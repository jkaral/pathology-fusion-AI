"""Proposal policies: what to evaluate next, with a stated reason.

Both proposers run the same fixed *coverage* set first, then adapt:

* ``DeterministicProposer``: hill-climbing from the incumbent. Untried one-knob neighbours
  of the incumbent in a fixed priority order; when exhausted, neighbours of the next-best
  trials; stops when nothing untried remains near the top configurations.
* ``LLMProposer``: an LLM reads the full trial table (fold-mean +/- SE, paired gain vs. the
  incumbent with corrected SE, acceptance decisions) and the search-space grammar, and
  returns JSON ``{"action", "config", "reasoning"}``. Proposals are validated against the
  space and de-duplicated; one retry with the error message, then the deterministic
  proposer is used for that step (logged as ``fallback``). The LLM can also ask to stop,
  which is honoured only after ``min_trials``.

Neither proposer can see test data: they only receive development-CV summaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pathfusion.agent.llm import ChatLLM, extract_json
from pathfusion.agent.space import SearchSpace
from pathfusion.agent.state import AgentState, TrialRecord
from pathfusion.fusion.spec import FusionSpec
from pathfusion.utils.logging import get_logger

log = get_logger("proposer")


@dataclass
class Proposal:
    spec: FusionSpec | None
    reasoning: str
    source: str
    stop: bool = False


class Proposer(ABC):
    def __init__(self, space: SearchSpace):
        self.space = space

    def coverage_next(self, state: AgentState) -> Proposal | None:
        tried = state.tried()
        for i, spec in enumerate(self.space.coverage_specs()):
            if spec.key not in tried:
                return Proposal(spec, f"coverage {i + 1}/{len(self.space.coverage_specs())}: "
                                      f"map the {spec.family} region before adapting", "fixed")
        return None

    @abstractmethod
    def propose(self, state: AgentState, min_trials: int) -> Proposal: ...


def _fmt(r: TrialRecord) -> str:
    return f"{r.spec.short()} = {r.result.mean:.3f}±{r.result.sd:.3f}"


class DeterministicProposer(Proposer):
    name = "deterministic"
    max_anchors = 4

    def propose(self, state: AgentState, min_trials: int) -> Proposal:
        cov = self.coverage_next(state)
        if cov is not None:
            return cov
        tried = state.tried()
        ranked = sorted(state.fusion_records, key=lambda r: -r.result.mean)
        anchors = ([state.incumbent] if state.incumbent and state.incumbent.phase != "reference" else []) + ranked
        seen: set[str] = set()
        for anchor in anchors:
            if anchor.spec.key in seen or anchor.spec.family not in self.space.families:
                continue
            seen.add(anchor.spec.key)
            for label, nb in self.space.neighbors(anchor.spec):
                if nb.key not in tried:
                    role = "incumbent" if anchor is state.incumbent else f"rank-{len(seen)} fusion trial"
                    return Proposal(nb, f"local move from {role} ({_fmt(anchor)}): {label}", self.name)
            if len(seen) >= self.max_anchors:  # only refine around the top few configurations
                break
        return Proposal(None, "no untried one-knob neighbours of the top configurations remain", self.name,
                        stop=True)


SYSTEM_PROMPT = """You are the search policy of an AutoML agent choosing {task} \
for 7-class lung adenocarcinoma growth-pattern classification.
Each trial is scored by stratified {k}-fold cross-validation on {n_dev} development patients ({metric}). \
With ~{per_fold} patients per fold the noise is large: a difference smaller than its standard error is NOT evidence. \
A new incumbent is accepted only if its fold-paired gain exceeds acceptance_z x corrected SE.
Your job: pick the next configuration that is most informative. Early on, probe under-explored regions; \
later, refine around configurations that are genuinely ahead. Do not re-run tried configurations. \
Prefer simpler configurations when results are within noise. If remaining improvements are plausibly \
below the noise level, you may stop.
Reply with ONE JSON object only, no prose outside it:
{{"action": "propose" or "stop", "config": {{...}} (omit when stopping), "reasoning": "<=2 sentences"}}"""


class LLMProposer(Proposer):
    def __init__(self, space: SearchSpace, llm: ChatLLM, acceptance_z: float, max_retries: int = 1,
                 n_dev: int = 162, n_folds: int = 5, metric: str = "macro_auroc"):
        super().__init__(space)
        task = ("how to fuse pathology foundation models (plus age and sex)" if space.mode == "fusion"
                else "the best single pathology foundation model configuration (plus age and sex)")
        self.system = SYSTEM_PROMPT.format(task=task, k=n_folds, n_dev=n_dev, metric=metric,
                                           per_fold=round(n_dev / n_folds))
        self.llm = llm
        self.acceptance_z = acceptance_z
        self.max_retries = max_retries
        self.fallback = DeterministicProposer(space)
        self.name = f"llm:{llm.name}"
        self.transcript: list[dict[str, str]] = []

    def _user_prompt(self, state: AgentState, min_trials: int, error: str | None) -> str:
        lines = [self.space.describe(), "", f"acceptance_z = {self.acceptance_z}",
                 f"budget remaining: {state.remaining} trials; minimum before stopping: "
                 f"{max(0, min_trials - state.trials_used)} more", "",
                 "Trials (cv_mean ± fold_sd | gain vs incumbent ± corrected SE | decision):"]
        for r in state.records:
            gain = (f"{r.diff_vs_incumbent:+.3f}±{r.se_diff:.3f}" if r.diff_vs_incumbent is not None else "n/a")
            dec = "ACCEPTED" if r.accepted else ("reference" if r.phase == "reference" else "rejected")
            lines.append(f"  #{r.index} [{r.phase}] {r.spec.short()} | {r.result.mean:.3f}±{r.result.sd:.3f} "
                         f"| {gain} | {dec} | balanced_acc={r.result.pooled['balanced_accuracy']:.3f}")
        if state.incumbent is not None:
            lines.append(f"Current incumbent: #{state.incumbent.index} {state.incumbent.spec.short()}")
        lines.append(f"Trials since last accepted improvement: {state.since_improvement}")
        if error:
            lines += ["", f"Your previous reply was rejected: {error}. Reply again with a valid, untried config."]
        return "\n".join(lines)

    def propose(self, state: AgentState, min_trials: int) -> Proposal:
        cov = self.coverage_next(state)
        if cov is not None:
            return cov
        error: str | None = None
        for _attempt in range(self.max_retries + 1):
            user = self._user_prompt(state, min_trials, error)
            try:
                reply = self.llm.chat(self.system, user)
            except Exception as exc:  # network / OOM / quota: degrade, never crash the search
                log.warning("LLM call failed (%s); using deterministic proposer", exc)
                fb = self.fallback.propose(state, min_trials)
                return Proposal(fb.spec, f"[LLM unavailable: {exc}] {fb.reasoning}", "fallback", fb.stop)
            self.transcript.append({"user": user, "reply": reply})
            try:
                obj = extract_json(reply)
                reasoning = str(obj.get("reasoning", "")).strip()
                if obj.get("action") == "stop":
                    if state.trials_used >= min_trials:
                        return Proposal(None, reasoning or "LLM requested stop", self.name, stop=True)
                    raise ValueError(f"stopping is not allowed before {min_trials} trials")
                cfg = dict(obj["config"])
                cfg.setdefault("use_meta", True)
                spec = self.space.validate(FusionSpec.from_dict(cfg))
                if spec.key in state.tried():
                    raise ValueError(f"config already tried: {spec.short()}")
                return Proposal(spec, reasoning, self.name)
            except (ValueError, KeyError, TypeError) as exc:
                error = str(exc)
                log.info("LLM proposal rejected: %s", error)
        fb = self.fallback.propose(state, min_trials)
        return Proposal(fb.spec, f"[LLM proposal invalid: {error}] {fb.reasoning}", "fallback", fb.stop)
