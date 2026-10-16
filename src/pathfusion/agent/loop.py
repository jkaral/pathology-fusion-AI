"""The agent loop: propose -> evaluate -> read result -> decide -> log -> repeat.

Guardrails against validation overfitting (the brief's failure mode):

1. **Selection data**: stratified 5-fold CV on the 162 development patients, never the
   22-patient validation split alone, never test.
2. **Explicit budget**: at most ``budget`` fusion trials (default 24) plus 3 references.
3. **Noise-aware acceptance**: a trial becomes the incumbent only if its fold-paired gain
   over the incumbent exceeds ``acceptance_z`` x Nadeau-Bengio-corrected SE.
4. **Stopping**: budget exhausted, ``patience`` trials without an accepted improvement,
   the proposer has nothing left, or the LLM asks to stop (only after ``min_trials``).
5. **Final pick by the one-standard-error rule**: among fusion trials within one SE of
   the best CV mean, take the least complex; ranking noise is not rewarded.
6. **Selection-noise estimate**: bootstrap winner's-curse optimism of the best CV score.

Every step is appended to ``decisions.jsonl`` as it happens; ``decisions.md`` renders
the full log, and ``selected.json`` holds the chosen spec for the final evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pathfusion.agent.evaluator import CVEvaluator, TrialResult, selection_optimism
from pathfusion.agent.proposers import Proposal, Proposer
from pathfusion.agent.state import AgentState, TrialRecord
from pathfusion.config import AgentCfg
from pathfusion.fusion.spec import FusionSpec
from pathfusion.utils.io import dump_json, to_json_line
from pathfusion.utils.logging import get_logger

log = get_logger("agent")


@dataclass
class AgentOutcome:
    selected: TrialRecord
    best_reference: TrialRecord
    state: AgentState
    optimism: dict[str, float]
    stop_reason: str
    out_dir: Path
    proposer_name: str = ""


class AgentLoop:
    def __init__(self, evaluator: CVEvaluator, space_refs: list[FusionSpec], proposer: Proposer,
                 cfg: AgentCfg, out_dir: Path, seed: int):
        self.ev = evaluator
        self.refs = space_refs
        self.proposer = proposer
        self.cfg = cfg
        self.out_dir = out_dir
        self.seed = seed
        self.state = AgentState(budget=cfg.budget)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(out_dir / "decisions.jsonl", "w")

    # ------------------------------------------------------------------ helpers
    def _se_mean(self, r: TrialResult) -> float:
        k = len(r.fold_scores)
        return float(np.sqrt((1.0 / k + self.ev.nb_ratio) * np.var(r.fold_scores, ddof=1))) if k > 1 else 0.0

    def _record(self, rec: TrialRecord) -> None:
        self.state.records.append(rec)
        r = rec.result
        row = {
            "index": rec.index, "phase": rec.phase, "source": rec.source, "reasoning": rec.reasoning,
            "config": rec.spec.to_dict(), "config_short": rec.spec.short(), "key": rec.spec.key,
            "fold_scores": [round(float(s), 4) for s in r.fold_scores],
            "cv_mean": round(r.mean, 4), "cv_sd": round(r.sd, 4), "cv_se_corrected": round(self._se_mean(r), 4),
            "pooled_oof": {k: round(v, 4) for k, v in r.pooled.items()},
            "gain_vs_incumbent": None if rec.diff_vs_incumbent is None else round(rec.diff_vs_incumbent, 4),
            "gain_se": None if rec.se_diff is None else round(rec.se_diff, 4),
            "accepted": rec.accepted, "incumbent_after": rec.incumbent_after,
            "seconds": round(r.seconds, 1), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._jsonl.write(to_json_line(row) + "\n")
        self._jsonl.flush()
        log.info("#%d %-10s %-60s cv=%.3f±%.3f %s", rec.index, rec.phase, rec.spec.short()[:60], r.mean, r.sd,
                 "ACCEPT" if rec.accepted else "")

    # ------------------------------------------------------------------ main loop
    def run(self) -> AgentOutcome:
        st = self.state
        # 1) references: single model + metadata. The best one is the bar to beat.
        for spec in self.refs:
            res = self.ev.evaluate(spec)
            rec = TrialRecord(len(st.records), "reference", "fixed",
                              "reference: single foundation model + metadata (the bar fusion must clear)", res)
            if st.incumbent is None or res.mean > st.incumbent.result.mean:
                st.incumbent = rec
                st.best_reference = rec
            rec.incumbent_after = st.incumbent.spec.short()
            self._record(rec)
        assert st.incumbent is not None and st.best_reference is not None

        stop_reason = "budget exhausted"
        while st.remaining > 0:
            prop: Proposal = self.proposer.propose(st, self.cfg.min_trials)
            if prop.stop or prop.spec is None:
                stop_reason = f"proposer stopped: {prop.reasoning}"
                break
            res = self.ev.evaluate(prop.spec)
            diff = self.ev.paired(res, st.incumbent.result)
            accepted = diff.mean > 0 and diff.mean > self.cfg.acceptance_z * diff.se
            phase = "coverage" if prop.source == "fixed" else "adaptive"
            rec = TrialRecord(len(st.records), phase, prop.source, prop.reasoning, res,
                              diff_vs_incumbent=diff.mean, se_diff=diff.se, accepted=accepted)
            if accepted:
                st.incumbent = rec
                st.since_improvement = 0
            else:
                st.since_improvement += 1
            rec.incumbent_after = st.incumbent.spec.short()
            self._record(rec)
            if st.since_improvement >= self.cfg.patience and st.trials_used >= self.cfg.min_trials:
                stop_reason = f"patience: {self.cfg.patience} trials without an accepted improvement"
                break

        selected = self._one_se_selection()
        optimism = selection_optimism(
            [r.result for r in st.fusion_records], self.ev.y_dev, self.ev.metric_name,
            self.ev.ctx.n_classes, self.cfg.bootstrap_optimism, self.seed,
        )
        self._jsonl.close()
        outcome = AgentOutcome(selected, st.best_reference, st, optimism, stop_reason, self.out_dir,
                               getattr(self.proposer, "name", type(self.proposer).__name__))
        write_reports(outcome, self.ev, self._se_mean)
        transcript = getattr(self.proposer, "transcript", None)
        if transcript:
            with open(self.out_dir / "llm_transcript.jsonl", "w") as fh:
                for turn in transcript:
                    fh.write(to_json_line(turn) + "\n")
        return outcome

    def _one_se_selection(self) -> TrialRecord:
        fusion = self.state.fusion_records
        if not fusion:
            raise RuntimeError("no fusion trials were evaluated")
        best = max(fusion, key=lambda r: r.result.mean)
        thresh = best.result.mean - self._se_mean(best.result)
        cands = [r for r in fusion if r.result.mean >= thresh]
        return min(cands, key=lambda r: (r.spec.complexity(), -r.result.mean))


def write_reports(out: AgentOutcome, ev: CVEvaluator, se_mean) -> None:  # type: ignore[no-untyped-def]
    st, d = out.state, out.out_dir
    sel, ref = out.selected, out.best_reference
    gain = ev.paired(sel.result, ref.result)
    summary = {
        "selected": sel.spec.to_dict(), "selected_short": sel.spec.short(), "selected_index": sel.index,
        "selected_cv_mean": sel.result.mean, "selected_cv_se": se_mean(sel.result),
        "best_reference": ref.spec.to_dict(), "best_reference_short": ref.spec.short(),
        "best_reference_cv_mean": ref.result.mean,
        "cv_gain_vs_best_single": gain.mean, "cv_gain_se": gain.se,
        "cv_gain_significant": bool(gain.mean > 0 and gain.mean > 1.0 * gain.se),
        "metric": ev.metric_name, "n_dev": len(ev.dev_ids), "n_folds": len(ev.splits),
        "fusion_trials": st.trials_used, "budget": st.budget, "stop_reason": out.stop_reason,
        "selection_optimism": out.optimism,
        "proposer": out.proposer_name,
    }
    dump_json(summary, d / "selected.json")
    np.savez_compressed(d / "oof_probs.npz", dev_ids=np.array(ev.dev_ids), y=ev.y_dev,
                        **{f"trial_{r.index}": r.result.oof_probs for r in st.records})
    lines = [
        "# Agent decision log", "",
        f"- Selection metric: **{ev.metric_name}**, stratified {len(ev.splits)}-fold CV on "
        f"{len(ev.dev_ids)} development patients (train+val; test never touched).",
        f"- Budget: {st.budget} fusion trials (+{len([r for r in st.records if r.phase == 'reference'])} "
        f"references); used {st.trials_used}. Stop reason: {out.stop_reason}.",
        "- Acceptance: paired fold gain > z x Nadeau-Bengio corrected SE. "
        "Final pick: one-SE rule (least complex fusion trial within 1 SE of the best).",
        f"- Best single model + metadata: **{ref.spec.short()}** (CV {ref.result.mean:.3f}).",
        f"- **Selected**: #{sel.index} **{sel.spec.short()}** (CV {sel.result.mean:.3f} ± "
        f"{se_mean(sel.result):.3f}); gain vs best single {gain.mean:+.3f} ± {gain.se:.3f} (corrected SE).",
        f"- Selection optimism (bootstrap winner's curse): {out.optimism.get('optimism', float('nan')):.3f}; "
        f"best pooled OOF {out.optimism.get('best_pooled', float('nan')):.3f} -> corrected "
        f"{out.optimism.get('corrected', float('nan')):.3f}.", "",
        "| # | phase | source | config | CV mean ± sd | gain vs inc. ± SE | pooled bal.acc | decision | reasoning |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in st.records:
        g = "" if r.diff_vs_incumbent is None else f"{r.diff_vs_incumbent:+.3f} ± {r.se_diff:.3f}"
        dec = "reference" if r.phase == "reference" else ("**accepted**" if r.accepted else "rejected")
        reason = r.reasoning.replace("|", "/").replace("\n", " ")
        lines.append(f"| {r.index} | {r.phase} | {r.source} | {r.spec.short()} | {r.result.mean:.3f} ± "
                     f"{r.result.sd:.3f} | {g} | {r.result.pooled['balanced_accuracy']:.3f} | {dec} | {reason} |")
    (d / "decisions.md").write_text("\n".join(lines) + "\n")
    with open(d / "trials.csv", "w") as fh:
        fh.write("index,phase,source,family,config,cv_mean,cv_sd,pooled_macro_auroc,pooled_bacc,accepted\n")
        for r in st.records:
            fh.write(f"{r.index},{r.phase},{r.source},{r.spec.family},\"{r.spec.short()}\",{r.result.mean:.4f},"
                     f"{r.result.sd:.4f},{r.result.pooled['macro_auroc']:.4f},"
                     f"{r.result.pooled['balanced_accuracy']:.4f},{r.accepted}\n")
    (d / "summary.txt").write_text(json.dumps(summary, indent=2))
