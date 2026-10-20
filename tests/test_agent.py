"""Agent: proposers (deterministic + scripted fake LLM), guardrails and an end-to-end run."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathfusion.agent.evaluator import CVEvaluator, TrialResult, selection_optimism  # noqa: E402
from pathfusion.agent.llm import ChatLLM, extract_json  # noqa: E402
from pathfusion.agent.loop import AgentLoop  # noqa: E402
from pathfusion.agent.proposers import DeterministicProposer, LLMProposer  # noqa: E402
from pathfusion.agent.space import SearchSpace  # noqa: E402
from pathfusion.agent.state import AgentState, TrialRecord  # noqa: E402
from pathfusion.data.splits import load_cv_folds  # noqa: E402
from pathfusion.fusion.spec import FusionSpec  # noqa: E402
from pathfusion.training.context import DataContext  # noqa: E402

MODELS = ["phikonv2", "uni2h", "virchow2"]


def _fake_record(i, spec, mean, phase="adaptive"):
    res = TrialResult(spec, np.full(5, mean) + np.linspace(-0.01, 0.01, 5), np.zeros((10, 7)),
                      {"macro_auroc": mean, "balanced_accuracy": 0.3}, 0.0)
    return TrialRecord(i, phase, "test", "", res)


def test_extract_json_tolerates_chatter():
    txt = 'Sure!\n```json\n{"action": "propose", "config": {"family": "late"}, "reasoning": "x {y}"}\n```'
    assert extract_json(txt)["config"]["family"] == "late"
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_space_coverage_spans_required_regions():
    space = SearchSpace(MODELS)
    cov = space.coverage_specs()
    fams = {(s.family, s.norm, s.pca_dim, s.late_rule, s.stream_fusion) for s in cov}
    assert ("early_concat", "none", None, "prob_mean", "gated") in fams          # raw concat
    assert any(s.family == "early_concat" and s.norm == "l2" for s in cov)       # with L2
    assert any(s.family == "early_concat" and s.pca_dim for s in cov)            # with dim. reduction
    assert {s.late_rule for s in cov if s.family == "late"} >= {"prob_mean", "weighted", "stacking"}
    assert any(s.family == "intermediate" for s in cov)
    assert len({s.key for s in cov}) == len(cov)


def test_space_rejects_invalid_proposals():
    space = SearchSpace(MODELS)
    with pytest.raises(ValueError):
        space.validate(FusionSpec("single", ("uni2h",)))
    with pytest.raises(ValueError):
        space.validate(FusionSpec("late", ("uni2h", "virchow2"), use_meta=False))
    with pytest.raises(ValueError):
        space.validate(FusionSpec("late", ("uni2h", "conch")))


def test_deterministic_proposer_coverage_then_neighbours():
    space = SearchSpace(MODELS)
    prop = DeterministicProposer(space)
    st = AgentState(budget=30)
    for i, s in enumerate(space.reference_specs()):
        st.records.append(_fake_record(i, s, 0.7, "reference"))
    st.incumbent = st.records[0]
    seen = []
    for i in range(len(space.coverage_specs())):
        p = prop.propose(st, 8)
        assert p.source == "fixed"
        st.records.append(_fake_record(len(st.records), p.spec, 0.7 + 0.01 * i, "coverage"))
        seen.append(p.spec.key)
    assert len(set(seen)) == len(seen)
    st.incumbent = max(st.fusion_records, key=lambda r: r.result.mean)
    p = prop.propose(st, 8)
    assert p.source == "deterministic" and p.spec.key not in st.tried()
    assert "incumbent" in p.reasoning


class ScriptedLLM(ChatLLM):
    def __init__(self, replies):
        self.replies = list(replies)
        self.name = "scripted"
        self.calls = 0

    def chat(self, system, user):
        self.calls += 1
        assert "Trials" in user and "acceptance_z" in user
        return self.replies.pop(0)


def _state_after_coverage(space):
    st = AgentState(budget=30)
    for s in space.reference_specs():
        st.records.append(_fake_record(len(st.records), s, 0.7, "reference"))
    for s in space.coverage_specs():
        st.records.append(_fake_record(len(st.records), s, 0.7, "coverage"))
    st.incumbent = st.records[0]
    return st


def test_llm_proposer_valid_invalid_duplicate_and_stop():
    space = SearchSpace(MODELS)
    st = _state_after_coverage(space)
    good = {"family": "late", "models": ["uni2h", "virchow2"], "late_rule": "logit_mean"}
    llm = ScriptedLLM([json.dumps({"action": "propose", "config": good, "reasoning": "try pair"})])
    p = LLMProposer(space, llm, 1.0).propose(st, 8)
    assert p.source == "llm:scripted" and p.spec.models == ("uni2h", "virchow2")

    dup = space.coverage_specs()[0].to_dict()
    llm = ScriptedLLM(["not json", json.dumps({"action": "propose", "config": dup, "reasoning": "dup"})])
    p = LLMProposer(space, llm, 1.0, max_retries=1).propose(st, 8)
    assert p.source == "fallback" and p.spec.key not in st.tried() and llm.calls == 2

    llm = ScriptedLLM([json.dumps({"action": "stop", "reasoning": "converged"})])
    p = LLMProposer(space, llm, 1.0).propose(st, 8)   # 8 fusion trials done -> allowed
    assert p.stop

    llm = ScriptedLLM([json.dumps({"action": "stop", "reasoning": "early"})] * 2)
    p = LLMProposer(space, llm, 1.0).propose(st, 50)  # stop before min_trials -> refused
    assert not p.stop and p.source == "fallback"


class BrokenLLM(ChatLLM):
    name = "broken"

    def chat(self, system, user):
        raise ConnectionError("no network")


def test_llm_outage_degrades_to_deterministic():
    space = SearchSpace(MODELS)
    st = _state_after_coverage(space)
    p = LLMProposer(space, BrokenLLM(), 1.0).propose(st, 8)
    assert p.source == "fallback" and p.spec is not None


def test_selection_optimism_positive_for_noise_trials():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(7), 20)
    results = []
    for _ in range(20):  # 20 equally useless configurations
        p = rng.dirichlet(np.ones(7), size=len(y))
        results.append(TrialResult(FusionSpec("meta_only"), np.zeros(5), p, {"macro_auroc": 0.5}, 0))
    for r in results:
        from pathfusion.eval.metrics import macro_auroc
        r.pooled["macro_auroc"] = macro_auroc(y, r.oof_probs, 7)
    opt = selection_optimism(results, y, "macro_auroc", 7, n_boot=100, seed=0)
    assert opt["optimism"] > 0.01, "picking the best of 20 noise configs must look optimistic"


def test_agent_end_to_end_on_synthetic(cfg, cohort, split, synthetic_dir, tmp_path):
    from pathfusion.embeddings.store import FeatureStore

    dev = cohort[cohort["patient_id"].isin(split.loc[split["split"] != "test", "patient_id"])]
    store = FeatureStore.from_dir(synthetic_dir, dev, cfg.extract.models, max_tiles=cfg.train.eval_tiles)
    ctx = DataContext.build(dev, store, cfg.data.classes, cfg.train)
    test_ids = set(split.loc[split["split"] == "test", "patient_id"])
    assert not test_ids & set(store.patients()), "agent context must not contain test patients"

    folds = load_cv_folds(cfg)
    ev = CVEvaluator(ctx, folds, "macro_auroc", seed=0)
    space = SearchSpace(cfg.extract.models)
    acfg = replace(cfg.agent, budget=10, min_trials=8, patience=5, bootstrap_optimism=30)
    out = AgentLoop(ev, space.reference_specs(), DeterministicProposer(space), acfg, tmp_path, 0).run()

    assert out.state.trials_used <= 10
    lines = (tmp_path / "decisions.jsonl").read_text().splitlines()
    assert len(lines) == 3 + out.state.trials_used
    rec = json.loads(lines[-1])
    assert {"reasoning", "cv_mean", "fold_scores", "gain_vs_incumbent", "accepted"} <= set(rec)
    sel = json.loads((tmp_path / "selected.json").read_text())
    assert sel["selected"]["family"] in ("early_concat", "late", "intermediate")
    assert (tmp_path / "decisions.md").exists() and (tmp_path / "oof_probs.npz").exists()
    # synthetic features are built so each model sees different classes: fusion should help
    assert out.selected.result.mean > out.best_reference.result.mean


def test_single_mode_space_and_proposals():
    space = SearchSpace(MODELS, mode="single")
    cov = space.coverage_specs()
    assert all(s.family == "single" and len(s.models) == 1 for s in cov)
    assert {s.models[0] for s in cov} == set(MODELS)
    with pytest.raises(ValueError):
        space.validate(FusionSpec("late", ("uni2h", "virchow2")))
    st = AgentState(budget=20)
    for s in space.reference_specs():
        st.records.append(_fake_record(len(st.records), s, 0.7, "reference"))
    for s in cov:
        st.records.append(_fake_record(len(st.records), s, 0.71, "coverage"))
    st.incumbent = st.records[-1]
    p = DeterministicProposer(space).propose(st, 4)
    assert p.spec is not None and p.spec.family == "single" and p.spec.key not in st.tried()


def test_single_search_can_select_a_reference(cfg, cohort, split, synthetic_dir, tmp_path):
    from pathfusion.embeddings.store import FeatureStore

    dev = cohort[cohort["patient_id"].isin(split.loc[split["split"] != "test", "patient_id"])]
    store = FeatureStore.from_dir(synthetic_dir, dev, cfg.extract.models, max_tiles=cfg.train.eval_tiles)
    ctx = DataContext.build(dev, store, cfg.data.classes, cfg.train)
    ev = CVEvaluator(ctx, load_cv_folds(cfg), "macro_auroc", seed=0)
    space = SearchSpace(cfg.extract.models, mode="single")
    acfg = replace(cfg.agent, budget=2, min_trials=1, patience=5, bootstrap_optimism=10)
    out = AgentLoop(ev, space.reference_specs(), DeterministicProposer(space), acfg, tmp_path, 0,
                    select_from_references=True).run()
    assert out.state.trials_used == 2
    assert out.selected.spec.family == "single"


def test_evaluator_disk_cache_resumes(ctx, cfg, tmp_path):
    from pathfusion.agent.llm import CachedChat

    folds = load_cv_folds(cfg)
    dev = sorted(folds["patient_id"].astype(str).unique())
    sub = DataContext.build(ctx.cohort[ctx.cohort["patient_id"].isin(dev)].reset_index(drop=True),
                            ctx.store, ctx.classes, ctx.train_cfg)
    ev = CVEvaluator(sub, folds, "macro_auroc", seed=0, cache_dir=tmp_path)
    spec = FusionSpec("meta_only")
    a = ev.evaluate(spec)
    b = CVEvaluator(sub, folds, "macro_auroc", seed=0, cache_dir=tmp_path).evaluate(spec)
    assert b.extras.get("cached") and np.allclose(a.fold_scores, b.fold_scores)
    assert np.allclose(a.oof_probs, b.oof_probs) and a.pooled == b.pooled

    inner = ScriptedLLM(["reply-1"])
    llm = CachedChat(inner, tmp_path)
    user = "Trials ... acceptance_z"
    assert llm.chat("sys", user) == "reply-1"
    assert CachedChat(ScriptedLLM([]), tmp_path).chat("sys", user) == "reply-1"  # replayed, no call


def test_one_se_rule_uses_paired_scale(tmp_path):
    """A simpler config that is consistently worse on every fold must not be picked just
    because fold difficulty makes each mean's own SE large (the pilot's real failure)."""
    from pathfusion.agent.evaluator import CVEvaluator as _CV

    folds = np.array([0.82, 0.88, 0.82, 0.91, 0.87])          # shared fold difficulty
    best = FusionSpec("early_concat", ("phikon", "uni2h", "virchow2"))
    worse_simple = FusionSpec("late", ("phikon", "uni2h", "virchow2"), late_rule="prob_mean")
    tie_simple = FusionSpec("late", ("phikon", "uni2h", "virchow2"), late_rule="logit_mean")

    def rec(i, spec, scores):
        return TrialRecord(i, "coverage", "fixed", "", TrialResult(spec, scores, np.zeros((5, 7)),
                                                                   {"macro_auroc": 0, "balanced_accuracy": 0}, 0))

    ev = _CV.__new__(_CV)          # paired() only needs nb_ratio
    ev.nb_ratio = 32 / 130
    loop = AgentLoop.__new__(AgentLoop)
    loop.ev, loop.select_from_references = ev, False
    loop.state = AgentState(budget=10)
    worse_scores = folds - 0.022 + np.array([0, .004, -.004, .002, -.002])
    tie_scores = folds - 0.001 + np.array([0, .01, -.01, .005, -.005])
    loop.state.records = [rec(0, best, folds), rec(1, worse_simple, worse_scores), rec(2, tie_simple, tie_scores)]
    pick = loop._one_se_selection()
    assert pick.spec == tie_simple.canonical()           # within paired noise and simpler
    loop.state.records = loop.state.records[:2]
    assert loop._one_se_selection().spec == best.canonical()  # consistently worse -> excluded
