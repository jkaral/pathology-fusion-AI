"""Stage 2: agent-driven fusion-strategy search on the development set (train + val).

The test split is never loaded here: the feature store is built for development patients
only, so nothing in this stage can see a test patient.

Usage:
    python scripts/02_run_agent.py                                  # proposer: auto
    python scripts/02_run_agent.py agent.proposer=deterministic     # no LLM
    python scripts/02_run_agent.py agent.budget=12 agent.patience=6
    python scripts/02_run_agent.py --embeddings-dir /kaggle/input/clwd-feats

Outputs: artifacts/agent/<run_id>/{decisions.jsonl, decisions.md, trials.csv,
selected.json, oof_probs.npz, llm_transcript.jsonl, config.json}; artifacts/agent/LATEST
points at the newest run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pathfusion.agent.evaluator import CVEvaluator  # noqa: E402
from pathfusion.agent.factory import make_llm, make_proposer, resolve_backend  # noqa: E402
from pathfusion.agent.loop import AgentLoop  # noqa: E402
from pathfusion.agent.space import SearchSpace  # noqa: E402
from pathfusion.config import load_config  # noqa: E402
from pathfusion.data.splits import load_cv_folds, load_split  # noqa: E402
from pathfusion.embeddings.store import FeatureStore  # noqa: E402
from pathfusion.eval.plots import plot_agent_trajectory  # noqa: E402
from pathfusion.training.context import DataContext  # noqa: E402
from pathfusion.utils.io import dump_json, git_commit  # noqa: E402
from pathfusion.utils.logging import get_logger  # noqa: E402
from pathfusion.utils.seed import seed_everything  # noqa: E402

log = get_logger("run_agent")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-dir", type=Path, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    seed_everything(cfg.seed)

    cohort = pd.read_csv(cfg.paths.resolve("cohort_csv"), dtype={"patient_id": str, "SampleNumber": str})
    split = load_split(cfg)
    folds = load_cv_folds(cfg)
    dev_ids = set(split.loc[split["split"] != "test", "patient_id"])
    dev_cohort = cohort[cohort["patient_id"].isin(dev_ids)].reset_index(drop=True)
    assert set(folds["patient_id"]) == dev_ids, "CV folds must cover exactly the development set"

    emb_dir = args.embeddings_dir or cfg.paths.resolve("embeddings_dir")
    log.info("loading features for %d development patients from %s", len(dev_cohort), emb_dir)
    store = FeatureStore.from_dir(emb_dir, dev_cohort, cfg.extract.models, max_tiles=cfg.train.eval_tiles)
    ctx = DataContext.build(dev_cohort, store, cfg.data.classes, cfg.train)

    space = SearchSpace(cfg.extract.models)
    evaluator = CVEvaluator(ctx, folds, cfg.agent.selection_metric, cfg.seed)
    llm = make_llm(cfg.agent)  # one (lazily loaded) LLM shared by both searches
    proposer = make_proposer(cfg.agent, space, n_dev=len(dev_ids), n_folds=len(evaluator.splits), llm=llm)
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + f"_{resolve_backend(cfg.agent)}"
    out_dir = cfg.paths.resolve("agent_dir") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json({"config": cfg.to_dict(), "git_commit": git_commit(), "embeddings_dir": str(emb_dir),
               "device": str(ctx.device)}, out_dir / "config.json")

    outcome = AgentLoop(evaluator, space.reference_specs(), proposer, cfg.agent, out_dir, cfg.seed).run()
    if cfg.agent.equal_budget_single_search:
        # Equal-budget control: the same agent/selection rule over single-model configs, with
        # exactly as many trials as the fusion search used. Shares the evaluator, so folds,
        # seed and cached reference fits are identical.
        single_space = SearchSpace(cfg.extract.models, mode="single")
        scfg = replace(cfg.agent, budget=outcome.state.trials_used)
        sprop = make_proposer(scfg, single_space, n_dev=len(dev_ids), n_folds=len(evaluator.splits), llm=llm)
        single = AgentLoop(evaluator, single_space.reference_specs(), sprop, scfg, out_dir / "single_search",
                           cfg.seed, select_from_references=True).run()
        gain = evaluator.paired(outcome.selected.result, single.selected.result)
        sel_path = out_dir / "selected.json"
        sel = json.loads(sel_path.read_text())
        sel.update({
            "best_single_tuned": single.selected.spec.to_dict(),
            "best_single_tuned_short": single.selected.spec.short(),
            "best_single_tuned_cv_mean": single.selected.result.mean,
            "single_search_trials": single.state.trials_used,
            "cv_gain_vs_tuned_single": gain.mean, "cv_gain_vs_tuned_single_se": gain.se,
            "cv_gain_vs_tuned_single_significant": bool(gain.mean > 0 and gain.mean > gain.se),
        })
        dump_json(sel, sel_path)
        log.info("equal-budget single search (%d trials): %s (CV %.3f); fused gain %+.3f ± %.3f",
                 single.state.trials_used, single.selected.spec.short(), single.selected.result.mean,
                 gain.mean, gain.se)
    (cfg.paths.resolve("agent_dir") / "LATEST").write_text(run_id + "\n")
    records = [json.loads(line) for line in (out_dir / "decisions.jsonl").read_text().splitlines()]
    plot_agent_trajectory(records, out_dir / "trajectory.png")
    log.info("stop: %s", outcome.stop_reason)
    log.info("best single + meta: %s (CV %.3f)", outcome.best_reference.spec.short(),
             outcome.best_reference.result.mean)
    log.info("selected: %s (CV %.3f) -> %s", outcome.selected.spec.short(), outcome.selected.result.mean,
             out_dir / "selected.json")


if __name__ == "__main__":
    main()
