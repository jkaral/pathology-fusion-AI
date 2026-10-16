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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pathfusion.agent.evaluator import CVEvaluator  # noqa: E402
from pathfusion.agent.factory import make_proposer, resolve_backend  # noqa: E402
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
    proposer = make_proposer(cfg.agent, space, n_dev=len(dev_ids), n_folds=len(evaluator.splits))
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S") + f"_{resolve_backend(cfg.agent)}"
    out_dir = cfg.paths.resolve("agent_dir") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_json({"config": cfg.to_dict(), "git_commit": git_commit(), "embeddings_dir": str(emb_dir),
               "device": str(ctx.device)}, out_dir / "config.json")

    outcome = AgentLoop(evaluator, space.reference_specs(), proposer, cfg.agent, out_dir, cfg.seed).run()
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
