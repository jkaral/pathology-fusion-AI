"""End-to-end smoke run of stages 2-4 on SYNTHETIC features in a temp directory.

Checks that the command-line stages chain together (agent -> test lock -> results ->
fact sheet) without GPUs, gated weights or network. Numbers produced here are meaningless
and never written to the real artifacts/ tree.

Usage: python scripts/smoke_e2e.py [--keep]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from pathfusion.data.synthetic import write_synthetic_features  # noqa: E402

FAST = ["train.hidden_dim=32", "train.attn_dim=16", "train.meta_dim=8", "train.epochs=6", "train.train_tiles=48",
        "train.eval_tiles=256", "train.lr=0.003", "train.device=cpu", "tracking.mode=disabled",
        "agent.budget=10", "agent.min_trials=8", "agent.patience=6", "agent.proposer=deterministic",
        "agent.bootstrap_optimism=50", "eval.seeds=[0,1]", "eval.n_bootstrap=200"]


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--keep", action="store_true", help="keep the temp directory")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="pathfusion_smoke_"))
    try:
        cohort = pd.read_csv(ROOT / "artifacts/data/cohort.csv", dtype={"patient_id": str})
        feats = write_synthetic_features(cohort, tmp / "feats", seed=0)
        paths = [f"paths.agent_dir={tmp / 'agent'}", f"paths.results_dir={tmp / 'results'}"]
        py = sys.executable
        run([py, "scripts/02_run_agent.py", "--embeddings-dir", str(feats), "--run-id", "smoke", *FAST, *paths])
        run([py, "scripts/03_final_evaluation.py", "--embeddings-dir", str(feats), *FAST, *paths])
        lock_refused = subprocess.run([py, "scripts/03_final_evaluation.py", "--embeddings-dir", str(feats),
                                       *FAST, *paths], cwd=ROOT, capture_output=True, text=True)
        assert lock_refused.returncode != 0 and "Refusing" in lock_refused.stderr, "test lock did not hold"
        print("test lock held on second run: OK")
        run([py, "scripts/04_report_facts.py", "--embeddings-dir", str(feats), *paths])
        expected = ["agent/smoke/decisions.md", "agent/smoke/selected.json", "agent/smoke/trajectory.png",
                    "results/results.md", "results/TEST_LOCK.json", "results/figures/curves.png",
                    "results/figures/confusion_fused.png", "results/figures/forest_macro_auroc.png",
                    "results/report_facts.md", "results/error_analysis.csv"]
        expected += ["agent/smoke/single_search/decisions.md"]
        import json as _json

        sel = _json.loads((tmp / "agent/smoke/selected.json").read_text())
        assert "best_single_tuned" in sel and sel["single_search_trials"] == sel["fusion_trials"], sel.keys()
        print("equal-budget single search:", sel["best_single_tuned_short"], "trials", sel["single_search_trials"])
        missing = [e for e in expected if not (tmp / e).exists()]
        assert not missing, f"missing outputs: {missing}"
        print(f"SMOKE OK: all {len(expected)} expected outputs present in {tmp}")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
