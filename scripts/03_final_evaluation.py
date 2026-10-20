"""Stage 3: train every results-table row on train, evaluate ONCE on test.

Reads the agent's choice from ``artifacts/agent/<run>/selected.json`` (default: LATEST).
Before any test prediction is made, ``artifacts/results/TEST_LOCK.json`` is written with
the selected spec, the agent run, the split checksum and the git commit. If the lock
already exists the script refuses to run: test is touched once. A deliberate re-run
(e.g. after a crash) needs ``--rerun-reason "..."``, which is appended to the lock's
history so the audit trail shows it.

Usage:
    python scripts/03_final_evaluation.py
    python scripts/03_final_evaluation.py --agent-run 20261016-120000_local_llm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathfusion.config import load_config  # noqa: E402
from pathfusion.data.splits import load_split  # noqa: E402
from pathfusion.embeddings.store import FeatureStore  # noqa: E402
from pathfusion.eval import plots  # noqa: E402
from pathfusion.eval.final import error_analysis, summarise, table_rows, train_rows  # noqa: E402
from pathfusion.fusion.spec import FusionSpec  # noqa: E402
from pathfusion.training.context import DataContext  # noqa: E402
from pathfusion.utils.io import dump_json, file_sha256, git_commit, load_json  # noqa: E402
from pathfusion.utils.logging import get_logger  # noqa: E402
from pathfusion.utils.seed import seed_everything  # noqa: E402

log = get_logger("final_eval")


def fmt_ci(m: dict) -> str:
    return f"{m['point']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


def write_markdown(res: dict, agent: dict, classes: list[str], path: Path) -> None:
    lines = [
        "# Results (test set, evaluated once)", "",
        f"Test patients: {res['n_test']}; seeds: {res['seeds']}. "
        "Seed mean ± sd = training-noise spread; [CI] = 95% stratified patient-bootstrap CI of the "
        "seed-ensemble.", "",
        "| Configuration | Macro AUROC (seed mean ± sd) | Macro AUROC ensemble [95% CI] | "
        "Balanced acc. (seed mean ± sd) | Balanced acc. ensemble [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for r in res["rows"]:
        lines.append(f"| {r['label']} | {r['auroc_seed_mean']:.3f} ± {r['auroc_seed_sd']:.3f} | "
                     f"{fmt_ci(r['macro_auroc'])} | {r['bacc_seed_mean']:.3f} ± {r['bacc_seed_sd']:.3f} | "
                     f"{fmt_ci(r['balanced_accuracy'])} |")
    lines += ["", "## Fused minus each other row (paired bootstrap on the same test patients)", "",
              "| vs | Δ macro AUROC [95% CI] | P(Δ≤0) | Δ balanced acc. [95% CI] | P(Δ≤0) |", "|---|---|---|---|---|"]
    for name, d in res["paired"].items():
        a, b = d["macro_auroc"], d["balanced_accuracy"]
        lines.append(f"| {name} | {a['diff']:+.3f} [{a['lo']:+.3f}, {a['hi']:+.3f}] | {a['p_le_0']:.2f} | "
                     f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}] | {b['p_le_0']:.2f} |")
    lines += ["", f"Fused beats every single-model row (ensemble point estimate): "
              f"{res['fused_beats_every_single_point']}; (seed means): {res['fused_beats_every_single_seed_mean']}",
              "", "## Per-class OvR AUROC (seed-ensemble)", "",
              "| Configuration | " + " | ".join(classes) + " |", "|---|" + "---|" * len(classes)]
    for r in res["rows"]:
        lines.append(f"| {r['label']} | " + " | ".join(f"{v:.3f}" for v in r["per_class_auroc"]) + " |")
    lines += ["", "## Agent selection (development CV, before test)", "",
              f"- Selected: {agent['selected_short']} "
              f"(CV {agent['selected_cv_mean']:.3f} ± {agent['selected_cv_se']:.3f})",
              f"- Best single + meta: {agent['best_reference_short']} (CV {agent['best_reference_cv_mean']:.3f})",
              f"- CV gain {agent['cv_gain_vs_best_single']:+.3f} ± {agent['cv_gain_se']:.3f} "
              f"(significant at 1 SE: {agent['cv_gain_significant']})",
              f"- Selection optimism: {agent['selection_optimism'].get('optimism', float('nan')):.3f}"]
    if "best_single_tuned_short" in agent:
        lines += [f"- Equal-budget single-model search ({agent['single_search_trials']} trials): "
                  f"{agent['best_single_tuned_short']} (CV {agent['best_single_tuned_cv_mean']:.3f}); fused gain vs it "
                  f"{agent['cv_gain_vs_tuned_single']:+.3f} ± {agent['cv_gain_vs_tuned_single_se']:.3f} "
                  f"(significant at 1 SE: {agent['cv_gain_vs_tuned_single_significant']})"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agent-run", default=None, help="run id under artifacts/agent (default: LATEST)")
    ap.add_argument("--embeddings-dir", type=Path, default=None)
    ap.add_argument("--rerun-reason", default=None, help="required if TEST_LOCK.json already exists")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    seed_everything(cfg.seed)
    res_dir = cfg.paths.resolve("results_dir")
    res_dir.mkdir(parents=True, exist_ok=True)

    agent_root = cfg.paths.resolve("agent_dir")
    run_id = args.agent_run or (agent_root / "LATEST").read_text().strip()
    agent = load_json(agent_root / run_id / "selected.json")
    fused = FusionSpec.from_dict(agent["selected"])
    # equal-budget tuned single model if the control search ran, else the default single
    best_single_spec = FusionSpec.from_dict(agent.get("best_single_tuned", agent["best_reference"]))
    best_single = best_single_spec.models[0]

    # ---------------------------------------------------------------- test lock
    lock_path = res_dir / "TEST_LOCK.json"
    entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "agent_run": run_id, "fused": fused.to_dict(),
             "best_single": best_single, "split_sha256": file_sha256(cfg.paths.resolve("split_csv")),
             "git_commit": git_commit(), "seeds": cfg.eval.seeds}
    if lock_path.exists():
        lock = load_json(lock_path)
        if not args.rerun_reason:
            sys.exit(f"Refusing: test set already used ({lock_path}). Pass --rerun-reason to document a re-run.")
        lock["history"].append({**entry, "rerun_reason": args.rerun_reason})
    else:
        lock = {"first_use": entry, "history": [entry]}
    dump_json(lock, lock_path)

    # ---------------------------------------------------------------- data
    cohort = pd.read_csv(cfg.paths.resolve("cohort_csv"), dtype={"patient_id": str, "SampleNumber": str})
    split = load_split(cfg)
    ids = {s: sorted(split.loc[split["split"] == s, "patient_id"]) for s in ("train", "val", "test")}
    emb_dir = args.embeddings_dir or cfg.paths.resolve("embeddings_dir")
    store = FeatureStore.from_dir(emb_dir, cohort, cfg.extract.models, max_tiles=cfg.train.eval_tiles)
    ctx = DataContext.build(cohort, store, cfg.data.classes, cfg.train)

    rows = table_rows(cfg.extract.models, best_single_spec, fused)
    train_rows(cfg, ctx, rows, ids["train"], ids["val"], ids["test"])
    res = summarise(cfg, ctx, rows, ids["test"])
    res["agent_run"] = run_id
    res["git_commit"] = git_commit()
    dump_json(res, res_dir / "results.json")
    write_markdown(res, agent, cfg.data.classes, res_dir / "results.md")

    # ---------------------------------------------------------------- predictions & errors
    preds = []
    for row in rows:
        for pid, p in zip(ids["test"], row.ensemble):
            probs = {c: round(float(v), 5) for c, v in zip(cfg.data.classes, p)}
            preds.append({"row": row.name, "patient_id": pid, **probs})
    pd.DataFrame(preds).to_csv(res_dir / "test_predictions.csv", index=False)
    errs = pd.DataFrame(error_analysis(ctx, next(r for r in rows if r.name == "fused"), ids["test"]))
    errs.to_csv(res_dir / "error_analysis.csv", index=False)
    np.savez_compressed(res_dir / "probs.npz", test_ids=np.array(ids["test"]), val_ids=np.array(ids["val"]),
                        **{f"{r.name}_test": np.array(r.test_probs) for r in rows},
                        **{f"{r.name}_val": np.array(r.val_probs) for r in rows})

    # ---------------------------------------------------------------- figures
    fig = res_dir / "figures"
    by = {r["name"]: r for r in res["rows"]}
    for name in ("fused", "best_single_meta"):
        plots.plot_confusion(np.array(by[name]["confusion"]), cfg.data.classes,
                             f"Test confusion: {by[name]['label']}", fig / f"confusion_{name}.png")
    forest_rows = [{"name": r["name"], "label": r["label"].split(":")[0], **r} for r in res["rows"]]
    plots.plot_forest(forest_rows, "macro_auroc", "Macro AUROC", fig / "forest_macro_auroc.png", "fused")
    plots.plot_forest(forest_rows, "balanced_accuracy", "Balanced accuracy", fig / "forest_bacc.png", "fused")
    plots.plot_curves({r.name: r.histories for r in rows if r.spec.family != "meta_only"}, fig / "curves.png")
    plots.plot_per_class_auroc({"metadata only": np.array(by["meta_only"]["per_class_auroc"]),
                                "best single + meta": np.array(by["best_single_meta"]["per_class_auroc"]),
                                "fused": np.array(by["fused"]["per_class_auroc"])},
                               cfg.data.classes, fig / "per_class_auroc.png")
    dec = [json.loads(line) for line in (agent_root / run_id / "decisions.jsonl").read_text().splitlines()]
    plots.plot_agent_trajectory(dec, fig / "agent_trajectory.png")
    log.info("results: %s", res_dir / "results.md")
    print((res_dir / "results.md").read_text())


if __name__ == "__main__":
    main()
