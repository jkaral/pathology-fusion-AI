"""Post-hoc study: does training on every slide of a patient help? (development CV only)

The headline run used one slide per patient (the reduced protocol, test set used once on
2026-10-23). The CLWD release has 144 further slides of the 162 development patients
(same hospital, scanner and labels; ``data/extra_slides.py``). This script asks whether
adding them as extra *training* bags improves the models, without touching the test set:

* the same 5 development folds as the agent; test patients are never loaded;
* a fixed, pre-registered list of configurations (``STUDY_SPECS``): the three single models,
  the agent's selected fusion and the best fusion trial by CV; no selection happens here;
* each configuration is trained twice per seed on identical folds: headline data only vs
  headline + extra training slides. Validation folds are scored on the headline slide, so
  both arms are scored on exactly the same patients and slides;
* fold scores are averaged over seeds, then compared fold-paired with the Nadeau-Bengio
  corrected SE (as in the agent's acceptance rule);
* secondary, a different protocol: patient-level prediction that also averages each held-out
  patient's extra slides at inference ("all slides").

Usage:
    python scripts/09_extra_slides_study.py --embeddings-dir DIR    # DIR holds headline + extra .h5

Outputs: artifacts/extra_slides_study/{results.md, results.json, fold_scores.csv}
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathfusion.agent.evaluator import CVEvaluator, TrialResult  # noqa: E402
from pathfusion.config import load_config  # noqa: E402
from pathfusion.data.extra_slides import extra_cohort_rows, extra_slide_table, extras_by_parent  # noqa: E402
from pathfusion.data.splits import load_cv_folds, load_split  # noqa: E402
from pathfusion.embeddings.store import FeatureStore  # noqa: E402
from pathfusion.fusion.spec import FusionSpec  # noqa: E402
from pathfusion.training.context import DataContext  # noqa: E402
from pathfusion.utils.io import dump_json, git_commit  # noqa: E402
from pathfusion.utils.logging import get_logger  # noqa: E402
from pathfusion.utils.seed import seed_everything  # noqa: E402

log = get_logger("extra_slides_study")

MODELS = ("phikonv2", "uni2h", "virchow2")
# Fixed before the study was run. The two fusions are the agent's pick (#6) and its best
# fusion trial by CV (#5) in artifacts/agent/20261023-184822_local_llm.
STUDY_SPECS = [
    FusionSpec("single", ("uni2h",)),
    FusionSpec("single", ("virchow2",)),
    FusionSpec("single", ("phikonv2",)),
    FusionSpec("late", MODELS, late_rule="prob_mean"),
    FusionSpec("early_concat", MODELS, norm="zscore", pca_dim=128),
]


def paired(a: np.ndarray, b: np.ndarray, nb_ratio: float) -> tuple[float, float]:
    """Mean of a - b over folds and its Nadeau-Bengio corrected SE."""
    d = a - b
    se = float(np.sqrt((1.0 / len(d) + nb_ratio) * np.var(d, ddof=1))) if len(d) > 1 else 0.0
    return float(d.mean()), se


def seed_mean(results: list[TrialResult], key: str | None = None) -> np.ndarray:
    """Fold scores averaged over seeds (``key`` selects an ``extras`` entry)."""
    rows = [np.asarray(r.extras[key]) if key else r.fold_scores for r in results]
    return np.mean(rows, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings-dir", type=Path, default=None)
    ap.add_argument("--seeds", default="0,1,2", help="comma-separated training seeds")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    seeds = [int(s) for s in args.seeds.split(",")]
    seed_everything(cfg.seed)
    out_dir = args.out_dir or cfg.paths.resolve("artifacts") / "extra_slides_study"
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = pd.read_csv(cfg.paths.resolve("cohort_csv"), dtype={"patient_id": str, "SampleNumber": str})
    split = load_split(cfg)
    folds = load_cv_folds(cfg)
    dev_ids = set(split.loc[split["split"] != "test", "patient_id"])
    dev_cohort = cohort[cohort["patient_id"].isin(dev_ids)].reset_index(drop=True)
    manifest = pd.read_csv(cfg.paths.resolve("hf_manifest_csv"))
    extra = extra_slide_table(cohort, split, manifest, cfg.data.classes)
    assert not set(extra["parent_id"]) - dev_ids, "extra slides must belong to development patients"
    extra_rows = extra_cohort_rows(dev_cohort, extra)
    by_parent = extras_by_parent(extra)
    log.info("development patients %d; extra slides %d from %d patients (%d with a different label)",
             len(dev_cohort), len(extra), len(by_parent),
             int((extra_rows.set_index("patient_id")["label"].to_numpy()
                  != dev_cohort.set_index("patient_id").loc[extra["parent_id"], "label"].to_numpy()).sum()))

    emb_dir = args.embeddings_dir or cfg.paths.resolve("embeddings_dir")
    full = pd.concat([dev_cohort, extra_rows], ignore_index=True)
    store = FeatureStore.from_dir(emb_dir, full, cfg.extract.models, max_tiles=cfg.train.eval_tiles)
    ctx = DataContext.build(full, store, cfg.data.classes, cfg.train)

    rows, fold_rows = [], []
    t0 = time.perf_counter()
    for spec in STUDY_SPECS:
        base, plus = [], []
        for s in seeds:
            base.append(CVEvaluator(ctx, folds, cfg.agent.selection_metric, s).evaluate(spec))
            plus.append(CVEvaluator(ctx, folds, cfg.agent.selection_metric, s, extra_train=by_parent).evaluate(spec))
            ctx.cache.clear()  # keep memory flat; nothing is re-used across specs here
        nb = CVEvaluator(ctx, folds, cfg.agent.selection_metric, seeds[0]).nb_ratio
        f_base, f_plus, f_all = seed_mean(base), seed_mean(plus), seed_mean(plus, "fold_scores_all_slides")
        d_plus, se_plus = paired(f_plus, f_base, nb)
        d_all, se_all = paired(f_all, f_base, nb)

        def pooled(res: list[TrialResult], key: str = "pooled") -> dict[str, float]:
            vals = [r.pooled if key == "pooled" else r.extras[key] for r in res]
            return {m: float(np.mean([v[m] for v in vals])) for m in ("macro_auroc", "balanced_accuracy")}

        row = {"config": spec.short(), "spec": spec.to_dict(),
               "cv_headline": float(f_base.mean()), "cv_plus_extra": float(f_plus.mean()),
               "gain_plus_extra": d_plus, "gain_plus_extra_se": se_plus,
               "cv_all_slides": float(f_all.mean()), "gain_all_slides": d_all, "gain_all_slides_se": se_all,
               "pooled_headline": pooled(base), "pooled_plus_extra": pooled(plus),
               "pooled_all_slides": pooled(plus, "pooled_all_slides")}
        rows.append(row)
        for k in range(len(f_base)):
            fold_rows.append({"config": spec.short(), "fold": k, "headline": f_base[k],
                              "plus_extra": f_plus[k], "all_slides": f_all[k]})
        log.info("%s: CV %.3f -> %.3f (gain %+.3f ± %.3f); all-slides %.3f (%+.3f ± %.3f)", spec.short(),
                 row["cv_headline"], row["cv_plus_extra"], d_plus, se_plus, row["cv_all_slides"], d_all, se_all)

    meta = {"seeds": seeds, "metric": cfg.agent.selection_metric, "n_dev": len(dev_cohort),
            "n_extra_slides": len(extra), "n_patients_with_extra": len(by_parent),
            "minutes": round((time.perf_counter() - t0) / 60, 1), "git_commit": git_commit(),
            "embeddings_dir": str(emb_dir), "test_set_used": False}
    dump_json({"meta": meta, "rows": rows}, out_dir / "results.json")
    pd.DataFrame(fold_rows).to_csv(out_dir / "fold_scores.csv", index=False)
    (out_dir / "results.md").write_text(render(meta, rows))
    log.info("wrote %s", out_dir / "results.md")


def render(meta: dict[str, object], rows: list[dict[str, object]]) -> str:
    def g(r: dict[str, object], k: str) -> str:
        return f"{r[k]:+.3f} ± {r[k + '_se']:.3f}"

    lines = [
        "# Extra-slides study (development CV only; test set not used)",
        "",
        f"{meta['n_dev']} development patients, 5 folds, seeds {meta['seeds']}; "
        f"{meta['n_extra_slides']} extra slides from {meta['n_patients_with_extra']} patients "
        "(same CLWD release). Fold scores averaged over seeds; gain = fold-paired difference "
        "± Nadeau-Bengio corrected SE. Generated by scripts/09_extra_slides_study.py.",
        "",
        "| Configuration | Macro AUROC, headline data | + extra training slides | Gain | "
        "All slides at inference | Gain | Bal. acc. headline → + extra → all slides |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        ph, pp, pa = r["pooled_headline"], r["pooled_plus_extra"], r["pooled_all_slides"]
        lines.append(
            f"| {r['config']} | {r['cv_headline']:.3f} | {r['cv_plus_extra']:.3f} | {g(r, 'gain_plus_extra')} | "
            f"{r['cv_all_slides']:.3f} | {g(r, 'gain_all_slides')} | "
            f"{ph['balanced_accuracy']:.3f} → {pp['balanced_accuracy']:.3f} → {pa['balanced_accuracy']:.3f} |"
        )
    lines += ["", "Columns 2-4 score the same headline slide per held-out patient. "
              "'All slides at inference' averages each held-out patient's extra slides too, "
              "which is a different protocol from the headline test evaluation.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
