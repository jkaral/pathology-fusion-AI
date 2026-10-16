"""Final evaluation: every results-table row trained identically, test touched once.

Rows (all with the shared ``TrainCfg`` recipe and the same seeds):

1. Metadata only (age + sex)              - logistic regression
2-4. Foundation model X alone              - ABMIL, image only (no metadata)
5. Best single model + metadata           - model chosen by development CV (agent references)
6. Fused (agent-selected) + metadata       - the agent's one-SE choice

Each row is trained on the *train* split with the validation split monitored for curves,
for every seed in ``eval.seeds``. Reported per row:

* mean +/- sd over seeds of test macro-AUROC and balanced accuracy (training noise);
* seed-ensemble (averaged probabilities) point estimate with a 95% stratified patient
  bootstrap CI (sampling noise of the 41 test patients);
* paired bootstrap differences fused - other rows on the same resampled patients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pathfusion.config import Config
from pathfusion.eval.metrics import (
    bootstrap_ci,
    compute_metrics,
    confusion,
    metric_fn,
    paired_bootstrap_diff,
    per_class_auroc,
)
from pathfusion.fusion.pipelines import build_pipeline
from pathfusion.fusion.spec import FusionSpec
from pathfusion.training.context import DataContext
from pathfusion.utils.logging import get_logger
from pathfusion.utils.tracking import Tracker

log = get_logger("final")


@dataclass
class Row:
    name: str
    label: str
    spec: FusionSpec
    test_probs: list[np.ndarray] = field(default_factory=list)   # per seed
    val_probs: list[np.ndarray] = field(default_factory=list)
    histories: list[list[dict]] = field(default_factory=list)
    extras: list[dict] = field(default_factory=list)

    @property
    def ensemble(self) -> np.ndarray:
        return np.mean(self.test_probs, axis=0)


def table_rows(models: list[str], best_single: str, fused: FusionSpec) -> list[Row]:
    rows = [Row("meta_only", "Metadata only (age + sex)", FusionSpec("meta_only"))]
    for m in models:
        rows.append(Row(f"single_{m}", f"{m} alone", FusionSpec("single", (m,), use_meta=False)))
    rows.append(Row("best_single_meta", f"Best single ({best_single}) + metadata",
                    FusionSpec("single", (best_single,), use_meta=True)))
    rows.append(Row("fused", f"Fused (agent): {fused.short()}", fused))
    return rows


def train_rows(cfg: Config, ctx: DataContext, rows: list[Row], train_ids: list[str], val_ids: list[str],
               test_ids: list[str]) -> None:
    for row in rows:
        for seed in cfg.eval.seeds:
            log.info("training %-18s seed %d", row.name, seed)
            tracker = Tracker(cfg.tracking, name=f"{row.name}-s{seed}", group="final", job_type="final",
                              config={"row": row.name, "spec": row.spec.to_dict(), "seed": seed,
                                      "train": cfg.to_dict()["train"]},
                              local_csv=cfg.paths.resolve("results_dir") / "curves" / f"{row.name}_s{seed}.csv")
            pipe = build_pipeline(row.spec)
            is_neural = row.spec.family != "meta_only"
            pipe.fit(ctx, train_ids, seed, val_ids=val_ids if is_neural else None,
                     log_fn=tracker.log if is_neural else None, log_train_metrics=is_neural)
            row.val_probs.append(pipe.predict(ctx, val_ids))
            row.test_probs.append(pipe.predict(ctx, test_ids))   # the single use of test, per row/seed
            row.histories.append(pipe.history)
            extra = {}
            if hasattr(pipe, "combiner"):
                extra["combiner"] = pipe.combiner.describe()
            row.extras.append(extra)
            yv = ctx.labels(val_ids)
            mv = compute_metrics(yv, row.val_probs[-1], ctx.n_classes)
            tracker.summary({"final/val_macro_auroc": mv["macro_auroc"],
                             "final/val_balanced_accuracy": mv["balanced_accuracy"]})
            tracker.finish()


def summarise(cfg: Config, ctx: DataContext, rows: list[Row], test_ids: list[str]) -> dict:
    y = ctx.labels(test_ids)
    n_c = ctx.n_classes
    out: dict = {"rows": [], "paired": {}, "n_test": len(test_ids), "seeds": cfg.eval.seeds}
    for row in rows:
        per_seed = [compute_metrics(y, p, n_c) for p in row.test_probs]
        ens = row.ensemble
        rec = {
            "name": row.name, "label": row.label, "spec": row.spec.to_dict(),
            "auroc_seed_mean": float(np.mean([m["macro_auroc"] for m in per_seed])),
            "auroc_seed_sd": float(np.std([m["macro_auroc"] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0,
            "bacc_seed_mean": float(np.mean([m["balanced_accuracy"] for m in per_seed])),
            "bacc_seed_sd": float(np.std([m["balanced_accuracy"] for m in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0,
            "macro_auroc": bootstrap_ci(y, ens, metric_fn("macro_auroc", n_c), cfg.eval.n_bootstrap, cfg.eval.ci, 0),
            "balanced_accuracy": bootstrap_ci(y, ens, metric_fn("balanced_accuracy", n_c), cfg.eval.n_bootstrap,
                                              cfg.eval.ci, 0),
            "per_class_auroc": per_class_auroc(y, ens, n_c).round(4).tolist(),
            "confusion": confusion(y, ens, n_c).tolist(),
            "extras": row.extras,
        }
        out["rows"].append(rec)
    fused = next(r for r in rows if r.name == "fused")
    for other in rows:
        if other.name == "fused":
            continue
        out["paired"][other.name] = {
            m: paired_bootstrap_diff(y, fused.ensemble, other.ensemble, metric_fn(m, n_c),
                                     cfg.eval.n_bootstrap, cfg.eval.ci, 1)
            for m in ("macro_auroc", "balanced_accuracy")
        }
    singles = [r for r in out["rows"] if r["name"].startswith("single_") or r["name"] == "best_single_meta"]
    f = next(r for r in out["rows"] if r["name"] == "fused")
    out["fused_beats_every_single_point"] = {
        m: bool(all(f[m]["point"] > s[m]["point"] for s in singles)) for m in ("macro_auroc", "balanced_accuracy")
    }
    out["fused_beats_every_single_seed_mean"] = {
        "macro_auroc": bool(all(f["auroc_seed_mean"] > s["auroc_seed_mean"] for s in singles)),
        "balanced_accuracy": bool(all(f["bacc_seed_mean"] > s["bacc_seed_mean"] for s in singles)),
    }
    return out


def error_analysis(ctx: DataContext, fused: Row, test_ids: list[str]) -> list[dict]:
    """Per-test-patient fused prediction with context for failure analysis.

    ``pred_in_multi_subtypes`` flags errors whose predicted pattern is listed among the
    patterns the pathologist recorded as present (HF ``Multi_Subtypes``), i.e. the model
    picked a real but non-dominant pattern.
    """
    probs = fused.ensemble
    out = []
    for pid, p in zip(test_ids, probs):
        row = ctx.cohort.loc[pid]
        true, pred = int(row["label_idx"]), int(p.argmax())
        multi = str(row.get("multi_subtypes", "") or "")
        pred_name = ctx.classes[pred]
        out.append({
            "patient_id": pid, "WSI_ID": row["WSI_ID"], "true": ctx.classes[true], "pred": pred_name,
            "correct": true == pred, "p_true": round(float(p[true]), 3), "p_pred": round(float(p[pred]), 3),
            "age": int(row["Age"]), "sex": row["Sex"], "diagnosis": row["diagnosis"], "multi_subtypes": multi,
            "pred_in_multi_subtypes": (true != pred) and pred_name.lower() in multi.lower(),
            "n_tiles": ctx.store.n_tiles(pid) if ctx.store is not None else None,
            "label_conflict": bool(row.get("label_conflict", False)),
        })
    return out
