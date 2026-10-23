"""Post-hoc extra-slides study: slide listing rules and the evaluator's extra training bags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pathfusion.data.extra_slides import SEP, extra_cohort_rows, extra_slide_table, extras_by_parent


@pytest.fixture(scope="module")
def extra(cfg, cohort, split) -> pd.DataFrame:
    manifest = pd.read_csv(cfg.paths.resolve("hf_manifest_csv"))
    return extra_slide_table(cohort, split, manifest, cfg.data.classes)


def test_extra_slides_never_include_test_or_headline_slides(extra, cohort, split):
    test_ids = set(split.loc[split["split"] == "test", "patient_id"])
    assert len(extra) == 144 and extra["parent_id"].nunique() == 109
    assert not set(extra["parent_id"]) & test_ids
    assert not set(extra["WSI_ID"]) & set(cohort["WSI_ID"])
    assert extra["WSI_ID"].is_unique


def test_extra_rows_keep_parent_metadata_and_own_label(extra, cohort):
    rows = extra_cohort_rows(cohort, extra)
    assert list(rows.columns) == list(cohort.columns)
    parents = cohort.set_index("patient_id")
    for _, r in rows.head(20).iterrows():
        pid, wsi = r["patient_id"].split(SEP)
        assert wsi == r["WSI_ID"]
        assert r["Age"] == parents.loc[pid, "Age"] and r["sex_code"] == parents.loc[pid, "sex_code"]
    # labels come from the slide itself; a few differ from the patient's headline label
    assert (rows["label"].to_numpy() != parents.loc[extra["parent_id"], "label"].to_numpy()).sum() == 4
    assert sum(len(v) for v in extras_by_parent(extra).values()) == len(extra)


def test_evaluator_adds_extra_training_bags(cfg, cohort, split, extra, synthetic_dir):
    pytest.importorskip("torch")
    from pathfusion.agent.evaluator import CVEvaluator
    from pathfusion.data.splits import load_cv_folds
    from pathfusion.data.synthetic import write_synthetic_features
    from pathfusion.embeddings.store import FeatureStore
    from pathfusion.fusion.spec import FusionSpec
    from pathfusion.training.context import DataContext

    dev = cohort[cohort["patient_id"].isin(set(split.loc[split["split"] != "test", "patient_id"]))]
    rows = extra_cohort_rows(dev.reset_index(drop=True), extra)
    write_synthetic_features(rows, synthetic_dir, seed=0)  # same prototypes as the headline slides
    full = pd.concat([dev, rows], ignore_index=True)
    store = FeatureStore.from_dir(synthetic_dir, full, cfg.extract.models, max_tiles=cfg.train.eval_tiles)
    ctx = DataContext.build(full, store, cfg.data.classes, cfg.train)
    folds = load_cv_folds(cfg)
    by_parent = extras_by_parent(extra)

    spec = FusionSpec("single", ("uni2h",))
    base = CVEvaluator(ctx, folds, "macro_auroc", seed=0).evaluate(spec)
    ev = CVEvaluator(ctx, folds, "macro_auroc", seed=0, extra_train=by_parent)
    plus = ev.evaluate(spec)
    assert len(base.fold_scores) == len(plus.fold_scores) == 5
    assert plus.oof_probs.shape == base.oof_probs.shape == (len(dev), len(cfg.data.classes))
    assert not np.allclose(base.oof_probs, plus.oof_probs)          # extra bags changed training
    assert len(plus.extras["fold_scores_all_slides"]) == 5
    assert not base.extras                                           # headline runs are unchanged
    with pytest.raises(ValueError, match="stacking"):
        ev.evaluate(FusionSpec("late", ("uni2h", "virchow2"), late_rule="stacking"))
