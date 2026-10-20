"""Cohort construction, quirk handling and the committed split."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pathfusion.config import load_config
from pathfusion.data.metadata import audit_quirks, build_cohort, load_slides
from pathfusion.data.splits import (
    assert_no_leakage,
    load_cv_folds,
    make_cv_folds,
    make_split,
)


def test_cohort_one_row_per_patient(cfg, cohort):
    assert cohort["patient_id"].is_unique
    assert cohort["WSI_ID"].is_unique
    assert len(cohort) == 203
    assert set(cohort["label"]) == set(cfg.data.classes)


def test_duplicate_id_merged_and_alternative(cfg):
    slides = load_slides(cfg)
    rep = audit_quirks(slides)
    assert set(rep.duplicate_ids["SampleNumber"]) == {"8377886"}
    cohort, _, _ = build_cohort(cfg)
    assert (cohort["SampleNumber"] == "8377886").sum() == 1
    split_cfg = replace(cfg, data=replace(cfg.data, duplicate_id_policy="split"))
    cohort2, _, _ = build_cohort(split_cfg)
    assert (cohort2["SampleNumber"] == "8377886").sum() == 2


def test_conflicting_labels_follow_selected_slide(cfg, cohort):
    conflicts = cohort[cohort["label_conflict"]]
    assert set(conflicts["patient_id"]) == {"8225322", "8240634", "8245975", "8248415", "8248805"}
    for _, r in conflicts.iterrows():
        slide_labels = dict(s.split(":") for s in r["slide_labels"].split("|"))
        assert slide_labels[r["WSI_ID"]] == r["label"]


def test_conflict_exclude_policy(cfg):
    c = replace(cfg, data=replace(cfg.data, conflict_policy="exclude"))
    cohort, excluded, _ = build_cohort(c)
    assert len(cohort) == 198
    assert (excluded["reason"] == "conflicting slide labels").sum() == 5


def test_patients_without_available_slide_excluded(cfg):
    _, excluded, _ = build_cohort(cfg)
    assert set(excluded["patient_id"]) == {"8388318", "8219606", "8221926", "8226738", "8238589", "8243660"}


def test_selected_slide_is_lowest_available(cfg, cohort):
    slides = load_slides(cfg)
    avail = slides[slides["available"]]
    expected = avail.sort_values("wsi_num").groupby("SampleNumber")["WSI_ID"].first()
    got = cohort.set_index("SampleNumber")["WSI_ID"]
    assert (expected.loc[got.index] == got).all()


def test_committed_split_is_valid(cfg, cohort, split):
    assert_no_leakage(split)
    assert set(split["patient_id"]) == set(cohort["patient_id"])
    tab = pd.crosstab(split["label"], split["split"])
    assert (tab > 0).all().all(), "every class must appear in every split"
    frac = split["split"].value_counts(normalize=True)
    assert abs(frac["test"] - 0.2) < 0.03 and abs(frac["val"] - 0.1) < 0.03


def test_split_is_reproducible(cfg, cohort, split):
    regen = make_split(cohort, cfg.split.fractions, cfg.split.seed)
    merged = regen.merge(split, on="patient_id", suffixes=("", "_c"))
    assert (merged["split"] == merged["split_c"]).all()


def test_cv_folds_dev_only_and_stratified(cfg, split):
    folds = load_cv_folds(cfg)
    dev = set(split.loc[split["split"] != "test", "patient_id"])
    assert set(folds["patient_id"]) == dev
    assert folds.groupby("repeat")["patient_id"].apply(lambda s: s.is_unique).all()
    regen = make_cv_folds(split, cfg.split.cv_folds, cfg.split.cv_repeats, cfg.split.seed)
    assert regen.equals(folds)
    lab = split.set_index("patient_id")["label"]
    per_fold = folds.assign(label=folds["patient_id"].map(lab)).groupby("fold")["label"].nunique()
    assert (per_fold == 7).all()


def test_config_overrides_and_unknown_keys():
    c = load_config(overrides=["agent.budget=5", "train.lr=0.1"])
    assert c.agent.budget == 5 and c.train.lr == 0.1
    try:
        load_config(overrides=["agent.nope=1"])
    except KeyError:
        pass
    else:
        raise AssertionError("unknown key accepted")
