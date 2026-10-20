"""Patient-level, subtype-stratified split and the development-set CV folds.

Both are written once by ``scripts/00_prepare_data.py`` and committed. Every later stage
loads them through :func:`load_split` / :func:`load_cv_folds`, which verify that the
files have not changed (checksum) and that no patient crosses partitions.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pathfusion.config import Config
from pathfusion.utils.io import file_sha256

SPLITS = ("train", "val", "test")


def _round_half_up(x: float) -> int:
    return int(np.floor(x + 0.5))


def make_split(cohort: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Per-class allocation so every class appears in every split (Cribriform has ~15 pts).

    For each class with ``n`` patients: ``n_test = round(f_test * n)``,
    ``n_val = max(1, round(f_val * n))``, remainder to train. Patients are shuffled
    with a seeded RNG before allocation.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1, got {fractions}")
    rng = np.random.default_rng(seed)
    assign: dict[str, str] = {}
    for _, grp in cohort.groupby("label", sort=True):
        ids = grp["patient_id"].sort_values().to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_test = _round_half_up(fractions["test"] * n)
        n_val = max(1, _round_half_up(fractions["val"] * n))
        if n - n_test - n_val < 1:
            raise ValueError(f"Class with {n} patients too small to split")
        for i, pid in enumerate(ids):
            assign[pid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    out = cohort[["patient_id", "SampleNumber", "WSI_ID", "label"]].copy()
    out["split"] = out["patient_id"].map(assign)
    return out.sort_values(["split", "patient_id"]).reset_index(drop=True)


def make_cv_folds(split: pd.DataFrame, n_folds: int, n_repeats: int, seed: int) -> pd.DataFrame:
    """Stratified K-fold over the development set (train + val). Test is never included."""
    dev = split[split["split"] != "test"].sort_values("patient_id").reset_index(drop=True)
    rows = []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + r)
        for k, (_, held) in enumerate(skf.split(dev["patient_id"], dev["label"])):
            for pid in dev.loc[held, "patient_id"]:
                rows.append({"patient_id": pid, "repeat": r, "fold": k})
    return pd.DataFrame(rows).sort_values(["repeat", "fold", "patient_id"]).reset_index(drop=True)


def write_with_checksum(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    digest = file_sha256(path)
    path.with_suffix(".sha256").write_text(digest + "\n")
    return digest


def _verify(path: Path) -> None:
    ref = path.with_suffix(".sha256")
    if ref.exists() and ref.read_text().strip() != file_sha256(path):
        raise RuntimeError(f"{path} does not match its committed checksum; refusing to use it")


def load_split(cfg: Config) -> pd.DataFrame:
    path = cfg.paths.resolve("split_csv")
    _verify(path)
    split = pd.read_csv(path, dtype={"patient_id": str, "SampleNumber": str})
    assert_no_leakage(split)
    return split


def load_cv_folds(cfg: Config) -> pd.DataFrame:
    path = cfg.paths.resolve("cv_folds_csv")
    _verify(path)
    folds = pd.read_csv(path, dtype={"patient_id": str})
    split = load_split(cfg)
    test_ids = set(split.loc[split["split"] == "test", "patient_id"])
    if test_ids & set(folds["patient_id"]):
        raise RuntimeError("CV folds contain test patients")
    return folds


def assert_no_leakage(split: pd.DataFrame) -> None:
    """A patient (and a SampleNumber) must belong to exactly one partition."""
    for key in ("patient_id", "SampleNumber", "WSI_ID"):
        n_parts = split.groupby(key)["split"].nunique()
        if (n_parts > 1).any():
            raise RuntimeError(f"Leakage: {key} in several splits: {list(n_parts[n_parts > 1].index)}")
    if set(split["split"]) - set(SPLITS):
        raise RuntimeError(f"Unexpected split names {set(split['split'])}")


def split_summary(split: pd.DataFrame) -> pd.DataFrame:
    tab = pd.crosstab(split["label"], split["split"]).reindex(columns=list(SPLITS), fill_value=0)
    tab.loc["TOTAL"] = tab.sum()
    return tab
