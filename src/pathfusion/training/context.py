"""Everything a pipeline needs to fit/predict, bundled once per run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import torch

from pathfusion.config import TrainCfg
from pathfusion.embeddings.store import FeatureStore


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class DataContext:
    cohort: pd.DataFrame                 # one row per patient, index = patient_id
    store: FeatureStore | None           # None for metadata-only runs
    classes: list[str]
    train_cfg: TrainCfg
    device: torch.device
    cache: dict[Any, Any] = field(default_factory=dict)  # fitted sub-models, keyed by (spec, ids, seed)

    @classmethod
    def build(cls, cohort: pd.DataFrame, store: FeatureStore | None, classes: list[str],
              train_cfg: TrainCfg) -> DataContext:
        c = cohort.copy()
        c["patient_id"] = c["patient_id"].astype(str)
        return cls(c.set_index("patient_id", drop=False), store, classes, train_cfg,
                   resolve_device(train_cfg.device))

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def labels(self, ids: list[str]) -> np.ndarray:
        return self.cohort.loc[ids, "label_idx"].to_numpy(np.int64)

    def age_stats(self, ids: list[str]) -> tuple[float, float]:
        a = self.cohort.loc[ids, "Age"].to_numpy(float)
        return float(a.mean()), float(a.std() + 1e-6)

    def covariates(self, ids: list[str], age_stats: tuple[float, float]) -> np.ndarray:
        sub = self.cohort.loc[ids]
        age_z = (sub["Age"].to_numpy(float) - age_stats[0]) / age_stats[1]
        return np.stack([age_z, sub["sex_code"].to_numpy(float)], 1).astype(np.float32)
