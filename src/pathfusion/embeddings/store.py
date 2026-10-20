"""In-memory access to cached tile embeddings, keyed by patient.

All agent trials and final models read features from here; foundation models are never
re-run. Bags are held as float16 tensors (~5 KB/tile for all three models combined).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from pathfusion.data.tiling import slide_seed


class FeatureStore:
    def __init__(self, bags: dict[str, dict[str, torch.Tensor]], dims: dict[str, int]):
        self._bags = bags          # model -> patient_id -> (N, D) float16
        self.dims = dims
        self.models = sorted(dims)

    @classmethod
    def from_dir(
        cls, emb_dir: str | Path, cohort: pd.DataFrame, models: list[str], max_tiles: int | None = None
    ) -> FeatureStore:
        """Load ``<WSI_ID>.h5`` for every cohort patient. Missing files raise."""
        emb_dir = Path(emb_dir)
        bags: dict[str, dict[str, torch.Tensor]] = {m: {} for m in models}
        dims: dict[str, int] = {}
        missing = [w for w in cohort["WSI_ID"] if not (emb_dir / f"{w}.h5").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} feature files missing, e.g. {missing[:5]}")
        for pid, wsi in zip(cohort["patient_id"], cohort["WSI_ID"]):
            with h5py.File(emb_dir / f"{wsi}.h5", "r") as fh:
                n = fh["coords"].shape[0]
                if n == 0:
                    raise ValueError(f"{wsi} has no tissue tiles")
                idx = np.arange(n)
                if max_tiles is not None and n > max_tiles:
                    rng = np.random.default_rng(slide_seed(wsi) + 1)
                    idx = np.sort(rng.choice(n, size=max_tiles, replace=False))
                for m in models:
                    arr = fh[f"feats/{m}"][...][idx]
                    bags[m][str(pid)] = torch.from_numpy(arr.astype(np.float16))
                    dims[m] = arr.shape[1]
        return cls(bags, dims)

    def bag(self, model: str, patient_id: str) -> torch.Tensor:
        return self._bags[model][patient_id]

    def n_tiles(self, patient_id: str) -> int:
        return int(self._bags[self.models[0]][patient_id].shape[0])

    def patients(self) -> list[str]:
        return sorted(self._bags[self.models[0]])

    def subset_models(self, models: list[str]) -> FeatureStore:
        return FeatureStore({m: self._bags[m] for m in models}, {m: self.dims[m] for m in models})
