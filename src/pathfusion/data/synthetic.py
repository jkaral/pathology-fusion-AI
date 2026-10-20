"""Synthetic tile-embedding files for tests and pipeline smoke runs (NOT results).

Writes ``<WSI_ID>.h5`` files with the same layout as real extraction output. Each fake
"model" carries signal for a different subset of classes, so a working fusion pipeline
should beat each single model; this checks the plumbing, not any scientific claim.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_MODELS = {"uni2h": 32, "virchow2": 48, "phikonv2": 24}
# classes (by index) each fake model can "see"
INFORMATIVE = {"uni2h": [0, 1, 2, 3], "virchow2": [3, 4, 5, 6], "phikonv2": [0, 2, 4, 6]}


def write_synthetic_features(
    cohort: pd.DataFrame, out_dir: str | Path, n_classes: int = 7, models: dict[str, int] | None = None,
    tiles: tuple[int, int] = (40, 120), signal: float = 0.9, tumor_frac: float = 0.3, seed: int = 0,
) -> Path:
    models = models or DEFAULT_MODELS
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    protos = {m: rng.normal(size=(n_classes, d)) for m, d in models.items()}
    for _, row in cohort.iterrows():
        n = int(rng.integers(*tiles))
        y = int(row["label_idx"])
        tumor = rng.random(n) < tumor_frac
        coords = np.stack([rng.integers(0, 10_000, n), rng.integers(0, 10_000, n)], 1).astype(np.int32)
        with h5py.File(out_dir / f"{row['WSI_ID']}.h5", "w") as fh:
            fh.create_dataset("coords", data=coords)
            g = fh.create_group("feats")
            for m, d in models.items():
                x = rng.normal(size=(n, d)) * 1.0
                if y in INFORMATIVE[m]:
                    x[tumor] += signal * protos[m][y]
                else:  # uninformative: tumour tiles look like a random class
                    x[tumor] += signal * protos[m][rng.integers(0, n_classes)]
                g.create_dataset(m, data=x.astype(np.float16))
            fh.attrs["wsi_id"] = row["WSI_ID"]
            fh.attrs["n_tissue_tiles"] = n
            fh.attrs["synthetic"] = True
    return out_dir
