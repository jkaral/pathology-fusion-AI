"""Streaming extraction with stand-in encoders (no gated weights needed)."""

from __future__ import annotations

from dataclasses import replace

import h5py
import numpy as np
import pandas as pd
import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from pathfusion.embeddings.base import Preprocess, TileEncoder  # noqa: E402
from pathfusion.embeddings.extract import is_complete, local_source, run_extraction  # noqa: E402
from pathfusion.embeddings.store import FeatureStore  # noqa: E402
from tests.test_tiling import fake_slide  # noqa: E402


class FakeEncoder(TileEncoder):
    def __init__(self, name: str, dim: int, mean: float):
        super().__init__()
        self.name, self.embed_dim, self._mean = name, dim, mean

    def _build(self):
        torch.manual_seed(0)
        net = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(4), torch.nn.Flatten(), torch.nn.Linear(48, self.embed_dim))
        return net, Preprocess(32, (self._mean,) * 3, (0.25,) * 3, "bilinear")

    def _forward(self, x):
        return self.model(x)


def test_extraction_writes_aligned_features_and_resumes(cfg, tmp_path):
    slides = tmp_path / "jpg"
    slides.mkdir()
    for i, wsi in enumerate(["WSI-1", "WSI-2"]):
        Image.fromarray(fake_slide(seed=i)).save(slides / f"{wsi}.jpg", quality=90)
    c = replace(cfg, tiling=replace(cfg.tiling, tile_size=64, stride=64, max_tiles=None),
                extract=replace(cfg.extract, batch_size=16))
    encs = [FakeEncoder("uni2h", 12, 0.5).load("cpu"), FakeEncoder("virchow2", 20, 0.4).load("cpu")]
    out = tmp_path / "feats"
    man = run_extraction(c, ["WSI-1", "WSI-2"], local_source(slides), encs, out, tmp_path / "qc", {})
    for wsi in ("WSI-1", "WSI-2"):
        assert is_complete(out / f"{wsi}.h5", ["uni2h", "virchow2"])
        with h5py.File(out / f"{wsi}.h5") as fh:
            n = fh["coords"].shape[0]
            assert n > 5 and fh["feats/uni2h"].shape == (n, 12) and fh["feats/virchow2"].shape == (n, 20)
            assert fh["feats/uni2h"].dtype == np.float16
            assert int(fh.attrs["width_20x"]) == 3072 // 4
        assert (tmp_path / "qc" / f"{wsi}.jpg").exists()
    assert len(pd.read_csv(man)) == 2
    # resume: nothing left to do, manifest unchanged
    run_extraction(c, ["WSI-1", "WSI-2"], local_source(slides), encs, out, None, {})
    assert len(pd.read_csv(man)) == 2
    cohort = pd.DataFrame({"patient_id": ["a", "b"], "WSI_ID": ["WSI-1", "WSI-2"]})
    store = FeatureStore.from_dir(out, cohort, ["uni2h", "virchow2"], max_tiles=5)
    assert store.bag("uni2h", "a").shape == (5, 12) and store.dims == {"uni2h": 12, "virchow2": 20}
