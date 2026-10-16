"""Shared fixtures: real metadata + committed split, synthetic features, tiny training recipe."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathfusion.config import Config, load_config  # noqa: E402

TINY_TRAIN = [
    "train.hidden_dim=16", "train.attn_dim=8", "train.meta_dim=4", "train.epochs=4",
    "train.batch_size=16", "train.train_tiles=32", "train.eval_tiles=256", "train.warmup_epochs=1",
    "train.lr=0.005", "train.device=cpu", "tracking.mode=disabled",
]


@pytest.fixture(scope="session")
def cfg() -> Config:
    return load_config(overrides=TINY_TRAIN)


@pytest.fixture(scope="session")
def cohort(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.paths.resolve("cohort_csv"), dtype={"patient_id": str, "SampleNumber": str})


@pytest.fixture(scope="session")
def split(cfg: Config) -> pd.DataFrame:
    from pathfusion.data.splits import load_split

    return load_split(cfg)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory, cohort: pd.DataFrame) -> Path:
    pytest.importorskip("torch")
    from pathfusion.data.synthetic import write_synthetic_features

    out = tmp_path_factory.mktemp("synthetic_feats")
    return write_synthetic_features(cohort, out, seed=0)


@pytest.fixture(scope="session")
def ctx(cfg: Config, cohort: pd.DataFrame, synthetic_dir: Path):
    from pathfusion.embeddings.store import FeatureStore
    from pathfusion.training.context import DataContext

    store = FeatureStore.from_dir(synthetic_dir, cohort, cfg.extract.models, max_tiles=cfg.train.eval_tiles)
    return DataContext.build(cohort, store, cfg.data.classes, cfg.train)
