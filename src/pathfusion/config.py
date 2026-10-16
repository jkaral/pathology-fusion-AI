"""Typed configuration loaded from ``conf/default.yaml`` with dotted CLI overrides.

Example::

    cfg = load_config(overrides=["agent.budget=16", "seed=1"])
    cfg.agent.budget  # -> 16
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "conf" / "default.yaml"


@dataclass
class PathsCfg:
    metadata_csv: str = "data/CLWD.csv"
    metadata_hf_csv: str = "data/CLWD_hf.csv"
    hf_manifest_csv: str = "data/hf_file_manifest.csv"
    artifacts: str = "artifacts"
    cohort_csv: str = "artifacts/data/cohort.csv"
    split_csv: str = "artifacts/splits/split_v1.csv"
    cv_folds_csv: str = "artifacts/splits/cv_folds_v1.csv"
    embeddings_dir: str = "artifacts/embeddings"
    qc_dir: str = "artifacts/qc"
    agent_dir: str = "artifacts/agent"
    results_dir: str = "artifacts/results"

    def resolve(self, name: str) -> Path:
        """Absolute path for a configured entry (relative paths are repo-relative)."""
        p = Path(getattr(self, name))
        return p if p.is_absolute() else REPO_ROOT / p


@dataclass
class DataCfg:
    hf_repo: str = "kmmuleelab/Lung_Pathology_Image_JPG"
    label_column: str = "Tumor Subtype"
    classes: list[str] = field(
        default_factory=lambda: [
            "In situ", "Lepidic", "Acinar", "Papillary", "Micropapillary", "Solid", "Cribriform",
        ]
    )
    duplicate_id_policy: str = "merge"
    conflict_policy: str = "selected_slide"
    slide_selection: str = "lowest_available_wsi_id"


@dataclass
class SplitCfg:
    version: str = "v1"
    fractions: dict[str, float] = field(
        default_factory=lambda: {"train": 0.7, "val": 0.1, "test": 0.2}
    )
    seed: int = 2026
    cv_folds: int = 5
    cv_repeats: int = 1


@dataclass
class TilingCfg:
    native_magnification: int = 80
    target_magnification: int = 20
    tile_size: int = 224
    stride: int = 224
    thumbnail_downsample: int = 16
    min_tissue_fraction: float = 0.5
    saturation_floor: float = 0.05
    max_tiles: int | None = 4096

    @property
    def downsample(self) -> int:
        return self.native_magnification // self.target_magnification


@dataclass
class ExtractCfg:
    models: list[str] = field(default_factory=lambda: ["uni2h", "virchow2", "phikonv2"])
    batch_size: int = 128
    num_workers: int = 4
    dtype: str = "float16"
    prefetch: int = 1
    shard_index: int = 0
    num_shards: int = 1
    keep_jpg: bool = False


@dataclass
class TrainCfg:
    hidden_dim: int = 256
    attn_dim: int = 128
    dropout: float = 0.25
    meta_dim: int = 16
    lr: float = 2e-4
    weight_decay: float = 1e-2
    epochs: int = 30
    batch_size: int = 16
    train_tiles: int = 512
    eval_tiles: int = 4096
    class_weighting: bool = True
    warmup_epochs: int = 2
    device: str = "auto"


@dataclass
class LLMCfg:
    local_model: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 400
    temperature: float = 0.0
    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    model: str = "gemini-2.5-flash"
    api_key_env: str = "GEMINI_API_KEY"
    max_retries: int = 1


@dataclass
class AgentCfg:
    budget: int = 24
    min_trials: int = 8
    patience: int = 8
    acceptance_z: float = 1.0
    selection_metric: str = "macro_auroc"
    bootstrap_optimism: int = 200
    proposer: str = "auto"
    llm: LLMCfg = field(default_factory=LLMCfg)


@dataclass
class EvalCfg:
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    n_bootstrap: int = 2000
    ci: float = 0.95


@dataclass
class TrackingCfg:
    wandb_project: str = "pathology-fusion"
    wandb_entity: str | None = None
    mode: str = "auto"


@dataclass
class Config:
    seed: int = 0
    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    tiling: TilingCfg = field(default_factory=TilingCfg)
    extract: ExtractCfg = field(default_factory=ExtractCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    agent: AgentCfg = field(default_factory=AgentCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    tracking: TrackingCfg = field(default_factory=TrackingCfg)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls: type, data: dict[str, Any]) -> Any:
    """Recursively instantiate dataclass ``cls`` from a dict, rejecting unknown keys."""
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint) and isinstance(value, dict):
            kwargs[name] = _build(hint, value)  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _set_dotted(tree: dict[str, Any], dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    node = tree
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load YAML config and apply ``key.sub=value`` overrides (values parsed as YAML)."""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key=value, got {item!r}")
        key, val = item.split("=", 1)
        _set_dotted(raw, key.strip(), yaml.safe_load(val))
    return _build(Config, raw)
