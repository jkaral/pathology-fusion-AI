"""Common interface for tile encoders.

Each encoder owns its preprocessing: input size, interpolation and normalisation statistics
are read from *that model's own* published config (timm ``pretrained_cfg`` or the HF image
processor), never shared. Tiles arrive as uint8 (B, H, W, 3) tensors already on the GPU,
and normalisation happens on-device, which avoids a per-tile PIL round trip per model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Preprocess:
    input_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bicubic"

    def __call__(self, tiles_uint8: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) uint8 -> (B, 3, S, S) normalised float32."""
        x = tiles_uint8.permute(0, 3, 1, 2).float().div_(255.0)
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            mode = self.interpolation if self.interpolation in ("bilinear", "bicubic") else "bilinear"
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode=mode,
                              antialias=True, align_corners=False)
        mean = torch.tensor(self.mean, device=x.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std


class TileEncoder(ABC):
    """A frozen foundation model mapping tiles to embeddings."""

    name: str = ""
    hf_repo: str = ""
    embed_dim: int = 0

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.preprocess: Preprocess | None = None
        self.device = torch.device("cpu")

    @abstractmethod
    def _build(self) -> tuple[torch.nn.Module, Preprocess]:
        """Instantiate pretrained weights and return (model, preprocess)."""

    @abstractmethod
    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalised (B, 3, S, S) -> (B, embed_dim)."""

    def load(self, device: torch.device | str) -> TileEncoder:
        self.device = torch.device(device)
        model, self.preprocess = self._build()
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        return self

    @torch.inference_mode()
    def encode(self, tiles_uint8: torch.Tensor) -> torch.Tensor:
        """Mixed precision: fp16 on CUDA; bf16 on CPUs with native bf16 (AMX/AVX512-BF16)."""
        assert self.model is not None and self.preprocess is not None, "call load() first"
        x = self.preprocess(tiles_uint8.to(self.device, non_blocking=True))
        if self.device.type == "cuda":
            amp_dtype, use_amp = torch.float16, True
        else:
            amp_dtype, use_amp = torch.bfloat16, cpu_has_native_bf16()
        with torch.autocast(device_type=self.device.type, dtype=amp_dtype, enabled=use_amp):
            out = self._forward(x)
        if out.shape[-1] != self.embed_dim:
            raise RuntimeError(f"{self.name}: expected dim {self.embed_dim}, got {out.shape[-1]}")
        return out.float()

    def describe(self) -> dict[str, object]:
        pp = self.preprocess
        return {
            "name": self.name, "hf_repo": self.hf_repo, "embed_dim": self.embed_dim,
            "input_size": pp.input_size if pp else None, "mean": pp.mean if pp else None,
            "std": pp.std if pp else None, "interpolation": pp.interpolation if pp else None,
        }


def cpu_has_native_bf16() -> bool:
    """True on CPUs with AMX/AVX512-BF16, where bf16 autocast is ~3x faster than fp32."""
    try:
        flags = open("/proc/cpuinfo").read()
    except OSError:
        return False
    return "amx_bf16" in flags or "avx512_bf16" in flags


def hf_processor_preprocess(repo: str) -> Preprocess:
    """Read size/mean/std/resampling from a model's own Hugging Face image processor."""
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(repo)
    size = proc.crop_size if getattr(proc, "do_center_crop", False) else proc.size
    side = int(size.get("height") or size.get("shortest_edge") or 224)
    interp = {2: "bilinear", 3: "bicubic"}.get(int(getattr(proc, "resample", 3)), "bicubic")
    return Preprocess(side, tuple(proc.image_mean), tuple(proc.image_std), interp)  # type: ignore[arg-type]


def timm_preprocess(model: torch.nn.Module) -> Preprocess:
    """Read the model's own data config from timm (mean/std/size/interpolation)."""
    from timm.data import resolve_data_config

    cfg = resolve_data_config({}, model=model)
    return Preprocess(
        input_size=int(cfg["input_size"][-1]),
        mean=tuple(float(v) for v in cfg["mean"]),  # type: ignore[arg-type]
        std=tuple(float(v) for v in cfg["std"]),  # type: ignore[arg-type]
        interpolation=str(cfg.get("interpolation", "bicubic")),
    )
