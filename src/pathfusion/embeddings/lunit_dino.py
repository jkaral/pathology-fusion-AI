"""Lunit DINO ViT-S/16 (Kang et al., CVPR 2023); tile encoder, 384-d.

Ungated timm port ``1aurent/vit_small_patch16_224.lunit_dino`` (used for the CPU pilot).
The port's hub config reports generic mean/std 0.5, but Lunit trained and evaluates with
its own dataset statistics (lunit-io/benchmark-ssl-pathology), so those are pinned here.
"""

from __future__ import annotations

import torch

from pathfusion.embeddings.base import Preprocess, TileEncoder

LUNIT_MEAN = (0.70322989, 0.53606487, 0.66096631)
LUNIT_STD = (0.21716536, 0.26081574, 0.20723464)


class LunitDINO(TileEncoder):
    name = "lunit_dino"
    hf_repo = "1aurent/vit_small_patch16_224.lunit_dino"
    embed_dim = 384

    def _build(self) -> tuple[torch.nn.Module, Preprocess]:
        import timm

        model = timm.create_model(f"hf-hub:{self.hf_repo}", pretrained=True, num_classes=0)
        return model, Preprocess(224, LUNIT_MEAN, LUNIT_STD, "bicubic")

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(x)
