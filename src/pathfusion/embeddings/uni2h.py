"""UNI2-h (Mahmood Lab): ViT-H/14 with 8 register tokens, SwiGLU MLP; tile encoder, 1536-d.

Architecture kwargs follow the model card (https://huggingface.co/MahmoodLab/UNI2-h).
Output: the pooled class token (timm ``num_classes=0``). Preprocessing is pinned to the
model card's transform (ImageNet mean/std, 224 px) rather than timm's architecture default
(mean 0.5 for this ViT family); the hub config is checked against it and any mismatch logged.
"""

from __future__ import annotations

import torch

from pathfusion.embeddings.base import Preprocess, TileEncoder, timm_preprocess
from pathfusion.utils.logging import get_logger

log = get_logger("uni2h")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class UNI2h(TileEncoder):
    name = "uni2h"
    hf_repo = "MahmoodLab/UNI2-h"
    embed_dim = 1536

    def _build(self) -> tuple[torch.nn.Module, Preprocess]:
        import timm

        kwargs = {
            "img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24,
            "init_values": 1e-5, "embed_dim": 1536, "mlp_ratio": 2.66667 * 2,
            "num_classes": 0, "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU,
            "reg_tokens": 8, "dynamic_img_size": True,
        }
        model = timm.create_model(f"hf-hub:{self.hf_repo}", pretrained=True, **kwargs)
        card = Preprocess(224, IMAGENET_MEAN, IMAGENET_STD, "bilinear")
        hub = timm_preprocess(model)
        if any(abs(a - b) > 1e-3 for a, b in zip(hub.mean + hub.std, card.mean + card.std)):
            log.warning("hub config normalisation %s/%s differs from model card; using model card",
                        hub.mean, hub.std)
        return model, card

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(x)
