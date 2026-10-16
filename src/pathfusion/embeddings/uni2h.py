"""UNI2-h (Mahmood Lab): ViT-H/14 with 8 register tokens, SwiGLU MLP; tile encoder, 1536-d.

Architecture kwargs follow the model card (https://huggingface.co/MahmoodLab/UNI2-h).
Output: the pooled class token (timm ``num_classes=0``). Preprocessing: ImageNet mean/std
at 224 px, as resolved from the checkpoint's own timm config.
"""

from __future__ import annotations

import torch

from pathfusion.embeddings.base import Preprocess, TileEncoder, timm_preprocess


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
        return model, timm_preprocess(model)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(x)
