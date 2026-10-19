"""Virchow2 (Paige AI): ViT-H/14 with 4 register tokens, SwiGLU MLP; tile encoder, 2560-d.

Following the model card (https://huggingface.co/paige-ai/Virchow2), the tile embedding is
``concat(class_token, mean(patch_tokens))`` = 1280 + 1280 = 2560 dims, where patch tokens
exclude the class and register tokens (``num_prefix_tokens`` = 5).
"""

from __future__ import annotations

import torch

from pathfusion.embeddings.base import Preprocess, TileEncoder, timm_preprocess


class Virchow2(TileEncoder):
    name = "virchow2"
    hf_repo = "paige-ai/Virchow2"
    embed_dim = 2560

    def _build(self) -> tuple[torch.nn.Module, Preprocess]:
        import timm

        model = timm.create_model(
            f"hf-hub:{self.hf_repo}", pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,
        )
        return model, timm_preprocess(model)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        tokens = self.model.forward_features(x)  # type: ignore[operator]          # (B, 1 + 4 + 256, 1280)
        n_prefix = int(getattr(self.model, "num_prefix_tokens", 5))
        cls_tok = tokens[:, 0]
        patch_mean = tokens[:, n_prefix:].mean(dim=1)
        return torch.cat([cls_tok, patch_mean], dim=-1)
