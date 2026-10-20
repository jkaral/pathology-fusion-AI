"""Phikon-v2 (Owkin): ViT-L/16 DINOv2-trained on PANCAN-XL; tile encoder, 1024-d.

Following the model card (https://huggingface.co/owkin/phikon-v2) the embedding is the
CLS token of ``last_hidden_state``. Mean/std/size come from its own HF image processor.
"""

from __future__ import annotations

import torch

from pathfusion.embeddings.base import Preprocess, TileEncoder, hf_processor_preprocess


class PhikonV2(TileEncoder):
    name = "phikonv2"
    hf_repo = "owkin/phikon-v2"
    embed_dim = 1024

    def _build(self) -> tuple[torch.nn.Module, Preprocess]:
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.hf_repo), hf_processor_preprocess(self.hf_repo)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(pixel_values=x).last_hidden_state[:, 0, :]
