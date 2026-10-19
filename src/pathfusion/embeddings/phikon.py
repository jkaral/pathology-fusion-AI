"""Phikon (Owkin): ViT-B/16 trained with iBOT on 40M TCGA tiles; tile encoder, 768-d.

Ungated, so it runs without credentials (used for the CPU pilot). Embedding: CLS token of
``last_hidden_state`` (model card). Mean/std/size come from its own HF image processor.
"""

from __future__ import annotations

import torch

from pathfusion.embeddings.base import Preprocess, TileEncoder, hf_processor_preprocess


class Phikon(TileEncoder):
    name = "phikon"
    hf_repo = "owkin/phikon"
    embed_dim = 768

    def _build(self) -> tuple[torch.nn.Module, Preprocess]:
        from transformers import AutoModel

        return AutoModel.from_pretrained(self.hf_repo), hf_processor_preprocess(self.hf_repo)

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(pixel_values=x).last_hidden_state[:, 0, :]
