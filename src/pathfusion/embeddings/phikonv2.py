"""Phikon-v2 (Owkin): ViT-L/16 DINOv2-trained on PANCAN-XL; tile encoder, 1024-d.

Following the model card (https://huggingface.co/owkin/phikon-v2) the embedding is the
CLS token of ``last_hidden_state``. Mean/std/size come from its own HF image processor.
"""

from __future__ import annotations

import torch

from pathfusion.embeddings.base import Preprocess, TileEncoder


class PhikonV2(TileEncoder):
    name = "phikonv2"
    hf_repo = "owkin/phikon-v2"
    embed_dim = 1024

    def _build(self) -> tuple[torch.nn.Module, Preprocess]:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(self.hf_repo)
        model = AutoModel.from_pretrained(self.hf_repo)
        size = processor.crop_size if getattr(processor, "do_center_crop", False) else processor.size
        side = int(size.get("height") or size.get("shortest_edge") or 224)
        resample = getattr(processor, "resample", 3)
        interp = {2: "bilinear", 3: "bicubic"}.get(int(resample), "bicubic")
        pp = Preprocess(
            input_size=side, mean=tuple(processor.image_mean),  # type: ignore[arg-type]
            std=tuple(processor.image_std), interpolation=interp,  # type: ignore[arg-type]
        )
        return model, pp

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        return self.model(pixel_values=x).last_hidden_state[:, 0, :]
