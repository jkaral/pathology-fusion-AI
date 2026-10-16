"""Encoder modules on randomly initialised architectures (no gated weights, no network).

Checks output dimensionality, Virchow2's token handling (CLS + mean of patch tokens,
register tokens excluded) and on-device preprocessing.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")

from pathfusion.embeddings.base import Preprocess, timm_preprocess  # noqa: E402
from pathfusion.embeddings.registry import EMBED_DIMS, get_encoder  # noqa: E402
from pathfusion.embeddings.uni2h import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

X = torch.randint(0, 255, (2, 224, 224, 3), dtype=torch.uint8)


def _attach(enc, model, pp):
    enc.model, enc.preprocess = model.eval(), pp
    return enc


def test_registry_dims():
    assert EMBED_DIMS == {"uni2h": 1536, "virchow2": 2560, "phikonv2": 1024}


def test_uni2h_architecture_and_card_preprocessing():
    kw = {"img_size": 224, "patch_size": 14, "depth": 2, "num_heads": 24, "init_values": 1e-5,
          "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
          "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU, "reg_tokens": 8,
          "dynamic_img_size": True}
    model = timm.create_model("vit_giant_patch14_224", pretrained=False, **kw)
    enc = _attach(get_encoder("uni2h"), model, Preprocess(224, IMAGENET_MEAN, IMAGENET_STD))
    assert enc.encode(X).shape == (2, 1536)


def test_virchow2_concatenates_cls_and_patch_mean():
    model = timm.create_model("vit_huge_patch14_224", pretrained=False, depth=2, img_size=224, init_values=1e-5,
                              num_classes=0, mlp_ratio=5.3375, global_pool="", reg_tokens=4,
                              mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU)
    enc = _attach(get_encoder("virchow2"), model, timm_preprocess(model))
    out = enc.encode(X)
    assert out.shape == (2, 2560)
    tokens = model.forward_features(enc.preprocess(X))
    assert model.num_prefix_tokens == 5 and tokens.shape[1] == 5 + 256
    assert torch.allclose(out[:, :1280], tokens[:, 0], atol=1e-4)
    assert torch.allclose(out[:, 1280:], tokens[:, 5:].mean(1), atol=1e-4)


def test_phikonv2_cls_token():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Dinov2Config(hidden_size=1024, num_hidden_layers=1, num_attention_heads=16,
                                    intermediate_size=256, patch_size=14, image_size=224)
    model = transformers.Dinov2Model(cfg)
    enc = _attach(get_encoder("phikonv2"), model, Preprocess(224, IMAGENET_MEAN, IMAGENET_STD))
    assert enc.encode(X).shape == (2, 1024)


def test_preprocess_resizes_and_normalises():
    pp = Preprocess(112, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), "bicubic")
    out = pp(torch.full((1, 224, 224, 3), 255, dtype=torch.uint8))
    assert out.shape == (1, 3, 112, 112) and torch.allclose(out, torch.ones_like(out), atol=1e-3)
