"""MIL networks, feature transforms and every fusion pipeline on synthetic features."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pathfusion.fusion.pipelines import build_pipeline  # noqa: E402
from pathfusion.fusion.spec import FusionSpec  # noqa: E402
from pathfusion.fusion.transforms import StreamTransform  # noqa: E402
from pathfusion.models.mil import MILClassifier, MultiStreamMIL, class_weights  # noqa: E402

MODELS = ("phikonv2", "uni2h", "virchow2")


def test_padding_does_not_change_prediction():
    torch.manual_seed(0)
    net = MILClassifier(8, 7, 16, 8, 0.0, meta_in=2, meta_dim=4).eval()
    x = torch.randn(1, 5, 8)
    meta = torch.randn(1, 2)
    out = net([x], torch.ones(1, 5, dtype=torch.bool), meta)
    xp = torch.cat([x, torch.randn(1, 3, 8)], dim=1)
    mask = torch.tensor([[True] * 5 + [False] * 3])
    assert torch.allclose(out, net([xp], mask, meta), atol=1e-5)


@pytest.mark.parametrize("fusion", ["gated", "attention", "concat"])
def test_multistream_shapes(fusion):
    net = MultiStreamMIL([8, 12], 7, 16, 8, 0.1, fusion=fusion, modality_dropout=0.5, meta_in=2, meta_dim=4)
    mask = torch.ones(3, 6, dtype=torch.bool)
    out = net([torch.randn(3, 6, 8), torch.randn(3, 6, 12)], mask, torch.randn(3, 2))
    assert out.shape == (3, 7)
    if fusion != "concat":
        assert net.last_gate is not None and net.last_gate.shape == (3, 2)


def test_modality_dropout_keeps_a_stream():
    net = MultiStreamMIL([4, 4], 7, 8, 4, 0.0, modality_dropout=0.99).train()
    hs = torch.ones(64, 2, 8)
    dropped = net._drop_streams(hs)
    assert (dropped.abs().sum(dim=(1, 2)) > 0).all()


def test_class_weights_balance():
    w = class_weights(torch.tensor([0, 0, 0, 1]), 3)
    assert w[2] == 0 and w[1] > w[0]


def test_transforms():
    bags = [torch.randn(50, 32, dtype=torch.float16) * 5 for _ in range(4)]
    l2 = StreamTransform("l2").fit(bags)
    assert torch.allclose(l2(bags[0]).norm(dim=-1), torch.ones(50), atol=1e-3)
    pca = StreamTransform("zscore", 16).fit(bags, seed=0)
    out = pca(bags[1])
    assert out.shape == (50, 16) and pca.out_dim == 16
    again = StreamTransform("zscore", 16).fit(bags, seed=0)
    assert torch.allclose(again(bags[1]), out)
    assert StreamTransform().fit(bags).is_identity


SPECS = [
    FusionSpec("meta_only"),
    FusionSpec("single", ("uni2h",), use_meta=False),
    FusionSpec("single", ("virchow2",)),
    FusionSpec("early_concat", MODELS, norm="l2"),
    FusionSpec("early_concat", MODELS, norm="zscore", pca_dim=64),
    FusionSpec("late", MODELS, late_rule="prob_mean"),
    FusionSpec("late", MODELS, late_rule="logit_mean"),
    FusionSpec("late", MODELS, late_rule="weighted"),
    FusionSpec("late", MODELS, late_rule="stacking", stack_c=0.1),
    FusionSpec("intermediate", MODELS, stream_fusion="gated", modality_dropout=0.1),
    FusionSpec("intermediate", ("uni2h", "virchow2"), stream_fusion="attention"),
    FusionSpec("intermediate", MODELS, stream_fusion="concat"),
]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.short())
def test_every_pipeline_fits_and_predicts(ctx, split, spec):
    train = sorted(split.loc[split["split"] == "train", "patient_id"])
    val = sorted(split.loc[split["split"] == "val", "patient_id"])
    logs = []
    pipe = build_pipeline(spec).fit(ctx, train, seed=0, val_ids=val,
                                    log_fn=(lambda r, s: logs.append(r)) if spec.family != "meta_only" else None,
                                    log_train_metrics=spec.family != "meta_only")
    p = pipe.predict(ctx, val)
    assert p.shape == (len(val), 7)
    assert np.allclose(p.sum(1), 1, atol=1e-4) and (p >= 0).all()
    if spec.family != "meta_only":
        assert len(logs) == ctx.train_cfg.epochs
        assert {"train/loss", "val/macro_auroc", "val/balanced_accuracy", "train/macro_auroc"} <= set(logs[-1])


def test_training_is_seed_deterministic(ctx, split):
    train = sorted(split.loc[split["split"] == "train", "patient_id"])[:60]
    val = sorted(split.loc[split["split"] == "val", "patient_id"])
    spec = FusionSpec("intermediate", MODELS, stream_fusion="gated")
    p1 = build_pipeline(spec).fit(ctx, train, seed=3).predict(ctx, val)
    p2 = build_pipeline(spec).fit(ctx, train, seed=3).predict(ctx, val)
    assert np.allclose(p1, p2, atol=1e-6)
