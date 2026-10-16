"""Metrics, bootstrap, fusion-spec validation."""

from __future__ import annotations

import numpy as np
import pytest

from pathfusion.eval.metrics import (
    bootstrap_ci,
    compute_metrics,
    macro_auroc,
    metric_fn,
    paired_bootstrap_diff,
)
from pathfusion.fusion.spec import FusionSpec


def _data(n=70, c=7, seed=0, noise=1.0):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(c), n // c)
    logits = rng.normal(size=(len(y), c)) * noise
    logits[np.arange(len(y)), y] += 2
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, p


def test_perfect_and_random_auroc():
    y, _ = _data()
    onehot = np.eye(7)[y]
    assert macro_auroc(y, onehot, 7) == pytest.approx(1.0)
    rnd = np.random.default_rng(1).random((len(y), 7))
    assert 0.3 < macro_auroc(y, rnd, 7) < 0.7


def test_bootstrap_ci_brackets_point():
    y, p = _data()
    ci = bootstrap_ci(y, p, metric_fn("macro_auroc", 7), n_boot=300)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["hi"] - ci["lo"] > 0


def test_paired_diff_identical_is_zero():
    y, p = _data()
    d = paired_bootstrap_diff(y, p, p, metric_fn("balanced_accuracy", 7), n_boot=100)
    assert d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0


def test_compute_metrics_keys():
    y, p = _data()
    m = compute_metrics(y, p, 7)
    assert set(m) == {"macro_auroc", "balanced_accuracy", "accuracy", "log_loss"}


def test_spec_validation_and_canonical():
    a = FusionSpec("late", ("virchow2", "uni2h"), late_rule="prob_mean", stream_fusion="attention")
    b = FusionSpec("late", ("uni2h", "virchow2"), late_rule="prob_mean")
    assert a.canonical() == b.canonical() and a.key == b.key
    assert FusionSpec.from_dict(a.to_dict()) == b.canonical()
    with pytest.raises(ValueError):
        FusionSpec("early_concat", ("uni2h",))
    with pytest.raises(ValueError):
        FusionSpec("late", ("uni2h", "virchow2"), late_rule="median")
    with pytest.raises(ValueError):
        FusionSpec("meta_only", ("uni2h",))


def test_complexity_orders_families():
    s = FusionSpec("single", ("uni2h",))
    late = FusionSpec("late", ("uni2h", "virchow2"))
    inter = FusionSpec("intermediate", ("uni2h", "virchow2"), stream_fusion="attention")
    assert s.complexity() < late.complexity() < inter.complexity()
