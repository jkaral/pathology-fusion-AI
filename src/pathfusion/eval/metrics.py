"""Classification metrics and patient-level bootstrap.

Primary metrics: macro one-vs-rest AUROC and balanced accuracy. The bootstrap is
*stratified* by true class (resampling within each class) so every replicate contains
every class and macro-AUROC is always defined; class counts match the test set.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss, roc_auc_score


def per_class_auroc(y: np.ndarray, probs: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.full(n_classes, np.nan)
    for c in range(n_classes):
        pos = y == c
        if 0 < pos.sum() < len(y):
            out[c] = roc_auc_score(pos.astype(int), probs[:, c])
    return out


def macro_auroc(y: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    """Mean OvR AUROC over classes present in ``y`` (all classes in our stratified splits)."""
    return float(np.nanmean(per_class_auroc(y, probs, n_classes)))


def balanced_accuracy(y: np.ndarray, probs: np.ndarray) -> float:
    return float(balanced_accuracy_score(y, probs.argmax(1)))


def compute_metrics(y: np.ndarray, probs: np.ndarray, n_classes: int) -> dict[str, float]:
    probs = np.clip(probs, 1e-7, 1.0)
    probs = probs / probs.sum(1, keepdims=True)
    return {
        "macro_auroc": macro_auroc(y, probs, n_classes),
        "balanced_accuracy": balanced_accuracy(y, probs),
        "accuracy": float((probs.argmax(1) == y).mean()),
        "log_loss": float(log_loss(y, probs, labels=list(range(n_classes)))),
    }


def confusion(y: np.ndarray, probs: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y, probs.argmax(1), labels=list(range(n_classes)))


def stratified_bootstrap_indices(y: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) indices resampled with replacement within each class."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    out = np.empty((n_boot, len(y)), dtype=np.int64)
    for b in range(n_boot):
        out[b] = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
    return out


MetricFn = Callable[[np.ndarray, np.ndarray], float]


def bootstrap_ci(
    y: np.ndarray, probs: np.ndarray, metric: MetricFn, n_boot: int = 2000, ci: float = 0.95, seed: int = 0
) -> dict[str, float]:
    idx = stratified_bootstrap_indices(y, n_boot, seed)
    vals = np.array([metric(y[i], probs[i]) for i in idx])
    lo, hi = np.nanquantile(vals, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"point": metric(y, probs), "lo": float(lo), "hi": float(hi), "sd": float(np.nanstd(vals))}


def paired_bootstrap_diff(
    y: np.ndarray, probs_a: np.ndarray, probs_b: np.ndarray, metric: MetricFn,
    n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
) -> dict[str, float]:
    """Bootstrap of metric(a) - metric(b) on the *same* resampled patients."""
    idx = stratified_bootstrap_indices(y, n_boot, seed)
    d = np.array([metric(y[i], probs_a[i]) - metric(y[i], probs_b[i]) for i in idx])
    lo, hi = np.nanquantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {
        "diff": metric(y, probs_a) - metric(y, probs_b), "lo": float(lo), "hi": float(hi),
        "p_le_0": float(np.mean(d <= 0)),
    }


def metric_fn(name: str, n_classes: int) -> MetricFn:
    if name == "macro_auroc":
        return lambda y, p: macro_auroc(y, p, n_classes)
    if name == "balanced_accuracy":
        return balanced_accuracy
    raise KeyError(name)
