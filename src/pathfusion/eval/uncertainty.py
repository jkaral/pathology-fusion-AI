"""How much can a 41-patient test set resolve? Exact granularity + simulation.

* ``bacc_granularity``: one patient of class c moves balanced accuracy by 1 / (K * n_c).
* ``simulate_test_noise``: draws synthetic test sets with the real class counts from a
  latent-score model, scores two correlated classifiers on each, and reports the
  sampling SD of each metric and of their paired difference. The minimum detectable
  difference (two-sided alpha 0.05, power 0.8) is (1.96 + 0.84) x SD(diff).

The latent model: patient i of class y gets class scores z = s * onehot(y) + e_i + u_a,
where e_i is patient noise *shared* by both classifiers (correlation ``rho``) and u_a is
classifier-specific noise. Larger ``s`` means a better classifier. This is a deliberately
simple stand-in. It sizes the noise; it does not predict our accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pathfusion.eval.metrics import balanced_accuracy, macro_auroc

Z_ALPHA, Z_POWER = 1.96, 0.84


def bacc_granularity(class_counts: dict[str, int]) -> dict[str, float]:
    k = len(class_counts)
    return {c: 1.0 / (k * n) for c, n in class_counts.items()}


@dataclass
class NoiseSummary:
    signal_a: float
    signal_b: float
    rho: float
    auroc_a: float
    auroc_b: float
    sd_auroc: float
    sd_diff_auroc: float
    bacc_a: float
    bacc_b: float
    sd_bacc: float
    sd_diff_bacc: float

    @property
    def mdd_auroc(self) -> float:
        return (Z_ALPHA + Z_POWER) * self.sd_diff_auroc

    @property
    def mdd_bacc(self) -> float:
        return (Z_ALPHA + Z_POWER) * self.sd_diff_bacc


def _scores(y: np.ndarray, k: int, signal: float, shared: np.ndarray, rho: float, rng: np.random.Generator):
    own = rng.normal(size=(len(y), k))
    z = signal * np.eye(k)[y] + np.sqrt(rho) * shared + np.sqrt(1 - rho) * own
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def simulate_test_noise(
    class_counts: list[int], signal_a: float, signal_b: float, rho: float = 0.6, n_sim: int = 2000, seed: int = 0
) -> NoiseSummary:
    rng = np.random.default_rng(seed)
    k = len(class_counts)
    y = np.repeat(np.arange(k), class_counts)
    rec = np.zeros((n_sim, 4))
    for s in range(n_sim):
        shared = rng.normal(size=(len(y), k))
        pa = _scores(y, k, signal_a, shared, rho, rng)
        pb = _scores(y, k, signal_b, shared, rho, rng)
        rec[s] = (macro_auroc(y, pa, k), macro_auroc(y, pb, k), balanced_accuracy(y, pa), balanced_accuracy(y, pb))
    return NoiseSummary(
        signal_a, signal_b, rho,
        auroc_a=float(rec[:, 0].mean()), auroc_b=float(rec[:, 1].mean()),
        sd_auroc=float(rec[:, 0].std()), sd_diff_auroc=float((rec[:, 0] - rec[:, 1]).std()),
        bacc_a=float(rec[:, 2].mean()), bacc_b=float(rec[:, 3].mean()),
        sd_bacc=float(rec[:, 2].std()), sd_diff_bacc=float((rec[:, 2] - rec[:, 3]).std()),
    )
