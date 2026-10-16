"""Late-fusion combiners over per-model class-probability vectors.

``weighted`` and ``stacking`` are fitted on *out-of-fold* probabilities from an inner
cross-fit on the training patients, never on in-sample predictions (which would be
over-confident and make the combiner trust whichever model memorised best).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

EPS = 1e-7


def _log(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, EPS, 1.0))


class LateCombiner:
    def __init__(self, rule: str, stack_c: float = 1.0, seed: int = 0):
        self.rule = rule
        self.stack_c = stack_c
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.stacker: LogisticRegression | None = None

    @property
    def needs_oof(self) -> bool:
        return self.rule in ("weighted", "stacking")

    def fit(self, probs: list[np.ndarray], y: np.ndarray, meta: np.ndarray | None, n_classes: int) -> LateCombiner:
        if self.rule == "weighted":
            m = len(probs)
            stack = np.stack(probs)  # (M, n, C)

            def nll(theta: np.ndarray) -> float:
                w = softmax(theta)
                mix = np.tensordot(w, stack, axes=1)
                return float(-_log(mix[np.arange(len(y)), y]).mean() + 1e-2 * np.sum(theta**2))

            res = minimize(nll, np.zeros(m), method="L-BFGS-B")
            self.weights = softmax(res.x)
        elif self.rule == "stacking":
            x = self._stack_features(probs, meta)
            self.stacker = LogisticRegression(C=self.stack_c, class_weight="balanced", max_iter=5000,
                                              random_state=self.seed)
            self.stacker.fit(x, y)
            if len(self.stacker.classes_) != n_classes:
                raise ValueError("stacker training fold lacks a class")
        return self

    @staticmethod
    def _stack_features(probs: list[np.ndarray], meta: np.ndarray | None) -> np.ndarray:
        feats = [_log(p) for p in probs]
        if meta is not None:
            feats.append(meta)
        return np.concatenate(feats, axis=1)

    def predict(self, probs: list[np.ndarray], meta: np.ndarray | None) -> np.ndarray:
        if self.rule == "prob_mean":
            return np.mean(probs, axis=0)
        if self.rule == "logit_mean":
            return softmax(np.mean([_log(p) for p in probs], axis=0), axis=1)
        if self.rule == "weighted":
            assert self.weights is not None
            return np.tensordot(self.weights, np.stack(probs), axes=1)
        if self.rule == "stacking":
            assert self.stacker is not None
            return self.stacker.predict_proba(self._stack_features(probs, meta))
        raise ValueError(self.rule)

    def describe(self) -> dict[str, object]:
        out: dict[str, object] = {"rule": self.rule}
        if self.weights is not None:
            out["weights"] = [round(float(w), 4) for w in self.weights]
        return out
