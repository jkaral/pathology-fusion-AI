"""Cross-validated scoring of a fusion spec on the development set (train + val only).

Why CV rather than the 22-patient validation split: one validation patient moves
balanced accuracy by ~5 points, so ranking dozens of configs on it selects noise. The
agent instead scores every trial on the same stratified K folds over the 162 development
patients and compares trials *paired by fold*.

Paired comparison uses the Nadeau-Bengio corrected variance, which accounts for the
overlap between CV training sets (plain fold SD understates uncertainty):

    var_corr = (1/K + n_test/n_train) * var(d_k),   d_k = score_new,k - score_ref,k
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from pathfusion.eval.metrics import compute_metrics, metric_fn
from pathfusion.fusion.pipelines import build_pipeline, fit_cached
from pathfusion.fusion.spec import FusionSpec
from pathfusion.training.context import DataContext
from pathfusion.utils.io import stable_hash


@dataclass
class TrialResult:
    spec: FusionSpec
    fold_scores: np.ndarray          # (R*K,) selection metric per held-out fold
    oof_probs: np.ndarray            # (n_dev, C) out-of-fold probabilities (avg over repeats)
    pooled: dict[str, float]         # metrics on pooled OOF predictions
    seconds: float
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def sd(self) -> float:
        return float(np.std(self.fold_scores, ddof=1)) if len(self.fold_scores) > 1 else float("nan")


@dataclass
class PairedDiff:
    mean: float
    se: float          # Nadeau-Bengio corrected
    t: float


class CVEvaluator:
    def __init__(self, ctx: DataContext, folds: pd.DataFrame, metric: str, seed: int,
                 cache_dir: Path | None = None):
        """``cache_dir``: persist each finished trial so an interrupted search resumes for free."""
        self.cache_dir = cache_dir
        self._folds_key = stable_hash(folds.sort_values(["repeat", "fold", "patient_id"]).to_dict("list"))
        self.ctx = ctx
        self.metric_name = metric
        self.metric = metric_fn(metric, ctx.n_classes)
        self.seed = seed
        self.dev_ids = sorted(folds["patient_id"].astype(str).unique())
        self._pos = {pid: i for i, pid in enumerate(self.dev_ids)}
        self.splits: list[tuple[int, list[str], list[str]]] = []
        for (r, _k), grp in folds.groupby(["repeat", "fold"], sort=True):
            held = sorted(grp["patient_id"].astype(str))
            held_set = set(held)
            train = [p for p in self.dev_ids if p not in held_set]
            self.splits.append((int(r), train, held))
        self.n_repeats = folds["repeat"].nunique()
        self.y_dev = ctx.labels(self.dev_ids)
        n_te = np.mean([len(h) for _, _, h in self.splits])
        self.nb_ratio = float(n_te / (len(self.dev_ids) - n_te))

    def _cache_path(self, spec: FusionSpec) -> Path | None:
        if self.cache_dir is None:
            return None
        key = stable_hash({"spec": spec.key, "folds": self._folds_key, "seed": self.seed,
                           "metric": self.metric_name, "train": asdict(self.ctx.train_cfg),
                           "dims": self.ctx.store.dims if self.ctx.store is not None else None})
        return self.cache_dir / f"trial_{key}.npz"

    def evaluate(self, spec: FusionSpec) -> TrialResult:
        path = self._cache_path(spec)
        if path is not None and path.exists():
            z = np.load(path, allow_pickle=False)
            return TrialResult(spec, z["fold_scores"], z["oof_probs"], json.loads(str(z["pooled"])),
                               float(z["seconds"]), extras={"cached": True})
        res = self._evaluate(spec)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, fold_scores=res.fold_scores, oof_probs=res.oof_probs,
                     pooled=json.dumps(res.pooled), seconds=res.seconds)
            tmp.replace(path)
        return res

    def _evaluate(self, spec: FusionSpec) -> TrialResult:
        t0 = time.perf_counter()
        oof = np.zeros((self.n_repeats, len(self.dev_ids), self.ctx.n_classes))
        scores = []
        for r, train, held in self.splits:
            # single-model fits are cached and re-used as late-fusion members
            if spec.family == "single":
                pipe = fit_cached(self.ctx, spec, train, self.seed)
            else:
                pipe = build_pipeline(spec).fit(self.ctx, train, self.seed)
            p = pipe.predict(self.ctx, held)
            idx = [self._pos[h] for h in held]
            oof[r, idx] = p
            scores.append(self.metric(self.ctx.labels(held), p))
        oof_mean = oof.mean(0)
        pooled = compute_metrics(self.y_dev, oof_mean, self.ctx.n_classes)
        return TrialResult(spec, np.array(scores), oof_mean, pooled, time.perf_counter() - t0)

    def paired(self, a: TrialResult, b: TrialResult) -> PairedDiff:
        """Fold-paired difference a - b with corrected standard error."""
        d = a.fold_scores - b.fold_scores
        k = len(d)
        var = np.var(d, ddof=1) if k > 1 else 0.0
        se = float(np.sqrt((1.0 / k + self.nb_ratio) * var))
        mean = float(d.mean())
        if se > 0:
            t = mean / se
        else:  # identical fold scores: no evidence either way (avoids 0 * inf = nan)
            t = 0.0 if mean == 0 else float(np.copysign(np.inf, mean))
        return PairedDiff(mean, se, t)


def selection_optimism(
    results: list[TrialResult], y: np.ndarray, metric_name: str, n_classes: int, n_boot: int, seed: int,
) -> dict[str, float]:
    """Bootstrap estimate of the winner's-curse optimism of picking the best of N trials.

    For each bootstrap resample of development patients, pick the trial with the best
    pooled-OOF score *on the resample* and record how much its in-bag score exceeds its
    out-of-bag score. The mean excess is the expected inflation of the best CV score
    caused by selection itself (cf. Tibshirani & Tibshirani 2009).
    """
    if not results or n_boot <= 0:
        return {"optimism": float("nan"), "best_pooled": float("nan"), "corrected": float("nan")}
    metric = metric_fn(metric_name, n_classes)
    rng = np.random.default_rng(seed)
    n = len(y)
    gaps, picks = [], []
    for _ in range(n_boot):
        inbag = np.concatenate([rng.choice(np.flatnonzero(y == c), size=(y == c).sum()) for c in np.unique(y)])
        oob = np.setdiff1d(np.arange(n), inbag)
        if len(np.unique(y[oob])) < len(np.unique(y)):
            continue
        in_scores = [metric(y[inbag], r.oof_probs[inbag]) for r in results]
        k = int(np.argmax(in_scores))
        picks.append(k)
        gaps.append(in_scores[k] - metric(y[oob], results[k].oof_probs[oob]))
    best = max(r.pooled[metric_name] for r in results)
    opt = float(np.mean(gaps)) if gaps else float("nan")
    pick_freq = np.bincount(picks, minlength=len(results)) / max(1, len(picks))
    return {"optimism": opt, "best_pooled": float(best), "corrected": float(best - opt),
            "n_boot_used": len(gaps), "top_pick_frequency": float(pick_freq.max())}
