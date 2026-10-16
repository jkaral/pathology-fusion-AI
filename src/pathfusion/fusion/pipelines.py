"""Fusion strategies as fit/predict pipelines, registered by family name.

=============  ===============================================================
family         what is trained
=============  ===============================================================
meta_only      multinomial logistic regression on (age_z, sex)
single         one ABMIL on one model's tiles (+ metadata)
early_concat   one ABMIL on per-tile concatenation of the (normalised/PCA'd) models
late           one ABMIL per model; probabilities combined by mean / logit mean /
               OOF-weighted vote / OOF stacking (stacker also sees metadata)
intermediate   multi-stream ABMIL, slide embeddings fused by gate/attention/concat
=============  ===============================================================

Every image pipeline uses the same shared recipe (``TrainCfg``), so results-table rows
differ only in their inputs and fusion operator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from pathfusion.fusion.late import LateCombiner
from pathfusion.fusion.spec import FusionSpec
from pathfusion.fusion.transforms import StreamTransform
from pathfusion.models.mil import MILClassifier, MultiStreamMIL
from pathfusion.training.context import DataContext
from pathfusion.training.trainer import LogFn, StreamInput, predict, train_mil
from pathfusion.utils.io import stable_hash

INNER_FOLDS = 3


class Pipeline(ABC):
    def __init__(self, spec: FusionSpec):
        self.spec = spec
        self.history: list[dict[str, float]] = []

    @abstractmethod
    def fit(self, ctx: DataContext, train_ids: list[str], seed: int, val_ids: list[str] | None = None,
            log_fn: LogFn | None = None, log_train_metrics: bool = False) -> Pipeline: ...

    @abstractmethod
    def predict(self, ctx: DataContext, ids: list[str]) -> np.ndarray: ...


class MetaOnlyPipeline(Pipeline):
    def fit(self, ctx, train_ids, seed, val_ids=None, log_fn=None, log_train_metrics=False):  # type: ignore[override]
        self.age_stats = ctx.age_stats(train_ids)
        self.clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)
        self.clf.fit(ctx.covariates(train_ids, self.age_stats), ctx.labels(train_ids))
        return self

    def predict(self, ctx, ids):  # type: ignore[override]
        p = np.zeros((len(ids), ctx.n_classes))
        p[:, self.clf.classes_] = self.clf.predict_proba(ctx.covariates(ids, self.age_stats))
        return p


class NeuralPipeline(Pipeline):
    """single / early_concat / intermediate: one network trained end-to-end."""

    def _build(self, ctx: DataContext, inputs: StreamInput) -> nn.Module:
        c = ctx.train_cfg
        meta_in = 2 if self.spec.use_meta else 0
        if self.spec.family == "intermediate":
            return MultiStreamMIL(inputs.in_dims, ctx.n_classes, c.hidden_dim, c.attn_dim, c.dropout,
                                  fusion=self.spec.stream_fusion, modality_dropout=self.spec.modality_dropout,
                                  meta_in=meta_in, meta_dim=c.meta_dim)
        return MILClassifier(inputs.in_dims[0], ctx.n_classes, c.hidden_dim, c.attn_dim, c.dropout,
                             meta_in=meta_in, meta_dim=c.meta_dim)

    def fit(self, ctx, train_ids, seed, val_ids=None, log_fn=None, log_train_metrics=False):  # type: ignore[override]
        assert ctx.store is not None, "image pipelines need a FeatureStore"
        models = list(self.spec.models)
        transforms = [
            StreamTransform(self.spec.norm, self.spec.pca_dim).fit(
                [ctx.store.bag(m, pid) for pid in train_ids], seed=seed)
            for m in models
        ]
        self.inputs = StreamInput(models, transforms, early_concat=self.spec.family == "early_concat")
        self.age_stats = ctx.age_stats(train_ids)
        torch.manual_seed(seed)  # weight init must depend on the seed only
        self.model = self._build(ctx, self.inputs)
        cov = ctx.covariates(train_ids, self.age_stats) if self.spec.use_meta else None
        cov_val = ctx.covariates(val_ids, self.age_stats) if (self.spec.use_meta and val_ids) else None
        self.history = train_mil(self.model, ctx, self.inputs, train_ids, cov, seed, val_ids, cov_val,
                                 log_fn=log_fn, log_train_metrics=log_train_metrics)
        return self

    def predict(self, ctx, ids):  # type: ignore[override]
        cov = ctx.covariates(ids, self.age_stats) if self.spec.use_meta else None
        return predict(self.model, ctx, self.inputs, ids, cov)


def _single_spec(spec: FusionSpec, model: str) -> FusionSpec:
    return FusionSpec(family="single", models=(model,), use_meta=spec.use_meta, norm=spec.norm,
                      pca_dim=spec.pca_dim)


def fit_cached(ctx: DataContext, spec: FusionSpec, train_ids: list[str], seed: int) -> Pipeline:
    """Fit (or reuse) a pipeline: identical (spec, training patients, seed) -> same model.

    Late fusion re-uses the per-model networks across combiner rules, and the agent's
    single-model reference trials are re-used as late-fusion members.
    """
    key = (spec.key, stable_hash(sorted(train_ids)), seed)
    if key not in ctx.cache:
        ctx.cache[key] = build_pipeline(spec).fit(ctx, train_ids, seed)
    return ctx.cache[key]


class LatePipeline(Pipeline):
    def fit(self, ctx, train_ids, seed, val_ids=None, log_fn=None, log_train_metrics=False):  # type: ignore[override]
        spec = self.spec
        self.members = [_single_spec(spec, m) for m in spec.models]
        self.combiner = LateCombiner(spec.late_rule, spec.stack_c, seed)
        self.age_stats = ctx.age_stats(train_ids)
        y = ctx.labels(train_ids)
        if self.combiner.needs_oof:
            oof = [np.zeros((len(train_ids), ctx.n_classes)) for _ in self.members]
            skf = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed)
            for tr, te in skf.split(np.zeros(len(y)), y):
                inner_tr = [train_ids[i] for i in tr]
                inner_te = [train_ids[i] for i in te]
                for k, ms in enumerate(self.members):
                    oof[k][te] = fit_cached(ctx, ms, inner_tr, seed).predict(ctx, inner_te)
            meta = ctx.covariates(train_ids, self.age_stats) if spec.use_meta else None
            self.combiner.fit(oof, y, meta, ctx.n_classes)
        if val_ids or log_fn:
            # Final (logged) runs: train members with validation curves, prefixed per model,
            # then log the member-averaged curve under the standard keys.
            self.fitted = [
                build_pipeline(ms).fit(ctx, train_ids, seed, val_ids, None, log_train_metrics)
                for ms in self.members
            ]
            self.history = _average_histories([p.history for p in self.fitted])
            if log_fn:  # one log call per epoch keeps wandb steps monotonic
                for e, rec in enumerate(self.history):
                    merged = dict(rec)
                    for ms, p in zip(self.members, self.fitted):
                        merged.update({f"member_{ms.models[0]}/{k}": v
                                       for k, v in p.history[e].items() if k != "epoch"})
                    log_fn(merged, int(rec["epoch"]))
        else:
            self.fitted = [fit_cached(ctx, ms, train_ids, seed) for ms in self.members]
        return self

    def predict(self, ctx, ids):  # type: ignore[override]
        probs = [p.predict(ctx, ids) for p in self.fitted]
        meta = ctx.covariates(ids, self.age_stats) if self.spec.use_meta else None
        return self.combiner.predict(probs, meta)


def _average_histories(hs: list[list[dict[str, float]]]) -> list[dict[str, float]]:
    hs = [h for h in hs if h]
    if not hs:
        return []
    out = []
    for recs in zip(*hs):
        keys = set.intersection(*(set(r) for r in recs))
        out.append({k: float(np.mean([r[k] for r in recs])) for k in keys})
    return out


REGISTRY: dict[str, Callable[[FusionSpec], Pipeline]] = {
    "meta_only": MetaOnlyPipeline,
    "single": NeuralPipeline,
    "early_concat": NeuralPipeline,
    "intermediate": NeuralPipeline,
    "late": LatePipeline,
}


def build_pipeline(spec: FusionSpec) -> Pipeline:
    return REGISTRY[spec.family](spec)
