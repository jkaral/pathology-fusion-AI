"""Training loop for (multi-stream) attention-MIL models over cached tile embeddings.

* Mini-batches of bags: each step samples ``train_tiles`` tiles per bag (the *same* tile
  indices across models, so streams stay spatially aligned), pads, and masks.
* Class-weighted cross-entropy (inverse frequency) for the 3:1 imbalance.
* AdamW, linear warm-up then cosine decay, gradient clipping at 1.0.
* Fixed epoch budget (no early stopping on the 22-patient validation set: its loss is too
  noisy to select on). Validation metrics are *logged* every epoch for the curves.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from pathfusion.eval.metrics import compute_metrics
from pathfusion.fusion.transforms import StreamTransform
from pathfusion.models.mil import class_weights, weighted_ce
from pathfusion.training.context import DataContext

LogFn = Callable[[dict[str, float], int], None]


@dataclass
class StreamInput:
    """Which cached models feed the network and how their tiles are preprocessed."""

    models: list[str]
    transforms: list[StreamTransform]
    early_concat: bool = False   # True: concatenate per tile into one stream

    @property
    def in_dims(self) -> list[int]:
        dims = [t.out_dim for t in self.transforms]
        return [sum(dims)] if self.early_concat else dims


def make_batch(
    ctx: DataContext, ids: list[str], inputs: StreamInput, cov: np.ndarray | None,
    max_tiles: int, rng: np.random.Generator | None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor | None]:
    """Gather (optionally subsampled) bags -> padded streams on device, mask, covariates."""
    assert ctx.store is not None
    dev = ctx.device
    picks = []
    for pid in ids:
        n = ctx.store.n_tiles(pid)
        if n > max_tiles:
            idx = rng.choice(n, size=max_tiles, replace=False) if rng is not None else np.arange(max_tiles)
        else:
            idx = np.arange(n)
        picks.append(torch.from_numpy(np.sort(idx)))
    t_max = max(len(p) for p in picks)
    mask = torch.zeros(len(ids), t_max, dtype=torch.bool)
    for i, p in enumerate(picks):
        mask[i, : len(p)] = True
    streams = []
    for model, tf in zip(inputs.models, inputs.transforms):
        d = ctx.store.dims[model]
        x = torch.zeros(len(ids), t_max, d, dtype=torch.float16)
        for i, (pid, p) in enumerate(zip(ids, picks)):
            x[i, : len(p)] = ctx.store.bag(model, pid)[p]
        streams.append(tf(x.to(dev, non_blocking=True)))
    if inputs.early_concat:
        streams = [torch.cat(streams, dim=-1)]
    meta = torch.from_numpy(cov).to(dev) if cov is not None else None
    return streams, mask.to(dev), meta


@torch.no_grad()
def predict(
    model: nn.Module, ctx: DataContext, inputs: StreamInput, ids: list[str], cov: np.ndarray | None,
    batch_size: int = 8,
) -> np.ndarray:
    """Softmax probabilities in the order of ``ids``."""
    model.eval()
    assert ctx.store is not None
    order = sorted(range(len(ids)), key=lambda i: ctx.store.n_tiles(ids[i]))  # type: ignore[union-attr]
    out = np.zeros((len(ids), ctx.n_classes), dtype=np.float64)
    for s in range(0, len(order), batch_size):
        chunk = order[s:s + batch_size]
        streams, mask, meta = make_batch(
            ctx, [ids[i] for i in chunk], inputs, cov[chunk] if cov is not None else None,
            ctx.train_cfg.eval_tiles, rng=None,
        )
        logits = model(streams, mask, meta)
        out[chunk] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
    return out


def _lr_lambda(total: int, warmup: int) -> Callable[[int], float]:
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return f


def train_mil(
    model: nn.Module, ctx: DataContext, inputs: StreamInput, train_ids: list[str],
    cov_train: np.ndarray | None, seed: int,
    val_ids: list[str] | None = None, cov_val: np.ndarray | None = None,
    log_fn: LogFn | None = None, log_train_metrics: bool = False,
) -> list[dict[str, float]]:
    """Train ``model`` in place for the configured fixed number of epochs; return history."""
    cfg = ctx.train_cfg
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.to(ctx.device)
    y_train = ctx.labels(train_ids)
    y_t = torch.from_numpy(y_train.copy()).to(ctx.device)
    weight = None
    if cfg.class_weighting:
        weight = class_weights(torch.from_numpy(y_train.copy()), ctx.n_classes).to(ctx.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(train_ids) / cfg.batch_size)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, _lr_lambda(cfg.epochs * steps_per_epoch, cfg.warmup_epochs * steps_per_epoch)
    )
    history: list[dict[str, float]] = []
    for epoch in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(train_ids))
        total, count = 0.0, 0
        for s in range(0, len(perm), cfg.batch_size):
            b = perm[s:s + cfg.batch_size]
            streams, mask, meta = make_batch(
                ctx, [train_ids[i] for i in b], inputs, cov_train[b] if cov_train is not None else None,
                cfg.train_tiles, rng,
            )
            logits = model(streams, mask, meta)
            loss = weighted_ce(logits, y_t[b], weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item() * len(b)
            count += len(b)
        rec: dict[str, float] = {"epoch": epoch + 1, "train/loss": total / count, "lr": float(sched.get_last_lr()[0])}
        if log_train_metrics:
            p = predict(model, ctx, inputs, train_ids, cov_train)
            m = compute_metrics(y_train, p, ctx.n_classes)
            rec.update({"train/macro_auroc": m["macro_auroc"], "train/balanced_accuracy": m["balanced_accuracy"]})
        if val_ids:
            p = predict(model, ctx, inputs, val_ids, cov_val)
            m = compute_metrics(ctx.labels(val_ids), p, ctx.n_classes)
            rec.update({"val/loss": m["log_loss"], "val/macro_auroc": m["macro_auroc"],
                        "val/balanced_accuracy": m["balanced_accuracy"]})
        history.append(rec)
        if log_fn is not None:
            log_fn(rec, epoch + 1)
    return history
