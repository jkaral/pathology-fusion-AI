"""Per-model tile-feature preprocessing fitted on training patients only.

Embeddings from different foundation models live on very different scales (and have
1024-2560 dims), so concatenating them raw lets the largest-norm model dominate. Options:

* ``l2``     : unit-normalise each tile vector per model.
* ``zscore`` : standardise each dimension with training-tile mean/std per model.
* ``pca_dim``: project each model to k dims (PCA on a training-tile sample), equalising
  the dimensional footprint of each model in a concatenation.

Fitting uses a seeded tile sample from the *training fold* only (no leakage).
"""

from __future__ import annotations

import numpy as np
import torch


class StreamTransform:
    def __init__(self, norm: str = "none", pca_dim: int | None = None):
        self.norm = norm
        self.pca_dim = pca_dim
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.pca_mean: torch.Tensor | None = None
        self.components: torch.Tensor | None = None
        self.in_dim: int | None = None

    @property
    def is_identity(self) -> bool:
        return self.norm == "none" and self.pca_dim is None

    @property
    def out_dim(self) -> int:
        assert self.in_dim is not None, "fit first"
        return self.pca_dim if self.pca_dim is not None else self.in_dim

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm == "l2":
            return torch.nn.functional.normalize(x, dim=-1)
        if self.norm == "zscore":
            assert self.mean is not None and self.std is not None
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x

    def fit(self, bags: list[torch.Tensor], max_tiles: int = 20000, seed: int = 0) -> StreamTransform:
        self.in_dim = int(bags[0].shape[1])
        if self.is_identity:
            return self
        rng = np.random.default_rng(seed)
        per_bag = max(1, max_tiles // max(1, len(bags)))
        sample = torch.cat([
            b[torch.from_numpy(rng.choice(len(b), size=min(per_bag, len(b)), replace=False))]
            for b in bags
        ]).float()
        if self.norm == "zscore":
            self.mean = sample.mean(0)
            self.std = sample.std(0).clamp(min=1e-6)
        sample = self._norm(sample)
        if self.pca_dim is not None:
            k = min(self.pca_dim, sample.shape[0] - 1, sample.shape[1])
            self.pca_mean = sample.mean(0)
            g = torch.Generator().manual_seed(seed)
            _, _, v = _svd_lowrank(sample - self.pca_mean, k, g)
            comps = v[:, :k].T.contiguous()
            if k < self.pca_dim:  # tiny synthetic data: pad so out_dim stays as configured
                comps = torch.cat([comps, torch.zeros(self.pca_dim - k, comps.shape[1])])
            self.components = comps
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """(..., D) float16/32 -> (..., out_dim) float32."""
        x = self._norm(x.float())
        if self.components is not None:
            assert self.pca_mean is not None
            x = (x - self.pca_mean.to(x.device)) @ self.components.to(x.device).T
        return x


def _svd_lowrank(x: torch.Tensor, k: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic randomized SVD (seeded generator)."""
    q = min(k + 10, min(x.shape))
    omega = torch.randn(x.shape[1], q, generator=gen)
    y = x @ omega
    for _ in range(4):
        y = x @ (x.T @ y)
    qmat, _ = torch.linalg.qr(y)
    b = qmat.T @ x
    ub, s, vh = torch.linalg.svd(b, full_matrices=False)
    return qmat @ ub, s, vh.T
