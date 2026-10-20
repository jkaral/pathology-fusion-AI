"""Attention-based MIL (gated ABMIL, Ilse et al. 2018) and multi-stream fusion networks.

All networks share the same building blocks so that differences between results-table
rows come from *how* streams are fused, not from different backbones:

* ``BagEncoder``: Linear -> ReLU -> Dropout per tile, gated-attention pooling over tiles.
  Padded tiles are masked out of the softmax so batches of unequal bags are exact.
* ``MetaEncoder``: (age_z, sex) -> small embedding, concatenated before the classifier.
* ``MILClassifier``: one stream (single model, or tile-level concatenation = early fusion).
* ``MultiStreamMIL``: one BagEncoder per foundation model, fused at the slide-embedding
  level by a learned gate, attention over stream tokens, or concatenation (intermediate
  fusion), with optional modality dropout.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionPool(nn.Module):
    def __init__(self, dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.v = nn.Sequential(nn.Linear(dim, attn_dim), nn.Tanh(), nn.Dropout(dropout))
        self.u = nn.Sequential(nn.Linear(dim, attn_dim), nn.Sigmoid(), nn.Dropout(dropout))
        self.w = nn.Linear(attn_dim, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h: (B, N, D), mask: (B, N) bool -> pooled (B, D), attention (B, N)."""
        scores = self.w(self.v(h) * self.u(h)).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=1)
        return torch.bmm(attn.unsqueeze(1), h).squeeze(1), attn


class BagEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, attn_dim: int, dropout: float):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.pool = GatedAttentionPool(hidden_dim, attn_dim, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pool(self.proj(x), mask)


class MetaEncoder(nn.Module):
    def __init__(self, meta_in: int, meta_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(meta_in, meta_dim), nn.ReLU())

    def forward(self, meta: torch.Tensor) -> torch.Tensor:
        return self.net(meta)


class MILClassifier(nn.Module):
    """Single-stream ABMIL (+ metadata)."""

    def __init__(self, in_dim: int, n_classes: int, hidden_dim: int, attn_dim: int,
                 dropout: float, meta_in: int = 0, meta_dim: int = 16):
        super().__init__()
        self.encoder = BagEncoder(in_dim, hidden_dim, attn_dim, dropout)
        self.meta = MetaEncoder(meta_in, meta_dim) if meta_in else None
        head_in = hidden_dim + (meta_dim if meta_in else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, n_classes))

    def forward(self, streams: list[torch.Tensor], mask: torch.Tensor,
                meta: torch.Tensor | None = None) -> torch.Tensor:
        z, _ = self.encoder(streams[0], mask)
        if self.meta is not None:
            assert meta is not None
            z = torch.cat([z, self.meta(meta)], dim=-1)
        return self.head(z)


class MultiStreamMIL(nn.Module):
    """Intermediate fusion: per-model ABMIL streams fused at the slide-embedding level."""

    FUSIONS = ("gated", "attention", "concat")

    def __init__(self, in_dims: list[int], n_classes: int, hidden_dim: int, attn_dim: int,
                 dropout: float, fusion: str = "gated", modality_dropout: float = 0.0,
                 meta_in: int = 0, meta_dim: int = 16, n_heads: int = 4):
        super().__init__()
        if fusion not in self.FUSIONS:
            raise ValueError(f"fusion must be one of {self.FUSIONS}")
        self.fusion = fusion
        self.modality_dropout = modality_dropout
        self.streams = nn.ModuleList(BagEncoder(d, hidden_dim, attn_dim, dropout) for d in in_dims)
        self.meta = MetaEncoder(meta_in, meta_dim) if meta_in else None
        m = len(in_dims)
        meta_out = meta_dim if meta_in else 0
        if fusion == "gated":
            self.gate = nn.Linear(m * hidden_dim + meta_out, m)
            fused_dim = hidden_dim
        elif fusion == "attention":
            self.stream_emb = nn.Parameter(torch.zeros(m, hidden_dim))
            self.meta_token = nn.Linear(meta_out, hidden_dim) if meta_in else None
            self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            fused_dim = hidden_dim
        else:
            fused_dim = m * hidden_dim
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(fused_dim + meta_out, n_classes))
        self.last_gate: torch.Tensor | None = None

    def _drop_streams(self, hs: torch.Tensor) -> torch.Tensor:
        """Zero whole streams per sample during training; at least one stream is kept."""
        if not self.training or self.modality_dropout <= 0:
            return hs
        b, m, _ = hs.shape
        keep = torch.rand(b, m, device=hs.device) >= self.modality_dropout
        none = ~keep.any(dim=1)
        if none.any():
            keep[none, torch.randint(0, m, (int(none.sum()),), device=hs.device)] = True
        scale = m / keep.sum(dim=1, keepdim=True).clamp(min=1)
        return hs * keep.unsqueeze(-1) * scale.unsqueeze(-1)

    def forward(self, streams: list[torch.Tensor], mask: torch.Tensor,
                meta: torch.Tensor | None = None) -> torch.Tensor:
        hs = torch.stack([enc(x, mask)[0] for enc, x in zip(self.streams, streams)], dim=1)  # (B, M, H)
        hs = self._drop_streams(hs)
        e = self.meta(meta) if self.meta is not None and meta is not None else None
        if self.fusion == "gated":
            gate_in = hs.flatten(1) if e is None else torch.cat([hs.flatten(1), e], dim=-1)
            g = torch.softmax(self.gate(gate_in), dim=-1)
            self.last_gate = g.detach()
            z = (g.unsqueeze(-1) * hs).sum(dim=1)
        elif self.fusion == "attention":
            tokens = hs + self.stream_emb
            if e is not None and self.meta_token is not None:
                tokens = torch.cat([tokens, self.meta_token(e).unsqueeze(1)], dim=1)
            out, w = self.attn(tokens, tokens, tokens, need_weights=True)
            tokens = self.norm(tokens + out)
            self.last_gate = w.detach().mean(dim=1)[:, : hs.shape[1]]
            z = tokens[:, : hs.shape[1]].mean(dim=1)
        else:
            z = hs.flatten(1)
        if e is not None:
            z = torch.cat([z, e], dim=-1)
        return self.head(z)


def class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalised to mean 1 (absent classes get weight 0)."""
    counts = torch.bincount(labels, minlength=n_classes).float()
    w = torch.where(counts > 0, counts.sum() / (n_classes * counts.clamp(min=1)), torch.zeros_like(counts))
    return w / w[counts > 0].mean()


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    return F.cross_entropy(logits, y, weight=weight)
