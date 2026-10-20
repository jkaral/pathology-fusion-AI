"""Macenko stain normalisation (Macenko et al., ISBI 2009), fitted once per slide.

The stain matrix and the 99th-percentile stain concentrations are estimated from the
tissue pixels of the slide's 1.25x thumbnail. Each 20x tile is then re-expressed with a
fixed reference stain matrix, removing between-slide (batch) colour variation before
the frozen encoders see the tiles. Reference values are the widely used defaults from
the staintools / torchstain implementations.

Enabled with ``tiling.stain_normalization: macenko`` (default ``none``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HE_REF = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
MAX_C_REF = np.array([1.9705, 1.0308])
IO = 240.0          # transmitted light intensity
BETA = 0.15         # OD threshold for "stained" pixels
ALPHA = 1.0         # angular percentile for the stain vectors


def _od(rgb: np.ndarray) -> np.ndarray:
    return -np.log((rgb.astype(np.float64) + 1.0) / IO)


@dataclass
class MacenkoNormalizer:
    he: np.ndarray       # (3, 2) slide stain matrix (columns: H, E)
    max_c: np.ndarray    # (2,) 99th-percentile concentrations

    @classmethod
    def fit(cls, rgb: np.ndarray, mask: np.ndarray | None = None) -> MacenkoNormalizer:
        """Estimate stain vectors from (tissue) pixels of an RGB image."""
        px = rgb.reshape(-1, 3) if mask is None else rgb[mask.astype(bool)]
        od = _od(px)
        od = od[(od > BETA).all(axis=1)]
        if len(od) < 100:
            raise ValueError("too few stained pixels to fit Macenko")
        _, vecs = np.linalg.eigh(np.cov(od.T))
        plane = vecs[:, 1:3]                                     # two largest eigenvectors
        proj = od @ plane
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        lo, hi = np.percentile(phi, ALPHA), np.percentile(phi, 100 - ALPHA)
        v1 = plane @ np.array([np.cos(lo), np.sin(lo)])
        v2 = plane @ np.array([np.cos(hi), np.sin(hi)])
        he = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)  # haematoxylin first
        he = he * np.sign(he.sum(0, keepdims=True))             # positive orientation
        he = he / np.linalg.norm(he, axis=0, keepdims=True)
        conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
        max_c = np.percentile(conc, 99, axis=1)
        return cls(he=he, max_c=np.maximum(max_c, 1e-6))

    def transform(self, tiles: np.ndarray) -> np.ndarray:
        """(..., 3) uint8 -> same shape uint8, re-stained with the reference matrix."""
        shape = tiles.shape
        od = _od(tiles.reshape(-1, 3))
        conc = np.linalg.lstsq(self.he, od.T, rcond=None)[0]     # (2, N)
        conc *= (MAX_C_REF / self.max_c)[:, None]
        out = IO * np.exp(-(HE_REF @ conc))
        return np.clip(out.T, 0, 255).astype(np.uint8).reshape(shape)
