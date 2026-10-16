"""Slide decoding at 20x, tissue segmentation and tile-grid generation.

Design notes
------------
* **80x -> 20x at decode time.** JPEG supports DCT-domain scaling by 1/2, 1/4, 1/8, so the
  4x downsample is done *while decoding* (libvips ``shrink=4`` or Pillow ``draft``). A
  65k x 65k 80x slide never materialises at full size: the 20x image is <= 16k x 16k
  (~0.8 GB RGB), which fits comfortably in Kaggle/Colab RAM.
* **Tissue mask** at 1.25x (20x / 16): HSV saturation, median-filtered, Otsu threshold
  (floored so blank slides do not yield noise), pen-ink hues removed, morphological
  closing and small-object removal. This is the standard CLAM-style recipe.
* **Tiles** are 224 x 224 px at 20x (~112 um at ~0.5 um/px), stride 224 (no overlap),
  kept if >= 50% of their footprint is tissue. 224 px is the native input size of all
  three encoders, so no per-model resampling is needed.
* Tile coordinates are computed once and shared by all three models, which makes
  tile-level early fusion (concatenation per tile) well defined.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from pathfusion.config import TilingCfg

Image.MAX_IMAGE_PIXELS = None  # WSIs exceed Pillow's decompression-bomb guard by design

# Hue band (degrees) of green/blue marker ink; H&E sits in the pink/purple range.
PEN_HUE_RANGE = (70.0, 200.0)
PEN_MIN_SATURATION = 0.25


@dataclass
class TileGrid:
    coords: np.ndarray        # (N, 2) int32 top-left (x, y) in 20x pixel space
    mask: np.ndarray          # (h, w) bool tissue mask at thumbnail resolution
    thumbnail: np.ndarray     # (h, w, 3) uint8
    image_size: tuple[int, int]  # (width, height) at 20x
    n_candidates: int         # tissue tiles before the max_tiles cap


def load_slide_at(path: str | Path, downsample: int) -> np.ndarray:
    """Decode a JPEG WSI at ``1/downsample`` scale into an (H, W, 3) uint8 array."""
    if downsample not in (1, 2, 4, 8):
        raise ValueError("JPEG shrink-on-load supports downsample in {1, 2, 4, 8}")
    try:
        import pyvips  # streaming decoder, lowest peak memory

        img = pyvips.Image.new_from_file(str(path), shrink=downsample, access="sequential")
        if img.bands == 4:
            img = img.flatten()
        elif img.bands == 1:
            img = img.colourspace("srgb")
        return np.asarray(img.numpy(), dtype=np.uint8)
    except ImportError:
        pass
    with Image.open(path) as im:
        full_w, full_h = im.size
        target = (full_w // downsample, full_h // downsample)
        im.draft("RGB", target)  # DCT scaling; no-op for non-JPEG inputs
        im = im.convert("RGB")
        if im.size != target:
            im = im.resize(target, Image.Resampling.BOX)
        return np.asarray(im)


def _otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rgb.astype(np.float32) / 255.0
    mx, mn = x.max(-1), x.min(-1)
    delta = mx - mn
    sat = np.divide(delta, mx, out=np.zeros_like(mx), where=mx > 0)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    safe = np.where(delta > 0, delta, 1.0)
    hue = np.where(mx == r, ((g - b) / safe) % 6, np.where(mx == g, (b - r) / safe + 2, (r - g) / safe + 4))
    hue = np.where(delta > 0, hue * 60.0, 0.0)
    return hue, sat, mx


def make_thumbnail(img: np.ndarray, factor: int) -> np.ndarray:
    """Box-filter downsample with Pillow's C ``reduce`` (no float copy of the 20x image)."""
    return np.asarray(Image.fromarray(img).reduce(factor))


def tissue_mask(thumb: np.ndarray, saturation_floor: float) -> np.ndarray:
    hue, sat, val = _hsv(thumb)
    sat = ndimage.median_filter(sat, size=5)
    thr = max(_otsu(sat), saturation_floor)
    mask = (sat > thr) & (val > 0.15)  # drop near-black (folds, debris, scanner edge)
    pen = (hue >= PEN_HUE_RANGE[0]) & (hue <= PEN_HUE_RANGE[1]) & (sat > PEN_MIN_SATURATION)
    mask &= ~pen
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=2)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndimage.label(mask)
    if n:
        sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
        min_size = max(16, int(0.0005 * mask.size))
        keep = np.isin(labels, np.flatnonzero(sizes >= min_size) + 1)
        mask = keep
    return mask


def grid_coords(
    mask: np.ndarray, image_size: tuple[int, int], tile: int, stride: int, factor: int, min_frac: float
) -> np.ndarray:
    """Top-left coords (20x px) of grid tiles whose tissue fraction >= ``min_frac``."""
    width, height = image_size
    integral = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    xs = np.arange(0, width - tile + 1, stride)
    ys = np.arange(0, height - tile + 1, stride)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    gx, gy = np.meshgrid(xs, ys)
    x0 = np.clip(gx // factor, 0, mask.shape[1])
    y0 = np.clip(gy // factor, 0, mask.shape[0])
    x1 = np.clip(-(-(gx + tile) // factor), 0, mask.shape[1])
    y1 = np.clip(-(-(gy + tile) // factor), 0, mask.shape[0])
    area = np.maximum((x1 - x0) * (y1 - y0), 1)
    tissue = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    keep = (tissue / area) >= min_frac
    return np.stack([gx[keep], gy[keep]], axis=1).astype(np.int32)


def slide_seed(slide_id: str) -> int:
    """Stable per-slide seed (independent of PYTHONHASHSEED)."""
    return zlib.crc32(slide_id.encode())


def make_tile_grid(img: np.ndarray, cfg: TilingCfg, slide_id: str) -> TileGrid:
    height, width = img.shape[:2]
    thumb = make_thumbnail(img, cfg.thumbnail_downsample)
    mask = tissue_mask(thumb, cfg.saturation_floor)
    coords = grid_coords(mask, (width, height), cfg.tile_size, cfg.stride,
                         cfg.thumbnail_downsample, cfg.min_tissue_fraction)
    n_candidates = len(coords)
    if cfg.max_tiles is not None and n_candidates > cfg.max_tiles:
        rng = np.random.default_rng(slide_seed(slide_id))
        idx = np.sort(rng.choice(n_candidates, size=cfg.max_tiles, replace=False))
        coords = coords[idx]
    return TileGrid(coords=coords, mask=mask, thumbnail=thumb,
                    image_size=(width, height), n_candidates=n_candidates)


def iter_tile_batches(img: np.ndarray, coords: np.ndarray, tile: int, batch_size: int):
    """Yield (start_index, uint8 array (B, tile, tile, 3)) batches in coordinate order."""
    for start in range(0, len(coords), batch_size):
        chunk = coords[start:start + batch_size]
        batch = np.stack([img[y:y + tile, x:x + tile] for x, y in chunk])
        yield start, batch


def save_qc_overlay(grid: TileGrid, tile: int, factor: int, path: str | Path, max_side: int = 1600) -> None:
    """Thumbnail with tissue outline (green) and kept tiles (blue boxes) for human QC."""
    base = Image.fromarray(grid.thumbnail).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    edge = grid.mask ^ ndimage.binary_erosion(grid.mask, iterations=1)
    ys, xs = np.nonzero(edge)
    for x, y in zip(xs, ys):
        draw.point((int(x), int(y)), fill=(0, 200, 0, 255))
    s = tile / factor
    for x, y in grid.coords:
        draw.rectangle([x / factor, y / factor, x / factor + s, y / factor + s], outline=(30, 90, 255, 110))
    out = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    out.thumbnail((max_side, max_side))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(path, quality=85)
