"""Tissue detection, grid generation and 1/4-scale JPEG decoding on synthetic slides."""

from __future__ import annotations

import numpy as np
from PIL import Image

from pathfusion.config import TilingCfg
from pathfusion.data.tiling import grid_coords, load_slide_at, make_tile_grid, save_qc_overlay


def fake_slide(h: int = 2048, w: int = 3072, seed: int = 0) -> np.ndarray:
    """White background with two pink 'tissue' ellipses and a green pen stroke."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 242, np.uint8)
    yy, xx = np.mgrid[:h, :w]
    for cy, cx, ry, rx in ((600, 800, 400, 600), (1400, 2200, 450, 500)):
        inside = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 < 1
        img[inside] = (np.array([200, 110, 170]) + rng.integers(-15, 15, (inside.sum(), 3))).clip(0, 255)
    img[1800:1850, 200:1500] = (40, 160, 60)  # pen ink
    return img


def test_tissue_grid_finds_tissue_not_background_or_pen():
    img = fake_slide()
    cfg = TilingCfg(tile_size=128, stride=128, thumbnail_downsample=16, max_tiles=None)
    grid = make_tile_grid(img, cfg, "WSI-test")
    assert grid.n_candidates > 20
    for x, y in grid.coords:
        patch = img[y:y + 128, x:x + 128]
        assert (patch.mean(-1) < 235).mean() > 0.4, "kept tile should be mostly tissue"
        assert not ((y < 1850) and (y + 128 > 1800) and (x < 1500) and (patch[..., 1] > 150).mean() > 0.5)
    w, h = grid.image_size
    assert (grid.coords[:, 0] + 128 <= w).all() and (grid.coords[:, 1] + 128 <= h).all()


def test_max_tiles_cap_is_deterministic():
    img = fake_slide()
    cfg = TilingCfg(tile_size=64, stride=64, thumbnail_downsample=16, max_tiles=50)
    a = make_tile_grid(img, cfg, "WSI-7").coords
    b = make_tile_grid(img, cfg, "WSI-7").coords
    assert len(a) == 50 and (a == b).all()


def test_grid_coords_full_mask():
    mask = np.ones((10, 10), bool)
    coords = grid_coords(mask, (160, 160), tile=32, stride=32, factor=16, min_frac=0.5)
    assert len(coords) == 25


def test_jpeg_decoded_at_quarter_scale(tmp_path):
    img = fake_slide(1024, 1536)
    path = tmp_path / "slide.jpg"
    Image.fromarray(img).save(path, quality=90)
    small = load_slide_at(path, 4)
    assert small.shape == (256, 384, 3)
    assert abs(float(small.mean()) - float(img.mean())) < 5


def test_qc_overlay_written(tmp_path):
    img = fake_slide()
    grid = make_tile_grid(img, TilingCfg(tile_size=128, stride=128, max_tiles=None), "WSI-1")
    save_qc_overlay(grid, 128, 16, tmp_path / "qc.jpg")
    assert (tmp_path / "qc.jpg").stat().st_size > 1000
