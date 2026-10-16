"""Single-pass, streaming feature extraction for all encoders.

Why single pass: the one-slide-per-patient subset is ~218 GB of JPEG. Each slide is
downloaded once, decoded at 20x once, tiled once, and *all* encoders run on the same tile
batch before the JPEG is deleted. Peak disk use is ~2 slides (current + prefetched).

Output per slide: ``<embeddings_dir>/<WSI_ID>.h5`` with

* ``coords``            (N, 2) int32 top-left tile coordinates at 20x
* ``feats/<model>``     (N, D_model) float16 tile embeddings
* attrs: image size, tissue-tile count before cap, timings, encoder preprocessing

Files are written to ``*.tmp`` and renamed on success, so an interrupted run never leaves
a half-written file; reruns skip completed slides (resume). ``--shard i/n`` splits the
slide list across several sessions.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from pathfusion.config import Config
from pathfusion.data.tiling import (
    iter_tile_batches,
    load_slide_at,
    make_tile_grid,
    save_qc_overlay,
)
from pathfusion.embeddings.base import TileEncoder
from pathfusion.utils.logging import get_logger

log = get_logger("extract")

MANIFEST_FIELDS = [
    "WSI_ID", "n_tiles", "n_tissue_tiles", "width_20x", "height_20x", "download_s",
    "decode_s", "tiling_s", "file_gb",
]


class SlideSource:
    """Yields local JPEG paths for slide IDs, prefetching ahead in a background thread."""

    def __init__(self, fetch: Callable[[str], Path], cleanup: Callable[[Path], None], prefetch: int = 1):
        self._fetch = fetch
        self._cleanup = cleanup
        self._pool = ThreadPoolExecutor(max_workers=max(1, prefetch))
        self.prefetch = prefetch

    def iterate(self, slide_ids: list[str]) -> Iterator[tuple[str, Path, float]]:
        futures: dict[int, Future[tuple[Path, float]]] = {}

        def timed(sid: str) -> tuple[Path, float]:
            t0 = time.perf_counter()
            return self._fetch(sid), time.perf_counter() - t0

        for i in range(min(len(slide_ids), self.prefetch + 1)):
            futures[i] = self._pool.submit(timed, slide_ids[i])
        for i, sid in enumerate(slide_ids):
            path, dt = futures.pop(i).result()
            nxt = i + self.prefetch + 1
            if nxt < len(slide_ids):
                futures[nxt] = self._pool.submit(timed, slide_ids[nxt])
            yield sid, path, dt

    def release(self, path: Path) -> None:
        self._cleanup(path)


def hf_source(repo_id: str, cache_dir: Path, prefetch: int, keep: bool) -> SlideSource:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)

    def fetch(sid: str) -> Path:
        return Path(hf_hub_download(repo_id=repo_id, filename=f"{sid}.jpg", repo_type="dataset",
                                    local_dir=str(cache_dir)))

    def cleanup(path: Path) -> None:
        if not keep:
            path.unlink(missing_ok=True)

    return SlideSource(fetch, cleanup, prefetch)


def local_source(directory: Path) -> SlideSource:
    """Slides already on disk (e.g. a Kaggle input dataset); never deleted."""

    def fetch(sid: str) -> Path:
        path = directory / f"{sid}.jpg"
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    return SlideSource(fetch, lambda _p: None, prefetch=0)


def assign_devices(encoders: list[TileEncoder]) -> list[torch.device]:
    """Spread encoders over visible GPUs, largest first (2x T4 on Kaggle)."""
    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        return [torch.device("cpu")] * len(encoders)
    order = sorted(range(len(encoders)), key=lambda i: -encoders[i].embed_dim)
    load = [0] * n_gpu
    devices: list[torch.device] = [torch.device("cpu")] * len(encoders)
    for i in order:
        g = int(np.argmin(load))
        devices[i] = torch.device(f"cuda:{g}")
        load[g] += encoders[i].embed_dim
    return devices


@dataclass
class SlideResult:
    wsi_id: str
    n_tiles: int
    seconds: dict[str, float]


def extract_slide(
    wsi_id: str, jpg: Path, encoders: list[TileEncoder], cfg: Config, out_dir: Path, qc_dir: Path | None
) -> SlideResult:
    tcfg = cfg.tiling
    t0 = time.perf_counter()
    img = load_slide_at(jpg, tcfg.downsample)
    t_decode = time.perf_counter() - t0

    t0 = time.perf_counter()
    grid = make_tile_grid(img, tcfg, wsi_id)
    t_tiling = time.perf_counter() - t0
    if qc_dir is not None:
        save_qc_overlay(grid, tcfg.tile_size, tcfg.thumbnail_downsample, qc_dir / f"{wsi_id}.jpg")

    n = len(grid.coords)
    feats = {e.name: np.zeros((n, e.embed_dim), dtype=np.float16) for e in encoders}
    gpu_ms = {e.name: 0.0 for e in encoders}
    for start, batch in iter_tile_batches(img, grid.coords, tcfg.tile_size, cfg.extract.batch_size):
        tiles = torch.from_numpy(batch)
        if torch.cuda.is_available():
            tiles = tiles.pin_memory()
        pending = []
        for enc in encoders:  # launch on every device before syncing -> GPUs overlap
            if enc.device.type == "cuda":
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                with torch.cuda.device(enc.device):
                    ev0.record()
                    out = enc.encode(tiles)
                    ev1.record()
                pending.append((enc, out, ev0, ev1))
            else:
                c0 = time.perf_counter()
                out = enc.encode(tiles)
                gpu_ms[enc.name] += 1000 * (time.perf_counter() - c0)
                pending.append((enc, out, None, None))
        for enc, out, ev0, ev1 in pending:
            feats[enc.name][start:start + len(batch)] = out.cpu().numpy().astype(np.float16)
            if ev0 is not None:
                ev1.synchronize()
                gpu_ms[enc.name] += ev0.elapsed_time(ev1)
    del img

    tmp = out_dir / f"{wsi_id}.h5.tmp"
    with h5py.File(tmp, "w") as fh:
        fh.create_dataset("coords", data=grid.coords)
        g = fh.create_group("feats")
        for name, arr in feats.items():
            g.create_dataset(name, data=arr)
        fh.attrs["wsi_id"] = wsi_id
        fh.attrs["width_20x"], fh.attrs["height_20x"] = grid.image_size
        fh.attrs["n_tissue_tiles"] = grid.n_candidates
        fh.attrs["tile_size"] = tcfg.tile_size
        fh.attrs["magnification"] = tcfg.target_magnification
        fh.attrs["encoders"] = json.dumps([e.describe() for e in encoders])
        fh.attrs["encode_ms"] = json.dumps(gpu_ms)
    os.replace(tmp, out_dir / f"{wsi_id}.h5")
    secs = {"decode_s": t_decode, "tiling_s": t_tiling, **{f"{k}_s": v / 1000 for k, v in gpu_ms.items()}}
    return SlideResult(wsi_id, n, secs)


def is_complete(path: Path, models: list[str]) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as fh:
            return all(f"feats/{m}" in fh for m in models)
    except OSError:
        return False


def run_extraction(
    cfg: Config, slide_ids: list[str], source: SlideSource, encoders: list[TileEncoder],
    out_dir: Path, qc_dir: Path | None, sizes: dict[str, int] | None = None,
) -> Path:
    """Process ``slide_ids`` (skipping completed ones); returns the timing manifest path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [e.name for e in encoders]
    todo = [s for s in slide_ids if not is_complete(out_dir / f"{s}.h5", names)]
    log.info("%d/%d slides to process (%d already done)", len(todo), len(slide_ids), len(slide_ids) - len(todo))
    manifest = out_dir / "extraction_manifest.csv"
    new_file = not manifest.exists()
    fields = MANIFEST_FIELDS + [f"{n}_s" for n in names]
    with open(manifest, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        t_start = time.perf_counter()
        for k, (sid, path, t_dl) in enumerate(source.iterate(todo), 1):
            try:
                res = extract_slide(sid, path, encoders, cfg, out_dir, qc_dir)
            finally:
                source.release(path)
            with h5py.File(out_dir / f"{sid}.h5", "r") as h5:
                row = {"WSI_ID": sid, "n_tiles": res.n_tiles, "n_tissue_tiles": int(h5.attrs["n_tissue_tiles"]),
                       "width_20x": int(h5.attrs["width_20x"]), "height_20x": int(h5.attrs["height_20x"]),
                       "download_s": round(t_dl, 2),
                       "file_gb": round((sizes or {}).get(sid, 0) / 1e9, 3)}
            row.update({k2: round(v, 2) for k2, v in res.seconds.items()})
            writer.writerow(row)
            fh.flush()
            elapsed = time.perf_counter() - t_start
            eta = elapsed / k * (len(todo) - k)
            log.info("[%d/%d] %s: %d tiles | %s | ETA %.0f min", k, len(todo), sid, res.n_tiles,
                     " ".join(f"{a}={b:.1f}s" for a, b in res.seconds.items()), eta / 60)
    return manifest


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def default_cache_dir() -> Path:
    return Path(os.environ.get("PATHFUSION_JPG_CACHE", tempfile.gettempdir())) / "clwd_jpg"
