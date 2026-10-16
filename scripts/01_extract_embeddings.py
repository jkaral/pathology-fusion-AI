"""Stage 1: tile the selected slides at 20x and embed every tile with all three encoders.

Run on a GPU runtime (Kaggle 2x T4 recommended). Needs ``HF_TOKEN`` with access to the
gated UNI2-h and Virchow2 repos.

Usage:
    python scripts/01_extract_embeddings.py                      # all cohort slides, from HF
    python scripts/01_extract_embeddings.py --shard 0/2          # first half (resumable)
    python scripts/01_extract_embeddings.py --limit 2            # smoke test on 2 slides
    python scripts/01_extract_embeddings.py --source local --local-dir /kaggle/input/clwd
    python scripts/01_extract_embeddings.py --summarize          # timing/size summary only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from pathfusion.config import load_config  # noqa: E402
from pathfusion.embeddings.extract import (  # noqa: E402
    assign_devices,
    default_cache_dir,
    free_disk_gb,
    hf_source,
    local_source,
    run_extraction,
)
from pathfusion.utils.logging import get_logger  # noqa: E402

log = get_logger("extract")


def summarize(emb_dir: Path, models: list[str]) -> dict[str, object]:
    man = pd.read_csv(emb_dir / "extraction_manifest.csv").drop_duplicates("WSI_ID", keep="last")
    out: dict[str, object] = {
        "slides": int(len(man)),
        "tiles_total": int(man["n_tiles"].sum()),
        "tiles_per_slide_median": float(man["n_tiles"].median()),
        "slides_capped": int((man["n_tissue_tiles"] > man["n_tiles"]).sum()),
        "download_hours": round(man["download_s"].sum() / 3600, 2),
        "decode_hours": round(man["decode_s"].sum() / 3600, 2),
        "tiling_hours": round(man["tiling_s"].sum() / 3600, 2),
    }
    for m in models:
        col = f"{m}_s"
        if col in man:
            out[f"{m}_gpu_hours"] = round(man[col].sum() / 3600, 2)
            out[f"{m}_tiles_per_s"] = round(man["n_tiles"].sum() / max(man[col].sum(), 1e-9), 1)
    size_gb = sum(p.stat().st_size for p in emb_dir.glob("*.h5")) / 1e9
    out["feature_cache_gb"] = round(size_gb, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "local"], default="hf")
    ap.add_argument("--local-dir", type=Path, default=None)
    ap.add_argument("--shard", default=None, help="i/n: process every n-th slide starting at i")
    ap.add_argument("--limit", type=int, default=None, help="only the first N slides (smoke test)")
    ap.add_argument("--out-dir", type=Path, default=None, help="defaults to paths.embeddings_dir")
    ap.add_argument("--no-qc", action="store_true", help="skip QC overlay images")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(overrides=args.overrides)
    out_dir = args.out_dir or cfg.paths.resolve("embeddings_dir")
    models = cfg.extract.models

    if args.summarize:
        summary = summarize(out_dir, models)
        print(json.dumps(summary, indent=2))
        (out_dir / "extraction_summary.json").write_text(json.dumps(summary, indent=2))
        return

    cohort = pd.read_csv(cfg.paths.resolve("cohort_csv"), dtype={"patient_id": str})
    slides = sorted(cohort["WSI_ID"], key=lambda s: int(s.split("-")[1]))
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        slides = slides[i::n]
    if args.limit:
        slides = slides[: args.limit]
    sizes = dict(zip(cohort["WSI_ID"], cohort["size_bytes"]))

    from pathfusion.embeddings.registry import get_encoder

    encoders = [get_encoder(m) for m in models]
    for enc, dev in zip(encoders, assign_devices(encoders)):
        log.info("loading %s on %s", enc.name, dev)
        enc.load(dev)
        log.info("  preprocessing: %s", enc.describe())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "encoders.json").write_text(json.dumps([e.describe() for e in encoders], indent=2))

    if args.source == "hf":
        cache = default_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        log.info("JPEG cache %s (%.0f GB free)", cache, free_disk_gb(cache))
        source = hf_source(cfg.data.hf_repo, cache, cfg.extract.prefetch, cfg.extract.keep_jpg)
    else:
        if args.local_dir is None:
            ap.error("--local-dir is required with --source local")
        source = local_source(args.local_dir)

    qc_dir = None if args.no_qc else cfg.paths.resolve("qc_dir")
    run_extraction(cfg, slides, source, encoders, out_dir, qc_dir, sizes)
    print(json.dumps(summarize(out_dir, models), indent=2))


if __name__ == "__main__":
    main()
