# CPU pilot on real data: design

## Why a pilot

The headline encoders (UNI2-h, Virchow2) are gated and the headline run needs a GPU.
This environment has neither credentials nor a GPU, but it can reach the dataset and
**ungated** encoders. The pilot runs the real pipeline on all 203 real slides to:

1. validate decoding, tissue detection and tiling on real JPEGs (not synthetic ones);
2. measure real timings and tile counts, and use them to plan the headline run;
3. run the agent (fusion search and equal-budget single search) on real features with a
   real LLM policy, to check the decision logic on real noise levels;
4. test the acquisition-order shortcut hypothesis (`scripts/07_batch_probe.py`).

## What is the same as the headline run

The cohort, the split, the development CV folds, tiling (224 px at 20x, ≥50% tissue), the
MIL recipe, the search space, the agent policy and its guardrails, and the equal-budget
control.

## What differs (`conf/pilot.yaml`)

| | Headline (`default.yaml`) | Pilot (`pilot.yaml`) |
|---|---|---|
| Encoders | UNI2-h, Virchow2, Phikon-v2 | Phikon-v2, Phikon (ViT-B), Lunit-DINO (ViT-S) |
| Tiles / slide | all tissue tiles (cap 4096 never binds) | 256 (seeded random subset) |
| Hardware | Kaggle 2x T4, fp16 | 4-core CPU, bf16 (AMX) |
| LLM policy | Qwen2.5-7B-Instruct (GPU) | Qwen2.5-3B-Instruct (CPU) |
| Stage 3 (test) | run once | **never run**: test stays reserved |

## How to read pilot numbers

Pilot scores are **development-CV** estimates for weaker, cheaper encoders with 6-16x fewer
tiles. They are lower bounds on what the headline configuration should reach, not results
for the brief's table. Their value lies in the *relative* comparisons (fusion vs. the
tuned single model at equal budget; metadata vs. images), in what the agent did, and in
the batch probe.

## Reproduce

```bash
export PATHFUSION_CONFIG=conf/pilot.yaml
python scripts/01_extract_embeddings.py                   # ~3.5 h on 4 CPU cores
python scripts/01_extract_embeddings.py --thumbnails-only # backfill thumbnails if needed
python scripts/06_tiling_qc_report.py --embeddings-dir artifacts/embeddings_pilot \
    --qc-dir artifacts/qc --out artifacts/pilot/tiling
python scripts/07_batch_probe.py
python scripts/02_run_agent.py --run-id pilot_qwen3b
```

Results: `artifacts/pilot/` (see `PILOT_RESULTS.md` there once generated).
