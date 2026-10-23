# Report fact sheet (auto-generated; numbers only)

## Data

- Cohort: 203 patients, one slide each (lowest available WSI_ID). See artifacts/data/data_audit.md.
- Split (patients): {'train': 140, 'test': 41, 'val': 22}; stratified per class, seed 2026.
- Development CV: 5 folds x 1 repeat(s) over train+val.
- Policies: duplicate_id=merge, conflicts=selected_slide.

## Tiling

- 80x -> 20x by JPEG DCT-domain shrink-on-load (4x linear).
- Tissue mask at 1.25x: HSV saturation, median filter, Otsu
  (floor 0.05), pen-hue removal, closing/opening, small-object removal.
- Tiles 224px at 20x, stride 224, kept if tissue >= 50%;
  cap 4096/slide.

## extraction_summary.json

```json
{
  "slides": 203,
  "tiles_total": 368250,
  "tiles_per_slide_median": 1870.0,
  "slides_capped": 0,
  "download_hours": 0.53,
  "decode_hours": 1.0,
  "tiling_hours": 0.06,
  "uni2h_gpu_hours": 3.33,
  "uni2h_tiles_per_s": 30.7,
  "virchow2_gpu_hours": 2.47,
  "virchow2_tiles_per_s": 41.4,
  "phikonv2_gpu_hours": 1.08,
  "phikonv2_tiles_per_s": 95.0,
  "feature_cache_gb": 3.78
}
```

## encoders.json

```json
[
  {
    "name": "uni2h",
    "hf_repo": "MahmoodLab/UNI2-h",
    "embed_dim": 1536,
    "input_size": 224,
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ],
    "interpolation": "bilinear"
  },
  {
    "name": "virchow2",
    "hf_repo": "paige-ai/Virchow2",
    "embed_dim": 2560,
    "input_size": 224,
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ],
    "interpolation": "bicubic"
  },
  {
    "name": "phikonv2",
    "hf_repo": "owkin/phikon-v2",
    "embed_dim": 1024,
    "input_size": 224,
    "mean": [
      0.485,
      0.456,
      0.406
    ],
    "std": [
      0.229,
      0.224,
      0.225
    ],
    "interpolation": "bicubic"
  }
]
```

## Shared MIL recipe (identical for all rows)

- Gated ABMIL: Linear(D->256)+ReLU+Dropout(0.25), attention dim 128; metadata
  MLP 2->16. AdamW lr 0.0002, wd 0.01, 30 epochs, batch 16 bags,
  512 sampled tiles/bag in training, <= 4096 at inference, class-weighted CE,
  2-epoch warm-up + cosine. Seeds [0, 1, 2, 3, 4].

## Agent

- Run: artifacts/agent/20261023-184822_local_llm (decisions.md, decisions.jsonl, trials.csv, llm_transcript.jsonl).
- Proposer: llm:local:Qwen/Qwen2.5-7B-Instruct; budget 24, used 16; stop: patience: 8 trials without an accepted improvement.
- Trials by source: {'fixed': 11, 'fallback': 4, 'llm:local:Qwen/Qwen2.5-7B-Instruct': 4}; accepted: 0.
- Best single + meta (CV): single uni2h +meta = 0.891.
- Selected: late phikonv2+uni2h+virchow2 prob_mean +meta = 0.882 +/- 0.030;
  gain -0.009 +/- 0.006.
- Selection optimism: {"optimism": 0.010019772585829777, "best_pooled": 0.8882992279006644, "corrected": 0.8782794553148346, "n_boot_used": 200, "top_pick_frequency": 0.41}

## Results (test set, evaluated once)

Test patients: 41; seeds: [0, 1, 2, 3, 4]. Seed mean ± sd = training-noise spread; [CI] = 95% stratified patient-bootstrap CI of the seed-ensemble.

| Configuration | Macro AUROC (seed mean ± sd) | Macro AUROC ensemble [95% CI] | Balanced acc. (seed mean ± sd) | Balanced acc. ensemble [95% CI] |
|---|---|---|---|---|
| Metadata only (age + sex) | 0.678 ± 0.000 | 0.678 [0.599, 0.750] | 0.230 ± 0.000 | 0.230 [0.137, 0.327] |
| uni2h alone | 0.861 ± 0.003 | 0.863 [0.815, 0.906] | 0.460 ± 0.029 | 0.456 [0.332, 0.580] |
| virchow2 alone | 0.853 ± 0.003 | 0.855 [0.803, 0.902] | 0.418 ± 0.014 | 0.414 [0.295, 0.536] |
| phikonv2 alone | 0.832 ± 0.009 | 0.831 [0.768, 0.887] | 0.457 ± 0.022 | 0.473 [0.338, 0.602] |
| Best single + metadata: single uni2h +meta | 0.867 ± 0.007 | 0.871 [0.821, 0.916] | 0.458 ± 0.031 | 0.474 [0.354, 0.598] |
| Fused (agent): late phikonv2+uni2h+virchow2 prob_mean +meta | 0.869 ± 0.003 | 0.874 [0.832, 0.913] | 0.420 ± 0.016 | 0.414 [0.290, 0.537] |

## Fused minus each other row (paired bootstrap on the same test patients)

| vs | Δ macro AUROC [95% CI] | P(Δ≤0) | Δ balanced acc. [95% CI] | P(Δ≤0) |
|---|---|---|---|---|
| meta_only | +0.196 [+0.110, +0.289] | 0.00 | +0.184 [+0.034, +0.334] | 0.01 |
| single_uni2h | +0.011 [-0.008, +0.034] | 0.14 | -0.042 [-0.101, +0.000] | 1.00 |
| single_virchow2 | +0.020 [-0.002, +0.043] | 0.04 | -0.000 [-0.048, +0.071] | 0.63 |
| single_phikonv2 | +0.043 [-0.011, +0.100] | 0.06 | -0.058 [-0.204, +0.077] | 0.80 |
| best_single_meta | +0.003 [-0.020, +0.027] | 0.38 | -0.060 [-0.125, +0.000] | 1.00 |

Fused beats every single-model row (ensemble point estimate): {'macro_auroc': True, 'balanced_accuracy': False}; (seed means): {'macro_auroc': True, 'balanced_accuracy': False}

## Per-class OvR AUROC (seed-ensemble)

| Configuration | In situ | Lepidic | Acinar | Papillary | Micropapillary | Solid | Cribriform |
|---|---|---|---|---|---|---|---|
| Metadata only (age + sex) | 0.764 | 0.760 | 0.500 | 0.729 | 0.347 | 0.809 | 0.838 |
| uni2h alone | 0.997 | 0.865 | 0.814 | 0.886 | 0.789 | 0.838 | 0.851 |
| virchow2 alone | 0.983 | 0.872 | 0.881 | 0.822 | 0.750 | 0.886 | 0.789 |
| phikonv2 alone | 0.993 | 0.865 | 0.829 | 0.716 | 0.828 | 0.710 | 0.877 |
| Best single + metadata: single uni2h +meta | 0.997 | 0.905 | 0.800 | 0.909 | 0.800 | 0.833 | 0.851 |
| Fused (agent): late phikonv2+uni2h+virchow2 prob_mean +meta | 1.000 | 0.912 | 0.867 | 0.860 | 0.800 | 0.848 | 0.833 |

## Agent selection (development CV, before test)

- Selected: late phikonv2+uni2h+virchow2 prob_mean +meta (CV 0.882 ± 0.030)
- Best single + meta: single uni2h +meta (CV 0.891)
- CV gain -0.009 ± 0.006 (significant at 1 SE: False)
- Selection optimism: 0.010
- Equal-budget single-model search (14 trials): single uni2h +meta (CV 0.891); fused gain vs it -0.009 ± 0.006 (significant at 1 SE: False)

## Error analysis (fused, test)

- Errors: 21/41; of which predicted pattern is listed in the slide's Multi_Subtypes:
  4.
- Most frequent confusions (true -> pred): Micropapillary->Papillary (3), Lepidic->In situ (2), Acinar->Micropapillary (2), Papillary->Micropapillary (2), Solid->Cribriform (2), Acinar->Papillary (1), Acinar->Lepidic (1), Cribriform->Solid (1).
- Errors by diagnosis: {'invasive adenocarcinoma (IA)': 19, 'minimally invasive adenocarcinoma (MIA)': 2}
  (test patients: {'Adenocarcinoma in situ(AIS)': 9, 'invasive adenocarcinoma (IA)': 29, 'minimally invasive adenocarcinoma (MIA)': 3}).
- Figures: artifacts/results/figures/ (confusion, forest plots, per-class AUROC, curves, agent trajectory).
