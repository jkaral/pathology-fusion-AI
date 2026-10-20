# Pilot results: real slides, ungated encoders, CPU

**What these numbers are:** the pipeline on all **203 real slides**, but with three *ungated*
encoders (Phikon-v2, Phikon, Lunit-DINO) at **256 tiles/slide**, scored by **5-fold CV on the
162 development patients**. Stage 3 was never run, so the test split is untouched. These are
not the brief's headline results (UNI2-h / Virchow2 / Phikon-v2 on GPU); see `docs/PILOT.md`.
Every number below is copied from a generated file named alongside it.

## 1. Extraction (`artifacts/embeddings_pilot/extraction_summary.json`)

- 203 slides, 51,843 tiles in total (one slide has only 131 tissue tiles; the rest are capped at 256).
- Wall clock on 4 CPU cores (bf16):
  - download 0.71 h
  - decode at 20x 0.97 h
  - tissue mask + grid 0.06 h
  - encoders: Phikon-v2 2.53 h (5.7 tiles/s), Phikon 0.74 h (19.4 tiles/s), Lunit-DINO 0.22 h (65.7 tiles/s)
- Feature cache: 0.23 GB.
- Tissue tiles per slide at 20x: median 1,871 (IQR 1,466-2,246, range 131-3,067), so the
  headline cap of 4096 truncates **no** slide (`artifacts/pilot/tiling/tiling_report.md`).

## 2. Agent search (`artifacts/pilot/agent/pilot_qwen3b/`)

Development-CV macro AUROC (mean of 5 folds; `trials.csv`, `decisions.md`):

| Trial | Configuration (+ age, sex) | CV macro AUROC | Pooled bal. acc. |
|---|---|---|---|
| ref | Lunit-DINO alone | 0.843 | 0.519 |
| ref | **Phikon alone (best single)** | **0.848** | 0.489 |
| ref | Phikon-v2 alone | 0.817 | 0.479 |
| cov | **Early concat, raw (selected)** | **0.869** | 0.538 |
| cov | Early concat, z-score + PCA-128 | 0.866 | 0.522 |
| cov | Intermediate, gated | 0.857 | 0.539 |
| cov | Intermediate, attention | 0.856 | 0.513 |
| cov | Late, OOF-weighted vote | 0.852 | 0.528 |
| cov | Late, probability mean | 0.847 | 0.540 |
| cov | Late, stacking | 0.822 | 0.441 |
| cov | Early concat, L2 | 0.801 | 0.396 |

- **Fused vs best single, paired over folds: +0.021 ± 0.008** (Nadeau-Bengio corrected SE, about 2.6 SE).
- **Equal-budget control** (`single_search/`): the same agent gave the single-model space
  14 trials (model × normalisation × PCA). It kept *default Phikon + metadata* (0.848), so
  the fused gain against a single model with matched tuning is the same **+0.021 ± 0.008**.
- Selection optimism (bootstrap winner's curse over the 16 fusion trials): **0.010**. Best
  pooled OOF AUROC 0.863 → **0.853 after correction**.
- Search dynamics:
  - 3 references + 8 coverage trials + 8 adaptive trials.
  - The search stopped by **patience**: 8 adaptive trials without an accepted improvement.
  - The only accepted improvement was coverage trial #3; nothing adaptive beat it beyond noise.
- LLM policy (Qwen2.5-3B, CPU):
  - 5 of the 8 adaptive fusion proposals came from the LLM; 3 fell back to the deterministic
    proposer after an invalid or duplicate reply. In the single search only 1 of 8 adaptive proposals came from the LLM (7 fallbacks).
  - Its free-text reasoning is sometimes **factually wrong**. In #12 it describes the incumbent
    as "L2 + PCA-128", but the incumbent is raw concatenation.
  - Validation, de-duplication and the noise-aware acceptance rule contained this: no
    hallucinated reasoning changed the outcome. That is an argument for the 7B model on GPU.
- **Rule fix found here:** the one-SE final pick originally used the marginal SE of the best
  trial (±0.024). That SE is dominated by shared fold difficulty, so the rule chose
  late/prob-mean (0.847), which is no better than the best single model. It now uses the paired
  corrected SE, as the acceptance rule does (commit `d8cc938`, with a regression test). The
  search was replayed from cache: the trajectory was identical and only the final pick changed.
  This change was made after seeing development-CV results; the test set was not involved.

## 3. Shortcut / batch analysis

- **Slide ID alone predicts subtype:** development-CV macro AUROC 0.78 (IDs are grouped by subtype).
- **Images carry an acquisition signature** (`batch_probe/batch_probe.md`, all 162 development slides):
  - mean-pooled features from each encoder predict within-subtype slide-ID rank (CV R² 0.24-0.26, permutation p = 0.005);
  - colour statistics alone predict subtype at AUROC 0.64;
  - JPEG file size tracks ID order within subtype (Spearman ρ = +0.27, p = 0.0007), while tissue amount and slide dimensions do not. This points to scan or encoding settings drifting with acquisition order.
  - The preliminary 82-slide probe had said "no trace". It lacked the ID ranges where the effect lives, and the full run supersedes it.
- **But classification does not depend on it** (`within_batch/within_batch.md`). Scoring each
  model only *within* slide-ID blocks, where batch cannot separate classes:

| Model | Mean within-block macro AUROC |
|---|---|
| ID-only classifier | 0.561 (≈ chance) |
| Phikon-v2 alone | 0.808 |
| Lunit-DINO alone | 0.844 |
| Phikon alone | 0.845 |
| **Selected fusion** | **0.858** |

## 4. What this implies for the headline run

1. The agent, its guardrails and the equal-budget control behave as intended on real noise
   levels, and the fused-beats-single claim survives matched tuning (on development CV).
2. Keep the within-block evaluation in the report as the answer to the shortcut risk; run
   `scripts/08_within_batch_eval.py` on the headline agent run too.
3. Stain normalisation is optional. The batch signature looks like encoding settings more
   than colour, and it does not drive within-block performance.
4. Prefer the 7B LLM policy on Kaggle; the 3B model's rationales are unreliable.
