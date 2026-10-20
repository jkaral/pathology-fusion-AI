# Agent decision log

- Selection metric: **macro_auroc**, stratified 5-fold CV on 162 development patients (train+val; test never touched).
- Budget: 16 fusion trials (+3 references); used 14. Stop reason: patience: 8 trials without an accepted improvement.
- Acceptance: paired fold gain > z x Nadeau-Bengio corrected SE. Final pick: one-SE rule on the fold-paired scale (least complex trial within 1 paired SE of the best).
- Best single model + metadata: **single phikon +meta** (CV 0.848).
- **Selected**: #1 **single phikon +meta** (CV 0.848 ± 0.026); gain vs best single +0.000 ± 0.000 (corrected SE).
- Selection optimism (bootstrap winner's curse): 0.011; best pooled OOF 0.845 -> corrected 0.834.

| # | phase | source | config | CV mean ± sd | gain vs inc. ± SE | pooled bal.acc | decision | reasoning |
|---|---|---|---|---|---|---|---|---|
| 0 | reference | fixed | single lunit_dino +meta | 0.843 ± 0.034 |  | 0.519 | reference | reference: single foundation model + metadata (the bar fusion must clear) |
| 1 | reference | fixed | single phikon +meta | 0.848 ± 0.039 |  | 0.489 | reference | reference: single foundation model + metadata (the bar fusion must clear) |
| 2 | reference | fixed | single phikonv2 +meta | 0.817 ± 0.048 |  | 0.479 | reference | reference: single foundation model + metadata (the bar fusion must clear) |
| 3 | coverage | fixed | single lunit_dino norm=l2 +meta | 0.741 ± 0.069 | -0.107 ± 0.025 | 0.291 | rejected | coverage 1/6: map the single region before adapting |
| 4 | coverage | fixed | single lunit_dino norm=zscore pca=128 +meta | 0.812 ± 0.042 | -0.036 ± 0.008 | 0.443 | rejected | coverage 2/6: map the single region before adapting |
| 5 | coverage | fixed | single phikon norm=l2 +meta | 0.726 ± 0.047 | -0.122 ± 0.024 | 0.368 | rejected | coverage 3/6: map the single region before adapting |
| 6 | coverage | fixed | single phikon norm=zscore pca=128 +meta | 0.790 ± 0.034 | -0.058 ± 0.024 | 0.432 | rejected | coverage 4/6: map the single region before adapting |
| 7 | coverage | fixed | single phikonv2 norm=l2 +meta | 0.701 ± 0.047 | -0.147 ± 0.017 | 0.344 | rejected | coverage 5/6: map the single region before adapting |
| 8 | coverage | fixed | single phikonv2 norm=zscore pca=128 +meta | 0.834 ± 0.030 | -0.014 ± 0.034 | 0.433 | rejected | coverage 6/6: map the single region before adapting |
| 9 | adaptive | fallback | single phikonv2 pca=128 +meta | 0.752 ± 0.033 | -0.096 ± 0.027 | 0.358 | rejected | [LLM proposal invalid: config already tried: single phikonv2 norm=zscore pca=128 +meta] local move from rank-1 fusion trial (single phikonv2 norm=zscore pca=128 +meta = 0.834±0.030): norm zscore->none |
| 10 | adaptive | fallback | single phikonv2 norm=l2 pca=128 +meta | 0.602 ± 0.078 | -0.246 ± 0.055 | 0.222 | rejected | [LLM proposal invalid: config already tried: single lunit_dino norm=zscore pca=128 +meta] local move from rank-1 fusion trial (single phikonv2 norm=zscore pca=128 +meta = 0.834±0.030): norm zscore->l2 |
| 11 | adaptive | fallback | single phikonv2 norm=zscore +meta | 0.845 ± 0.038 | -0.003 ± 0.009 | 0.470 | rejected | [LLM proposal invalid: single needs exactly one model] local move from rank-1 fusion trial (single phikonv2 norm=zscore pca=128 +meta = 0.834±0.030): pca_dim 128->None |
| 12 | adaptive | fallback | single phikonv2 norm=zscore pca=64 +meta | 0.780 ± 0.049 | -0.068 ± 0.014 | 0.397 | rejected | [LLM proposal invalid: config already tried: single lunit_dino +meta] local move from rank-1 fusion trial (single phikonv2 norm=zscore +meta = 0.845±0.038): pca_dim None->64 |
| 13 | adaptive | fallback | single phikonv2 norm=zscore pca=256 +meta | 0.809 ± 0.041 | -0.039 ± 0.017 | 0.429 | rejected | [LLM proposal invalid: config already tried: single lunit_dino norm=zscore pca=128 +meta] local move from rank-1 fusion trial (single phikonv2 norm=zscore +meta = 0.845±0.038): pca_dim None->256 |
| 14 | adaptive | llm:local:Qwen/Qwen2.5-3B-Instruct | single lunit_dino norm=zscore pca=64 +meta | 0.793 ± 0.044 | -0.055 ± 0.016 | 0.421 | rejected | The incumbent single phikon +meta has shown slightly better performance than the reference single lunit_dino +meta. The next trial should explore under-explored regions to find potential improvements. |
| 15 | adaptive | fallback | single lunit_dino norm=zscore +meta | 0.840 ± 0.050 | -0.008 ± 0.013 | 0.532 | rejected | [LLM proposal invalid: config already tried: single lunit_dino norm=zscore pca=64 +meta] local move from rank-1 fusion trial (single phikonv2 norm=zscore +meta = 0.845±0.038): models phikonv2->lunit_dino |
| 16 | adaptive | fallback | single phikon norm=zscore +meta | 0.839 ± 0.030 | -0.009 ± 0.011 | 0.491 | rejected | [LLM proposal invalid: config already tried: single lunit_dino norm=zscore pca=128 +meta] local move from rank-1 fusion trial (single phikonv2 norm=zscore +meta = 0.845±0.038): models phikonv2->phikon |
