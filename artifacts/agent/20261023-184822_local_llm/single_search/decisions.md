# Agent decision log

- Selection metric: **macro_auroc**, stratified 5-fold CV on 162 development patients (train+val; test never touched).
- Budget: 16 fusion trials (+3 references); used 14. Stop reason: patience: 8 trials without an accepted improvement.
- Acceptance: paired fold gain > z x Nadeau-Bengio corrected SE. Final pick: one-SE rule on the fold-paired scale (least complex trial within 1 paired SE of the best).
- Best single model + metadata: **single uni2h +meta** (CV 0.891).
- **Selected**: #1 **single uni2h +meta** (CV 0.891 ± 0.036); gain vs best single +0.000 ± 0.000 (corrected SE).
- Selection optimism (bootstrap winner's curse): 0.007; best pooled OOF 0.881 -> corrected 0.874.

| # | phase | source | config | CV mean ± sd | gain vs inc. ± SE | pooled bal.acc | decision | reasoning |
|---|---|---|---|---|---|---|---|---|
| 0 | reference | fixed | single phikonv2 +meta | 0.819 ± 0.047 |  | 0.477 | reference | reference: single foundation model + metadata (the bar fusion must clear) |
| 1 | reference | fixed | single uni2h +meta | 0.891 ± 0.053 |  | 0.576 | reference | reference: single foundation model + metadata (the bar fusion must clear) |
| 2 | reference | fixed | single virchow2 +meta | 0.871 ± 0.050 |  | 0.487 | reference | reference: single foundation model + metadata (the bar fusion must clear) |
| 3 | coverage | fixed | single phikonv2 norm=l2 +meta | 0.706 ± 0.037 | -0.185 ± 0.023 | 0.344 | rejected | coverage 1/6: map the single region before adapting |
| 4 | coverage | fixed | single phikonv2 norm=zscore pca=128 +meta | 0.803 ± 0.039 | -0.088 ± 0.027 | 0.441 | rejected | coverage 2/6: map the single region before adapting |
| 5 | coverage | fixed | single uni2h norm=l2 +meta | 0.844 ± 0.064 | -0.047 ± 0.022 | 0.446 | rejected | coverage 3/6: map the single region before adapting |
| 6 | coverage | fixed | single uni2h norm=zscore pca=128 +meta | 0.860 ± 0.054 | -0.032 ± 0.012 | 0.546 | rejected | coverage 4/6: map the single region before adapting |
| 7 | coverage | fixed | single virchow2 norm=l2 +meta | 0.832 ± 0.045 | -0.059 ± 0.011 | 0.390 | rejected | coverage 5/6: map the single region before adapting |
| 8 | coverage | fixed | single virchow2 norm=zscore pca=128 +meta | 0.863 ± 0.040 | -0.028 ± 0.009 | 0.487 | rejected | coverage 6/6: map the single region before adapting |
| 9 | adaptive | llm:local:Qwen/Qwen2.5-7B-Instruct | single uni2h norm=l2 pca=256 +meta | 0.691 ± 0.069 | -0.200 ± 0.037 | 0.241 | rejected | The current incumbent is uni2h, which has the highest macro_auroc. To further refine, we should explore variations of uni2h, specifically focusing on normalization and PCA to see if they can improve performance. |
| 10 | adaptive | llm:local:Qwen/Qwen2.5-7B-Instruct | single phikonv2 norm=zscore pca=64 +meta | 0.784 ± 0.046 | -0.107 ± 0.016 | 0.365 | rejected | The incumbent is uni2h with meta. Probing other normalization methods and PCA dimensions can provide more information about the potential for improvement. |
| 11 | adaptive | fallback | single virchow2 pca=128 +meta | 0.861 ± 0.036 | -0.030 ± 0.015 | 0.508 | rejected | [LLM proposal invalid: single needs exactly one model] local move from rank-1 fusion trial (single virchow2 norm=zscore pca=128 +meta = 0.863±0.040): norm zscore->none |
| 12 | adaptive | llm:local:Qwen/Qwen2.5-7B-Instruct | single uni2h norm=zscore pca=64 +meta | 0.837 ± 0.059 | -0.054 ± 0.021 | 0.492 | rejected | The incumbent is #1 (uni2h), but there is potential to improve by exploring different normalization and PCA settings. The previous trial with uni2h and z-score normalization showed some promise, achieving a macro_auroc of 0.860, which is close to the incumbent's score of 0.891. |
| 13 | adaptive | fallback | single virchow2 norm=l2 pca=128 +meta | 0.666 ± 0.086 | -0.225 ± 0.069 | 0.288 | rejected | [LLM proposal invalid: single needs exactly one model] local move from rank-1 fusion trial (single virchow2 norm=zscore pca=128 +meta = 0.863±0.040): norm zscore->l2 |
| 14 | adaptive | fallback | single virchow2 norm=zscore +meta | 0.875 ± 0.050 | -0.016 ± 0.016 | 0.547 | rejected | [LLM proposal invalid: config already tried: single phikonv2 norm=zscore pca=128 +meta] local move from rank-1 fusion trial (single virchow2 norm=zscore pca=128 +meta = 0.863±0.040): pca_dim 128->None |
| 15 | adaptive | fallback | single virchow2 norm=zscore pca=64 +meta | 0.845 ± 0.050 | -0.047 ± 0.020 | 0.492 | rejected | [LLM proposal invalid: config already tried: single uni2h norm=zscore pca=128 +meta] local move from rank-1 fusion trial (single virchow2 norm=zscore +meta = 0.875±0.050): pca_dim None->64 |
| 16 | adaptive | llm:local:Qwen/Qwen2.5-7B-Instruct | single uni2h norm=zscore pca=256 +meta | 0.874 ± 0.045 | -0.017 ± 0.014 | 0.550 | rejected | The incumbent is uni2h with meta, which has the highest macro_auroc. To potentially improve, we should explore variations of uni2h, particularly focusing on normalization and PCA, as these have shown mixed results in previous trials. |
