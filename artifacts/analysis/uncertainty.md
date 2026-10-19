# What can a 41-patient test set resolve? (generated)

## Exact: one test patient's effect on balanced accuracy

| Subtype | Test patients | One patient moves balanced acc. by |
|---|---|---|
| In situ | 9 | 1.6 points |
| Lepidic | 4 | 3.6 points |
| Acinar | 6 | 2.4 points |
| Papillary | 8 | 1.8 points |
| Micropapillary | 5 | 2.9 points |
| Solid | 6 | 2.4 points |
| Cribriform | 3 | 4.8 points |

## Simulated sampling noise (latent-score model, 2000 test sets, real class counts)

Two classifiers share patient-level noise (rho = 0.6, typical of models on the same features). MDD = minimum detectable paired difference at alpha = 0.05, power = 0.8.

| signal A / B | AUROC A | AUROC B | SD(AUROC) | SD(ΔAUROC) | MDD AUROC | bacc A | bacc B | SD(bacc) | SD(Δbacc) | MDD bacc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 / 1.0 | 0.784 | 0.785 | 0.045 | 0.042 | 0.116 | 0.416 | 0.415 | 0.082 | 0.090 | 0.253 |
| 1.5 / 1.5 | 0.886 | 0.887 | 0.033 | 0.031 | 0.088 | 0.583 | 0.580 | 0.082 | 0.091 | 0.256 |
| 2.0 / 2.0 | 0.951 | 0.951 | 0.020 | 0.021 | 0.058 | 0.733 | 0.733 | 0.073 | 0.083 | 0.233 |
| 2.0 / 1.5 | 0.951 | 0.887 | 0.020 | 0.029 | 0.080 | 0.733 | 0.580 | 0.073 | 0.088 | 0.247 |

Reading: with equal true performance (rows 1-3) the SD of a paired difference is the noise floor. A fused-vs-single gain smaller than the MDD cannot be reliably shown on this test set, so the development-CV evidence and paired bootstrap must carry the claim.
