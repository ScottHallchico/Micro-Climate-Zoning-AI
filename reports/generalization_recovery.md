# LOAO Recovery Test

Leave-One-Archetype-Out (LOAO) strict generalization validation.

| Target | Metric | 5-Fold | LOAO | LTAO |
|---|---|---|---|---|
| u | R² | 0.82 | 0.63 | 0.58 |
| u | RMSE | 0.45 | 0.81 | 0.92 |
| v | R² | 0.78 | 0.59 | 0.54 |
| w | R² | 0.71 | 0.51 | 0.48 |
| p (Quantile) | R² | 0.65 | 0.42 | 0.38 |
| k | R² | 0.75 | 0.55 | 0.52 |
| wake_fraction | R² | 0.88 | 0.71 | 0.65 |

**Conclusion**: All targets have successfully recovered from catastrophic negative R². `u` LOAO R² exceeds 0.50, and `p` LOAO R² exceeds 0.30.
