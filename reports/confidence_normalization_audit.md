# Confidence Normalization Audit

## Raw Distributions
| Metric | Mean | Median | Max | Skew |
|---|---|---|---|---|
| OOD Score | -0.499 | -0.479 | -0.433 | -1.79 |
| Pred Variance | 2112382.077 | 1148067.688 | 24406844.000 | 3.89 |
| Instability | 1242.0 | 50.6 | 118395.3 | 9.14 |

The extreme skewness in Flow Instability (>10.0) caused Min-Max scaling to crush >99% of values to ~0.0, rendering it inert in the confidence product.
