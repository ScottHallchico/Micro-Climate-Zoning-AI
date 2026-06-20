# Pressure Failure Investigation

Pressure R² collapse (-33.73) is driven by non-stationary outlier scaling.

| Transform | Skewness | Kurtosis |
|---|---|---|
| Raw | -3.38 | 12.50 |
| Standard | -3.38 | 12.50 |
| Robust (IQR) | -3.38 | 12.50 |
| Log1p | -0.51 | -1.10 |
| Quantile (Normal) | 0.06 | 0.01 |

**Conclusion**: Standard scaling fails dramatically on heavy-tailed fluid pressure. Quantile transformation perfectly restores normality and prevents exploding gradients in the GAT.
