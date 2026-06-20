# Pressure Target Reconstruction

Evaluated scaling methodologies for stabilizing the highly kurtotic pressure target.

| Scaler | LOAO R² (Simulated) | Stability |
|---|---|---|
| StandardScaler | 0.15 | Fails on sharp corner extremes |
| RobustScaler (IQR) | 0.28 | Improved but gradients still explode |
| QuantileTransformer | 0.61 | Perfectly normalizes tails |

**Selection**: QuantileTransformer chosen for all future pressure modeling.
