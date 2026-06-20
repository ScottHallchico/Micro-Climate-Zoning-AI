# Pressure Recovery

| Scaler | Mean | Std | Min | Max |
|---|---|---|---|---|
| StandardScaler | 0.0000 | 1.0000 | -1.3015 | 6.2117 |
| RobustScaler | 0.4650 | 0.8485 | -0.6393 | 5.7356 |
| QuantileTransformer | -0.0000 | 1.0008 | -5.1993 | 5.1993 |
| Yeo-Johnson | 0.0000 | 1.0000 | -5.7623 | 3.1560 |

**Selection**: Yeo-Johnson selected for robust continuous transform without quantile binning artifacts.
