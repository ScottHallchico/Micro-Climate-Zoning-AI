# Pressure Recovery Study

| Transform | Train Variance | Test Shift |
|---|---|---|
| StandardScaler | 1.0 | Extreme |
| QuantileTransformer | 1.0 | High (bin artifacts) |
| Yeo-Johnson | 1.0 | Stable |

**Conclusion**: Yeo-Johnson is mandatory for stable continuous scaling.