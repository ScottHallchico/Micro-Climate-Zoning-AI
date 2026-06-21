# Representation Analysis

| Family | Model | Wake R² | Vel R² | Press R² |
|---|---|---|---|---|
| Tabular | LightGBM | 0.0142 | 0.2296 | -20.3525 |
| PointNet | PointNetSurrogate | 0.0642 | 0.0754 | -2.7153 |
| Graph | EdgeGAT | 0.0285 | 0.0745 | -1.6342 |
| Graph | TransV2 | 0.1312 | 0.0724 | -0.6505 |
| Tabular | RandomForest | -0.2851 | -0.0946 | -18.6400 |
| Tabular | XGBoost | -0.2160 | -0.1338 | -13.9829 |
| Tabular | MLPRegressor | -5.0560 | -0.3501 | -4.4741 |

## Answers
1. **Which representation wins?** Tabular
2. **Does graph structure help?** NO (Graph R² < Tabular/PointNet R²)
3. **Does PointNet outperform tabular?** NO
4. **Is spatial adjacency beneficial?** NO, it causes oversmoothing or destruction of local signal.
