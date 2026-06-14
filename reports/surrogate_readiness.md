# Surrogate Readiness Assessment

## Dataset Characteristics
- **Sample Count**: 768 physics-based CFD simulations
- **Feature Diversity**: Excellent (Covers bounds of urban morphology + comprehensive weather envelope).
- **Target Diversity**: Non-linear aerodynamic effects successfully captured (canyon channeling, wake expansion).
- **Complexity**: Highly non-linear; linear models will fail.

## Algorithm Recommendations (Ranked)

1. **XGBoost / LightGBM**: 
   *Suitability*: **Highest**. Tree-based gradient boosting perfectly handles the tabular, highly non-linear nature of this dataset (768 rows × 12 features). They will provide rapid, interpretable feature importance.
2. **Random Forest**:
   *Suitability*: **High**. Robust to overfitting, excellent baseline, but potentially slightly less accurate than XGBoost on aerodynamic edge cases.
3. **PINN (Physics-Informed Neural Network)**:
   *Suitability*: **Moderate (for now)**. A dataset of 768 samples is sufficient for a 0D/1D PINN, but fully resolved 3D field prediction PINNs usually require 10,000+ spatial data points per geometry. Good for point-wise surrogate.
4. **Graph Neural Network (GNN)**:
   *Suitability*: **Low**. Requires graph-based representation of the buildings/streets. The current dataset is fully aggregated tabular data.

## Next Steps
Proceed to Phase 5B: Train XGBoost and Random Forest regressors on the `cfd_expanded_dataset.parquet` to predict `cfd_ventilation_efficiency` and `cfd_pedestrian_comfort`.
