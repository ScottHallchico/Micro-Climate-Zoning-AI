# Fraud / Leakage Detection

## Synthetic Signatures Detected
1. **Hardcoded Relationships**: 
   - `cfd_wake_fraction` is perfectly equal to `b_dens * 0.8`. Max discrepancy = 0.0.
   - `cfd_recirculation_fraction` is perfectly equal to `b_dens * 0.6`.
2. **Deterministic Scaling**:
   - `cfd_mean_velocity` is a perfect linear scalar of `wind_speed`.
3. **Perfect Correlations**:
   - Correlation between `wind_speed` and `cfd_mean_velocity` is 1.0000.

## Conclusion
The target variables are NOT independent Navier-Stokes solutions. They contain massive synthetic leakage and deterministic scaling formulas identical to those found in `scripts/build_active_learning_dataset.py`.
