# Final Surrogate Model Comparison

Based on R², RMSE, and physical plausibility, the models are ranked as follows:

1. **XGBoost**: Highest R² and lowest RMSE overall. Demonstrates excellent stability and captures non-linear aerodynamic relationships best.
2. **LightGBM**: Very similar performance to XGBoost but slightly faster training.
3. **Random Forest**: Solid baseline but slightly worse extrapolation stability compared to gradient boosting.

**Recommendation**: Deploy **XGBoost** as the primary CFD surrogate for Phase 5B. It offers the best balance of interpolation accuracy and robustness.
