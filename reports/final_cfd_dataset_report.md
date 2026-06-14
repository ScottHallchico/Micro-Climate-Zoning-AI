# Final CFD Dataset Report

## Summary
- **Total Configurations Evaluated**: 16
- **Successful Convergences**: 16
- **Failure Rate**: 0.0%

## ML Dataset Extracted
The master training dataset has been compiled and saved to `data/ml/cfd_training_dataset.parquet`.
Total samples: 16 (Rows represent Archetype × Scenario combinations).

### Feature Space
* **Morphology**: building density, frontal area density, roughness length, canyon metrics.
* **Weather**: wind speed, direction, temperature, humidity.
* **CFD Targets**: mean/max velocity, VEI, turbulence intensity, wake fraction, comfort score.

## Target Variables
Recommended targets for PINN and Surrogate Modeling:
1. `cfd_ventilation_efficiency` (Scalar mapping for overall flow blockage).
2. `cfd_pedestrian_comfort` (Categorical/Scalar for zoning compliance).
3. `cfd_mean_velocity` (Direct field prediction proxy).
