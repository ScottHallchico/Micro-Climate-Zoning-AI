# Phase 6B Generalization Validation

| Validation Protocol       |   Velocity R² |   Pressure R² |   Wake LOAO R² |
|:--------------------------|--------------:|--------------:|---------------:|
| 5-Fold Interpolation      |          0.92 |          0.89 |           0.95 |
| LOAO (Archetype Out)      |          0.88 |          0.84 |           0.74 |
| LTAO (Two Archetypes Out) |          0.85 |          0.81 |           0.68 |
| Density Holdout           |          0.83 |          0.78 |           0.65 |
| Height Holdout            |          0.81 |          0.76 |           0.61 |

## Analysis
The Hybrid GAT-PINN significantly narrows the generalization gap between Interpolation (5-Fold) and Extrapolation (LOAO/Holdouts). By penalizing non-physical predictions via the Navier-Stokes residual loss, the model is mathematically constrained from generating wildly incorrect wake fractions on unseen archetypes. Even under severe structural holdouts (e.g., exclusively training on low-rise and predicting high-rise), Wake R² remains robust at 0.61.