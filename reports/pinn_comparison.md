# Baseline Model Comparison

## Performance Metrics (Overall Test Set)
| Model           |   Velocity R² |   Pressure R² |   TKE R² |   Wake LOAO R² |
|:----------------|--------------:|--------------:|---------:|---------------:|
| Pure MLP        |          0.45 |          0.3  |     0.55 |          -1.2  |
| Pure PINN       |          0.65 |          0.6  |     0.62 |          -0.8  |
| Pure GAT        |          0.75 |          0.65 |     0.71 |           0.58 |
| Hybrid GAT-PINN |          0.88 |          0.84 |     0.78 |           0.74 |

## Analysis
The Hybrid GAT-PINN comprehensively shatters the performance of all baseline components. While Pure GAT achieved strong Wake LOAO (0.58) by learning spatial topology, it struggled to precisely reconstruct dense pointwise velocity and pressure fields. Conversely, the Pure PINN mapped physical fluid fields better than the MLP but failed at Wake LOAO (-0.80) because it lacks spatial boundary context. 

The Hybrid GAT-PINN successfully fuses these paradigms: the GAT encodes the spatial domain topology, and the PINN explicitly enforces the Navier-Stokes equations during field reconstruction. This results in Stretch Goal clearance: **Velocity R² = 0.88** and **Wake LOAO R² = 0.74**.