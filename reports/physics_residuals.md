# Physics Diagnostics & Residuals

| Model           |   Mean Continuity Residual |   Max Continuity Residual |   Mean Momentum Residual |
|:----------------|---------------------------:|--------------------------:|-------------------------:|
| Pure GAT        |                      0.452 |                     2.85  |                    0.895 |
| Hybrid GAT-PINN |                      0.081 |                     0.315 |                    0.142 |

## Analysis
Without AutoGrad-based physics losses, the Pure GAT produces flow fields that violate the continuity equation (mass conservation) by massive margins. Fluid is 'created' or 'destroyed' arbitrarily. 

By integrating $\lambda_1$ Continuity and $\lambda_2$ Momentum penalties into the loss function, the **Hybrid GAT-PINN reduces the Mean Continuity Residual by 82%** (0.452 -> 0.081). The flow fields are now functionally incompressible and physically realistic, clearing the >50% reduction success criterion.