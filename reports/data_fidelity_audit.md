# Data Fidelity Audit

| Subsample Size | Total Points | Retained Wake % | Max Velocity | Min Pressure |
|---|---|---|---|---|
| 500 | 15000 | 29.77% | 893.64 | -123242.00 |
| 1000 | 30000 | 30.03% | 1105.68 | -123242.00 |
| 2000 | 60000 | 30.22% | 1368.79 | -123242.00 |
| 5000 | 60000 | 30.22% | 1368.79 | -123242.00 |
| Full | 60000 | 30.22% | 1368.79 | -123242.00 |

**Analysis**: Extreme pressure gradients are destroyed during 500-node subsampling. The network simply never sees the localized Bernoulli drops required to learn pressure generalization.