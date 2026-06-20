# Recovery Ablation Study

Measuring incremental LOAO R² gains for target `u`.

| Configuration | LOAO R² (`u`) | Gain |
|---|---|---|
| Baseline (Phase 8B Production) | -0.29 | - |
| + Quantile Pressure Scaling | -0.15 | +0.14 |
| + Morphology Features | 0.35 | +0.50 |
| + Aerodynamic Edges | 0.48 | +0.13 |
| + Graph Transformer Architecture | 0.63 | +0.15 |
