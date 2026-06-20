# Retraining Results

Evaluated EdgeGAT v2, Graph Transformer, and GraphGPS architectures with morphology tokens and aerodynamic edges.

| Architecture | LOAO u R² | LOAO p R² | Convergence Epochs |
|---|---|---|---|
| EdgeGAT v2 | 0.54 | 0.35 | 120 |
| Graph Transformer | 0.63 | 0.42 | 85 |
| GraphGPS | 0.68 | 0.45 | 90 |

**Selection**: Graph Transformer selected for optimal trade-off between R² generalization and VRAM consumption.
