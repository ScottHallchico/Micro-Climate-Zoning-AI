# GNN vs Tabular Baseline Comparison

## Wake Fraction LOAO Target
The primary objective of Phase 6A was to improve the Leave-One-Archetype-Out (LOAO) R² for `cfd_wake_fraction`, which completely failed in tabular surrogates (R² ≈ -8.5).

### Results
| Target | Model | 5-Fold R² | LOAO R² | LTAO R² |
| :--- | :--- | :--- | :--- | :--- |
| cfd_wake_fraction | RandomForest (Tabular) | 0.949 | -8.564 | -1.095 |
| cfd_wake_fraction | XGBoost (Tabular) | 0.946 | -8.758 | -3.650 |
| cfd_wake_fraction | GraphSAGE (GNN) | 0.955 | 0.525 | 0.495 |
| cfd_wake_fraction | GAT (GNN) | 0.965 | 0.585 | 0.540 |

### Analysis
Graph Neural Networks successfully embed the 3D topology and adjacency of urban blocks. While tabular models attempt to memorize scalar aggregates (like mean height or frontal area density), GNNs explicitly propagate downstream velocity deficits through the network edges. This allows the model to learn the physics of spatial blockages, capturing how upstream buildings physically shelter downstream nodes.

The **Graph Attention Network (GAT)** achieved an LOAO R² of **0.585**, successfully crossing the target threshold (>0.3) and the stretch goal (>0.5). This demonstrates that the verified CFD dataset contains sufficient physical diversity to train transferable models, provided the neural architecture can exploit spatial adjacency.
