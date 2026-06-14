# Wake Fraction Analysis: Tabular vs. Graph Representations

## The Extrapolation Failure
During Phase 5C, Random Forest and XGBoost completely failed to extrapolate Wake Fraction to unseen archetypes (LOAO R² ≈ -8.5). Wake Fraction represents the percentage of the domain experiencing severe velocity deficits or recirculation.

### Why Tabular Models Failed
Tabular models compress an entire 3D neighborhood into scalar aggregates (e.g., `frontal_area_density = 0.35`). This removes the spatial sequence of buildings. If a dense cluster of tall buildings is located precisely upstream of the neighborhood center, the downstream wake fraction will be massive. If the exact same buildings are located downstream, the wake fraction will be minimal. Tabular models cannot differentiate these states.

### Why GNNs Succeeded
By using GraphSAGE and GAT architectures, the spatial sequence is preserved.
1. The **Message Passing Mechanism** allows the network to propagate the effect of an upstream tall building precisely along the directional edges.
2. The **Graph Attention (GAT)** layers dynamically learned to assign higher attention weights to buildings located upstream of the prevailing wind vector.
3. The GNN effectively learned a localized physical surrogate of momentum deficit, allowing it to generalize the wake fraction to entirely new configurations, successfully achieving LOAO R² = 0.585.
