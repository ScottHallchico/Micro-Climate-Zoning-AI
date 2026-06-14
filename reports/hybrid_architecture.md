# Hybrid GAT-PINN Architecture

## Graph Encoder
- **Type**: Graph Attention Network (GAT)
- **Input Features**: Building height, area, perimeter, compactness.
- **Output Dimension**: 128 (z_graph embedding)

## Physics Decoder
- **Type**: Fully Connected MLP
- **Architecture**: 131 -> 256 -> 256 -> 128 -> 5
- **Outputs**: u, v, w, p, k

## Physics Losses
- **Continuity**: Enforced via AutoGrad tracking the spatial coordinates.
- **Momentum**: Enforced via RANS approximation.
- **Lambda Multipliers**: λ₁=0.10, λ₂=0.05.
