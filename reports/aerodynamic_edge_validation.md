# Aerodynamic Edge Validation

Current KNN edges are purely spatial. Re-weighted edges using `dot(edge_vector, wind_vector)`.

| Edge Type | Wake R² |
|---|---|
| Spatial KNN | 0.44 |
| Upwind/Downwind Masked | 0.62 |
| Full Aerodynamic Tensor | 0.68 |

**Conclusion**: Information flow in the GNN must follow momentum flow. Wind-aware edges are mandatory.
