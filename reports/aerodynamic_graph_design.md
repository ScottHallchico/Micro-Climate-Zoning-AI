# Aerodynamic Graph Design

Replaced isotropic spatial KNN with Wind-Aware Edges.

## Edge Pruning Logic
```python
edge_vector = target_pos - source_pos
alignment = dot(normalize(edge_vector), wind_vector)
valid_edges = alignment > 0.0 # Only allow message passing DOWNWIND
```

## Edge Attributes Added
- Distance
- Bearing
- Wind Alignment
- Height Differential
- Blockage Coefficient
