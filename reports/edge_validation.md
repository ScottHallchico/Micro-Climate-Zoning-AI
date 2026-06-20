# Aerodynamic Edge Validation

Audited adjacency matrix construction logic.

## Wind-Aware Edge Logic
- Edge vectors computed directly from `(x_target - x_source)`.
- Masking vector extracted from global `wind_direction` inlet boundary condition.
- Independent of any CFD flow fields (velocity vectors are explicitly forbidden during graph construction).

**Verdict**: Aerodynamic edge pruning is physically legitimate and strictly zero-leakage.
