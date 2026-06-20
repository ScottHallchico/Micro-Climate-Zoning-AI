# Surrogate Recovery Strategy

## Root Cause Diagnosis
The LOAO generalization collapse is caused by a compound failure:
1. **Target Scaling**: Standard scaling on heavy-tailed pressure causes gradient explosion.
2. **Edge Connectivity**: Isotropic KNN allows back-propagation against the wind, blurring wakes.
3. **Feature Poverty**: XYZ coordinates alone force the network to implicitly learn geometry, which fails on unseen shapes.

## Recommendation: B & A
Requires Feature Redesign (Morphological Tokens) and Architecture Upgrade (Graph Transformer with Aerodynamic Edges).
