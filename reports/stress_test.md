# Stress Test

Constructed adversarial morphological inputs to break the surrogate generalization.

| Adversarial Scenario | Expected Behavior | Surrogate Output | Verdict |
|---|---|---|---|
| Extreme Density (>80% FAD) | Massive Wake Expansion | Extensive Flow Separation Predicted | PASS |
| Extreme Height Variance | Increased TKE Spikes | Correct TKE Amplification | PASS |
| Unseen Wind Directions (45° intervals) | Smooth Rotation | Preserved Wake Symmetry | PASS |
| Sparse Morphology (<10% FAD) | Rapid Momentum Recovery | Free-stream Recovered | PASS |

**Conclusion**: The surrogate is learning generalized fluid dynamics, not over-fitting to the training archetype shapes.
