# Loss Decomposition

Evaluating individual MSE magnitudes on identical scale (Yeo-Johnson norm):
- Velocity MSE: ~0.8
- Pressure MSE: ~4.2
- Wake BCE: ~0.6

**Analysis**: Pressure gradients dominate the aggregate loss backpropagation by 5x. Equal weighting destroys velocity convergence.