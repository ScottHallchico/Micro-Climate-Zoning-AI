# Active Learning Strategy

1. **Initial Prior**: 20 cases selected via Latin Hypercube Sampling.
2. **Round 2 (Uncertainty Sampling)**: 40 cases selected by maximizing the predictive variance of a Matern Gaussian Process.
3. **Round 3 (Expected Improvement)**: 40 cases selected focusing on boundary gradients.

Total real CFD simulations executed: 100.
