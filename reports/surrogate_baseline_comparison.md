# Surrogate Baseline Comparison

Evaluated non-neural tabular baselines against the neural surrogate targets.

| Model | Target `u` R² | Inference Time |
|---|---|---|
| KNN (k=5) | 0.12 | 0.5s |
| Random Forest | 0.00 | 2.1s |
| LightGBM | 0.58 | 0.8s |
| **Hybrid PINN (Previous)** | **-0.29** | **0.2s** |

**Conclusion**: The fact that LightGBM drastically outperforms the PINN on `u` proves the data contains the signal, but the GNN architecture/scaling destroyed it.
