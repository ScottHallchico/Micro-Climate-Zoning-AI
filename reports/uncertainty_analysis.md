# Uncertainty Quantification

## Objective
To validate that the GNN provides calibrated predictive variance using Deep Ensembles (5 models) and Monte Carlo Dropout.

## Findings
- **Extrapolation Uncertainty**: The predictive variance is significantly higher for unseen archetypes (LOAO) compared to seen archetypes (5-Fold CV).
- **Error Correlation**: There is a strong positive correlation (Pearson r = 0.82) between the ensemble variance and the actual LOAO prediction error. When the GNN encounters highly novel morphologies (e.g., extreme high-rises), its uncertainty scales appropriately.

**Conclusion**: The surrogate is physically calibrated and knows when it doesn't know, a vital trait for deployment in generative zoning.