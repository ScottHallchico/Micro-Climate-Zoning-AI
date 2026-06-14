# Data Leakage Assessment

## Status: CRITICAL LEAKAGE DETECTED

## Mechanism of Leakage
The target variables in the current `.parquet` datasets were deterministically computed from the input variables using closed-form python expressions, rather than being extracted from independent, physics-based Navier-Stokes solutions. 

For example, in `phase5_expansion.py`, Ventilation Efficiency (VEI) was generated via:
`vei = max(0.1, 1.0 - (fad * 1.5)) * dir_mod`

If a Machine Learning model (e.g., XGBoost, Random Forest, or a Neural Network) is trained on this dataset, it will simply perform **symbolic regression**. The algorithm will effortlessly "learn" the Python coefficients (e.g., the $-1.5$ multiplier on `frontal_area_density`), yield a near-perfect $R^2 \approx 1.0$, and present a catastrophic illusion of predictive success.

This model would have zero aerodynamic validity and would fail instantly when presented with real-world fluid dynamic variance.

## Implication
**NO MACHINE LEARNING CAN COMMENCE** using the current `cfd_expanded_dataset.parquet` file. The ML models must learn the underlying non-linear physics of the Navier-Stokes equations from OpenFOAM, not the heuristics coded in the generation scripts.

---

## Active Learning Strategy Recommendation

To replace these synthetic placeholders with genuine, computationally expensive CFD data, we must strategically sample the parameter space. Given a compute budget, we cannot brute-force 768 simulations immediately. We recommend an **Active Learning (AL) Loop** to intelligently select the first 50–100 cases.

### Phase 1: Latin Hypercube Baseline (Cases 1–20)
1.  **Objective**: Establish a maximally diverse initial prior.
2.  **Method**: Use Latin Hypercube Sampling (LHS) across the `(Archetype, Wind Speed, Wind Direction)` design space.
3.  **Action**: Run these 20 OpenFOAM simulations to full convergence, extract real metrics, and train an initial weak Gaussian Process (GP) Surrogate Model.

### Phase 2: Uncertainty Sampling (Cases 21–60)
1.  **Objective**: Run simulations where the surrogate model is most confused.
2.  **Method**: Query the GP Surrogate for predictions across the remaining 748 unevaluated configurations. Select the 40 cases with the highest predictive variance (epistemic uncertainty).
3.  **Action**: Run these 40 OpenFOAM simulations. The surrogate will specifically request cases involving aerodynamic edge-cases (e.g., high-speed wind entering complex diagonal canyons).

### Phase 3: Gradient/Boundary Exploration (Cases 61–100)
1.  **Objective**: Refine the decision boundaries for critical outputs like "Pedestrian Discomfort Thresholds."
2.  **Method**: Expected Improvement (EI) or query-by-committee. Find regions where small changes in morphology or wind direction trigger massive non-linear shifts in Wake Fraction or VEI.
3.  **Action**: Execute the final 40 OpenFOAM runs focused on these highly-sensitive gradient zones.

### Conclusion
By adopting an Active Learning framework, we can achieve surrogate model performance with ~100 real CFD simulations that would normally require the full 768-case parametric sweep, saving ~85% of compute costs while completely eliminating the synthetic data leakage.
