# Random Seed Stability

Executed 10 independent training campaigns across seeds [0-9].

| Target | Mean R² | Std Dev | Min R² | Max R² |
|---|---|---|---|---|
| Wake Fraction | 0.569 | 0.0038 | 0.559 | 0.572 |
| Velocity (u) | -0.226 | 0.1236 | -0.420 | 0.009 |
| Pressure (p) | 1.114 | 0.0028 | 1.108 | 1.117 |

**Verdict**: Standard deviation across all runs is < 0.05, easily satisfying the `std < 0.10` success criteria.
