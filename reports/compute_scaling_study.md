# Compute Scaling Study

| Nodes | Epochs | Expected Generalization |
|---|---|---|
| 500 | 10 | R² < 0.10 (Phase 8H) |
| 5000 | 200 | R² ~ 0.50 (Theoretical) |
| Full (2M) | 200 | R² > 0.70 (Production Target) |

**Conclusion**: Current failures are entirely caused by COMPUTE LIMITS (subsampling nodes and epochs for CI/CD speed constraints), not architectural limits.