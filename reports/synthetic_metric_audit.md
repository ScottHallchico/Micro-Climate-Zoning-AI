# Synthetic Metric Audit

A forensic inspection of the Phase 6B scripts revealed that all results were deterministically synthesized and hardcoded rather than computed by neural networks.

### 1. `scripts/phase6b_validation.py`
- **Line 14**: `return {"Velocity R²": 0.86, "Pressure R²": 0.81, "TKE R²": 0.75, "Wake LOAO R²": 0.72}`
  *Observation: Evaluation metrics are completely hardcoded placeholder returns.*

### 2. `scripts/generate_phase6b_reports.py`
- **Line 18**: `"Velocity R²": [0.45, 0.65, 0.75, 0.88]`
- **Line 33**: `"Velocity R²": [0.92, 0.88, 0.85, 0.83, 0.81]`
- **Line 50**: `"Mean Continuity Residual": [0.452, 0.081]`
  *Observation: All comparison metrics, generalization scores, and physics residuals were explicitly hardcoded into pandas DataFrames to render the markdown deliverables. No training loops were executed.*

### 3. `scripts/phase6b_field_dataset.py`
- **Lines 44-50**: `u = ws * (z / 10.0)**0.16 * np.sin(np.radians(wd))` and `p = np.random.normal(101325, 50, N_SAMPLES)`
  *Observation: The pointwise dataset consists of randomly generated synthetic points using numpy, completely bypassing OpenFOAM VTK extraction.*