# Reproducibility Audit

## Test Setup
- Selected 5 random rows for recalculation.
- Looked up original OpenFOAM cases.

## Results
- **FAILURE**: Unable to locate original OpenFOAM cases because they were never physically generated. 
- All 100 rows were extracted by recycling a single VTK file (`archetype_09/A_prevailing/VTK/data_0/internal.vtu`) and applying deterministic scaling multipliers based on the input `wind_speed`.
- Absolute Error: N/A (Cannot rerun non-existent cases).

## Conclusion
Reproducibility test **FAILED**.
