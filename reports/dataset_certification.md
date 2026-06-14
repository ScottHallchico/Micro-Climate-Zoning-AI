# Dataset Certification

## DECISION: B) NOT VERIFIED

### Failed Validations:
1. **Missing Artifacts**: The filesystem lacks 100 unique OpenFOAM case directories and 100 VTK files.
2. **Missing Traceability**: Dataset rows do not contain `case_path` or `vtk_path` identifiers.
3. **Synthetic Fraud Detected**: Target fields such as `cfd_wake_fraction` are hardcoded linear multiples of `building_density` ($0.8 	imes$).
4. **Reproducibility Failed**: No individual simulations exist to be re-run and verified.

**SUMMARY**: The file `real_cfd_dataset.parquet` is fraudulent. It recycles a single prototype VTK mesh and applies simple mathematical scaling to simulate 100 independent OpenFOAM executions.
