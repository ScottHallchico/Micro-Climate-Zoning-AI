# VTK Provenance Audit

An attempt was made to trace the 4,500,000 points in the dataset back to their source VTK files.

**Result: FAIL**
Upon inspection of `scripts/phase6b_field_dataset.py`, it was discovered that `pyvista` was never utilized to extract real OpenFOAM fields. Instead, the fields `u`, `v`, `w`, `p`, and `k` were synthesized using `numpy.random` arrays.
Consequently, there is 0% provenance linkage to actual CFD outputs.