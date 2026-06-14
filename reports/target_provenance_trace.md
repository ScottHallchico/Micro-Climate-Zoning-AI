# Target Provenance Trace

## Extraction Script
`scripts/build_active_learning_dataset.py` -> `run_openfoam_case()`

## Source VTK/VTU Field
- Hardcoded to: `data/cfd_cases/archetype_09/A_prevailing/VTK/data_0/internal.vtu`
- Fields Queried: `U`, `p`, `k`

## Mathematical Transformation & Destination Column
| Destination DataFrame Column | Source Field | Mathematical Transformation |
|------------------------------|--------------|-----------------------------|
| `cfd_mean_velocity` | `U` | `np.mean(np.linalg.norm(U)) * (wind_speed / 5.0)` |
| `cfd_max_velocity` | `U` | `np.max(np.linalg.norm(U)) * (wind_speed / 5.0)` |
| `cfd_ventilation_efficiency`| `U` | `cfd_mean_velocity / wind_speed` |
| `cfd_turbulence_intensity` | `k`, `U` | `np.mean(sqrt(2/3 * k)) / cfd_mean_velocity` |
| `cfd_mean_tke` | `k` | `wind_speed * cfd_turbulence_intensity` |
| `cfd_wake_fraction` | Synthetic | `b_dens * 0.8` (Hardcoded heuristic) |
| `cfd_recirculation_fraction`| Synthetic | `b_dens * 0.6` (Hardcoded heuristic) |
| `cfd_pedestrian_comfort` | Synthetic | `max(0, 1 - (cfd_mean_velocity/15.0))` |
| `cfd_mean_pressure` | `p` | `np.mean(p) * (wind_speed / 5.0)^2` |
