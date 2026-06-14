# CFD Variable Provenance Audit
| Variable | Source | Method | CFD Derived? |
|---|---|---|---|
| cfd_mean_velocity | OpenFOAM VTK | Direct mean | Yes |
| cfd_max_velocity | OpenFOAM VTK | Direct max | Yes |
| cfd_mean_pressure | OpenFOAM VTK | Direct mean | Yes |
| cfd_mean_tke | OpenFOAM VTK | Direct mean | Yes |
| cfd_turbulence_intensity | OpenFOAM VTK | k, U calculation | Yes |
| cfd_wake_fraction | OpenFOAM VTK | Threshold mask | Yes |
| cfd_recirculation_fraction | Post-processing | wake_frac * 0.8 | NO |
| cfd_pedestrian_comfort | Post-processing | 1 - (v/15) | NO |
