# Machine Learning Dataset Audit

- Total Rows: 50
- Columns: 17
- Duplicate Rows: 0

## Null Counts
|                            |   0 |
|:---------------------------|----:|
| simulation_id              |   0 |
| archetype                  |   0 |
| wind_speed                 |   0 |
| wind_direction             |   0 |
| season                     |   0 |
| case_path                  |   0 |
| vtk_path                   |   0 |
| runtime_seconds            |   0 |
| final_residuals            |   0 |
| cfd_mean_velocity          |   0 |
| cfd_max_velocity           |   0 |
| cfd_ventilation_efficiency |   0 |
| cfd_turbulence_intensity   |   0 |
| cfd_mean_tke               |   0 |
| cfd_wake_fraction          |   0 |
| cfd_mean_pressure          |   0 |
| mesh_cells                 |   0 |

## Feature Distributions
|       |   wind_speed |   wind_direction |
|:------|-------------:|-----------------:|
| count |     50       |           50     |
| mean  |      6.88    |          146.7   |
| std   |      3.75059 |          112.016 |
| min   |      2       |            0     |
| 25%   |      4       |           45     |
| 50%   |      6       |          135     |
| 75%   |     10       |          225     |
| max   |     12       |          315     |

## Target Distributions
|       |   cfd_mean_velocity |   cfd_max_velocity |   cfd_mean_pressure |   cfd_mean_tke |   cfd_turbulence_intensity |   cfd_wake_fraction |
|:------|--------------------:|-------------------:|--------------------:|---------------:|---------------------------:|--------------------:|
| count |            50       |           50       |            50       |     50         |                50          |          50         |
| mean  |            10.2663  |          284.395   |         -8406.19    |      0.368605  |                 0.0499424  |           0.276206  |
| std   |            13.7137  |          796.98    |         35992.1     |      0.431491  |                 0.0345039  |           0.0252697 |
| min   |             1.01978 |            3.06103 |       -207734       |      0.0281025 |                 0.00564529 |           0.250705  |
| 25%   |             2.90658 |           17.2712  |           -28.0734  |      0.0499059 |                 0.0291157  |           0.254294  |
| 50%   |             6.55358 |           36.3481  |            -0.72815 |      0.169587  |                 0.0396908  |           0.268629  |
| 75%   |            10.4552  |          136.324   |           388.438   |      0.532619  |                 0.0563637  |           0.305309  |
| max   |            73.2494  |         4584.84    |          5094.27    |      1.46977   |                 0.137297   |           0.331312  |
