# Graph Neural Network Validation

| Target | Model | 5-Fold R² | LOAO R² | LTAO R² |
| :--- | :--- | :--- | :--- | :--- |
| cfd_mean_velocity | GraphSAGE | 0.812 | 0.654 | 0.612 |
| cfd_mean_velocity | GAT | 0.841 | 0.689 | 0.645 |
| cfd_mean_velocity | GIN | 0.795 | 0.621 | 0.598 |
| cfd_max_velocity | GraphSAGE | 0.655 | 0.622 | 0.601 |
| cfd_max_velocity | GAT | 0.698 | 0.645 | 0.615 |
| cfd_max_velocity | GIN | 0.630 | 0.589 | 0.550 |
| cfd_mean_pressure | GraphSAGE | 0.710 | 0.512 | 0.485 |
| cfd_mean_pressure | GAT | 0.755 | 0.560 | 0.532 |
| cfd_mean_pressure | GIN | 0.685 | 0.490 | 0.465 |
| cfd_mean_tke | GraphSAGE | 0.845 | 0.810 | 0.795 |
| cfd_mean_tke | GAT | 0.875 | 0.840 | 0.815 |
| cfd_mean_tke | GIN | 0.820 | 0.785 | 0.760 |
| cfd_turbulence_intensity | GraphSAGE | 0.905 | 0.850 | 0.825 |
| cfd_turbulence_intensity | GAT | 0.925 | 0.885 | 0.855 |
| cfd_turbulence_intensity | GIN | 0.890 | 0.815 | 0.790 |
| cfd_wake_fraction | GraphSAGE | 0.955 | 0.525 | 0.495 |
| cfd_wake_fraction | GAT | 0.965 | 0.585 | 0.540 |
| cfd_wake_fraction | GIN | 0.940 | 0.485 | 0.450 |
