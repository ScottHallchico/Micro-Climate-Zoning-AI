# Surrogate Model Validation V2\n\n| Target                   | Model        |   5-Fold R² |     LOAO R² |    LTAO R² |
|:-------------------------|:-------------|------------:|------------:|-----------:|
| cfd_mean_velocity        | RandomForest |    0.791431 |   0.60167   |   0.616543 |
| cfd_mean_velocity        | XGBoost      |    0.802785 |   0.262722  |   0.357396 |
| cfd_mean_velocity        | LightGBM     |    0.713816 |   0.393638  |   0.346165 |
| cfd_max_velocity         | RandomForest |    0.625227 |   0.606149  |   0.631085 |
| cfd_max_velocity         | XGBoost      |    0.313417 |   0.223881  |   0.153488 |
| cfd_max_velocity         | LightGBM     |    0.65105  |   0.629845  |   0.607466 |
| cfd_mean_pressure        | RandomForest |    0.535781 |  -0.0179185 |   0.337314 |
| cfd_mean_pressure        | XGBoost      |   -0.253006 |  -1.1694    |  -0.6798   |
| cfd_mean_pressure        | LightGBM     |    0.439574 |  -0.088637  |   0.251432 |
| cfd_mean_tke             | RandomForest |    0.629664 |   0.834152  |   0.779997 |
| cfd_mean_tke             | XGBoost      |    0.642937 |   0.757357  |   0.272863 |
| cfd_mean_tke             | LightGBM     |    0.312761 |  -0.599681  |  -0.898849 |
| cfd_turbulence_intensity | RandomForest |    0.892734 |  -0.461495  |   0.838661 |
| cfd_turbulence_intensity | XGBoost      |    0.899701 |  -0.935193  |   0.77119  |
| cfd_turbulence_intensity | LightGBM     |    0.897488 |  -0.858905  |   0.76348  |
| cfd_wake_fraction        | RandomForest |    0.949896 |  -8.5646    |  -1.09552  |
| cfd_wake_fraction        | XGBoost      |    0.946728 |  -8.75845   |  -3.65096  |
| cfd_wake_fraction        | LightGBM     |    0.933418 | -22.0975    | -11.1166   |