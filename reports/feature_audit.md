# Feature Audit

## Permutation Importance (Velocity u)
| Feature | Importance |
|---|---|
| wind_direction | 0.4969 |
| wind_speed | 0.2140 |
| y | 0.1802 |
| local_density_50m | 0.0994 |
| z | 0.0901 |
| dist_to_tallest | 0.0020 |
| building_density | 0.0000 |
| frontal_area_density | 0.0000 |
| mean_height | 0.0000 |
| max_height | 0.0000 |
| n_buildings | 0.0000 |
| roughness_length | 0.0000 |
| canyon_aspect_ratio | 0.0000 |
| height_std | 0.0000 |
| local_h_var_50m | -0.0015 |
| local_h_mean_50m | -0.0030 |
| x | -0.0167 |

**Coordinate Dominance**: 0.2537 (Sum of x,y,z importance)
If coordinate dominance is > 0.5, the model may be overfitting spatially.
