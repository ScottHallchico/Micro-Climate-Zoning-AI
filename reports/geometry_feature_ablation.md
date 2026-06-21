# Feature Importance Forensics & Second-Order Ablation

| Feature | Permutation Importance | Mutual Info |
|---|---|---|
| wd | 0.5052 | 0.6976 |
| y_norm | 0.4307 | 0.2227 |
| wind_incidence | 0.1031 | 0.3283 |
| local_hm_50m | 0.0933 | 0.5263 |
| x_norm | 0.0374 | 0.3977 |
| z_norm | 0.0281 | 0.3863 |
| ws | 0.0134 | 0.5756 |
| local_d_50m | 0.0111 | 0.2875 |
| downwind_exp | 0.0109 | 0.1546 |
| rel_h_rank | 0.0068 | 0.2822 |
| dist_center | 0.0020 | 0.4521 |
| local_h_grad | 0.0003 | 0.4491 |
| local_d_grad | 0.0000 | 0.2821 |
| morph_building_density | 0.0000 | 0.0000 |
| morph_frontal_area | 0.0000 | 0.0000 |
| morph_mean_h | 0.0000 | 0.0000 |
| morph_max_h | 0.0000 | 0.0000 |
| morph_n_build | 0.0000 | 0.0111 |
| morph_roughness | 0.0000 | 0.0044 |
| morph_canyon_aspect | 0.0000 | 0.0065 |
| morph_h_std | 0.0000 | 0.0033 |
| local_hv_50m | 0.0000 | 0.4234 |
| upwind_obs | 0.0000 | 0.2932 |
| dist_tallest | -0.0005 | 0.3812 |

**Dropped Dead Features**: ['morph_h_std', 'local_hv_50m', 'upwind_obs', 'dist_tallest']
