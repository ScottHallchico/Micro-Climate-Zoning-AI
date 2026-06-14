# Urban Archetype Report

Generated: 2026-06-14 01:01:46

## Methodology
- **Algorithm**: MiniBatchKMeans (scalable for 1M+ buildings)
- **Feature set**: Morphology-only (10 features) — climate proxies excluded
- **Optimal K**: **11**
- **Selection strategy**: Composite rank (Silhouette, Davies-Bouldin, Calinski-Harabasz)

## Silhouette Metric Evaluation
- **Sweep Search (10k sample)**: Max 0.1950
- **Final Validation (100k sample)**: 0.1576

### Features Used for Clustering
- `building_density_100m`
- `mean_neighbor_height`
- `height_variance`
- `pad_100m`
- `roughness_length_proxy`
- `canyon_aspect_ratio`
- `multi_svf`
- `green_coverage_score`
- `tree_cooling_index`
- `fad_100m`

### Features Excluded (climate proxies, reported only)
- `ventilation_efficiency_index`
- `thermal_trapping_index`

## Cluster Validation Metrics (Sweep)

| K | Silhouette (10k) ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | Composite Rank |
|---|-------------------|-----------------|--------------------:|---------------:|
| 8 | 0.1950 | 1.4826 | 174794 | 4.67 |
| 9 | 0.1491 | 1.4685 | 173178 | 6.00 |
| 10 | 0.1254 | 1.5344 | 172437 | 11.67 |
| 11 | 0.1576 | 1.3951 | 183130 | 1.67 **←** |
| 12 | 0.1569 | 1.4298 | 177029 | 3.33 |
| 13 | 0.1504 | 1.5305 | 164998 | 8.33 |
| 14 | 0.1615 | 1.4574 | 160532 | 5.00 |
| 15 | 0.1445 | 1.4238 | 159579 | 6.33 |
| 16 | 0.1386 | 1.5030 | 152271 | 10.33 |
| 17 | 0.1384 | 1.4464 | 150178 | 8.67 |
| 18 | 0.1380 | 1.4972 | 135731 | 12.00 |
| 19 | 0.1457 | 1.4707 | 137442 | 9.00 |
| 20 | 0.1415 | 1.4059 | 134949 | 8.33 |
| 21 | 0.1269 | 1.5043 | 130344 | 14.00 |
| 22 | 0.1372 | 1.4637 | 137095 | 10.67 |

## Archetype Profiles

### Archetype 0 (132,507 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 95.8526 | 27.1376 |
| `mean_neighbor_height` | 22.6859 | 3.4413 |
| `height_variance` | 8.2023 | 2.1051 |
| `pad_100m` | 0.2507 | 0.0514 |
| `roughness_length_proxy` | 7.3073 | 2.5457 |
| `canyon_aspect_ratio` | 1.2912 | 0.4360 |
| `multi_svf` | 0.6354 | 0.0899 |
| `green_coverage_score` | 0.3846 | 0.0643 |
| `tree_cooling_index` | 0.0495 | 0.0357 |
| `fad_100m` | 0.6363 | 0.1557 |
| `ventilation_efficiency_index` | 6.9166 | 8.4488 |
| `thermal_trapping_index` | 10.3561 | 4.1389 |

### Archetype 1 (243,158 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 77.4290 | 23.7918 |
| `mean_neighbor_height` | 24.1961 | 3.7890 |
| `height_variance` | 7.2824 | 2.8880 |
| `pad_100m` | 0.2348 | 0.0463 |
| `roughness_length_proxy` | 6.9974 | 2.3612 |
| `canyon_aspect_ratio` | 2.9462 | 0.9685 |
| `multi_svf` | 0.9523 | 0.0529 |
| `green_coverage_score` | 0.3251 | 0.0469 |
| `tree_cooling_index` | 0.0334 | 0.0277 |
| `fad_100m` | 0.5702 | 0.1381 |
| `ventilation_efficiency_index` | 12.6262 | 14.9245 |
| `thermal_trapping_index` | 1.3830 | 1.5776 |

### Archetype 2 (76,818 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 98.5563 | 31.1021 |
| `mean_neighbor_height` | 23.0130 | 4.2366 |
| `height_variance` | 7.6779 | 2.5909 |
| `pad_100m` | 0.2526 | 0.0543 |
| `roughness_length_proxy` | 7.7698 | 3.5594 |
| `canyon_aspect_ratio` | 2.7144 | 1.2224 |
| `multi_svf` | 0.9070 | 0.1218 |
| `green_coverage_score` | 0.4710 | 0.0791 |
| `tree_cooling_index` | 0.1829 | 0.0817 |
| `fad_100m` | 0.6581 | 0.1848 |
| `ventilation_efficiency_index` | 8.7157 | 10.6913 |
| `thermal_trapping_index` | 2.2335 | 3.0970 |

### Archetype 3 (20,304 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 47.1221 | 21.5615 |
| `mean_neighbor_height` | 82.2953 | 23.8834 |
| `height_variance` | 53.6931 | 28.0408 |
| `pad_100m` | 0.4731 | 0.1022 |
| `roughness_length_proxy` | 89.9856 | 36.2436 |
| `canyon_aspect_ratio` | 9.9811 | 7.9596 |
| `multi_svf` | 0.7758 | 0.1820 |
| `green_coverage_score` | 0.2943 | 0.0590 |
| `tree_cooling_index` | 0.0441 | 0.0389 |
| `fad_100m` | 2.1942 | 0.5779 |
| `ventilation_efficiency_index` | 0.1046 | 1.5028 |
| `thermal_trapping_index` | 13.8225 | 12.4716 |

### Archetype 4 (69,946 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 76.9898 | 32.3122 |
| `mean_neighbor_height` | 30.2251 | 8.0151 |
| `height_variance` | 12.7582 | 5.6952 |
| `pad_100m` | 0.3823 | 0.0717 |
| `roughness_length_proxy` | 13.5868 | 6.6295 |
| `canyon_aspect_ratio` | 2.0747 | 1.0684 |
| `multi_svf` | 0.6534 | 0.1371 |
| `green_coverage_score` | 0.2851 | 0.0422 |
| `tree_cooling_index` | 0.0266 | 0.0227 |
| `fad_100m` | 0.8722 | 0.2321 |
| `ventilation_efficiency_index` | 4.2781 | 8.2301 |
| `thermal_trapping_index` | 16.9368 | 7.3201 |

### Archetype 5 (141,092 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 97.5247 | 22.0495 |
| `mean_neighbor_height` | 26.6343 | 4.0352 |
| `height_variance` | 9.2499 | 2.7407 |
| `pad_100m` | 0.3650 | 0.0542 |
| `roughness_length_proxy` | 11.9718 | 3.4751 |
| `canyon_aspect_ratio` | 3.6866 | 1.3307 |
| `multi_svf` | 0.9507 | 0.0474 |
| `green_coverage_score` | 0.2877 | 0.0381 |
| `tree_cooling_index` | 0.0296 | 0.0220 |
| `fad_100m` | 0.8859 | 0.1532 |
| `ventilation_efficiency_index` | 8.0352 | 12.4413 |
| `thermal_trapping_index` | 2.3272 | 2.2857 |

### Archetype 6 (130,792 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 147.6308 | 24.9377 |
| `mean_neighbor_height` | 21.7070 | 2.5873 |
| `height_variance` | 7.8625 | 1.6260 |
| `pad_100m` | 0.3235 | 0.0498 |
| `roughness_length_proxy` | 9.5607 | 2.2058 |
| `canyon_aspect_ratio` | 2.6253 | 1.1061 |
| `multi_svf` | 0.9079 | 0.1132 |
| `green_coverage_score` | 0.3451 | 0.0555 |
| `tree_cooling_index` | 0.0577 | 0.0357 |
| `fad_100m` | 0.8740 | 0.1258 |
| `ventilation_efficiency_index` | 8.7741 | 10.9656 |
| `thermal_trapping_index` | 3.7191 | 4.8385 |

### Archetype 7 (2,970 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 22.0175 | 11.9840 |
| `mean_neighbor_height` | 195.4459 | 76.2162 |
| `height_variance` | 154.3823 | 58.0387 |
| `pad_100m` | 0.5199 | 0.1375 |
| `roughness_length_proxy` | 371.6933 | 208.0273 |
| `canyon_aspect_ratio` | 26.3222 | 27.8993 |
| `multi_svf` | 0.6720 | 0.2379 |
| `green_coverage_score` | 0.2498 | 0.0414 |
| `tree_cooling_index` | 0.0082 | 0.0171 |
| `fad_100m` | 3.7809 | 1.1588 |
| `ventilation_efficiency_index` | 0.0000 | 0.0000 |
| `thermal_trapping_index` | 23.8775 | 19.3953 |

### Archetype 8 (34,530 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 33.9437 | 18.5454 |
| `mean_neighbor_height` | 48.2244 | 16.5369 |
| `height_variance` | 22.2065 | 12.0668 |
| `pad_100m` | 0.3231 | 0.0995 |
| `roughness_length_proxy` | 20.3519 | 10.7448 |
| `canyon_aspect_ratio` | 8.2463 | 5.0407 |
| `multi_svf` | 0.9244 | 0.0879 |
| `green_coverage_score` | 0.2881 | 0.0543 |
| `tree_cooling_index` | 0.0284 | 0.0264 |
| `fad_100m` | 0.8304 | 0.3115 |
| `ventilation_efficiency_index` | 2.6949 | 9.3315 |
| `thermal_trapping_index` | 3.1233 | 3.6916 |

### Archetype 9 (164,446 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 56.3930 | 24.0033 |
| `mean_neighbor_height` | 23.9315 | 4.7654 |
| `height_variance` | 7.0037 | 3.4044 |
| `pad_100m` | 0.1807 | 0.0558 |
| `roughness_length_proxy` | 5.1732 | 2.2737 |
| `canyon_aspect_ratio` | 2.7589 | 1.0050 |
| `multi_svf` | 0.9396 | 0.0724 |
| `green_coverage_score` | 0.4377 | 0.0632 |
| `tree_cooling_index` | 0.0442 | 0.0342 |
| `fad_100m` | 0.4245 | 0.1507 |
| `ventilation_efficiency_index` | 15.0742 | 15.3641 |
| `thermal_trapping_index` | 1.0543 | 1.2538 |

### Archetype 10 (66,382 buildings)
| Feature | Mean | Std |
|---------|------|-----|
| `building_density_100m` | 91.9389 | 22.2964 |
| `mean_neighbor_height` | 39.5170 | 8.3722 |
| `height_variance` | 11.2630 | 5.1086 |
| `pad_100m` | 0.4032 | 0.0569 |
| `roughness_length_proxy` | 26.9823 | 10.9679 |
| `canyon_aspect_ratio` | 5.1974 | 1.9558 |
| `multi_svf` | 0.9099 | 0.0876 |
| `green_coverage_score` | 0.3557 | 0.0667 |
| `tree_cooling_index` | 0.0721 | 0.0417 |
| `fad_100m` | 1.3236 | 0.2725 |
| `ventilation_efficiency_index` | 0.7742 | 4.4294 |
| `thermal_trapping_index` | 4.3193 | 4.5030 |
