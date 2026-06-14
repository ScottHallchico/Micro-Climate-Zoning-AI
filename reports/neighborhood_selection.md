# Representative Neighborhood Selection Report

Generated: 2026-06-14 01:01:46

## Neighborhood Size Sensitivity (250m vs 500m)

| Archetype | Rep. Building | Centroid Dist | 250m Bldgs | 500m Bldgs | 500m Mean H | 500m PAD | Complexity |
|-----------|---------------|---------------|------------|------------|-------------|----------|------------|
| 0 | `4579696` | 0.167 | 169 | 623 | 22.3 | 0.23 | Medium |
| 1 | `5106159` | 0.153 | 146 | 593 | 24.9 | 0.23 | Medium |
| 2 | `4542579` | 0.249 | 180 | 606 | 23.6 | 0.27 | High |
| 3 | `1057177` | 0.766 | 98 | 237 | 77.2 | 0.41 | High |
| 4 | `2019150` | 0.304 | 174 | 692 | 26.0 | 0.32 | High |
| 5 | `3147526` | 0.185 | 216 | 702 | 26.6 | 0.34 | High |
| 6 | `2060982` | 0.175 | 291 | 851 | 23.0 | 0.29 | High |
| 7 | `1041900` | 1.927 | 60 | 242 | 129.3 | 0.49 | High |
| 8 | `3116137` | 0.548 | 69 | 258 | 42.9 | 0.42 | High |
| 9 | `2075472` | 0.181 | 95 | 357 | 25.0 | 0.19 | Medium |
| 10 | `3055772` | 0.260 | 160 | 467 | 42.1 | 0.36 | High |

## Recommendation
For initial CFD experiments, 250m patches are recommended for High-complexity archetypes
to keep mesh cell counts feasible. 500m patches are suitable for Low/Medium complexity.