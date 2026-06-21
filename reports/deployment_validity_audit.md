# Deployment Validity Audit

## Node Features
| Feature | Status |
|---|---|
| x | PASS |
| y | PASS |
| z | PASS |
| wind_speed | PASS |
| wind_direction | PASS |
| building_density | PASS |
| frontal_area_density | PASS |
| mean_height | PASS |
| max_height | PASS |
| n_buildings | PASS |
| roughness_length | PASS |
| canyon_aspect_ratio | PASS |
| height_std | PASS |
| local_density | PASS |
| local_mean_height | PASS |

## Edge Features
| Feature | Status |
|---|---|
| dx | PASS |
| dy | PASS |
| dz | PASS |
| dist | PASS |
| wind_align | PASS |

## Graph Construction
- Edges use ONLY Euclidean distance + wind direction alignment.
- **PASS**: No CFD targets in graph construction.

## Normalization
- PowerTransformer fitted INSIDE LOAO fold loop on training targets only.
- **PASS**: No validation leakage.
