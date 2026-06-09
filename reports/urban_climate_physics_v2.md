# Urban Climate Physics & Morphology Report

Generated: 2026-06-09 16:35:02

This report details the advanced, physically-grounded indices used for CFD simulation geometries and PINN loss function weighting.

## Target Variable Dictionary & Statistics

### `pad_100m`
- **Definition**: Plan Area Density (100m)
- **Physical Interpretation**: Proxy for impervious surface and thermal mass.
- **Physics Formulation**: `sum(area) / (pi*100^2)`
- **Units**: Ratio (0-1)
- **Statistics**: 
  - Mean: 0.2854 ± 0.0951
  - Range: [0.0000, 1.0000]
  - Quartiles: (25%: 0.2178, 50%: 0.2746, 75%: 0.3483)

### `pad_250m`
- **Definition**: Plan Area Density (250m)
- **Physical Interpretation**: Neighborhood scale building density.
- **Physics Formulation**: `sum(area) / (pi*250^2)`
- **Units**: Ratio (0-1)
- **Statistics**: 
  - Mean: 0.2718 ± 0.0879
  - Range: [0.0000, 0.7912]
  - Quartiles: (25%: 0.2078, 50%: 0.2641, 75%: 0.3359)

### `fad_100m`
- **Definition**: Frontal Area Density (100m)
- **Physical Interpretation**: Aerodynamic obstruction facing wind vectors.
- **Physics Formulation**: `sum(equiv_width * height) / (pi*100^2)`
- **Units**: Ratio
- **Statistics**: 
  - Mean: 0.7535 ± 0.3931
  - Range: [0.0000, 7.8539]
  - Quartiles: (25%: 0.5232, 50%: 0.6957, 75%: 0.8930)

### `roughness_length_proxy`
- **Definition**: Aerodynamic Roughness Length (Z0)
- **Physical Interpretation**: Lettau (1969) adaptation for mechanical wind turbulence.
- **Physics Formulation**: `0.5 * mean_height * FAD`
- **Units**: Meters
- **Statistics**: 
  - Mean: 12.4033 ± 25.7928
  - Range: [0.0000, 2422.6218]
  - Quartiles: (25%: 5.9322, 50%: 8.2545, 75%: 11.4957)

### `multi_svf`
- **Definition**: Multi-Building Sky View Factor
- **Physical Interpretation**: True 3D angular sky obstruction replacing 2D street canyon cos(arctan).
- **Physics Formulation**: `1 - mean(sin(obstruction_angle_to_neighbors))`
- **Units**: Ratio (0-1)
- **Statistics**: 
  - Mean: 0.8759 ± 0.1469
  - Range: [0.0444, 1.0000]
  - Quartiles: (25%: 0.7985, 50%: 0.9439, 75%: 0.9857)

### `local_mean_elevation`
- **Definition**: Mean Ground Elevation
- **Physical Interpretation**: Determines baseline atmospheric pressures and localized cold pooling.
- **Physics Formulation**: `mean(elevation in 100m)`
- **Units**: Meters
- **Statistics**: 
  - Mean: 55.1353 ± 41.1878
  - Range: [-2.0000, 404.0000]
  - Quartiles: (25%: 25.2000, 50%: 45.7111, 75%: 73.5636)

### `local_elevation_variance`
- **Definition**: Ground Elevation Variance
- **Physical Interpretation**: Highlights complex, non-flat urban terrain.
- **Physics Formulation**: `std(elevation in 100m)`
- **Units**: Meters
- **Statistics**: 
  - Mean: 3.5141 ± 3.7367
  - Range: [0.0000, 160.5983]
  - Quartiles: (25%: 1.3924, 50%: 2.3743, 75%: 4.4455)

### `local_slope_proxy`
- **Definition**: Proxy for Local Slope
- **Physical Interpretation**: Indicates steep topological features dictating drainage flows.
- **Physics Formulation**: `max_elev_diff / 100`
- **Units**: Ratio
- **Statistics**: 
  - Mean: 0.1468 ± 0.1574
  - Range: [0.0000, 3.6100]
  - Quartiles: (25%: 0.0600, 50%: 0.1000, 75%: 0.1800)

### `urban_morphology_index`
- **Definition**: Urban Morphology Intensity
- **Physical Interpretation**: Combined structural obstruction and thermal capacity.
- **Physics Formulation**: `Norm(PAD) * Norm(Roughness)`
- **Units**: Index (0-100)
- **Statistics**: 
  - Mean: 0.2574 ± 0.8198
  - Range: [0.0000, 100.0000]
  - Quartiles: (25%: 0.0756, 50%: 0.1300, 75%: 0.2282)

### `ventilation_efficiency_index`
- **Definition**: Airflow Flushing Potential
- **Physical Interpretation**: How easily wind penetrates and removes heat/pollutants.
- **Physics Formulation**: `Norm(Wind) * SVF / (Norm(FAD) + 0.1)`
- **Units**: Index (0-100)
- **Statistics**: 
  - Mean: 9.1068 ± 12.7888
  - Range: [0.0000, 100.0000]
  - Quartiles: (25%: 0.0000, 50%: 4.9091, 75%: 7.6944)

### `thermal_trapping_index`
- **Definition**: Radiative Trapping Potential
- **Physical Interpretation**: Identifies zones with restricted sky cooling and high thermal mass.
- **Physics Formulation**: `(1-SVF) * Norm(PAD) * (1-Green)`
- **Units**: Index (0-100)
- **Statistics**: 
  - Mean: 4.4315 ± 6.1732
  - Range: [0.0000, 100.0000]
  - Quartiles: (25%: 0.3928, 50%: 1.7356, 75%: 6.4027)
