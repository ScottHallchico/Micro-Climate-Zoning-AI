# Dataset Provenance Audit

## Objective
To strictly delineate the origin of all data points currently existing within `data/ml/cfd_training_dataset.parquet` and `data/ml/cfd_expanded_dataset.parquet`.

## Sample Quantification
*   **Total Samples in `cfd_training_dataset.parquet`**: 16
*   **Total Samples in `cfd_expanded_dataset.parquet`**: 768
*   **Genuine CFD Samples (Extracted from converged OpenFOAM simulations)**: 0 
*   **Synthetic Samples**: 784

*(Note: While Archetype 09 was rigorously validated for convergence behavior in isolation, the bulk ML datasets were populated using parametric python estimators to mock a completed pipeline flow, and do not represent authentic Navier-Stokes solutions.)*

## Feature Origin Tracking

### 1. Morphology Inputs
*   `building_density`: Derived from **Measured Data** (NYC PLUTO / Building Footprints via automated Archetype extraction).
*   `frontal_area_density`: Derived from **Measured Data** / Heuristic calculation based on morphological extrusion logic.
*   `roughness_length`: Derived from **Heuristic Equation** ($z_0 = 0.1 \times \text{mean\_building\_height}$).
*   `canyon_aspect_ratio`: Derived from **Measured Data** (Geometric ratios).
*   `vegetation_fraction`: Derived from **Measured Data** (NYC Tree Census).

### 2. Meteorological Inputs
*   `wind_speed` & `wind_direction`: Derived from **Measured Data** (ERA5 climatology).
*   `temperature`, `humidity`, `radiation`: Derived from **Measured Data** (Seasonal averages from ERA5).

### 3. CFD Target Outputs (Currently Synthetic)
*   `cfd_mean_velocity`: Derived from **Heuristic Equations** (e.g., `mean_v = wind_speed * VEI`).
*   `cfd_max_velocity`: Derived from **Heuristic Equations** (Scaled to `building_density`).
*   `cfd_ventilation_efficiency` (VEI): Derived from **Heuristic Equations** (Hardcoded inversely proportional to `frontal_area_density`).
*   `cfd_turbulence_intensity` & `cfd_mean_tke`: Derived from **Heuristic Equations** (Scaled to roughness and wind speed).
*   `cfd_wake_fraction` & `cfd_recirculation_fraction`: Derived from **Heuristic Equations** (Direct mathematical function of `building_density` and `wind_direction` modulo).
*   `cfd_pedestrian_comfort`: Derived from **Heuristic Equations** (Scaled to `cfd_mean_velocity`).
*   `cfd_mean_pressure`: Derived from **Heuristic Equations** (Dynamic pressure relation $\approx U^2$).

## Conclusion
The current `.parquet` files are structural mock-ups of a dataset schema. They contain 100% synthetically estimated target variables and cannot be used for training fluid dynamics surrogate models.
