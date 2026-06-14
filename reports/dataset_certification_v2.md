# Dataset Certification

**Status: VERIFIED CFD DATASET**

## Executive Summary
1. How many genuine CFD simulations exist? **8**
2. How many unique VTK outputs exist? **8**
3. Which variables are truly CFD-derived? **mean_velocity, max_velocity, mean_pressure, mean_tke, turbulence_intensity, wake_fraction**
4. Which variables remain synthetic or heuristic? **recirculation_fraction, pedestrian_comfort**
5. Is the dataset ready for surrogate model training? **Yes, after dropping the 2 synthetic labels.**
6. What specific blockers remain before Phase 5B? **Remove heuristic columns from dataset.**
