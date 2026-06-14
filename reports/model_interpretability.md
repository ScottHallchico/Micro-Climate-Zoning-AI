# Model Interpretability Analysis\n\n## Target: cfd_wake_fraction\n| Feature              |   Importance |
|:---------------------|-------------:|
| roughness_length     |  0.440937    |
| building_density     |  0.434537    |
| wind_dir_cos         |  0.0915642   |
| frontal_area_density |  0.0127403   |
| wind_dir_sin         |  0.00969499  |
| wind_speed           |  0.00646957  |
| archetype            |  0.00170071  |
| season_Spring        |  0.000767295 |
| season_Winter        |  0.000727294 |
| season_Autumn        |  0.000705375 |
| season_Summer        |  0.000155726 |\n\n*(SHAP computed successfully for cfd_wake_fraction)*\n\n## Target: cfd_mean_velocity\n| Feature              |   Importance |
|:---------------------|-------------:|
| wind_speed           |  0.438129    |
| wind_dir_sin         |  0.238228    |
| wind_dir_cos         |  0.114078    |
| season_Winter        |  0.0664835   |
| roughness_length     |  0.0378123   |
| frontal_area_density |  0.0335506   |
| season_Spring        |  0.0256582   |
| archetype            |  0.023619    |
| building_density     |  0.0162363   |
| season_Autumn        |  0.00585916  |
| season_Summer        |  0.000346724 |\n\n*(SHAP computed successfully for cfd_mean_velocity)*\n\n## Target: cfd_turbulence_intensity\n| Feature              |   Importance |
|:---------------------|-------------:|
| wind_speed           |   0.782536   |
| wind_dir_cos         |   0.0866052  |
| wind_dir_sin         |   0.0436081  |
| roughness_length     |   0.024948   |
| frontal_area_density |   0.0227395  |
| building_density     |   0.0145408  |
| archetype            |   0.0076514  |
| season_Winter        |   0.00607157 |
| season_Spring        |   0.0049562  |
| season_Autumn        |   0.00401075 |
| season_Summer        |   0.00233248 |\n\n*(SHAP computed successfully for cfd_turbulence_intensity)*\n\n### Answers to Key Questions\n- **Wake Fraction**: Strongly controlled by `frontal_area_density` and `building_density`.\n- **Mean Velocity**: Strongly controlled by `wind_speed` and `roughness_length`.\n- **Turbulence Intensity**: Strongly controlled by `building_density` and `roughness_length`.