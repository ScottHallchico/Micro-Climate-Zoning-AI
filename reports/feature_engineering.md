# Feature Engineering Report

1. **Morphological Features Mapped**:
   Mapped `building_density`, `frontal_area_density` (pad), and `roughness_length` (derived from mean_height) using the archetype metadata.
   
2. **Wind Direction Encoding**:
   Applied cyclic sine and cosine transformations to `wind_direction`.

3. **Season Encoding**:
   Applied one-hot encoding to the categorical `season` variable.

Final Model Input Features:
archetype, building_density, frontal_area_density, roughness_length, wind_speed, wind_dir_sin, wind_dir_cos, season_Autumn, season_Spring, season_Summer, season_Winter
