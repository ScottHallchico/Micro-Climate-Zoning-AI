# Actual Data Sources

Project city: Mumbai, India

Project coordinate used: `19.0760 N, 72.8777 E`

## Weather and Surface-Adjacent Temperature

Source: Open-Meteo forecast API

Nearest returned model point:

- Latitude: `19.086115`
- Longitude: `72.85291`
- Elevation: `6 m`
- Time zone: `Asia/Kolkata`

Saved file:

- `data/actual_weather_measurements.csv`

Fields collected:

- air temperature at 2 m
- relative humidity at 2 m
- wind speed at 10 m
- wind direction at 10 m
- shortwave solar radiation
- surface pressure
- soil temperature at 0 cm

Important note: `soil_temperature_0cm_c` is real public data, but it is not the same as satellite land-surface temperature. Use it as a temporary surface-temperature proxy until Landsat or ECOSTRESS thermal data is added.

## Thermal Data Still Needed

For true thermal mapping, use one of these:

- Landsat 8/9 Collection 2 Level 2 land surface temperature
- ECOSTRESS land surface temperature
- Local drone thermal camera survey
- Field sensor readings per block

The final thermal file should look like:

```csv
block_id,surface_temp_c,air_temp_c,uhi_intensity_c,heat_storage_wm2,thermal_risk,source,timestamp
BLK-01,38.2,33.1,3.4,210,high,landsat,2026-05-13T10:30:00+05:30
```

## OpenStreetMap Urban Structure

Source: official OpenStreetMap API map extract for bounding box around the project coordinate.

Bounding box:

- south: `12.9072`
- south: `19.0710`
- west: `72.8727`
- north: `19.0810`
- east: `72.8827`

Saved file:

- `data/actual_blocks.csv`

Fields collected or derived:

- `osm_building_count`: actual OSM building way count per dashboard block cell
- `osm_road_count`: actual OSM road way count per dashboard block cell
- `osm_green_feature_count`: actual OSM green/natural/park feature count per dashboard block cell
- `measured_avg_building_levels`: actual OSM `building:levels` when present
- `max_height_m`: actual levels converted to meters where levels exist, otherwise an OSM-density estimate
- `svf`, `lambda_p`, `hw_ratio`, `green_cover_pct`: morphology estimates derived from OSM feature density

Important note: OSM usually does not contain complete measured building heights. When `measured_avg_building_levels` is blank, height is estimated from OSM building density and should be replaced by municipal GIS, LiDAR, drone survey, or measured field data when available.
