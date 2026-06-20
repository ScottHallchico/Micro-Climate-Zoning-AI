# Spatial Zone Generation

- **Zone polygons exported**: 10
- **Output file**: `data/zones/climate_zones.geojson`
- **CRS**: WGS84 (EPSG:4326) for web display
- **Dissolved from**: 60000 classified CFD points

Each polygon contains:
- `zone_id`, `zone_type`, `zone_name`
- `vei`, `wake_fraction`, `mean_velocity`, `tke`
- `confidence_score` (inverse TKE percentile)
