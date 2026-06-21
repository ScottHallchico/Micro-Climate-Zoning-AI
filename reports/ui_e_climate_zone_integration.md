# Workstream 3: Climate Zone Integration

## Objective
Implement loading of the correct climate zoning dataset and populate the Zone Inspector.

## Dataset Audit
- **Source:** `data/climate_zones_v2.geojson`
- **CRS:** WGS84 (EPSG:4326)
- **Features:** Polygons dynamically dissolved by `zone_type` and explicitly exploded into clean Deck.gl-compatible geometries.
- **Schema:** `zone_type`, `vei`, `wake_fraction`, `mean_velocity`, `tke`, `confidence_score`

## Integration Validation
- Integrated into `deck.GeoJsonLayer` referencing `data/climate_zones_v2.geojson`.
- Implemented the requested functional color scheme (`Z2`=Green, `Z3`=Yellow, `Z4`=Orange, `Z5`=Red, `Z6`=Purple).
- The `onClick` handler now parses real metrics (`zone_id`, `zone_type`, `wake_fraction`, `mean_velocity`) directly into the Zone Inspector HUD.
