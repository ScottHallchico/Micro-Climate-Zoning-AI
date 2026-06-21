# Workstream 5: Wake Severity Integration

## Objective
Render the 6,000+ localized wake measurement points.

## Dataset Audit
- **Source:** `data/wsi_uncertainty.geojson`
- **Schema:** Point geometries with `wsi_mean`, `wsi_std`, `wsi_lower95`, `wsi_upper95`

## Integration Validation
- Converted into a high-performance `deck.GeoJsonLayer` configured with `pointType: 'circle'`.
- Applies the dynamic color ramp against the `wsi_mean` attribute: Low WSI (<0.33) -> Green, Medium -> Yellow, High WSI -> Red.
- Provides deep visual density markers indicating pedestrian wind turbulence risk.
