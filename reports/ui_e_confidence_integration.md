# Workstream 6: Confidence Layer Integration

## Objective
Generate dynamic, mathematically sound confidence uncertainty visualizations without using dummy polygons.

## Implementation Details
- Instead of using fake zones, the application now directly parses the statistical uncertainty spread from the `wsi_uncertainty.geojson` model outputs.
- Computes `confidence = 1 - (wsi_std / (wsi_upper95 - wsi_lower95))` natively inside the MapLibre fragment shader loop.
- The layer accurately paints standard deviation risk: High Confidence points glow Green, while highly uncertain predictions glow Red indicating Out-of-Distribution (OOD) risks.
- The Zone Inspector computes and reveals the precise percentage.
