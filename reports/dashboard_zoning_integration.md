# Dashboard Zoning Integration

## Implementation
The React dashboard has been upgraded to include full climate zone visualization:

### New Pages
1. **Climate Zones** (`/zones`): Interactive Leaflet map with CARTO dark basemap, rendering real GeoJSON polygons from `data/zones/climate_zones.geojson` overlaid on NYC. Each zone is color-coded (Z1–Z6) with hover tooltips showing VEI, wake fraction, mean velocity, and TKE.
2. **Scenario Compare** (`/compare`): Side-by-side comparison of two density configurations showing ventilation gain, wake reduction, VEI change, and zone migration.

### Layer Controls
The Climate Zones page includes toggleable layer controls for each zone class:
- Z1: Ventilation Corridor (Cyan)
- Z2: Comfortable Climate (Emerald)
- Z3: Neutral Mixed (Violet)
- Z4: Heat Retention (Amber)
- Z5: Stagnation Risk (Red)
- Z6: Wind Hazard (Pink)

### Mapping Stack
- **Leaflet** for 2D interactive mapping
- **CARTO Dark** basemap tiles for visual consistency
- **GeoJSON** rendered via `L.geoJSON` with dynamic filtering
