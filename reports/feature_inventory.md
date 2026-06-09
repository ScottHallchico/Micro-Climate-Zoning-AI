# Feature Inventory

Generated: 2026-06-09 14:46:31

| # | Feature | Type | Non-Null | Mean | Min | Max | Source |
|---|---------|------|----------|------|-----|-----|--------|
| 1 | building_id | str | 1,082,945 | 1082921 unique | — | — | Building Footprints |
| 2 | geometry | MultiPolygon | 1,082,945 | — | — | — | Building Footprints |
| 3 | height_roof | float64 | 1,082,945 | 27.7893 | 0.0000 | 2026.0000 | Building Footprints |
| 4 | ground_elevation | float64 | 1,082,382 | 55.1310 | -16.0000 | 407.0000 | Building Footprints |
| 5 | construction_year | float64 | 1,072,836 | 1939.8685 | 1652.0000 | 2026.0000 | Building Footprints |
| 6 | zoning | str | 920,139 | 193 unique | — | — | MapPLUTO |
| 7 | land_use | float64 | 919,968 | 1.6688 | 1.0000 | 11.0000 | MapPLUTO |
| 8 | far | float64 | 920,499 | 0.9547 | 0.0000 | 71.6700 | MapPLUTO |
| 9 | lot_area | float64 | 920,802 | 128969.6203 | 0.0000 | 214378390.0000 | MapPLUTO |
| 10 | nearby_tree_count | int32 | 1,082,945 | 41.0908 | 0.0000 | 169.0000 | Tree Census (spatial) |
| 11 | canopy_density | float32 | 1,082,945 | 0.0028 | 0.0000 | 0.0451 | Tree Census (derived) |
| 12 | ndvi | float32 | 1,082,944 | 0.2541 | -0.5175 | 0.9135 | Sentinel-2 |
| 13 | surface_temperature | float32 | 1,082,945 | 39.3042 | -273.1500 | 56.0685 | Landsat 8 thermal |
| 14 | temperature | float32 | 1,082,945 | 13.5743 | 13.2868 | 13.8337 | ERA5 |
| 15 | humidity | float32 | 1,082,945 | 68.0688 | 65.4585 | 69.9601 | ERA5 (derived) |
| 16 | wind_speed | float32 | 1,082,945 | 0.7557 | 0.7273 | 0.8831 | ERA5 (derived) |
| 17 | wind_direction | float32 | 1,082,945 | 276.1555 | 261.4691 | 284.9021 | ERA5 (derived) |
| 18 | solar_radiation | float32 | 1,082,945 | 171.8106 | 169.5071 | 175.7906 | ERA5 |