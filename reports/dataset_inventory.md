# NYC Micro-Climate Zoning AI – Dataset Inventory

Generated: 2026-06-08 17:20:45

## Summary Table

| # | Dataset | Format | Status | Size | Rows | CRS |
|---|---------|--------|--------|------|------|-----|
| 1 | NYC Building Footprints | geojson | OK | 961.8 MB | 1,082,945 | EPSG:4326 |
| 2 | MapPLUTO – Shoreline Clipped (All Boroughs) | csv | OK | 268.1 MB | 721,973 | EPSG:4326 (inferred from lat/l… |
| 3 | 2015 Street Tree Census – Tree Data | csv | OK | 210.1 MB | 683,788 | EPSG:4326 (inferred from lat/l… |
| 4 | Landsat 8 Collection 2 Level 2 (Surface Reflectance) | geotiff | OK | 18.1 MB | — | EPSG:4326 |
| 5 | Sentinel-2 SR Harmonized | geotiff | OK | 16.6 MB | — | EPSG:4326 |
| 6 | ERA5 Hourly Reanalysis (2m Temperature, 10m Wind, Surface Radiation) | netcdf | OK | 1.4 MB | — | EPSG:4326 (ERA5 standard) |

## Detailed Reports

### NYC Building Footprints
- **Source:** NYC Open Data (pre-downloaded)
- **File:** `data/raw/BUILDING_20260602.geojson`
- **Format:** geojson
- **Status:** OK
- **Size:** 961.8 MB
- **Row count:** 1,082,945
- **CRS:** EPSG:4326
- **Bounds:** [-74.255496924591, 40.498437523627, -73.700063823166, 40.915058408069]
- **Missing values (sample):**
  - `name`: 998 (99.8%)
  - `construction_year`: 5 (0.5%)
  - `ground_elevation`: 3 (0.3%)
  - `last_status_type`: 1 (0.1%)

### MapPLUTO – Shoreline Clipped (All Boroughs)
- **Source:** NYC Open Data – Socrata API
- **File:** `data/raw/mappluto.csv`
- **Format:** csv
- **Status:** OK
- **Size:** 268.1 MB
- **Row count:** 721,973
- **Columns:** 108
- **CRS:** EPSG:4326 (inferred from lat/lon columns)
- **Missing values (sample):**
  - `zonedist2`: 4922 (98.44%)
  - `zonedist3`: 5000 (100.0%)
  - `zonedist4`: 5000 (100.0%)
  - `overlay1`: 4605 (92.1%)
  - `overlay2`: 4999 (99.98%)
  - `spdist1`: 4812 (96.24%)
  - `spdist2`: 4999 (99.98%)
  - `spdist3`: 5000 (100.0%)
  - `ltdheight`: 5000 (100.0%)
  - `landuse`: 9 (0.18%)
- **Year range (yearbuilt):** 1887 → 2025
- **Year range (yearalter1):** 1926 → 2026
- **Year range (yearalter2):** 1984 → 2025
- **Date range (appdate):** 1985-04-16 00:00:00 → 2026-03-27 00:00:00

### 2015 Street Tree Census – Tree Data
- **Source:** NYC Open Data – Socrata API
- **File:** `data/raw/tree_census_2015.csv`
- **Format:** csv
- **Status:** OK
- **Size:** 210.1 MB
- **Row count:** 683,788
- **Columns:** 45
- **CRS:** EPSG:4326 (inferred from lat/lon columns)
- **Missing values (sample):**
  - `health`: 206 (4.12%)
  - `spc_latin`: 206 (4.12%)
  - `spc_common`: 206 (4.12%)
  - `steward`: 3328 (66.56%)
  - `guards`: 4088 (81.76%)
  - `sidewalk`: 206 (4.12%)
  - `problems`: 2888 (57.76%)
  - `council district`: 46 (0.92%)
  - `census tract`: 46 (0.92%)
  - `bin`: 66 (1.32%)
- **Date range (created_at):** 2015-05-19 00:00:00 → 2015-09-17 00:00:00

### Landsat 8 Collection 2 Level 2 (Surface Reflectance)
- **Source:** Google Earth Engine – LANDSAT/LC08/C02/T1_L2
- **File:** `data/raw/landsat8_nyc.tif`
- **Format:** geotiff
- **Status:** OK
- **Size:** 18.1 MB
- **CRS:** EPSG:4326
- **Bounds:** [-74.25923296174025, 40.4770087240847, -73.70030119196109, 40.91844085470103]

### Sentinel-2 SR Harmonized
- **Source:** Google Earth Engine – COPERNICUS/S2_SR_HARMONIZED
- **File:** `data/raw/sentinel2_nyc.tif`
- **Format:** geotiff
- **Status:** OK
- **Size:** 16.6 MB
- **CRS:** EPSG:4326
- **Bounds:** [-74.25941262479706, 40.47736805019834, -73.70030119196107, 40.918261191644206]

### ERA5 Hourly Reanalysis (2m Temperature, 10m Wind, Surface Radiation)
- **Source:** Copernicus Climate Data Store (CDS)
- **File:** `data/raw/era5_nyc_hourly.nc`
- **Format:** netcdf
- **Status:** OK
- **Size:** 1.4 MB
- **CRS:** EPSG:4326 (ERA5 standard)

## Authentication-Required Datasets

The following datasets require API credentials. See `SETUP.md` for configuration instructions.

- **Landsat 8 Collection 2 Level 2 (Surface Reflectance)** → Google Earth Engine
- **Sentinel-2 SR Harmonized** → Google Earth Engine
- **ERA5 Hourly Reanalysis (2m Temperature, 10m Wind, Surface Radiation)** → Copernicus CDS API
