"""
Centralized configuration for the NYC Micro-Climate Zoning AI data acquisition pipeline.

All paths, URLs, and dataset metadata are defined here to keep download and
validation scripts DRY and consistent.
"""

from pathlib import Path

# ── Project Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ── NYC Bounding Box (WGS-84) ───────────────────────────────────────────────
NYC_BBOX = {
    "west": -74.2591,
    "south": 40.4774,
    "east": -73.7004,
    "north": 40.9176,
}

# ── Dataset Definitions ─────────────────────────────────────────────────────
# Each entry contains download URL / instructions, expected filename, and type.

DATASETS = {
    "building_footprints": {
        "description": "NYC Building Footprints",
        "filename": "BUILDING_20260602.geojson",
        "source": "NYC Open Data (pre-downloaded)",
        "url": None,  # Already acquired
        "format": "geojson",
        "auth_required": False,
    },
    "mappluto": {
        "description": "MapPLUTO – Shoreline Clipped (All Boroughs)",
        "filename": "mappluto.csv",
        "source": "NYC Open Data – Socrata API",
        "url": "https://data.cityofnewyork.us/api/views/64uk-42ks/rows.csv?accessType=DOWNLOAD",
        "format": "csv",
        "auth_required": False,
    },
    "tree_census": {
        "description": "2015 Street Tree Census – Tree Data",
        "filename": "tree_census_2015.csv",
        "source": "NYC Open Data – Socrata API",
        "url": "https://data.cityofnewyork.us/api/views/uvpi-gqnh/rows.csv?accessType=DOWNLOAD",
        "format": "csv",
        "auth_required": False,
    },
    "landsat8": {
        "description": "Landsat 8 Collection 2 Level 2 (Surface Reflectance)",
        "filename": "landsat8_nyc.tif",
        "source": "Google Earth Engine – LANDSAT/LC08/C02/T1_L2",
        "url": None,  # Requires Earth Engine authentication
        "format": "geotiff",
        "auth_required": True,
        "auth_provider": "Google Earth Engine",
    },
    "sentinel2": {
        "description": "Sentinel-2 SR Harmonized",
        "filename": "sentinel2_nyc.tif",
        "source": "Google Earth Engine – COPERNICUS/S2_SR_HARMONIZED",
        "url": None,  # Requires Earth Engine authentication
        "format": "geotiff",
        "auth_required": True,
        "auth_provider": "Google Earth Engine",
    },
    "era5": {
        "description": "ERA5 Hourly Reanalysis (2m Temperature, 10m Wind, Surface Radiation)",
        "filename": "era5_nyc_hourly.nc",
        "source": "Copernicus Climate Data Store (CDS)",
        "url": None,  # Requires CDS API key
        "format": "netcdf",
        "auth_required": True,
        "auth_provider": "Copernicus CDS API",
    },
}
