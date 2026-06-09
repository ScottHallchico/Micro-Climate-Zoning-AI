# NYC Micro-Climate Zoning AI — Setup Guide

## Table of Contents

1. [System Prerequisites](#1-system-prerequisites)
2. [Python Environment Setup](#2-python-environment-setup)
3. [Download Public Datasets](#3-download-public-datasets)
4. [Google Earth Engine Setup](#4-google-earth-engine-setup)
5. [Copernicus CDS API Setup](#5-copernicus-cds-api-setup)
6. [Validate All Datasets](#6-validate-all-datasets)
7. [Project Directory Structure](#7-project-directory-structure)

---

## 1. System Prerequisites

The pipeline depends on several C/C++ geospatial libraries. Install them **before**
creating the Python environment.

### Ubuntu / Debian

```bash
sudo apt update
sudo apt install -y \
    python3-dev python3-venv \
    gdal-bin libgdal-dev \
    libgeos-dev \
    libproj-dev proj-data \
    libnetcdf-dev libhdf5-dev \
    libspatialindex-dev \
    build-essential
```

### macOS (Homebrew)

```bash
brew install gdal geos proj netcdf hdf5 spatialindex
```

### Verify system libraries

```bash
gdal-config --version   # ≥ 3.6
geos-config --version   # ≥ 3.11
proj 2>&1 | head -1     # ≥ 9.0
nc-config --version     # ≥ 4.9
```

---

## 2. Python Environment Setup

```bash
# Clone the repository (if you haven't already)
cd Micro-Climate-Zoning-AI

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies (vector GIS, raster, climate, reporting)
pip install -r requirements.txt

# Install remote-sensing dependencies (Earth Engine + CDS API)
# → Only after completing Steps 4 and 5 below
pip install -r requirements-remote.txt

# (Optional) Install development tools
pip install -r requirements-dev.txt
```

---

## 3. Download Public Datasets

These datasets are freely available from NYC Open Data with **no authentication**:

| Dataset | Source | Auto-Download |
|---------|--------|:---:|
| Building Footprints | NYC Open Data | ✓ (pre-downloaded) |
| MapPLUTO (Shoreline Clipped) | NYC Open Data | ✓ |
| 2015 Street Tree Census | NYC Open Data | ✓ |

```bash
source .venv/bin/activate
python scripts/download_open_data.py
```

To re-download (overwrite existing files):

```bash
python scripts/download_open_data.py --force
```

---

## 4. Google Earth Engine Setup

Required for: **Landsat 8 Collection 2 Level 2** and **Sentinel-2 SR Harmonized**.

### Step 1: Register for Earth Engine

1. Go to <https://earthengine.google.com/>
2. Sign in with your Google account
3. Request access if you haven't already (approval is usually instant for
   research/education use)

### Step 2: Install & Authenticate

```bash
source .venv/bin/activate
pip install -r requirements-remote.txt

# Authenticate (opens a browser for Google OAuth)
earthengine authenticate
```

This stores credentials at `~/.config/earthengine/credentials`.

### Step 3: Verify

```python
import ee
ee.Initialize()
print(ee.Image("LANDSAT/LC08/C02/T1_L2/LC08_013032_20240701").bandNames().getInfo())
```

### Step 4: Download

```bash
python scripts/download_earth_engine.py              # Both datasets
python scripts/download_earth_engine.py landsat8      # Landsat only
python scripts/download_earth_engine.py sentinel2     # Sentinel only
```

---

## 5. Copernicus CDS API Setup

Required for: **ERA5 Hourly Reanalysis**.

### Step 1: Register for CDS

1. Go to <https://cds.climate.copernicus.eu/>
2. Create a free account
3. Log in and navigate to your profile page
4. Copy your **API key** (UID and Key)

### Step 2: Create Configuration File

Create `~/.cdsapirc`:

```bash
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api
key: <YOUR-UID>:<YOUR-API-KEY>
EOF
chmod 600 ~/.cdsapirc
```

Replace `<YOUR-UID>:<YOUR-API-KEY>` with your actual credentials from the CDS
profile page (e.g., `12345:abcdef12-3456-7890-abcd-ef1234567890`).

### Step 3: Accept License Terms

Before your first download, you must accept the ERA5 license:

1. Visit: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels>
2. Scroll to the bottom and click **"Accept Terms"**

### Step 4: Download

```bash
python scripts/download_era5.py                         # Full year 2024
python scripts/download_era5.py --year 2024 --months 6 7 8  # Summer only
```

---

## 6. Validate All Datasets

After downloading, run the validation suite:

```bash
source .venv/bin/activate
python scripts/validate_datasets.py
```

This produces:

| Output | Path |
|--------|------|
| JSON validation results | `reports/validation_results.json` |
| Markdown inventory report | `reports/dataset_inventory.md` |

---

## 7. Project Directory Structure

```
Micro-Climate-Zoning-AI/
├── data/
│   ├── raw/                          # ← All raw downloads go here
│   │   ├── BUILDING_20260602.geojson # Building Footprints (pre-downloaded)
│   │   ├── mappluto.csv              # MapPLUTO (auto-downloaded)
│   │   ├── tree_census_2015.csv      # Tree Census (auto-downloaded)
│   │   ├── landsat8_nyc.tif          # Landsat 8 (Earth Engine)
│   │   ├── sentinel2_nyc.tif         # Sentinel-2 (Earth Engine)
│   │   └── era5_nyc_hourly.nc        # ERA5 (CDS API)
│   └── processed/                    # ← Future: cleaned/transformed data
├── reports/
│   ├── validation_results.json       # Machine-readable validation output
│   └── dataset_inventory.md          # Human-readable inventory report
├── scripts/
│   ├── __init__.py
│   ├── config.py                     # Central configuration
│   ├── download_all.py               # Master orchestrator
│   ├── download_open_data.py         # NYC Open Data downloader
│   ├── download_earth_engine.py      # GEE satellite downloader
│   ├── download_era5.py              # CDS ERA5 downloader
│   └── validate_datasets.py          # Dataset validator + reporter
├── requirements.txt                  # Core dependencies
├── requirements-remote.txt           # Earth Engine + CDS API deps
├── requirements-dev.txt              # Dev/test tools
├── SETUP.md                          # This file
└── .venv/                            # Python virtual environment
```

---

## Troubleshooting

### `rasterio` fails to install

Ensure GDAL is installed at the system level:

```bash
sudo apt install gdal-bin libgdal-dev
export GDAL_CONFIG=/usr/bin/gdal-config
pip install rasterio
```

### `netCDF4` fails to install

Ensure HDF5 and NetCDF C libraries are installed:

```bash
sudo apt install libhdf5-dev libnetcdf-dev
pip install netCDF4
```

### Earth Engine `ee.Initialize()` fails

```bash
earthengine authenticate --force
```

### CDS API returns 403

- Verify `~/.cdsapirc` has the correct key
- Ensure you've accepted the ERA5 dataset license on the CDS website
