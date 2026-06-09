#!/usr/bin/env python3
"""
download_earth_engine.py
────────────────────────
Downloads satellite imagery from Google Earth Engine for the NYC area:
  • Landsat 8 Collection 2 Level 2
  • Sentinel-2 SR Harmonized

Prerequisites:
    pip install earthengine-api
    earthengine authenticate

This script creates a cloud-optimised median composite for a given date range,
clips it to the NYC bounding box, and exports it as a GeoTIFF.

Usage:
    source .venv/bin/activate
    python scripts/download_earth_engine.py             # both datasets
    python scripts/download_earth_engine.py landsat8     # Landsat only
    python scripts/download_earth_engine.py sentinel2    # Sentinel only
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import DATASETS, RAW_DIR, NYC_BBOX


def _check_ee():
    """Verify that the Earth Engine Python API is importable and authenticated."""
    try:
        import ee  # noqa: F811
    except ImportError:
        print("ERROR: earthengine-api is not installed.")
        print("       pip install earthengine-api")
        sys.exit(1)

    try:
        ee.Initialize()
        print("  ✓  Earth Engine authenticated and initialized.")
    except Exception as exc:
        print(f"ERROR: Earth Engine authentication failed → {exc}")
        print("       Run: earthengine authenticate")
        sys.exit(1)

    return ee


def _nyc_geometry(ee):
    """Return an ee.Geometry.Rectangle for NYC."""
    return ee.Geometry.Rectangle(
        [NYC_BBOX["west"], NYC_BBOX["south"], NYC_BBOX["east"], NYC_BBOX["north"]]
    )


# ── Landsat 8 ────────────────────────────────────────────────────────────────

def download_landsat8(ee, start_date="2024-06-01", end_date="2024-08-31"):
    """
    Download a cloud-free Landsat 8 median composite (RGB + Thermal) for NYC.
    """
    dest = RAW_DIR / DATASETS["landsat8"]["filename"]
    if dest.exists():
        print(f"  ⊘  {dest.name} already exists. Skipping.")
        return dest

    aoi = _nyc_geometry(ee)

    def _apply_scale_factors(image):
        optical = image.select("SR_B.*").multiply(0.0000275).add(-0.2)
        thermal = image.select("ST_B.*").multiply(0.00341802).add(149.0)
        return image.addBands(optical, overwrite=True).addBands(thermal, overwrite=True)

    collection = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUD_COVER", 20))
        .map(_apply_scale_factors)
    )

    count = collection.size().getInfo()
    print(f"  ℹ  Landsat 8 scenes matching filters: {count}")
    if count == 0:
        print("  ⚠  No scenes found. Try widening the date range or cloud cover threshold.")
        return None

    composite = collection.median().clip(aoi)
    bands = ["SR_B4", "SR_B3", "SR_B2", "SR_B5", "ST_B10"]  # R, G, B, NIR, Thermal
    composite = composite.select(bands)

    # Export via getDownloadURL (small-area approach; for large areas use Export.image.toDrive)
    print("  ↓  Requesting GeoTIFF download from Earth Engine…")
    url = composite.getDownloadURL({
        "name": "landsat8_nyc",
        "bands": bands,
        "region": aoi,
        "scale": 60,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })

    import requests
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    print(f"  ✓  Saved → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# ── Sentinel-2 ───────────────────────────────────────────────────────────────

def download_sentinel2(ee, start_date="2024-06-01", end_date="2024-08-31"):
    """
    Download a cloud-free Sentinel-2 SR median composite (RGB + NIR) for NYC.
    """
    dest = RAW_DIR / DATASETS["sentinel2"]["filename"]
    if dest.exists():
        print(f"  ⊘  {dest.name} already exists. Skipping.")
        return dest

    aoi = _nyc_geometry(ee)

    def _mask_clouds(image):
        qa = image.select("QA60")
        cloud_bit = 1 << 10
        cirrus_bit = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
        return image.updateMask(mask).divide(10000)

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
        .map(_mask_clouds)
    )

    count = collection.size().getInfo()
    print(f"  ℹ  Sentinel-2 scenes matching filters: {count}")
    if count == 0:
        print("  ⚠  No scenes found. Try widening the date range.")
        return None

    bands = ["B4", "B3", "B2", "B8"]  # R, G, B, NIR
    composite = collection.median().clip(aoi).select(bands)

    print("  ↓  Requesting GeoTIFF download from Earth Engine…")
    url = composite.getDownloadURL({
        "name": "sentinel2_nyc",
        "bands": bands,
        "region": aoi,
        "scale": 40,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })

    import requests
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)

    print(f"  ✓  Saved → {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


# ── CLI ──────────────────────────────────────────────────────────────────────

TARGETS = {
    "landsat8": download_landsat8,
    "sentinel2": download_sentinel2,
}

if __name__ == "__main__":
    ee = _check_ee()

    target = sys.argv[1] if len(sys.argv) > 1 else None

    print("=" * 70)
    print("NYC Micro-Climate Zoning AI – Earth Engine Downloader")
    print("=" * 70)

    if target:
        if target not in TARGETS:
            print(f"Unknown target: {target}. Available: {', '.join(TARGETS)}")
            sys.exit(1)
        TARGETS[target](ee)
    else:
        for key, fn in TARGETS.items():
            print(f"\n[{key}]")
            fn(ee)

    print("\n" + "=" * 70)
    print("Done.")
    print("=" * 70)
