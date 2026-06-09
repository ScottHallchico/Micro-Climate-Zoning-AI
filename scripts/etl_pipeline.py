#!/usr/bin/env python3
"""
etl_pipeline.py — Spatial ETL for NYC Micro-Climate Zoning AI.

Joins Building Footprints + MapPLUTO + Tree Census + Landsat 8 +
Sentinel-2 + ERA5 into a single building_master.parquet.

Usage:
    source .venv/bin/activate
    python scripts/etl_pipeline.py
"""

import sys, time, json, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import RAW_DIR, PROCESSED_DIR, REPORTS_DIR

# ── Constants ────────────────────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"
CRS_PROJECTED = "EPSG:32618"  # UTM 18N (meters) for NYC spatial ops
TREE_BUFFER_M = 100           # meters around each building centroid

LOG = []  # Collect step timings for ETL report


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] {msg}")


def _step(name):
    """Context-manager-like timing helper."""
    class _Timer:
        def __init__(self):
            self.name = name
            self.t0 = None
        def __enter__(self):
            _log(f"▶ {name}")
            self.t0 = time.time()
            return self
        def __exit__(self, *_):
            elapsed = time.time() - self.t0
            _log(f"✓ {name} ({elapsed:.1f}s)")
            LOG.append({"step": name, "seconds": round(elapsed, 1)})
    return _Timer()


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1 — Load Datasets
# ═══════════════════════════════════════════════════════════════════════

def load_buildings():
    """Load building footprints, keep only needed columns."""
    with _step("Load Building Footprints"):
        keep = ["bin", "mappluto_bbl", "height_roof", "ground_elevation",
                "construction_year", "geometry"]
        gdf = gpd.read_file(RAW_DIR / "BUILDING_20260602.geojson")
        gdf = gdf[[c for c in keep if c in gdf.columns]]
        gdf = gdf.set_crs(CRS_WGS84, allow_override=True)

        # Coerce numeric columns
        for col in ["height_roof", "ground_elevation", "construction_year"]:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

        _log(f"  rows={len(gdf):,}, cols={list(gdf.columns)}")
    return gdf


def load_pluto():
    """Load MapPLUTO, select planning-relevant columns."""
    with _step("Load MapPLUTO"):
        cols_map = {
            "BBL": "bbl", "zonedist1": "zoning", "landuse": "land_use",
            "builtfar": "far", "lotarea": "lot_area", "numfloors": "num_floors",
            "bldgarea": "bldg_area", "unitsres": "units_res",
        }
        df = pd.read_csv(RAW_DIR / "mappluto.csv",
                         usecols=list(cols_map.keys()), low_memory=False)
        df = df.rename(columns=cols_map)

        for col in ["far", "lot_area", "num_floors", "bldg_area"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Safely convert BBL to string, avoiding '.0' suffixes caused by float64 nulls
        df["bbl"] = pd.to_numeric(df["bbl"], errors="coerce").astype("Int64").astype(str).str.strip()
        df = df[df["bbl"] != "<NA>"]  # Remove rows with null BBLs

        df = df.drop_duplicates(subset="bbl", keep="first")
        _log(f"  rows={len(df):,}, cols={list(df.columns)}")
    return df


def load_trees():
    """Load tree census, create point GeoDataFrame."""
    with _step("Load Tree Census"):
        cols = ["tree_id", "tree_dbh", "latitude", "longitude",
                "spc_common", "health", "status"]
        df = pd.read_csv(RAW_DIR / "tree_census_2015.csv",
                         usecols=cols, low_memory=False)
        df = df.dropna(subset=["latitude", "longitude"])
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs=CRS_WGS84,
        )
        gdf = gdf.drop(columns=["latitude", "longitude"])
        _log(f"  rows={len(gdf):,}")
    return gdf


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2 — Join Buildings ↔ MapPLUTO
# ═══════════════════════════════════════════════════════════════════════

def join_buildings_pluto(buildings, pluto):
    """Left-join buildings to MapPLUTO on BBL."""
    with _step("Join Buildings ↔ MapPLUTO"):
        buildings["_join_bbl"] = buildings["mappluto_bbl"].astype(str).str.strip()
        master = buildings.merge(pluto, left_on="_join_bbl", right_on="bbl",
                                 how="left", suffixes=("", "_pluto"))
        master = master.drop(columns=["_join_bbl"], errors="ignore")

        matched = master["zoning"].notna().sum()
        _log(f"  matched={matched:,}/{len(master):,} "
             f"({matched/len(master)*100:.1f}%)")
    return master


# ═══════════════════════════════════════════════════════════════════════
#  STEP 3 — Vegetation Features (Tree Census)
# ═══════════════════════════════════════════════════════════════════════

def compute_tree_features(master, trees):
    """Count nearby trees & canopy density within buffer around each building."""
    from scipy.spatial import cKDTree

    with _step("Compute Tree Features (spatial)"):
        # Project to meters for distance calculations
        master_proj = master.to_crs(CRS_PROJECTED)
        trees_proj = trees.to_crs(CRS_PROJECTED)

        # Building centroids
        centroids = master_proj.geometry.centroid
        bldg_coords = np.column_stack([centroids.x, centroids.y])

        # Tree coordinates and pre-computed canopy areas
        tree_coords = np.column_stack([trees_proj.geometry.x,
                                       trees_proj.geometry.y])
        dbh = trees["tree_dbh"].fillna(0).values.astype(float)
        # Crown spread (m) ≈ (1 + 0.3 * dbh_inches) * 0.3048
        crown_d = (1.0 + 0.3 * dbh) * 0.3048
        tree_canopy_area = np.pi * (crown_d / 2.0) ** 2

        buffer_area = np.pi * TREE_BUFFER_M ** 2

        # Build KD-tree from tree locations
        _log("  Building spatial index (KD-tree)…")
        tree_kd = cKDTree(tree_coords)

        # Query all building centroids at once
        _log(f"  Querying {len(bldg_coords):,} buildings × "
             f"{len(tree_coords):,} trees (r={TREE_BUFFER_M}m)…")
        results = tree_kd.query_ball_point(bldg_coords, TREE_BUFFER_M,
                                           workers=-1)

        # Vectorized count
        nearby_counts = np.array([len(r) for r in results], dtype=np.int32)

        # Canopy density
        canopy_dens = np.zeros(len(results), dtype=np.float32)
        for i, idxs in enumerate(results):
            if idxs:
                canopy_dens[i] = min(tree_canopy_area[idxs].sum()
                                     / buffer_area, 1.0)

        master["nearby_tree_count"] = nearby_counts
        master["canopy_density"] = canopy_dens
        _log(f"  mean trees/bldg={nearby_counts.mean():.1f}, "
             f"mean canopy={canopy_dens.mean():.4f}")
    return master


# ═══════════════════════════════════════════════════════════════════════
#  STEP 4 — Remote Sensing Features (NDVI, Surface Temperature)
# ═══════════════════════════════════════════════════════════════════════

def compute_raster_features(master):
    """Extract NDVI (Sentinel-2) and surface temperature (Landsat 8)."""
    import rasterio

    with _step("Compute Raster Features"):
        centroids = master.geometry.centroid
        coords = list(zip(centroids.x, centroids.y))

        # ── Sentinel-2 NDVI (bands: B4=Red[0], B3=Green[1], B2=Blue[2], B8=NIR[3])
        s2_path = RAW_DIR / "sentinel2_nyc.tif"
        _log(f"  Sampling Sentinel-2 ({len(coords):,} points)…")
        with rasterio.open(s2_path) as src:
            red_vals = np.full(len(coords), np.nan, dtype=np.float32)
            nir_vals = np.full(len(coords), np.nan, dtype=np.float32)
            for i, val in enumerate(src.sample(coords)):
                red_vals[i] = val[0]   # B4 = Red
                nir_vals[i] = val[3]   # B8 = NIR

        denom = nir_vals + red_vals
        ndvi = np.where(denom != 0, (nir_vals - red_vals) / denom, np.nan)
        ndvi = np.clip(ndvi, -1, 1)
        master["ndvi"] = ndvi.astype(np.float32)

        # ── Landsat 8 surface temperature (bands: R[0],G[1],B[2],NIR[3],Thermal[4])
        l8_path = RAW_DIR / "landsat8_nyc.tif"
        _log(f"  Sampling Landsat 8 thermal ({len(coords):,} points)…")
        with rasterio.open(l8_path) as src:
            thermal_vals = np.full(len(coords), np.nan, dtype=np.float32)
            for i, val in enumerate(src.sample(coords)):
                thermal_vals[i] = val[4]  # ST_B10 = Thermal (Kelvin)

        # Convert Kelvin → Celsius
        surface_temp_c = thermal_vals - 273.15
        master["surface_temperature"] = surface_temp_c.astype(np.float32)

        valid_ndvi = np.isfinite(master["ndvi"]).sum()
        valid_temp = np.isfinite(master["surface_temperature"]).sum()
        _log(f"  NDVI valid={valid_ndvi:,}, mean={np.nanmean(ndvi):.4f}")
        _log(f"  LST valid={valid_temp:,}, "
             f"mean={np.nanmean(surface_temp_c):.1f}°C")
    return master


# ═══════════════════════════════════════════════════════════════════════
#  STEP 5 — Weather Features (ERA5)
# ═══════════════════════════════════════════════════════════════════════

def compute_weather_features(master):
    """Extract annual-mean ERA5 climate features for each building."""
    import xarray as xr

    with _step("Compute Weather Features (ERA5)"):
        ds = xr.open_dataset(RAW_DIR / "era5_nyc_hourly.nc")

        # Compute temporal means across all hours (city-wide annual means)
        # ERA5 grid is ~25 km — only 2×3 cells cover NYC
        t2m_mean = float(ds["t2m"].mean()) - 273.15     # K → °C
        d2m_mean = float(ds["d2m"].mean()) - 273.15     # K → °C
        u10_mean = float(ds["u10"].mean())
        v10_mean = float(ds["v10"].mean())
        sp_mean  = float(ds["sp"].mean())
        ssrd_mean = float(ds["ssrd"].mean()) / 3600.0   # J/m² → W/m²

        # Derived: wind speed & direction
        wind_speed = np.sqrt(u10_mean**2 + v10_mean**2)
        wind_dir = (270 - np.degrees(np.arctan2(v10_mean, u10_mean))) % 360

        # Derived: relative humidity (Magnus formula)
        # RH = 100 * exp(17.625*Td / (243.04+Td)) / exp(17.625*T / (243.04+T))
        rh = 100.0 * (np.exp(17.625 * d2m_mean / (243.04 + d2m_mean))
                       / np.exp(17.625 * t2m_mean / (243.04 + t2m_mean)))
        rh = np.clip(rh, 0, 100)

        # Also compute spatially-varying values per building centroid
        # using nearest-neighbor interpolation from the ERA5 grid
        centroids = master.geometry.centroid
        lats = centroids.y.values
        lons = centroids.x.values

        # ERA5 grid coords
        era5_lats = ds["latitude"].values
        era5_lons = ds["longitude"].values

        # Find nearest grid cell for each building
        lat_idx = np.abs(era5_lats[None, :] - lats[:, None]).argmin(axis=1)
        lon_idx = np.abs(era5_lons[None, :] - lons[:, None]).argmin(axis=1)

        # Extract temporal means at each grid cell
        t2m_grid = ds["t2m"].mean(dim="valid_time").values  # (lat, lon)
        d2m_grid = ds["d2m"].mean(dim="valid_time").values
        u10_grid = ds["u10"].mean(dim="valid_time").values
        v10_grid = ds["v10"].mean(dim="valid_time").values
        ssrd_grid = ds["ssrd"].mean(dim="valid_time").values

        # Assign per-building values
        master["temperature"] = (t2m_grid[lat_idx, lon_idx] - 273.15).astype(np.float32)

        d2m_vals = d2m_grid[lat_idx, lon_idx] - 273.15
        t2m_vals = t2m_grid[lat_idx, lon_idx] - 273.15
        master["humidity"] = np.clip(
            100.0 * np.exp(17.625 * d2m_vals / (243.04 + d2m_vals))
                  / np.exp(17.625 * t2m_vals / (243.04 + t2m_vals)),
            0, 100
        ).astype(np.float32)

        u_vals = u10_grid[lat_idx, lon_idx]
        v_vals = v10_grid[lat_idx, lon_idx]
        master["wind_speed"] = np.sqrt(u_vals**2 + v_vals**2).astype(np.float32)
        master["wind_direction"] = (
            (270 - np.degrees(np.arctan2(v_vals, u_vals))) % 360
        ).astype(np.float32)

        master["solar_radiation"] = (
            ssrd_grid[lat_idx, lon_idx] / 3600.0
        ).astype(np.float32)

        ds.close()

        _log(f"  temp={t2m_mean:.1f}°C, RH={rh:.1f}%, "
             f"wind={wind_speed:.1f}m/s @ {wind_dir:.0f}°, "
             f"solar={ssrd_mean:.0f} W/m²")
    return master


# ═══════════════════════════════════════════════════════════════════════
#  STEP 6 — Finalize Schema & Save
# ═══════════════════════════════════════════════════════════════════════

FINAL_SCHEMA = [
    # Building features
    "building_id", "geometry", "height_roof", "ground_elevation",
    "construction_year",
    # Planning features
    "zoning", "land_use", "far", "lot_area",
    # Vegetation features
    "nearby_tree_count", "canopy_density",
    # Remote sensing features
    "ndvi", "surface_temperature",
    # Weather features
    "temperature", "humidity", "wind_speed", "wind_direction",
    "solar_radiation",
]


def finalize_and_save(master):
    """Rename columns, enforce schema, save to parquet."""
    with _step("Finalize & Save"):
        # Rename bin → building_id
        if "bin" in master.columns:
            master = master.rename(columns={"bin": "building_id"})

        # Keep only final schema columns
        available = [c for c in FINAL_SCHEMA if c in master.columns]
        missing = [c for c in FINAL_SCHEMA if c not in master.columns]
        if missing:
            _log(f"  ⚠ Missing columns: {missing}")

        master = master[available]
        master = master.set_crs(CRS_WGS84, allow_override=True)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        out_path = PROCESSED_DIR / "building_master.parquet"
        master.to_parquet(out_path, index=False)
        _log(f"  Saved → {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
        _log(f"  Rows: {len(master):,}, Cols: {len(master.columns)}")
    return master


# ═══════════════════════════════════════════════════════════════════════
#  STEP 7 — Generate Reports
# ═══════════════════════════════════════════════════════════════════════

def generate_reports(master):
    """Produce ETL report, feature inventory, and data quality report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── ETL Report ──
    with _step("Generate Reports"):
        lines = [
            "# ETL Pipeline Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Output: `data/processed/building_master.parquet`",
            f"- **Rows:** {len(master):,}",
            f"- **Columns:** {len(master.columns)}",
            f"- **CRS:** {master.crs}",
            "",
            "## Processing Steps",
            "",
            "| Step | Duration |",
            "|------|----------|",
        ]
        total_s = 0
        for entry in LOG:
            lines.append(f"| {entry['step']} | {entry['seconds']}s |")
            total_s += entry["seconds"]
        lines.append(f"| **Total** | **{total_s:.1f}s** |")

        # Join statistics
        lines += ["", "## Join Statistics", ""]
        matched = master["zoning"].notna().sum()
        lines.append(f"- Buildings ↔ MapPLUTO: {matched:,}/{len(master):,} "
                      f"({matched/len(master)*100:.1f}% match)")
        tree_present = (master["nearby_tree_count"] > 0).sum()
        lines.append(f"- Buildings with ≥1 tree within {TREE_BUFFER_M}m: "
                      f"{tree_present:,}/{len(master):,} "
                      f"({tree_present/len(master)*100:.1f}%)")

        (REPORTS_DIR / "etl_report.md").write_text("\n".join(lines))

        # ── Feature Inventory ──
        inv_lines = [
            "# Feature Inventory",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| # | Feature | Type | Non-Null | Mean | Min | Max | Source |",
            "|---|---------|------|----------|------|-----|-----|--------|",
        ]
        sources = {
            "building_id": "Building Footprints",
            "geometry": "Building Footprints",
            "height_roof": "Building Footprints",
            "ground_elevation": "Building Footprints",
            "construction_year": "Building Footprints",
            "zoning": "MapPLUTO",
            "land_use": "MapPLUTO",
            "far": "MapPLUTO",
            "lot_area": "MapPLUTO",
            "nearby_tree_count": "Tree Census (spatial)",
            "canopy_density": "Tree Census (derived)",
            "ndvi": "Sentinel-2",
            "surface_temperature": "Landsat 8 thermal",
            "temperature": "ERA5",
            "humidity": "ERA5 (derived)",
            "wind_speed": "ERA5 (derived)",
            "wind_direction": "ERA5 (derived)",
            "solar_radiation": "ERA5",
        }

        for i, col in enumerate(master.columns, 1):
            if col == "geometry":
                inv_lines.append(
                    f"| {i} | geometry | MultiPolygon | {len(master):,} "
                    f"| — | — | — | Building Footprints |")
                continue
            dtype = str(master[col].dtype)
            nn = master[col].notna().sum()
            src = sources.get(col, "—")
            if pd.api.types.is_numeric_dtype(master[col]):
                vals = master[col].dropna()
                if len(vals) > 0:
                    inv_lines.append(
                        f"| {i} | {col} | {dtype} | {nn:,} "
                        f"| {vals.mean():.4f} | {vals.min():.4f} "
                        f"| {vals.max():.4f} | {src} |")
                else:
                    inv_lines.append(
                        f"| {i} | {col} | {dtype} | {nn:,} "
                        f"| — | — | — | {src} |")
            else:
                nuniq = master[col].nunique()
                inv_lines.append(
                    f"| {i} | {col} | {dtype} | {nn:,} "
                    f"| {nuniq} unique | — | — | {src} |")

        (REPORTS_DIR / "feature_inventory.md").write_text("\n".join(inv_lines))

        # ── Data Quality Report ──
        dq_lines = [
            "# Data Quality Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Completeness",
            "",
            "| Feature | Total | Non-Null | Null | Completeness |",
            "|---------|-------|----------|------|-------------|",
        ]
        total = len(master)
        for col in master.columns:
            if col == "geometry":
                continue
            nn = int(master[col].notna().sum())
            nl = total - nn
            pct = nn / total * 100
            dq_lines.append(
                f"| {col} | {total:,} | {nn:,} | {nl:,} | {pct:.1f}% |")

        dq_lines += ["", "## Numeric Feature Distributions", ""]
        num_cols = [c for c in master.columns
                    if c != "geometry" and pd.api.types.is_numeric_dtype(master[c])]
        dq_lines.append(
            "| Feature | Mean | Std | P5 | P25 | P50 | P75 | P95 |")
        dq_lines.append(
            "|---------|------|-----|----|----|-----|-----|-----|")
        for col in num_cols:
            s = master[col].dropna()
            if len(s) == 0:
                continue
            pcts = np.nanpercentile(s, [5, 25, 50, 75, 95])
            dq_lines.append(
                f"| {col} | {s.mean():.4f} | {s.std():.4f} | "
                f"{pcts[0]:.4f} | {pcts[1]:.4f} | {pcts[2]:.4f} | "
                f"{pcts[3]:.4f} | {pcts[4]:.4f} |")

        (REPORTS_DIR / "data_quality.md").write_text("\n".join(dq_lines))

        _log("  Reports saved → reports/etl_report.md, "
             "feature_inventory.md, data_quality.md")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 64 + "╗")
    print("║  NYC Micro-Climate Zoning AI — ETL Pipeline                   ║")
    print("╚" + "═" * 64 + "╝")

    t0 = time.time()

    # Load
    buildings = load_buildings()
    pluto = load_pluto()
    trees = load_trees()

    # Join
    master = join_buildings_pluto(buildings, pluto)
    del buildings, pluto  # free memory

    # Enrich
    master = compute_tree_features(master, trees)
    del trees

    master = compute_raster_features(master)
    master = compute_weather_features(master)

    # Save
    master = finalize_and_save(master)
    generate_reports(master)

    elapsed = time.time() - t0
    print(f"\n{'═' * 66}")
    print(f"  ✓ ETL complete in {elapsed/60:.1f} minutes")
    print(f"  Output: data/processed/building_master.parquet")
    print(f"  Reports: reports/etl_report.md, feature_inventory.md, "
          f"data_quality.md")
    print(f"{'═' * 66}")


if __name__ == "__main__":
    main()
