#!/usr/bin/env python3
"""
validate_datasets.py – Validates all downloaded datasets and produces reports.

Checks: existence, file size, row counts, CRS, missing values, date ranges.
Outputs: reports/dataset_inventory.md  +  reports/validation_results.json

Usage:
    source .venv/bin/activate
    python scripts/validate_datasets.py
"""
import sys, json, os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import DATASETS, RAW_DIR, REPORTS_DIR


def sizeof_fmt(num):
    for u in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024:
            return f"{num:.1f} {u}"
        num /= 1024
    return f"{num:.1f} TB"


def validate_geojson(filepath):
    """Validate a GeoJSON file using geopandas (streaming for large files)."""
    import geopandas as gpd

    result = {"status": "OK", "errors": []}
    try:
        # Use fiona to get metadata without loading everything
        import fiona
        with fiona.open(filepath) as src:
            result["row_count"] = len(src)
            result["crs"] = str(src.crs) if src.crs else "Unknown"
            result["schema"] = {
                "geometry": src.schema["geometry"],
                "properties": list(src.schema["properties"].keys())
            }
            result["bounds"] = list(src.bounds)
            result["driver"] = src.driver

        # Sample first 1000 rows for missing value analysis
        gdf = gpd.read_file(filepath, rows=1000)
        missing = gdf.isnull().sum()
        total = len(gdf)
        result["missing_values_sample"] = {
            col: {"count": int(v), "pct": round(v / total * 100, 2)}
            for col, v in missing.items() if v > 0
        }
        result["columns"] = list(gdf.columns)
        result["sample_crs"] = str(gdf.crs) if gdf.crs else "Unknown"

    except Exception as e:
        result["status"] = "ERROR"
        result["errors"].append(str(e))

    return result


def validate_csv(filepath):
    """Validate a CSV file."""
    import pandas as pd

    result = {"status": "OK", "errors": []}
    try:
        # Get row count efficiently
        with open(filepath, "r") as f:
            row_count = sum(1 for _ in f) - 1  # subtract header
        result["row_count"] = row_count

        # Sample first 5000 rows for schema + missing analysis
        df = pd.read_csv(filepath, nrows=5000, low_memory=False)
        result["columns"] = list(df.columns)
        result["column_count"] = len(df.columns)
        result["dtypes"] = {col: str(dt) for col, dt in df.dtypes.items()}

        missing = df.isnull().sum()
        total = len(df)
        result["missing_values_sample"] = {
            col: {"count": int(v), "pct": round(v / total * 100, 2)}
            for col, v in missing.items() if v > 0
        }

        # Detect coordinate columns for CRS inference
        coord_cols = [c for c in df.columns if c.lower() in
                      ("latitude", "longitude", "lat", "lng", "lon",
                       "x", "y", "xcoord", "ycoord", "x_coord", "y_coord")]
        if coord_cols:
            result["coordinate_columns"] = coord_cols
            result["crs"] = "EPSG:4326 (inferred from lat/lon columns)"
        else:
            result["crs"] = "N/A (tabular data)"

        # Try to find date columns
        date_cols = [c for c in df.columns if any(kw in c.lower()
                     for kw in ("date", "year", "created", "updated", "time"))]
        if date_cols:
            result["date_columns"] = date_cols
            for dc in date_cols[:5]:
                try:
                    col = df[dc].dropna()
                    if len(col) == 0:
                        continue
                    # Integer year columns (yearbuilt, yearalter1, etc.)
                    if pd.api.types.is_numeric_dtype(col):
                        vals = col[col > 0]
                        if len(vals) > 0 and vals.min() > 1600 and vals.max() < 2100:
                            result[f"year_range_{dc}"] = {
                                "min": int(vals.min()),
                                "max": int(vals.max()),
                            }
                            continue
                    # String date columns (created_at, appdate, etc.)
                    dates = pd.to_datetime(col, errors="coerce").dropna()
                    if len(dates) > 0:
                        result[f"date_range_{dc}"] = {
                            "min": str(dates.min()),
                            "max": str(dates.max()),
                        }
                except Exception:
                    pass

    except Exception as e:
        result["status"] = "ERROR"
        result["errors"].append(str(e))

    return result


def validate_geotiff(filepath):
    """Validate a GeoTIFF file (basic check)."""
    result = {"status": "OK", "errors": []}
    try:
        # Try rasterio if available, otherwise basic check
        try:
            import rasterio
            with rasterio.open(filepath) as src:
                result["crs"] = str(src.crs)
                result["bounds"] = list(src.bounds)
                result["resolution"] = list(src.res)
                result["shape"] = list(src.shape)
                result["band_count"] = src.count
                result["dtypes"] = list(src.dtypes)
        except ImportError:
            result["note"] = "rasterio not installed; basic validation only"
            result["crs"] = "Unknown (install rasterio for full validation)"
    except Exception as e:
        result["status"] = "ERROR"
        result["errors"].append(str(e))
    return result


def validate_netcdf(filepath):
    """Validate a NetCDF file (basic check)."""
    result = {"status": "OK", "errors": []}
    try:
        try:
            import netCDF4 as nc
            ds = nc.Dataset(filepath)
            result["variables"] = list(ds.variables.keys())
            result["dimensions"] = {k: v.size for k, v in ds.dimensions.items()}
            result["crs"] = "EPSG:4326 (ERA5 standard)"

            if "time" in ds.variables:
                import cftime
                times = nc.num2date(ds.variables["time"][:],
                                    ds.variables["time"].units)
                result["date_range"] = {
                    "min": str(times[0]),
                    "max": str(times[-1]),
                }
            ds.close()
        except ImportError:
            result["note"] = "netCDF4 not installed; basic validation only"
    except Exception as e:
        result["status"] = "ERROR"
        result["errors"].append(str(e))
    return result


VALIDATORS = {
    "geojson": validate_geojson,
    "csv": validate_csv,
    "geotiff": validate_geotiff,
    "netcdf": validate_netcdf,
}


def validate_all():
    """Run validation on all datasets and produce reports."""
    print("=" * 70)
    print("NYC Micro-Climate Zoning AI – Dataset Validator")
    print("=" * 70)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    inventory = {}
    all_ok = True

    for key, meta in DATASETS.items():
        filepath = RAW_DIR / meta["filename"]
        print(f"\n[{key}] {meta['description']}")

        entry = {
            "description": meta["description"],
            "source": meta["source"],
            "filename": meta["filename"],
            "format": meta["format"],
            "auth_required": meta["auth_required"],
        }

        if not filepath.exists():
            entry["status"] = "MISSING"
            entry["file_size"] = None
            if meta["auth_required"]:
                prov = meta.get("auth_provider", "unknown")
                entry["note"] = f"Requires {prov} auth. See SETUP.md."
                print(f"  ⊘  Not downloaded (requires {prov} authentication)")
            else:
                entry["note"] = "File not found – run download script"
                print(f"  ✗  MISSING: {filepath}")
                all_ok = False
        else:
            fsize = filepath.stat().st_size
            entry["file_size_bytes"] = fsize
            entry["file_size"] = sizeof_fmt(fsize)
            print(f"  📁 Size: {sizeof_fmt(fsize)}")

            validator = VALIDATORS.get(meta["format"])
            if validator:
                print(f"  🔍 Validating ({meta['format']})…")
                vresult = validator(filepath)
                entry.update(vresult)

                if vresult["status"] == "OK":
                    print(f"  ✓  Valid")
                    if "row_count" in vresult:
                        print(f"     Rows: {vresult['row_count']:,}")
                    if "crs" in vresult:
                        print(f"     CRS: {vresult['crs']}")
                else:
                    print(f"  ✗  ERRORS: {vresult['errors']}")
                    all_ok = False
            else:
                entry["status"] = "SKIPPED"
                print(f"  ⊘  No validator for format: {meta['format']}")

        inventory[key] = entry

    # ── Write JSON report ─────────────────────────────────────────
    json_path = REPORTS_DIR / "validation_results.json"
    report = {
        "generated_at": datetime.now().isoformat(),
        "datasets": inventory,
    }
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 JSON report → {json_path}")

    # ── Write Markdown inventory ──────────────────────────────────
    md_path = REPORTS_DIR / "dataset_inventory.md"
    _write_markdown_report(md_path, inventory)
    print(f"📄 Markdown report → {md_path}")

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    present = sum(1 for v in inventory.values() if v.get("status") != "MISSING")
    total = len(inventory)
    auth_pending = sum(1 for v in inventory.values()
                       if v.get("status") == "MISSING" and v.get("auth_required"))
    missing_public = sum(1 for v in inventory.values()
                         if v.get("status") == "MISSING" and not v.get("auth_required"))

    print(f"  Datasets present:         {present}/{total}")
    print(f"  Awaiting authentication:  {auth_pending}")
    if missing_public:
        print(f"  ⚠ Missing (public):       {missing_public}")
    if all_ok or (missing_public == 0):
        print("  ✓  All downloadable datasets validated successfully!")
    print("=" * 70)

    return inventory


def _write_markdown_report(md_path, inventory):
    lines = [
        "# NYC Micro-Climate Zoning AI – Dataset Inventory",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary Table",
        "",
        "| # | Dataset | Format | Status | Size | Rows | CRS |",
        "|---|---------|--------|--------|------|------|-----|",
    ]
    for i, (key, v) in enumerate(inventory.items(), 1):
        status = v.get("status", "Unknown")
        size = v.get("file_size", "—")
        rows = f"{v['row_count']:,}" if "row_count" in v else "—"
        crs = v.get("crs", v.get("sample_crs", "—"))
        if len(str(crs)) > 30:
            crs = str(crs)[:30] + "…"
        lines.append(
            f"| {i} | {v['description']} | {v['format']} | "
            f"{status} | {size} | {rows} | {crs} |"
        )

    lines += ["", "## Detailed Reports", ""]

    for key, v in inventory.items():
        lines.append(f"### {v['description']}")
        lines.append(f"- **Source:** {v['source']}")
        lines.append(f"- **File:** `data/raw/{v['filename']}`")
        lines.append(f"- **Format:** {v['format']}")
        lines.append(f"- **Status:** {v.get('status', 'Unknown')}")
        if v.get("file_size"):
            lines.append(f"- **Size:** {v['file_size']}")
        if "row_count" in v:
            lines.append(f"- **Row count:** {v['row_count']:,}")
        if "column_count" in v:
            lines.append(f"- **Columns:** {v['column_count']}")
        if v.get("crs") or v.get("sample_crs"):
            lines.append(f"- **CRS:** {v.get('crs', v.get('sample_crs'))}")
        if v.get("bounds"):
            lines.append(f"- **Bounds:** {v['bounds']}")

        mv = v.get("missing_values_sample", {})
        if mv:
            lines.append("- **Missing values (sample):**")
            for col, info in list(mv.items())[:10]:
                lines.append(f"  - `{col}`: {info['count']} ({info['pct']}%)")

        # Date ranges
        for dk, dv in v.items():
            if dk.startswith("date_range"):
                col_name = dk.replace("date_range_", "")
                lines.append(f"- **Date range ({col_name}):** {dv['min']} → {dv['max']}")
            elif dk.startswith("year_range"):
                col_name = dk.replace("year_range_", "")
                lines.append(f"- **Year range ({col_name}):** {dv['min']} → {dv['max']}")

        if v.get("note"):
            lines.append(f"- **Note:** {v['note']}")
        if v.get("errors"):
            lines.append(f"- **Errors:** {v['errors']}")
        lines.append("")

    lines += [
        "## Authentication-Required Datasets",
        "",
        "The following datasets require API credentials. "
        "See `SETUP.md` for configuration instructions.",
        "",
    ]
    for key, v in inventory.items():
        if v.get("auth_required"):
            prov = DATASETS[key].get("auth_provider", "Unknown")
            lines.append(f"- **{v['description']}** → {prov}")
    lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    validate_all()
