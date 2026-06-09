#!/usr/bin/env python3
"""
download_era5.py – Downloads ERA5 Hourly reanalysis from Copernicus CDS.

Variables: 2m temperature, 10m wind (u,v), surface solar & thermal radiation.

Prerequisites:
    pip install cdsapi
    Create ~/.cdsapirc with your CDS API key.

Usage:
    python scripts/download_era5.py
    python scripts/download_era5.py --year 2024 --months 6 7 8
"""
import sys, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import DATASETS, RAW_DIR, NYC_BBOX


def download_era5(year=2024, months=None):
    try:
        import cdsapi
    except ImportError:
        print("ERROR: pip install cdsapi"); sys.exit(1)

    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        print("ERROR: ~/.cdsapirc not found. See SETUP.md."); sys.exit(1)

    dest = RAW_DIR / DATASETS["era5"]["filename"]
    if dest.exists():
        print(f"  Already exists: {dest}"); return dest

    months = months or list(range(1, 13))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    c = cdsapi.Client()

    downloaded_files = []
    
    for m in months:
        month_dest = RAW_DIR / f"era5_nyc_hourly_{year}_{m:02d}.nc"
        downloaded_files.append(month_dest)
        
        if month_dest.exists():
            print(f"  Already downloaded: {month_dest.name}")
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "2m_temperature", 
                "2m_dewpoint_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "surface_pressure",
                "total_precipitation",
                "surface_solar_radiation_downwards",
                "surface_thermal_radiation_downwards",
            ],
            "year": [str(year)],
            "month": [f"{m:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [NYC_BBOX["north"], NYC_BBOX["west"],
                     NYC_BBOX["south"], NYC_BBOX["east"]],
        }
        print(f"  Downloading ERA5 for {year}-{m:02d}...")
        c.retrieve("reanalysis-era5-single-levels", request, str(month_dest))
        print(f"  Saved: {month_dest.name}")

    # Merge files if we downloaded them successfully
    print("  Extracting and merging monthly files into a single dataset...")
    try:
        import zipfile
        import xarray as xr
        import tempfile
        import shutil
        
        extracted_nc_files = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            
            # Extract all files
            for i, f in enumerate(downloaded_files):
                if zipfile.is_zipfile(f):
                    with zipfile.ZipFile(f, 'r') as zip_ref:
                        # Extract files, rename them to be unique to avoid overwriting
                        for member in zip_ref.namelist():
                            target_name = f"month_{i}_{member}"
                            extracted_path = tmp_path / target_name
                            with open(extracted_path, "wb") as out_f:
                                out_f.write(zip_ref.read(member))
                            extracted_nc_files.append(extracted_path)
                else:
                    # It might be an actual nc file if CDS didn't zip it
                    extracted_nc_files.append(f)
            
            print(f"    Found {len(extracted_nc_files)} NetCDF parts to merge...")
            ds = xr.open_mfdataset(extracted_nc_files, combine='by_coords')
            ds.to_netcdf(dest)
            ds.close()
            
        print(f"  Saved final merged file: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        
        # Cleanup partial zip files
        for f in downloaded_files:
            f.unlink()
            
    except ImportError:
        print("  xarray not installed. Cannot merge. Please pip install xarray")
        sys.exit(1)
    except Exception as e:
        print(f"  Error during merging: {e}")
        sys.exit(1)

    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--months", type=int, nargs="+", default=None)
    args = parser.parse_args()
    print("=" * 60)
    print("ERA5 Downloader")
    print("=" * 60)
    download_era5(year=args.year, months=args.months)
    print("Done.")
