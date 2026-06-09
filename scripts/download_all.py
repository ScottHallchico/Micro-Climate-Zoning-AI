#!/usr/bin/env python3
"""
download_all.py – Master orchestrator for the data acquisition pipeline.

Downloads all publicly accessible datasets, then prints instructions
for datasets that require authentication.

Usage:
    source .venv/bin/activate
    python scripts/download_all.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.config import DATASETS, RAW_DIR
from scripts.download_open_data import download_all as download_public


def main():
    print("╔" + "═" * 68 + "╗")
    print("║  NYC Micro-Climate Zoning AI — Data Acquisition Pipeline          ║")
    print("╚" + "═" * 68 + "╝")

    # ── Phase 1: Public datasets ─────────────────────────────────────────
    print("\n┌─ Phase 1: Public Datasets (no auth required) ──────────────────┐")
    download_public()

    # ── Phase 2: Earth Engine (check availability) ───────────────────────
    print("\n┌─ Phase 2: Google Earth Engine ──────────────────────────────────┐")
    try:
        import ee
        ee.Initialize()
        print("  ✓  Earth Engine authenticated.")
        from scripts.download_earth_engine import download_landsat8, download_sentinel2
        print("\n  [landsat8]")
        download_landsat8(ee)
        print("\n  [sentinel2]")
        download_sentinel2(ee)
    except ImportError:
        print("  ⊘  earthengine-api not installed.")
        print("     → pip install -r requirements-remote.txt")
        print("     → earthengine authenticate")
        print("     → python scripts/download_earth_engine.py")
    except Exception as exc:
        print(f"  ⊘  Earth Engine not authenticated: {exc}")
        print("     → See SETUP.md §4 for configuration steps.")

    # ── Phase 3: ERA5 (check availability) ───────────────────────────────
    print("\n┌─ Phase 3: ERA5 Reanalysis (Copernicus CDS) ────────────────────┐")
    try:
        import cdsapi
        rc = Path.home() / ".cdsapirc"
        if rc.exists():
            print("  ✓  CDS API configured.")
            from scripts.download_era5 import download_era5
            download_era5()
        else:
            raise FileNotFoundError("~/.cdsapirc not found")
    except ImportError:
        print("  ⊘  cdsapi not installed.")
        print("     → pip install -r requirements-remote.txt")
        print("     → See SETUP.md §5 for CDS API setup.")
    except Exception as exc:
        print(f"  ⊘  CDS API not configured: {exc}")
        print("     → See SETUP.md §5 for configuration steps.")

    # ── Phase 4: Validate ────────────────────────────────────────────────
    print("\n┌─ Phase 4: Validation ──────────────────────────────────────────┐")
    from scripts.validate_datasets import validate_all
    validate_all()

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n╔" + "═" * 68 + "╗")
    print("║  Pipeline Complete                                              ║")
    print("╚" + "═" * 68 + "╝")

    present = []
    missing = []
    for key, meta in DATASETS.items():
        fp = RAW_DIR / meta["filename"]
        if fp.exists():
            present.append(key)
        else:
            missing.append(key)

    print(f"\n  Downloaded: {len(present)}/{len(DATASETS)}")
    for k in present:
        print(f"    ✓ {k}")
    if missing:
        print(f"\n  Pending ({len(missing)}):")
        for k in missing:
            prov = DATASETS[k].get("auth_provider", "manual download")
            print(f"    ⊘ {k} → requires {prov}")
        print("\n  See SETUP.md for authentication instructions.")


if __name__ == "__main__":
    main()
