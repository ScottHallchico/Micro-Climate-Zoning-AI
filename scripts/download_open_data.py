#!/usr/bin/env python3
"""
download_open_data.py
─────────────────────
Downloads publicly available NYC datasets that require NO authentication:
  • MapPLUTO (Shoreline Clipped)
  • NYC 2015 Street Tree Census

Usage:
    source .venv/bin/activate
    python scripts/download_open_data.py          # download all
    python scripts/download_open_data.py mappluto # download one
"""

import sys
import time
import hashlib
import requests
from pathlib import Path

# ── resolve imports when run as script ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.config import DATASETS, RAW_DIR


# ── Helpers ──────────────────────────────────────────────────────────────────

def _sizeof_fmt(num_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def _md5(filepath: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(key: str, force: bool = False) -> Path:
    """
    Stream-download a single dataset to data/raw/.
    Returns the path to the saved file.
    """
    meta = DATASETS[key]
    url = meta["url"]
    dest = RAW_DIR / meta["filename"]

    if dest.exists() and not force:
        print(f"  ⊘  {meta['filename']} already exists ({_sizeof_fmt(dest.stat().st_size)}). Skipping.")
        return dest

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  ↓  Downloading {meta['description']}…")
    print(f"     URL: {url}")

    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r     {_sizeof_fmt(downloaded)} / {_sizeof_fmt(total)}  ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r     {_sizeof_fmt(downloaded)} downloaded", end="", flush=True)

    elapsed = time.time() - t0
    print(f"\n  ✓  Saved → {dest}  ({_sizeof_fmt(dest.stat().st_size)}, {elapsed:.1f}s)")
    print(f"     MD5: {_md5(dest)}")
    return dest


# ── Public download targets ─────────────────────────────────────────────────

PUBLIC_DATASETS = [k for k, v in DATASETS.items() if v["url"] is not None and not v["auth_required"]]


def download_all(force: bool = False):
    """Download every public (no-auth) dataset."""
    print("=" * 70)
    print("NYC Micro-Climate Zoning AI – Public Data Downloader")
    print("=" * 70)

    for key in PUBLIC_DATASETS:
        print(f"\n[{key}]")
        try:
            download_dataset(key, force=force)
        except Exception as exc:
            print(f"  ✗  FAILED: {exc}")

    print("\n" + "=" * 70)
    print("Done.  Run `python scripts/validate_datasets.py` to verify integrity.")
    print("=" * 70)


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    force = "--force" in sys.argv

    if target and target != "--force":
        if target not in DATASETS:
            print(f"Unknown dataset key: {target}")
            print(f"Available: {', '.join(PUBLIC_DATASETS)}")
            sys.exit(1)
        download_dataset(target, force=force)
    else:
        download_all(force=force)
