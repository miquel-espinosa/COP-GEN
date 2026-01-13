#!/usr/bin/env python3
"""
check_s1rtc_shapes.py
---------------------------------
Quick utility to traverse a directory tree and identify GeoTIFF files that do
not match the expected Sentinel-1 RTC tile size of 1068 × 1068 pixels.

Usage
-----
python check_s1rtc_shapes.py /path/to/Core-S1RTC/ [--workers 8] [--out wrong_shapes_s1rtc.txt]

The script searches **recursively** for files ending in ``.tif`` (case-insensitive),
then uses a thread pool to open each file with *rasterio* and compare its
``width``/``height`` to 1068.  Any file that deviates is printed to stdout and
written to the specified output text file.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
from pathlib import Path
from typing import List, Optional

import rasterio

EXPECTED_SIZE = 1068  # both width and height should equal this value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locate GeoTIFFs with unexpected shape (should be 1068×1068).")
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory that will be searched recursively for *.tif files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of threads to use when checking file shapes (default: 8)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("wrong_shapes_s1rtc.txt"),
        help="File to write offending file paths to (default: wrong_shapes_s1rtc.txt)",
    )
    return parser.parse_args()


def get_all_tifs(root: Path) -> List[Path]:
    """Return a (sorted) list of all *.tif files under *root* (recursive)."""
    if not root.is_dir():
        print(f"❌ The provided path is not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    # Use rglob for recursive search; match both .tif and .TIF
    tifs = sorted(p for p in root.rglob("*.tif") if p.is_file())
    tifs.extend(sorted(p for p in root.rglob("*.TIF") if p.is_file()))
    return tifs


def check_shape(tif_path: Path) -> Optional[str]:
    """Return *tif_path* as str if its dimensions are not 1068×1068, else None."""
    try:
        with rasterio.open(tif_path) as src:
            if src.width != EXPECTED_SIZE or src.height != EXPECTED_SIZE:
                return str(tif_path)
    except Exception as err:
        # Treat any read error as a failure and record it as well
        return f"{tif_path}   ERROR: {err}"
    return None


def main() -> None:
    args = parse_args()

    tif_files = get_all_tifs(args.root)
    total = len(tif_files)
    print(f"🔍 Discovered {total} .tif files under {args.root}")

    bad_files: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx, result in enumerate(executor.map(check_shape, tif_files), start=1):
            # executor.map preserves order of *tif_files*; iterate to collect results
            if result is not None:
                print(result)
                bad_files.append(result)
            # Provide a minimal progress indicator every 1000 files
            if idx % 1000 == 0:
                print(f"✔ {idx}/{total} files checked …")

    # Write results to disk
    args.out.write_text("\n".join(bad_files))
    print(f"\n✅ Scan complete. {len(bad_files)} problematic files written to {args.out}")


if __name__ == "__main__":
    main() 