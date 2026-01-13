import argparse
import os
import pickle
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from ddm.pre_post_process_data import encode_date


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def load_cells(txt_path: str) -> List[str]:
    """Return non-empty, stripped lines from *txt_path*."""
    with open(txt_path, "r") as f:
        cells = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(cells)} grid-cell labels")
    return cells


def ensure_unique_grid_cells(df: pd.DataFrame, label: str) -> None:
    """Raise an error if *df* contains multiple rows for the same grid_cell."""
    dup_cells = (
        df.loc[df.duplicated(subset=["grid_cell"], keep=False), "grid_cell"].unique()
    )
    if dup_cells.size > 0:
        sample = ", ".join(dup_cells[:10])
        more = "…" if dup_cells.size > 10 else ""
        raise ValueError(
            f"Found {dup_cells.size} grid_cells with duplicate rows in {label}: {sample}{more}"
        )


def remove_duplicate_grid_cells(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Remove duplicate grid_cells from dataframe, keeping the first occurrence.
    
    This is useful when working with full metadata that may contain multiple
    observations per grid cell, but we only need one representative timestamp
    per cell (e.g., the earliest one).
    """
    initial_count = len(df)
    dup_cells = df.duplicated(subset=["grid_cell"], keep="first")
    
    if dup_cells.any():
        num_duplicates = dup_cells.sum()
        print(f"Found {num_duplicates} duplicate grid_cells in {label}, keeping first occurrence")
        df = df[~dup_cells].copy()
        print(f"Reduced {label} from {initial_count} to {len(df)} rows")
    
    return df


def is_leap_year(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


# def decode_date(sin_doy, cos_doy, year_norm, min_year=1950, max_year=2050):
#     """
#     This function is used in the pre_post_process_data.py file to decode the encoded date back to a datetime object.
#     """
#     # Decode year
#     year = ((year_norm + 1) / 2) * (max_year - min_year) + min_year
#     year = int(round(year))

#     days_in_year = 366 if is_leap_year(year) else 365

#     # Decode day of year
#     angle = np.arctan2(sin_doy, cos_doy)
#     if angle < 0:
#         angle += 2 * np.pi

#     doy = int(round(angle * days_in_year / (2 * np.pi))) % days_in_year
#     if doy == 0:
#         doy = days_in_year

#     decoded_date = datetime(year, 1, 1) + timedelta(days=doy - 1)
#     return decoded_date


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------


def compute_mean_timestamps(
    cells: List[str],
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
) -> Tuple[Dict[str, date], List[int]]:
    """Compute the per-cell mean date between Sentinel-1 and Sentinel-2 captures.

    Returns (mapping, day_diffs) where `mapping` is a mapping ``grid_cell → date``
    storing only *day-month-year* (no time-of-day) and `day_diffs` contains the
    absolute day differences per processed cell.
    """
    mapping: Dict[str, date] = {}
    diffs_days: List[int] = []

    # Earliest timestamp per cell (fast, one pass)
    s1_min = s1_df.groupby("grid_cell", sort=False)["timestamp"].min()
    s2_min = s2_df.groupby("grid_cell", sort=False)["timestamp"].min()

    # Combine in a single dataframe
    merged = pd.concat({"s1": s1_min, "s2": s2_min}, axis=1, join="inner").dropna()

    # Keep only the cells in common_cells.txt (if that list is smaller)
    merged = merged.loc[merged.index.intersection(cells)]

    for cell in tqdm(cells):
        s1_ts = merged.loc[cell, "s1"]
        s2_ts = merged.loc[cell, "s2"]

        if pd.isna(s1_ts) or pd.isna(s2_ts):
            # Skip cells missing data in either modality
            print(f"Skipping cell {cell} because it is missing data in either modality")
            continue

        # Ensure pandas.Timestamp for both
        if not isinstance(s1_ts, pd.Timestamp):
            s1_ts = pd.to_datetime(s1_ts)
        if not isinstance(s2_ts, pd.Timestamp):
            s2_ts = pd.to_datetime(s2_ts)

        # Compute absolute difference in days
        diff_days = abs((s2_ts - s1_ts).days)
        diffs_days.append(diff_days)

        # Mid-point timestamp (mean of the two)
        mean_ts = s1_ts + (s2_ts - s1_ts) / 2

        # Store only date component (day-month-year)
        mapping[cell] = encode_date(mean_ts.date(), add_spatial_dims=False)

    return mapping, diffs_days


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        "Pre-compute per-grid-cell mean timestamps between Sentinel-1 RTC and Sentinel-2 L2A observations"
    )
    parser.add_argument("--s1_metadata", default=None, help="Path to S1RTC metadata.parquet (optional, will download if not provided)")
    parser.add_argument("--s2_metadata", default=None, help="Path to S2L2A metadata.parquet (optional, will download if not provided)")
    parser.add_argument("--grid_cells_txt", required=True, help="Path to common grid-cells txt file")
    parser.add_argument("--output", required=True, help="Where to save the pickle mapping {cell → date}")
    parser.add_argument("--histogram", default="time_diffs_hist.png", help="Output path for histogram image")
    args = parser.parse_args()

    print("Loading grid cells…")
    cells = load_cells(args.grid_cells_txt)

    print("Reading metadata parquet files…")
    
    # Handle S1 metadata
    if args.s1_metadata:
        s1_df = pd.read_parquet(args.s1_metadata, columns=["grid_cell", "timestamp"])
    else:
        print("No S1 metadata provided, downloading from HuggingFace...")
        access_url = "https://huggingface.co/datasets/Major-TOM/Core-S1RTC/resolve/main/metadata.parquet?download=true"
        local_url = Path("./data/majorTOM/Core-S1RTC/metadata.parquet")
        local_url.parent.mkdir(exist_ok=True, parents=True)
        
        if local_url.exists():
            print("Using cached S1 metadata")
            s1_df = pd.read_parquet(local_url, columns=["grid_cell", "timestamp"])
        else:
            print("Downloading S1 metadata...")
            import urllib.request
            local_url, response = urllib.request.urlretrieve(access_url, local_url)
            s1_df = pd.read_parquet(local_url, columns=["grid_cell", "timestamp"])
    
    # Handle S2 metadata
    if args.s2_metadata:
        s2_df = pd.read_parquet(args.s2_metadata, columns=["grid_cell", "timestamp"])
    else:
        print("No S2 metadata provided, downloading from HuggingFace...")
        access_url = "https://huggingface.co/datasets/Major-TOM/Core-S2L2A/resolve/main/metadata.parquet?download=true"
        local_url = Path("./data/majorTOM/Core-S2L2A/metadata.parquet")
        local_url.parent.mkdir(exist_ok=True, parents=True)
        
        if local_url.exists():
            print("Using cached S2 metadata")
            s2_df = pd.read_parquet(local_url, columns=["grid_cell", "timestamp"])
        else:
            print("Downloading S2 metadata...")
            import urllib.request
            local_url, response = urllib.request.urlretrieve(access_url, local_url)
            s2_df = pd.read_parquet(local_url, columns=["grid_cell", "timestamp"])

    # Ensure timestamps are datetime64
    s1_df["timestamp"] = pd.to_datetime(s1_df["timestamp"])
    s2_df["timestamp"] = pd.to_datetime(s2_df["timestamp"])

    # ----------------------------------------------------------------------
    # Remove duplicates (keep first occurrence per grid_cell)
    # ----------------------------------------------------------------------
    s1_df = remove_duplicate_grid_cells(s1_df, "S1 metadata")
    s2_df = remove_duplicate_grid_cells(s2_df, "S2 metadata")

    # ----------------------------------------------------------------------
    # Uniqueness check
    # ----------------------------------------------------------------------
    ensure_unique_grid_cells(s1_df, "S1 metadata")
    ensure_unique_grid_cells(s2_df, "S2 metadata")
    print("No duplicates per grid-cell in S1 and S2 metadata.")
    
    print("Computing mean timestamps…")
    mapping, diffs_days = compute_mean_timestamps(cells, s1_df, s2_df)

    if not mapping:
        print("Warning: No grid-cells had timestamps in both datasets. Nothing to save.")
        return

    print(f"Computed mean timestamps for {len(mapping)} grid-cells")

    # Save pickle mapping
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(mapping, f)
    print(f"Saved mapping → {args.output}")

    # Plot and save histogram of day differences
    plt.figure(figsize=(8, 5))
    plt.hist(diffs_days, bins=150, color="steelblue", edgecolor="black")
    plt.xlabel("|Δ days| between S1 and S2 observations")
    plt.ylabel("Count of grid-cells")
    plt.title("Distribution of observation time differences")
    plt.yscale('log')
    plt.xscale('log')
    # Add more x-axis labels
    ax = plt.gca()
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    # Set custom tick locations to include 5 and 50
    ax.xaxis.set_major_locator(plt.FixedLocator([1, 5, 10, 50, 100, 1000]))
    # Add minor ticks between major ticks
    ax.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    # Set x-axis to start at a small positive value (can't be exactly 0 with log scale)
    plt.xlim(left=1)
    plt.tight_layout()
    plt.savefig(args.histogram, dpi=150)
    print(f"Histogram saved → {args.histogram}")


if __name__ == "__main__":
    main() 