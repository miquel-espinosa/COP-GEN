import argparse
import csv
import os
import glob
import sys
from typing import List, Tuple, Optional

import json
import numpy as np
import torch
import requests
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape
import cartopy.crs as ccrs
import urllib.request
import tempfile
import time
import pyarrow.parquet as pq
from io import BytesIO
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local project imports (run from repository root or ensure repo root is on PYTHONPATH)
from ddm.pre_post_process_data import decode_lat_lon
from datasets import _LatLonMemmapCache
from scripts.grid import Grid
from majortom.download_world import is_valid_parquet
from majortom.metadata_helpers import filter_download


KOPPEN_URL = "https://raw.githubusercontent.com/rjerue/koppen-map/master/raw-data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize and compare lat-lon predictions for a grid cell (GT vs COPGEN vs Terramind) on a world climate zones map."
    )
    parser.add_argument(
        "grid_dir",
        type=str,
        help="Path to the grid directory (e.g., /path/to/143D_1481R) containing caches and outputs/ folders.",
    )
    parser.add_argument(
        "--koppen-path",
        type=str,
        default=None,
        help="Optional path to the Köppen–Geiger JSON file. If not provided, it will be saved/loaded next to this script.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the output figure. Defaults to <grid_dir>/outputs/lat_lon_comparison.png",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display the figure interactively.")
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the figure.",
    )
    parser.add_argument(
        "--vis-gridcells",
        action="store_true",
        help="If set, overlay grid_cell labels and download per-cell thumbnails for predictions.",
    )
    parser.add_argument(
        "--highlight",
        type=str,
        default=None,
        help="Grid cell id (e.g., 380U_1238R) to highlight among COPGEN predictions.",
    )
    return parser.parse_args()


def ensure_koppen_json(koppen_path: Optional[str]) -> str:
    """Ensure the Köppen–Geiger JSON file exists locally, downloading if necessary."""
    if koppen_path is None:
        # Save next to this script by default, using URL basename
        script_dir = os.path.dirname(os.path.abspath(__file__))
        koppen_path = os.path.join(script_dir, os.path.basename(KOPPEN_URL))

    if not os.path.exists(koppen_path):
        try:
            print(f"Downloading climate dataset to {koppen_path} ...")
            r = requests.get(KOPPEN_URL, timeout=120)
            r.raise_for_status()
            with open(koppen_path, "wb") as f:
                f.write(r.content)
            print("Download complete.")
        except Exception as e:
            raise RuntimeError(f"Failed to download Köppen–Geiger dataset: {e}")

    return koppen_path


def load_climate_gdf_from_json(koppen_json_path: str) -> gpd.GeoDataFrame:
    """Load the rjerue/koppen-map raw-data.json into a GeoDataFrame."""
    with open(koppen_json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON-like FeatureCollection structure in the JSON file.")
    features = data.get("features", [])
    records = []
    for feat in features:
        geom = feat.get("geometry")
        props = feat.get("properties", {}) or {}
        if geom is None:
            continue
        try:
            geom_obj = shape(geom)
        except Exception:
            continue
        record = {"geometry": geom_obj}
        # Normalize property name to 'climate' when available
        if "climate" in props:
            record["climate"] = props["climate"]
        # Preserve original props minimally
        for k, v in props.items():
            if k not in record:
                record[k] = v
        records.append(record)
    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    
    def _koppen_group(code: str) -> str:
        # Keep the meaningful groups; drop only the last warm/cold letter
        if code in ("ET", "EF", "Af", "Am", "Aw"):
            return code
        if code.startswith("BW"):
            return "BW"  # Arid-Desert
        if code.startswith("BS"):
            return "BS"  # Arid-Steppe
        if code.startswith("C"):
            # Cf/Cw/Cs families
            return f"C{code[1]}" if len(code) >= 2 and code[1] in ("f", "w", "s") else "C"
        if code.startswith("D"):
            # Df/Dw/Ds families
            return f"D{code[1]}" if len(code) >= 2 and code[1] in ("f", "w", "s") else "D"
        return code[:1]  # fallback to major group

    LABELS = {
        "Af": "Af Tropical-Rainforest",
        "Am": "Am Tropical-Monsoon",
        "Aw": "Aw Tropical-Savanna",
        "BW": "BW Arid-Desert",
        "BS": "BS Arid-Steppe",
        "Cf": "Cf Temperate (no dry season)",
        "Cw": "Cw Temperate (dry winter)",
        "Cs": "Cs Temperate (dry summer)",
        "Df": "Df Cold (no dry season)",
        "Dw": "Dw Cold (dry winter)",
        "Ds": "Ds Cold (dry summer)",
        "ET": "ET Polar-Tundra",
        "EF": "EF Polar-Frost",
    }

    # Extract code and coarsened groups
    gdf["koppen_code"] = gdf["climate"].str.split().str[0]
    gdf["koppen_group"] = gdf["koppen_code"].apply(_koppen_group)
    gdf["koppen_label"] = gdf["koppen_group"].map(LABELS).fillna(gdf["koppen_group"])
    
    MAJOR = {"A": "Tropical", "B": "Arid", "C": "Temperate", "D": "Cold", "E": "Polar"}
    gdf["koppen_major"] = gdf["koppen_code"].str[0].map(MAJOR)
    
    return gdf

def read_single_lat_lon_csv(csv_path: str, expected_tile_id: str) -> Optional[Tuple[float, float]]:
    """Read a lat_lon.csv file and return (lat, lon) for the expected tile_id.
    If multiple rows exist, prefer the one matching tile_id; otherwise fall back to the first data row.
    """
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if len(rows) == 0:
                return None
            # Prefer matching tile_id
            for row in rows:
                tile_id = row.get("tile_id", "").strip()
                if tile_id == expected_tile_id:
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                    return lat, lon
            # Fallback to first row
            lat = float(rows[0]["lat"])
            lon = float(rows[0]["lon"])
            return lat, lon
    except Exception:
        return None


def collect_predictions(root: str, tile_id: str) -> List[Tuple[float, float]]:
    """Collect all (lat, lon) predictions under root/*/lat_lon.csv."""
    lat_lons: List[Tuple[float, float]] = []
    pattern = os.path.join(root, "*", "lat_lon.csv")
    for csv_path in sorted(glob.glob(pattern)):
        pair = read_single_lat_lon_csv(csv_path, tile_id)
        if pair is not None:
            lat_lons.append(pair)
        else:
            print(f"No predictions found for {csv_path}")
    return lat_lons


def compute_gt_lat_lon(grid_dir: str, tile_id: str) -> Tuple[float, float]:
    """Compute ground-truth lat/lon from 3D cartesian cache using provided helpers."""
    cache_dir = os.path.join(grid_dir, "3d_cartesian_lat_lon_cache")
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(
            f"Could not find 3d_cartesian_lat_lon_cache at '{cache_dir}'."
        )

    cache = _LatLonMemmapCache(cache_dir)
    cartesian_np = cache[(tile_id, -1, -1)]  # (3,)
    cartesian_t = torch.tensor(cartesian_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    latlon_t = decode_lat_lon(cartesian_t, "3d_cartesian_lat_lon")  # (1,1,1,2)
    lat = float(latlon_t[0, 0, 0, 0].item())
    lon = float(latlon_t[0, 0, 0, 1].item())
    return lat, lon


def _scatter_with_halo(
    ax,
    lons: List[float],
    lats: List[float],
    color: str,
    marker: str,
    size: float,
    edgecolor: str = "k",
    linewidth: float = 0.6,
    zorder: float = 4,
    label: Optional[str] = None,
):
    """Plot points with a white halo underlay to improve visibility on busy backgrounds."""
    # Halo layer (white, slightly larger)
    # ax.scatter(
    #     lons,
    #     lats,
    #     marker=marker,
    #     s=size * 1.8,
    #     c="white",
    #     alpha=0.95,
    #     edgecolor="none",
    #     zorder=zorder - 0.1,
    #     label=None,
    # )
    # Foreground colored layer
    return ax.scatter(
        lons,
        lats,
        marker=marker,
        s=size,
        c=color,
        alpha=0.95,
        edgecolor=edgecolor,
        linewidths=linewidth,
        zorder=zorder,
        label=label,
    )


def _annotate_points(ax, lats: List[float], lons: List[float], color: str) -> None:
    """Add (lat, lon) labels next to points."""
    if len(lats) != len(lons):
        return
    for lat, lon in zip(lats, lons):
        ax.text(
            lon,
            lat + 0.25,  # slight northward offset to avoid overlap
            f"{lat:.2f},{lon:.2f}",
            fontsize=8,
            color=color,
            ha="center",
            va="bottom",
            zorder=6,
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
        )

def _annotate_cells(ax, lats: List[float], lons: List[float], grid: Grid, color: str) -> None:
    """Add grid_cell labels next to points."""
    if len(lats) != len(lons):
        return
    rows, cols = grid.latlon2rowcol(lats, lons)
    for lat, lon, r, c in zip(lats, lons, rows, cols):
        cell = f"{r}_{c}"
        ax.text(
            lon,
            lat + 0.25,
            cell,
            fontsize=8,
            color=color,
            ha="center",
            va="bottom",
            zorder=6,
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5),
        )


def _ensure_s2l2a_metadata(cache_dir: str) -> gpd.GeoDataFrame:
    """Load S2L2A metadata (cached locally)."""
    os.makedirs(cache_dir, exist_ok=True)
    dataset_name = "Core-S2L2A"
    parquet_url = f"https://huggingface.co/datasets/Major-TOM/{dataset_name}/resolve/main/metadata.parquet?download=true"
    local_parquet_path = os.path.join(cache_dir, f"{dataset_name}.parquet")
    # Cache-aware loading (avoid re-downloading if file exists)
    if os.path.exists(local_parquet_path):
        # Minimal columns are fine, but keep all for flexibility
        df = pq.read_table(local_parquet_path).to_pandas()
    else:
        print("Downloading S2L2A metadata...")
        local_parquet_path, _ = urllib.request.urlretrieve(parquet_url, local_parquet_path)
        df = pq.read_table(local_parquet_path).to_pandas()
    # Parse timestamp if present
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df['centre_lon'], df['centre_lat']),
        crs=df['crs'].iloc[0] if 'crs' in df.columns and len(df) > 0 else "EPSG:4326",
    )
    return gdf

def _center_crop(pil_img, size: int = 192):
    w, h = pil_img.size
    crop_w = min(size, w)
    crop_h = min(size, h)
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    right = left + crop_w
    bottom = top + crop_h
    return pil_img.crop((left, top, right, bottom))


def _save_thumbnails_for_predictions(
    latlons: List[Tuple[float, float]],
    out_dir: str,
    gdf_s2: gpd.GeoDataFrame,
    grid: Grid,
) -> None:
    """Fetch and save 192x192 center-cropped thumbnails for predicted points using metadata_helpers.filter_download."""
    if not latlons:
        return
    os.makedirs(out_dir, exist_ok=True)

    # 1) Map predicted lat/lon to grid_cells; group latlons by cell
    lats = [lat for lat, _ in latlons]
    lons = [lon for _, lon in latlons]
    rows, cols = grid.latlon2rowcol(lats, lons)
    cell_to_latlons: dict = {}
    for (lat, lon), r, c in zip(latlons, rows, cols):
        cell = f"{r}_{c}"
        cell_to_latlons.setdefault(cell, []).append((lat, lon))

    # 2) For each grid_cell, select a single S2L2A row (latest by timestamp if available)
    cell_to_target: dict = {}  # cell -> row (Series)
    for cell, _list in cell_to_latlons.items():
        subset = gdf_s2[gdf_s2["grid_cell"] == cell]
        if subset.empty:
            print(f"No S2L2A metadata found for grid cell {cell} ?")
            continue
        try:
            target_row = subset.sort_values("timestamp", ascending=False).iloc[0] if "timestamp" in subset.columns else subset.iloc[0]
            cell_to_target[cell] = target_row
        except Exception as e:
            print(f"Error selecting row for cell {cell}: {e}")

    if not cell_to_target:
        return

    # 3) Build a minimal dataframe with only the target rows (one per cell) and download thumbnails via helpers
    target_df = pd.DataFrame([cell_to_target[cell] for cell in cell_to_target])
    # Ensure we only keep required columns
    keep_cols = [c for c in ["grid_cell", "parquet_url", "parquet_row", "product_id", "timestamp"] if c in target_df.columns]
    target_df = target_df[keep_cols]

    # 4) Use metadata_helpers.filter_download to download thumbnails only (no bands)
    vis_root = os.path.dirname(out_dir)
    dl_root = os.path.join(vis_root, "s2_thumbnails_download")
    try:
        filter_download(target_df, local_dir=dl_root, source_name="Core-S2L2A", by_row=True, verbose=False, tif_columns=[])
    except Exception as e:
        print(f"Error during thumbnail download via helpers: {e}")

    # 5) For each grid_cell, copy + crop the downloaded thumbnail and save once per cell as <grid_cell>.png
    for cell, latlon_list in cell_to_latlons.items():
        if cell not in cell_to_target:
            continue
        target = cell_to_target[cell]
        product_id = str(target.get("product_id", "id"))
        row_code = cell.split("_")[0]
        thumb_path = os.path.join(dl_root, "Core-S2L2A", row_code, cell, product_id, "thumbnail.png")
        try:
            if not os.path.exists(thumb_path):
                print(f"Missing downloaded thumbnail for cell {cell} at {thumb_path}")
                continue
            img = Image.open(thumb_path)
            crop = _center_crop(img, size=192)
            out_path = os.path.join(out_dir, f"{cell}.png")
            if os.path.exists(out_path):
                print(f"Thumbnail already exists for cell {cell}")
                continue
            crop.save(out_path)
        except Exception as e:
            print(f"Error processing cell {cell}: {e}")


def plot_world_with_predictions(
    koppen_path: str,
    tile_id: str,
    gt_latlon: Tuple[float, float],
    copgen_latlons: List[Tuple[float, float]],
    terramind_latlons: List[Tuple[float, float]],
    save_path: str,
    title: Optional[str] = None,
    show: bool = True,
    annotate_cells: bool = False,
    highlight_cell: Optional[str] = None,
) -> None:
    print("Loading climate dataset...")
    climate = load_climate_gdf_from_json(koppen_path)

    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Plot the climate zones (biomes)
    column_name = "climate" if "climate" in climate.columns else ("zone" if "zone" in climate.columns else None)
    if column_name is not None:
        # Choose coarsened column and a discrete palette with enough unique colors
        # groups_col = "koppen_label"
        groups_col = "koppen_major"
        
        # Define custom color mapping for Köppen major climate zones
        color_map = {
            "Arid": "#fb8072",        # Salmon/Red
            "Tropical": "#b3de69",    # Light green
            "Temperate": "#ffffb3",   # Light Yellow
            "Cold": "#bebada",        # Light Purple
            "Polar": "#8dd3c7",       # blue
        }
        
        # Map colors to the climate data
        climate["plot_color"] = climate[groups_col].map(color_map)
        
        climate.plot(
            ax=ax,
            color=climate["plot_color"],
            alpha=0.65,
            edgecolor="#bdbdbd",
            linewidth=0.25,
            legend=False,
        )
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, edgecolor="#bdbdbd", label=label, alpha=0.60) 
                          for label, color in color_map.items() if label in climate[groups_col].values]
        ax.legend(handles=legend_elements, loc="lower left", fontsize=10, 
                 title="Köppen–Geiger (coarse)")
    else:
        climate.plot(ax=ax, color="#efefef", edgecolor="#bdbdbd", linewidth=0.15, legend=False, alpha=0.6)

    ax.coastlines(resolution='50m', color='#424242', linewidth=0.75, zorder=3)

    # Overlay predictions
    # Ground Truth (red star)
    gt_lat, gt_lon = gt_latlon
    _scatter_with_halo(
        ax=ax,
        lons=[gt_lon],
        lats=[gt_lat],
        color="#d62728",            # vivid red
        marker="*",
        size=220,
        edgecolor="k",
        linewidth=0.7,
        zorder=5,
        label="Ground Truth",
    )

    # COPGEN (dark green)
    if len(copgen_latlons) > 0:
        copgen_lats = [x[0] for x in copgen_latlons]
        copgen_lons = [x[1] for x in copgen_latlons]
        _scatter_with_halo(
            ax=ax,
            lons=copgen_lons,
            lats=copgen_lats,
            color="#00c853",         # vivid green
            marker="o",
            size=60,
            edgecolor="k",
            linewidth=0.6,
            zorder=4.5,
            label=f"COPGEN (n={len(copgen_latlons)})",
        )
        # Optional highlight of a specific grid cell among COPGEN predictions
        if highlight_cell:
            try:
                grid_for_highlight = Grid(10)
                rows_h, cols_h = grid_for_highlight.latlon2rowcol(copgen_lats, copgen_lons)
                cell_ids = [f"{r}_{c}" for r, c in zip(rows_h, cols_h)]
                idxs = [i for i, cid in enumerate(cell_ids) if cid == highlight_cell]
                for i in idxs:
                    hi_lat = copgen_lats[i]
                    hi_lon = copgen_lons[i]
                    # Draw an emphasized ring around the highlighted point
                    ax.scatter(
                        [hi_lon],
                        [hi_lat],
                        marker="o",
                        s=220,
                        facecolors="none",
                        edgecolors="#ffeb3b",  # bright yellow ring
                        linewidths=2.2,
                        zorder=6,
                        transform=ccrs.PlateCarree(),
                    )
                if len(idxs) == 0:
                    print(f"Warning: --highlight '{highlight_cell}' not found among COPGEN predictions.")
            except Exception as e:
                print(f"Highlight error: {e}")
        if annotate_cells:
            grid_for_labels = Grid(10)
            _annotate_cells(ax, copgen_lats, copgen_lons, grid_for_labels, color="#1b5e20")

    # Terramind (blue)
    if len(terramind_latlons) > 0:
        terr_lats = [x[0] for x in terramind_latlons]
        terr_lons = [x[1] for x in terramind_latlons]
        _scatter_with_halo(
            ax=ax,
            lons=terr_lons,
            lats=terr_lats,
            color="#2979ff",         # vivid blue
            marker="^",              # different shape for contrast
            size=60,
            edgecolor="k",
            linewidth=0.6,
            zorder=4.5,
            label=f"Terramind (n={len(terramind_latlons)})",
        )
        if annotate_cells:
            grid_for_labels = Grid(10)
            _annotate_cells(ax, terr_lats, terr_lons, grid_for_labels, color="#0d47a1")

    # Ax cosmetics
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_axis_off()

    # Title
    ax.set_title(
        title if title is not None else f"Latitude-Longitude Predictions for {tile_id}",
        fontsize=18,
    )

    # Legends: keep climate legend (already added), add a separate one for predictions
    climate_legend = ax.get_legend()
    pred_legend = ax.legend(loc="upper right", title="Predictions", frameon=True, markerscale=1.2)
    if climate_legend is not None:
        ax.add_artist(climate_legend)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    grid_dir = os.path.abspath(args.grid_dir)
    if not os.path.isdir(grid_dir):
        raise NotADirectoryError(f"grid_dir does not exist or is not a directory: {grid_dir}")

    tile_id = os.path.basename(os.path.normpath(grid_dir))
    print(f"Tile ID: {tile_id}")

    # Default save path
    save_path = args.save
    if save_path is None:
        if args.vis_gridcells:
            if args.highlight:
                save_path = os.path.join(grid_dir, "outputs", f"lat_lon_comparison_with_gridcells_{args.highlight}.png")
            else:
                save_path = os.path.join(grid_dir, "outputs", "lat_lon_comparison_with_gridcells.png")
        else:
            if args.highlight:
                save_path = os.path.join(grid_dir, "outputs", f"lat_lon_comparison_{args.highlight}.png")
            else:
                save_path = os.path.join(grid_dir, "outputs", "lat_lon_comparison.png")

    # Ensure climate zones data
    koppen_path = ensure_koppen_json(args.koppen_path)

    # Ground truth
    print("Computing ground-truth lat/lon from cache...")
    gt_latlon = compute_gt_lat_lon(grid_dir, tile_id)
    print(f"GT lat/lon: {gt_latlon}")

    # Predictions
    copgen_root = os.path.join(grid_dir, "outputs", "copgen")
    terramind_root = os.path.join(grid_dir, "outputs", "terramind")

    print("Collecting COPGEN predictions...")
    copgen_latlons = collect_predictions(copgen_root, tile_id) if os.path.isdir(copgen_root) else []
    print(f"Found {len(copgen_latlons)} COPGEN predictions")

    print("Collecting Terramind predictions...")
    terramind_latlons = collect_predictions(terramind_root, tile_id) if os.path.isdir(terramind_root) else []
    print(f"Found {len(terramind_latlons)} Terramind predictions")

    # Prepare visualization output directories
    vis_root = os.path.join(grid_dir, "outputs", "lat_lon_vis")
    copgen_vis_dir = os.path.join(vis_root, "copgen-lat-lon-vis")
    terramind_vis_dir = os.path.join(vis_root, "terramind-lat-lon-vis")
    os.makedirs(copgen_vis_dir, exist_ok=True)
    os.makedirs(terramind_vis_dir, exist_ok=True)

    if args.vis_gridcells:
        # Load S2L2A metadata (cached)
        try:
            print("Loading S2L2A metadata for thumbnail retrieval...")
            s2_gdf = _ensure_s2l2a_metadata(cache_dir=vis_root)
        except Exception as e:
            print(f"Warning: Failed to load S2L2A metadata ({e}). Skipping thumbnail downloads.")
            s2_gdf = None
    else:
        s2_gdf = None

    # Save thumbnails for predictions
    if s2_gdf is not None and args.vis_gridcells:
        grid = Grid(10)
        print("Saving COPGEN thumbnail visualizations...")
        _save_thumbnails_for_predictions(copgen_latlons, copgen_vis_dir, s2_gdf, grid)
        print("Saving Terramind thumbnail visualizations...")
        _save_thumbnails_for_predictions(terramind_latlons, terramind_vis_dir, s2_gdf, grid)

    # Plot
    plot_world_with_predictions(
        koppen_path=koppen_path,
        tile_id=tile_id,
        gt_latlon=gt_latlon,
        copgen_latlons=copgen_latlons,
        terramind_latlons=terramind_latlons,
        save_path=save_path,
        title=args.title,
        show=not args.no_show,
        annotate_cells=args.vis_gridcells,
        highlight_cell=args.highlight,
    )


if __name__ == "__main__":
    main()
