"""
Produce a publication-ready 512x512 thumbnail centered on a Majortom grid cell.

Usage example:
    python paper_figures/lat-lon-thumbnail.py 380U_1238R \
        --output /tmp/380U_1238R.png --zoom-deg 10
"""
import argparse
import json
import os
from typing import Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import requests
from PIL import Image
from shapely.geometry import box, shape

from scripts.grid import Grid

KOPPEN_URL = "https://raw.githubusercontent.com/rjerue/koppen-map/master/raw-data.json"
DEFAULT_DIST_KM = 10.0
FIGURE_SIZE_INCH = 4.0  # 4in at 128 DPI -> 512px square

# Pleasant Köppen color palette (matches other paper figures)
CLIMATE_COLORS = {
    "Arid": "#fb8072",
    "Tropical": "#b3de69",
    "Temperate": "#ffffb3",
    "Cold": "#bebada",
    "Polar": "#8dd3c7",
}
FALLBACK_CLIMATE_COLOR = "#e0e0e0"
BACKGROUND_COLOR = "#fdfbf7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a cropped climate-zone thumbnail around a Majortom grid cell."
    )
    parser.add_argument(
        "grid_cell",
        type=str,
        help="Majortom grid cell name (e.g., 380U_1238R).",
    )
    parser.add_argument(
        "--dist-km",
        type=float,
        default=DEFAULT_DIST_KM,
        help=f"Grid spacing in km. Default: {DEFAULT_DIST_KM}",
    )
    parser.add_argument(
        "--zoom-deg",
        type=float,
        default=10.0,
        help="Half-width of the map window in degrees.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path. Defaults to ./lat_lon_thumbnail_<grid_cell>.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=128,
        help="Figure DPI. 4in * 128dpi => 512px.",
    )
    parser.add_argument(
        "--koppen-path",
        type=str,
        default=None,
        help="Optional local path to the Köppen–Geiger JSON file.",
    )
    parser.add_argument(
        "--no-climate",
        action="store_true",
        help="Disable climate zone polygons (only coastlines + point).",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_coords",
        help="Print lat-lon coordinates and exit (no PNG generation).",
    )
    return parser.parse_args()


def ensure_koppen_json(koppen_path: Optional[str]) -> str:
    """Ensure the Köppen–Geiger JSON file exists locally, downloading if needed."""
    if koppen_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        koppen_path = os.path.join(script_dir, os.path.basename(KOPPEN_URL))

    if not os.path.exists(koppen_path):
        try:
            print(f"Downloading climate dataset to {koppen_path} ...")
            response = requests.get(KOPPEN_URL, timeout=120)
            response.raise_for_status()
            with open(koppen_path, "wb") as fh:
                fh.write(response.content)
            print("Download complete.")
        except Exception as exc:
            raise RuntimeError(f"Failed to download Köppen–Geiger dataset: {exc}")

    return koppen_path


def load_climate_gdf_from_json(koppen_json_path: str) -> gpd.GeoDataFrame:
    """Load the rjerue/koppen-map raw-data.json into a GeoDataFrame."""
    with open(koppen_json_path, "r") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or data.get("type") != "FeatureCollection":
        raise ValueError("Expected a GeoJSON-like FeatureCollection structure.")

    records = []
    for feat in data.get("features", []):
        geom = feat.get("geometry")
        props = feat.get("properties", {}) or {}
        if geom is None:
            continue
        try:
            geom_obj = shape(geom)
        except Exception:
            continue
        record = {"geometry": geom_obj}
        if "climate" in props:
            record["climate"] = props["climate"]
        for key, value in props.items():
            if key not in record:
                record[key] = value
        records.append(record)

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    def _koppen_group(code: str) -> str:
        if code in ("ET", "EF", "Af", "Am", "Aw"):
            return code
        if code.startswith("BW"):
            return "BW"
        if code.startswith("BS"):
            return "BS"
        if code.startswith("C"):
            return f"C{code[1]}" if len(code) >= 2 and code[1] in ("f", "w", "s") else "C"
        if code.startswith("D"):
            return f"D{code[1]}" if len(code) >= 2 and code[1] in ("f", "w", "s") else "D"
        return code[:1]

    labels = {
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
    major = {"A": "Tropical", "B": "Arid", "C": "Temperate", "D": "Cold", "E": "Polar"}

    gdf["koppen_code"] = gdf["climate"].str.split().str[0]
    gdf["koppen_group"] = gdf["koppen_code"].apply(_koppen_group)
    gdf["koppen_label"] = gdf["koppen_group"].map(labels).fillna(gdf["koppen_group"])
    gdf["koppen_major"] = gdf["koppen_code"].str[0].map(major)

    return gdf


def grid_cell_center(grid: Grid, grid_cell: str) -> Tuple[float, float]:
    """Return (lat, lon) of the grid cell centroid."""
    matches = grid.points[grid.points["name"] == grid_cell]
    if matches.empty:
        raise ValueError(f"Grid cell '{grid_cell}' not found for dist={grid.dist}km.")
    point = matches.iloc[0]
    bbox = grid.get_bounded_footprint(point)
    centroid = bbox.centroid
    return float(centroid.y), float(centroid.x)


def compute_extent(lat: float, lon: float, zoom_deg: float) -> Tuple[float, float, float, float]:
    """Compute lon/lat bounds around the target point."""
    lat_min = max(-89.5, lat - zoom_deg)
    lat_max = min(89.5, lat + zoom_deg)
    lon_min = lon - zoom_deg
    lon_max = lon + zoom_deg
    return lon_min, lon_max, lat_min, lat_max


def center_crop_square(image_path: str) -> None:
    """Center crop an image to a square and overwrite the file."""
    with Image.open(image_path) as img:
        width, height = img.size
        if width == height:
            return  # Already square
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        cropped = img.crop((left, top, right, bottom))
        cropped.save(image_path)


def plot_thumbnail(
    *,
    lat: float,
    lon: float,
    extent: Tuple[float, float, float, float],
    climate: Optional[gpd.GeoDataFrame],
    output_path: str,
    dpi: int,
) -> None:
    fig = plt.figure(figsize=(FIGURE_SIZE_INCH, FIGURE_SIZE_INCH), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection=ccrs.PlateCarree())
    ax.set_facecolor(BACKGROUND_COLOR)

    lon_min, lon_max, lat_min, lat_max = extent
    bbox_geom = box(lon_min, lat_min, lon_max, lat_max)

    if climate is not None and not climate.empty:
        climate = climate.copy()
        climate["plot_color"] = climate["koppen_major"].map(CLIMATE_COLORS).fillna(FALLBACK_CLIMATE_COLOR)
        try:
            target = gpd.clip(climate, bbox_geom)
            if target.empty:
                target = climate[climate.intersects(bbox_geom)]
        except Exception:
            target = climate[climate.intersects(bbox_geom)]
        if not target.empty:
            target.plot(
                ax=ax,
                color=target["plot_color"],
                edgecolor="#bdbdbd",
                linewidth=0.2,
                alpha=0.7,
                legend=False,
                zorder=1,
                transform=ccrs.PlateCarree(),
            )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=1.2,         # was 0.8
        edgecolor="#2e2e2e",
        zorder=4,
    )
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#2e2e2e", zorder=3)

    ax.gridlines(
        draw_labels=False,
        color="#8d8d8d",
        linestyle=":",
        linewidth=0.4,
        zorder=2,
        alpha=0.6,
    )

    ax.scatter(
        [lon],
        [lat],
        marker="*",
        s=520,                 # larger
        c="#c62828",
        edgecolors="white",
        linewidths=1.8,
        transform=ccrs.PlateCarree(),
        zorder=10,
    )


    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.set_axis_off()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(
        output_path,
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)

    # Center crop to ensure the image is perfectly square
    center_crop_square(output_path)

    print(f"Saved thumbnail to {output_path}")


def main() -> None:
    args = parse_args()
    grid = Grid(dist=args.dist_km)
    lat, lon = grid_cell_center(grid, args.grid_cell)

    if args.print_coords:
        print(f"{lat},{lon}")
        return

    print(f"Processing grid cell: {args.grid_cell}")
    extent = compute_extent(lat, lon, args.zoom_deg)

    if args.output is None:
        output_path = os.path.join(os.getcwd(), f"lat_lon_thumbnail_{args.grid_cell}.png")
    elif os.path.isdir(args.output) or args.output.endswith(os.sep):
        # If output is a directory, append the default filename
        os.makedirs(args.output, exist_ok=True)
        output_path = os.path.join(args.output, f"lat_lon_thumbnail_{args.grid_cell}.png")
    else:
        output_path = args.output

    climate_gdf = None
    if not args.no_climate:
        koppen_path = ensure_koppen_json(args.koppen_path)
        try:
            print("Loading climate dataset...")
            climate_gdf = load_climate_gdf_from_json(koppen_path)
        except Exception as exc:
            print(f"Warning: failed to load climate data ({exc}); continuing without it.")

    plot_thumbnail(
        lat=lat,
        lon=lon,
        extent=extent,
        climate=climate_gdf,
        output_path=output_path,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
