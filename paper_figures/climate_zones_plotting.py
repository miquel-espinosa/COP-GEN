"""
Plot generated lat/lon predictions for each LULC class on a world map.

Expected input layout (folder names are lowercased and stripped):
    climate_zones/
        water/
        trees/
        flooded vegetation/
        crops/
        built-up areas/
        bare ground/
        clouds/
        rangeland/

Within each class directory we look for:
    outputs/copgen/input_LULC_output_DEM_S1RTC_S2L1C_S2L2A_cloud_mask_lat_lon_timestamps_seed_*/lat_lon.csv
"""
import argparse
import colorsys
import csv
import glob
import json
import os
import zipfile
from typing import Dict, List, Tuple, Optional

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject, calculate_default_transform
import requests
from shapely.geometry import shape


# Köppen–Geiger source (same as lat_lon_comparison.py)
KOPPEN_URL = "https://raw.githubusercontent.com/rjerue/koppen-map/master/raw-data.json"

# Natural Earth water body sources
WATER_OCEAN_URL = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_ocean.zip"
WATER_LAKES_URL = "https://naturalearth.s3.amazonaws.com/50m_physical/ne_50m_lakes.zip"

# NASA topo/bathy raster used for mountain basemap
MOUNTAIN_BASEMAP_URL = (
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/"
    "world.topo.bathy.200412.3x21600x10800.jpg"
)
MOUNTAIN_IMAGE_PIXEL_LIMIT = 400_000_000

# Global Percent Tree Cover tiles (Global Map ver 2)
TREECOVER_TILE_BASE_URL = "https://github.com/globalmaps/gm_ve_v2/raw/master"
TREECOVER_TILE_FILES = [
    "gm_ve_v2_1_1.zip",
    "gm_ve_v2_1_2.zip",
    "gm_ve_v2_1_3.zip",
    "gm_ve_v2_1_4.zip",
    "gm_ve_v2_1_5.zip",
    "gm_ve_v2_1_6.zip",
    "gm_ve_v2_2_1.zip",
    "gm_ve_v2_2_2.zip",
    "gm_ve_v2_2_3.zip",
    "gm_ve_v2_2_4.zip",
    "gm_ve_v2_2_5.zip",
    "gm_ve_v2_2_6.zip",
]

# WorldPop global population density raster (2020, 1km resolution, ~1GB)
# Users can provide their own raster via --population-raster-path
POPULATION_RASTER_URL = (
    "https://data.worldpop.org/GIS/Population_Density/Global_2000_2020_1km/"
    "2020/0_Mosaicked/gpw_v4_population_density_rev11_2020_30_sec.tif"
)
# Fallback: smaller SEDAC GPW v4 2.5 arc-minute (~5km) resolution
POPULATION_RASTER_URL_FALLBACK = (
    "https://sedac.ciesin.columbia.edu/downloads/data/gpw-v4/"
    "gpw-v4-population-density-rev11/gpw-v4-population-density-rev11_2020_2pt5_min_tif.zip"
)

# Unified LULC classes with CLI-friendly keys (no spaces)
# Maps: cli_key -> {label, color, copgen_folder, terramind_folder}
LULC_CLASSES_UNIFIED = [
    {
        "key": "water",
        "label": "Water",
        "color": (26 / 255, 91 / 255, 171 / 255),
        "copgen_folder": "water",
        "terramind_folder": "water",
    },
    {
        "key": "trees",
        "label": "Trees",
        "color": (53 / 255, 130 / 255, 33 / 255),
        "copgen_folder": "trees",
        "terramind_folder": "trees",
    },
    {
        "key": "flooded_vegetation",
        "label": "Flooded vegetation",
        "color": (135 / 255, 209 / 255, 158 / 255),
        "copgen_folder": "flooded vegetation",
        "terramind_folder": "flooded vegetation",
    },
    {
        "key": "crops",
        "label": "Crops",
        "color": (255 / 255, 219 / 255, 92 / 255),
        "copgen_folder": "crops",
        "terramind_folder": "crops",
    },
    {
        "key": "built_area",
        "label": "Built-up areas",
        "color": (237 / 255, 2 / 255, 42 / 255),
        "copgen_folder": "built-up areas",
        "terramind_folder": "built area",
    },
    {
        "key": "bare_ground",
        "label": "Bare ground",
        "color": (227 / 255, 226 / 255, 195 / 255),
        "copgen_folder": "bare ground",
        "terramind_folder": "bare ground",
    },
    {
        "key": "snow_ice",
        "label": "Snow/Ice",
        "color": (168 / 255, 235 / 255, 255 / 255),
        "copgen_folder": "snow_ice",
        "terramind_folder": "snow_ice",
    },
    {
        "key": "clouds",
        "label": "Clouds",
        "color": (97 / 255, 97 / 255, 97 / 255),
        "copgen_folder": "clouds",
        "terramind_folder": "clouds",
    },
    {
        "key": "rangeland",
        "label": "Rangeland",
        "color": (165 / 255, 155 / 255, 143 / 255),
        "copgen_folder": "rangeland",
        "terramind_folder": "rangeland",
    },
]

# Build lookup by CLI key
LULC_CLASS_BY_KEY = {c["key"]: c for c in LULC_CLASSES_UNIFIED}

# Model markers for comparison mode
# hue_offset: fraction of hue wheel to rotate (0.0 = no change, 0.5 = complementary)
MODEL_MARKERS = {
    "copgen": {"marker": "o", "label": "CopGen", "hue_offset": 0.0},
    "terramind": {"marker": "^", "label": "TerraMind", "hue_offset": 0.15},
}


def _shift_hue(
    rgb_tuple: Tuple[float, float, float], hue_offset: float
) -> Tuple[float, float, float]:
    """Shift the hue of an RGB color by a specified fraction of the hue wheel.

    Args:
        rgb_tuple: RGB color as (r, g, b) with values in [0, 1].
        hue_offset: Fraction of hue wheel to rotate (0.0-1.0, e.g., 0.5 = complementary).

    Returns:
        New RGB color tuple with shifted hue.
    """
    if hue_offset == 0.0:
        return rgb_tuple
    r, g, b = rgb_tuple
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + hue_offset) % 1.0
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
    return (r_new, g_new, b_new)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot generated lat/lon predictions for each LULC class on a world map."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="climate_zones",
        help="Path to the climate_zones root directory.",
    )
    parser.add_argument(
        "--koppen-path",
        type=str,
        default=None,
        help="Optional path to the Köppen–Geiger JSON file. If not provided, it will be downloaded next to this script.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the output figure. Defaults to <root>/lat_lon_by_lulc.png",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the figure.",
    )
    parser.add_argument(
        "--basemap",
        type=str,
        choices=("climates", "mountains", "population", "treecover"),
        default="climates",
        help="Background basemap to render: climates (default), mountains, population, or treecover.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="copgen",
        help="Model to use for collecting predictions.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional list of LULC class keys to include (default: all classes). "
            "Valid keys: water, trees, flooded_vegetation, crops, built_area, "
            "bare_ground, snow_ice, clouds, rangeland."
        ),
    )
    parser.add_argument(
        "--climate-alpha",
        type=float,
        default=0.65,
        help="Alpha transparency (0-1) for the climate zone polygons.",
    )
    parser.add_argument(
        "--scatter-size",
        type=float,
        default=48.0,
        help="Marker size (Matplotlib 's' parameter) for scatter points.",
    )
    parser.add_argument(
        "--water-ocean-path",
        type=str,
        default=None,
        help="Path to an ocean polygon shapefile (defaults to downloading Natural Earth 50m oceans).",
    )
    parser.add_argument(
        "--water-lakes-path",
        type=str,
        default=None,
        help="Path to a freshwater polygon shapefile (defaults to downloading Natural Earth 50m lakes).",
    )
    parser.add_argument(
        "--water-alpha",
        type=float,
        default=1,
        help="Alpha transparency (0-1) for the water body polygons.",
    )
    parser.add_argument(
        "--no-oceans",
        action="store_true",
        help="Disable plotting ocean water body polygons.",
    )
    parser.add_argument(
        "--no-lakes",
        action="store_true",
        help="Disable plotting lake/freshwater body polygons.",
    )
    parser.add_argument(
        "--mountain-image-path",
        type=str,
        default=None,
        help="Custom path to a mountain basemap image (defaults to NASA world topo/bathy).",
    )
    parser.add_argument(
        "--mountain-alpha",
        type=float,
        default=0.85,
        help="Alpha transparency (0-1) for the mountain basemap image.",
    )
    parser.add_argument(
        "--mountain-max-size",
        type=int,
        default=4096,
        help="Maximum pixel dimension for the mountain basemap (<=0 keeps original).",
    )
    parser.add_argument(
        "--legend-position",
        type=str,
        choices=("top-right", "top-left", "bottom-right", "bottom-left"),
        default="top-right",
        help="Choose where to place the LULC legend.",
    )
    parser.add_argument(
        "--population-alpha",
        type=float,
        default=0.8,
        help="Alpha transparency (0-1) for the population density basemap.",
    )
    parser.add_argument(
        "--population-cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap name for the population density basemap.",
    )
    parser.add_argument(
        "--population-max-density",
        type=float,
        default=None,
        help="Optional manual upper bound (people/km^2) for population density scaling.",
    )
    parser.add_argument(
        "--population-raster-path",
        type=str,
        default=None,
        help=(
            "Path to a gridded population density GeoTIFF raster. "
            "If not provided, attempts to download WorldPop global data (large file). "
            "Recommended sources: WorldPop, SEDAC GPW v4, or GHS-POP."
        ),
    )
    parser.add_argument(
        "--population-res-deg",
        type=float,
        default=0.25,
        help="Approximate degree resolution for resampling population raster (e.g., 0.25 = ~25km).",
    )
    parser.add_argument(
        "--population-cache-dir",
        type=str,
        default=None,
        help="Optional directory to cache population raster data (default: alongside this script).",
    )
    parser.add_argument(
        "--treecover-alpha",
        type=float,
        default=0.8,
        help="Alpha transparency (0-1) for the tree cover basemap.",
    )
    parser.add_argument(
        "--treecover-res-deg",
        type=float,
        default=0.5,
        help="Approximate degree resolution for the tree cover mosaic (e.g., 0.5).",
    )
    parser.add_argument(
        "--treecover-cache-dir",
        type=str,
        default=None,
        help="Optional directory to cache tree cover tiles/mosaics (default: alongside this script).",
    )
    parser.add_argument(
        "--comparison",
        action="store_true",
        help="Enable comparison mode to plot both CopGen and TerraMind predictions on the same map.",
    )
    parser.add_argument(
        "--copgen-root",
        type=str,
        default=None,
        help="Path to the CopGen climate_zones root directory (used in --comparison mode).",
    )
    parser.add_argument(
        "--terramind-root",
        type=str,
        default=None,
        help="Path to the TerraMind climate_zones root directory (used in --comparison mode).",
    )
    parser.add_argument(
        "--terramind-hue-offset",
        type=float,
        default=None,
        help=(
            "Hue offset (0-1) to apply to TerraMind colors in comparison mode. "
            "If not specified, uses tab10 colormap for distinct model colors. "
            "0.0 = same colors as CopGen, 0.5 = complementary colors."
        ),
    )
    return parser.parse_args()


def ensure_koppen_json(koppen_path: Optional[str]) -> str:
    """Ensure the Köppen–Geiger JSON file exists locally, downloading if necessary."""
    if koppen_path is None:
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


def _ensure_vector_dataset(
    description: str, provided_path: Optional[str], fallback_url: str
) -> str:
    """Ensure a vector dataset exists locally, downloading if needed."""
    if provided_path is not None:
        if not os.path.exists(provided_path):
            raise FileNotFoundError(f"{description} path does not exist: {provided_path}")
        return provided_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, os.path.basename(fallback_url))
    if not os.path.exists(local_path):
        try:
            print(f"Downloading {description} dataset to {local_path} ...")
            r = requests.get(fallback_url, timeout=180)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            print("Download complete.")
        except Exception as exc:
            raise RuntimeError(f"Failed to download {description}: {exc}")
    return local_path


def ensure_mountain_image(provided_path: Optional[str]) -> str:
    """Ensure the mountain basemap image is available locally."""
    if provided_path is not None:
        if not os.path.exists(provided_path):
            raise FileNotFoundError(
                f"Mountain basemap image path does not exist: {provided_path}"
            )
        return provided_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, os.path.basename(MOUNTAIN_BASEMAP_URL))
    if not os.path.exists(local_path):
        try:
            print(f"Downloading mountain basemap to {local_path} ...")
            r = requests.get(MOUNTAIN_BASEMAP_URL, timeout=300)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            print("Download complete.")
        except Exception as exc:
            raise RuntimeError(f"Failed to download mountain basemap image: {exc}")
    return local_path


def load_mountain_image(image_path: str, max_size: Optional[int]) -> np.ndarray:
    """Load the mountain basemap image and optionally downscale it."""
    try:
        if hasattr(Image, "MAX_IMAGE_PIXELS"):
            current_limit = Image.MAX_IMAGE_PIXELS
            if current_limit is not None and current_limit < MOUNTAIN_IMAGE_PIXEL_LIMIT:
                Image.MAX_IMAGE_PIXELS = MOUNTAIN_IMAGE_PIXEL_LIMIT
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if max_size is not None and max_size > 0:
                longest = max(img.size)
                if longest > max_size:
                    scale = max_size / float(longest)
                    new_size = (
                        max(1, int(round(img.width * scale))),
                        max(1, int(round(img.height * scale))),
                    )
                    resampling = getattr(Image, "Resampling", None)
                    if resampling is not None:
                        img = img.resize(new_size, resampling.LANCZOS)
                    else:
                        img = img.resize(new_size, Image.LANCZOS)
            return np.asarray(img)
    except Exception as exc:
        raise RuntimeError(f"Failed to load mountain basemap image: {exc}")


def _download_file(url: str, dest_path: str, chunk_size: int = 1 << 20) -> None:
    """Download a remote file to disk with streaming."""
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)


def _resolve_treecover_cache_dir(cache_dir: Optional[str]) -> str:
    if cache_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "treecover_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _ensure_treecover_tiles(cache_dir: Optional[str]) -> List[str]:
    """Download and extract Global Map tree cover tiles when missing."""
    resolved_dir = _resolve_treecover_cache_dir(cache_dir)
    tif_paths: List[str] = []
    for filename in TREECOVER_TILE_FILES:
        zip_path = os.path.join(resolved_dir, filename)
        if not os.path.exists(zip_path):
            url = f"{TREECOVER_TILE_BASE_URL}/{filename}"
            print(f"Downloading tree cover tile {filename} ...")
            try:
                _download_file(url, zip_path)
            except Exception as exc:
                raise RuntimeError(f"Failed to download {filename}: {exc}")
        tif_name = filename.replace(".zip", ".tif")
        tif_path = os.path.join(resolved_dir, tif_name)
        if not os.path.exists(tif_path):
            print(f"Extracting {tif_name} ...")
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extract(tif_name, resolved_dir)
            except Exception as exc:
                raise RuntimeError(f"Failed to extract {filename}: {exc}")
        tif_paths.append(tif_path)
    return tif_paths


def load_treecover_array(
    cache_dir: Optional[str], target_res_deg: float
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Load or build a coarse global percent tree cover raster."""
    if target_res_deg <= 0:
        raise ValueError("--treecover-res-deg must be greater than zero.")
    resolved_dir = _resolve_treecover_cache_dir(cache_dir)
    cache_key = f"{target_res_deg:.3f}".replace(".", "p")
    cache_file = os.path.join(resolved_dir, f"treecover_res_{cache_key}.npz")
    if os.path.exists(cache_file):
        with np.load(cache_file) as data:
            return data["image"], tuple(float(v) for v in data["extent"])

    tif_paths = _ensure_treecover_tiles(resolved_dir)
    datasets = [rasterio.open(path) for path in tif_paths]
    try:
        mosaic, transform = merge(
            datasets,
            bounds=(-180.0, -90.0, 180.0, 90.0),
            res=target_res_deg,
            nodata=255,
            resampling=Resampling.bilinear,
        )
    finally:
        for ds in datasets:
            ds.close()

    arr = mosaic[0]
    arr = np.where(arr >= 254, np.nan, arr)
    arr = np.clip(arr, 0, 100)
    extent = (-180.0, 180.0, -90.0, 90.0)
    np.savez_compressed(cache_file, image=arr, extent=np.array(extent, dtype=np.float32))
    return arr, extent


def _resolve_population_cache_dir(cache_dir: Optional[str]) -> str:
    """Resolve and create the population raster cache directory."""
    if cache_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(script_dir, "population_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _ensure_population_raster(
    cache_dir: Optional[str], raster_path: Optional[str]
) -> str:
    """Ensure population raster is available, downloading if necessary.

    Args:
        cache_dir: Directory for caching downloaded data.
        raster_path: User-provided path to a population raster. If None, attempts download.

    Returns:
        Path to the population density GeoTIFF.
    """
    if raster_path is not None:
        if not os.path.exists(raster_path):
            raise FileNotFoundError(f"Population raster not found: {raster_path}")
        return raster_path

    resolved_dir = _resolve_population_cache_dir(cache_dir)
    # Check for any existing .tif files in cache
    existing_tifs = glob.glob(os.path.join(resolved_dir, "*.tif"))
    if existing_tifs:
        print(f"Using cached population raster: {existing_tifs[0]}")
        return existing_tifs[0]

    # Attempt download (note: WorldPop/SEDAC files are large and may require manual download)
    local_tif = os.path.join(resolved_dir, "population_density.tif")
    print(
        "\n"
        "=" * 70 + "\n"
        "NOTE: Gridded population density raster not found.\n"
        "Please download a population density GeoTIFF and provide it via:\n"
        "  --population-raster-path /path/to/population_density.tif\n"
        "\n"
        "Recommended sources (free, global coverage):\n"
        "  - WorldPop: https://www.worldpop.org/geodata/listing?id=64\n"
        "  - SEDAC GPW v4: https://sedac.ciesin.columbia.edu/data/collection/gpw-v4\n"
        "  - GHS-POP: https://ghsl.jrc.ec.europa.eu/ghs_pop.php\n"
        "=" * 70 + "\n"
    )
    raise FileNotFoundError(
        f"Population raster not found. Please provide one via --population-raster-path "
        f"or place a .tif file in {resolved_dir}"
    )


def load_population_raster_array(
    cache_dir: Optional[str],
    raster_path: Optional[str],
    target_res_deg: float,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Load a gridded population density raster and resample to target resolution.

    Args:
        cache_dir: Directory for caching resampled data.
        raster_path: Path to source population density GeoTIFF.
        target_res_deg: Target resolution in degrees (e.g., 0.25 for ~25km).

    Returns:
        Tuple of (array, extent) where extent is (left, right, bottom, top).
    """
    if target_res_deg <= 0:
        raise ValueError("--population-res-deg must be greater than zero.")

    resolved_dir = _resolve_population_cache_dir(cache_dir)
    source_path = _ensure_population_raster(cache_dir, raster_path)

    # Check for cached resampled version
    cache_key = f"{target_res_deg:.3f}".replace(".", "p")
    source_basename = os.path.splitext(os.path.basename(source_path))[0]
    cache_file = os.path.join(resolved_dir, f"population_{source_basename}_res_{cache_key}.npz")

    if os.path.exists(cache_file):
        print(f"Loading cached population raster from {cache_file}")
        with np.load(cache_file) as data:
            return data["image"], tuple(float(v) for v in data["extent"])

    print(f"Loading and resampling population raster to {target_res_deg}° resolution...")

    with rasterio.open(source_path) as src:
        # Calculate output dimensions for global extent at target resolution
        out_width = int(360.0 / target_res_deg)
        out_height = int(180.0 / target_res_deg)

        # Read and resample to global grid
        dst_crs = CRS.from_epsg(4326)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs, dst_crs,
            src.width, src.height,
            *src.bounds,
            dst_width=out_width, dst_height=out_height
        )

        # Create output array
        arr = np.empty((out_height, out_width), dtype=np.float32)
        arr.fill(np.nan)

        reproject(
            source=rasterio.band(src, 1),
            destination=arr,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )

    # Handle nodata and negative values
    arr = np.where(arr < 0, np.nan, arr)
    # Clip extreme values (some datasets have artifacts)
    arr = np.clip(arr, 0, np.nanpercentile(arr[np.isfinite(arr)], 99.9))

    extent = (-180.0, 180.0, -90.0, 90.0)
    np.savez_compressed(cache_file, image=arr, extent=np.array(extent, dtype=np.float32))
    print(f"Cached resampled population raster to {cache_file}")

    return arr, extent


def _ensure_population_dataset() -> str:
    """Download the Natural Earth 110m admin dataset if needed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(script_dir, "ne_110m_admin_0_countries.zip")
    if not os.path.exists(local_path):
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        try:
            print(f"Downloading population dataset to {local_path} ...")
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            print("Download complete.")
        except Exception as exc:
            raise RuntimeError(f"Failed to download population dataset: {exc}")
    return local_path


def load_population_density_gdf() -> gpd.GeoDataFrame:
    """Load Natural Earth countries and compute population density."""
    dataset_path = _ensure_population_dataset()
    read_path = f"zip://{dataset_path}"
    world = gpd.read_file(read_path)
    world.columns = [str(c).lower() for c in world.columns]
    if "pop_est" not in world.columns:
        available = ", ".join(world.columns)
        raise RuntimeError(f"'pop_est' column missing from population dataset (available: {available})")
    world = world[(world["pop_est"].notna()) & (world["pop_est"] > 0)]
    if world.empty:
        raise RuntimeError("Natural Earth population dataset is empty.")
    world_equal_area = world.to_crs("EPSG:6933")
    world_equal_area["area_km2"] = world_equal_area.geometry.area / 1_000_000.0
    world_equal_area = world_equal_area[world_equal_area["area_km2"] > 0]
    if world_equal_area.empty:
        raise RuntimeError("Failed to compute positive polygon areas for population dataset.")
    world_equal_area["pop_density"] = world_equal_area["pop_est"] / world_equal_area["area_km2"]
    world_geo = world_equal_area.to_crs("EPSG:4326")
    world_geo = world_geo[["name", "pop_est", "pop_density", "geometry"]].copy()
    world_geo["pop_density"] = world_geo["pop_density"].clip(lower=0)
    return world_geo


def _load_vector_gdf(path: str) -> gpd.GeoDataFrame:
    """Load a vector dataset (shapefile / zipped shapefile) into a GeoDataFrame."""
    read_path = path
    if path.lower().endswith(".zip"):
        read_path = f"zip://{path}"
    return gpd.read_file(read_path)


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
        if "climate" in props:
            record["climate"] = props["climate"]
        for k, v in props.items():
            if k not in record:
                record[k] = v
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

    gdf["koppen_code"] = gdf["climate"].str.split().str[0]
    gdf["koppen_group"] = gdf["koppen_code"].apply(_koppen_group)
    gdf["koppen_label"] = gdf["koppen_group"].map(LABELS).fillna(gdf["koppen_group"])
    MAJOR = {"A": "Tropical", "B": "Arid", "C": "Temperate", "D": "Cold", "E": "Polar"}
    gdf["koppen_major"] = gdf["koppen_code"].str[0].map(MAJOR)
    return gdf


def _scatter_with_halo(
    ax,
    lons: List[float],
    lats: List[float],
    color,
    marker: str,
    size: float,
    edgecolor: str = "k",
    linewidth: float = 0.6,
    zorder: float = 4,
    label: Optional[str] = None,
):
    """Plot points with a white halo underlay to improve visibility."""
    ax.scatter(
        lons,
        lats,
        marker=marker,
        s=size * 1.8,
        c="white",
        alpha=0.95,
        edgecolor="none",
        zorder=zorder - 0.1,
        label=None,
        transform=ccrs.PlateCarree(),
    )
    return ax.scatter(
        lons,
        lats,
        marker=marker,
        s=size,
        c=[color],
        alpha=0.95,
        edgecolor=edgecolor,
        linewidths=linewidth,
        zorder=zorder,
        label=label,
        transform=ccrs.PlateCarree(),
    )


def _slugify_class_key(key: str) -> str:
    """Turn a LULC key into a filesystem-friendly slug."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in key)
    slug = slug.strip("_")
    return slug or "class"


def _extract_lat_lon(row: Dict[str, str]) -> Optional[Tuple[float, float]]:
    """Extract a (lat, lon) pair from a CSV row with forgiving column names."""
    lat_keys = ("lat", "latitude", "Lat", "Latitude")
    lon_keys = ("lon", "longitude", "Lon", "Longitude")
    lat_val = None
    lon_val = None
    for k in lat_keys:
        if k in row and row[k] not in ("", None):
            lat_val = row[k]
            break
    for k in lon_keys:
        if k in row and row[k] not in ("", None):
            lon_val = row[k]
            break
    if lat_val is None or lon_val is None:
        return None
    try:
        return float(lat_val), float(lon_val)
    except Exception:
        return None


def read_lat_lon_csv(csv_path: str) -> List[Tuple[float, float]]:
    """Read all lat/lon rows from a lat_lon.csv file."""
    pairs: List[Tuple[float, float]] = []
    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pair = _extract_lat_lon(row)
                if pair is not None:
                    pairs.append(pair)
    except Exception as e:
        print(f"Warning: failed to read {csv_path} ({e})")
    return pairs


def collect_class_predictions(root: str, class_key: str, model: str) -> List[Tuple[float, float]]:
    """Collect all (lat, lon) predictions for a single class.

    Args:
        root: Root directory containing class folders.
        class_key: Unified CLI key (e.g., 'bare_ground', 'built_area').
        model: Model name ('copgen' or 'terramind').

    Returns:
        List of (lat, lon) tuples.
    """
    # Resolve folder name from unified class definition
    class_info = LULC_CLASS_BY_KEY.get(class_key)
    if class_info is None:
        print(f"Warning: unknown class key '{class_key}'")
        return []

    if model == "copgen":
        folder_name = class_info["copgen_folder"]
    elif model == "terramind":
        folder_name = class_info["terramind_folder"]
    else:
        print(f"Warning: unknown model '{model}'")
        return []

    class_dir = os.path.join(root, folder_name)
    if not os.path.isdir(class_dir):
        print(f"Warning: missing class directory {class_dir}")
        return []

    if model == "copgen":
        # Try both possible copgen formats
        pattern1 = os.path.join(
            class_dir,
            "outputs",
            "copgen",
            "input_LULC_output_DEM_S1RTC_S2L1C_S2L2A_cloud_mask_lat_lon_timestamps_seed_*",
            "lat_lon.csv",
        )
        pattern2 = os.path.join(
            class_dir,
            "outputs",
            "copgen",
            "input_LULC_cloud_mask_output_DEM_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_*",
            "lat_lon.csv",
        )
        lat_lons: List[Tuple[float, float]] = []
        for csv_path in sorted(glob.glob(pattern1)):
            lat_lons.extend(read_lat_lon_csv(csv_path))
        for csv_path in sorted(glob.glob(pattern2)):
            lat_lons.extend(read_lat_lon_csv(csv_path))
    elif model == "terramind":
        pattern = os.path.join(
            class_dir,
            "outputs",
            "terramind",
            "input_LULC_output_coords_seed_*",
            "lat_lon.csv",
        )
        lat_lons: List[Tuple[float, float]] = []
        for csv_path in sorted(glob.glob(pattern)):
            lat_lons.extend(read_lat_lon_csv(csv_path))
    else:
        lat_lons: List[Tuple[float, float]] = []

    if len(lat_lons) == 0:
        print(f"No predictions found for class '{class_key}' (folder: {folder_name})")
    return lat_lons


def plot_lat_lon_by_class(
    class_to_latlons: Dict[str, List[Tuple[float, float]]],
    class_styles: Dict[str, Dict[str, object]],
    save_path: str,
    title: Optional[str],
    climate: Optional[gpd.GeoDataFrame],
    climate_alpha: float,
    scatter_size: float,
    water_layers: Optional[List[Dict[str, object]]],
    water_alpha: float,
    mountain_basemap: Optional[Dict[str, object]],
    population_basemap: Optional[Dict[str, object]],
    treecover_basemap: Optional[Dict[str, object]],
    legend_loc: str,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={"projection": ccrs.PlateCarree()})

    if treecover_basemap is not None:
        img = ax.imshow(
            treecover_basemap["image"],
            extent=treecover_basemap["extent"],
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap=treecover_basemap["cmap"],
            norm=treecover_basemap["norm"],
            alpha=treecover_basemap["alpha"],
            zorder=0.2,
        )
        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation="vertical",
            fraction=0.025,
            pad=0.01,
        )
        cbar.set_label("Percent tree cover (%)")
    elif population_basemap is not None:
        img = ax.imshow(
            population_basemap["image"],
            extent=population_basemap["extent"],
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap=population_basemap["cmap"],
            norm=population_basemap["norm"],
            alpha=population_basemap["alpha"],
            zorder=0.2,
        )
        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation="vertical",
            fraction=0.025,
            pad=0.01,
        )
        cbar.set_label("Population density (people/km²)")
    elif mountain_basemap is not None:
        ax.imshow(
            mountain_basemap["image"],
            extent=[-180, 180, -90, 90],
            origin="upper",
            transform=ccrs.PlateCarree(),
            zorder=0.2,
            alpha=mountain_basemap["alpha"],
        )

    if water_layers:
        for layer in water_layers:
            color = layer["color"]
            edgecolor = layer["edgecolor"]
            linewidth = layer["linewidth"]
            zorder = layer["zorder"]
            layer_gdf = layer["gdf"]
            if layer_gdf is None or layer_gdf.empty:
                continue
            layer_gdf.plot(
                ax=ax,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=water_alpha,
                zorder=zorder,
                legend=False,
            )

    climate_legend = None
    if climate is not None:
        groups_col = "koppen_major"
        color_map = {
            "Arid": "#fb8072",
            "Tropical": "#b3de69",
            "Temperate": "#ffffb3",
            "Cold": "#bebada",
            "Polar": "#8dd3c7",
        }
        climate["plot_color"] = climate[groups_col].map(color_map)
        climate.plot(
            ax=ax,
            color=climate["plot_color"],
            alpha=climate_alpha,
            edgecolor="#bdbdbd",
            linewidth=0.25,
            legend=False,
        )
        legend_elements = [
            Patch(facecolor=color, edgecolor="#bdbdbd", label=label, alpha=climate_alpha)
            for label, color in color_map.items()
            if label in climate[groups_col].values
        ]
        climate_legend = ax.legend(
            handles=legend_elements,
            loc="lower left",
            fontsize=10,
            title="Köppen–Geiger (coarse)",
        )

    ax.coastlines(resolution="50m", color="#424242", linewidth=0.75, zorder=3)

    for class_key, latlons in class_to_latlons.items():
        if len(latlons) == 0:
            continue
        lats = [lat for lat, _ in latlons]
        lons = [lon for _, lon in latlons]
        style = class_styles[class_key]
        label = f"{style['label']} (n={len(latlons)})"
        _scatter_with_halo(
            ax=ax,
            lons=lons,
            lats=lats,
            color=style["color"],
            marker="o",
            size=scatter_size,
            edgecolor="k",
            linewidth=0.55,
            zorder=4.5,
            label=label,
        )

    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=18)
    pred_legend = ax.legend(
        loc=legend_loc, title="LULC predictions", frameon=True, markerscale=1.2
    )
    if climate_legend is not None:
        ax.add_artist(climate_legend)

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


def plot_comparison(
    copgen_data: Dict[str, List[Tuple[float, float]]],
    terramind_data: Dict[str, List[Tuple[float, float]]],
    class_styles: Dict[str, Dict[str, object]],
    save_path: str,
    title: Optional[str],
    climate: Optional[gpd.GeoDataFrame],
    climate_alpha: float,
    scatter_size: float,
    water_layers: Optional[List[Dict[str, object]]],
    water_alpha: float,
    mountain_basemap: Optional[Dict[str, object]],
    population_basemap: Optional[Dict[str, object]],
    treecover_basemap: Optional[Dict[str, object]],
    legend_loc: str,
    terramind_hue_offset: Optional[float] = None,
) -> None:
    """Plot comparison of CopGen and TerraMind predictions on the same map."""
    fig, ax = plt.subplots(figsize=(16, 10), subplot_kw={"projection": ccrs.PlateCarree()})

    if treecover_basemap is not None:
        img = ax.imshow(
            treecover_basemap["image"],
            extent=treecover_basemap["extent"],
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap=treecover_basemap["cmap"],
            norm=treecover_basemap["norm"],
            alpha=treecover_basemap["alpha"],
            zorder=0.2,
        )
        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation="vertical",
            fraction=0.025,
            pad=0.01,
        )
        cbar.set_label("Percent tree cover (%)")
    elif population_basemap is not None:
        img = ax.imshow(
            population_basemap["image"],
            extent=population_basemap["extent"],
            origin="upper",
            transform=ccrs.PlateCarree(),
            cmap=population_basemap["cmap"],
            norm=population_basemap["norm"],
            alpha=population_basemap["alpha"],
            zorder=0.2,
        )
        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation="vertical",
            fraction=0.025,
            pad=0.01,
        )
        cbar.set_label("Population density (people/km²)")
    elif mountain_basemap is not None:
        ax.imshow(
            mountain_basemap["image"],
            extent=[-180, 180, -90, 90],
            origin="upper",
            transform=ccrs.PlateCarree(),
            zorder=0.2,
            alpha=mountain_basemap["alpha"],
        )

    if water_layers:
        for layer in water_layers:
            color = layer["color"]
            edgecolor = layer["edgecolor"]
            linewidth = layer["linewidth"]
            zorder = layer["zorder"]
            layer_gdf = layer["gdf"]
            if layer_gdf is None or layer_gdf.empty:
                continue
            layer_gdf.plot(
                ax=ax,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=water_alpha,
                zorder=zorder,
                legend=False,
            )

    climate_legend = None
    if climate is not None:
        groups_col = "koppen_major"
        color_map = {
            "Arid": "#fb8072",
            "Tropical": "#b3de69",
            "Temperate": "#ffffb3",
            "Cold": "#bebada",
            "Polar": "#8dd3c7",
        }
        climate["plot_color"] = climate[groups_col].map(color_map)
        climate.plot(
            ax=ax,
            color=climate["plot_color"],
            alpha=climate_alpha,
            edgecolor="#bdbdbd",
            linewidth=0.25,
            legend=False,
        )
        legend_elements = [
            Patch(facecolor=color, edgecolor="#bdbdbd", label=label, alpha=climate_alpha)
            for label, color in color_map.items()
            if label in climate[groups_col].values
        ]
        climate_legend = ax.legend(
            handles=legend_elements,
            loc="lower left",
            fontsize=10,
            title="Köppen–Geiger (coarse)",
        )

    ax.coastlines(resolution="50m", color="#424242", linewidth=0.75, zorder=3)

    # Plot both models with different markers and colors
    model_data = [
        ("copgen", copgen_data, MODEL_MARKERS["copgen"]),
        ("terramind", terramind_data, MODEL_MARKERS["terramind"]),
    ]

    # Determine color mode: tab10 (default) or hue-shifted LULC colors
    use_tab10 = terramind_hue_offset is None
    if use_tab10:
        tab10_cmap = plt.get_cmap("tab10")
        color_idx = 0

    for model_name, data, marker_info in model_data:
        hue_offset = 0.0 if model_name == "copgen" else (terramind_hue_offset or 0.0)
        for class_key, latlons in data.items():
            if len(latlons) == 0:
                continue
            lats = [lat for lat, _ in latlons]
            lons = [lon for _, lon in latlons]
            style = class_styles.get(class_key)
            if style is None:
                continue

            # Choose color based on mode
            if use_tab10:
                # Use tab10 colormap for distinct colors per (model, class)
                plot_color = tab10_cmap(color_idx % 10)[:3]  # RGB only, no alpha
                color_idx += 1
            else:
                # Apply hue shift to LULC class colors
                base_color = style["color"]
                plot_color = _shift_hue(base_color, hue_offset)

            label = f"{marker_info['label']} - {style['label']} (n={len(latlons)})"
            _scatter_with_halo(
                ax=ax,
                lons=lons,
                lats=lats,
                color=plot_color,
                marker=marker_info["marker"],
                size=scatter_size,
                edgecolor="k",
                linewidth=0.55,
                zorder=4.5,
                label=label,
            )

    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=18)
    pred_legend = ax.legend(
        loc=legend_loc, title="Model predictions", frameon=True, markerscale=1.2
    )
    if climate_legend is not None:
        ax.add_artist(climate_legend)

    out_dir = os.path.dirname(save_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    print(f"Saved comparison figure to: {save_path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    # Validate common arguments
    if not (0.0 <= args.climate_alpha <= 1.0):
        raise ValueError("--climate-alpha must be within [0, 1].")
    if not (0.0 <= args.water_alpha <= 1.0):
        raise ValueError("--water-alpha must be within [0, 1].")
    if args.scatter_size <= 0:
        raise ValueError("--scatter-size must be greater than zero.")
    if not (0.0 <= args.mountain_alpha <= 1.0):
        raise ValueError("--mountain-alpha must be within [0, 1].")
    if args.mountain_max_size < 0:
        raise ValueError("--mountain-max-size must be non-negative.")
    if not (0.0 <= args.population_alpha <= 1.0):
        raise ValueError("--population-alpha must be within [0, 1].")
    if args.population_res_deg <= 0:
        raise ValueError("--population-res-deg must be greater than zero.")
    if not (0.0 <= args.treecover_alpha <= 1.0):
        raise ValueError("--treecover-alpha must be within [0, 1].")
    if args.treecover_res_deg <= 0:
        raise ValueError("--treecover-res-deg must be greater than zero.")
    if args.terramind_hue_offset is not None and not (0.0 <= args.terramind_hue_offset <= 1.0):
        raise ValueError("--terramind-hue-offset must be within [0, 1].")

    # Validate and select classes using unified class system
    valid_keys = sorted(LULC_CLASS_BY_KEY.keys())
    if args.classes:
        selected_classes = []
        missing = []
        seen = set()
        for raw_key in args.classes:
            norm_key = raw_key.strip().lower()
            if norm_key not in LULC_CLASS_BY_KEY:
                missing.append(raw_key)
                continue
            if norm_key in seen:
                continue
            seen.add(norm_key)
            selected_classes.append(LULC_CLASS_BY_KEY[norm_key])
        if missing:
            raise ValueError(
                f"Unknown class(es): {', '.join(missing)}. "
                f"Valid options: {', '.join(valid_keys)}"
            )
    else:
        selected_classes = LULC_CLASSES_UNIFIED

    # Build class styles from selected classes
    class_styles = {c["key"]: {"label": c["label"], "color": c["color"]} for c in selected_classes}

    # Handle comparison mode vs single model mode
    if args.comparison:
        # Comparison mode: need both copgen-root and terramind-root
        if args.copgen_root is None or args.terramind_root is None:
            raise ValueError(
                "--comparison requires both --copgen-root and --terramind-root to be specified."
            )
        copgen_root = os.path.abspath(args.copgen_root)
        terramind_root = os.path.abspath(args.terramind_root)
        if not os.path.isdir(copgen_root):
            raise NotADirectoryError(f"copgen-root does not exist or is not a directory: {copgen_root}")
        if not os.path.isdir(terramind_root):
            raise NotADirectoryError(f"terramind-root does not exist or is not a directory: {terramind_root}")

        # Collect predictions from both models using unified keys
        copgen_data: Dict[str, List[Tuple[float, float]]] = {}
        terramind_data: Dict[str, List[Tuple[float, float]]] = {}

        for c in selected_classes:
            class_key = c["key"]
            copgen_data[class_key] = collect_class_predictions(copgen_root, class_key, model="copgen")
            terramind_data[class_key] = collect_class_predictions(terramind_root, class_key, model="terramind")

        # Determine save path
        filename = "lat_lon_comparison"
        if args.classes:
            slug = "_".join(_slugify_class_key(c["key"]) for c in selected_classes)
            filename = f"{filename}_{slug}"
        filename = f"{filename}.png"

        if args.save is not None:
            save_folder = args.save
        else:
            save_folder = copgen_root
        save_path = os.path.join(save_folder, filename)

    else:
        # Single model mode
        root_dir = os.path.abspath(args.root)
        if not os.path.isdir(root_dir):
            raise NotADirectoryError(f"root does not exist or is not a directory: {root_dir}")

        if args.model not in ("copgen", "terramind"):
            raise ValueError(f"Invalid model: {args.model}. Must be 'copgen' or 'terramind'.")

        filename = "lat_lon_by_lulc"
        if args.classes:
            slug = "_".join(_slugify_class_key(c["key"]) for c in selected_classes)
            filename = f"{filename}_{slug}"
        filename = f"{filename}.png"

        if args.save is not None:
            save_folder = args.save
        else:
            save_folder = root_dir
        save_path = os.path.join(save_folder, filename)

        # Collect predictions for single model
        class_to_latlons: Dict[str, List[Tuple[float, float]]] = {}
        for c in selected_classes:
            class_to_latlons[c["key"]] = collect_class_predictions(root_dir, c["key"], model=args.model)

    mountain_basemap: Optional[Dict[str, object]] = None
    population_basemap: Optional[Dict[str, object]] = None
    treecover_basemap: Optional[Dict[str, object]] = None
    climate_gdf = None

    if args.basemap == "mountains":
        mountain_path = ensure_mountain_image(args.mountain_image_path)
        max_size = args.mountain_max_size if args.mountain_max_size > 0 else None
        mountain_image = load_mountain_image(mountain_path, max_size)
        mountain_basemap = {"image": mountain_image, "alpha": args.mountain_alpha}
    elif args.basemap == "climates":
        koppen_path = ensure_koppen_json(args.koppen_path)
        try:
            print("Loading climate dataset...")
            climate_gdf = load_climate_gdf_from_json(koppen_path)
        except Exception as e:
            print(f"Warning: failed to load climate background ({e}); continuing without it.")
            climate_gdf = None
    elif args.basemap == "population":
        try:
            population_array, population_extent = load_population_raster_array(
                args.population_cache_dir,
                args.population_raster_path,
                args.population_res_deg,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load population basemap: {exc}") from exc

        # Compute density statistics for normalization
        valid_densities = population_array[np.isfinite(population_array)]
        if valid_densities.size == 0:
            raise RuntimeError("Population density raster did not contain valid values.")

        vmin = max(0.1, float(np.nanpercentile(valid_densities, 5)))
        if args.population_max_density is not None and args.population_max_density > vmin:
            vmax = args.population_max_density
        else:
            vmax = float(np.nanpercentile(valid_densities, 99))
        if vmax <= vmin:
            vmax = vmin + 1.0

        try:
            population_cmap = plt.get_cmap(args.population_cmap)
        except ValueError as exc:
            raise ValueError(f"Invalid population colormap '{args.population_cmap}': {exc}") from exc

        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        population_basemap = {
            "image": population_array,
            "extent": population_extent,
            "alpha": args.population_alpha,
            "cmap": population_cmap,
            "norm": norm,
        }
    elif args.basemap == "treecover":
        treecover_array, treecover_extent = load_treecover_array(
            args.treecover_cache_dir, args.treecover_res_deg
        )
        treecover_basemap = {
            "image": treecover_array,
            "extent": treecover_extent,
            "alpha": args.treecover_alpha,
            "cmap": plt.get_cmap("YlGn"),
            "norm": mcolors.Normalize(vmin=0, vmax=100),
        }
    water_layers: List[Dict[str, object]] = []
    if not args.no_oceans:
        try:
            ocean_path = _ensure_vector_dataset(
                "ocean water bodies", args.water_ocean_path, WATER_OCEAN_URL
            )
            ocean_gdf = _load_vector_gdf(ocean_path)
            if not ocean_gdf.empty:
                water_layers.append(
                    {
                        "gdf": ocean_gdf,
                        "color": "#e8f6ff",
                        "edgecolor": "#80bce0",
                        "linewidth": 0.15,
                        "zorder": 1.2,
                    }
                )
        except Exception as e:
            print(f"Warning: failed to load ocean water bodies ({e}); continuing without them.")
    if not args.no_lakes:
        try:
            lakes_path = _ensure_vector_dataset(
                "freshwater bodies", args.water_lakes_path, WATER_LAKES_URL
            )
            lakes_gdf = _load_vector_gdf(lakes_path)
            if not lakes_gdf.empty:
                water_layers.append(
                    {
                        "gdf": lakes_gdf,
                        "color": "#b3e2ff",
                        "edgecolor": "#76a4cf",
                        "linewidth": 0.12,
                        "zorder": 1.3,
                    }
                )
        except Exception as e:
            print(f"Warning: failed to load freshwater bodies ({e}); continuing without them.")
    if len(water_layers) == 0:
        water_layers = None

    legend_loc_map = {
        "top-right": "upper right",
        "top-left": "upper left",
        "bottom-right": "lower right",
        "bottom-left": "lower left",
    }
    legend_loc = legend_loc_map[args.legend_position]

    if args.comparison:
        plot_comparison(
            copgen_data=copgen_data,
            terramind_data=terramind_data,
            class_styles=class_styles,
            save_path=save_path,
            title=args.title,
            climate=climate_gdf,
            climate_alpha=args.climate_alpha,
            scatter_size=args.scatter_size,
            water_layers=water_layers,
            water_alpha=args.water_alpha,
            mountain_basemap=mountain_basemap,
            population_basemap=population_basemap,
            treecover_basemap=treecover_basemap,
            legend_loc=legend_loc,
            terramind_hue_offset=args.terramind_hue_offset,
        )
    else:
        plot_lat_lon_by_class(
            class_to_latlons=class_to_latlons,
            class_styles=class_styles,
            save_path=save_path,
            title=args.title,
            climate=climate_gdf,
            climate_alpha=args.climate_alpha,
            scatter_size=args.scatter_size,
            water_layers=water_layers,
            water_alpha=args.water_alpha,
            mountain_basemap=mountain_basemap,
            population_basemap=population_basemap,
            treecover_basemap=treecover_basemap,
            legend_loc=legend_loc,
        )


if __name__ == "__main__":
    main()
