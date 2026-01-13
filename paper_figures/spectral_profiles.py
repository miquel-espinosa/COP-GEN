"""
Generate spectral profile plots comparing Copgen S2L2A outputs against ground truth.

For each LULC class found in the image, selects a representative pixel and plots
the spectral values across all S2L2A bands.

Example usage:
    # GT vs Copgen only (default)
    python spectral_profiles.py \
        --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/143D_1481R \
        --terramind-exp input_DEM_LULC_output_S2L2A_seed_111 \
        --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_111
    
    # Include Terramind comparison (generates all modes: best-copgen, best-terramind, best-per-model)
    python spectral_profiles.py \
        --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/143D_1481R \
        --terramind-exp input_DEM_LULC_output_S2L2A_seed_111 \
        --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_111 \
        --comparison
    
    # Single mode with custom output
    python spectral_profiles.py \
        --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/143D_1481R \
        --terramind-exp input_DEM_LULC_output_S2L2A_seed_111 \
        --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_111 \
        --single-mode best-copgen \
        --output custom_output.png
"""
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
from scipy.ndimage import binary_erosion


# S2L2A band order (matching benchmark/common/modalities.py)
S2L2A_BAND_ORDER = [
    "B01", "B02", "B03", "B04", "B05", "B06", 
    "B07", "B08", "B8A", "B09", "B11", "B12"
]

# Sentinel-2 L2A center wavelengths in nanometers (for x-axis)
S2L2A_WAVELENGTHS = {
    "B01": 443,   # Coastal aerosol
    "B02": 490,   # Blue
    "B03": 560,   # Green
    "B04": 665,   # Red
    "B05": 705,   # Red edge 1
    "B06": 740,   # Red edge 2
    "B07": 783,   # Red edge 3
    "B08": 842,   # NIR broad
    "B8A": 865,   # NIR narrow
    "B09": 945,   # Water vapor
    "B11": 1610,  # SWIR 1
    "B12": 2190,  # SWIR 2
}

# Band names for display
S2L2A_BAND_NAMES = {
    "B01": "Coastal Aerosol",
    "B02": "Blue",
    "B03": "Green", 
    "B04": "Red",
    "B05": "Red Edge 1",
    "B06": "Red Edge 2",
    "B07": "Red Edge 3",
    "B08": "NIR Broad",
    "B8A": "NIR Narrow",
    "B09": "Water Vapor",
    "B11": "SWIR 1",
    "B12": "SWIR 2",
}

# LULC class mapping (from ddm/pre_post_process_data.py)
# Maps index value in tif to class info
LULC_CLASSES = {
    0: {"name": "No Data", "color": "#000000", "strong-color": "#000000"},
    1: {"name": "Water", "color": "#1a5bab", "strong-color": "#144a8a"},
    2: {"name": "Trees", "color": "#358221", "strong-color": "#2a6a1b"},
    3: {"name": "Flooded Vegetation", "color": "#87d19e", "strong-color": "#6fb786"},
    4: {"name": "Crops", "color": "#ffdb5c", "strong-color": "#e6c24f"},
    5: {"name": "Built-up Areas", "color": "#ed022a", "strong-color": "#c40222"},
    6: {"name": "Bare Ground", "color": "#e3e2c3", "strong-color": "#c9c8aa"},
    7: {"name": "Snow/Ice", "color": "#a8ebff", "strong-color": "#8fd1e6"},
    8: {"name": "Clouds", "color": "#616161", "strong-color": "#4e4e4e"},
    9: {"name": "Rangeland", "color": "#a59b8f", "strong-color": "#8c8378"},
}

# Mapping from lowercase/underscore names to class indices for CLI
LULC_NAME_TO_INDEX = {
    name.lower().replace(" ", "_").replace("/", "_"): idx
    for idx, info in LULC_CLASSES.items()
    for name in [info["name"]]
}
# Add common aliases
LULC_NAME_TO_INDEX.update({
    "water": 1,
    "trees": 2,
    "flooded_vegetation": 3,
    "flooded": 3,
    "crops": 4,
    "built_up": 5,
    "built_up_areas": 5,
    "urban": 5,
    "bare_ground": 6,
    "bare": 6,
    "snow_ice": 7,
    "snow": 7,
    "ice": 7,
    "clouds": 8,
    "rangeland": 9,
})


def parse_skip_classes(skip_names: List[str]) -> set:
    """Convert class names to indices for skipping.
    
    Args:
        skip_names: List of class names (case-insensitive, underscores for spaces)
        
    Returns:
        Set of class indices to skip
    """
    skip_indices = set()
    for name in skip_names:
        normalized = name.lower().replace(" ", "_").replace("-", "_")
        if normalized in LULC_NAME_TO_INDEX:
            skip_indices.add(LULC_NAME_TO_INDEX[normalized])
        else:
            available = sorted(set(LULC_NAME_TO_INDEX.keys()))
            raise ValueError(
                f"Unknown LULC class '{name}'. Available: {', '.join(available)}"
            )
    return skip_indices


# Original LULC values to index mapping (for raw data)
LULC_VALUE_TO_INDEX = {
    0: 0,   # No data
    1: 1,   # Water
    2: 2,   # Trees
    4: 3,   # Flooded vegetation
    5: 4,   # Crops
    7: 5,   # Built-up areas
    8: 6,   # Bare Ground
    9: 7,   # Snow/Ice
    10: 8,  # Clouds
    11: 9,  # Rangeland
}


def find_lulc_file(tile_root: Path) -> Path:
    """Find the LULC file for the tile."""
    # Try terramind_inputs first (stacked format)
    lulc_dir = tile_root / "terramind_inputs" / "LULC"
    if lulc_dir.exists():
        lulc_files = list(lulc_dir.glob("*.tif"))
        if lulc_files:
            return lulc_files[0]
    
    # Try Core-LULC format
    core_lulc = tile_root / "Core-LULC"
    if core_lulc.exists():
        lulc_files = list(core_lulc.rglob("*.tif"))
        if lulc_files:
            return lulc_files[0]
    
    raise FileNotFoundError(f"No LULC file found in {tile_root}")


def load_lulc(lulc_path: Path) -> np.ndarray:
    """Load LULC data and return as 2D array with class indices 0-9."""
    with rasterio.open(lulc_path) as src:
        lulc = src.read(1)
    
    # Check if values need remapping (raw values vs already indexed)
    unique_vals = np.unique(lulc)
    max_val = unique_vals.max()
    
    # If max value > 9, assume it's raw values that need remapping
    if max_val > 9:
        mapped = np.zeros_like(lulc, dtype=np.uint8)
        for orig_val, idx in LULC_VALUE_TO_INDEX.items():
            mapped[lulc == orig_val] = idx
        return mapped
    return lulc.astype(np.uint8)


def load_terramind_s2l2a(terramind_gen_path: Path) -> np.ndarray:
    """
    Load S2L2A from Terramind output (stacked .tif with all bands).
    Returns array of shape (n_bands, H, W).
    """
    # Find the .tif file in generations folder
    gen_dir = terramind_gen_path / "generations"
    tif_files = list(gen_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif file found in {gen_dir}")
    
    with rasterio.open(tif_files[0]) as src:
        data = src.read()  # Shape: (n_bands, H, W)
    return data


def load_copgen_s2l2a(copgen_gen_path: Path) -> np.ndarray:
    """
    Load S2L2A from Copgen output (individual band .tif files).
    Resamples all bands to 10m resolution (highest res bands).
    Returns array of shape (n_bands, H, W).
    """
    gen_dir = copgen_gen_path / "generations" / "Core-S2L2A"
    if not gen_dir.exists():
        raise FileNotFoundError(f"Copgen S2L2A directory not found: {gen_dir}")
    
    # Find the product directory (navigate through nested structure)
    # Structure: Core-S2L2A/<vstrip>/<tile>/<product>/
    product_dirs = list(gen_dir.rglob("B02.tif"))
    if not product_dirs:
        raise FileNotFoundError(f"No S2L2A band files found in {gen_dir}")
    
    product_dir = product_dirs[0].parent
    
    # First, get target shape from a 10m band (B02, B03, B04, or B08)
    with rasterio.open(product_dir / "B02.tif") as src:
        target_shape = (src.height, src.width)
    
    # Load bands in order, resampling to target shape if needed
    bands = []
    for band_name in S2L2A_BAND_ORDER:
        band_path = product_dir / f"{band_name}.tif"
        if not band_path.exists():
            raise FileNotFoundError(f"Band file not found: {band_path}")
        with rasterio.open(band_path) as src:
            data = src.read(1)
            # Resample if shape doesn't match
            if data.shape != target_shape:
                from scipy.ndimage import zoom
                zoom_factors = (target_shape[0] / data.shape[0], 
                               target_shape[1] / data.shape[1])
                data = zoom(data, zoom_factors, order=1)  # Bilinear interpolation
        bands.append(data)
    
    return np.stack(bands, axis=0)


def load_ground_truth_s2l2a(tile_root: Path) -> Optional[np.ndarray]:
    """Load ground truth S2L2A if available."""
    # Try terramind_inputs first (already stacked and resampled)
    gt_dir = tile_root / "terramind_inputs" / "S2L2A"
    if gt_dir.exists():
        tif_files = list(gt_dir.glob("*.tif"))
        if tif_files:
            with rasterio.open(tif_files[0]) as src:
                return src.read()
    
    # Try Core-S2L2A (individual bands, may need resampling)
    core_s2l2a = tile_root / "Core-S2L2A"
    if core_s2l2a.exists():
        band_files = list(core_s2l2a.rglob("B02.tif"))
        if band_files:
            product_dir = band_files[0].parent
            
            # Get target shape from 10m band
            with rasterio.open(product_dir / "B02.tif") as src:
                target_shape = (src.height, src.width)
            
            bands = []
            for band_name in S2L2A_BAND_ORDER:
                band_path = product_dir / f"{band_name}.tif"
                if band_path.exists():
                    with rasterio.open(band_path) as src:
                        data = src.read(1)
                        # Resample if needed
                        if data.shape != target_shape:
                            from scipy.ndimage import zoom
                            zoom_factors = (target_shape[0] / data.shape[0],
                                           target_shape[1] / data.shape[1])
                            data = zoom(data, zoom_factors, order=1)
                        bands.append(data)
            if bands:
                return np.stack(bands, axis=0)
    
    return None


def create_interior_mask(
    lulc: np.ndarray, 
    class_idx: int, 
    frame_margin: int = 5,
    class_boundary_margin: int = 3
) -> np.ndarray:
    """
    Create a mask of interior pixels for a class, excluding:
    1. Pixels within frame_margin of image boundaries
    2. Pixels within class_boundary_margin of different class pixels
    
    Args:
        lulc: LULC array (H, W)
        class_idx: Class to create mask for
        frame_margin: Pixels to exclude from image frame edges
        class_boundary_margin: Pixels to exclude from class boundaries
    
    Returns:
        Boolean mask where True = valid interior pixel
    """
    h, w = lulc.shape
    
    # Start with class mask
    class_mask = lulc == class_idx
    
    if not np.any(class_mask):
        return class_mask
    
    # Erode class mask to get interior pixels (away from class boundaries)
    if class_boundary_margin > 0:
        # Create structuring element for erosion
        struct = np.ones((2 * class_boundary_margin + 1, 2 * class_boundary_margin + 1), dtype=bool)
        interior_mask = binary_erosion(class_mask, structure=struct)
    else:
        interior_mask = class_mask
    
    # Exclude pixels near image frame boundaries
    if frame_margin > 0:
        frame_mask = np.ones((h, w), dtype=bool)
        frame_mask[:frame_margin, :] = False  # Top
        frame_mask[-frame_margin:, :] = False  # Bottom
        frame_mask[:, :frame_margin] = False  # Left
        frame_mask[:, -frame_margin:] = False  # Right
        interior_mask = interior_mask & frame_mask
    
    return interior_mask


def select_representative_pixels(
    lulc: np.ndarray, 
    target_size: Tuple[int, int] = None,
    frame_margin: int = 5,
    class_boundary_margin: int = 3,
    skip_classes: set = None
) -> Dict[int, Tuple[int, int]]:
    """
    For each LULC class present, select a representative pixel (center-most within the region).
    Excludes pixels near image boundaries and class boundaries.
    
    Args:
        lulc: LULC array (H, W)
        target_size: Optional size to center-crop LULC to
        frame_margin: Pixels to exclude from image frame edges
        class_boundary_margin: Pixels to exclude from class boundaries
        skip_classes: Set of class indices to skip (default: None = skip none)
    
    Returns:
        Dict mapping class_idx -> (row, col)
    """
    if skip_classes is None:
        skip_classes = set()
    
    if target_size is not None:
        # Center crop LULC to match model generation region
        h, w = lulc.shape
        th, tw = target_size
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        lulc_cropped = lulc[start_h:start_h+th, start_w:start_w+tw]
    else:
        lulc_cropped = lulc
    
    pixels = {}
    for class_idx in range(10):  # Classes 0-9
        if class_idx == 0 or class_idx in skip_classes:  # Skip no-data and user-specified classes
            continue
        
        # Get interior mask (excluding boundaries)
        interior_mask = create_interior_mask(
            lulc_cropped, class_idx, frame_margin, class_boundary_margin
        )
        
        # Fall back to full class mask if no interior pixels
        if not np.any(interior_mask):
            interior_mask = lulc_cropped == class_idx
            if not np.any(interior_mask):
                continue
        
        # Find indices where this class exists in interior
        rows, cols = np.where(interior_mask)
        # Select the center-most pixel of this class within the cropped region
        center_row = np.median(rows).astype(int)
        center_col = np.median(cols).astype(int)
        # Get actual pixel that's closest to median
        distances = (rows - center_row)**2 + (cols - center_col)**2
        closest_idx = np.argmin(distances)
        # Coordinates are in the cropped region space (0 to target_size)
        pixels[class_idx] = (rows[closest_idx], cols[closest_idx])
    
    return pixels


def compute_spectral_error(pred_spectral: np.ndarray, gt_spectral: np.ndarray, 
                          metric: str = "mae") -> float:
    """
    Compute error between predicted and ground truth spectral profiles.
    
    Args:
        pred_spectral: Predicted spectral values (n_bands,)
        gt_spectral: Ground truth spectral values (n_bands,)
        metric: Error metric - "mae", "mse", or "cosine"
    
    Returns:
        Error value (lower is better for mae/mse, higher is better for cosine)
    """
    if metric == "mae":
        return np.mean(np.abs(pred_spectral - gt_spectral))
    elif metric == "mse":
        return np.mean((pred_spectral - gt_spectral) ** 2)
    elif metric == "cosine":
        # Cosine similarity (1 = identical, 0 = orthogonal)
        norm_pred = np.linalg.norm(pred_spectral)
        norm_gt = np.linalg.norm(gt_spectral)
        if norm_pred == 0 or norm_gt == 0:
            return 0.0
        return np.dot(pred_spectral, gt_spectral) / (norm_pred * norm_gt)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def select_best_matching_pixels(
    lulc: np.ndarray,
    pred_data: np.ndarray,
    gt_data: np.ndarray,
    target_size: Tuple[int, int] = None,
    metric: str = "mae",
    max_pixels_per_class: int = 1000,
    frame_margin: int = 5,
    class_boundary_margin: int = 3,
    skip_classes: set = None
) -> Dict[int, Tuple[int, int]]:
    """
    For each LULC class, find the pixel that best matches GT spectral profile.
    Excludes pixels near image boundaries and class boundaries.
    
    Args:
        lulc: LULC array (H, W)
        pred_data: Predicted S2L2A data (C, H, W) - already normalized
        gt_data: Ground truth S2L2A data (C, H, W) - already normalized  
        target_size: Size to crop LULC to
        metric: Error metric for comparison
        max_pixels_per_class: Max pixels to search (for efficiency)
        frame_margin: Pixels to exclude from image frame edges
        class_boundary_margin: Pixels to exclude from class boundaries
        skip_classes: Set of class indices to skip (default: None = skip none)
    
    Returns:
        Dict mapping class_idx -> (row, col) of best matching pixel
    """
    if skip_classes is None:
        skip_classes = set()
    
    if target_size is not None:
        # Center crop LULC to match model generation region
        h, w = lulc.shape
        th, tw = target_size
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        lulc_cropped = lulc[start_h:start_h+th, start_w:start_w+tw]
    else:
        lulc_cropped = lulc
    
    pixels = {}
    for class_idx in range(10):
        if class_idx == 0 or class_idx in skip_classes:  # Skip no-data and user-specified classes
            continue
        
        # Get interior mask (excluding boundaries)
        interior_mask = create_interior_mask(
            lulc_cropped, class_idx, frame_margin, class_boundary_margin
        )
        
        # Fall back to full class mask if no interior pixels
        if not np.any(interior_mask):
            interior_mask = lulc_cropped == class_idx
            if not np.any(interior_mask):
                continue
            
        rows, cols = np.where(interior_mask)
        n_pixels = len(rows)
        
        # Sample if too many pixels
        if n_pixels > max_pixels_per_class:
            indices = np.random.choice(n_pixels, max_pixels_per_class, replace=False)
            rows = rows[indices]
            cols = cols[indices]
        
        # Find best matching pixel
        best_error = float('inf') if metric != "cosine" else float('-inf')
        best_pixel = None
        
        for r, c in zip(rows, cols):
            pred_spectral = pred_data[:, r, c]
            gt_spectral = gt_data[:, r, c]
            error = compute_spectral_error(pred_spectral, gt_spectral, metric)
            
            if metric == "cosine":
                # Higher is better for cosine
                if error > best_error:
                    best_error = error
                    best_pixel = (r, c)
            else:
                # Lower is better for mae/mse
                if error < best_error:
                    best_error = error
                    best_pixel = (r, c)
        
        if best_pixel is not None:
            pixels[class_idx] = best_pixel
    
    return pixels


def select_best_matching_pixels_for_models(
    lulc: np.ndarray,
    model_data: Dict[str, np.ndarray],
    gt_data: np.ndarray,
    target_size: Tuple[int, int] = None,
    metric: str = "mae",
    max_pixels_per_class: int = 1000,
    frame_margin: int = 5,
    class_boundary_margin: int = 3,
    skip_classes: set = None
) -> Dict[str, Dict[int, Tuple[int, int]]]:
    """
    For each model, find the best matching pixels per LULC class.
    
    Args:
        lulc: LULC array (H, W)
        model_data: Dict of model_name -> S2L2A data (C, H, W)
        gt_data: Ground truth S2L2A data (C, H, W)
        target_size: Size to crop LULC to
        metric: Error metric
        max_pixels_per_class: Max pixels to search
        frame_margin: Pixels to exclude from image frame edges
        class_boundary_margin: Pixels to exclude from class boundaries
        skip_classes: Set of class indices to skip (default: None = skip none)
    
    Returns:
        Dict of model_name -> {class_idx -> (row, col)}
    """
    result = {}
    for model_name, pred_data in model_data.items():
        print(f"  Finding best pixels for {model_name}...")
        result[model_name] = select_best_matching_pixels(
            lulc, pred_data, gt_data, target_size, metric, max_pixels_per_class,
            frame_margin, class_boundary_margin, skip_classes
        )
    return result


def normalize_reflectance(data: np.ndarray) -> np.ndarray:
    """Normalize S2L2A data to reflectance values [0, 1]."""
    # S2L2A values are typically stored as uint16 with scale factor 10000
    return data.astype(np.float32) / 10000.0


def extract_spectral_profile(data: np.ndarray, row: int, col: int) -> np.ndarray:
    """Extract spectral values for a pixel at (row, col)."""
    return data[:, row, col]


def plot_spectral_profiles(
    profiles: Dict[str, Dict[int, np.ndarray]],  # source -> class_idx -> spectral values
    output_path: Path,
    title: str = "S2L2A Spectral Profiles of Features of Interest"
):
    """
    Create a spectral profile plot comparing multiple sources.
    Similar to the reference showing spectral characteristics of different land cover types.
    
    Args:
        profiles: Dict mapping source name -> {class_idx: spectral_values}
        output_path: Where to save the plot
        title: Plot title
    """
    # Get wavelengths for x-axis
    wavelengths = [S2L2A_WAVELENGTHS[b] for b in S2L2A_BAND_ORDER]
    
    # Setup figure with clean styling
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    ax.set_facecolor('#fafafa')
    
    # Add subtle grid
    ax.grid(True, linestyle='-', alpha=0.3, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    
    # Define line styles for different sources
    line_styles = {
        "Ground Truth": {"linestyle": "-", "linewidth": 2.5, "alpha": 1.0},
        "Terramind": {"linestyle": "--", "linewidth": 2, "alpha": 0.85},
        "Copgen": {"linestyle": ":", "linewidth": 2.5, "alpha": 0.85},
    }
    
    # Collect all classes present across all sources
    all_classes = set()
    for source_profiles in profiles.values():
        all_classes.update(source_profiles.keys())
    all_classes = sorted(all_classes)
    
    # Plot each class with distinctive markers
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X']
    
    for idx, class_idx in enumerate(all_classes):
        class_info = LULC_CLASSES.get(class_idx, {"name": f"Class {class_idx}", "color": "#888888"})
        color = class_info["color"]
        marker = markers[idx % len(markers)]
        
        for source_name, source_profiles in profiles.items():
            if class_idx in source_profiles:
                spectral = source_profiles[class_idx]
                style = line_styles.get(source_name, {"linestyle": "-", "linewidth": 2, "alpha": 0.8})
                ax.plot(wavelengths, spectral, 
                       color=color, 
                       linestyle=style["linestyle"],
                       linewidth=style["linewidth"],
                       marker=marker,
                       markersize=6,
                       alpha=style["alpha"],
                       zorder=10)
    
    # Create legend box on the right side (outside plot)
    # Class legend
    class_handles = []
    for idx, class_idx in enumerate(all_classes):
        class_info = LULC_CLASSES.get(class_idx, {"name": f"Class {class_idx}", "color": "#888888"})
        marker = markers[idx % len(markers)]
        handle = Line2D([0], [0], color=class_info["color"], linewidth=2, 
                       marker=marker, markersize=6, label=class_info["name"])
        class_handles.append(handle)
    
    # Source legend
    source_handles = []
    for source_name, style in line_styles.items():
        if source_name in profiles:
            handle = Line2D([0], [0], color='#555555', linestyle=style["linestyle"], 
                          linewidth=2, label=source_name)
            source_handles.append(handle)
    
    # Position legends outside plot area
    legend1 = ax.legend(handles=class_handles, loc='upper left', 
                       bbox_to_anchor=(1.02, 1.0), title='Profiles', 
                       fontsize=10, title_fontsize=11, framealpha=0.95,
                       edgecolor='#cccccc')
    ax.add_artist(legend1)
    legend2 = ax.legend(handles=source_handles, loc='upper left',
                       bbox_to_anchor=(1.02, 0.45), title='Source',
                       fontsize=10, title_fontsize=11, framealpha=0.95,
                       edgecolor='#cccccc')
    
    # Styling
    ax.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set axis limits
    ax.set_xlim(400, 2300)
    y_max = 0.6
    for source_profiles in profiles.values():
        for spectral in source_profiles.values():
            y_max = max(y_max, np.max(spectral) * 1.15)
    ax.set_ylim(0, y_max)
    
    # Custom x-axis ticks at specific wavelengths
    ax.set_xticks([wl for wl in wavelengths if wl < 1000] + [1000, 2000])
    
    # Add secondary x-axis with band names
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(wavelengths)
    ax2.set_xticklabels(S2L2A_BAND_ORDER, fontsize=8, rotation=45, ha='left')
    ax2.tick_params(axis='x', which='major', pad=2)
    
    # Add a light box around the plot
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved spectral profile plot to {output_path}")


def plot_spectral_profiles_separate(
    profiles: Dict[str, Dict[int, np.ndarray]],
    output_path: Path,
    title: str = "Spectral Profiles Comparison by LULC Class"
):
    """
    Create separate subplot for each LULC class comparing sources.
    Plots in the same row share y-axis, plots in the same column share x-axis.
    """
    wavelengths = [S2L2A_WAVELENGTHS[b] for b in S2L2A_BAND_ORDER]
    
    # Collect all classes
    all_classes = set()
    for source_profiles in profiles.values():
        all_classes.update(source_profiles.keys())
    all_classes = sorted(all_classes)
    
    n_classes = len(all_classes)
    if n_classes == 0:
        print("No classes found to plot!")
        return
    
    # Create subplots with shared axes
    n_cols = min(3, n_classes)
    n_rows = (n_classes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5*n_cols, 5.5*n_rows), 
                            squeeze=False, facecolor='white',
                            sharex='col', sharey='row')
    
    # Distinctive colors for each source
    source_styles = {
        "Ground Truth": {"color": "#27ae60", "linestyle": "-", "marker": "o"},   # Green
        "Terramind": {"color": "#2980b9", "linestyle": "-", "marker": "s"},     # Blue  
        "Copgen": {"color": "#c0392b", "linestyle": "-", "marker": "^"},         # Red
    }
    
    # Font sizes for paper readability
    TITLE_FONTSIZE = 22
    AXIS_LABEL_FONTSIZE = 18
    TICK_FONTSIZE = 16
    LEGEND_FONTSIZE = 16
    MARKER_SIZE = 12
    LINE_WIDTH = 4
    
    # First pass: compute y-axis limits per row based on actual data range
    row_y_min = {}
    row_y_max = {}
    for idx, class_idx in enumerate(all_classes):
        row = idx // n_cols
        for source_profiles in profiles.values():
            if class_idx in source_profiles:
                spectral = source_profiles[class_idx]
                current_min = row_y_min.get(row, float('inf'))
                current_max = row_y_max.get(row, float('-inf'))
                row_y_min[row] = min(current_min, np.min(spectral))
                row_y_max[row] = max(current_max, np.max(spectral))
    
    # Add padding (10% on each side)
    for row in row_y_min:
        data_range = row_y_max[row] - row_y_min[row]
        padding = data_range * 0.1
        row_y_min[row] = row_y_min[row] - padding
        row_y_max[row] = row_y_max[row] + padding
    
    for idx, class_idx in enumerate(all_classes):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        ax.set_facecolor('#fafafa')
        
        class_info = LULC_CLASSES.get(class_idx, {"name": f"Class {class_idx}", "color": "#888888"})
        
        for source_name, source_profiles in profiles.items():
            if class_idx in source_profiles:
                spectral = source_profiles[class_idx]
                style = source_styles.get(source_name, {"color": "#888888", "linestyle": "-", "marker": "o"})
                ax.plot(wavelengths, spectral,
                       color=style["color"],
                       linestyle=style["linestyle"],
                       linewidth=LINE_WIDTH,
                       marker=style["marker"],
                       markersize=MARKER_SIZE,
                       label=source_name,
                       alpha=0.85)
        
        # Title with colored background strip
        ax.set_title(class_info["name"], fontsize=TITLE_FONTSIZE, fontweight='bold',
                    color='white', backgroundcolor=class_info["color"],
                    pad=12)
        # Determine if this is a bottom row plot (last row or above an empty cell)
        is_bottom = row == n_rows - 1 or idx + n_cols >= n_classes
        
        # Only show x-axis label and ticks on bottom row
        if is_bottom:
            ax.set_xlabel('Wavelength (nm)', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.tick_params(axis='x', labelbottom=False, bottom=False)
        
        # Only show y-axis label and ticks on leftmost column
        if col == 0:
            ax.set_ylabel('Reflectance', fontsize=AXIS_LABEL_FONTSIZE)
        else:
            ax.tick_params(axis='y', labelleft=False, left=False)
        
        # Set tick label font size
        ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
        
        ax.set_xlim(400, 2300)
        ax.set_ylim(row_y_min.get(row, 0), row_y_max.get(row, 0.6))
        ax.legend(fontsize=LEGEND_FONTSIZE, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3, color='#cccccc')
        
        # Style spines
        for spine in ax.spines.values():
            spine.set_color('#cccccc')
    
    # Hide unused subplots
    for idx in range(n_classes, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    # fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save with _separate suffix
    sep_path = output_path.parent / f"{output_path.stem}_by_class{output_path.suffix}"
    plt.savefig(sep_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved per-class spectral profile plot to {sep_path}")


def center_crop(data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Center crop a (C, H, W) array to target shape."""
    _, h, w = data.shape
    th, tw = target_shape
    start_h = (h - th) // 2
    start_w = (w - tw) // 2
    return data[:, start_h:start_h+th, start_w:start_w+tw]


def top_left_crop(data: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Top-left crop a (C, H, W) array to target shape."""
    th, tw = target_shape
    return data[:, :th, :tw]


def find_dem_visualization(tile_root: Path, copgen_exp: str) -> Optional[Path]:
    """Find DEM visualization from copgen experiment."""
    dem_path = tile_root / "outputs" / "copgen" / copgen_exp / "visualisations" / "DEM_DEM" / "visualisations-0" / "band_0_gray_raw.png"
    if dem_path.exists():
        return dem_path
    # Fallback to terramind if copgen doesn't have it
    return None

def find_copgen_s2l2a_visualization(tile_root: Path, copgen_exp: str) -> Optional[Path]:
    """Find S2L2A visualization from copgen experiment."""
    s2l2a_path = tile_root / "outputs" / "copgen" / copgen_exp / "visualisations" / "S2L2A_B02_B03_B04_B08" / "visualisations-0" / "composite_0_1_2_raw.png"
    if s2l2a_path.exists():
        return s2l2a_path
    return None

def find_terramind_s2l2a_visualization(tile_root: Path, tile_id: str, terramind_exp: str) -> Optional[Path]:
    """Find S2L2A visualization from terramind experiment."""
    s2l2a_path = tile_root / "outputs" / "terramind" / terramind_exp / "visualisations" / tile_id / "pred_S2L2A.png"
    if s2l2a_path.exists():
        return s2l2a_path
    return None

def find_GT_s2l2a_visualization(tile_root: Path, tile_id: str, terramind_exp: str) -> Optional[Path]:
    """Find S2L2A visualization from terramind experiment."""
    s2l2a_path = tile_root / "outputs" / "terramind" / terramind_exp / "visualisations" / tile_id / "gt_S2L2A.png"
    if s2l2a_path.exists():
        return s2l2a_path
    return None


def annotate_s2l2a_with_pixel_frames(
    img_path: Path,
    output_path: Path,
    pixels: Dict[int, Tuple[int, int]],
    frame_size: int = 12,
    frame_width: int = 3
):
    """
    Annotate S2L2A image with colored frames around sampled pixel locations.
    
    Args:
        img_path: Path to input S2L2A PNG
        output_path: Path to save annotated PNG
        pixels: Dict of class_idx -> (row, col)
        frame_size: Size of the square frame (e.g., 3 for 3x3, 5 for 5x5)
        frame_width: Width of the frame border (default 1 pixel)
    """
    img = Image.open(img_path)
    img_array = np.array(img)
    h, w = img_array.shape[:2]
    
    # Calculate the radius from center pixel
    radius = frame_size // 2
    
    for class_idx, (row, col) in pixels.items():
        class_info = LULC_CLASSES.get(class_idx, {"name": f"Class {class_idx}", "color": "#888888"})
        # Convert hex color to RGB
        hex_color = class_info.get("strong-color", class_info["color"])
        rgb = tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        # Draw frame around the pixel
        # We draw the border pixels, skipping the interior
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                # Skip the center pixel(s) - only draw the frame border
                if abs(dr) < radius - frame_width + 1 and abs(dc) < radius - frame_width + 1:
                    continue
                r, c = row + dr, col + dc
                # Check bounds
                if 0 <= r < h and 0 <= c < w:
                    if img_array.ndim == 3:
                        img_array[r, c, :3] = rgb
                    else:
                        # Grayscale - convert to RGB first
                        pass
    
    Image.fromarray(img_array).save(output_path)


def find_lulc_visualization(tile_root: Path, copgen_exp: str) -> Optional[Path]:
    """Find LULC visualization from copgen experiment."""
    lulc_path = tile_root / "outputs" / "copgen" / copgen_exp / "visualisations" / "LULC_LULC" / "visualisations-0" / "band_0_raw.png"
    if lulc_path.exists():
        return lulc_path
    return None


def annotate_lulc_with_pixels(
    lulc_img_path: Path,
    output_path: Path,
    pixels_per_model: Dict[str, Dict[int, Tuple[int, int]]],
    lulc_shape: Tuple[int, int],
    target_shape: Tuple[int, int],
    title: str = "LULC with Sampled Pixels"
):
    """
    Create annotated LULC visualization with pixel markers.
    
    Args:
        lulc_img_path: Path to raw LULC PNG
        output_path: Where to save annotated image
        pixels_per_model: Dict of model_name -> {class_idx -> (row, col)}
        lulc_shape: Original LULC array shape (H, W)
        target_shape: Target cropped shape used for sampling
        title: Title for the plot
    """
    # Load the LULC image
    lulc_img = Image.open(lulc_img_path)
    img_array = np.array(lulc_img)
    img_h, img_w = img_array.shape[:2]
    
    # The raw LULC PNG from copgen visualizations is already the center-cropped region
    # So if img size matches target_shape, pixel coords map directly
    # Otherwise we need to scale
    target_h, target_w = target_shape
    
    # Check if the image is already the cropped region or the full LULC
    if img_h == target_h and img_w == target_w:
        # Image IS the cropped region - direct mapping
        scale_h = 1.0
        scale_w = 1.0
        offset_h = 0
        offset_w = 0
        is_cropped_view = True
    else:
        # Image is full LULC - need to calculate offset and scale
        lulc_h, lulc_w = lulc_shape
        offset_h = (lulc_h - target_h) // 2
        offset_w = (lulc_w - target_w) // 2
        scale_h = img_h / lulc_h
        scale_w = img_w / lulc_w
        is_cropped_view = False
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3, 3), facecolor='white')
    ax.imshow(img_array)
    
    def get_annotation_offset(x, y, img_w, img_h, margin=0.2):
        """Calculate annotation offset to keep text inside image bounds."""
        # margin is fraction of image size to consider as "near edge"
        edge_x = img_w * margin
        edge_y = img_h * margin
        
        # Default: place text to upper-left of marker
        offset_x, offset_y = -10, -20
        ha, va = 'right', 'bottom'
        
        # Adjust horizontal position
        if x < edge_x:
            # Near left edge: place text to the right
            offset_x = 10
            ha = 'left'
        elif x > img_w - edge_x:
            # Near right edge: place text to the left
            offset_x = -10
            ha = 'right'
        
        # Adjust vertical position
        if y < edge_y:
            # Near top edge: place text below
            offset_y = 20
            va = 'top'
        elif y > img_h - edge_y:
            # Near bottom edge: place text above
            offset_y = -20
            va = 'bottom'
        
        return (offset_x, offset_y), ha, va
    
    # Model markers and colors
    model_styles = {
        "Ground Truth": {"marker": "o", "color": "#27ae60", "size": 300},
        "Terramind": {"marker": "s", "color": "#2980b9", "size": 300},
        "Copgen": {"marker": "^", "color": "#c0392b", "size": 300},
        "Shared": {"marker": "*", "color": "#f39c12", "size": 400},
    }
    
    # Check if all models use same pixels
    all_same = True
    reference_pixels = None
    for model_name, pixels in pixels_per_model.items():
        if reference_pixels is None:
            reference_pixels = pixels
        elif pixels != reference_pixels:
            all_same = False
            break
    
    if all_same and reference_pixels:
        # All models use same pixels - show single marker
        for class_idx, (row, col) in reference_pixels.items():
            if is_cropped_view:
                # Direct mapping - pixel coords are already in image space
                img_y = row
                img_x = col
            else:
                # Convert from target crop coordinates to original LULC coordinates
                lulc_row = row + offset_h
                lulc_col = col + offset_w
                # Convert to image coordinates
                img_y = lulc_row * scale_h
                img_x = lulc_col * scale_w
            
            class_info = LULC_CLASSES.get(class_idx, {"name": f"Class {class_idx}", "color": "#888888"})
            ax.scatter(img_x, img_y, marker='*', s=400, 
                      c='white', edgecolors=class_info["strong-color"], 
                      linewidths=2, zorder=10)
            offset, ha, va = get_annotation_offset(img_x, img_y, img_w, img_h)
            ax.annotate(class_info["name"], (img_x, img_y), 
                       xytext=offset, textcoords='offset points',
                       fontsize=9, color='white', fontweight='bold',
                       ha=ha, va=va,
                       bbox=dict(boxstyle='round,pad=0.2',
                       facecolor=class_info["color"], alpha=0.8))
    else:
        # Different pixels per model - show separate markers with offset
        offsets = {"Terramind": (-5, 0), "Copgen": (5, 0), "Ground Truth": (0, -5)}
        
        for model_name, pixels in pixels_per_model.items():
            style = model_styles.get(model_name, model_styles["Shared"])
            dx, dy = offsets.get(model_name, (0, 0))
            
            for class_idx, (row, col) in pixels.items():
                if is_cropped_view:
                    img_y = row + dy
                    img_x = col + dx
                else:
                    lulc_row = row + offset_h
                    lulc_col = col + offset_w
                    img_y = lulc_row * scale_h + dy
                    img_x = lulc_col * scale_w + dx
                
                class_info = LULC_CLASSES.get(class_idx, {"name": f"Class {class_idx}", "color": "#888888"})
                ax.scatter(img_x, img_y, marker=style["marker"], s=style["size"],
                          c=style["color"], edgecolors='white', linewidths=2, zorder=10)
                
                # Add class label for each marker with smart positioning
                offset, ha, va = get_annotation_offset(img_x, img_y, img_w, img_h)
                ax.annotate(class_info["name"], (img_x, img_y), 
                           xytext=offset, textcoords='offset points',
                           fontsize=8, color='white', fontweight='bold',
                           ha=ha, va=va,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor=class_info["color"], alpha=0.7))
        
        # Add model legend
        legend_handles = []
        for model_name in pixels_per_model.keys():
            style = model_styles.get(model_name, model_styles["Shared"])
            handle = plt.Line2D([0], [0], marker=style["marker"], color='w',
                               markerfacecolor=style["color"], markersize=12,
                               markeredgecolor='white', markeredgewidth=1.5,
                               label=model_name, linestyle='None')
            legend_handles.append(handle)
        ax.legend(handles=legend_handles, loc='upper right', fontsize=10, framealpha=0.9)
    
    # Only draw crop rectangle if showing full LULC
    if not is_cropped_view:
        from matplotlib.patches import Rectangle
        rect_y = offset_h * scale_h
        rect_x = offset_w * scale_w
        rect_h = target_h * scale_h
        rect_w = target_w * scale_w
        rect = Rectangle((rect_x, rect_y), rect_w, rect_h, 
                         linewidth=2, edgecolor='white', facecolor='none',
                         linestyle='--', alpha=0.8)
        ax.add_patch(rect)
    
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"  Saved annotated LULC to {output_path}")


def get_tile_id(tile_root: Path) -> str:
    """Extract tile ID from tile root path."""
    return tile_root.name


def generate_all_plots(
    tile_root: Path,
    terramind_exp: str,
    copgen_exp: str,
    metric: str = "mae",
    max_pixels: int = 1000,
    seed: int = 42,
    include_terramind: bool = False,
    skip_classes: set = None
):
    """
    Generate all spectral profile plots and supporting visualizations.
    
    Creates output structure:
        ./paper_figures/spectral_profiles/<tile_id>_<seed>/
            ├── best-copgen/
            │   ├── spectral_profiles.png
            │   ├── spectral_profiles_by_class.png
            │   ├── DEM_raw.png
            │   └── LULC_annotated.png
            ├── best-terramind/
            │   └── ...
            └── best-per-model/
                └── ...
    
    Args:
        skip_classes: Set of LULC class indices to skip (default: None = skip none)
    """
    if skip_classes is None:
        skip_classes = set()
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    tile_id = get_tile_id(tile_root)
    base_output_dir = Path("./paper_figures/spectral_profiles") / f"{tile_id}_seed{seed}"
    
    print(f"\n{'='*60}")
    print(f"Generating all spectral profile plots for tile: {tile_id}")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*60}\n")
    
    # Load all data once
    print("Loading data...")
    copgen_path = tile_root / "outputs" / "copgen" / copgen_exp
    
    lulc_path = find_lulc_file(tile_root)
    lulc = load_lulc(lulc_path)
    lulc_shape = lulc.shape
    print(f"  LULC shape: {lulc_shape}, classes: {np.unique(lulc)}")
    
    copgen_data = load_copgen_s2l2a(copgen_path)
    
    # Load Terramind data only if comparison mode is enabled
    terramind_data = None
    terramind_refl = None
    if include_terramind:
        terramind_path = tile_root / "outputs" / "terramind" / terramind_exp
        terramind_data = load_terramind_s2l2a(terramind_path)
        min_h = min(terramind_data.shape[1], copgen_data.shape[1])
        min_w = min(terramind_data.shape[2], copgen_data.shape[2])
        target_shape = (min_h, min_w)
        terramind_data = center_crop(terramind_data, target_shape)
        terramind_refl = normalize_reflectance(terramind_data)
    else:
        target_shape = (copgen_data.shape[1], copgen_data.shape[2])
    
    print(f"  Target shape: {target_shape}")
    
    # Both Terramind and Copgen generate center crops of the original tile
    copgen_data = center_crop(copgen_data, target_shape)
    copgen_refl = normalize_reflectance(copgen_data)
    
    gt_data = load_ground_truth_s2l2a(tile_root)
    if gt_data is None:
        raise ValueError("Ground truth required for best-pixel selection")
    gt_data = center_crop(gt_data, target_shape)
    gt_refl = normalize_reflectance(gt_data)
    
    # Find input visualizations
    dem_vis_path = find_dem_visualization(tile_root, copgen_exp)
    lulc_vis_path = find_lulc_visualization(tile_root, copgen_exp)
    copgen_s2l2a_vis_path = find_copgen_s2l2a_visualization(tile_root, copgen_exp)
    terramind_s2l2a_vis_path = None
    gt_s2l2a_vis_path = None
    if include_terramind:
        terramind_s2l2a_vis_path = find_terramind_s2l2a_visualization(tile_root, tile_id, terramind_exp)
        gt_s2l2a_vis_path = find_GT_s2l2a_visualization(tile_root, tile_id, terramind_exp)
    
    # Define modes to generate based on whether Terramind comparison is enabled
    if include_terramind:
        modes = ["best-copgen", "best-terramind", "best-per-model"]
    else:
        modes = ["best-copgen"]
    
    for mode in modes:
        print(f"\n{'-'*40}")
        print(f"Generating: {mode}")
        print(f"{'-'*40}")
        
        mode_dir = base_output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        
        # Select pixels based on mode
        profiles = {}
        pixels_for_annotation = {}
        
        if mode == "best-copgen":
            pixels = select_best_matching_pixels(
                lulc, copgen_refl, gt_refl, target_shape, metric, max_pixels,
                skip_classes=skip_classes
            )
            print(f"  Best pixels for Copgen:")
            for class_idx, (r, c) in pixels.items():
                error = compute_spectral_error(copgen_refl[:, r, c], gt_refl[:, r, c], metric)
                print(f"    {LULC_CLASSES[class_idx]['name']}: ({r}, {c}), {metric}={error:.4f}")
            
            if include_terramind:
                profiles["Terramind"] = {c: extract_spectral_profile(terramind_refl, r, col) 
                                         for c, (r, col) in pixels.items()}
            profiles["Copgen"] = {c: extract_spectral_profile(copgen_refl, r, col) 
                                 for c, (r, col) in pixels.items()}
            profiles["Ground Truth"] = {c: extract_spectral_profile(gt_refl, r, col) 
                                        for c, (r, col) in pixels.items()}
            pixels_for_annotation = {"Shared": pixels}
            # title_suffix = "(Best Pixel for Copgen)"
            
        elif mode == "best-terramind":
            pixels = select_best_matching_pixels(
                lulc, terramind_refl, gt_refl, target_shape, metric, max_pixels,
                skip_classes=skip_classes
            )
            print(f"  Best pixels for Terramind:")
            for class_idx, (r, c) in pixels.items():
                error = compute_spectral_error(terramind_refl[:, r, c], gt_refl[:, r, c], metric)
                print(f"    {LULC_CLASSES[class_idx]['name']}: ({r}, {c}), {metric}={error:.4f}")
            
            profiles["Terramind"] = {c: extract_spectral_profile(terramind_refl, r, col) 
                                     for c, (r, col) in pixels.items()}
            profiles["Copgen"] = {c: extract_spectral_profile(copgen_refl, r, col) 
                                 for c, (r, col) in pixels.items()}
            profiles["Ground Truth"] = {c: extract_spectral_profile(gt_refl, r, col) 
                                        for c, (r, col) in pixels.items()}
            pixels_for_annotation = {"Shared": pixels}
            # title_suffix = "(Best Pixel for Terramind)"
            
        elif mode == "best-per-model":
            terramind_pixels = select_best_matching_pixels(
                lulc, terramind_refl, gt_refl, target_shape, metric, max_pixels,
                skip_classes=skip_classes
            )
            copgen_pixels = select_best_matching_pixels(
                lulc, copgen_refl, gt_refl, target_shape, metric, max_pixels,
                skip_classes=skip_classes
            )
            
            print(f"  Best pixels for Terramind:")
            for class_idx, (r, c) in terramind_pixels.items():
                error = compute_spectral_error(terramind_refl[:, r, c], gt_refl[:, r, c], metric)
                print(f"    {LULC_CLASSES[class_idx]['name']}: ({r}, {c}), {metric}={error:.4f}")
            
            print(f"  Best pixels for Copgen:")
            for class_idx, (r, c) in copgen_pixels.items():
                error = compute_spectral_error(copgen_refl[:, r, c], gt_refl[:, r, c], metric)
                print(f"    {LULC_CLASSES[class_idx]['name']}: ({r}, {c}), {metric}={error:.4f}")
            
            profiles["Terramind"] = {c: extract_spectral_profile(terramind_refl, r, col) 
                                     for c, (r, col) in terramind_pixels.items()}
            profiles["Copgen"] = {c: extract_spectral_profile(copgen_refl, r, col) 
                                 for c, (r, col) in copgen_pixels.items()}
            # GT uses copgen pixels for display
            profiles["Ground Truth"] = {c: extract_spectral_profile(gt_refl, r, col) 
                                        for c, (r, col) in copgen_pixels.items()}
            pixels_for_annotation = {"Terramind": terramind_pixels, "Copgen": copgen_pixels}
            # title_suffix = "(Best Pixel per Model)"
        
        # Generate spectral plots
        output_path = mode_dir / "spectral_profiles.png"
        plot_spectral_profiles(profiles, output_path) 
                            #   title="S2L2A Spectral Profiles")
                            #   title=f"Spectral Profiles {title_suffix}")
        plot_spectral_profiles_separate(profiles, output_path)
                                    #    title="S2L2A Spectral Profiles by Class")
                            #   title=f"Spectral Profiles {title_suffix}")
        
        # Copy DEM visualization
        if dem_vis_path:
            dem_dest = mode_dir / "DEM_raw.png"
            shutil.copy(dem_vis_path, dem_dest)
            print(f"  Copied DEM visualization to {dem_dest}")
        # Copy LULC visualization
        if lulc_vis_path:
            lulc_dest = mode_dir / "LULC_raw.png"
            shutil.copy(lulc_vis_path, lulc_dest)
            print(f"  Copied LULC visualization to {lulc_dest}")
        
        # Determine which pixels to use for each model's annotation
        if mode == "best-per-model":
            copgen_annotation_pixels = copgen_pixels
            terramind_annotation_pixels = terramind_pixels
        else:
            # In best-copgen and best-terramind modes, all use the same pixels
            copgen_annotation_pixels = pixels
            terramind_annotation_pixels = pixels
        
        # Copy Copgen S2L2A visualization (raw and annotated)
        if copgen_s2l2a_vis_path:
            copgen_s2l2a_dest = mode_dir / "Copgen_S2L2A_raw.png"
            shutil.copy(copgen_s2l2a_vis_path, copgen_s2l2a_dest)
            print(f"  Copied Copgen S2L2A visualization to {copgen_s2l2a_dest}")
            
            # Create annotated version with pixel frames
            copgen_s2l2a_annotated = mode_dir / "Copgen_S2L2A_annotated.png"
            annotate_s2l2a_with_pixel_frames(
                copgen_s2l2a_vis_path, copgen_s2l2a_annotated, copgen_annotation_pixels
            )
            print(f"  Created annotated Copgen S2L2A to {copgen_s2l2a_annotated}")
        
        # Copy Terramind S2L2A visualization (center crop from 256x256 to 192x192)
        if terramind_s2l2a_vis_path:
            terramind_s2l2a_dest = mode_dir / "Terramind_S2L2A_raw.png"
            terramind_img = Image.open(terramind_s2l2a_vis_path)
            img_array = np.array(terramind_img)
            h, w = img_array.shape[:2]
            crop_h, crop_w = 192, 192
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            cropped = img_array[start_h:start_h+crop_h, start_w:start_w+crop_w]
            Image.fromarray(cropped).save(terramind_s2l2a_dest)
            print(f"  Saved center-cropped Terramind S2L2A visualization to {terramind_s2l2a_dest}")
            
            # Create annotated version with pixel frames
            terramind_s2l2a_annotated = mode_dir / "Terramind_S2L2A_annotated.png"
            annotate_s2l2a_with_pixel_frames(
                terramind_s2l2a_dest, terramind_s2l2a_annotated, terramind_annotation_pixels
            )
            print(f"  Created annotated Terramind S2L2A to {terramind_s2l2a_annotated}")
        
        # Copy GT S2L2A visualization (center crop from 256x256 to 192x192)
        if gt_s2l2a_vis_path:
            gt_s2l2a_dest = mode_dir / "GT_S2L2A_raw.png"
            gt_img = Image.open(gt_s2l2a_vis_path)
            img_array = np.array(gt_img)
            h, w = img_array.shape[:2]
            crop_h, crop_w = 192, 192
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            cropped = img_array[start_h:start_h+crop_h, start_w:start_w+crop_w]
            Image.fromarray(cropped).save(gt_s2l2a_dest)
            print(f"  Saved center-cropped GT S2L2A visualization to {gt_s2l2a_dest}")
            
            # Create annotated version with pixel frames (use copgen pixels since it's our model)
            gt_s2l2a_annotated = mode_dir / "GT_S2L2A_annotated.png"
            annotate_s2l2a_with_pixel_frames(
                gt_s2l2a_dest, gt_s2l2a_annotated, copgen_annotation_pixels
            )
            print(f"  Created annotated GT S2L2A to {gt_s2l2a_annotated}")
        
        # Create annotated LULC visualization
        if lulc_vis_path:
            lulc_dest = mode_dir / "LULC_annotated.png"
            annotate_lulc_with_pixels(
                lulc_vis_path, lulc_dest,
                pixels_for_annotation, lulc_shape, target_shape,
                # title="LULC with Sampled Pixels"
                # title=f"LULC with Sampled Pixels {title_suffix}"
            )
    
    print(f"\n{'='*60}")
    print(f"All plots generated successfully!")
    print(f"Output directory: {base_output_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate spectral profile plots comparing model outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # GT vs Copgen only (default)
    # Output: ./paper_figures/spectral_profiles/<tile_id>/best-copgen/
    python spectral_profiles.py \\
        --tile-root ./paper_figures/paper_figures_datasets/one_tile_datasets_DEM_LULC_to_S2L2A/143D_1481R \\
        --terramind-exp input_DEM_LULC_output_S2L2A_seed_111 \\
        --copgen-exp input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_111
    
    # Include Terramind comparison (generates all modes)
    # Output: ./paper_figures/spectral_profiles/<tile_id>/{best-copgen,best-terramind,best-per-model}/
    python spectral_profiles.py \\
        --tile-root ... --terramind-exp ... --copgen-exp ... \\
        --comparison
    
    # Single mode with custom output
    python spectral_profiles.py \\
        --tile-root ... --terramind-exp ... --copgen-exp ... \\
        --single-mode best-copgen \\
        --output custom_output.png
        """
    )
    parser.add_argument("--tile-root", type=str, required=True,
                       help="Root directory for the tile (contains terramind_inputs, outputs, etc.)")
    parser.add_argument("--terramind-exp", type=str, default=None,
                       help="Terramind experiment name (folder name under outputs/terramind/). Required if --comparison is used.")
    parser.add_argument("--copgen-exp", type=str, required=True,
                       help="Copgen experiment name (folder name under outputs/copgen/)")
    parser.add_argument("--single-mode", type=str, default=None,
                       choices=["center", "best-per-model", "best-copgen", "best-terramind"],
                       help="""Run in single mode with specific pixel selection.
                           If not specified, generates ALL modes automatically.""")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path (only used with --single-mode)")
    parser.add_argument("--include-gt", action="store_true", default=True,
                       help="Include ground truth in the comparison (default: True)")
    parser.add_argument("--no-gt", action="store_true",
                       help="Exclude ground truth from the comparison")
    parser.add_argument("--metric", type=str, default="mae",
                       choices=["mae", "mse", "cosine"],
                       help="Metric for best-pixel selection (default: mae)")
    parser.add_argument("--max-pixels", type=int, default=1000,
                       help="Max pixels to search per class for best-pixel selection (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--comparison", action="store_true",
                       help="Include Terramind in comparison. If not set, only GT vs Copgen is shown.")
    parser.add_argument("--skip", type=str, nargs='*', default=[],
                       help="LULC class names to skip (e.g., --skip flooded_vegetation trees clouds). "
                            "Names are case-insensitive and use underscores for spaces.")
    
    args = parser.parse_args()
    
    # Validate: --terramind-exp is required if --comparison is used
    if args.comparison and args.terramind_exp is None:
        parser.error("--terramind-exp is required when --comparison is used")
    
    # Parse skip classes
    skip_classes = set()
    if args.skip:
        skip_classes = parse_skip_classes(args.skip)
        print(f"Skipping LULC classes: {[LULC_CLASSES[i]['name'] for i in skip_classes]}")
    
    tile_root = Path(args.tile_root)
    if not tile_root.exists():
        raise FileNotFoundError(f"Tile root not found: {tile_root}")
    
    # If no single mode specified, generate ALL plots
    if args.single_mode is None:
        generate_all_plots(
            tile_root=tile_root,
            terramind_exp=args.terramind_exp,
            copgen_exp=args.copgen_exp,
            metric=args.metric,
            max_pixels=args.max_pixels,
            skip_classes=skip_classes,
            seed=args.seed,
            include_terramind=args.comparison
        )
        return
    
    # Single mode - original behavior
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    include_terramind = args.comparison
    pixel_selection = args.single_mode
    
    # Validate mode when Terramind comparison is disabled
    if not include_terramind and pixel_selection in ["best-terramind", "best-per-model"]:
        print(f"WARNING: Mode '{pixel_selection}' requires --comparison flag. Falling back to 'best-copgen'.")
        pixel_selection = "best-copgen"
    
    output_path = Path(args.output) if args.output else Path(f"spectral_profiles_{pixel_selection}.png")
    
    copgen_path = tile_root / "outputs" / "copgen" / args.copgen_exp
    
    print(f"Loading data from tile: {tile_root}")
    if include_terramind:
        print(f"  Terramind experiment: {args.terramind_exp}")
    print(f"  Copgen experiment: {args.copgen_exp}")
    print(f"  Pixel selection: {pixel_selection}")
    print(f"  Include Terramind: {include_terramind}")
    
    # Load LULC for pixel selection
    lulc_path = find_lulc_file(tile_root)
    print(f"  LULC file: {lulc_path}")
    lulc = load_lulc(lulc_path)
    print(f"  LULC shape: {lulc.shape}, unique classes: {np.unique(lulc)}")
    
    # Load S2L2A data
    print("Loading Copgen S2L2A...")
    copgen_data = load_copgen_s2l2a(copgen_path)
    print(f"  Copgen shape: {copgen_data.shape}")
    
    # Load Terramind data only if comparison mode is enabled
    terramind_data = None
    terramind_refl = None
    if include_terramind:
        terramind_path = tile_root / "outputs" / "terramind" / args.terramind_exp
        print("Loading Terramind S2L2A...")
        terramind_data = load_terramind_s2l2a(terramind_path)
        print(f"  Terramind shape: {terramind_data.shape}")
        min_h = min(terramind_data.shape[1], copgen_data.shape[1])
        min_w = min(terramind_data.shape[2], copgen_data.shape[2])
        target_shape = (min_h, min_w)
        terramind_data = center_crop(terramind_data, target_shape)
        terramind_refl = normalize_reflectance(terramind_data)
    else:
        target_shape = (copgen_data.shape[1], copgen_data.shape[2])
    
    print(f"Using target shape: {target_shape}")
    
    # Both Terramind and Copgen generate center crops of the original tile
    copgen_data = center_crop(copgen_data, target_shape)
    
    # Normalize to reflectance
    copgen_refl = normalize_reflectance(copgen_data)
    
    # Load ground truth (required for best-pixel selection)
    print("Loading ground truth S2L2A...")
    gt_data = load_ground_truth_s2l2a(tile_root)
    gt_refl = None
    if gt_data is not None:
        print(f"  GT shape: {gt_data.shape}")
        gt_data = center_crop(gt_data, target_shape)
        gt_refl = normalize_reflectance(gt_data)
    else:
        print("  Ground truth not found.")
        if pixel_selection != "center":
            print("  WARNING: Best-pixel selection requires ground truth. Falling back to center.")
            pixel_selection = "center"
    
    include_gt = args.include_gt and not args.no_gt and gt_refl is not None
    
    # Select pixels based on strategy
    profiles = {}
    
    if pixel_selection == "center":
        # Original behavior: center-most pixel per class
        pixels = select_representative_pixels(lulc, target_shape, skip_classes=skip_classes)
        print(f"\nSelected CENTER pixels for classes: {list(pixels.keys())}")
        for class_idx, (r, c) in pixels.items():
            print(f"  {LULC_CLASSES[class_idx]['name']}: pixel ({r}, {c})")
        
        # All sources use same pixels
        if include_terramind:
            profiles["Terramind"] = {c: extract_spectral_profile(terramind_refl, r, col) 
                                     for c, (r, col) in pixels.items()}
        profiles["Copgen"] = {c: extract_spectral_profile(copgen_refl, r, col) 
                             for c, (r, col) in pixels.items()}
        if include_gt:
            profiles["Ground Truth"] = {c: extract_spectral_profile(gt_refl, r, col) 
                                        for c, (r, col) in pixels.items()}
    
    elif pixel_selection == "best-per-model":
        # Best matching pixel for each model independently
        # This mode requires --comparison flag (validated above)
        print(f"\nFinding BEST-PER-MODEL pixels (metric: {args.metric})...")
        model_pixels = select_best_matching_pixels_for_models(
            lulc,
            {"Terramind": terramind_refl, "Copgen": copgen_refl},
            gt_refl,
            target_shape,
            args.metric,
            args.max_pixels,
            skip_classes=skip_classes
        )
        
        # Each model uses its own best pixels
        for model_name, pixels in model_pixels.items():
            print(f"\n  {model_name} best pixels:")
            for class_idx, (r, c) in pixels.items():
                print(f"    {LULC_CLASSES[class_idx]['name']}: pixel ({r}, {c})")
            
            model_data = terramind_refl if model_name == "Terramind" else copgen_refl
            profiles[model_name] = {c: extract_spectral_profile(model_data, r, col) 
                                   for c, (r, col) in pixels.items()}
        
        # GT uses union of all pixels found (for reference)
        if include_gt:
            # Use Copgen pixels for GT display (our model)
            copgen_pixels = model_pixels.get("Copgen", {})
            profiles["Ground Truth"] = {c: extract_spectral_profile(gt_refl, r, col) 
                                        for c, (r, col) in copgen_pixels.items()}
    
    elif pixel_selection in ["best-copgen", "best-terramind"]:
        # Best pixel for one model, same location for all
        favor_model = "Copgen" if pixel_selection == "best-copgen" else "Terramind"
        favor_data = copgen_refl if favor_model == "Copgen" else terramind_refl
        
        print(f"\nFinding BEST pixels favoring {favor_model} (metric: {args.metric})...")
        pixels = select_best_matching_pixels(
            lulc, favor_data, gt_refl, target_shape, args.metric, args.max_pixels,
            skip_classes=skip_classes
        )
        
        print(f"\n  Best pixels for {favor_model}:")
        for class_idx, (r, c) in pixels.items():
            # Also compute error for this pixel
            pred = favor_data[:, r, c]
            gt = gt_refl[:, r, c]
            error = compute_spectral_error(pred, gt, args.metric)
            print(f"    {LULC_CLASSES[class_idx]['name']}: pixel ({r}, {c}), {args.metric}={error:.4f}")
        
        # All sources use the same pixels
        if include_terramind:
            profiles["Terramind"] = {c: extract_spectral_profile(terramind_refl, r, col) 
                                     for c, (r, col) in pixels.items()}
        profiles["Copgen"] = {c: extract_spectral_profile(copgen_refl, r, col) 
                             for c, (r, col) in pixels.items()}
        if include_gt:
            profiles["Ground Truth"] = {c: extract_spectral_profile(gt_refl, r, col) 
                                        for c, (r, col) in pixels.items()}
    
    # Generate plots
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine title based on selection mode
    title_suffix = {
        "center": "(Center Pixel)",
        "best-per-model": "(Best Pixel per Model)",
        "best-copgen": "(Best Pixel for Copgen)",
        "best-terramind": "(Best Pixel for Terramind)",
    }.get(pixel_selection, "")
    
    # Main combined plot
    plot_spectral_profiles(profiles, output_path, 
                          title=f"Spectral Profiles of Features of Interest {title_suffix}")
    
    # Per-class subplot version
    plot_spectral_profiles_separate(profiles, output_path,
                                   title=f"Spectral Profiles Comparison {title_suffix}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

