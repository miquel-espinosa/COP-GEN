from __future__ import annotations

import argparse
import colorsys
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
import rioxarray as rxr

"""
Command for generating the plots

TILES=("143D_1481R" "95U_112R" "195D_669L" "211D_500R" "215U_1019L" "248U_978R" "250U_409R" "256U_1125L" "272D_1525R")

for TILE in "${TILES[@]}"; do
    python3 paper_figures/narrowing_distributions.py \
        ./paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/$TILE 
done
"""



S2_BAND_ORDER = {
    "S2L2A": [
        "B01.tif",
        "B02.tif",
        "B03.tif",
        "B04.tif",
        "B05.tif",
        "B06.tif",
        "B07.tif",
        "B08.tif",
        "B8A.tif",
        "B09.tif",
        "B11.tif",
        "B12.tif",
    ]
}

IGNORE_SCENARIOS = ["DEM+LULC+S1RTC+timestamps+S2L1C"]

# Canonical order for input-condition labels in legends
CANONICAL_INPUT_ORDER: List[str] = [
    "DEM",
    "cloud_mask",
    "LULC",
    "S1RTC",
    "timestamps",
    "lat_lon",
    "S2L1C",
]

# Tokens that can appear but we don't want to treat as inputs for the legend
OUTPUT_MODALITIES: set[str] = {"S2L2A"}  # outputs often include these

# Mapping from TerraMind naming to canonical naming
TERRAMIND_TO_CANONICAL: Dict[str, str] = {
    "coords": "lat_lon",
    # TerraMind doesn't support timestamps
}

# Number of bands in S2L2A stacked TIF
S2L2A_NUM_BANDS = 12

# Darkening factor for TerraMind colors (0.0 = black, 1.0 = no change)
TERRAMIND_DARKEN_FACTOR = 0.65


def _darken_color(color, factor: float = TERRAMIND_DARKEN_FACTOR):
    """
    Darken a color by reducing its value (brightness) in HSV space.
    
    Args:
        color: Any color format accepted by matplotlib (RGB tuple, hex string, named color)
        factor: How much to darken (0.0 = black, 1.0 = no change). Default 0.65.
    
    Returns:
        RGB tuple of the darkened color
    """
    # Convert to RGB tuple (values 0-1)
    rgb = mcolors.to_rgb(color)
    # Convert to HSV
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    # Reduce value (brightness)
    v_new = v * factor
    # Convert back to RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v_new)
    return (r, g, b)


def _label_without_timestamps(label: str) -> str:
    """
    Remove 'timestamps' from a label to create a match key.
    This allows matching CopGen scenarios (with timestamps) to TerraMind scenarios (without).
    
    E.g., "DEM+LULC+S1RTC+timestamps+lat_lon" -> "DEM+LULC+S1RTC+lat_lon"
    """
    parts = label.split("+")
    filtered = [p for p in parts if p != "timestamps"]
    return "+".join(filtered)


@dataclass(frozen=True)
class PlotConfig:
    kde: bool = True
    dpi: int = 300
    figsize: Tuple[float, float] = (18.0, 12.0)  # 3x4 grid
    style: str = "white"
    palette_name: str = "colorblind"
    combined: bool = False  # Whether to also generate combined distribution plots
    terramind: bool = False  # Whether to include TerraMind comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot S2L2A value distributions (histogram + KDE) for a tile, "
            "showing how distributions narrow as more input conditions are used."
        )
    )
    parser.add_argument(
        "tile_path",
        type=str,
        help="Path to tile experiments, e.g. paper_figures/paper_figures_datasets/one_tile_distribution_narrowing/143D_1481R",
    )
    parser.add_argument(
        "--no-kde", action="store_true", help="Disable KDE overlay (histograms only)"
    )
    parser.add_argument(
        "--dpi", type=int, default=PlotConfig.dpi, help="Figure DPI when saving"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the figure. Defaults to <tile_path>/narrowing_histograms",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure window after saving",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Also generate combined distribution plots (all scenarios in one figure per band)",
    )
    parser.add_argument(
        "--terramind",
        action="store_true",
        help="Include TerraMind comparison in the plots (requires terramind outputs to exist)",
    )
    return parser.parse_args()


def _canonical_label_from_inputs(tokens: Iterable[str]) -> str:
    present = {t for t in tokens if t in CANONICAL_INPUT_ORDER}
    ordered = [t for t in CANONICAL_INPUT_ORDER if t in present]
    return "+".join(ordered)


def _parse_inputs_from_folder_name(dirname: str) -> Optional[str]:
    """
    Folder naming examples:
      - input_DEM_cloud_mask_output_LULC_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_111
      - input_DEM_LULC_cloud_mask_output_S1RTC_S2L1C_S2L2A_lat_lon_timestamps_seed_12
      - input_DEM_LULC_S1RTC_cloud_mask_output_S2L1C_S2L2A_lat_lon_timestamps_seed_12
      - input_DEM_LULC_S1RTC_cloud_mask_timestamps_output_S2L1C_S2L2A_lat_lon_seed_12
      - input_DEM_LULC_S1RTC_S2L1C_cloud_mask_timestamps_output_S2L2A_lat_lon_seed_12
    """
    m = re.search(r"input_(.+?)_output", dirname)
    if not m:
        return None
    raw = m.group(1)
    # Replace "lat_lon" with a placeholder to treat it as a single token
    raw = raw.replace("lat_lon", "lat-lon")
    tokens = [t for t in raw.split("_") if t]
    # Convert placeholder back to "lat_lon"
    tokens = [t.replace("lat-lon", "lat_lon") for t in tokens]
    # Ensure no output modalities leak into inputs
    offending = sorted(set(tokens) & OUTPUT_MODALITIES)
    if offending:
        print(f"\033[93mWarning:\033[0m Found output modalities in inputs ({offending}) for folder '{dirname}'. They will be ignored.")
        tokens = [t for t in tokens if t not in OUTPUT_MODALITIES]
    if not tokens:
        return None
    return _canonical_label_from_inputs(tokens)


def _discover_scenarios(copgen_dir: Path) -> Dict[str, List[Path]]:
    scenarios: Dict[str, List[Path]] = defaultdict(list)
    if not copgen_dir.exists():
        return scenarios
    for child in copgen_dir.iterdir():
        if not child.is_dir():
            continue
        label = _parse_inputs_from_folder_name(child.name)
        if not label:
            continue
        if label in IGNORE_SCENARIOS:
            print(f"\033[94mInfo:\033[0m Ignoring scenario '{label}' (user-specified).")
            continue
        scenarios[label].append(child)
    return scenarios


def _parse_inputs_from_terramind_folder_name(dirname: str) -> Optional[str]:
    """
    TerraMind folder naming examples:
      - input_DEM_LULC_output_S2L2A_seed_111
      - input_DEM_LULC_S1RTC_output_S2L2A_seed_12
      - input_DEM_LULC_S1RTC_coords_output_S2L2A_seed_12

    TerraMind uses 'coords' instead of 'lat_lon' and doesn't support 'timestamps'.
    """
    m = re.search(r"input_(.+?)_output", dirname)
    if not m:
        return None
    raw = m.group(1)
    tokens = [t for t in raw.split("_") if t]
    # Map TerraMind naming to canonical naming
    tokens = [TERRAMIND_TO_CANONICAL.get(t, t) for t in tokens]
    # Ensure no output modalities leak into inputs
    offending = sorted(set(tokens) & OUTPUT_MODALITIES)
    if offending:
        print(f"\033[93mWarning:\033[0m Found output modalities in inputs ({offending}) for folder '{dirname}'. They will be ignored.")
        tokens = [t for t in tokens if t not in OUTPUT_MODALITIES]
    if not tokens:
        return None
    return _canonical_label_from_inputs(tokens)


def _discover_terramind_scenarios(terramind_dir: Path) -> Dict[str, List[Path]]:
    """
    Discover TerraMind scenarios from the terramind output directory.
    Groups seeds by input conditions.
    """
    scenarios: Dict[str, List[Path]] = defaultdict(list)
    if not terramind_dir.exists():
        return scenarios
    for child in terramind_dir.iterdir():
        if not child.is_dir():
            continue
        label = _parse_inputs_from_terramind_folder_name(child.name)
        if not label:
            continue
        if label in IGNORE_SCENARIOS:
            print(f"\033[94mInfo:\033[0m Ignoring TerraMind scenario '{label}' (user-specified).")
            continue
        scenarios[label].append(child)
    return scenarios


def _tile_id_parts(tile_path: Path) -> Tuple[str, str]:
    """
    Return (tile_code, tile_id) for a path ending in e.g. .../143D_1481R
    """
    tile_id = tile_path.name
    tile_code = tile_id.split("_")[0] if "_" in tile_id else tile_id
    return tile_code, tile_id


def _find_gt_scene_dir(tile_path: Path) -> Path:
    """
    Expected GT path pattern:
        <tile_path>/Core-S2L2A/<TILE_CODE>/<TILE_ID>/<SCENE_NAME>/
    where <SCENE_NAME> looks like S2B_MSIL2A_... (exact folder name).
    """
    tile_code, tile_id = _tile_id_parts(tile_path)
    base = tile_path / "Core-S2L2A" / tile_code / tile_id
    if not base.exists():
        raise FileNotFoundError(f"Ground-truth base dir not found: {base}")
    scene_dirs = [d for d in base.iterdir() if d.is_dir()]
    if len(scene_dirs) == 0:
        raise FileNotFoundError(f"No GT scene directory under: {base}")
    # Choose the first sorted to be deterministic (usually there's only one)
    scene_dirs.sort()
    return scene_dirs[0]


def _band_paths_in_dir(scene_dir: Path, bands_filenames: Sequence[str]) -> Dict[str, Path]:
    """
    Map band short name (e.g., B02) -> Path to its tif within scene_dir.
    If direct '<band>.tif' is missing, try '<scene>_<band>.tif'.
    """
    mapping: Dict[str, Path] = {}
    scene_name = scene_dir.name
    for fname in bands_filenames:
        band = fname.replace(".tif", "").replace(".TIF", "")
        p = scene_dir / fname
        if not p.exists():
            alt = scene_dir / f"{scene_name}_{band}.tif"
            p = alt if alt.exists() else p
        if p.exists():
            mapping[band] = p
    return mapping


def _read_band_values(
    tif_path: Path,
) -> np.ndarray:
    """
    Read a single-band GeoTIFF, return 1D float array scaled to [0, 1] if uint16.
    Nodata values are masked out.
    """
    if tif_path.exists():
        arr = rxr.open_rasterio(tif_path).squeeze().values
        data = arr.ravel() / 10000.0
        return data.astype(np.float32)
    else:
        return np.array([], dtype=np.float32)


def _gather_generated_values_for_band(
    seed_dirs: List[Path],
    tile_code: str,
    tile_id: str,
    scene_name: str,
    band_fname: str,
) -> np.ndarray:
    band = band_fname.replace(".tif", "").replace(".TIF", "")
    vals: List[np.ndarray] = []
    for seed_dir in seed_dirs:
        gen_scene_dir = seed_dir / "generations" / "Core-S2L2A" / tile_code / tile_id / scene_name
        tif_path = gen_scene_dir / f"{band}.tif"
        v = _read_band_values(tif_path)
        vals.append(v)
    return np.concatenate(vals, axis=0)


def _read_band_from_stacked_tif(
    tif_path: Path,
    band_index: int,
) -> np.ndarray:
    """
    Read a specific band from a stacked GeoTIFF (TerraMind format).
    Returns 1D float array scaled to [0, 1].

    Args:
        tif_path: Path to the stacked TIF file
        band_index: 0-based index of the band to read
    """
    if tif_path.exists():
        arr = rxr.open_rasterio(tif_path)
        # arr shape is (bands, height, width)
        if band_index >= arr.shape[0]:
            print(f"\033[93mWarning:\033[0m Band index {band_index} out of range for {tif_path} (has {arr.shape[0]} bands)")
            return np.array([], dtype=np.float32)
        band_arr = arr[band_index].values
        data = band_arr.ravel() / 10000.0
        return data.astype(np.float32)
    else:
        return np.array([], dtype=np.float32)


def _gather_terramind_values_for_band(
    seed_dirs: List[Path],
    tile_id: str,
    band_index: int,
) -> np.ndarray:
    """
    Gather values for a specific band from TerraMind stacked TIF outputs.

    Args:
        seed_dirs: List of seed directories containing generations
        tile_id: Tile identifier (e.g., "143D_1481R")
        band_index: 0-based index of the band in the stacked TIF
    """
    vals: List[np.ndarray] = []
    for seed_dir in seed_dirs:
        tif_path = seed_dir / "generations" / f"{tile_id}.tif"
        v = _read_band_from_stacked_tif(tif_path, band_index)
        vals.append(v)
    if not vals:
        return np.array([], dtype=np.float32)
    return np.concatenate(vals, axis=0)


#
# Note: we rely on Matplotlib's default binning for histograms, so no custom bin logic here.


def _legend_order(labels: Iterable[str]) -> List[str]:
    """
    Order legend: GT first, then scenarios by number of inputs ascending and canonical order.
    """
    def sort_key(lbl: str) -> Tuple[int, List[int]]:
        if lbl == "GT":
            return (0, [])
        parts = lbl.split("+") if lbl else []
        idxs = [CANONICAL_INPUT_ORDER.index(p) for p in parts if p in CANONICAL_INPUT_ORDER]
        return (1 + len(parts), idxs)

    ordered = sorted(set(labels), key=sort_key)
    # Ensure GT is first if present
    if "GT" in ordered:
        ordered.remove("GT")
        ordered = ["GT"] + ordered
    return ordered


def plot_distributions_for_tile(
    tile_path: Path,
    cfg: PlotConfig,
    save_dir: Optional[Path] = None,
) -> Path:
    sns.set_style(cfg.style)

    tile_code, tile_id = _tile_id_parts(tile_path)
    gt_scene_dir = _find_gt_scene_dir(tile_path)
    scene_name = gt_scene_dir.name

    # Discover scenarios from COPGEN outputs (group seeds by input conditions)
    copgen_dir = tile_path / "outputs" / "copgen"
    copgen_scenarios = _discover_scenarios(copgen_dir)

    # Discover scenarios from TerraMind outputs (only if --terramind flag is set)
    terramind_scenarios: Dict[str, List[Path]] = {}
    if cfg.terramind:
        terramind_dir = tile_path / "outputs" / "terramind"
        terramind_scenarios = _discover_terramind_scenarios(terramind_dir)

    # Use CopGen labels as primary labels. TerraMind scenarios are matched by removing
    # 'timestamps' from the label (since TerraMind doesn't support timestamps).
    # E.g., CopGen "DEM+LULC+S1RTC+timestamps+lat_lon" matches TerraMind "DEM+LULC+S1RTC+lat_lon"
    all_scenario_labels = set(copgen_scenarios.keys())
    # Also include any TerraMind-only scenarios (those without a matching CopGen scenario)
    for tm_label in terramind_scenarios.keys():
        # Check if there's already a CopGen label that would match this TerraMind label
        has_copgen_match = any(
            _label_without_timestamps(cg_label) == tm_label
            for cg_label in copgen_scenarios.keys()
        )
        if not has_copgen_match:
            all_scenario_labels.add(tm_label)

    band_files = S2_BAND_ORDER["S2L2A"]
    gt_band_paths = _band_paths_in_dir(gt_scene_dir, band_files)

    # Colors - assign same color to same scenario label
    scenario_labels_sorted = _legend_order(list(all_scenario_labels))
    palette = sns.color_palette(cfg.palette_name, n_colors=max(3, len(scenario_labels_sorted)))
    color_map: Dict[str, str] = {}
    for i, lbl in enumerate(scenario_labels_sorted):
        color_map[lbl] = palette[i % len(palette)]
    gt_color = "#2b2b2b"
    hist_alpha = 0.25
    kde_alpha = 0.95

    # Output directory
    out_root = save_dir if save_dir is not None else (tile_path / "narrowing_histograms")
    out_root.mkdir(parents=True, exist_ok=True)

    # Iterate bands (one figure per band)
    for band_idx, fname in enumerate(tqdm(band_files, desc="Plotting bands")):
        band = fname.replace(".tif", "").replace(".TIF", "")

        # Gather values (needed for both combined and stacked plots)
        copgen_values_by_label: Dict[str, np.ndarray] = {}
        terramind_values_by_label: Dict[str, np.ndarray] = {}

        # Ground truth
        gt_vals = _read_band_values(gt_band_paths[band])

        # CopGen scenarios (aggregate across seeds)
        for scen_label in scenario_labels_sorted:
            seed_dirs = copgen_scenarios.get(scen_label, [])
            if seed_dirs:
                scen_vals = _gather_generated_values_for_band(
                    seed_dirs=seed_dirs,
                    tile_code=tile_code,
                    tile_id=tile_id,
                    scene_name=scene_name,
                    band_fname=fname,
                )
                copgen_values_by_label[scen_label] = scen_vals

        # TerraMind scenarios (aggregate across seeds)
        # Use match key (label without timestamps) to find corresponding TerraMind scenario
        for scen_label in scenario_labels_sorted:
            tm_match_key = _label_without_timestamps(scen_label)
            seed_dirs = terramind_scenarios.get(tm_match_key, [])
            if seed_dirs:
                scen_vals = _gather_terramind_values_for_band(
                    seed_dirs=seed_dirs,
                    tile_id=tile_id,
                    band_index=band_idx,
                )
                terramind_values_by_label[scen_label] = scen_vals

        # Combined distribution plot (optional, enabled with --combined flag)
        if cfg.combined:
            fig, ax = plt.subplots(figsize=(6.0, 4.0))

            # Plot histograms
            # GT first (neutral color)
            ax.hist(
                gt_vals,
                bins=256,
                density=True,
                alpha=0.35,
                color=gt_color,
                edgecolor="none",
            )
            if cfg.kde:
                sns.kdeplot(
                    gt_vals,
                    ax=ax,
                    color=gt_color,
                    lw=1.6,
                    alpha=0.95,
                    label=None,  # keep legend simple (one label per dataset)
                )

            # CopGen scenarios (solid lines)
            for scen_label in scenario_labels_sorted:
                if scen_label not in copgen_values_by_label:
                    continue
                vals = copgen_values_by_label[scen_label]
                ax.hist(
                    vals,
                    bins=256,
                    density=True,
                    alpha=hist_alpha,
                    color=color_map[scen_label],
                    edgecolor="none",
                )
                if cfg.kde:
                    sns.kdeplot(
                        vals,
                        ax=ax,
                        color=color_map[scen_label],
                        lw=1.3,
                        alpha=kde_alpha,
                        linestyle="-",
                        label=None,
                    )

            # TerraMind scenarios (dashed lines and darker colors to distinguish from CopGen)
            for scen_label in scenario_labels_sorted:
                if scen_label not in terramind_values_by_label:
                    continue
                vals = terramind_values_by_label[scen_label]
                tm_match_key = _label_without_timestamps(scen_label)
                tm_color = _darken_color(color_map[scen_label])
                ax.hist(
                    vals,
                    bins=256,
                    density=True,
                    alpha=hist_alpha,
                    color=tm_color,
                    edgecolor="none",
                    hatch="//",  # hatched pattern to distinguish
                )
                if cfg.kde:
                    sns.kdeplot(
                        vals,
                        ax=ax,
                        color=tm_color,
                        lw=1.3,
                        alpha=kde_alpha,
                        linestyle="--",  # dashed line for TerraMind
                        label=None,
                    )

            ax.set_title(f"{tile_id} — {band}", fontsize=11)
            ax.set_xlabel("Reflectance (0-1)")
            ax.set_ylabel("Density")
            ax.set_yscale("symlog", linthresh=0.01)
            ax.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.5, alpha=0.8)
            ax.set_yticks([])

            # Build custom legend with line handles (solid for GT/CopGen, dashed for TerraMind)
            legend_handles = []
            legend_labels = []

            # GT
            gt_handle = mlines.Line2D([], [], color=gt_color, linewidth=1.6, linestyle="-")
            legend_handles.append(gt_handle)
            legend_labels.append("GT (n=1)")

            # CopGen scenarios (solid lines)
            for scen_label in scenario_labels_sorted:
                if scen_label in copgen_values_by_label:
                    n_runs = len(copgen_scenarios[scen_label])
                    handle = mlines.Line2D([], [], color=color_map[scen_label], linewidth=1.3, linestyle="-")
                    legend_handles.append(handle)
                    legend_labels.append(f"CopGen {scen_label} (n={n_runs})")

            # TerraMind scenarios (dashed lines with darker color)
            for scen_label in scenario_labels_sorted:
                if scen_label in terramind_values_by_label:
                    tm_match_key = _label_without_timestamps(scen_label)
                    n_runs = len(terramind_scenarios[tm_match_key])
                    tm_color = _darken_color(color_map[scen_label])
                    handle = mlines.Line2D([], [], color=tm_color, linewidth=1.3, linestyle="--")
                    legend_handles.append(handle)
                    legend_labels.append(f"TerraMind {tm_match_key} (n={n_runs})")

            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    labels=legend_labels,
                    fontsize=8.5,
                    frameon=True,
                    framealpha=0.9,
                    loc="best",
                )
            plt.tight_layout()
            out_path = out_root / f"{tile_id}_S2L2A_{band}_distribution.png"
            fig.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved figure to {out_path}")

        # Also create a stacked per-scenario figure (GT vs each scenario), sharing x-axis
        # Each row shows GT + CopGen (+ TerraMind if --terramind flag) for one scenario label
        nrows = len(scenario_labels_sorted)
        if nrows == 0:
            continue  # No scenarios to plot
        fig_stack, axes_stack = plt.subplots(
            nrows=nrows, ncols=1, figsize=(8, max(2.7 * nrows, 3.5)), sharex=True
        )
        if nrows == 1:
            axes_list = [axes_stack]
        else:
            axes_list = list(axes_stack)

        # Determine common x-limits across GT + all scenarios (both CopGen and TerraMind)
        all_vals_for_xlim = [gt_vals]
        all_vals_for_xlim.extend(copgen_values_by_label.values())
        all_vals_for_xlim.extend(terramind_values_by_label.values())
        try:
            xmin = min(float(np.nanmin(v)) for v in all_vals_for_xlim if v.size > 0)
            xmax = max(float(np.nanmax(v)) for v in all_vals_for_xlim if v.size > 0)
        except ValueError:
            xmin, xmax = 0.0, 1.0
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin >= xmax:
            xmin, xmax = 0.0, 1.0

        for ax_row, scen_label in zip(axes_list, scenario_labels_sorted):
            # GT
            ax_row.hist(
                gt_vals,
                bins=256,
                density=True,
                alpha=0.35,
                color=gt_color,
                edgecolor="none",
            )
            if cfg.kde:
                sns.kdeplot(
                    gt_vals,
                    ax=ax_row,
                    color=gt_color,
                    lw=1.6,
                    alpha=0.95,
                )

            # CopGen scenario (if available)
            if scen_label in copgen_values_by_label:
                copgen_vals = copgen_values_by_label[scen_label]
                ax_row.hist(
                    copgen_vals,
                    bins=256,
                    density=True,
                    alpha=hist_alpha,
                    color=color_map[scen_label],
                    edgecolor="none",
                )
                if cfg.kde:
                    sns.kdeplot(
                        copgen_vals,
                        ax=ax_row,
                        color=color_map[scen_label],
                        lw=1.3,
                        alpha=kde_alpha,
                        linestyle="-",
                    )

            # TerraMind scenario (if available)
            if scen_label in terramind_values_by_label:
                terramind_vals = terramind_values_by_label[scen_label]
                tm_match_key = _label_without_timestamps(scen_label)
                tm_color = _darken_color(color_map[scen_label])
                ax_row.hist(
                    terramind_vals,
                    bins=256,
                    density=True,
                    alpha=hist_alpha,
                    color=tm_color,
                    edgecolor="none",
                    hatch="//",
                )
                if cfg.kde:
                    sns.kdeplot(
                        terramind_vals,
                        ax=ax_row,
                        color=tm_color,
                        lw=1.3,
                        alpha=kde_alpha,
                        linestyle="--",
                    )

            ax_row.set_xlim([xmin, xmax])
            ax_row.set_ylabel("Density", fontsize=15)
            ax_row.set_yscale("symlog", linthresh=0.01)
            ax_row.grid(True, color="#e0e0e0", linestyle="--", linewidth=0.5, alpha=0.8)
            ax_row.set_title(scen_label, fontsize=18)
            ax_row.tick_params(axis='both', labelsize=15)
            ax_row.set_yticks([])

            # Build custom legend with line handles for this row
            row_handles = []
            row_labels = []

            # GT (always present)
            gt_handle = mlines.Line2D([], [], color=gt_color, linewidth=2, linestyle="-")
            row_handles.append(gt_handle)
            row_labels.append("GT (n=1)")

            # CopGen for this scenario (solid line)
            # Only prefix with "CopGen" if terramind comparison is enabled
            if scen_label in copgen_values_by_label:
                copgen_n_runs = len(copgen_scenarios[scen_label])
                handle = mlines.Line2D([], [], color=color_map[scen_label], linewidth=2, linestyle="-")
                row_handles.append(handle)
                if cfg.terramind:
                    row_labels.append(f"CopGen {scen_label} (n={copgen_n_runs})")
                else:
                    # row_labels.append(f"{scen_label} (n={copgen_n_runs})")
                    row_labels.append(f"COPGEN (n={copgen_n_runs})")

            # TerraMind for this scenario (dashed line, darker color)
            if scen_label in terramind_values_by_label:
                tm_match_key = _label_without_timestamps(scen_label)
                terramind_n_runs = len(terramind_scenarios[tm_match_key])
                tm_color = _darken_color(color_map[scen_label])
                handle = mlines.Line2D([], [], color=tm_color, linewidth=2, linestyle="--")
                row_handles.append(handle)
                row_labels.append(f"TerraMind {tm_match_key} (n={terramind_n_runs})")

            if row_handles:
                ax_row.legend(
                    handles=row_handles,
                    labels=row_labels,
                    fontsize=14.0,
                    frameon=True,
                    framealpha=0.9,
                    loc="best",
                )

        # Only the bottom row carries the x-axis label
        axes_list[-1].set_xlabel("Reflectance (0-1)", fontsize=14)
        fig_stack.suptitle(f"{tile_id} — {band}", fontsize=18)
        plt.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.8)
        out_path_stacked = out_root / f"{tile_id}_S2L2A_{band}_stacked.png"
        fig_stack.savefig(out_path_stacked, dpi=cfg.dpi, bbox_inches="tight")
        plt.close(fig_stack)
        print(f"Saved figure to {out_path_stacked}")

    return out_root


def main(args: argparse.Namespace) -> None:
    cfg = PlotConfig(
        kde=(not args.no_kde),
        dpi=args.dpi,
        combined=args.combined,
        terramind=args.terramind,
    )
    tile_path = Path(args.tile_path).resolve()
    explicit_save_dir = Path(args.save_dir).resolve() if args.save_dir is not None else None
    print(f"Plotting distributions for tile: {tile_path}")
    out_dir = plot_distributions_for_tile(tile_path, cfg, save_dir=explicit_save_dir)
    print(f"Saved band histograms to: {out_dir}")
    if args.show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    args = parse_args()
    main(args)

