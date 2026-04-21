"""
Benchmark footprint extraction.

This module documents *exactly* how the 192x192 evaluation footprint is
extracted from the 1056x1056 Major TOM tiles.  It is the reference used
to score COP-GEN and TerraMind outputs against real Sentinel-2 L2A data
in the benchmark published alongside the paper.

The extracted window is **approximately but not exactly** the geometric
centre of the tile.  This is because:

1.  COP-GEN and TerraMind were trained on the *old* Major TOM v1 grid
    (1068x1068 px at 10m) and their 192x192 outputs correspond to a
    centre crop of that old tile::

        old_grid_offset = (1068 - 192) // 2 = 438 px = 4380 m

2.  The benchmark is released on the *new* Major TOM v2 grid
    (1056x1056 px at 10m).  The v2 grid snaps each cell's upper-left
    corner to the nearest 60 m node relative to the MGRS tile origin;
    the v1 grid did not.  As a result, the v1 origin and the v2 origin
    differ by up to ~60 m per cell (and may also differ in CRS for
    three cells where the two grids chose different UTM zones).

3.  Re-expressing the v1 centre-crop footprint in v2 pixel coordinates
    therefore yields offsets in the range **406--457 px** (mean
    432 +/- 9), rather than the geometric v2 centre at (1056-192)/2 =
    432 px.  The per-cell offset is deterministic and reproducible from
    the grid geometry alone.

The function below takes a 1056 tile and returns the exact 192x192
window used for evaluation, handling CRS mismatches where they occur.

Usage
-----
>>> import rasterio
>>> from benchmark_footprint import load_grid, crop_benchmark_footprint
>>>
>>> grid = load_grid("benchmark_grid.json")   # ships with the dataset
>>> with rasterio.open("real/MT10_106D_246R_00.tif") as src:
...     window = crop_benchmark_footprint(src, cell_id="106D_246R", grid=grid)
...     # window is a (bands, 192, 192) numpy array (uint16 reflectance)

The file ``benchmark_grid.json`` is distributed with the HuggingFace
dataset under ``metadata/benchmark_grid.json``.  It contains one entry
per cell::

    {
        "106D_246R": {
            "ul_x": 653683.14, "ul_y": 8957579.3, "crs": "EPSG:32734"
        },
        ...
    }

where ``(ul_x, ul_y)`` is the upper-left corner of the 192x192 model
output footprint in the cell's native UTM CRS.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Edge length of the model output footprint in pixels.
MODEL_PX: int = 192

#: Edge length of the old Major TOM v1 tile in pixels.
PARENT_PX: int = 1068

#: Edge length of the new Major TOM v2 tile in pixels.
NEW_TILE_PX: int = 1056

#: Ground-sample distance in metres. All bands are resampled to 10 m.
PIXEL_SIZE: int = 10

#: Centre-crop offset in the old v1 tile, in pixels.
CENTRE_OFFSET_PX: int = (PARENT_PX - MODEL_PX) // 2  # 438

#: Centre-crop offset in the old v1 tile, in metres.
CENTRE_OFFSET_M: int = CENTRE_OFFSET_PX * PIXEL_SIZE  # 4380


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_grid(path: Union[str, Path]) -> Dict[str, Dict]:
    """Load the benchmark grid JSON.

    Parameters
    ----------
    path : str or Path
        Path to ``benchmark_grid.json``.

    Returns
    -------
    dict
        Mapping ``cell_id -> {"ul_x", "ul_y", "crs"}`` where ``(ul_x, ul_y)``
        is the upper-left corner of the 192x192 evaluation window in the
        cell's native UTM CRS.
    """
    with open(path) as f:
        return json.load(f)


def crop_benchmark_footprint(
    src: "rasterio.io.DatasetReader",
    cell_id: str,
    grid: Dict[str, Dict],
) -> np.ndarray:
    """Extract the 192x192 evaluation footprint from a 1056x1056 tile.

    The function reads the cell's old-grid origin from ``grid``, computes
    the geographic bounds of the 192x192 evaluation window, and extracts
    the corresponding pixel window from the open rasterio dataset.

    Handles CRS mismatches between the old and new grids via
    ``rasterio.warp.transform_bounds``.  Returns zeros in regions that
    fall outside the tile.

    Notes
    -----
    * **Real subset**: returns 100% valid reflectance (the 1056 tile
      covers the full evaluation footprint).
    * **COP-GEN / TerraMind subsets**: typically 95--100% valid.  The
      ~5% of zero pixels occur at edges where the model's native 192
      output does not perfectly align with the v2 grid centre (the two
      grids snap to different 60 m nodes).  This matches the behaviour
      used during benchmark evaluation, so cropped windows from all
      three subsets cover the same geographic footprint (just with
      marginally different edge coverage for the synthetic data).
    * Six cells (~1%) have the model footprint partially outside the S2
      swath.  These are excluded from the benchmark evaluation in the
      paper but their tiles are still included in the HuggingFace
      release.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Opened rasterio handle to a 1056x1056 tile (any band count).
    cell_id : str
        Cell identifier, e.g. ``"106D_246R"``.
    grid : dict
        Loaded benchmark grid (see :func:`load_grid`).

    Returns
    -------
    ndarray, shape ``(bands, 192, 192)``
        Pixel data for the evaluation footprint, in the source tile's
        dtype.  Zero-padded at any edge that falls outside the tile.

    Raises
    ------
    KeyError
        If ``cell_id`` is not in ``grid``.
    """
    info = grid[cell_id]
    old_ul_x: float = info["ul_x"]
    old_ul_y: float = info["ul_y"]
    old_crs: str = info["crs"]

    # Geographic bounds of the 192x192 evaluation window in old-grid CRS
    model_left = old_ul_x + CENTRE_OFFSET_M
    model_top = old_ul_y - CENTRE_OFFSET_M
    model_right = model_left + MODEL_PX * PIXEL_SIZE
    model_bottom = model_top - MODEL_PX * PIXEL_SIZE

    # Reproject to source CRS if the two differ (rare)
    src_crs = str(src.crs)
    if old_crs != src_crs:
        model_left, model_bottom, model_right, model_top = transform_bounds(
            old_crs, src_crs, model_left, model_bottom, model_right, model_top,
        )

    # Convert to source pixel coordinates
    col_off = round((model_left - src.transform.c) / src.transform.a)
    row_off = round((src.transform.f - model_top) / abs(src.transform.e))

    fully_inside = (
        col_off >= 0
        and row_off >= 0
        and col_off + MODEL_PX <= src.width
        and row_off + MODEL_PX <= src.height
    )

    if fully_inside:
        return src.read(window=Window(col_off, row_off, MODEL_PX, MODEL_PX))

    # Clamp window to the tile and zero-pad the missing edges
    c0 = max(0, col_off)
    r0 = max(0, row_off)
    c1 = min(src.width, col_off + MODEL_PX)
    r1 = min(src.height, row_off + MODEL_PX)
    crop = src.read(window=Window(c0, r0, c1 - c0, r1 - r0))
    out = np.zeros((src.count, MODEL_PX, MODEL_PX), dtype=crop.dtype)
    paste_r = max(0, -row_off)
    paste_c = max(0, -col_off)
    out[:, paste_r : paste_r + crop.shape[1], paste_c : paste_c + crop.shape[2]] = crop
    return out


def benchmark_footprint_bounds(
    cell_id: str,
    grid: Dict[str, Dict],
) -> Tuple[float, float, float, float]:
    """Return the geographic bounds (left, bottom, right, top) of the
    192x192 evaluation footprint in the cell's native UTM CRS.

    Useful for users who want to pre-align other rasters to the
    benchmark footprint before loading.
    """
    info = grid[cell_id]
    left = info["ul_x"] + CENTRE_OFFSET_M
    top = info["ul_y"] - CENTRE_OFFSET_M
    right = left + MODEL_PX * PIXEL_SIZE
    bottom = top - MODEL_PX * PIXEL_SIZE
    return (left, bottom, right, top)
