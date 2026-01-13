import argparse
import os
import pickle
from typing import Dict, Tuple, List, Optional
import math
import numpy as np
import torch
from tqdm import tqdm
import multiprocessing
import json

from grid import Grid
from ddm.pre_post_process_data import encode_lat_lon
# from locationencoder.pe import SphericalHarmonics

# Global 10-km grid instance, identical to the dataset implementation
GLOBAL_GRID_10KM = Grid(10)

LatLon = Tuple[float, float]
Key = Tuple[str, int, int]  # (grid_cell, patch_row, patch_col) – use (-1, -1) for the whole cell

# Supported output formats for the coordinates
_SUPPORTED_FORMATS = {"spherical_harmonics", "3d_cartesian", "lat_lon"}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_cells(txt_path: str) -> List[str]:
    """Return non-empty, stripped lines from *txt_path*."""
    with open(txt_path, "r") as f:
        cells = [ln.strip() for ln in f if ln.strip()]
    print(f"Loaded {len(cells)} grid-cell labels")
    return cells


class _RowMeta:
    """Pre-computed geometry for one latitude row of the grid."""

    __slots__ = (
        "lat_bottom",
        "row_height",
        "lon_left_arr",
        "col_width_arr",
        "col_name_to_idx",
    )

    def __init__(self, lat_bottom: float, row_height: float, lon_left_arr: np.ndarray, col_width_arr: np.ndarray, col_names: List[str]):
        self.lat_bottom = lat_bottom
        self.row_height = row_height
        self.lon_left_arr = lon_left_arr
        self.col_width_arr = col_width_arr
        self.col_name_to_idx = {name: i for i, name in enumerate(col_names)}


def _build_row_meta(grid: Grid) -> Dict[str, _RowMeta]:
    """Vectorised extraction of per-row bounds & widths (no shapely calls later)."""
    meta: Dict[str, _RowMeta] = {}
    lats = grid.lats
    n_rows = len(lats)

    for row_idx, row_label in enumerate(grid.rows):
        lat_bottom = lats[row_idx]
        # Height = lat diff to next row; extrapolate at the top edge
        if row_idx + 1 < n_rows:
            lat_top = lats[row_idx + 1]
        else:
            lat_top = lat_bottom + (lat_bottom - lats[row_idx - 1])
        row_height = lat_top - lat_bottom

        df_row = grid.points_by_row[row_idx].sort_values("col_idx")
        lon_left_arr = df_row.geometry.x.to_numpy()
        # Widths: diff to next lon; extrapolate at rightmost edge
        lon_right = np.empty_like(lon_left_arr)
        lon_right[:-1] = lon_left_arr[1:]
        lon_right[-1] = lon_left_arr[-1] + (lon_left_arr[-1] - lon_left_arr[-2])
        col_width_arr = lon_right - lon_left_arr

        meta[row_label] = _RowMeta(
            lat_bottom=lat_bottom,
            row_height=row_height,
            lon_left_arr=lon_left_arr,
            col_width_arr=col_width_arr,
            col_names=df_row["col"].tolist(),
        )
    return meta


# -----------------------------------------------------------------------------
# Core computation
# -----------------------------------------------------------------------------


def compute_centres(
    cells: List[str],
    side: int,
    output_format: str,
    patch_size: Optional[Tuple[float, float]] = None,
    legendre_polys: int = 20,
) -> Dict[Key, LatLon]:
    """Compute (lat, lon) centre for every patch of every grid-cell *and* the
    centre of the entire grid-cell itself.

    For the whole cell, the key uses patch indices ``(-1, -1)`` so that the
    returned mapping can be queried uniformly via ``(cell_label, pr, pc)``.

    *side* is the number of patch subdivisions per side (rows = cols).
    """
    grid = GLOBAL_GRID_10KM
    row_meta = _build_row_meta(grid)

    mapping: Dict[Key, LatLon] = {}
    
    if output_format == "spherical_harmonics":
        raise ImportError(
            "Spherical harmonics output format requires additional dependencies. "
            "Please install locationencoder by cloning the repository and running installation instructions."
        )
        sh = SphericalHarmonics(legendre_polys=legendre_polys)

    for cell in tqdm(cells, desc="centres"):
        try:
            row_label, col_label = cell.split("_")
        except ValueError:
            raise ValueError(f"Malformed cell label '{cell}' - expected 'ROW_COL'")

        row_info = row_meta[row_label]
        col_idx = row_info.col_name_to_idx[col_label]

        lon_left = row_info.lon_left_arr[col_idx]
        col_width = row_info.col_width_arr[col_idx]
        lat_bottom = row_info.lat_bottom
        row_height = row_info.row_height

        # Store centre of the entire grid-cell (patch_row = patch_col = -1)
        cell_lat = lat_bottom + row_height * 0.5
        cell_lon = lon_left + col_width * 0.5        
        if output_format == "spherical_harmonics":
            raise ImportError(
                "Spherical harmonics output format requires additional dependencies. "
                "Please install locationencoder by cloning the repository and running installation instructions."
            )
            centre_vec = sh(torch.tensor([[cell_lon, cell_lat]], dtype=torch.float32))
            centre_vec = centre_vec.squeeze(0).detach().numpy()
        elif output_format == "3d_cartesian":
            lat_rad = math.radians(cell_lat)
            lon_rad = math.radians(cell_lon)
            x = math.cos(lat_rad) * math.cos(lon_rad)
            y = math.cos(lat_rad) * math.sin(lon_rad)
            z = math.sin(lat_rad)
            centre_vec = np.array([x, y, z], dtype=np.float32)
        elif output_format == "lat_lon":
            centre_vec = np.array([cell_lat, cell_lon], dtype=np.float32)

        mapping[(cell, -1, -1)] = centre_vec
        
        # Patch size
        if patch_size is None:
            patch_w = col_width / side
            patch_h = row_height / side
        else:
            patch_w, patch_h = patch_size

        for pr in range(side):  # patch row (0 = top)
            for pc in range(side):  # patch col (0 = left)
                # Horizontal placement (right-align last col like dataset)
                patch_minx = lon_left + pc * patch_w if pc < side - 1 else lon_left + col_width - patch_w
                # Vertical placement – row 0 is the northern-most patch
                patch_miny = lat_bottom + (row_height - patch_h) - pr * patch_h if pr < side - 1 else lat_bottom

                lat = patch_miny + patch_h * 0.5
                lon = patch_minx + patch_w * 0.5

                if output_format == "spherical_harmonics":
                    raise ImportError(
                        "Spherical harmonics output format requires additional dependencies. "
                        "Please install locationencoder by cloning the repository and running installation instructions."
                    )
                    lat_lon_vec = sh(torch.tensor([[lon, lat]], dtype=torch.float32)).squeeze(0).detach().numpy()
                elif output_format == "3d_cartesian":
                    lat_lon_vec = encode_lat_lon(lat, lon, output_format='3d_cartesian_lat_lon', add_spatial_dims=False)
                elif output_format == "lat_lon":
                    lat_lon_vec = np.array([lat, lon], dtype=np.float32)
                
                mapping[(cell, pr, pc)] = lat_lon_vec

    return mapping


# -----------------------------------------------------------------------------
# Parallel version
# -----------------------------------------------------------------------------


def compute_centres_parallel(
    cells: List[str],
    side: int,
    output_format: str,
    patch_size: Optional[Tuple[float, float]] = None,
    legendre_polys: int = 20,
    workers: int = 1,
) -> Dict[Key, LatLon]:
    """Compute centres in parallel using *workers* processes.

    Falls back to the serial :pyfunc:`compute_centres` implementation when
    ``workers`` is set to ``1``.
    """

    # Guard against invalid worker counts
    workers = max(1, int(workers))

    # Serial execution shortcut
    if workers == 1 or len(cells) == 0:
        return compute_centres(cells, side, output_format, patch_size, legendre_polys)

    # Split the cell list into roughly equal chunks for each worker
    chunk_size = math.ceil(len(cells) / workers)
    chunks: List[List[str]] = [cells[i : i + chunk_size] for i in range(0, len(cells), chunk_size)]

    # Prepare the iterable of arguments for starmap
    args_iter = [
        (chunk, side, output_format, patch_size, legendre_polys) for chunk in chunks
    ]

    # Use ``fork`` on Unix to avoid expensive re-initialisation cost of the Grid
    ctx = multiprocessing.get_context("fork") if hasattr(multiprocessing, "get_context") else multiprocessing
    with ctx.Pool(processes=workers) as pool:
        results = pool.starmap(compute_centres, args_iter)

    # Merge individual dictionaries (later entries cannot duplicate keys)
    merged: Dict[Key, LatLon] = {}
    for mapping in results:
        merged.update(mapping)

    return merged

# -----------------------------------------------------------------------------
# Fast array writer (for memory-mapped loading)
# -----------------------------------------------------------------------------

def _write_npy_dir(
    out_dir: str,
    cells: List[str],
    centres: Dict[Key, LatLon],
    side: int,
    coord_format: str,
    legendre_polys: int,
    patch_size: Optional[Tuple[float, float]],
):
    os.makedirs(out_dir, exist_ok=True)

    # Determine vector dimension
    any_vec = next(iter(centres.values()))
    vec_dim = int(np.asarray(any_vec, dtype=np.float32).shape[-1])

    num_cells = len(cells)
    # Prepare arrays
    latlon_cells = np.empty((num_cells, vec_dim), dtype=np.float32)
    latlon_patches = np.empty((num_cells, side, side, vec_dim), dtype=np.float32)

    # Fill arrays in the same order as input cells
    for i, cell in enumerate(cells):
        latlon_cells[i] = np.asarray(centres[(cell, -1, -1)], dtype=np.float32)
        for pr in range(side):
            for pc in range(side):
                latlon_patches[i, pr, pc] = np.asarray(centres[(cell, pr, pc)], dtype=np.float32)

    # Save arrays for memory-mapped loading
    max_len = max(len(c) for c in cells) if cells else 1
    cells_np = np.asarray(cells, dtype=f"<U{max_len}")

    np.save(os.path.join(out_dir, "cells.npy"), cells_np)
    np.save(os.path.join(out_dir, "latlon_cells.npy"), latlon_cells)
    np.save(os.path.join(out_dir, "latlon_patches.npy"), latlon_patches)

    # Write metadata
    meta = {
        "side": int(side),
        "format": coord_format,
        "legendre_polys": int(legendre_polys),
        "patch_width": None if patch_size is None else float(patch_size[0]),
        "patch_height": None if patch_size is None else float(patch_size[1]),
        "vec_dim": vec_dim,
        "version": 1,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"Wrote memmap-ready arrays to '{out_dir}'")


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser("Pre-compute lat/lon centres for COPGEN patches (fast version)")
    parser.add_argument("--grid_cells_txt", required=True)
    parser.add_argument("--patch_side", type=int, required=True, help="Number of patches per side (rows = cols)")
    parser.add_argument("--patch_width", type=float, help="Optional explicit patch width (degrees)")
    parser.add_argument("--patch_height", type=float, help="Optional explicit patch height (degrees)")
    parser.add_argument("--output", required=True, help="Where to save the pickle mapping or npy directory")
    parser.add_argument(
        "--format",
        required=True,
        choices=sorted(_SUPPORTED_FORMATS),
        help="Coordinate format to store: 'lat-lon', 'spherical_harmonics', or '3d_cartesian'",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel worker processes to use (default: all available CPU cores)",
    )
    parser.add_argument("--legendre_polys", type=int, default=20, help="Number of Legendre polynomials for spherical harmonics")
    parser.add_argument(
        "--storage",
        choices=["pickle", "npy_dir"],
        default="pickle",
        help="Storage format: 'pickle' (backward compatible) or 'npy_dir' (fast, memory-mapped)",
    )
    args = parser.parse_args()

    if (args.patch_width is None) ^ (args.patch_height is None):
        parser.error("--patch_width and --patch_height must be supplied together or both omitted")

    patch_sz = (args.patch_width, args.patch_height) if args.patch_width is not None else None

    print("Loading grid cells...")
    cells = load_cells(args.grid_cells_txt)
    print("Computing patch centres...")
    centres = compute_centres_parallel(
        cells=cells,
        side=args.patch_side,
        output_format=args.format,
        patch_size=patch_sz,
        legendre_polys=args.legendre_polys,
        workers=args.workers,
    )
    print("Done!")
    
    print("Writing to file...")
    if args.storage == "pickle":
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "wb") as f:
            pickle.dump(centres, f)
        print(f"Saved {len(centres)} entries → {args.output}")
    else:
        # NPY directory mode: args.output is a directory (create if not exists)
        _write_npy_dir(
            out_dir=args.output,
            cells=cells,
            centres=centres,
            side=args.patch_side,
            coord_format=args.format,
            legendre_polys=args.legendre_polys,
            patch_size=patch_sz,
        )


if __name__ == "__main__":
    main() 