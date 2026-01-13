import argparse
from typing import List, Tuple, Sequence
from pathlib import Path
import os
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch as MplPatch
from matplotlib.collections import PolyCollection
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm
import matplotlib.colors as mcolors

import sys
sys.path.append("../../scripts")
from grid import Grid


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

def load_cell_groups(paths: List[str]) -> List[Tuple[str, List[str]]]:
    groups: List[Tuple[str, List[str]]] = []
    for p in paths:
        print(f"Reading cells from {p} ...")
        cells: List[str] = []
        with open(p, "r") as f:
            for line in f:
                s = line.strip()
                if s:
                    cells.append(s)
        if len(cells) == 0:
            print(f"Warning: file {p} contained no entries")
        # De-duplicate within this file while preserving order
        seen = set()
        uniq_cells: List[str] = []
        for c in cells:
            if c not in seen:
                uniq_cells.append(c)
                seen.add(c)
        name = Path(p).stem
        print(f"Loaded {len(uniq_cells)} unique grid-cell labels for group '{name}'")
        groups.append((name, uniq_cells))
    total = sum(len(g[1]) for g in groups)
    if total == 0:
        raise ValueError("No grid cells found in the provided txt files")
    print(f"Total unique labels across groups (per-file deduped): {total}")
    return groups


# -----------------------------------------------------------------------------
# Grid helpers (fast, vectorised meta extracted once from Grid)
# -----------------------------------------------------------------------------

class _RowMeta:
    __slots__ = ("lat_bottom", "row_height", "lon_left_arr", "col_width_arr", "col_name_to_idx")

    def __init__(self, lat_bottom: float, row_height: float, lon_left_arr: np.ndarray, col_width_arr: np.ndarray, col_names: List[str]):
        self.lat_bottom = lat_bottom
        self.row_height = row_height
        self.lon_left_arr = lon_left_arr
        self.col_width_arr = col_width_arr
        self.col_name_to_idx = {name: i for i, name in enumerate(col_names)}


def build_row_meta(grid: Grid) -> dict:
    meta = {}
    lats = grid.lats
    n_rows = len(lats)

    print("Building grid row metadata ...")
    for row_idx, row_label in enumerate(tqdm(grid.rows, desc="rows")):
        lat_bottom = lats[row_idx]
        if row_idx + 1 < n_rows:
            lat_top = lats[row_idx + 1]
        else:
            lat_top = lat_bottom + (lat_bottom - lats[row_idx - 1])
        row_height = lat_top - lat_bottom

        df_row = grid.points_by_row[row_idx].sort_values("col_idx")
        lon_left_arr = df_row.geometry.x.to_numpy()
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
# Geometry construction
# -----------------------------------------------------------------------------

def rect_lonlat_for_cell(cell_label: str, row_meta: dict) -> Tuple[float, float, float, float]:
    try:
        row_label, col_label = cell_label.split("_")
    except ValueError:
        raise ValueError(f"Malformed cell label '{cell_label}' - expected 'ROW_COL'")

    if row_label not in row_meta:
        raise KeyError(f"Row '{row_label}' not found in grid meta")
    row_info: _RowMeta = row_meta[row_label]

    try:
        col_idx = row_info.col_name_to_idx[col_label]
    except KeyError:
        raise KeyError(f"Column '{col_label}' not found in row '{row_label}'")

    lon_left = float(row_info.lon_left_arr[col_idx])
    col_width = float(row_info.col_width_arr[col_idx])
    lat_bottom = float(row_info.lat_bottom)
    row_height = float(row_info.row_height)
    return lon_left, lat_bottom, col_width, row_height


def split_rect_antimeridian(lon_left: float, lat_bottom: float, width: float, height: float) -> List[List[Tuple[float, float]]]:
    lon_right = lon_left + width

    if lon_right <= 180.0:
        # Single rectangle in standard range
        return [[
            (lon_left, lat_bottom),
            (lon_right, lat_bottom),
            (lon_right, lat_bottom + height),
            (lon_left, lat_bottom + height),
        ]]

    # Crosses the antimeridian: split into two polygons
    right_over = lon_right - 180.0  # amount beyond 180
    left_part = [
        (lon_left, lat_bottom),
        (180.0, lat_bottom),
        (180.0, lat_bottom + height),
        (lon_left, lat_bottom + height),
    ]
    right_part = [
        (-180.0, lat_bottom),
        (-180.0 + right_over, lat_bottom),
        (-180.0 + right_over, lat_bottom + height),
        (-180.0, lat_bottom + height),
    ]
    return [left_part, right_part]


# -----------------------------------------------------------------------------
# Basemap drawing
# -----------------------------------------------------------------------------

def make_basemap(style: str = "light", figsize: Tuple[int, int] = (48, 24), dpi: int = 167) -> Tuple[plt.Figure, plt.Axes, Basemap]:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    m = Basemap(projection="sinu", lat_0=0, lon_0=0, resolution="l", ax=ax)

    if style == "light":
        land = "#9eba9b"; borders = "#666666"; ocean = "#CCDDFF"
    else:
        land = "#242424"; borders = "#000000"; ocean = "#242424"

    m.fillcontinents(color=land, lake_color=ocean)
    m.drawmapboundary(fill_color=ocean)
    m.drawcountries(color=borders, linewidth=1)
    m.drawcoastlines(color=borders, linewidth=1)

    ax.set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    return fig, ax, m


_ROW_META_WORKER = None


def _init_worker(row_meta):
    global _ROW_META_WORKER
    _ROW_META_WORKER = row_meta


def _build_polys_for_cells_chunk(cells_chunk: List[str]) -> Tuple[List[np.ndarray], int]:
    # Uses global row meta in worker
    polys_list: List[np.ndarray] = []
    for cell in cells_chunk:
        lon_left, lat_bottom, width, height = rect_lonlat_for_cell(cell, _ROW_META_WORKER)
        polys = split_rect_antimeridian(lon_left, lat_bottom, width, height)
        for poly_lonlat in polys:
            arr = np.asarray(poly_lonlat, dtype=float)
            polys_list.append(arr)
    return polys_list, len(cells_chunk)


def _compute_group_polys_lonlat(cells: List[str], row_meta: dict, workers: int) -> List[np.ndarray]:
    # Serial fast path
    if workers <= 1 or len(cells) == 0:
        out: List[np.ndarray] = []
        for cell in tqdm(cells, desc="tiles", leave=False):
            lon_left, lat_bottom, width, height = rect_lonlat_for_cell(cell, row_meta)
            polys = split_rect_antimeridian(lon_left, lat_bottom, width, height)
            for poly_lonlat in polys:
                out.append(np.asarray(poly_lonlat, dtype=float))
        return out

    # Parallel: chunk cells and compute in workers
    chunk_size = max(500, min(5000, len(cells) // (workers * 2) or 1))
    chunks = [cells[i : i + chunk_size] for i in range(0, len(cells), chunk_size)]
    ctx = multiprocessing.get_context("fork") if hasattr(multiprocessing, "get_context") else multiprocessing
    out: List[np.ndarray] = []
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(row_meta,)) as pool:
        pbar = tqdm(total=len(cells), desc="tiles(par)")
        for polys_chunk, n_cells in pool.imap_unordered(_build_polys_for_cells_chunk, chunks):
            out.extend(polys_chunk)
            pbar.update(n_cells)
        pbar.close()
    return out


def plot_cell_groups_on_basemap(groups: List[Tuple[str, List[str]]], grid_km: int, colors: Sequence[str], edgecolor: str, alpha: float, style: str, output: str, dpi: int, workers: int) -> None:
    print("Creating grid ...")
    grid = Grid(grid_km)
    row_meta = build_row_meta(grid)

    print("Preparing basemap ...")
    fig, ax, m = make_basemap(style=style, dpi=dpi)

    # Assign colors per group (cycle if fewer colors provided)
    handles: List[MplPatch] = []
    for i, (name, cells) in enumerate(groups):
        if len(cells) == 0:
            continue
        color = colors[i % len(colors)]
        print(f"Computing polygons for group '{name}' with {len(cells)} tiles ...")
        polys_lonlat = _compute_group_polys_lonlat(cells, row_meta, workers=workers)

        # Transform all vertices in one batch for this group
        if len(polys_lonlat) == 0:
            continue
        lengths = [poly.shape[0] for poly in polys_lonlat]
        all_lons = np.concatenate([poly[:, 0] for poly in polys_lonlat])
        all_lats = np.concatenate([poly[:, 1] for poly in polys_lonlat])
        x_all, y_all = m(all_lons, all_lats)

        coords_list = []
        offset = 0
        for L in lengths:
            coords = np.column_stack([x_all[offset:offset+L], y_all[offset:offset+L]])
            coords_list.append(coords)
            offset += L

        coll = PolyCollection(
            coords_list,
            facecolors=color,
            edgecolors=edgecolor if edgecolor else "none",
            linewidths=0.5 if edgecolor else 0.0,
            antialiased=True,
        )
        ax.add_collection(coll)

        handles.append(MplPatch(facecolor=color, edgecolor=edgecolor if edgecolor else "none", label=name, alpha=alpha))

    if len(handles) > 0:
        ax.legend(
            handles=handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),  # keep away from absolute corner
            frameon=True,
            framealpha=0.9,
            fontsize=18,
            borderpad=1.2,
            labelspacing=0.8,
            handlelength=1.5,
            borderaxespad=1.0,
        )

    print(f"Saving figure to {output} ...")
    fig.savefig(output, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Folium export (interactive map)
# -----------------------------------------------------------------------------

def _color_to_hex(color) -> str:
    if color is None:
        return None
    if isinstance(color, str) and color.startswith("#"):
        return color
    try:
        return mcolors.to_hex(color, keep_alpha=False)
    except Exception:
        # Fallback to original string if conversion fails
        return str(color)


def _close_ring(coords_arr: np.ndarray) -> List[List[float]]:
    coords: List[List[float]] = coords_arr.astype(float).tolist()
    if len(coords) == 0:
        return coords
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def export_cell_groups_to_folium(groups: List[Tuple[str, List[str]]], grid_km: int, colors: Sequence[str], edgecolor: str, alpha: float, style: str, output_html: str, workers: int) -> None:
    try:
        import folium
        from folium import GeoJson, LayerControl
    except ImportError:
        print("Folium is not installed. Skipping interactive map export.")
        return

    print("Creating grid for interactive map ...")
    grid = Grid(grid_km)
    row_meta = build_row_meta(grid)

    tiles = "CartoDB positron" if style == "light" else "CartoDB dark_matter"
    fmap = folium.Map(location=[0.0, 0.0], zoom_start=2, tiles=tiles, control_scale=True)

    for i, (name, cells) in enumerate(groups):
        if len(cells) == 0:
            continue
        print(f"Computing polygons for interactive group '{name}' with {len(cells)} tiles ...")
        polys_lonlat = _compute_group_polys_lonlat(cells, row_meta, workers=workers)
        if len(polys_lonlat) == 0:
            continue

        fill_color = _color_to_hex(colors[i % len(colors)])
        line_color = _color_to_hex(edgecolor) if edgecolor else fill_color
        weight = 1 if edgecolor else 0
        fill_opacity = float(alpha)
        if fill_opacity < 0.0:
            fill_opacity = 0.0
        if fill_opacity > 1.0:
            fill_opacity = 1.0

        # Build a GeoJSON MultiPolygon (list of polygons, each with one outer ring)
        multi_coords: List[List[List[List[float]]]] = []
        for poly in polys_lonlat:
            ring = _close_ring(poly)
            # GeoJSON is [lon, lat]
            multi_coords.append([ring])

        feature = {
            "type": "Feature",
            "properties": {
                "name": name,
                "tiles": len(cells),
            },
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": multi_coords,
            },
        }

        def _style_fn(_):
            return {
                "fillColor": fill_color,
                "color": line_color,
                "weight": weight,
                "fillOpacity": fill_opacity,
            }

        gj = GeoJson(
            data=feature,
            name=name,
            style_function=_style_fn,
            tooltip=folium.GeoJsonTooltip(fields=["name", "tiles"], aliases=["Group", "Num tiles"]),
        )
        gj.add_to(fmap)

    LayerControl(collapsed=False).add_to(fmap)

    print(f"Saving interactive map to {output_html} ...")
    fmap.save(output_html)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser("Visualise MajorTOM grid-cell coverage on a world map")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more txt files with grid-cell labels (e.g. 0U_1022L)")
    parser.add_argument("--output", required=True, help="Path to save the output image (e.g. coverage.png)")
    parser.add_argument("--output_html", default=None, help="Optional path to save an interactive HTML map via Folium")
    parser.add_argument("--grid_km", type=int, default=10, help="Grid cell size in km (default: 10)")
    parser.add_argument("--style", choices=["light", "dark"], default="light", help="Basemap style")
    parser.add_argument("--colors", nargs="+", help="Optional list of colors per input; cycles if fewer provided")
    parser.add_argument("--edgecolor", default=None, help="Edge color for tiles (default: none)")
    parser.add_argument("--alpha", type=float, default=0.9, help="Tile fill alpha (default: 0.9)")
    parser.add_argument("--dpi", type=int, default=167, help="Figure DPI (default: 167)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers for polygon prep (default: all cores)")

    args = parser.parse_args()

    groups = load_cell_groups(args.inputs)

    # Resolve colors
    if args.colors:
        colors: List[str] = list(args.colors)
    else:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(len(groups))]

    plot_cell_groups_on_basemap(
        groups=groups,
        grid_km=args.grid_km,
        colors=colors,
        edgecolor=args.edgecolor,
        alpha=args.alpha,
        style=args.style,
        output=args.output,
        dpi=args.dpi,
        workers=max(1, int(args.workers)),
    )

    if args.output_html:
        export_cell_groups_to_folium(
            groups=groups,
            grid_km=args.grid_km,
            colors=colors,
            edgecolor=args.edgecolor,
            alpha=args.alpha,
            style=args.style,
            output_html=args.output_html,
            workers=max(1, int(args.workers)),
        )


if __name__ == "__main__":
    main()


