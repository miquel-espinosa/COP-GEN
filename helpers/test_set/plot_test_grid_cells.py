import argparse
import os
import re
from typing import List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Local project imports
from scripts.grid import Grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot locations of Major TOM grid cells from a txt file on a world map with labels."
    )
    parser.add_argument(
        "txt_path",
        type=str,
        help="Path to a txt file with one grid cell id per line (e.g., 106D_246R).",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the output figure (PNG). Defaults to <txt_dir>/plot_<txt_basename>.png",
    )
    parser.add_argument(
        "--grid-dist",
        type=int,
        default=10,
        help="Grid distance in km used by Major TOM (default: 10).",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    return parser.parse_args()


_CELL_RE = re.compile(r"^\s*(\d+)([UD])_(\d+)([LR])\s*$")


def load_cell_ids(txt_path: str) -> List[str]:
    with open(txt_path, "r") as f:
        cells = [ln.strip() for ln in f if ln.strip()]
    return cells


def parse_cell_id(cell_id: str) -> Tuple[str, str]:
    """
    Parse a Major TOM grid cell id like '106D_246R' into ('106D', '246R').
    """
    m = _CELL_RE.match(cell_id)
    if not m:
        raise ValueError(f"Invalid grid cell id format: {cell_id}")
    row = f"{m.group(1)}{m.group(2)}"
    col = f"{m.group(3)}{m.group(4)}"
    return row, col


def cells_to_latlons(cells: List[str], grid: Grid) -> Tuple[List[float], List[float]]:
    rows: List[str] = []
    cols: List[str] = []
    for cid in tqdm(cells, desc="Processing cells"):
        r, c = parse_cell_id(cid)
        rows.append(r)
        cols.append(c)
    lats, lons = grid.rowcol2latlon_vectorized_needs_testing(rows, cols)
    return lats, lons


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
    label: str = None,
):
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
        transform=ccrs.PlateCarree(),
    )


def _annotate_labels(ax, labels: List[str], lats: List[float], lons: List[float], color: str) -> None:
    for lab, lat, lon in zip(labels, lats, lons):
        ax.text(
            lon,
            lat + 0.25,
            lab,
            fontsize=2,
            color=color,
            ha="center",
            va="bottom",
            zorder=6,
            transform=ccrs.PlateCarree(),
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1.5),
        )


def main() -> None:
    args = parse_args()
    txt_path = os.path.abspath(args.txt_path)
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"txt_path does not exist: {txt_path}")

    save_path = args.save
    if save_path is None:
        base = os.path.splitext(os.path.basename(txt_path))[0]
        save_path = os.path.join(os.path.dirname(txt_path), f"plot_{base}.png")

    cells = load_cell_ids(txt_path)
    if len(cells) == 0:
        raise ValueError("No grid cell ids found in the txt file.")
    
    print(f"Loading grid ...")
    grid = Grid(args.grid_dist)
    print(f"Converting cells to latlons ...")
    lats, lons = cells_to_latlons(cells, grid)

    print(f"Plotting {len(cells)} cells from {txt_path} ...")
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": ccrs.PlateCarree()})
    ax.coastlines(resolution="50m", color="#424242", linewidth=0.8, zorder=3)
    ax.set_global()
    ax.set_axis_off()

    _scatter_with_halo(
        ax=ax,
        lons=lons,
        lats=lats,
        color="#2979ff",
        marker="o",
        size=10,
        edgecolor="k",
        linewidth=0.4,
        zorder=4.5,
        label=f"Cells (n={len(cells)})",
    )
    _annotate_labels(ax, cells, lats, lons, color="#0d47a1")

    ax.set_title(args.title if args.title else "Major TOM Grid Cells", fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()


