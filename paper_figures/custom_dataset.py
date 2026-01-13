
GRID_IDS = [
    "143D_1481R",
    "95U_112R",
    "195D_669L",
    "211D_500R",
    "215U_1019L",
    "248U_978R",
    "250U_409R",
    "256U_1125L",
    "272D_1525R",
]

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple


def parse_grid_id(grid_id: str) -> Tuple[str, str]:
    """
    Split a grid_id like '106D_246R' into ('106D', '246R').
    """
    parts = grid_id.split("_")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Malformed grid_id: {grid_id!r}")
    return parts[0], parts[1]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _copytree(src: Path, dst: Path) -> None:
    """
    Copy a directory tree, merging if the destination exists.
    """
    ensure_dir(dst.parent)
    shutil.copytree(src, dst, dirs_exist_ok=True)


def copy_modalities_for_grid(
    root_dir: Path,
    out_dir: Path,
    grid_id: str,
    modalities: Iterable[str],
    force_copy: bool = False,
) -> None:
    """
    Copy data for one grid_id across the selected modalities, preserving structure.
    Missing modalities or cells are skipped with a warning.
    """
    tile, row = parse_grid_id(grid_id)
    for modality in modalities:
        if modality in {"DEM", "S1RTC", "S2L1C", "S2L2A"}:
            src = root_dir / f"Core-{modality}" / tile / grid_id
            dst = out_dir / f"Core-{modality}" / tile / grid_id
            if not src.exists():
                print(f"[warn] Missing {modality} for {grid_id}: {src}")
                continue
            if dst.exists() and not force_copy:
                # Already copied
                continue
            print(f"[copy] {modality} {grid_id} → {dst}")
            _copytree(src, dst)
        elif modality == "LULC":
            src_dir = root_dir / "Core-LULC" / tile / row
            if not src_dir.exists():
                print(f"[warn] Missing LULC dir for {grid_id}: {src_dir}")
                continue
            dst_dir = out_dir / "Core-LULC" / tile / row
            ensure_dir(dst_dir)
            # Prefer exact-year match; fall back to any matching tif
            candidates: List[Path] = []
            exact = src_dir / f"{grid_id}_2021.tif"
            if exact.exists():
                candidates = [exact]
            else:
                candidates = list(src_dir.glob(f"{grid_id}_*.tif"))
            if not candidates:
                print(f"[warn] No LULC tif for {grid_id} in {src_dir}")
                continue
            for tif in candidates:
                dst = dst_dir / tif.name
                if dst.exists() and not force_copy:
                    continue
                print(f"[copy] LULC {grid_id} → {dst}")
                shutil.copy2(tif, dst)
        else:
            print(f"[skip] Unknown modality: {modality}")


def write_test_txt(out_dir: Path, grid_ids: Iterable[str]) -> Path:
    """
    Write test.txt under out_dir with one grid_id per line.
    """
    p = out_dir / "test.txt"
    ensure_dir(p.parent)
    content = "".join(f"{g}\n" for g in grid_ids)
    if p.exists():
        try:
            if p.read_text() == content:
                print(f"[ok] test.txt up-to-date at {p}")
                return p
        except Exception:
            pass
    p.write_text(content)
    print(f"[write] {p}")
    return p


def find_repo_root() -> Path:
    # file: <repo>/paper_figures/DEM_to_S2_distribution.py
    return Path(__file__).resolve().parent.parent


def caches_present_for_modalities(out_dir: Path, modalities: Iterable[str]) -> bool:
    """
    Heuristic: consider caches present if each available modality dir contains its .cache_<modality>.pkl.
    Modalities missing in the dataset are ignored.
    """
    all_ok = True
    for modality in modalities:
        mdir = out_dir / f"Core-{modality}"
        if not mdir.exists():
            # Not available; do not require cache
            continue
        cache = mdir / f".cache_{modality}.pkl"
        if not cache.exists():
            all_ok = False
    return all_ok


def maybe_run_create_pkl_files(out_dir: Path, modalities: Iterable[str], force: bool = False) -> None:
    if not force and caches_present_for_modalities(out_dir, modalities):
        print("[ok] Modality caches already present; skipping create_pkl_files")
        return
    scripts_dir = find_repo_root() / "scripts"
    cmd = [
        sys.executable,
        str(scripts_dir / "create_pkl_files.py"),
        "--root_dir",
        str(out_dir),
        "--modalities",
        *list(modalities),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def maybe_run_precompute_time_mean(test_txt: Path, out_dir: Path, force: bool = False) -> None:
    out_pkl = out_dir / "mean_timestamps_cache.pkl"
    out_png = out_dir / "mean_timestamps_diff.png"
    if not force and out_pkl.exists() and out_png.exists():
        print("[ok] mean timestamps already computed; skipping")
        return
    scripts_dir = find_repo_root() / "scripts"
    cmd = [
        sys.executable,
        str(scripts_dir / "precompute_time_mean.py"),
        "--grid_cells_txt",
        str(test_txt),
        "--output",
        str(out_pkl),
        "--histogram",
        str(out_png),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def latlon_cache_ready(latlon_dir: Path) -> bool:
    return (latlon_dir / "cells.npy").exists() and (latlon_dir / "latlon_patches.npy").exists()


def maybe_run_precompute_lat_lon(
    test_txt: Path,
    out_dir: Path,
    patch_side: int,
    workers: int,
    force: bool = False,
) -> None:
    latlon_dir = out_dir / "3d_cartesian_lat_lon_cache"
    if not force and latlon_cache_ready(latlon_dir):
        print("[ok] 3d_cartesian_lat_lon_cache already present; skipping")
        return
    scripts_dir = find_repo_root() / "scripts"
    cmd = [
        sys.executable,
        str(scripts_dir / "precompute_lat_lon.py"),
        "--grid_cells_txt",
        str(test_txt),
        "--patch_side",
        str(patch_side),
        "--format",
        "3d_cartesian",
        "--storage",
        "npy_dir",
        "--output",
        str(latlon_dir),
        "--workers",
        str(int(workers)),
    ]
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def build_dataset(
    root_dataset: Path,
    output_dataset: Path,
    grid_ids: Iterable[str],
    modalities: Iterable[str],
    patch_side: int,
    workers: int,
    force_copy: bool = False,
    force_processing: bool = False,
) -> None:
    root_dataset = root_dataset.resolve()
    output_dataset = output_dataset.resolve()
    print(f"[info] Source: {root_dataset}")
    print(f"[info] Output: {output_dataset}")
    ensure_dir(output_dataset)

    # Copy per grid_id
    for gid in grid_ids:
        copy_modalities_for_grid(root_dataset, output_dataset, gid, modalities, force_copy=force_copy)

    # Write test.txt
    test_txt = write_test_txt(output_dataset, grid_ids)

    # Derived caches
    maybe_run_create_pkl_files(output_dataset, modalities, force=force_processing)
    maybe_run_precompute_time_mean(test_txt, output_dataset, force=force_processing)
    maybe_run_precompute_lat_lon(test_txt, output_dataset, patch_side=patch_side, workers=workers, force=force_processing)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Create a custom dataset restricted to selected grid_ids.")
    parser.add_argument("--root_dataset", required=True, help="Path to the existing full dataset root (contains Core-*)")
    parser.add_argument("--output_dataset", required=True, help="Where to write the new custom dataset")
    parser.add_argument("--grid_ids", nargs="+", help="Grid ids like 106D_246R (space-separated). If omitted, use built-in GRID_IDS.")
    parser.add_argument("--grid_ids_txt", help="Optional path to a txt file with one grid id per line")
    parser.add_argument("--modalities", nargs="+", default=["DEM", "S2L1C", "S2L2A", "S1RTC", "LULC"], help="Modalities to include")
    parser.add_argument("--patch_side", type=int, default=6, help="Patch side passed to precompute_lat_lon")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2), help="Workers for lat/lon precompute")
    parser.add_argument("--force_copy", action="store_true", help="Force re-copy even if destination exists")
    parser.add_argument("--force_processing", action="store_true", help="Force re-run the processing steps")
    return parser.parse_args()


def _load_grid_ids(args: argparse.Namespace) -> List[str]:
    if args.grid_ids_txt:
        p = Path(args.grid_ids_txt)
        if not p.exists():
            raise FileNotFoundError(p)
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    if args.grid_ids:
        return list(args.grid_ids)
    return list(GRID_IDS)


def main() -> None:
    args = _parse_args()
    grid_ids = _load_grid_ids(args)
    build_dataset(
        root_dataset=Path(args.root_dataset),
        output_dataset=Path(args.output_dataset),
        grid_ids=grid_ids,
        modalities=args.modalities,
        patch_side=int(args.patch_side),
        workers=int(args.workers),
        force_copy=bool(args.force_copy),
        force_processing=bool(args.force_processing),
    )


if __name__ == "__main__":
    main()