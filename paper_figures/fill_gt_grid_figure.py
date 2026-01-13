"""
Script to generate thumbnail PNGs for S1 and S2 data subfolders.

Given a folder structure like:
./paper_figures/paper_figures_datasets/fill_gt_grid_figure/143D_1481R
├── S1A_IW_GRDH_..._rtc/
│   ├── vh.tif
│   └── vv.tif
├── S2A_MSIL2A_.../
│   ├── B02.tif, B03.tif, B04.tif, ...

This script will generate 192x192 center-cropped thumbnails as PNGs
saved in the parent directory.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import rasterio
from PIL import Image


def read_tif(path: Path) -> np.ndarray:
    """Read a tif file and return the array."""
    with rasterio.open(path) as src:
        arr = src.read()
    return arr


def center_crop(arr: np.ndarray, size: int = 192) -> np.ndarray:
    """Center crop an array to size x size. 
    Expects array shape (C, H, W) or (H, W).
    """
    if arr.ndim == 2:
        h, w = arr.shape
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        return arr[top:top + size, left:left + size]
    else:
        _, h, w = arr.shape
        top = max(0, (h - size) // 2)
        left = max(0, (w - size) // 2)
        return arr[:, top:top + size, left:left + size]


def save_tensor_as_png(tensor: torch.Tensor, path: Path):
    """
    Save a visualization tensor as a PNG.
    Accepts CHW or HWC; values expected in [0, 1].
    """
    arr = tensor.detach().cpu().numpy()
    
    # Remove batch dimension if present
    if arr.ndim == 4:
        arr = np.squeeze(arr, axis=0)
    
    # Convert channel-first (C,H,W) to (H,W,C) when C in {1,3}
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    
    # Clip to [0,1] and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255.0).round().astype(np.uint8)
    
    # If single-channel with trailing dim 1, squeeze it for 'L' mode
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 1:
        arr_uint8 = arr_uint8[:, :, 0]
    
    Image.fromarray(arr_uint8).save(path)


def is_s1_folder(folder_name: str) -> bool:
    """Check if folder name indicates S1 data."""
    return (folder_name.startswith("S1A_") or folder_name.startswith("S1B_")) and "_rtc" in folder_name


def is_s2_folder(folder_name: str) -> bool:
    """Check if folder name indicates S2 data."""
    return folder_name.startswith("S2A_") or folder_name.startswith("S2B_")


def process_s1_folder(folder: Path, output_dir: Path, crop_size: int = 192):
    """Process an S1 folder and save thumbnail."""
    from visualisations.thumbnails_vis import s1rtc_thumbnail_torch
    
    vh_path = folder / "vh.tif"
    vv_path = folder / "vv.tif"
    
    if not vh_path.exists() or not vv_path.exists():
        print(f"  Skipping {folder.name}: missing vh.tif or vv.tif")
        return
    
    # Read tifs
    vh = read_tif(vh_path).squeeze()
    vv = read_tif(vv_path).squeeze()
    
    # Center crop
    vh = center_crop(vh, crop_size)
    vv = center_crop(vv, crop_size)
    
    # Convert to tensors
    vh_tensor = torch.from_numpy(vh).float()
    vv_tensor = torch.from_numpy(vv).float()
    
    # Generate thumbnail
    thumbnail = s1rtc_thumbnail_torch(vv_tensor, vh_tensor, return_uint8=False)
    
    # Save
    output_path = output_dir / f"{folder.name}.png"
    save_tensor_as_png(thumbnail, output_path)
    print(f"  Saved: {output_path.name}")


def process_s2_folder(folder: Path, output_dir: Path, crop_size: int = 192):
    """Process an S2 folder and save thumbnail."""
    from visualisations.thumbnails_vis import s2_thumbnail_torch
    
    b02_path = folder / "B02.tif"
    b03_path = folder / "B03.tif"
    b04_path = folder / "B04.tif"
    
    if not b02_path.exists() or not b03_path.exists() or not b04_path.exists():
        print(f"  Skipping {folder.name}: missing B02.tif, B03.tif, or B04.tif")
        return
    
    # Read tifs
    b02 = read_tif(b02_path).squeeze()
    b03 = read_tif(b03_path).squeeze()
    b04 = read_tif(b04_path).squeeze()
    
    # Center crop
    b02 = center_crop(b02, crop_size)
    b03 = center_crop(b03, crop_size)
    b04 = center_crop(b04, crop_size)
    
    # Convert to tensors
    b02_tensor = torch.from_numpy(b02).float()
    b03_tensor = torch.from_numpy(b03).float()
    b04_tensor = torch.from_numpy(b04).float()
    
    # Generate thumbnail (B04=red, B03=green, B02=blue)
    thumbnail = s2_thumbnail_torch(b04_tensor, b03_tensor, b02_tensor, return_uint8=False)
    
    # Save
    output_path = output_dir / f"{folder.name}.png"
    save_tensor_as_png(thumbnail, output_path)
    print(f"  Saved: {output_path.name}")


def process_directory(input_dir: Path, crop_size: int = 192):
    """Process all subfolders in the input directory."""
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Output directory is the same as input (parent directory of subfolders)
    output_dir = input_dir
    
    # Get all subfolders
    subfolders = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    
    print(f"Found {len(subfolders)} subfolders in {input_dir}")
    
    s1_count = 0
    s2_count = 0
    skipped_count = 0
    
    for folder in subfolders:
        folder_name = folder.name
        
        if is_s1_folder(folder_name):
            print(f"Processing S1: {folder_name}")
            process_s1_folder(folder, output_dir, crop_size)
            s1_count += 1
        elif is_s2_folder(folder_name):
            print(f"Processing S2: {folder_name}")
            process_s2_folder(folder, output_dir, crop_size)
            s2_count += 1
        else:
            print(f"Skipping unknown folder type: {folder_name}")
            skipped_count += 1
    
    print(f"\nSummary:")
    print(f"  S1 folders processed: {s1_count}")
    print(f"  S2 folders processed: {s2_count}")
    print(f"  Skipped: {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate 192x192 center-cropped thumbnails for S1 and S2 data."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default="./paper_figures/paper_figures_datasets/fill_gt_grid_figure/143D_1481R",
        help="Path to the directory containing S1 and S2 subfolders."
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=192,
        help="Size of the center crop (default: 192)."
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    process_directory(input_dir, args.crop_size)


if __name__ == "__main__":
    main()

