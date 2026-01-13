"""
Steps for creating a subset of the edinburgh dataset.

Copy tiles to subset directory:
- python3 scripts/create_subset.py

Create pkl files for each modality:
- python3 scripts/create_pkl_files.py --root_dir ./data/majorTOM/subset --modalities DEM S2L1C S2L2A S1RTC LULC

Create common grid cells:
accelerate launch --num_processes 1 encode_moments_vae.py \
  --cfg ./configs/vae/final/S2L2A/copgen_ae_kl_192x192_S2L2A_B4_3_2_8_latent_8.yaml \
  --output_dir ./data/majorTOM/subset \
  --patchify \
  --lulc_align \
  --save_common_grid_cells common_grid_cells.txt

Create lat-lon files:
python ./scripts/precompute_lat_lon.py \
    --grid_cells_txt ./data/majorTOM/subset/common_grid_cells.txt \ 
    --patch_side 6 \
    --format 3d_cartesian \
    --storage npy_dir \
    --output ./data/majorTOM/subset/3d_cartesian_lat_lon \       
    --workers 16

Create time mean files:
python3 ./scripts/precompute_time_mean.py \
    --s1_metadata ./data/majorTOM/subset/Core-S1RTC/metadata.parquet \
    --s2_metadata ./data/majorTOM/subset/Core-S2L2A/metadata.parquet \
    --grid_cells_txt ./data/majorTOM/subset/common_grid_cells.txt \
    --output ./data/majorTOM/subset/mean_timestamps.pkl \
    --histogram ./data/majorTOM/subset/mean_timestamps_diff.png
"""


import os
import shutil

# Source and destination directories
src_root = "edinburgh"
dst_root = "subset"

# Tiles to include
tiles_to_copy = {
    "618U_30L",
    "619U_15L",
    "619U_24L",
    "619U_30L",
    "619U_31L",
    "620U_25L",
    "622U_15L",
    "623U_23L",
    "624U_20L",
    "624U_24L",
    "625U_19L",
    "625U_31L",
    "626U_17L",
    "626U_18L",
    "627U_21L",
    "628U_18L",
    "632U_15L",
}

# Helper to safely copy directories
def copy_dir(src_dir, dst_dir):
    os.makedirs(os.path.dirname(dst_dir), exist_ok=True)
    print(f"Copying {src_dir} -> {dst_dir}")
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

# Process each core folder
for core_folder in os.listdir(src_root):
    core_path = os.path.join(src_root, core_folder)
    if not os.path.isdir(core_path):
        continue

    print(f"\nProcessing {core_folder}...")

    # Handle LULC differently
    if core_folder == "Core-LULC":
        for tile in tiles_to_copy:
            # Extract the upper ID (e.g. 619U) and lower ID (e.g. 15L)
            try:
                upper, lower = tile.split("_")
            except ValueError:
                continue  # Skip malformed names
            src_dir = os.path.join(core_path, upper, lower)
            if os.path.exists(src_dir):
                rel_path = os.path.relpath(src_dir, src_root)
                dst_dir = os.path.join(dst_root, rel_path)
                copy_dir(src_dir, dst_dir)

    else:
        # Standard structure: e.g. Core-DEM/Core-S2L1C/etc.
        for root, dirs, files in os.walk(core_path):
            for d in dirs:
                if d in tiles_to_copy:
                    src_dir = os.path.join(root, d)
                    rel_path = os.path.relpath(src_dir, src_root)
                    dst_dir = os.path.join(dst_root, rel_path)
                    copy_dir(src_dir, dst_dir)
