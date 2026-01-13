import os
import glob
from collections import defaultdict
import numpy as np
import rasterio

def find_cloud_masks(root):
    """Recursively find all cloud_mask.tif files under the given root path."""
    pattern = os.path.join(root, '**', 'cloud_mask.tif')
    files = glob.glob(pattern, recursive=True)
    
    matched_files = {}
    for f in files:
        # Extract relative matching key for comparison (excluding satellite type)
        rel_path = os.path.relpath(f, root)
        parts = rel_path.split(os.sep)
        # Expect structure like: "100U/100U_825L/S2B_MSIL2A_20181230T152639.../cloud_mask.tif"
        if len(parts) >= 4:
            # Remove the Core-* and timestamp at the end
            key = os.path.join(*parts[0:3])  # "100U/100U_825L/S2B_MSIL2A_20181230T152639"
            key = "_".join(key.split("_")[0:4])  # Strip the trailing timestamp
            matched_files[key] = f
            # Remove L2A/L1C if it exists from the key
            key = key.replace("L2A_", "_")
            key = key.replace("L1C_", "_")
            matched_files[key] = f
    return matched_files

def compare_cloud_masks(root1, root2):
    masks1 = find_cloud_masks(root1)
    masks2 = find_cloud_masks(root2)
    
    common_keys = set(masks1.keys()) & set(masks2.keys())
    print(f"Found {len(common_keys)} common cloud_mask.tif files.")

    unique_values1 = set()
    unique_values2 = set()
    min_val1, max_val1 = float('inf'), float('-inf')
    min_val2, max_val2 = float('inf'), float('-inf')

    for key in sorted(common_keys):
        file1 = masks1[key]
        file2 = masks2[key]
        
        with rasterio.open(file1) as src1, rasterio.open(file2) as src2:
            data1 = src1.read(1)
            data2 = src2.read(1)

            # Collect unique values
            unique_values1.update(np.unique(data1))
            unique_values2.update(np.unique(data2))
            
            # Update min/max
            min_val1 = min(min_val1, np.min(data1))
            max_val1 = max(max_val1, np.max(data1))
            min_val2 = min(min_val2, np.min(data2))
            max_val2 = max(max_val2, np.max(data2))

            if np.array_equal(data1, data2):
                print(f"[MATCH] {key}", np.sum(data1), np.sum(data2))
            else:
                print(f"\033[91m[DIFFERENT] {key}\033[0m")

    print(f"\nRoot 1 ({root1}):")
    print(f"Unique values: {sorted(unique_values1)}")
    print(f"Min value: {min_val1}")
    print(f"Max value: {max_val1}")
    
    print(f"\nRoot 2 ({root2}):")
    print(f"Unique values: {sorted(unique_values2)}")
    print(f"Min value: {min_val2}")
    print(f"Max value: {max_val2}")

compare_cloud_masks('data/majorTOM/small_world/Core-S2L2A', 'data/majorTOM/small_world/Core-S2L1C')
