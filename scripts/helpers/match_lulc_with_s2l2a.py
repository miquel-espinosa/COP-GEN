import pickle
import shutil
from pathlib import Path
from tqdm import tqdm

def load_pkl(pkl_path):
    """Load pickle file and return the data."""
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)

def save_pkl(data, pkl_path):
    """Save data to pickle file."""
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

def extract_grid_ids_from_s2l2a(s2l2a_entries):
    """
    Extract unique grid IDs from S2L2A entries.
    S2L2A entries are like: ('626U_18L', 'S2A_MSIL2A_...')
    We need to extract the grid ID part: '626U_18L'
    """
    grid_ids = set()
    for entry in s2l2a_entries:
        if isinstance(entry, tuple) and len(entry) >= 1:
            grid_id = entry[0]  # e.g., '626U_18L'
            grid_ids.add(grid_id)
    return grid_ids

def extract_grid_ids_from_lulc(lulc_entries):
    """
    Extract unique grid IDs from LULC entries.
    LULC entries are like: ('626U_999L', '626U_999L_2022')
    We need to extract the grid ID part: '626U_999L'
    """
    grid_ids = set()
    for entry in lulc_entries:
        if isinstance(entry, tuple) and len(entry) >= 1:
            grid_id = entry[0]  # e.g., '626U_999L'
            grid_ids.add(grid_id)
    return grid_ids

def main():
    # Define paths
    base_dir = Path('./data/majorTOM/edinburgh')
    s2l2a_pkl_path = base_dir / 'Core-S2L2A' / '.cache_S2L2A.pkl'
    lulc_pkl_path = base_dir / 'Core-LULC' / '.cache_LULC.pkl'
    lulc_root = base_dir / 'Core-LULC'
    
    print("Loading pickle files...")
    s2l2a_entries = load_pkl(s2l2a_pkl_path)
    lulc_entries = load_pkl(lulc_pkl_path)
    
    print(f"S2L2A entries: {len(s2l2a_entries)}")
    print(f"LULC entries: {len(lulc_entries)}")
    
    # Extract grid IDs
    s2l2a_grid_ids = extract_grid_ids_from_s2l2a(s2l2a_entries)
    lulc_grid_ids = extract_grid_ids_from_lulc(lulc_entries)
    
    print(f"\nUnique S2L2A grid IDs: {len(s2l2a_grid_ids)}")
    print(f"Unique LULC grid IDs: {len(lulc_grid_ids)}")
    
    # Find LULC grid IDs that are not in S2L2A
    extra_lulc_grid_ids = lulc_grid_ids - s2l2a_grid_ids
    
    print(f"\nExtra LULC grid IDs (not in S2L2A): {len(extra_lulc_grid_ids)}")
    
    if extra_lulc_grid_ids:
        print(f"\nRemoving {len(extra_lulc_grid_ids)} extra LULC folders...")
        
        # Delete extra LULC folders
        for grid_id in tqdm(extra_lulc_grid_ids, desc="Deleting folders"):
            # Parse grid_id to get folder structure: '626U_999L' -> '626U/999L'
            parts = grid_id.split('_')
            if len(parts) == 2:
                grid_prefix = parts[0]  # '626U'
                grid_suffix = parts[1]  # '999L'
                folder_path = lulc_root / grid_prefix / grid_suffix
                
                if folder_path.exists():
                    shutil.rmtree(folder_path)
                    print(f"  Deleted: {folder_path}")
        
        # Update LULC pkl file to only include entries that match S2L2A
        filtered_lulc_entries = [
            entry for entry in lulc_entries
            if isinstance(entry, tuple) and len(entry) >= 1 and entry[0] in s2l2a_grid_ids
        ]
        
        print(f"\nUpdating LULC pkl file...")
        print(f"  Original entries: {len(lulc_entries)}")
        print(f"  Filtered entries: {len(filtered_lulc_entries)}")
        
        save_pkl(filtered_lulc_entries, lulc_pkl_path)
        print(f"  Saved updated pkl to: {lulc_pkl_path}")
    else:
        print("\nNo extra LULC folders to remove. All LULC grid IDs match S2L2A.")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
