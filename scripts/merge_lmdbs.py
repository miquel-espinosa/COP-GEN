import os
import lmdb
import pickle
import argparse
from tqdm import tqdm

import sys
sys.path.append("./")
from encode_moments_vae import load_zarr_zip_blob

"""
This script merges multiple LMDBs into one based on shared keys.

Usage:
    python merge_lmdbs.py \
        --input_dir /path/to/lmdbs \
        --output_dir /path/to/merged_output
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Merge multiple LMDBs into one based on shared keys")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to folder containing individual LMDB subfolders"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Path to write merged LMDB (default: input_dir/merged_lmdb)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1000,
        help="Number of entries to write in each batch (default: 1000)"
    )
    parser.add_argument(
        "--max_entries", type=int, default=None,
        help="Maximum number of shared entries to process (default: process all)"
    )
    parser.add_argument(
        "--reference_modality", type=str, default=None,
        help="Manually specify which modality folder name to use as the reference (default: smallest LMDB)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_root = args.input_dir
    output_path = args.output_dir or os.path.join(input_root, "merged_lmdb")
    os.makedirs(output_path, exist_ok=True)
    batch_size = args.batch_size
    max_entries = args.max_entries
    
    print(f"🔍 Scanning for LMDB directories in: {input_root}")
    if max_entries:
        print(f"🎯 Will process maximum {max_entries} shared entries")
    
    # Find all LMDB directories (must contain data.mdb)
    lmdb_dirs = []
    for d in os.listdir(input_root):
        dir_path = os.path.join(input_root, d)
        if os.path.isdir(dir_path) and os.path.isfile(os.path.join(dir_path, "data.mdb")):
            lmdb_dirs.append(dir_path)
    
    if len(lmdb_dirs) != 10:
        print(f"⚠️  Warning: Found {len(lmdb_dirs)} LMDB directories, expected 10")
    
    # Create modality mapping: modality_name -> lmdb_pathds
    lmdb_paths = {os.path.basename(path): path for path in lmdb_dirs}
    modalities = list(lmdb_paths.keys())
    print(f"📦 Found modalities: {modalities}")
    
    # Determine reference modality (user-specified or smallest)
    smallest_modality = min(modalities, key=lambda m: os.path.getsize(os.path.join(lmdb_paths[m], "data.mdb")))

    if args.reference_modality:
        if args.reference_modality not in modalities:
            raise ValueError(
                f"Specified reference modality '{args.reference_modality}' not found among detected modalities: {modalities}"
            )
        reference_modality = args.reference_modality
        print(f"📌 Using user-specified reference modality: {reference_modality}")
    else:
        reference_modality = smallest_modality
        print(f"🔍 Using smallest modality as reference: {reference_modality}")
    
    # Open all LMDB environments (keep them open for efficiency)
    print("🚪 Opening all LMDB environments...")
    envs = {}
    txns = {}
    for modality, path in lmdb_paths.items():
        envs[modality] = lmdb.open(path, readonly=True, lock=False)
        txns[modality] = envs[modality].begin()
    
    # Open output LMDB
    map_size = 10 * 1024 ** 4  # 10 TB - generous size for merged data
    env_out = lmdb.open(output_path, map_size=map_size)
    
    try:
        # Get all keys from the reference (smallest) LMDB
        print(f"📋 Getting all keys from reference modality: {reference_modality}")
        reference_keys = []
        cursor = txns[reference_modality].cursor()
        for key, _ in cursor:
            reference_keys.append(key)
        cursor.close()
        
        print(f"🔢 Found {len(reference_keys)} keys in reference LMDB")
        
        # Process keys in batches for efficient writing
        shared_count = 0
        batch_data = []
        
        print("🔄 Processing keys and checking for shared entries...")
        
        with tqdm(total=len(reference_keys), desc="Processing keys") as pbar:
            for key in reference_keys:
                # Check if we've reached the maximum entries limit
                if max_entries:
                    if  shared_count >= max_entries:
                        print(f"\n🎯 Reached maximum entries limit: {max_entries}")
                        break
                
                # Check if key exists in all modalities
                modality_data = {}
                key_exists_in_all = True
                
                for modality in modalities:
                    value = txns[modality].get(key)
                    if value is None:
                        key_exists_in_all = False
                        break
                    if 'GOOGLE_EMBEDS' in modality:
                        zarr_array = load_zarr_zip_blob(value)
                        modality_data[modality] = zarr_array[:] # to numpy array
                    else:
                        modality_data[modality] = pickle.loads(value)
                
                if key_exists_in_all:
                    # Add to batch
                    batch_data.append((key, modality_data))
                    shared_count += 1
                    
                    # Write batch when full
                    if len(batch_data) >= batch_size:
                        write_batch(env_out, batch_data)
                        batch_data = []
                
                pbar.update(1)
        
        # Write remaining batch
        if batch_data:
            write_batch(env_out, batch_data)
        
        print(f"\n✅ Successfully merged {shared_count} shared entries")
        print(f"📊 Shared key ratio: {shared_count}/{len(reference_keys)} ({shared_count/len(reference_keys)*100:.1f}%)")
        
    finally:
        # Clean up: close all transactions and environments
        print("🧹 Cleaning up...")
        for txn in txns.values():
            txn.abort()  # Close read transactions
        for env in envs.values():
            env.close()
        env_out.close()
    
    print(f"🎉 Merge complete! Output LMDB: {output_path}")

def write_batch(env_out, batch_data):
    """Write a batch of data to the output LMDB efficiently"""
    with env_out.begin(write=True) as txn_out:
        for key, modality_data in batch_data:
            # Store as a dictionary with modality names as keys
            merged_value = pickle.dumps(modality_data)
            txn_out.put(key, merged_value)

if __name__ == "__main__":
    main()

