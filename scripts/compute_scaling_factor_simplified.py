import argparse
import torch
import numpy as np
import os
import sys
from glob import glob
from tqdm import tqdm
import libs.autoencoder
import importlib
import json
import lmdb
import pickle
import random

# Modality shape mapping for reconstructing arrays from bytes
MODALITY_SHAPES = {
    "S2L2A_B02_B03_B04_B08": (16, 24, 24),
    "S2L2A_B05_B06_B07_B11_B12_B8A": (16, 12, 12),
    "S2L2A_B01_B09": (16, 4, 4),
    "S2L1C_B02_B03_B04_B08": (16, 24, 24),
    "S2L1C_B05_B06_B07_B11_B12_B8A": (16, 12, 12),
    "S2L1C_B01_B09_B10": (16, 4, 4),
    "S2L1C_cloud_mask": (16, 24, 24),
    "S1RTC_vh_vv": (16, 24, 24),
    "DEM_DEM": (16, 8, 8),
    "LULC_LULC": (16, 24, 24),
}

def load_config(config_path):
    """Load configuration from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.get_config()

def compute_scaling_factor(latent_path, model_path, modality, selected_keys, batch_size=32):
    """Compute scaling factor for a single modality using LMDB and a subset of keys."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{modality}] Using model from {model_path}")
    
    modality_path = os.path.join(latent_path, modality)

    # Check if LMDB exists
    lmdb_data_file = os.path.join(modality_path, "data.mdb")
    if not os.path.exists(lmdb_data_file):
        print(f"[{modality}] WARNING: No data.mdb file found in {modality_path}")
        return None

    # Open LMDB environment
    env = lmdb.open(
        modality_path,
        max_readers=1,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    # Ensure keys are in bytes and unique
    keys = [k.encode('ascii') if isinstance(k, str) else k for k in selected_keys]
    total_entries = len(keys)
    print(f"[{modality}] Processing {total_entries} keys")

    if total_entries == 0:
        print(f"[{modality}] WARNING: No keys provided for processing")
        env.close()
        return None

    # Get expected shape for this modality
    if modality not in MODALITY_SHAPES:
        print(f"[{modality}] WARNING: Unknown modality shape, skipping")
        env.close()
        return None

    expected_shape = MODALITY_SHAPES[modality]

    # Process in batches
    # Online statistics accumulators
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    batch_count = (total_entries + batch_size - 1) // batch_size
 
    # Keep a single read transaction for all batches
    with env.begin(write=False) as txn:
        for batch_idx in tqdm(range(batch_count), desc=f"[{modality}] Processing"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_entries)
            batch_keys = keys[start_idx:end_idx]

            for key in batch_keys:
                try:
                    value = txn.get(key)
                    if value is None:
                        continue  # key missing in this LMDB

                    data = pickle.loads(value)

                    # Depending on storage format, extract bytes
                    if isinstance(data, dict):
                        bytes_data = data.get(modality)
                        if bytes_data is None:
                            continue
                    else:
                        bytes_data = data

                    latent_np = np.frombuffer(bytes_data, dtype=np.float32).reshape(expected_shape)
                    latent_tensor = torch.from_numpy(latent_np)

                    mean, logvar = torch.chunk(latent_tensor, 2, dim=0)
                    logvar = torch.clamp(logvar, -30.0, 20.0)
                    std = torch.exp(0.5 * logvar)
                    z = mean + std * torch.randn_like(mean)

                    # Flatten to compute running sums
                    flat = z.view(-1)
                    total_sum += flat.sum().item()
                    total_sq_sum += (flat * flat).sum().item()
                    total_count += flat.numel()

                except Exception as e:
                    print(f"[{modality}] WARNING: Error processing key {key}: {e}")
                    continue
 
    env.close()
 
    if total_count == 0:
        raise ValueError(f"[{modality}] No valid samples generated")

    mean_val = total_sum / total_count
    var_val = max(total_sq_sum / total_count - mean_val ** 2, 0.0)
    std = var_val ** 0.5
 
    scaling_factor = 1.0 / std

    return {"std": std, "scaling_factor": scaling_factor, "modality": modality}

def main():
    parser = argparse.ArgumentParser(description='Compute scaling factors for latent representations')
    parser.add_argument('--latent_path', type=str, required=True, 
                        help='Path to the directory containing modality subfolders with latent .npy files')
    parser.add_argument('--config_path', type=str, required=True,
                        help='Path to the configuration file (e.g., small_world_S2.py)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--grid_cells_file', type=str, required=True,
                        help='Path to text file containing grid cell identifiers')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of random grid_cell_patch entries to sample (default: use all)')
    args = parser.parse_args()
    
    torch.set_grad_enabled(False)
    
    print(f"Loading configuration from {args.config_path}")
    config = load_config(args.config_path)

    # Load grid cells and create keys
    with open(args.grid_cells_file, 'r') as f:
        grid_cells = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(grid_cells)} unique grid cells from {args.grid_cells_file}")

    # Expand into grid_cell_patch keys (0-35)
    expanded_keys = []
    for cell in grid_cells:
        for num in range(36):
            expanded_keys.append(f"{cell}_patch_{num}")

    print(f"Total expanded keys: {len(expanded_keys)}")

    # Randomly sample if needed
    if args.num_samples is not None and args.num_samples > 0 and args.num_samples < len(expanded_keys):
        selected_keys = random.sample(expanded_keys, args.num_samples)
        print(f"Randomly selected {len(selected_keys)} keys for scaling factor computation")
    else:
        selected_keys = expanded_keys
        print("Using all expanded keys for scaling factor computation")

    selected_keys.sort()

    results = {}
    for modality, autoencoder_config in config.autoencoders.items():
        print(f"\nProcessing modality: {modality}")
        results[modality] = compute_scaling_factor(
            args.latent_path,
            autoencoder_config.pretrained_path,
            modality,
            selected_keys,
            args.batch_size
        )
        
    
    # Pretty print the results
    print("\nSummary of scaling factors:")
    print("-" * 50)
    for modality, result in results.items():
        print(f"Modality: {modality}")
        print(f"  Standard deviation: {result['std']:.6f}")
        print(f"  scale_factor: {result['scaling_factor']:.6f}")
        print("-" * 50)
    

if __name__ == "__main__":
    main() 