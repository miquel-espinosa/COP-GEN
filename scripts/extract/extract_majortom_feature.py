from libs.autoencoder import get_model
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import argparse
from pathlib import Path
import glob

from majortom.NMajorTOM import NMajorTOM
import pyarrow.parquet as pq
import geopandas as gpd
import pandas as pd

torch.manual_seed(0)
np.random.seed(0)

PATCH_SIZE = 256
GRID_SIZE = 4  # 4x4 grid of patches

SATELLITE_CONFIGS = {
    'S2L2A': {
        'tif_bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'cloud_mask'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),  # Crop to 1024x1024
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    },
    'S2L1C': {
        'tif_bands': ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'cloud_mask'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    },
    'S1RTC': {
        'tif_bands': ['vv', 'vh'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    },
    'DEM': {
        'tif_bands': ['DEM', 'compressed'],
        'png_bands': ['thumbnail'],
        'tif_transforms': [],
        'png_transforms': [
            transforms.Resize(1068), # First, interpolate to match the resolution of the other modalities (1068x1068)
            transforms.CenterCrop(PATCH_SIZE * GRID_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ]
    }
}

def fix_crs(df):
    if df['crs'].iloc[0].startswith('EPSG:EPSG:'):
        df['crs'] = df['crs'].str.replace('EPSG:EPSG:', 'EPSG:', regex=False)
    return df

def load_metadata(path):
    df = pq.read_table(path).to_pandas()
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df.timestamp)
    df = fix_crs(df)
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.centre_lon, df.centre_lat), crs=df.crs.iloc[0]
    )
    return gdf

def process_satellite(subset_path, satellite_types, bands_per_type, ratio_train_test, seed):
    """Process multiple satellite types simultaneously while ensuring they're paired"""
    modalities = {}
    filtered_dfs = {}
    
    # First, load metadata for all satellite types
    for sat_type in satellite_types:
        metadata_path = os.path.join(subset_path, f"Core-{sat_type}", "metadata.parquet")
        if not os.path.exists(metadata_path):
            print(f"Skipping {sat_type}: metadata not found at {metadata_path}")
            continue
        
        gdf = load_metadata(metadata_path)
        local_dir = os.path.join(subset_path, f"Core-{sat_type}")
        
        # Split bands into tif and png based on configuration
        tif_bands = [b for b in bands_per_type[sat_type] if b in SATELLITE_CONFIGS[sat_type]['tif_bands']]
        png_bands = [b for b in bands_per_type[sat_type] if b in SATELLITE_CONFIGS[sat_type]['png_bands']]
        
        print(f"\nChecking files for {sat_type}...")
        
        # Check which indices have all required files
        valid_indices = []
        
        for idx in tqdm(range(len(gdf)), desc=f"Validating {sat_type} samples", unit="samples"):
            row = gdf.iloc[idx]
            grid_cell = row.grid_cell
            row_id = grid_cell.split('_')[0]
            product_id = row.product_id if 'product_id' in row.index else "id"
            
            base_path = os.path.join(local_dir, row_id, grid_cell, product_id)
            all_files_exist = True
            
            # Check TIF files
            for band in tif_bands:
                if not os.path.exists(os.path.join(base_path, f"{band}.tif")):
                    all_files_exist = False
                    break
            
            # Check PNG files
            if all_files_exist:  # Only check PNGs if TIFs exist
                for band in png_bands:
                    if not os.path.exists(os.path.join(base_path, f"{band}.png")):
                        all_files_exist = False
                        break
            
            if all_files_exist:
                valid_indices.append(idx)
        
        filtered_df = gdf.iloc[valid_indices].copy()
        print(f"Found {len(filtered_df)} valid samples out of {len(gdf)} for {sat_type}")
        filtered_dfs[sat_type] = filtered_df
    
    # Find common grid cells across all modalities
    grid_cell_sets = {
        source: set(df['grid_cell'].unique())
        for source, df in filtered_dfs.items()
    }
    
    # Find intersection of all grid cell sets
    common_grid_cells = set.intersection(*grid_cell_sets.values())
    print(f"\nFound {len(common_grid_cells)} common grid cells across all modalities")
    
    # Filter all modalities to keep only common grid cells
    for sat_type in satellite_types:
        if sat_type not in filtered_dfs:
            continue
            
        df = filtered_dfs[sat_type]
        df = df[df['grid_cell'].isin(common_grid_cells)]
        print(f"{sat_type}: {len(df)} samples for common grid cells")
        
        modalities[sat_type] = {
            'df': df,
            'local_dir': os.path.join(subset_path, f"Core-{sat_type}"),
            'tif_bands': tif_bands,
            'png_bands': png_bands,
            'tif_transforms': SATELLITE_CONFIGS[sat_type]['tif_transforms'],
            'png_transforms': SATELLITE_CONFIGS[sat_type]['png_transforms']
        }
    
    dataset = NMajorTOM(modalities=modalities, ratio_train_test=ratio_train_test, seed=seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
    )

    model = get_model("assets/stable-diffusion/autoencoder_kl.pth")
    # model = nn.DataParallel(model)
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    
    return model, dataloader, device

def extract_features(model, dataloader, satellite_types, bands_per_type, output_dir, device, flip=False, save_png=False):
    """Extract features for all modalities simultaneously while ensuring they're paired"""
    train_idx = 0
    test_idx = 0
    
    # First, create all directories
    for sat_type in satellite_types:
        sat_base_dir = f"{sat_type}_{'_'.join(bands_per_type[sat_type])}"
        train_dir = os.path.join(output_dir, 'train', sat_base_dir)
        test_dir = os.path.join(output_dir, 'test', sat_base_dir)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

    # Create DataFrames to store metadata
    train_metadata = []
    test_metadata = []
    
    for batch in tqdm(dataloader, desc="Processing paired modalities"):
        
        # Verify that all modalities have the same grid_cells and same splits
        modalities = list(batch.keys())
        ref_grid_cells = batch[modalities[0]]['grid_cell']
        ref_splits = batch[modalities[0]]['split']
        for modality in modalities:
            if batch[modality]['grid_cell'] != ref_grid_cells:
                raise ValueError(f"Mismatched grid_cells found: {batch[modality]['grid_cell']} != {ref_grid_cells}")
            if batch[modality]['split'] != ref_splits:
                raise ValueError(f"Mismatched splits found: {batch[modality]['split']} != {ref_splits}")        
        
        # Process each modality in the paired batch
        for sat_type in satellite_types:
            if sat_type not in batch:
                continue
            
            modality_data = batch[sat_type]
            img = torch.cat([v for k, v in modality_data.items() if k in bands_per_type[sat_type]], dim=1)
            
            # Split into patches
            B, C, H, W = img.shape
            patches = img.unfold(2, PATCH_SIZE, PATCH_SIZE).unfold(3, PATCH_SIZE, PATCH_SIZE)
            patches = patches.contiguous().view(B, C, GRID_SIZE, GRID_SIZE, PATCH_SIZE, PATCH_SIZE)
            patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
            patches = patches.view(B * GRID_SIZE * GRID_SIZE, C, PATCH_SIZE, PATCH_SIZE)
            # Create flipped versions
            if flip:
                patches = torch.cat([patches, patches.flip(dims=[-1])], dim=0)
            patches = patches.to(device)
            
            if sat_type == 'DEM':
                patches = patches.repeat(1, 3, 1, 1)
            
            moments = model(inputs=patches, fn="encode_moments")
            moments = moments.detach().cpu().numpy()
            
            # Create train/test directories for this modality
            sat_base_dir = f"{sat_type}_{'_'.join(bands_per_type[sat_type])}"
            train_dir = os.path.join(output_dir, 'train', sat_base_dir)
            test_dir = os.path.join(output_dir, 'test', sat_base_dir)
            
            # Get split information and metadata for the current batch
            splits = modality_data['split']
            grid_cells = modality_data['grid_cell']
            
            for i in range(0, len(moments), 2 if flip else 1):
                patch_idx = i // (2 if flip else 1)
                batch_idx = patch_idx // (GRID_SIZE * GRID_SIZE)
                patch_pos = patch_idx % (GRID_SIZE * GRID_SIZE)
                patch_row = patch_pos // GRID_SIZE
                patch_col = patch_pos % GRID_SIZE
                
                # Determine which split this patch belongs to
                is_train = splits[batch_idx] == 'train'
                
                # Choose appropriate directory and counter
                save_dir = train_dir if is_train else test_dir
                current_idx = train_idx if is_train else test_idx
                
                metadata_entry = {
                    'feature_idx': current_idx + i,
                    'grid_cell': grid_cells[batch_idx],
                    'patch_row': patch_row,
                    'patch_col': patch_col,
                    'satellite': sat_type,
                    'bands': '_'.join(bands_per_type[sat_type]),                                        
                }
                
                # Save original patch features
                np.save(os.path.join(save_dir, f"{current_idx + i}.npy"), moments[i])
                if save_png:
                    patch_i = patches[i].detach().cpu()
                    patch_i = (patch_i + 1) / 2  # Denormalize from [-1,1] to [0,1]
                    torchvision.utils.save_image(patch_i, os.path.join(save_dir, f"{current_idx + i}.png"))
                
                if is_train:
                    train_metadata.append(metadata_entry)
                else:
                    test_metadata.append(metadata_entry)
                
                # Save flipped patch features if enabled
                if flip:
                    flipped_entry = metadata_entry.copy()
                    flipped_entry['feature_idx'] = current_idx + i + 1
                    flipped_entry['flipped'] = True
                    if is_train:
                        train_metadata.append(flipped_entry)
                    else:
                        test_metadata.append(flipped_entry)
                    
                    np.save(os.path.join(save_dir, f"{current_idx + i + 1}.npy"), moments[i + 1])
                    if save_png:
                        patch_i_flipped = patches[i + 1].detach().cpu()
                        patch_i_flipped = (patch_i_flipped + 1) / 2 # Denormalize from [-1,1] to [0,1]
                        torchvision.utils.save_image(patch_i_flipped, os.path.join(save_dir, f"{current_idx + i + 1}.png"))
            
        # Update counters based on the number of patches in this batch
        num_patches = len(moments)
        if is_train:
            train_idx += num_patches
        else:
            test_idx += num_patches
    
    # Save metadata
    train_df = pd.DataFrame(train_metadata)
    test_df = pd.DataFrame(test_metadata)
    train_df.to_parquet(os.path.join(output_dir, 'train_final_metadata.parquet'))
    test_df.to_parquet(os.path.join(output_dir, 'test_final_metadata.parquet'))
    
    print(f"Saved {train_idx} training and {test_idx} testing sets of patch features")
    print(f"Saved metadata for {len(train_df)} training and {len(test_df)} testing samples")

def main():
    parser = argparse.ArgumentParser(description='Extract features from MajorTOM dataset')
    parser.add_argument('--subset_path', required=True, help='Path to the subset folder')
    parser.add_argument('--bands', nargs='+', required=True, help='Bands to process (e.g., B1 B2 B3 DEM vv vh)')
    parser.add_argument('--ratio_train_test', type=float, default=0.95, help='Ratio of training to testing data')
    parser.add_argument('--flip', action='store_true', help='Flip the patches')
    parser.add_argument('--save_png', action='store_true', help='Save PNG visualisations of patches')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Get subset name from path
    subset_name = Path(args.subset_path).name
    
    print("Flip is set to", args.flip)
    print("Seed is set to", args.seed)
    print("Subset path is", args.subset_path)
    print("Bands are", args.bands)
    print("Ratio train test is", args.ratio_train_test)

    # Create the main output directory
    all_bands = '_'.join(sorted(args.bands))
    output_dir = os.path.join(args.subset_path, f"encoded_{subset_name}_{all_bands}")
    os.makedirs(output_dir, exist_ok=True)

    # Group bands by satellite type
    bands_per_type = {}
    satellite_types = []
    for sat_type, config in SATELLITE_CONFIGS.items():
        all_sat_bands = config['tif_bands'] + config['png_bands']
        sat_bands = [b for b in args.bands if b in all_sat_bands]
        if sat_bands:
            bands_per_type[sat_type] = sat_bands
            satellite_types.append(sat_type)

    if satellite_types:
        # Process all satellite types together
        model, dataloader, device = process_satellite(args.subset_path, satellite_types, bands_per_type, args.ratio_train_test, args.seed)
        extract_features(model, dataloader, satellite_types, bands_per_type, output_dir, device, flip=args.flip, save_png=args.save_png)

if __name__ == "__main__":
    main()