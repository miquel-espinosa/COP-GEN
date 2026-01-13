import argparse
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name
from train_vae import load_conf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Compute min/max values of satellite dataset")
    parser.add_argument("--cfg", help="Config file path", type=str, required=True)
    parser.add_argument("--save_dir", help="Path to save the visualisations", type=str, default=".")
    parser.add_argument("--batch_size", help="Batch size for processing", type=int, default=16)
    parser.add_argument("--num_workers", help="Number of workers for data loading", type=int, default=4)
    parser.add_argument("--use_percentiles", help="Use percentile clipping", action="store_true")
    parser.add_argument("--with_negatives", help="Include negative values in the analysis", action="store_true")
    parser.add_argument("--nodata_threshold", help="Nodata value threshold (default: -32768.0)", type=float, default=-32768.0)
    parser.add_argument("--data_dir", help="Override data.data_dir from config", type=str)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    if args.data_dir:
        args.cfg.setdefault("data", {})["data_dir"] = args.data_dir
    return args

def plot_band_histograms(band_values, bands, filename):
    """
    Plot histograms for each band in the dataset.
    
    Args:
        band_values (dict): Dictionary mapping band names to arrays of values
        bands (list): List of band names to plot
    """
    plt.figure(figsize=(10, 6))
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    for i, band in enumerate(bands):
        values = band_values[band]
        plt.hist(values, bins=100, alpha=0.5, color=colors[i % len(colors)], label=f'Band {band}')
    
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency') 
    # plt.ylim(0, 1000)
    plt.title(f'Histogram of Band Values')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved histogram plot to {filename}")
    plt.close()
    
def print_min_max_values(global_mins, global_maxs, tif_bands, name):
    """
    Print minimum and maximum values for each band and overall statistics.
    
    Args:
        global_mins (dict): Dictionary of minimum values per band
        global_maxs (dict): Dictionary of maximum values per band
        tif_bands (list): List of band names
        name (str): Name to print in the title
    """
    print("="*50)
    print(f"MIN/MAX VALUES FOR {name} DATASET")
    print("="*50)
    
    for band in tif_bands:
        print(f"Band {band}:")
        print(f"  Min: {global_mins[band]:.6f}")
        print(f"  Max: {global_maxs[band]:.6f}")
        print(f"  Range: {global_maxs[band] - global_mins[band]:.6f}")
        print("-"*50)
    
    all_min = min(global_mins.values())
    all_max = max(global_maxs.values())
    print("Overall statistics:")
    print(f"  Global min: {all_min:.6f}")
    print(f"  Global max: {all_max:.6f}")
    print(f"  Global range: {all_max - all_min:.6f}")
    print("="*50)
    print("\n")

def main(args):
    data_cfg = args.cfg["data"]
    data_cfg['preprocess_bands'] = False  # Crucial - we want raw values
    dataset = construct_class_by_name(**data_cfg)
    print(f"Dataset initialized with {len(dataset)} images")
    
    # Create plots subdirectory inside save_dir
    plots_dir = os.path.join(args.save_dir)
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Saving plots to directory: {plots_dir}")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # STEP 1: RAW BAND VALUES =============================================================
    
    global_mins = {band: float('inf') for band in data_cfg['tif_bands']}
    global_maxs = {band: float('-inf') for band in data_cfg['tif_bands']}
    band_values = {band: [] for band in data_cfg['tif_bands']}
    
    for num, batch in enumerate(tqdm(dataloader, desc="Computing min/max values")):
        images = batch['image']  # Shape: [B, C, H, W]
        for i, band in enumerate(data_cfg['tif_bands']):
            band_data = images[:, i, :, :]
            band_min = torch.min(band_data).item()
            band_max = torch.max(band_data).item()
            global_mins[band] = min(global_mins[band], band_min)
            global_maxs[band] = max(global_maxs[band], band_max)
            
            # Debugging code------------------------------------------------------
            # Debug: if min value is below -10000, visualise and save the band image 
            # raw_values = band_data.flatten().cpu().numpy()
            # # if np.any((raw_values > -30000)):
            # if np.any((raw_values < -20) & (raw_values > -32000)):
            #     print("Erroneous values found in tile")
            #     band_data = band_data.unsqueeze(1) # Add channel dimension
            #     from visualisations.visualise_bands import visualise_bands
            #     visualise_bands(None, band_data, plots_dir, milestone=num, n_images_to_log=1, satellite_type=data_cfg['satellite_type'])
                
            #     ## Plot histogram of the band
            #     plt.figure(figsize=(10, 6))
            #     plt.hist(raw_values, bins=100, alpha=0.5, color='blue', label=f'Band {band}')
            #     plt.xlabel('Pixel Value')
            #     plt.ylabel('Frequency')
            #     plt.ylim(0, 20)
            #     plt.title(f'Histogram of Band {band}')
            #     plt.savefig(os.path.join(plots_dir, f"visualisation-{num}-{band}.png"), dpi=300, bbox_inches='tight')
            #     plt.close()
            # Debugging code------------------------------------------------------
            
            band_values[band].append(band_data.flatten().cpu().numpy())
    
    # Handle case where same band appears multiple times (e.g. ['dem', 'dem', 'dem'])
    unique_bands = set(data_cfg['tif_bands'])
    for band in unique_bands:
        band_values[band] = np.concatenate(band_values[band])
        
    print("Plotting raw band histograms...")
    plot_band_histograms(band_values, data_cfg['tif_bands'], 
                         filename=os.path.join(plots_dir, "raw_band_histograms.png"))
    print_min_max_values(global_mins, global_maxs, data_cfg['tif_bands'], name="RAW BAND VALUES")
    
    # STEP 2: REMOVE OUTLIERS =============================================================
    
    nodata_threshold = args.nodata_threshold
    clean_mins = {band: float('inf') for band in data_cfg['tif_bands']}
    clean_maxs = {band: float('-inf') for band in data_cfg['tif_bands']}
    clean_band_values = {band: [] for band in data_cfg['tif_bands']}
    
    for band in data_cfg['tif_bands']:
        min_positive_value = band_values[band][band_values[band] > 0].min()
        num_of_zeros = np.sum(band_values[band] == 0)
        print(f"Minimum positive value for band {band}: {min_positive_value}")
        print(f"Number of zeros for band {band}: {num_of_zeros}")
        # Replace nodata values with the minimum positive value
        valid_mask = band_values[band] > nodata_threshold # mask for valid data: all values above the threshold
        valid_data = np.where(valid_mask, band_values[band], min_positive_value)
        
        # Replace negative or zero values with the minimum positive value
        if not args.with_negatives:
            valid_data = np.where(valid_data <= 0, min_positive_value, valid_data)
        
        if len(valid_data) > 0:
            if args.use_percentiles: # Optional percentile clipping
                p01 = np.percentile(valid_data, 1)
                p99 = np.percentile(valid_data, 99)
                valid_data = np.clip(valid_data, p01, p99)
            
            clean_mins[band] = valid_data.min()
            clean_maxs[band] = valid_data.max()
            clean_band_values[band] = valid_data
    
    print("Plotting cleaned band histograms...")
    if args.use_percentiles and args.with_negatives:
        plot_band_histograms(clean_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "cleaned_band_histograms_with_percentile_clipping_and_negatives.png"))
    elif args.use_percentiles:
        plot_band_histograms(clean_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "cleaned_band_histograms_with_percentile_clipping.png"))
    elif args.with_negatives:
        plot_band_histograms(clean_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "cleaned_band_histograms_with_negatives.png"))
    else:
        plot_band_histograms(clean_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "cleaned_band_histograms.png"))
    print_min_max_values(clean_mins, clean_maxs, data_cfg['tif_bands'], name="CLEANED BAND VALUES")
    
    # STEP 2.5: OFFSET TO POSITIVE VALUES (e.g. DEM) =============================================================
    # if args.with_negatives:
    #     # Compute the minimum value across all bands
    #     min_value = min(clean_mins.values())
    #     print("="*50)
    #     print("OFFSETTING CLEANED DATA TO POSITIVE RANGE")
    #     print("="*50)
    #     print(f"Minimum value across all bands: {min_value:.6f}")
    #     print(f"Offset used: {abs(min_value):.6f}")
    #     print("="*50)
    #     # Offset the band values so that the minimum is 0
    #     for band in data_cfg['tif_bands']:
    #         clean_band_values[band] = clean_band_values[band] + abs(min_value)
    #     # Update clean_mins and clean_maxs
    #     for band in data_cfg['tif_bands']:
    #         clean_mins[band] = clean_mins[band] + abs(min_value)
    #         clean_maxs[band] = clean_maxs[band] + abs(min_value)
    #     print_min_max_values(clean_mins, clean_maxs, data_cfg['tif_bands'], name="CLEANED BAND VALUES (OFFSET TO POSITIVE VALUES)")
    #     print("="*50)
    #     print("\n")
    
    
    # STEP 3: APPLY DB SCALING =============================================================
    
    # db_band_values = {band: 10 * np.log10(clean_band_values[band]) for band in data_cfg['tif_bands']}
    db_band_values = {band: np.log1p(clean_band_values[band]) for band in data_cfg['tif_bands']}
    # db_band_values = {band: (np.log(np.maximum(clean_band_values[band], 1e-4)) - np.log(1e-4)) / np.log(1e-4)
    #                         for band in data_cfg['tif_bands']}
    # db_band_values = {band: (np.log1p(np.maximum(clean_band_values[band], 1e-4)) - np.log1p(1e-4)) / np.log1p(1e-4)
    #                         for band in data_cfg['tif_bands']}
    db_mins = {band: db_band_values[band].min() for band in data_cfg['tif_bands']}
    db_maxs = {band: db_band_values[band].max() for band in data_cfg['tif_bands']}
    print("Plotting log-scaled band histograms...")
    if args.use_percentiles and args.with_negatives:
        plot_band_histograms(db_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "log_scaled_band_histograms_with_percentile_clipping_and_negatives.png"))
    elif args.use_percentiles:
        plot_band_histograms(db_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "log_scaled_band_histograms_with_percentile_clipping.png"))
    elif args.with_negatives:
        plot_band_histograms(db_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "log_scaled_band_histograms_with_negatives.png"))
    else:
        plot_band_histograms(db_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "log_scaled_band_histograms.png"))
    print_min_max_values(db_mins, db_maxs, data_cfg['tif_bands'], name="LOG-SCALED BAND VALUES")
    
    # STEP 4: APPLY NORMALIZATION =============================================================
    
    normalized_band_values = {band: (db_band_values[band] - db_mins[band]) / (db_maxs[band] - db_mins[band]) for band in data_cfg['tif_bands']}
    normalized_mins = {band: normalized_band_values[band].min() for band in data_cfg['tif_bands']}
    normalized_maxs = {band: normalized_band_values[band].max() for band in data_cfg['tif_bands']}
    
    print("Plotting normalized band histograms...")
    if args.use_percentiles and args.with_negatives:
        plot_band_histograms(normalized_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "normalized_band_histograms_with_percentile_clipping_and_negatives.png"))
    elif args.use_percentiles:
        plot_band_histograms(normalized_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "normalized_band_histograms_with_percentile_clipping.png"))
    elif args.with_negatives:
        plot_band_histograms(normalized_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "normalized_band_histograms_with_negatives.png"))
    else:
        plot_band_histograms(normalized_band_values, data_cfg['tif_bands'], 
                             filename=os.path.join(plots_dir, "normalized_band_histograms.png"))
    print_min_max_values(normalized_mins, normalized_maxs, data_cfg['tif_bands'], name="NORMALIZED BAND VALUES")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
