import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import seaborn as sns
from scripts.extract_majortom_feature import process_satellite

def analyze_input_distribution(dataloader, sat_type, flips=False):
    """Analyze distribution of input images from dataloader"""
    all_data = []
    
    print(f"Analyzing input distribution for {sat_type}...")
    for data in tqdm(dataloader):
        img = data['thumbnail']  # Get RGB image directly
        if flips:
            img = torch.cat([img, img.flip(dims=[-1])], dim=0)
        all_data.append(img.cpu().numpy())
    
    all_data = np.concatenate(all_data, axis=0)
    
    print("all_data.shape", all_data.shape)
    
    # Calculate statistics
    mean_per_channel = np.mean(all_data, axis=(0, 2, 3))
    std_per_channel = np.std(all_data, axis=(0, 2, 3))
    
    return all_data, mean_per_channel, std_per_channel

def plot_distributions(input_data, encoded_data, sat_type, output_dir, kde=True):
    """Plot histograms with KDE for input and encoded distributions"""
    plt.figure(figsize=(15, 10))
    
    # Plot input distributions
    channel_names = ['R', 'G', 'B'] if input_data.shape[1] == 3 else ['Greyscale']
    for i in range(len(channel_names)):
        plt.subplot(2, 1, 1)
        sns.histplot(input_data[:, i, :, :].flatten(), 
                    label=f'Channel {channel_names[i]}',
                    stat='density',
                    alpha=0.3,
                    kde=kde)
    
    plt.title(f'{sat_type} Input Distribution')
    plt.legend()
    
    # Plot encoded distribution
    plt.subplot(2, 1, 2)
    sns.histplot(encoded_data.flatten(), 
                label='Encoded Features',
                stat='density',
                alpha=0.3,
                kde=kde)
    plt.title(f'{sat_type} Encoded Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{sat_type}_distributions.png'), dpi=300)
    plt.close()

def plot_all_distributions(results_dict, output_dir, kde=True):
    """Plot all input distributions in one figure and all encoded distributions in another"""
    
    # Create figure for input distributions
    n_types = len(results_dict)
    plt.figure(figsize=(15, 5*n_types))
    
    # First find global min/max for input data to set consistent x-axis
    input_min = float('inf')
    input_max = float('-inf')
    for sat_data in results_dict.values():
        input_min = min(input_min, np.min(sat_data['input_data']))
        input_max = max(input_max, np.max(sat_data['input_data']))
    
    # Plot input distributions
    for idx, (sat_type, sat_data) in enumerate(results_dict.items(), 1):
        plt.subplot(n_types, 1, idx)
        input_data = sat_data['input_data']
        channel_names = ['R', 'G', 'B'] if input_data.shape[1] == 3 else ['Greyscale']
        
        for i in range(len(channel_names)):
            sns.histplot(input_data[:, i, :, :].flatten(),
                        label=f'Channel {channel_names[i]}',
                        stat='density',
                        alpha=0.3,
                        kde=kde)
        
        plt.title(f'{sat_type} Input Distribution')
        plt.xlim(input_min, input_max)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_input_distributions.png'), dpi=300)
    plt.close()
    
    # Create figure for encoded distributions
    plt.figure(figsize=(15, 5*n_types))
    
    # Find global min/max for encoded data
    encoded_min = float('inf')
    encoded_max = float('-inf')
    for sat_data in results_dict.values():
        encoded_min = min(encoded_min, np.min(sat_data['encoded_data']))
        encoded_max = max(encoded_max, np.max(sat_data['encoded_data']))
    
    # Plot encoded distributions
    for idx, (sat_type, sat_data) in enumerate(results_dict.items(), 1):
        plt.subplot(n_types, 1, idx)
        sns.histplot(sat_data['encoded_data'].flatten(),
                    label='Encoded Features',
                    stat='density',
                    alpha=0.3,
                    kde=kde)
        
        plt.title(f'{sat_type} Encoded Distribution')
        plt.xlim(encoded_min, encoded_max)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_encoded_distributions.png'), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze data distributions')
    parser.add_argument('--subset_path', required=True, help='Path to the subset folder')
    parser.add_argument('--resolution', type=int, default=256, help='Resolution for processing')
    parser.add_argument('--output_dir', required=True, help='Directory to save analysis results')
    parser.add_argument('--flips', action='store_true', help='Include flipped images')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each satellite type
    sat_types = ['S2L2A', 'S2L1C', 'DEM']
    results = {}
    
    # Store all data for plotting
    plot_data = {}
    
    for sat_type in sat_types:
        print(f"\nProcessing {sat_type}...")
        
        # Process satellite data - using only thumbnail band
        model, dataloader, device = process_satellite(
            args.subset_path, 
            sat_type, 
            ['thumbnail'],  # Only use thumbnail for RGB
            args.resolution, 
        )
        
        if model is None or dataloader is None:
            print(f"Skipping {sat_type}: Could not process satellite data")
            continue
        
        # Analyze input distribution
        input_data, input_mean, input_std = analyze_input_distribution(dataloader, sat_type, args.flips)
        
        # Get encoded features
        model.eval()
        with torch.no_grad():
            # Process and encode all images
            encoded_data = []
            print(f"Encoding {sat_type} images...")
            for data in tqdm(dataloader):
                img = data['thumbnail']
                if args.flips:
                    img = torch.cat([img, img.flip(dims=[-1])], dim=0)
                img = img.to(device)
                if sat_type == 'DEM':
                    img = img.repeat(1, 3, 1, 1)
                moments = model(inputs=img, fn="encode_moments")
                encoded_data.append(moments.cpu().numpy())
        
        encoded_data = np.concatenate(encoded_data, axis=0)
        print("encoded_data.shape", encoded_data.shape)
        encoded_mean = np.mean(encoded_data)
        encoded_std = np.std(encoded_data)
        
        # Store data for plotting
        plot_data[sat_type] = {
            'input_data': input_data,
            'encoded_data': encoded_data,
            'input_mean': input_mean,
            'input_std': input_std,
            'encoded_mean': encoded_mean,
            'encoded_std': encoded_std
        }
        
        # Plot distributions
        print("Plotting distributions...")
        plot_distributions(input_data, encoded_data, sat_type, args.output_dir, kde=False)
        
        # Print statistics
        print(f"\n{sat_type} Statistics:")
        print(f"Input Mean per RGB channel: {input_mean}")
        print(f"Input Std per RGB channel: {input_std}")
        print(f"Encoded Mean: {encoded_mean}")
        print(f"Encoded Std: {encoded_std}")
    
    # Plot all distributions
    print("\nPlotting all distributions...")
    plot_all_distributions(plot_data, args.output_dir, kde=False)
    
    # Save results to file
    with open(os.path.join(args.output_dir, 'distribution_statistics.txt'), 'w') as f:
        for sat_type, stats in results.items():
            f.write(f"\n{sat_type} Statistics:\n")
            f.write(f"Input Mean per RGB channel: {stats['input_mean']}\n")
            f.write(f"Input Std per RGB channel: {stats['input_std']}\n")
            f.write(f"Encoded Mean: {stats['encoded_mean']}\n")
            f.write(f"Encoded Std: {stats['encoded_std']}\n")

if __name__ == "__main__":
    main()