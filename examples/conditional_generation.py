from pathlib import Path
import rasterio as rio
from rasterio.windows import Window
import torch
from importlib.machinery import SourceFileLoader
import numpy as np
import os
from libs.copgen import CopgenModel
from visualisations.visualise_bands import visualise_bands
from visualisations.merge_visualisations import merge_visualisations
from ddm.pre_post_process_data import pre_process_data, post_process_data, encode_date, encode_lat_lon
from datasets import _LatLonMemmapCache
from datetime import datetime
from benchmark.io.tiff import write_tif
import csv

def load_tif_band(path, size, window=None):
    '''
    Load a TIFF band from a path and optionally apply one-hot encoding.
    Args:
        path: Path to the TIFF file
        size: Size of the window to read
        window: Window to read. Tuple of top-left x, top-left y, width, height. If None, the center of the image is used.
    Returns:
        Band data as a numpy array
    '''
    with rio.open(path) as f:
        if window is None:
            height = f.height
            width = f.width
            start_y = (height - size) // 2
            start_x = (width - size) // 2
            window = Window(start_x, start_y, size, size)
        band_data = f.read(window=window)
        return band_data

def to_tensor(band_data, add_batch_dim=False):
    if not isinstance(band_data, torch.Tensor):
        band_data = torch.from_numpy(band_data).float()
    if add_batch_dim:
        band_data = band_data.unsqueeze(0)
    return band_data


def get_base_from_modality(modality: str) -> str:
    """Extract the base modality name (e.g., 'S2L2A' from 'S2L2A_B02_B03_B04_B08')."""
    for base in ("S2L2A", "S2L1C", "S1RTC", "DEM", "LULC"):
        if base in modality:
            return base
    return modality


def save_generated_tif(
    tensor: torch.Tensor,
    modality: str,
    config,
    outputs_path: Path,
    ref_path: Path = None,
    sample_idx: int = None
):
    """
    Save a generated tensor as TIF file(s).
    
    Args:
        tensor: Generated tensor of shape [C, H, W] or [1, C, H, W]
        modality: Modality name (e.g., 'S2L2A_B02_B03_B04_B08', 'DEM_DEM', 'LULC_LULC')
        config: Model config containing modality band information
        outputs_path: Directory to save outputs
        ref_path: Optional path to reference TIF for geo metadata
        sample_idx: Optional sample index for multi-sample generation
    """
    # Ensure tensor is [C, H, W]
    arr = tensor.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]  # Remove batch dim
    arr = arr.astype(np.float32)
    
    # Create modality output directory
    modality_dir = outputs_path / modality
    modality_dir.mkdir(parents=True, exist_ok=True)
    
    # Get band names from config
    bands = list(config.all_modality_configs[modality].bands)
    base = get_base_from_modality(modality)
    
    # Build suffix for multi-sample outputs
    suffix = f"_sample_{sample_idx}" if sample_idx is not None else ""
    
    # Handle scalar modalities (lat_lon, timestamps) - skip TIF saving
    if modality in ("3d_cartesian_lat_lon", "mean_timestamps"):
        return  # These are saved to CSV separately
    
    # Handle LULC (single-band categorical)
    if base == "LULC":
        out_path = modality_dir / f"LULC{suffix}.tif"
        label = arr[0] if arr.shape[0] == 1 else arr
        write_tif(out_path, label[np.newaxis, ...], ref_meta_or_path=ref_path, dtype=np.float32)
        print(f"  Saved: {out_path}")
        return
    
    # Handle DEM (single-band)
    if base == "DEM":
        out_path = modality_dir / f"DEM{suffix}.tif"
        write_tif(out_path, arr, ref_meta_or_path=ref_path, dtype=np.float32)
        print(f"  Saved: {out_path}")
        return
    
    # Handle multi-band modalities (S2L2A, S2L1C, S1RTC)
    EPSILON = 1e-12
    if base == "S1RTC":
        arr = np.clip(arr, EPSILON, None)
    
    for i, band in enumerate(bands):
        out_path = modality_dir / f"{band}{suffix}.tif"
        write_tif(out_path, arr[i:i+1], ref_meta_or_path=ref_path, dtype=np.float32)
        print(f"  Saved: {out_path}")


def save_scalar_outputs(
    samples: dict,
    outputs_path: Path,
    n_samples: int = 1
):
    """
    Save scalar modalities (lat_lon, timestamps) to CSV files.
    
    Args:
        samples: Dictionary of generated samples
        outputs_path: Directory to save outputs
        n_samples: Number of samples generated
    """
    # Save lat_lon to CSV
    if "3d_cartesian_lat_lon" in samples:
        latlon_path = outputs_path / "lat_lon.csv"
        tensor = samples["3d_cartesian_lat_lon"]
        arr = tensor.detach().cpu().numpy()
        
        with open(latlon_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "lat", "lon"])
            for i in range(arr.shape[0]):
                vals = arr[i].reshape(-1)
                lat, lon = float(vals[0]), float(vals[1])
                writer.writerow([f"sample_{i}", lat, lon])
        print(f"  Saved: {latlon_path}")
    
    # Save timestamps to CSV  
    if "mean_timestamps" in samples:
        time_path = outputs_path / "timestamp.csv"
        tensor = samples["mean_timestamps"]
        arr = tensor.detach().cpu().numpy()
        
        with open(time_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_id", "day", "month", "year"])
            for i in range(arr.shape[0]):
                vals = arr[i].reshape(-1)
                day, month, year = int(vals[0]), int(vals[1]), int(vals[2])
                writer.writerow([f"sample_{i}", day, month, year])
        print(f"  Saved: {time_path}")


def prepare_bands(config, path, satellite_type, bands=None, window=None):
    size = config.all_modality_configs[satellite_type].img_input_resolution[0]
    tensors = []
    use_bands = bands if bands is not None else config.all_modality_configs[satellite_type].bands
    for band in use_bands:
        tif = load_tif_band(path / f"{band}.tif", size=size, window=window)
        arr = pre_process_data(tif, satellite_type, config.dataset)
        arr = to_tensor(arr)
        tensors.append(arr)
    return to_tensor(torch.cat(tensors, dim=0), add_batch_dim=True)


def load_ground_truth(config, modality: str, gt_paths: dict, window=None) -> torch.Tensor:
    """
    Load ground truth data for a given modality from the dataset.
    
    Args:
        config: Model config containing modality band information
        modality: Modality name (e.g., 'S2L2A_B02_B03_B04_B08', 'DEM_DEM', 'LULC_LULC')
        gt_paths: Dictionary mapping modality names to their data paths
        window: Optional window for reading TIF files
        
    Returns:
        Ground truth tensor of shape [1, C, H, W] (post-processed)
    """
    if modality not in gt_paths:
        return None
    
    path = gt_paths[modality]
    base = get_base_from_modality(modality)
    
    # Skip scalar modalities for now (lat_lon, timestamps)
    if modality in ("3d_cartesian_lat_lon", "mean_timestamps"):
        return None
    
    # Load and prepare bands (same as conditions but for GT)
    size = config.all_modality_configs[modality].img_input_resolution[0]
    bands = list(config.all_modality_configs[modality].bands)
    
    tensors = []
    for band in bands:
        tif_path = path / f"{band}.tif"
        if not tif_path.exists():
            return None
        tif = load_tif_band(tif_path, size=size, window=window)
        arr = pre_process_data(tif, modality, config.dataset)
        arr = to_tensor(arr)
        tensors.append(arr)
    
    # Combine bands and post-process
    raw_tensor = to_tensor(torch.cat(tensors, dim=0), add_batch_dim=True)
    gt_post = post_process_data(processed_data=raw_tensor, satellite_type=modality, data_config=config.dataset)
    return gt_post

def print_run_info(config, conditions, modalities_to_generate, n_samples, batch_size, model_path, config_path):
    print()
    print("=" * 80)
    print("COPGEN GENERATION SUMMARY")
    print("=" * 80)
    print("All Available Modalities:")
    for m in config.all_modality_configs.keys():
        print(f"  - {m}")
    print("Condition Modalities:")
    for m in conditions.keys():
        print(f"  - {m}")
    print("Generate Modalities:")
    for m in modalities_to_generate:
        print(f"  - {m}")
    print(f"Number of Samples: {n_samples}")
    print(f"Batch Size: {batch_size}")
    print(f"Model Path: {model_path}")
    print(f"Config Path: {config_path}")
    print("=" * 80)
    print()

def main():
    
    # Generic paths
    data_path = Path('data/majorTOM/edinburgh')
    model_path = Path('models/copgen/cop_gen_base/500000_nnet_ema.pth')
    config_path = Path('configs/copgen/discrete/cop_gen_base.py')
    
    # Output paths (generated tifs and visualisations)
    generations_path = Path('data/majorTOM/edinburgh/outputs/conditional_example')
    visualisations_path = generations_path / "visualisations"
    outputs_path = generations_path / "generations"
    generations_path.mkdir(parents=True, exist_ok=True)
    visualisations_path.mkdir(parents=True, exist_ok=True)
    outputs_path.mkdir(parents=True, exist_ok=True)
    
    # Example specific paths for tile 618U_30L
    dem_path = data_path / "Core-DEM/618U/618U_30L/id"
    lulc_path = data_path / "Core-LULC/618U/30L"
    s1rtc_path = data_path / f"Core-S1RTC/618U/618U_30L/S1A_IW_GRDH_1SDV_20230418T063857_20230418T063922_048147_05C9E6_rtc"
    s2l1c_path = data_path / f"Core-S2L1C/618U/618U_30L/S2A_MSIL1C_20230420T114351_N0509_R123_T30UUG_20230420T151818"
    s2l2a_path = data_path / f"Core-S2L2A/618U/618U_30L/S2A_MSIL2A_20230420T114351_N0509_R123_T30UUG_20230420T165159"
    
    batch_size = 1
    n_samples = 4 # Generate 4 samples
    
    # Load config
    config = SourceFileLoader("config", str(config_path)).load_module().get_config()
    
    # Load data    
    lat_lon = to_tensor(encode_lat_lon(lat=61.8, lon=8.0, output_format='3d_cartesian_lat_lon', add_spatial_dims=True))
    date = to_tensor(encode_date(datetime(2023, 1, 1), add_spatial_dims=True))
    dem_tensor = prepare_bands(config, path=dem_path, satellite_type="DEM_DEM")
    lulc_tensor = prepare_bands(config, path=lulc_path, bands=['618U_30L_2023'], satellite_type="LULC_LULC")
    s1rtc_tensor = prepare_bands(config, path=s1rtc_path, satellite_type="S1RTC_vh_vv")
    s2l1c_b2348_tensor = prepare_bands(config, path=s2l1c_path, satellite_type="S2L1C_B02_B03_B04_B08")
    s2l1c_b56711128a_tensor = prepare_bands(config, path=s2l1c_path, satellite_type="S2L1C_B05_B06_B07_B11_B12_B8A")
    s2l1c_b1910_tensor = prepare_bands(config, path=s2l1c_path, satellite_type="S2L1C_B01_B09_B10")
    s2l2a_b2348_tensor = prepare_bands(config, path=s2l2a_path, satellite_type="S2L2A_B02_B03_B04_B08")
    s2l2a_b56711128a_tensor = prepare_bands(config, path=s2l2a_path, satellite_type="S2L2A_B05_B06_B07_B11_B12_B8A")
    s2l2a_b19_tensor = prepare_bands(config, path=s2l2a_path, satellite_type="S2L2A_B01_B09")
    s2l1c_cloud_mask_tensor = prepare_bands(config, path=s2l1c_path, satellite_type="S2L1C_cloud_mask")
    
    # Define conditions
    conditions = {
        # "3d_cartesian_lat_lon": lat_lon,
        # "mean_timestamps": date,            
        "DEM_DEM": dem_tensor,
        # "LULC_LULC": lulc_tensor,
        "S1RTC_vh_vv": s1rtc_tensor,
        # "S2L1C_B02_B03_B04_B08": s2l1c_b2348_tensor,
        # "S2L1C_B05_B06_B07_B11_B12_B8A": s2l1c_b56711128a_tensor,
        # "S2L1C_B01_B09_B10": s2l1c_b1910_tensor,
        # "S2L2A_B02_B03_B04_B08": s2l2a_b2348_tensor,
        # "S2L2A_B05_B06_B07_B11_B12_B8A": s2l2a_b56711128a_tensor,
        # "S2L2A_B01_B09": s2l2a_b19_tensor,
        # "S2L1C_cloud_mask": s2l1c_cloud_mask_tensor,
    }

    # Define generations
    modalities_to_generate = [
        "3d_cartesian_lat_lon",
        "mean_timestamps",
        # "DEM_DEM",
        "LULC_LULC",
        # "S1RTC_vh_vv",
        "S2L1C_B02_B03_B04_B08",
        "S2L1C_B05_B06_B07_B11_B12_B8A",
        "S2L1C_B01_B09_B10",
        "S2L2A_B02_B03_B04_B08",
        "S2L2A_B05_B06_B07_B11_B12_B8A",
        "S2L2A_B01_B09",
        "S2L1C_cloud_mask",
    ]
    
    # Print run information
    print_run_info(config, conditions, modalities_to_generate, n_samples, batch_size, model_path, config_path)
    
    # Load model
    model = CopgenModel(model_path=model_path, config_path=config_path, seed=1234)
    samples = model.generate(modalities=modalities_to_generate,
                             conditions=conditions,
                             n_samples=n_samples,
                             batch_size=batch_size,
                             return_latents=False)

    # Post-process and visualise
    all_samples = {**conditions, **samples}
    for modality, sample in all_samples.items():
        all_samples[modality] = post_process_data(processed_data=sample, satellite_type=modality, data_config=config.dataset)
    
    # Calculate max resolution for consistent scaling across all modalities
    max_img_input_resolution = max(
        config.all_modality_configs[modality].img_input_resolution[1] 
        for modality in all_samples.keys()
    )
    
    # Define ground truth paths for generated modalities (to show green frame comparisons)
    gt_paths = {
        "DEM_DEM": dem_path,
        "LULC_LULC": lulc_path,
        "S1RTC_vh_vv": s1rtc_path,
        "S2L1C_B02_B03_B04_B08": s2l1c_path,
        "S2L1C_B05_B06_B07_B11_B12_B8A": s2l1c_path,
        "S2L1C_B01_B09_B10": s2l1c_path,
        "S2L2A_B02_B03_B04_B08": s2l2a_path,
        "S2L2A_B05_B06_B07_B11_B12_B8A": s2l2a_path,
        "S2L2A_B01_B09": s2l2a_path,
        "S2L1C_cloud_mask": s2l1c_path,
    }
    # For LULC, we need to specify the band name differently
    lulc_gt_paths = {"LULC_LULC": (lulc_path, ['618U_30L_2023'])}
    
    # Visualise each modality with appropriate frame colors
    for modality, tensor in all_samples.items():
        save_dir = visualisations_path / modality
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if modality in conditions.keys():
            # Conditions: red frame, duplicate to match n_samples for better layout
            input_frame_color = None
            recon_frame_color = (1.0, 0.3, 0.2)  # softer red
            inputs = None
            # Repeat condition tensor to match n_samples
            if tensor.shape[0] == 1 and n_samples > 1:
                tensor = tensor.repeat(n_samples, 1, 1, 1)
        else:
            # Generated outputs: blue frame with green frame ground truth
            input_frame_color = (0.2, 0.9, 0.3)  # green for GT
            recon_frame_color = (0.3, 0.5, 1.0)  # blue for generated
            
            # Load ground truth for comparison
            if modality in lulc_gt_paths:
                # Special handling for LULC with custom band names
                gt_path, gt_bands = lulc_gt_paths[modality]
                inputs = prepare_bands(config, path=gt_path, bands=gt_bands, satellite_type=modality)
                inputs = post_process_data(processed_data=inputs, satellite_type=modality, data_config=config.dataset)
            elif modality in gt_paths:
                inputs = load_ground_truth(config, modality, gt_paths)
            else:
                inputs = None
            
            # Repeat GT to match n_samples if needed
            if inputs is not None and inputs.shape[0] == 1 and n_samples > 1:
                inputs = inputs.repeat(n_samples, 1, 1, 1)
        
        visualise_bands(
            inputs=inputs,
            reconstructions=tensor,
            save_dir=save_dir,
            milestone=0,
            repeat_n_times=None,
            satellite_type=modality,
            view_histogram=False,
            resize=max_img_input_resolution,
            input_frame_color=input_frame_color,
            recon_frame_color=recon_frame_color
        )
    
    # Merge visualisations
    merge_visualisations(results_path=str(visualisations_path), verbose=False, overwrite=True)
    
    # Save generated bands to TIF files
    print("\nSaving generated outputs to TIF files...")
    
    # Build reference paths for geo metadata (optional but recommended)
    ref_paths = {
        "DEM_DEM": dem_path / "DEM.tif",
        "LULC_LULC": lulc_path / "618U_30L_2023.tif",
        "S1RTC_vh_vv": s1rtc_path / "vv.tif",
        "S2L1C_B02_B03_B04_B08": s2l1c_path / "B02.tif",
        "S2L1C_B05_B06_B07_B11_B12_B8A": s2l1c_path / "B05.tif",
        "S2L1C_B01_B09_B10": s2l1c_path / "B01.tif",
        "S2L2A_B02_B03_B04_B08": s2l2a_path / "B02.tif",
        "S2L2A_B05_B06_B07_B11_B12_B8A": s2l2a_path / "B05.tif",
        "S2L2A_B01_B09": s2l2a_path / "B01.tif",
        "S2L1C_cloud_mask": s2l1c_path / "cloud_mask.tif",
    }
    
    # Save image-based modalities as TIF files
    for modality, tensor in all_samples.items():
        # Skip conditions - only save generated outputs
        if modality in conditions.keys():
            continue
        # Skip scalar modalities (handled separately)
        if modality in ("3d_cartesian_lat_lon", "mean_timestamps"):
            continue
            
        print(f"Saving {modality}...")
        ref_path = ref_paths.get(modality)
        
        # If multiple samples, save each sample separately
        if n_samples > 1:
            for s in range(n_samples):
                save_generated_tif(
                    tensor=tensor[s],
                    modality=modality,
                    config=config,
                    outputs_path=outputs_path,
                    ref_path=ref_path,
                    sample_idx=s
                )
        else:
            save_generated_tif(
                tensor=tensor[0] if tensor.dim() == 4 else tensor,
                modality=modality,
                config=config,
                outputs_path=outputs_path,
                ref_path=ref_path,
                sample_idx=None
            )
    
    # Save scalar modalities (lat_lon, timestamps) to CSV
    save_scalar_outputs(samples, outputs_path, n_samples)
    
    print(f"\nGeneration complete! Outputs saved to: {outputs_path}")


if __name__ == "__main__":
    main()