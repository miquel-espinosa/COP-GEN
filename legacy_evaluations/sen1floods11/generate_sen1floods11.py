import argparse
import pandas as pd
from pangaea.datasets.sen1floods11 import Sen1Floods11
from ddm.pre_post_process_data import pre_process_data, post_process_data, encode_date
from visualisations.visualise_bands import visualise_bands
from visualisations.merge_visualisations import merge_visualisations
from pathlib import Path
from importlib.machinery import SourceFileLoader
from libs.copgen import CopgenModel
import yaml
import torch
import torchvision.transforms.functional as T
from torch.utils.data import DataLoader
import rasterio
import numpy as np
import os
from tqdm import tqdm


class COPGEN_Sen1Floods11(Sen1Floods11):
    def __init__(self, model_config,
                 condition_modalities: list[str],
                 modality_mapping: dict,
                 band_mapping: dict,
                 crop_size: int,
                 resize: bool,
                 *args, **kwargs):
        super().__init__(*args, **kwargs) # Initialise the Sen1Floods11 dataset
        self.model_config = model_config
        self.condition_modalities = condition_modalities # List of modalities to condition on
        self.modality_mapping = modality_mapping
        self.band_mapping = band_mapping
        self.crop_size = crop_size
        self.resize = resize
        
    # Override original _get_date function to return timestamp as datetime object
    def _get_date(self, index):
        file_name = self.s2_image_list[index]
        location = os.path.basename(file_name).split("_")[0]
        if self.metadata[self.metadata["location"] == location].shape[0] != 1:
            date = pd.to_datetime("13-10-1998", dayfirst=True)
        else:
            date = pd.to_datetime(
                self.metadata[self.metadata["location"] == location]["s2_date"].item()
            )
        return date
        
    def __getitem__(self, index):
        item = super().__getitem__(index) # Get the item from the Sen1Floods11 dataset
        ref_paths = {
            "S2L2A": self.s2_image_list[index],
            "S2L1C": self.s2_image_list[index],
            "S1RTC": self.s1_image_list[index],
        }
        conditions = {}
        for condition_modality in self.condition_modalities:
            dataset_key = self.modality_mapping[condition_modality] # e.g. dataset_key = "sar"
            bands = []
            if condition_modality == "mean_timestamps":
                timestamp = encode_date(item["metadata"]["timestamp"], add_spatial_dims=False)
                timestamp_tensor = torch.from_numpy(timestamp).float().unsqueeze(0).unsqueeze(0)
                bands.append(timestamp_tensor)
            else:
                for band in self.model_config.all_modality_configs[condition_modality].bands:
                    band_key = self.band_mapping[condition_modality][band] # e.g. band vv in COPGEN is mapped to "VH" in Sen1Floods11
                    band_index = self.bands[dataset_key].index(band_key) # e.g. get index position of "vv" in "sar" bands list
                    band_tensor = item["image"][dataset_key][band_index]
                    if self.resize:
                        band_tensor = T.resize(band_tensor, self.model_config.all_modality_configs[condition_modality].img_input_resolution)
                    if 'S1RTC' in condition_modality:
                        # SEN1Floods11 S1RTC data is already in db range, convert to backscatter values
                        band_tensor_pre = pre_process_data(band_tensor, condition_modality, self.model_config.dataset, in_db=True)
                    else:
                        band_tensor_pre = pre_process_data(band_tensor, condition_modality, self.model_config.dataset)
                    bands.append(band_tensor_pre)
            conditions[condition_modality] = torch.cat(bands, dim=0)
        return conditions, ref_paths


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Sen1Floods11 samples using CopgenModel")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint")
    parser.add_argument("--model_config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument("--data_config", type=str, required=True,
                        help="Path to the CopGen YAML config (with dataset_yaml_path)")
    parser.add_argument("--condition_modalities", type=str, required=True, nargs="+",
                        help="List of modalities to condition on (e.g. S2L2A_B02_B03_B04_B08,S2L2A_B05_B06_B07_B11_B12_B8A,S1RTC_vh_vv)")
    # parser.add_argument("--generate_modalities", type=str, required=True, nargs="+",
    #                     help="Comma-separated list of modalities to generate (e.g. S2L2A_B01_B09,S2L2A_B02_B03_B04_B08,S2L2A_B05_B06_B07_B11_B12_B8A,S2L1C_B01_B09_B10,S2L1C_B02_B03_B04_B08,S2L1C_B05_B06_B07_B11_B12_B8A,S1RTC_vh_vv)")
    parser.add_argument("--root_output_path", type=str, required=True,
                        help="Path to save the outputs")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to use (train|val|test)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of workers")
    parser.add_argument("--visualise_every_n_batches", type=int, default=10,
                        help="Visualise every n batches")
    return parser.parse_args()


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def load_data_config(data_config_path: str) -> dict:
    """
    Loads the copgen mapping YAML and merges it with the pangaea dataset YAML.
    Args:
        data_config_path (str): Path to the data config YAML

    Returns:
        dict: Merged config
    """
    data_cfg = load_yaml(data_config_path) # Load copgen mapping YAML
    dataset_yaml_path = data_cfg.get("dataset_yaml_path")
    if dataset_yaml_path is None:
        raise ValueError("dataset_yaml_path must be specified in the data config YAML")
    dataset_cfg = load_yaml(dataset_yaml_path) # Load pangaea dataset YAML
    merged_cfg = {**dataset_cfg, **data_cfg}
    # Remove Hydra-specific entries and dataset_yaml_path from merged_cfg
    merged_cfg = {k: v for k, v in merged_cfg.items() if k != "_target_" and k != "dataset_yaml_path"}
    return merged_cfg


def print_experiment_summary(args, all_modalities, generate_modalities):
    print()
    print("=" * 80)
    print("COPGEN GENERATION SUMMARY")
    print("=" * 80)
    print("All Available Modalities:")
    for m in all_modalities:
        print(f"  - {m}")
    print("Condition Modalities:")
    for m in args.condition_modalities:
        print(f"  - {m}")
    print("Generate Modalities:")
    for m in generate_modalities:
        print(f"  - {m}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"Number of Workers: {args.num_workers}")
    print(f"Seed: {args.seed}")
    print(f"Split: {args.split}")
    print(f"Model Path: {args.model_path}")
    print(f"Model Config: {args.model_config}")
    print(f"Data Config: {args.data_config}")
    print(f"Root Output Path: {args.root_output_path}")
    print("=" * 80)
    print()
    
def setup_output_root(output_root: str, data_config: dict):
    """Set up output root with directory structure and needed files."""
    import shutil
    
    # Create output directories
    output_data_dir = f'{output_root}/v1.1/data'
    visualisations_dir = f'{output_root}/visualisations'
    os.makedirs(output_data_dir, exist_ok=True)
    os.makedirs(visualisations_dir, exist_ok=True)
    
    # Copy metadata
    metadata_file = os.path.join(data_config['root_path'], 'v1.1', 'Sen1Floods11_Metadata.geojson')
    shutil.copy(metadata_file, os.path.join(output_root, 'v1.1'))
    
    # Copy splits directory
    splits_source_dir = os.path.join(data_config['root_path'], 'v1.1', 'splits')
    splits_output_dir = os.path.join(output_root, 'v1.1', 'splits')
    shutil.copytree(splits_source_dir, splits_output_dir, dirs_exist_ok=True)

    return output_data_dir, visualisations_dir

def get_output_path(output_root: str, input_path: str, modality: str) -> Path:
    output_path = input_path.replace('./data/sen1floods11_v1.1/v1.1/data', output_root)
    output_path = Path(output_path.replace('<modality>', modality))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def resize_to_target(tensor: torch.Tensor, target_size: int = 192) -> torch.Tensor:
    """Resize tensor to target spatial size."""
    if tensor.shape[-1] == target_size and tensor.shape[-2] == target_size:
        return tensor
    return T.resize(tensor, (target_size, target_size))


def save_tensor_as_sen1floods11_tif(output_root: str, tensor: torch.Tensor, ref_paths: dict, modality: str):
    """Save tensor to GeoTIFF using reference raster metadata."""

    tensor_np = tensor.detach().cpu().numpy().astype(np.float32)
    if tensor_np.ndim == 2:
        tensor_np = np.expand_dims(tensor_np, axis=0)
        
    out_path = get_output_path(output_root, ref_paths["S2L2A"].replace("S2Hand", "<modality>"), modality)
    
    ref_path = ref_paths[modality] if modality in ref_paths else ref_paths["S2L2A"]
    with rasterio.open(ref_path) as ref:
        meta = ref.meta.copy()
    
    # Update metadata to match the tensor
    meta.update({
        "driver": "GTiff",
        "height": tensor_np.shape[1],
        "width": tensor_np.shape[2],
        "count": tensor_np.shape[0],
        "dtype": "float32"
    })
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(tensor_np)
    # print(f"✅ Saved generated output to: {out_path}")


def main():
    args = parse_args()
    
    # Load configs
    model_config = SourceFileLoader("config", str(args.model_config)).load_module().get_config()
    data_config = load_data_config(args.data_config)

    # Set up output root
    output_data_dir, visualisations_dir = setup_output_root(args.root_output_path, data_config)
    
    # Print experiment summary
    all_modalities = list(model_config.all_modality_configs.keys())
    generate_modalities = [m for m in all_modalities if m not in args.condition_modalities]
    print_experiment_summary(args, all_modalities, generate_modalities)
    
    # Checks
    s2l2a_modalities_used = [m for m in args.condition_modalities if "S2L2A" in m]
    s2l1c_modalities_used = [m for m in args.condition_modalities if "S2L1C" in m]
    s2l2a_modalities_all = [m for m in all_modalities if "S2L2A" in m]
    s2l1c_modalities_all = [m for m in all_modalities if "S2L1C" in m and 'cloud_mask' not in m]
    
    if s2l2a_modalities_used:
        if len(s2l2a_modalities_used) > 0 and len(s2l2a_modalities_used) != len(s2l2a_modalities_all):
            raise ValueError(f"If any S2L2A modality is used as condition, all S2L2A modalities must be included.")
    if s2l1c_modalities_used:
        if len(s2l1c_modalities_used) > 0 and len(s2l1c_modalities_used) != len(s2l1c_modalities_all):
            raise ValueError(f"If any S2L1C modality is used as condition, all S2L1C modalities must be included.")
    
    # Load dataset
    dataset = COPGEN_Sen1Floods11(
        model_config=model_config,
        condition_modalities=args.condition_modalities,
        split=args.split,
        **data_config,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Load model
    model = CopgenModel(model_path=args.model_path, config_path=model_config, seed=args.seed)
    
    # Generate samples
    for idx, batch in tqdm(enumerate(dataloader), desc="Generating samples", total=len(dataloader)):
        conditions, ref_paths = batch
        
        generated = model.generate(
            modalities=generate_modalities,
            conditions=conditions,
            n_samples=args.num_samples,
        )
        
        all_generated = {**conditions, **generated}
        
        # Post process
        for modality, generated_tensor in all_generated.items():
            all_generated[modality] = post_process_data(generated_tensor, modality, model_config.dataset)
                
        # Visualise if enabled
        if args.visualise_every_n_batches and idx % args.visualise_every_n_batches == 0:
            for modality, generated_tensor in all_generated.items():
                visualise_bands(inputs=None,
                                reconstructions=all_generated[modality],
                                save_dir=f'{visualisations_dir}/{modality}',
                                milestone=idx,
                                repeat_n_times=args.num_samples if modality in list(conditions.keys()) else None,
                                satellite_type=modality)
                        
            merge_visualisations(results_path=visualisations_dir, verbose=False, overwrite=True)
            
        # ================================= SEN1Floods11 formatting =================================
        merged = {}
        for satellite_type in ["S2L2A", "S2L1C"]:
            sat_specific_modalities = [m for m in all_generated.keys() if satellite_type in m
                                                                        and m in generate_modalities
                                                                        and 'cloud_mask' not in m]
            if sat_specific_modalities:
                resized_bands = {}
                for modality in sat_specific_modalities:
                    tensor = resize_to_target(all_generated[modality], data_config['crop_size'])
                    band_mapping = data_config['band_mapping'][modality]
                    assert set(list(band_mapping.keys())) == set(model_config.all_modality_configs[modality].bands), "Band mapping keys do not match cop-gen bands"
                    # Order cop-gen bands in band_mapping according to order in model_config.all_modality_configs[modality].bands
                    band_mapping = {band: band_mapping[band] for band in model_config.all_modality_configs[modality].bands}
                    for i, (copgen_band, sen_band) in enumerate(band_mapping.items()):
                        resized_bands[sen_band] = tensor[:, i:i+1, :, :]
                
                # Reorder according to target_band_order
                ordered_bands = [resized_bands[band] for band in data_config['bands']['optical'] if band in resized_bands]
                merged[satellite_type] = torch.cat(ordered_bands, dim=1)
        
        # Add the rest of generated modalities to merged
        for modality in generate_modalities:
            if ('S2L2A' not in modality and 'S2L1C' not in modality) or 'cloud_mask' in modality:
                if modality not in merged:
                    merged[modality] = all_generated[modality]
        # ==========================================================================================
        
        for modality, tensor in merged.items():
            for i in range(tensor.shape[0]):
                ref_paths_batch = {
                    "S2L2A": ref_paths["S2L2A"][i],
                    "S2L1C": ref_paths["S2L1C"][i],
                    "S1RTC": ref_paths["S1RTC"][i],
                }
                save_tensor_as_sen1floods11_tif(output_data_dir, tensor[i], ref_paths_batch, modality)

if __name__ == "__main__":
    main()