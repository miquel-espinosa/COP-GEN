import os
import json
from pathlib import Path
import torch
import rioxarray as rxr
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset
from benchmark.common.tile_matching import find_matching_file as find_tile
from benchmark.io.coords import coords_tensor_for_tile, load_coords_json
from benchmark.visualize.base import LULC_CLASS_NAMES

class CopGenDataset(Dataset):
    def __init__(self, tif_paths, input_modalities, input_dirs, crop_size, device, coords_file=None, input_overrides=None):
        self.tif_paths = tif_paths
        self.input_modalities = input_modalities
        self.input_dirs = input_dirs
        self.crop_size = crop_size
        self.crop = T.CenterCrop(crop_size)
        self.device = device
        self.input_overrides = input_overrides or {}
        
        self.coords_data = None
        if "coords" in input_modalities and coords_file:
            self.coords_data = load_coords_json(Path(coords_file))

    def find_matching_file(self, modality, base_tile_name):
        match = find_tile(self.input_dirs[modality], base_tile_name)
        if match is None:
            raise FileNotFoundError(f"No matching file found for {base_tile_name} in {modality}")
        return match
    
    def get_coord_tensor(self, tile_name):
        """Get coordinate tensor for CoordsTokenizer"""
        if not self.coords_data:
            raise ValueError("Coordinates data not loaded")
        return coords_tensor_for_tile(tile_name, self.coords_data, self.device)

    def _load_and_process(self, modality, path):
        arr = rxr.open_rasterio(path).squeeze().values.astype(np.float32)

        if modality == "S1RTC":
            epsilon = 1e-8
            arr = 10.0 * np.log10(np.clip(arr, epsilon, None))
            arr = np.clip(arr, -50, 10)
        elif modality == "LULC" and arr.ndim == 2:
            unique_classes = np.unique(arr)
            if len(unique_classes) > 10 or np.max(unique_classes) > 9:
                raise ValueError(f"LULC data has invalid classes: {unique_classes}. Expected classes 0-9 only.")
            arr = arr[None, ...]
        elif modality == "DEM" and arr.ndim == 2:
            arr = arr[None, ...]
        return arr

    def preprocess(self, modality, tif_path):
        tile_name = Path(tif_path).stem
        actual_file = self.find_matching_file(modality, tile_name)
        return self._load_and_process(modality, actual_file)

    def __getitem__(self, idx):
        tif_path = str(self.tif_paths[idx])
        input_dict = {}
        
        try:
            for modality in self.input_modalities:
                if modality in self.input_overrides:
                    # Handle override
                    override_val = self.input_overrides[modality]
                    if modality == "coords":
                        # Expect "lat,lon"
                        try:
                            lat_str, lon_str = [s.strip() for s in override_val.split(",")]
                            lat, lon = float(lat_str), float(lon_str)
                            # Shape [1, 2] with [lon, lat]
                            input_dict[modality] = torch.tensor([[lon, lat]], dtype=torch.float32, device=self.device)
                        except Exception:
                            raise ValueError(f"Invalid coords override '{override_val}'. Expected 'lat,lon'.")
                    
                    elif modality == "LULC" and not str(override_val).lower().endswith(('.tif', '.tiff')) and not Path(override_val).exists():
                         # Check if it is a class name
                         target_name = str(override_val).strip().lower()
                         name_to_id = {v.lower(): k for k, v in LULC_CLASS_NAMES.items()}
                         
                         if target_name in name_to_id:
                             class_id = name_to_id[target_name]
                             # Create full tensor of crop_size
                             h, w = self.crop_size if isinstance(self.crop_size, (tuple, list)) else (self.crop_size, self.crop_size)
                             # Create tensor [1, H, W] filled with class_id
                             tensor = torch.full((1, h, w), float(class_id), dtype=torch.float32)
                             input_dict[modality] = tensor.to(self.device)
                         else:
                             # If not a known class and not a path, raise error
                             raise ValueError(f"Override '{override_val}' for LULC is not a valid path and not a known class. Available classes: {', '.join(name_to_id.keys())}")

                    else:
                        # Image modality - treat override_val as path
                        if not Path(override_val).exists():
                            raise FileNotFoundError(f"Override path not found for {modality}: {override_val}")
                        
                        arr = self._load_and_process(modality, Path(override_val))
                        tensor = torch.tensor(arr).float().unsqueeze(0)
                        tensor = self.crop(tensor)
                        input_dict[modality] = tensor.to(self.device)
                        
                elif modality == "coords":
                    # Convert coordinate string to tensor for CoordsTokenizer
                    tile_name = Path(tif_path).stem
                    coord_tensor = self.get_coord_tensor(tile_name)
                    input_dict[modality] = coord_tensor.to(self.device)
                else:
                    # Regular tensor processing
                    arr = self.preprocess(modality, tif_path)
                    tensor = torch.tensor(arr).float().unsqueeze(0)
                    tensor = self.crop(tensor)
                    input_dict[modality] = tensor.to(self.device)
                    
            return tif_path, input_dict
        except (FileNotFoundError, KeyError) as e:
            print(f"Skipping {tif_path}: {e}")
            return None 

    def __len__(self):
        return len(self.tif_paths)