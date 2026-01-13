import torch
import torchvision.transforms as T
import torch.utils.data as data
import torch.nn as nn
from pathlib import Path
from functools import partial
from ddm.utils import exists, convert_image_to_fn, normalize_to_neg_one_to_one
from PIL import Image, ImageDraw
import torch.nn.functional as F
import math
import torchvision.transforms.functional as F2
import torchvision.datasets as datasets
from typing import Any, Callable, Optional, Tuple
import os
import pickle
import numpy as np
import copy
# import albumentations # We don't use this for now
import random
import rasterio as rio
from rasterio.windows import Window  # Added for efficient patch reads
from ddm.pre_post_process_data import pre_process_data, one_hot_encode, get_value_to_index
from tqdm import tqdm
import signal

# -----------------------------
# Time-out helper for GDAL reads
# -----------------------------
class Timeout:
    """Context manager that raises ``TimeoutError`` if the enclosed
    block takes longer than *seconds* seconds.  It uses SIGALRM, so it
    only works in the main thread of the current process (which is the
    case when the DataLoader is configured with ``num_workers = 0``)."""

    def __init__(self, seconds: int = 60):
        self.seconds = seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)  # cancel any scheduled alarm
        return False  # propagate exceptions

    @staticmethod
    def _handler(signum, frame):
        print("*** Raster read exceeded time limit ***")


class SatelliteDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        img_folder=None,  # Added for backward compatibility with other datasets
        image_size=[256, 256],
        tif_bands=['B04', 'B03', 'B02', 'B08'],  # R,G,B,IR by default
        augment_horizontal_flip=False,
        satellite_type=None,
        center_crop=False,
        normalize_to_neg_one_to_one=True,
        img_list=None,  # Can be provided or auto-generated
        preprocess_bands=True, # By default, we preprocess the bands
        min_db=None,
        max_db=None,
        min_positive=None,
        resize_to=None,
        one_hot_encode_n_classes=None,  # Number of classes for one-hot encoding
        patchify=False,
        patches_per_side=None,
        common_id_set=None, # Set of common IDs to filter the img_list to
        skip_done_entries=False,
        **kwargs
    ):
        super().__init__()
        self.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
        
        # Handle the case where img_folder is passed instead of data_dir (for compatibility)
        if img_folder is not None and data_dir is None:
            self.data_dir = Path(img_folder) if isinstance(img_folder, str) else img_folder
        
        self.image_size = image_size if isinstance(image_size, list) else [image_size, image_size]
        self.size = self.image_size[0]  # For simplicity use first dimension
        self.tif_bands = tif_bands if not isinstance(tif_bands, str) else [tif_bands]
        self.random_flip = augment_horizontal_flip
        self.center_crop = center_crop
        self.normalize_to_neg_one_to_one = normalize_to_neg_one_to_one
        self.satellite_type = satellite_type
        self.preprocess_bands = preprocess_bands
        self.min_db = min_db
        self.max_db = max_db
        self.min_positive = min_positive
        self.resize_to = resize_to
        self.one_hot_encode_n_classes = one_hot_encode_n_classes  # Store number of classes for one-hot encoding
        self.patchify = patchify
        self.patches_per_side = patches_per_side
        
        if self.patchify and self.patches_per_side is None:
            raise ValueError("patches_per_side must be set when patchify is True")
        
        # For LULC and cloud mask, we need to map the original class values to indices
        if self.satellite_type == 'LULC':
           self.value_to_index = get_value_to_index(self.satellite_type)
        if 'cloud_mask' in self.tif_bands:
            self.value_to_index = get_value_to_index('cloud_mask')

        print(f"Initializing SatelliteDataset with type: {self.satellite_type}, bands: {self.tif_bands}")
        
        # If img_list is not provided, we need to discover available products
        if img_list is None:
            # Cache file name based only on satellite type since tuples are (grid_cell, product_id)
            cache_file = self.data_dir / f".cache_{self.satellite_type}.pkl"
            
            # Try to load from cache first
            if cache_file.exists():
                try:
                    print(f"Loading cached products from {cache_file}")
                    with open(cache_file, 'rb') as f:
                        self.img_list = pickle.load(f)
                    print(f"Loaded {len(self.img_list)} products from cache")
                except:
                    print("Failed to load cache, discovering products...")
                    self.img_list = self._discover_products()
                    # Save to cache
                    with open(cache_file, 'wb') as f:
                        pickle.dump(self.img_list, f)
                    print(f"Saved {len(self.img_list)} products to cache")
            else:
                print("No cache found, discovering products...")
                self.img_list = self._discover_products()
                # Save to cache
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.img_list, f)
                print(f"Saved {len(self.img_list)} products to cache")
        else:
            self.img_list = img_list
            
        print(f"Found {len(self.img_list)} products")
        
        # Given the img_list, we need to filter it to only keep the common IDs
        if common_id_set is not None:
            self.img_list = [item for item in self.img_list if item[0] in common_id_set]
            print(f"Filtered to {len(self.img_list)} products, after LULC alignment")
        
        if self.satellite_type == 'LULC' or self.satellite_type == 'GOOGLE_EMBEDS':
            # For LULC, we can have duplicates in the grid_cell column due to different years
            # We will discard the file with the lowest year E.g. (8D_1933R_2020.tif, 8D_1933R_2021.tif) -> keep 8D_1933R_2021.tif
            filtered_list = {}
            for grid_cell, filename in self.img_list:
                # Extract year from filename (format: {row}_{grid_cell}_{year}.tif)
                year = int(filename.split('_')[-1].split('.')[0])
                
                if grid_cell not in filtered_list or year > int(filtered_list[grid_cell][1].split('_')[-1].split('.')[0]):
                    filtered_list[grid_cell] = (grid_cell, filename)
            
            self.img_list = list(filtered_list.values())
        
        if self.satellite_type == 'S2L1C' and self.tif_bands == ['cloud_mask'] and False:
            # For cloud mask, we should only keep 5% of the cloudless images
            raise NotImplementedError("Cloud mask filtering not implemented yet")
            
        # Check that there are no duplicates in the img_list in the grid_cell column
        grid_cells = [item[0] for item in self.img_list]
        if len(grid_cells) != len(set(grid_cells)):
            raise ValueError("There are duplicates in the grid_cell column of the img_list")
        else:
            print("No duplicates in the grid_cell column of the img_list")
        self.grid_cells = grid_cells
        
        # Remove _patch_<patch_idx> from the skip_done_entries set
        if skip_done_entries:
            img_id_skip_done_entries = [item.split('_patch_')[0] for item in skip_done_entries]
            print(f"We will skip {len(img_id_skip_done_entries)} grid cells.")
            img_id_skip_done_entries = set(img_id_skip_done_entries)
            print(f"First item of img_id skip done entries: {list(img_id_skip_done_entries)[0]}")
            print(f"First item of img_list: {self.img_list[0]}")
            self.img_list = [item for item in self.img_list if item[0] not in img_id_skip_done_entries]
            print(f"Filtered to {len(self.img_list)} products, after skipping done entries")
        
        # Run an entire epoch to test the dataloading there is no data corruption
        # errors = False
        # for j in tqdm(range(len(self.img_list)), desc="Testing dataloading"):
        #     try:
        #         self.__getitem__(j)
        #     except Exception as e:
        #         print(f"Error in dataloading: {e}")
        #         errors = True
        # if errors:
        #     raise ValueError("There are errors in the dataloading")
        # else:
        #     print("No errors in the dataloading")
        
    def _discover_products(self):
        """
        Auto-discover products in the data directory following MajorTOM structure
        """
        products = []
        
        # If the satellite type is LULC, the folder structure is like Core-LULC/{row}/{grid_cell}/{row}_{grid_cell}_{year}.tif
        # For example, Core-LULC/8D/1933R/8D_1933R_2020.tif
        if self.satellite_type == 'LULC' or self.satellite_type == 'GOOGLE_EMBEDS':
            # Get all rows (e.g. 8D)
            rows = [d for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            for row in rows:
                # Get all grid cells (e.g. 1933R) 
                grid_cells = [d for d in row.iterdir() if d.is_dir() and not d.name.startswith('.')]
                for grid_cell in grid_cells:
                    # Find all LULC tif files in this grid cell
                    tif_files = list(grid_cell.glob(f"{row.name}_{grid_cell.name}_*.tif"))
                    for tif_file in tif_files:
                        # Use row_gridcell as identifier and filename as product
                        products.append((f"{row.name}_{grid_cell.name}", tif_file.stem))
            return products
        
        # Check if we have a flat structure or MajorTOM structure
        # MajorTOM: row/grid_cell/product_id
        # Flat: product_id or grid_cell_product_id
        
        # First try to find row directories
        try:
            rows = [d for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
            
            if len(rows) > 0:
                # Seems like MajorTOM structure
                for row in tqdm(rows):
                    grid_cells = [d for d in row.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    for grid_cell in grid_cells:
                        product_ids = [d for d in grid_cell.iterdir() if d.is_dir() and not d.name.startswith('.')]
                        for product_id in product_ids:
                            # Check if at least one band exists
                            if any((product_id / f"{band}.tif").exists() for band in self.tif_bands):
                                products.append((grid_cell.name, product_id.name))
            else:
                # Try flat structure - direct product IDs
                product_ids = [d for d in self.data_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
                for product_id in product_ids:
                    # Check if product has required bands
                    if all((product_id / f"{band}.tif").exists() for band in self.tif_bands):
                        # Use the product_id as both grid_cell and product_id for consistency
                        products.append((product_id.name, product_id.name))
        except Exception as e:
            print(f"Error discovering products: {e}")
            # Fallback - look for any .tif files directly in the data_dir
            tif_files = list(self.data_dir.glob("*.tif"))
            if len(tif_files) > 0:
                # Extract unique base names without band identifier
                base_names = set()
                for tif_file in tif_files:
                    # Assume format product_id_bandname.tif
                    parts = tif_file.stem.split('_')
                    if len(parts) > 1:
                        base_name = '_'.join(parts[:-1])  # Everything except the last part (band name)
                        base_names.add(base_name)
                
                # Use these as product IDs
                for base_name in base_names:
                    if all((self.data_dir / f"{base_name}_{band}.tif").exists() for band in self.tif_bands):
                        products.append((base_name, base_name))
        
        return products

    def _preprocess_band_data(self, band_data):
        full_sat_name = f'{self.satellite_type}_{"_".join(self.tif_bands)}'
        custom_config = {
            'class_name': 'SatelliteDataset',
            'normalize_to_neg_one_to_one': self.normalize_to_neg_one_to_one,
            'min_db': self.min_db,
            'max_db': self.max_db,
            'min_positive': self.min_positive,
        }
        return pre_process_data(band_data, full_sat_name, custom_config)

    def __len__(self):
        if self.patchify:
            return len(self.img_list) * self.patches_per_side * self.patches_per_side
        else:
            return len(self.img_list)
    
    def _get_patch_coordinates(self, idx):
        """
        Calculate patch coordinates for a given index when patchify is enabled.
        Returns (img_idx, patch_row, patch_col) where patch_row and patch_col are the patch indices.
        """
        if not self.patchify:
            return idx, 0, 0
            
        img_idx = idx // (self.patches_per_side * self.patches_per_side)
        patch_idx = idx % (self.patches_per_side * self.patches_per_side)
        patch_row = patch_idx // self.patches_per_side
        patch_col = patch_idx % self.patches_per_side
        
        return img_idx, patch_row, patch_col

    def _one_hot_encode(self, data):
        """
        Convert categorical data to one-hot encoded format.
        Args:
            data: numpy array of shape [H, W] with class values
        Returns:
            torch.Tensor of shape [num_classes, H, W] with one-hot encoding
        """
        if self.one_hot_encode_n_classes is None:
            raise ValueError("one_hot_encode_n_classes must be set for categorical data")
        
        assert len(self.value_to_index.keys()) == self.one_hot_encode_n_classes, \
            "Number of classes in value_to_index must match one_hot_encode_n_classes"
            
        one_hot = one_hot_encode(data, value_to_index=self.value_to_index)
            
        return torch.from_numpy(one_hot)

    def __getitem__(self, idx):
        # Get patch coordinates if patchify is enabled
        img_idx, patch_row, patch_col = self._get_patch_coordinates(idx)
        
        # Get the product information from img_list
        grid_cell, product_id = self.img_list[img_idx]
        
        try:
            # Try MajorTOM structure
            # Extract row from grid_cell (first part before underscore)
            row = grid_cell.split('_')[0]
            
            # Construct path according to MajorTOM structure: row/grid_cell/product_id
            product_path = self.data_dir / row / grid_cell / product_id
            
            # If path doesn't exist, try alternate structures
            if not product_path.exists():
                raise FileNotFoundError(f"Product path not found: {product_path}")
                
        except (IndexError, FileNotFoundError):
            # Try flat structure - product_id only
            product_path = self.data_dir / product_id
            
            if not product_path.exists():
                # Try grid_cell_product_id format
                product_path = self.data_dir / f"{grid_cell}_{product_id}"
                
                if not product_path.exists():
                    # Last resort - try looking for files directly in data_dir
                    product_path = self.data_dir

        # Special handling for LULC and cloud mask data
        if 'LULC' in self.satellite_type or 'cloud_mask' in self.tif_bands or 'GOOGLE_EMBEDS' in self.satellite_type:
            try:
                # For LULC we need to add row and grid_cell and no band name
                # For cloud mask we need to load like any other band
                if 'LULC' in self.satellite_type or 'GOOGLE_EMBEDS' in self.satellite_type:
                    band_path = product_path / row / grid_cell.split('_')[1] / f'{product_id}.tif'
                    if not band_path.exists():
                        band_path = self.data_dir / row / grid_cell.split('_')[1] / f'{product_id}.tif'
                elif 'cloud_mask' in self.tif_bands:
                    band_path = product_path / row / grid_cell / product_id / f'{self.tif_bands[0]}.tif'
                    if not band_path.exists():
                        band_path = self.data_dir / row / grid_cell / product_id / f'{self.tif_bands[0]}.tif'
                
                # with Timeout(60):
                with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                    with rio.open(band_path) as f:
                        if 'GOOGLE_EMBEDS' in self.satellite_type:
                            # USING WINDOW MAKES IT MUCH FASTER! TODO: Needs to be implemented for the other modalities too
                            if self.patchify:
                                # Efficiently read only the requested patch to avoid loading the full 1068×1068×C tensor
                                patch_size = self.size
                                img_height, img_width = f.height, f.width
                                if img_height != img_width or img_height < 1068 or img_width < 1068:
                                    print(f"Faulty GOOGLE_EMBEDS grid cell {grid_cell} with shape {img_height}x{img_width}, skipping sample.")
                                    return None
                                # Compute patch top-left coordinate (same logic as _extract_patch)
                                start_y = img_height - patch_size if patch_row == self.patches_per_side - 1 else patch_row * patch_size
                                start_x = img_width  - patch_size if patch_col == self.patches_per_side - 1 else patch_col * patch_size
                                window = Window(start_x, start_y, patch_size, patch_size)
                                data = f.read(window=window)  # Shape [C, patch_size, patch_size]
                            else:
                                data = f.read()  # Read full image when patchify is disabled
                        else:
                            # Use windowed reads for categorical single-channel data
                            img_height, img_width = f.height, f.width
                            if self.patchify:
                                patch_size = self.size
                                start_y = img_height - patch_size if patch_row == self.patches_per_side - 1 else patch_row * patch_size
                                start_x = img_width  - patch_size if patch_col == self.patches_per_side - 1 else patch_col * patch_size
                            else:
                                if img_height >= self.size and img_width >= self.size:
                                    if self.center_crop:
                                        start_y = (img_height - self.size) // 2
                                        start_x = (img_width - self.size) // 2
                                    else:
                                        start_y = random.randint(0, img_height - self.size)
                                        start_x = random.randint(0, img_width - self.size)
                                else:
                                    start_y, start_x = 0, 0
                            read_h = min(self.size, img_height - start_y)
                            read_w = min(self.size, img_width - start_x)
                            data = f.read(1, window=Window(start_x, start_y, read_w, read_h))  # Read as [H, W] for other categorical datasets
                
                # Discard LULC samples with too many black (0) pixels in the selected window/patch
                # Filter out LULC samples that have too many black pixels
                if (self.satellite_type is not None) and ('LULC' in self.satellite_type):
                    zero_ratio = np.mean(data == 0)
                    if zero_ratio >= 0.05: # 5% of the pixels are black
                        return None
                
                stacked_tensor = torch.from_numpy(data).float()
                
                if self.preprocess_bands:
                    # One-hot encoding is handled in the preprocess_band_data function
                    stacked_tensor = self._preprocess_band_data(stacked_tensor)
                
                # Apply resize if needed (skip when patchify is enabled)
                if self.resize_to is not None and not self.patchify:
                    stacked_tensor = T.Resize(self.resize_to)(stacked_tensor)
                
                # Apply patchification or cropping
                if self.patchify and 'GOOGLE_EMBEDS' not in self.satellite_type and stacked_tensor.shape[1] > self.size and stacked_tensor.shape[2] > self.size:
                    # Extract patch from the full image (already done for GOOGLE_EMBEDS above)
                    print(f"MYERROR 1: The shape of the stacked_tensor is {stacked_tensor.shape}, self.size is {self.size}, patch_row is {patch_row}, patch_col is {patch_col}")
                    raise NotImplementedError("MYERROR 1: We should never reach this point as patchify is done before.")
                    stacked_tensor = self._extract_patch(stacked_tensor, patch_row, patch_col)
                else:
                    # Apply random crop or center crop if needed
                    if stacked_tensor.shape[1] > self.size and stacked_tensor.shape[2] > self.size:
                        if self.center_crop:
                            crop_x = (stacked_tensor.shape[1] - self.size) // 2
                            crop_y = (stacked_tensor.shape[2] - self.size) // 2
                        else:
                            crop_x = random.randint(0, stacked_tensor.shape[1] - self.size)
                            crop_y = random.randint(0, stacked_tensor.shape[2] - self.size)
                        
                        stacked_tensor = stacked_tensor[:, crop_x:crop_x+self.size, crop_y:crop_y+self.size]
                    elif stacked_tensor.shape[1] < self.size or stacked_tensor.shape[2] < self.size:
                        print(f"\033[91mStacked tensor is smaller than target size. Returning None.\033[0m")
                        return None
                        # raise NotImplementedError("MYERROR: We should never do padding.")
                        # If image is smaller than target size, pad it
                        # pad_h = max(0, self.size - stacked_tensor.shape[1])
                        # pad_w = max(0, self.size - stacked_tensor.shape[2])
                        
                        # if pad_h > 0 or pad_w > 0:
                        #     stacked_tensor = F.pad(stacked_tensor, (0, pad_w, 0, pad_h), mode='reflect')
                
                # Apply random flip if enabled
                if self.random_flip and random.random() < 0.5:
                    stacked_tensor = torch.flip(stacked_tensor, dims=[-1])
                
                if self.patchify:
                    # Include patch information in the filename
                    patch_id = patch_row * self.patches_per_side + patch_col
                    return {'image': stacked_tensor, 'filename': f"{grid_cell}_patch_{patch_id}"}
                else:
                    return {'image': stacked_tensor, 'filename': grid_cell}
                
            except Exception as e:
                print(f"\033[91mError loading {self.satellite_type} data: {e}. Band path: {band_path}\033[0m")
                # Skip this sample
                return None
        
        # For non-categorical data, continue with existing band loading logic
        # Dictionary to store loaded band data for potential reuse
        loaded_bands = {}
        
        # Pre-compute window for all bands based on the first band to avoid full reads
        try:
            first_band = 'B08' if (self.satellite_type == 'S2L2A' and self.tif_bands == ['NDVI']) else self.tif_bands[0]
            # Resolve first band path
            band_path = product_path / f'{first_band}.tif'
            if not band_path.exists():
                band_path = product_path / f'{product_id}_{first_band}.tif'
                if not band_path.exists():
                    band_path = self.data_dir / f'{product_id}_{first_band}.tif'
            with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                with rio.open(band_path) as f0:
                    img_height0, img_width0 = f0.height, f0.width
                    base_x, base_y = 0, 0
                    eff_width, eff_height = img_width0, img_height0
                    if self.satellite_type == 'S1RTC':
                        # Accept 1068x1068, 1068x1069, 1069x1068, 1069x1069, but crop to 1068x1068
                        if eff_height < 1068 or eff_width < 1068:
                            print(f"Faulty S1RTC grid cell {grid_cell} with shape {(eff_height, eff_width)}, skipping sample.")
                            return None
                        # If either dimension is 1069, crop off the extra pixel
                        if eff_height > 1068 or eff_width > 1068:
                            print(f"S1RTC grid cell {grid_cell} with shape {(eff_height, eff_width)}, cropping to 1068x1068.")
                            # Always crop from the center
                            base_x = (eff_width - 1068) // 2
                            base_y = (eff_height - 1068) // 2
                            eff_width = 1068
                            eff_height = 1068
                    # Determine crop start within effective region
                    if self.patchify:
                        patch_size = self.size
                        start_y_eff = eff_height - patch_size if patch_row == self.patches_per_side - 1 else patch_row * patch_size
                        start_x_eff = eff_width  - patch_size if patch_col == self.patches_per_side - 1 else patch_col * patch_size
                    else:
                        if eff_height >= self.size and eff_width >= self.size:
                            if self.center_crop:
                                start_y_eff = (eff_height - self.size) // 2
                                start_x_eff = (eff_width - self.size) // 2
                            else:
                                start_y_eff = random.randint(0, eff_height - self.size)
                                start_x_eff = random.randint(0, eff_width - self.size)
                        else:
                            start_y_eff, start_x_eff = 0, 0
                    # Final window in file coordinates
                    window_start_y = base_y + start_y_eff
                    window_start_x = base_x + start_x_eff
                    read_h = min(self.size, eff_height - start_y_eff)
                    read_w = min(self.size, eff_width - start_x_eff)
        except Exception as e:
            print(f"\033[91mError precomputing window for bands: {e}\033[0m")
            # Fallback to reading full bands
            window_start_y = 0
            window_start_x = 0
            read_h = None
            read_w = None
        
        # Load each band as a tensor
        # NDVI specific case (not fully working yet as bands are not loading)
        if self.satellite_type == 'S2L2A' and self.tif_bands == ['NDVI']:
            try:
                # Read B08
                band = 'B08'
                band_path = product_path / f'{band}.tif'
                if not band_path.exists():
                    band_path = product_path / f'{product_id}_{band}.tif'
                    if not band_path.exists():
                        band_path = self.data_dir / f'{product_id}_{band}.tif'
                with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                    with rio.open(band_path) as f:
                        if read_h is not None and read_w is not None:
                            b08 = f.read(1, window=Window(window_start_x, window_start_y, read_w, read_h))
                        else:
                            print("\033[93mWARNING 1: We are reading the full band\033[0m")
                            b08 = f.read(1)
                        
                        if self.preprocess_bands:
                            b08 = self._preprocess_band_data(b08)
                
                # Read B04
                band = 'B04'
                band_path = product_path / f'{band}.tif'
                if not band_path.exists():
                    band_path = product_path / f'{product_id}_{band}.tif'
                    if not band_path.exists():
                        band_path = self.data_dir / f'{product_id}_{band}.tif'
                with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                    with rio.open(band_path) as f:
                        if read_h is not None and read_w is not None:
                            b04 = f.read(1, window=Window(window_start_x, window_start_y, read_w, read_h))
                        else:
                            print("\033[93mWARNING 2: We are reading the full band\033[0m")
                            b04 = f.read(1)
                        
                        if self.preprocess_bands:
                            b04 = self._preprocess_band_data(b04)
                
                # Compute NDVI
                b08 = b08.astype(np.float32)
                b04 = b04.astype(np.float32)
                denom = b08 + b04
                ndvi = (b08 - b04) / (denom + 1e-6)
                
                stacked_tensor = torch.from_numpy(ndvi).float().unsqueeze(0)
                if self.resize_to is not None and not self.patchify:
                    stacked_tensor = T.Resize(self.resize_to)(stacked_tensor)
                # Pad if smaller than target size
                if not self.patchify:
                    if stacked_tensor.shape[1] < self.size or stacked_tensor.shape[2] < self.size:
                        print(f"\033[91mNDVI is smaller than target size. Returning None.\033[0m")
                        return None
                        # raise NotImplementedError("MYERROR: We should never do padding.")
                        # pad_h = max(0, self.size - stacked_tensor.shape[1])
                        # pad_w = max(0, self.size - stacked_tensor.shape[2])
                        # if pad_h > 0 or pad_w > 0:
                        #     stacked_tensor = F.pad(stacked_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            except Exception as e:
                print(f"\033[91mError computing NDVI: {e}\033[0m")
                # Skip this sample
                return None
        else: # All other cases
            band_tensors = []
            for band in self.tif_bands:
                # Regular band - load from file using windowed reads
                try:
                    # First try direct band file
                    band_path = product_path / f'{band}.tif'
                    if not band_path.exists():
                        # Try product_band format
                        band_path = product_path / f'{product_id}_{band}.tif'
                        
                        if not band_path.exists():
                            # Try direct in data_dir
                            band_path = self.data_dir / f'{product_id}_{band}.tif'
                    
                    with rio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR'):
                        with rio.open(band_path) as f:
                            # Read the band data using a window if available
                            if read_h is not None and read_w is not None:
                                band_data = f.read(1, window=Window(window_start_x, window_start_y, read_w, read_h))
                            else:
                                print("\033[93mWARNING: We are reading the full band\033[0m")
                                band_data = f.read(1)
                            
                            # Apply satellite-specific preprocessing
                            if self.preprocess_bands:
                                band_data = self._preprocess_band_data(band_data)
                            
                            # Convert to tensor
                            band_tensor = torch.from_numpy(band_data).float().unsqueeze(0)
                except Exception as e:
                    print(f"\033[91mError loading band {band}: {e}. Band path: {band_path}\033[0m")
                    # Skip this sample entirely
                    return None
            
                if self.resize_to is not None and not self.patchify:
                    band_tensor = T.Resize(self.resize_to)(band_tensor)
            
                # Apply patchification or cropping
                if self.patchify:
                    # For patchification, we already read only the requested patch
                    pass
                else:
                    # Apply random crop or center crop if needed
                    if band_tensor.shape[1] > self.size and band_tensor.shape[2] > self.size:
                        raise NotImplementedError("MYERROR: We should never reach this point as cropping is done before.")
                        if band == self.tif_bands[0]:  # Only get crop indices for first band
                            if self.center_crop:
                                # Calculate center crop indices
                                self.crop_x = (band_tensor.shape[1] - self.size) // 2
                                self.crop_y = (band_tensor.shape[2] - self.size) // 2
                            else:  
                                # Random crop indices
                                self.crop_x = random.randint(0, band_tensor.shape[1] - self.size)
                                self.crop_y = random.randint(0, band_tensor.shape[2] - self.size)
                    
                        # Apply same crop to all bands
                        band_tensor = band_tensor[:, self.crop_x:self.crop_x+self.size, self.crop_y:self.crop_y+self.size]
                    elif band_tensor.shape[1] < self.size or band_tensor.shape[2] < self.size:
                        print(f"\033[91mBand {band} is smaller than target size. Returning None.\033[0m")
                        return None
                        # raise NotImplementedError("MYERROR: We should never do padding.")
                        # # If image is smaller than target size, pad it
                        # pad_h = max(0, self.size - band_tensor.shape[1])
                        # pad_w = max(0, self.size - band_tensor.shape[2])
                        
                        # if pad_h > 0 or pad_w > 0:
                        #     band_tensor = F.pad(band_tensor, (0, pad_w, 0, pad_h), mode='reflect')
            
                band_tensors.append(band_tensor)
            
            # Stack bands along channel dimension
            stacked_tensor = torch.cat(band_tensors, dim=0)
        
        # Apply patchification if enabled
        if self.patchify and (stacked_tensor.shape[1] > self.size and stacked_tensor.shape[2] > self.size):
            raise NotImplementedError("MYERROR 2: We should never reach this point as patchify is done before.")
            stacked_tensor = self._extract_patch(stacked_tensor, patch_row, patch_col)
        
        # Apply random flip if enabled
        if self.random_flip and random.random() < 0.5:
            stacked_tensor = torch.flip(stacked_tensor, dims=[-1])
        
        # Ensure the tensor owns its storage (tensors created via torch.from_numpy share the
        # NumPy buffer which is marked as non-resizable, breaking default_collate when it tries
        # to create a bigger shared storage).  Cloning moves the data into a fresh, resizable
        # PyTorch storage with negligible overhead compared to the I/O cost we already paid.
        # stacked_tensor = stacked_tensor.clone()

        if self.patchify:
            # Include patch information in the filename
            patch_id = patch_row * self.patches_per_side + patch_col
            return {'image': stacked_tensor, 'filename': f"{grid_cell}_patch_{patch_id}"}
        else:
            return {'image': stacked_tensor, 'filename': grid_cell}

    def _extract_patch(self, tensor, patch_row, patch_col):
        """
        Extract a patch from the full image tensor.
        Args:
            tensor: Full image tensor of shape [C, H, W]
            patch_row: Row index of the patch
            patch_col: Column index of the patch
        Returns:
            Patch tensor of shape [C, patch_size, patch_size]
        """
        if not self.patchify:
            return tensor
            
        # Calculate patch size and number of patches per side
        patch_size = self.size
        
        # Calculate the starting coordinates for the patch
        # For the last patch in each row/column, we allow overlap to cover the entire image
        img_height, img_width = tensor.shape[1], tensor.shape[2]
        
        if patch_row == self.patches_per_side - 1:
            # Last row - right-align the patch
            start_y = img_height - patch_size
        else:
            start_y = patch_row * patch_size
            
        if patch_col == self.patches_per_side - 1:
            # Last column - right-align the patch
            start_x = img_width - patch_size
        else:
            start_x = patch_col * patch_size
            
        # Extract the patch
        patch = tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size]
        
        return patch


if __name__ == '__main__':
    dataset = CIFAR10(
        img_folder='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/cifar-10-python',
        augment_horizontal_flip=False
    )

    dataset = EdgeDataset(
        data_root='/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/data/BSDS_my_aug96',
        image_size=[320, 320],
    )
    for i in range(len(dataset)):
        d = dataset[i]
        mask = d['cond']
        print(mask.max())
    dl = data.DataLoader(dataset, batch_size=2, shuffle=False, pin_memory=True, num_workers=0)

    pause = 0