import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import rasterio as rio
from PIL import Image
import torchvision.transforms as transforms
import random
from ddm.pre_post_process_data import pre_process_data

class NMajorTOM(Dataset):
    """NMajorTOM Dataset with multiple modalities (https://huggingface.co/Major-TOM)

    Args:
        modalities (dict): Dictionary of modality configurations, where each key is a modality name
                          and value is a dict containing:
                          - df: Metadata dataframe for that modality
                          - local_dir: Root directory for that modality
                          - tif_bands: List of tif bands to read
                          - png_bands: List of png bands to read
                          - tif_transforms: List of transforms for tif files (Defined in ddm/pre_post_process_data.py)
                          - png_transforms: List of transforms for png files
        random_flip (bool): Whether to randomly flip all modalities together
        ratio_train_test (float): Ratio of training samples (e.g., 0.8 for 80% train, 20% test)
        seed (int): Random seed for reproducible train/test splits
    """
    
    def __init__(self, modalities, random_flip=True, ratio_train_test=0.8, seed=42):
        super().__init__()
        self.modalities = {}
        
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Process each modality's configuration
        for modality_name, config in modalities.items():
            # Drop rows that are complete duplicates across all relevant columns
            num_rows = len(config['df'])
            if modality_name == 'S1RTC':
                relevant_cols = ['grid_cell', 'grid_row_u', 'grid_col_r', 'product_id', 
                                    'timestamp', 'nodata', 'orbit_state', 'centre_lat', 
                                    'centre_lon', 'crs', 'parquet_url', 'geometry']
            else:
                relevant_cols = list(config['df'].keys())
            config['df'] = config['df'].drop_duplicates(subset=relevant_cols)
            print(f"Dropped {num_rows - len(config['df'])} duplicates from {modality_name}")
            
            # By now, we should have no duplicate grid_cells
            if config['df']['grid_cell'].duplicated().any():
                raise ValueError(f"Found rows with duplicate grid_cells but different values in modality {modality_name}")
                
            self.modalities[modality_name] = {
                'df': config['df'],
                'local_dir': Path(config['local_dir']) if isinstance(config['local_dir'], str) else config['local_dir'],
                'tif_bands': config['tif_bands'] if not isinstance(config['tif_bands'], str) else [config['tif_bands']],
                'png_bands': config['png_bands'] if not isinstance(config['png_bands'], str) else [config['png_bands']],
                'tif_transforms': transforms.Compose(config['tif_transforms']) if config['tif_transforms'] is not None else None,
                'png_transforms': transforms.Compose(config['png_transforms']) if config['png_transforms'] is not None else None
            }
        
        self.random_flip = random_flip
        
        # Get the set of grid_cells for each modality
        grid_cells_by_modality = {
            name: set(mod['df']['grid_cell'].values) 
            for name, mod in self.modalities.items()
        }
        
        # Check that all modalities share the same grid_cells
        if len(grid_cells_by_modality) > 0:
            reference_grid_cells = grid_cells_by_modality[list(grid_cells_by_modality.keys())[0]]
            for modality_name, grid_cells in grid_cells_by_modality.items():
                if grid_cells != reference_grid_cells:
                    missing = reference_grid_cells - grid_cells
                    extra = grid_cells - reference_grid_cells
                    error_msg = f"Modality {modality_name} has mismatched grid_cells.\n"
                    if missing:
                        error_msg += f"Missing grid_cells: {missing}\n"
                    if extra:
                        error_msg += f"Extra grid_cells: {extra}"
                    raise ValueError(error_msg)
        
        # Sort all dataframes by grid_cell for consistent sampling
        for modality in self.modalities.values():
            modality['df'] = modality['df'].sort_values('grid_cell').reset_index(drop=True)
            
        
        print("Creating train/test split...")
        
        # After sorting dataframes, create train/test split
        all_grid_cells = list(reference_grid_cells)
        random.shuffle(all_grid_cells)
        
        n_train = int(len(all_grid_cells) * ratio_train_test)
        self.train_grid_cells = set(all_grid_cells[:n_train])
        self.test_grid_cells = set(all_grid_cells[n_train:])
        
        # Let's create a dictionary of grid_cells to split
        self.grid_cell_to_split = {grid_cell: 'train' if grid_cell in self.train_grid_cells else 'test' for grid_cell in reference_grid_cells}
        
        print(f"Split dataset into {len(self.train_grid_cells)} train and {len(self.test_grid_cells)} test grid cells")

    def __len__(self):
        # Return length of any modality (they should all be the same)
        assert len(self.modalities) > 0, "No modalities provided"
        # Get len for each modality and make sure they are the same
        lengths = [len(mod['df']) for mod in self.modalities.values()]
        if not all(x == lengths[0] for x in lengths):
            raise ValueError("All modalities must have the same number of samples")
        return lengths[0]

    def __getitem__(self, idx):
        result = {}
        
        # Generate the same random flip decision for all modalities
        do_flip = self.random_flip and random.random() < 0.5
        
        # Get the grid cell for this index (they're all the same across modalities)
        first_modality = list(self.modalities.keys())[0]
        current_grid_cell = self.modalities[first_modality]['df'].iloc[idx]['grid_cell']
        
        # Determine if this sample is in train or test set
        split = self.grid_cell_to_split[current_grid_cell]
        
        for modality_name, modality in self.modalities.items():
            meta = modality['df'].iloc[idx]
            product_id = meta.product_id if 'product_id' in meta.index else "id"
            grid_cell = meta.grid_cell
            row = grid_cell.split('_')[0]
            
            path = modality['local_dir'] / Path(f"{row}/{grid_cell}/{product_id}")
            out_dict = {}
            
            # Process TIF bands
            for band in modality['tif_bands']:
                with rio.open(path / f'{band}.tif') as f:
                    # out = torch.from_numpy(f.read()).float()
                    out = f.read()
                    # TODO: Needs update with post_processing and new visualise_bands function
                    out = pre_process_data(out, modality_name, data_config=modality['data_config'])
                    out = torch.from_numpy(out).float()
                # if modality['tif_transforms'] is not None:
                #     out = modality['tif_transforms'](out)
                out_dict[band] = out
            
            # Process PNG bands
            for band in modality['png_bands']:
                out = Image.open(path / f'{band}.png')
                if modality['png_transforms'] is not None:
                    out = modality['png_transforms'](out)
                out_dict[band] = out
            
            # Apply the same random flip to all bands in this modality
            if do_flip:
                out_dict = {k: v.flip(-1) for k, v in out_dict.items()}
            
            # Add split information to the output dictionary
            out_dict['split'] = split
            out_dict['grid_cell'] = current_grid_cell
            
            result[modality_name] = out_dict
            
        # Assert the grid_cells are the same for all modalities in the resulting dictionary
        if len(result) > 0:
            first_modality = list(result.keys())[0]
            first_grid_cell = self.modalities[first_modality]['df'].iloc[idx]['grid_cell']
            for modality_name in result.keys():
                current_grid_cell = self.modalities[modality_name]['df'].iloc[idx]['grid_cell']
                if current_grid_cell != first_grid_cell:
                    raise ValueError(f"Mismatched grid_cells found: {current_grid_cell} != {first_grid_cell}")
                # Add grid_cell to the output dictionary for verification
                result[modality_name]['grid_cell'] = current_grid_cell
        
        return result