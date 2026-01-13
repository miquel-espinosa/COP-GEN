import torch
import rioxarray as rxr
import random
import json
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import logging
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from pathlib import Path
from typing import Union, List
from .base import render_output, print_lulc_stats
from benchmark.utils.plottingutils import plot_text, coords_to_text
from benchmark.common.paths import data_input_dir, data_output_dir, comparisons_vis_dir, coords_json_path
from benchmark.common.modalities import S2_BAND_ORDER
from benchmark.common.tile_matching import find_matching_file as find_tile
from benchmark.io.coords import load_coords_json
from PIL import Image
from visualisations.visualise_bands import visualise_bands

def _save_vis_image(img, path: Path):
    """
    Save a visualization tensor/array as a PNG without matplotlib.
    Accepts CHW, HWC or HW; values are expected in [0, 1].
    """
    if torch.is_tensor(img):
        arr = img.detach().cpu().numpy()
    else:
        arr = np.asarray(img)
    if arr.ndim == 4:
        arr = np.squeeze(arr, axis=0)
    # Convert channel-first (C,H,W) to (H,W,C) when C in {1,3}
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    # Clip to [0,1] and convert to uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr_uint8 = (arr * 255.0).round().astype(np.uint8)
    # If single-channel with trailing dim 1, squeeze it for 'L' mode
    if arr_uint8.ndim == 3 and arr_uint8.shape[2] == 1:
        arr_uint8 = arr_uint8[:, :, 0]
    Image.fromarray(arr_uint8).save(path)

class ComparisonVisualizer:
    def __init__(self, input_modality: Union[str, List[str]], output_modality: str, root, crop_size=256, seed=1234):
        self.root = Path(root)
        self.crop_size = crop_size 
        
        if isinstance(input_modality, str):
            self.input_modalities = [input_modality]
            self.input_modality = input_modality
        else:
            self.input_modalities = input_modality
            self.input_modality = "_".join(input_modality)
        
        self.output_modality = output_modality

        self.base_input_modality = self._extract_base(self.input_modality)
        self.base_output_modality = self._extract_base(output_modality)

        self.input_dirs = {mod: data_input_dir(self.root, mod) for mod in self.input_modalities}
        self.experiment_name = f"input_{'_'.join(self.input_modalities)}_output_{output_modality}_seed_{seed}"
        self.output_dir = data_output_dir(self.root, self.experiment_name)

        if self.base_output_modality == "coords":
            self.gt_dir = data_input_dir(self.root, "coords")
            self.coords_json = coords_json_path(self.root)
        else:
            self.gt_dir = self.root / "terramind_inputs" / self.base_output_modality

        self.vis_dir = comparisons_vis_dir(self.root, self.experiment_name)
        self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _extract_base(self, modality):
        return modality.split("_from_")[0] if "_from_" in modality else modality

    def _s2_group_defs(self, modality: str):
        if modality == "S2L2A":
            return [
                ("S2L2A_B02_B03_B04_B08", ["B02", "B03", "B04", "B08"]),
                ("S2L2A_B05_B06_B07_B11_B12_B8A", ["B05", "B06", "B07", "B11", "B12", "B8A"]),
                ("S2L2A_B01_B09", ["B01", "B09"]),
            ]
        if modality == "S2L1C":
            return [
                ("S2L1C_B02_B03_B04_B08", ["B02", "B03", "B04", "B08"]),
                ("S2L1C_B05_B06_B07_B11_B12_B8A", ["B05", "B06", "B07", "B11", "B12", "B8A"]),
                ("S2L1C_B01_B09_B10", ["B01", "B09", "B10"]),
            ]
        return []

    def _slice_s2_bands(self, tensor: torch.Tensor, modality: str, band_names: List[str]) -> torch.Tensor:
        # tensor shape: [1, C, H, W]
        order = S2_BAND_ORDER.get(modality, [])
        band_to_idx = {p.replace(".tif", "").replace(".TIF", ""): i for i, p in enumerate(order)}
        indices = []
        for b in band_names:
            if b not in band_to_idx:
                logging.warning(f"Band {b} not found in expected order for {modality}")
                continue
            idx = band_to_idx[b]
            if idx >= tensor.shape[1]:
                logging.warning(f"Band index {idx} for {b} exceeds tensor channels ({tensor.shape[1]})")
                continue
            indices.append(idx)
        if not indices:
            raise ValueError(f"No valid bands found for {modality} with requested {band_names}")
        return tensor[:, indices, ...]

    def _reorder_s1_vh_vv(self, tensor: torch.Tensor) -> torch.Tensor:
        # Expect 2 channels [VV, VH]; return [VH, VV] to match 'vh_vv' naming
        if tensor.shape[1] >= 2:
            return tensor[:, [1, 0], ...]
        return tensor

    def find_matching_file(self, directory, tile_name):
        return find_tile(directory, tile_name)

    def visualize(self, n_examples=5, stride_every: int = None, consecutive_visualisations: int = 4):
        logging.info('start visualization')
        output_files = list(self.output_dir.glob("*.tif"))
        
        # Always sort deterministically first
        output_files = sorted(output_files)
        
        # If a stride is provided, prefer deterministic striding over random sampling
        if stride_every is not None and isinstance(stride_every, int) and stride_every > 1:
            # Take every Nth file, then take consecutive_visualisations files at each stride point
            strided_files = []
            for i in range(0, len(output_files), stride_every):
                # Take consecutive_visualisations files starting from this stride point
                end_idx = min(i + consecutive_visualisations, len(output_files))
                strided_files.extend(output_files[i:end_idx])
            output_files = strided_files
        elif n_examples and n_examples < len(output_files):
            output_files = sorted(random.sample(output_files, n_examples))

        for i, output_path in enumerate(output_files):
            tile = output_path.stem
            tile_vis_dir = self.vis_dir / tile
            tile_vis_dir.mkdir(parents=True, exist_ok=True)
                        
            input_data = {}
            missing_inputs = []

            for modality in self.input_modalities:
                if modality == "coords":
                    coords_file = coords_json_path(self.root)
                    if coords_file.exists():
                        coords_data = load_coords_json(coords_file)
                        if tile in coords_data:
                            input_data[modality] = coords_data[tile]  # Store the coord string
                        else:
                            missing_inputs.append(modality)
                    else:
                        missing_inputs.append(modality)
                else:
                    matching_file = self.find_matching_file(self.input_dirs[modality], tile)
                    if matching_file is None:
                        missing_inputs.append(modality)
                    else:
                        input_data[modality] = matching_file
                
            gt_file = self.find_matching_file(self.gt_dir, tile)
            if gt_file is None:
                logging.warning(f"Skipping {tile}: missing ground truth {self.base_output_modality}")
                continue
            
            if missing_inputs:
                logging.warning(f"Skipping {tile}: missing {missing_inputs}")
                continue

            output_arr = rxr.open_rasterio(output_path).squeeze().values
            if output_arr.ndim == 2:
                output_arr = output_arr[None, ...]
            output_tensor = torch.tensor(output_arr).float().unsqueeze(0)

            if self.base_output_modality == "coords":
                output_vis = None
            else:
                output_vis = render_output(self.base_output_modality, output_tensor)

            gt_arr = rxr.open_rasterio(gt_file).squeeze().values
            if gt_arr.ndim == 2:
                gt_arr = gt_arr[None, ...]
            gt_tensor = torch.tensor(gt_arr).float().unsqueeze(0)

            if self.base_output_modality == "coords":
                gt_vis = None
            else:
                crop = T.CenterCrop(self.crop_size)
                gt_tensor_cropped = crop(gt_tensor)
                gt_vis = render_output(self.base_output_modality, gt_tensor_cropped)

            n_inputs = len(self.input_modalities)
            fig, axes = plt.subplots(1, n_inputs + 2, figsize=((n_inputs + 2) * 4, 6))

            for ax in axes:
                ax.set_aspect('equal', adjustable='box')    
            
            if n_inputs + 2 == 1:
                axes = [axes]

            crop = T.CenterCrop(self.crop_size)
            for i, (modality, input_data_item) in enumerate(input_data.items()):
                if modality == "coords":
                    coords_str = input_data_item 
                    
                    plot_text(coords_str, ax=axes[i])
                    axes[i].set_title("coords Input")
                else:
                    input_arr = rxr.open_rasterio(input_data_item).squeeze().values
                    if input_arr.ndim == 2:
                        input_arr = input_arr[None, ...]
                    
                    input_tensor = torch.tensor(input_arr).float().unsqueeze(0)
                    input_tensor = crop(input_tensor)
                    input_vis = render_output(modality, input_tensor)
                    
                    if input_vis.ndim == 4:
                        input_vis = input_vis.squeeze(0)
                    if input_vis.ndim == 3 and input_vis.shape[0] in [1, 3]:
                        input_vis = input_vis.permute(1, 2, 0)
                    
                    # Save raw input visualization image
                    mod_base = self._extract_base(modality)
                    _save_vis_image(input_vis, tile_vis_dir / f"input_{mod_base}.png")

                    axes[i].imshow(input_vis.cpu().numpy())
                    axes[i].set_title(f"{modality} Input ({self.crop_size}x{self.crop_size})")
                    axes[i].axis("off")

                    # Also save using COP-GEN visualisation tools with consistent grouping and colors
                    try:
                        if mod_base in ("S2L2A", "S2L1C"):
                            for group_name, bands in self._s2_group_defs(mod_base):
                                group_tensor = self._slice_s2_bands(input_tensor, mod_base, bands)
                                save_dir = tile_vis_dir / group_name
                                visualise_bands(
                                    inputs=None,
                                    reconstructions=group_tensor,
                                    save_dir=save_dir,
                                    n_images_to_log=1,
                                    milestone=None,
                                    satellite_type=group_name,
                                    view_difference=False,
                                    view_histogram=False,
                                    resize=self.crop_size,
                                    # Initial conditions recon-only use softer red frame
                                    recon_frame_color=(1.0, 0.3, 0.2),
                                )
                        elif mod_base == "S1RTC":
                            save_dir = tile_vis_dir / "S1RTC_vh_vv"
                            s1_tensor = self._reorder_s1_vh_vv(input_tensor)
                            visualise_bands(
                                inputs=None,
                                reconstructions=s1_tensor,
                                save_dir=save_dir,
                                n_images_to_log=1,
                                milestone=None,
                                satellite_type="S1RTC_vh_vv",
                                view_difference=False,
                                view_histogram=False,
                                resize=self.crop_size,
                                recon_frame_color=(1.0, 0.3, 0.2),
                            )
                        elif mod_base == "DEM":
                            save_dir = tile_vis_dir / "DEM_DEM"
                            visualise_bands(
                                inputs=None,
                                reconstructions=input_tensor,
                                save_dir=save_dir,
                                n_images_to_log=1,
                                milestone=None,
                                satellite_type="DEM_DEM",
                                view_difference=False,
                                view_histogram=False,
                                resize=self.crop_size,
                                recon_frame_color=(1.0, 0.3, 0.2),
                            )
                        elif mod_base == "LULC":
                            save_dir = tile_vis_dir / "LULC_LULC"
                            visualise_bands(
                                inputs=None,
                                reconstructions=input_tensor,
                                save_dir=save_dir,
                                n_images_to_log=1,
                                milestone=None,
                                satellite_type="LULC_LULC",
                                view_difference=False,
                                view_histogram=False,
                                resize=self.crop_size,
                                recon_frame_color=(1.0, 0.3, 0.2),
                            )
                        else:
                            # Skip coords and unknowns
                            pass
                    except Exception as vis_err:
                        logging.warning(f"visualise_bands for input {mod_base} failed on {tile}: {vis_err}")

            if self.base_output_modality == "coords":
                coord_text = coords_to_text(output_tensor)
                plot_text(coord_text, ax=axes[n_inputs])
                axes[n_inputs].set_title(f"Generated {self.base_output_modality}")
            else:
                if output_vis.ndim == 4:
                    output_vis = output_vis.squeeze(0)
                if output_vis.ndim == 3 and output_vis.shape[0] in [1, 3]:
                    output_vis = output_vis.permute(1, 2, 0)
                
                # Save raw prediction visualization image
                _save_vis_image(output_vis, tile_vis_dir / f"pred_{self.base_output_modality}.png")

                axes[n_inputs].imshow(output_vis.cpu().numpy())
                axes[n_inputs].set_title(f"Generated {self.base_output_modality}")
                axes[n_inputs].axis("off")

                # Also save using COP-GEN visualisation tools (with GT comparison) and consistent grouping and colors
                try:
                    # Crop output to match GT crop for consistent comparison tiles
                    output_tensor_cropped = T.CenterCrop(self.crop_size)(output_tensor)
                    if self.base_output_modality in ("S2L2A", "S2L1C"):
                        for group_name, bands in self._s2_group_defs(self.base_output_modality):
                            out_group = self._slice_s2_bands(output_tensor_cropped, self.base_output_modality, bands)
                            gt_group = self._slice_s2_bands(gt_tensor_cropped, self.base_output_modality, bands)
                            save_dir = tile_vis_dir / group_name
                            visualise_bands(
                                inputs=gt_group,
                                reconstructions=out_group,
                                save_dir=save_dir,
                                n_images_to_log=1,
                                milestone=None,
                                satellite_type=group_name,
                                view_difference=False,
                                view_histogram=False,
                                resize=self.crop_size,
                                # GT uses green, recon uses softer blue
                                input_frame_color=(0.2, 0.9, 0.3),
                                recon_frame_color=(0.3, 0.5, 1.0),
                            )
                    elif self.base_output_modality == "S1RTC":
                        save_dir = tile_vis_dir / "S1RTC_vh_vv"
                        out_s1 = self._reorder_s1_vh_vv(output_tensor_cropped)
                        gt_s1 = self._reorder_s1_vh_vv(gt_tensor_cropped)
                        visualise_bands(
                            inputs=gt_s1,
                            reconstructions=out_s1,
                            save_dir=save_dir,
                            n_images_to_log=1,
                            milestone=None,
                            satellite_type="S1RTC_vh_vv",
                            view_difference=False,
                            view_histogram=False,
                            resize=self.crop_size,
                            input_frame_color=(0.2, 0.9, 0.3),
                            recon_frame_color=(0.3, 0.5, 1.0),
                        )
                    elif self.base_output_modality == "DEM":
                        save_dir = tile_vis_dir / "DEM_DEM"
                        visualise_bands(
                            inputs=gt_tensor_cropped,
                            reconstructions=output_tensor_cropped,
                            save_dir=save_dir,
                            n_images_to_log=1,
                            milestone=None,
                            satellite_type="DEM_DEM",
                            view_difference=False,
                            view_histogram=False,
                            resize=self.crop_size,
                            input_frame_color=(0.2, 0.9, 0.3),
                            recon_frame_color=(0.3, 0.5, 1.0),
                        )
                    elif self.base_output_modality == "LULC":
                        save_dir = tile_vis_dir / "LULC_LULC"
                        visualise_bands(
                            inputs=gt_tensor_cropped,
                            reconstructions=output_tensor_cropped,
                            save_dir=save_dir,
                            n_images_to_log=1,
                            milestone=None,
                            satellite_type="LULC_LULC",
                            view_difference=False,
                            view_histogram=False,
                            resize=self.crop_size,
                            input_frame_color=(0.2, 0.9, 0.3),
                            recon_frame_color=(0.3, 0.5, 1.0),
                        )
                    else:
                        # Skip coords and unknowns
                        pass
                except Exception as vis_err:
                    logging.warning(f"visualise_bands for output {self.base_output_modality} failed on {tile}: {vis_err}")

            if self.base_output_modality == "coords":
                gt_coord_text = coords_to_text(gt_tensor)
                plot_text(gt_coord_text, ax=axes[n_inputs + 1])
                axes[n_inputs + 1].set_title(f"Ground Truth {self.base_output_modality}")
            else:
                if gt_vis.ndim == 4:
                    gt_vis = gt_vis.squeeze(0)
                if gt_vis.ndim == 3 and gt_vis.shape[0] in [1, 3]:
                    gt_vis = gt_vis.permute(1, 2, 0)

                # Save raw ground-truth visualization image
                _save_vis_image(gt_vis, tile_vis_dir / f"gt_{self.base_output_modality}.png")

                axes[n_inputs + 1].imshow(gt_vis.cpu().numpy())
                axes[n_inputs + 1].set_title(f"Ground Truth {self.base_output_modality}")
                axes[n_inputs + 1].axis("off")

            plt.suptitle(f"{self.input_modality} → {self.output_modality}: {tile}")
            plt.tight_layout()

            vis_path = self.vis_dir / f"{tile}.png"
            plt.savefig(vis_path, bbox_inches="tight", dpi=150)
            plt.close()
            print(f"Saved comparison: {vis_path}")

            if self.base_output_modality == "coords":
                gen_lat = output_arr[0] if len(output_arr) > 0 else np.nan
                gen_lon = output_arr[1] if len(output_arr) > 1 else np.nan
                gt_lat = gt_arr[0] if len(gt_arr) > 0 else np.nan
                gt_lon = gt_arr[1] if len(gt_arr) > 1 else np.nan
                
                print(f"Generated coords: lat={gen_lat:.2f}, lon={gen_lon:.2f}")
                print(f"Ground truth coords: lat={gt_lat:.2f}, lon={gt_lon:.2f}")
                
                if not (np.isnan(gen_lat) or np.isnan(gt_lat)):
                    lat_diff = abs(gen_lat - gt_lat)
                    lon_diff = abs(gen_lon - gt_lon)
                    print(f"Difference: lat={lat_diff:.3f}, lon={lon_diff:.3f}")

            # # Print LULC stats if applicable (using cropped data)
            # for modality, input_path in input_data.items():
            #     if modality == "LULC":
            #         input_arr = rxr.open_rasterio(input_path).squeeze().values
            #         if input_arr.ndim == 2:
            #             input_arr = input_arr[None, ...]
            #         cropped_tensor = crop(torch.tensor(input_arr).unsqueeze(0))
            #         print_lulc_stats("Input (cropped)", cropped_tensor.squeeze().numpy())
            
            # if self.base_output_modality == "LULC":
            #     print_lulc_stats("Generated", output_arr.squeeze())
            #     print_lulc_stats("Ground Truth (cropped)", gt_tensor_cropped.squeeze().numpy())