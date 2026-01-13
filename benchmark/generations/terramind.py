import rasterio
from pathlib import Path
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from terratorch.registry import FULL_MODEL_REGISTRY
from benchmark.dataloaders import CopGenDataset
from benchmark.evaluation import ImageRegressionMetrics, SpatialMetrics, SegmentationMetrics
from benchmark.evaluation.metrics import select_metrics
from benchmark.common.paths import data_input_dir, data_output_dir, coords_json_path
from benchmark.io.tiff import write_tif, default_meta_if_missing
import torchvision.transforms as T
import csv

class TerraMindGenerator:
    def __init__(self, input_modalities, output_modality, root, crop_size=(256, 256),
                 device=None, model_name="terramind_v1_base_generate", timesteps=50,
                 standardize=True, pretrained=True, seed=1234, input_overrides=None):

        self.root = Path(root)
        self.input_modalities = input_modalities
        self.input_overrides = input_overrides or {}
        self.output_modality = output_modality
        self.input_dirs = {mod: data_input_dir(self.root, mod) for mod in input_modalities}
        self.experiment_name = f"input_{'_'.join(input_modalities)}_output_{output_modality}_seed_{seed}"
        self.output_dir = data_output_dir(self.root, self.experiment_name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_size = crop_size
        self.timesteps = timesteps
        # CSV output path for coordinate predictions
        self.coords_csv_path = None
        if self.output_modality == "coords":
            self.coords_csv_path = self.root / "outputs" / "terramind" / self.experiment_name / "lat_lon.csv"
            if not self.coords_csv_path.exists():
                with self.coords_csv_path.open("w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["tile_id", "lat", "lon"])

        self.coords_file = coords_json_path(self.root) if "coords" in input_modalities else None

        # Initialize evaluation metrics based on output modality
        self.metrics = select_metrics(self.output_modality)
        if isinstance(self.metrics, SpatialMetrics):
            logging.info("Initialized spatial metrics for coordinate evaluation")
        elif isinstance(self.metrics, SegmentationMetrics):
            logging.info("Initialized segmentation metrics for LULC evaluation")
        else:
            logging.info(f"Initialized image regression metrics for {self.output_modality} evaluation")
        if device is None:
            if torch.cuda.is_available(): device = 'cuda'
            elif torch.backends.mps.is_available(): device = 'mps'
            else: device = 'cpu'
        self.device = device
        logging.info(f"Using device: {self.device}")

        self.model = FULL_MODEL_REGISTRY.build(
            model_name,
            modalities=input_modalities,
            output_modalities=[output_modality],
            pretrained=pretrained,
            standardize=standardize,
        ).to(self.device).eval()
        
        PRINT_MODEL_PARAMS = False
        if PRINT_MODEL_PARAMS:
            logging.info("Model structure:")
            for name, module in self.model.named_children():
                num_params = sum(p.numel() for p in module.parameters())
                logging.info(f"  {name}: {num_params:,} params ({num_params/1e6:.2f}M)")
                # Go one level deeper for tokenizer
                if hasattr(module, 'named_children'):
                    for sub_name, sub_module in module.named_children():
                        sub_params = sum(p.numel() for p in sub_module.parameters())
                        if sub_params > 0:
                            logging.info(f"    └─ {sub_name}: {sub_params:,} params ({sub_params/1e6:.2f}M)")
                            # Go one more level deeper
                            if hasattr(sub_module, 'named_children'):
                                for subsub_name, subsub_module in sub_module.named_children():
                                    subsub_params = sum(p.numel() for p in subsub_module.parameters())
                                    if subsub_params > 0:
                                        logging.info(f"      └─ {subsub_name}: {subsub_params:,} params ({subsub_params/1e6:.2f}M)")
            
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M), {trainable_params:,} trainable")
            
            if self.device == 'cuda':
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logging.info(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            exit(0)

    def _append_coords_csv(self, tile_id, lat, lon):
        if self.coords_csv_path is None:
            return
        # Write empty string if values are not finite
        lat_val = float(lat) if np.isfinite(lat) else ""
        lon_val = float(lon) if np.isfinite(lon) else ""
        with self.coords_csv_path.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([tile_id, lat_val, lon_val])

    def save_output(self, tif_path, generated, ground_truth=None):
        tile_name = Path(tif_path).stem
        out_path = self.output_dir / f"{tile_name}.tif"

        if self.output_modality == "coords":
            coords_list = generated[self.output_modality]
            
            if coords_list and isinstance(coords_list[0], (list, tuple)) and len(coords_list[0]) == 2:
                pred_lat, pred_lon = coords_list[0][1], coords_list[0][0]
                output = np.array([pred_lat, pred_lon], dtype=np.float32)
                # Save predicted coordinates to CSV
                self._append_coords_csv(tile_name, pred_lat, pred_lon)
                
                if ground_truth is not None and self.metrics is not None:
                    gt_lat, gt_lon = ground_truth
                    pred_coords = torch.tensor([[pred_lat, pred_lon]], dtype=torch.float32, device=self.device)
                    gt_coords = torch.tensor([[gt_lat, gt_lon]], dtype=torch.float32, device=self.device)
                    self.metrics.update(pred_coords, gt_coords)
                    
                    distance_km = SpatialMetrics._haversine_km_torch(pred_coords, gt_coords).item()
                    # logging.info(f"Ground truth: lat={gt_lat:.4f}, lon={gt_lon:.4f}")
                    # logging.info(f"Predicted: lat={pred_lat:.4f}, lon={pred_lon:.4f}")
                    # logging.info(f"Distance error: {distance_km:.2f} km")
            else:
                logging.warning("Invalid coordinates generated - using NaN values")
                output = np.array([np.nan, np.nan], dtype=np.float32)
                # Still append a row with empty coord fields for traceability
                self._append_coords_csv(tile_name, np.nan, np.nan)

            output_dtype = np.float32
            output_data = output.reshape(2, 1, 1)

        else:
            output = generated[self.output_modality].cpu().squeeze().numpy()

            # model_output_tensor = generated[self.output_modality]
            # logging.info(f"DEBUG - Model output shape for {self.output_modality}: {model_output_tensor.shape}")
            # if model_output_tensor.dim() >= 3:
            #     num_bands = model_output_tensor.shape[-3] if model_output_tensor.dim() == 4 else model_output_tensor.shape[0]
            #     logging.info(f"DEBUG - Number of bands in model output: {num_bands}")

            if ground_truth is not None and self.metrics is not None:
                # logging.info(f"DEBUG - Ground truth shape: {ground_truth.shape}")
                if ground_truth.dim() >= 3:
                    gt_bands = ground_truth.shape[-3] if ground_truth.dim() == 4 else ground_truth.shape[0]
                    # logging.info(f"DEBUG - Number of bands in ground truth: {gt_bands}")

                pred_tensor = generated[self.output_modality].cpu()
                gt_tensor = ground_truth
                if pred_tensor.dim() == 3:  # Add batch dimension if needed
                    pred_tensor = pred_tensor.unsqueeze(0)
                if gt_tensor.dim() == 3:
                    gt_tensor = gt_tensor.unsqueeze(0)

                # Remove B10 (index 9, 0-based) from ground truth to match prediction
                if self.output_modality == "S2L1C" and gt_tensor.shape[1] == 13 and pred_tensor.shape[1] == 12:
                    gt_tensor = torch.cat([gt_tensor[:, :9, :, :], gt_tensor[:, 10:, :, :]], dim=1)
                    # logging.info(f"Removed B10 from ground truth S2L1C: {ground_truth.shape} -> {gt_tensor.shape}")
                            
                # Center crop ground truth to match prediction size
                if gt_tensor.shape[2:] != pred_tensor.shape[2:]:
                    crop_transform = T.CenterCrop(self.crop_size)
                    gt_tensor = crop_transform(gt_tensor)
                    # logging.info(f"Center cropped ground truth from {gt_tensor.shape} to match prediction size")

                # Make tensors contiguous to fix memory error
                pred_tensor = pred_tensor.contiguous()
                gt_tensor = gt_tensor.contiguous()

                if self.output_modality == "LULC":
                    unique_classes = torch.unique(gt_tensor)
                    max_class = unique_classes.max().item()
                    min_class = unique_classes.min().item()
                    # logging.info(f"DEBUG - LULC ground truth classes range: {min_class} to {max_class}")
    
                    # For LULC classification convert ground truth to one-hot
                    if gt_tensor.shape[1] == 1: 
                        num_classes = 10
                        gt_onehot = torch.nn.functional.one_hot(
                            gt_tensor.squeeze(1).long(), 
                            num_classes=num_classes
                        ).permute(0, 3, 1, 2).float()
                    else:
                        gt_onehot = gt_tensor 
                    
                    self.metrics.update_from_logits(pred_tensor, gt_onehot)
                    # logging.info(f"Updated classification metrics for {tile_name}")
                else:
                    # For other modalities use regression metrics
                    # Physical units
                    if self.output_modality == "S1RTC":
                        pred_tensor = 10 ** (pred_tensor / 10.0)
                        gt_tensor = 10 ** (gt_tensor / 10.0)
                    elif self.output_modality in ["S2L2A", "S2L1C"]:
                        pred_tensor = pred_tensor / 10000.0
                        gt_tensor = gt_tensor / 10000.0
                    
                    self.metrics.update(pred_tensor, gt_tensor)
                    # logging.info(f"Updated regression metrics for {tile_name}")

                # self.metrics.update(pred_tensor, gt_tensor)          
                # logging.info(f"Updated image metrics for {tile_name}")

            if self.output_modality == "S1RTC":
                output = np.clip(output, -50, 10)
                output_data = (10 ** (output / 10)).astype(np.float32)
                output_dtype = np.float32
            elif self.output_modality in ["S2L2A", "S2L1C"]:
                output_dtype = np.uint16
                output_data = np.clip(output, 0, 10000).astype(np.uint16)
            elif self.output_modality == "LULC":
                output_data = np.round(np.clip(output, 0, 255)).astype(np.uint8)
                output_dtype = np.uint8
            elif self.output_modality == "DEM":
                output_dtype = np.float32
                output_data = output.astype(np.float32)
            else:
                output_dtype = np.float32
                output_data = output.astype(np.float32)

        ref_meta = None
        if Path(tif_path).exists():
            with rasterio.open(tif_path) as src:
                ref_meta = src.meta.copy()
        else:
            ref_meta = default_meta_if_missing()

        nodata_map = {np.uint8: 255, np.uint16: 65535, np.float32: -9999.0}
        write_tif(out_path, output_data, ref_meta_or_path=ref_meta, dtype=output_dtype, nodata=nodata_map.get(output_dtype))

        # logging.info(f"Saved coordinates to: {out_path}")
        if self.output_modality == "coords":
            num_bands = output_data.shape[0] if output_data.ndim == 3 else 1
            lat_val = output_data[0, 0, 0] if num_bands >= 1 and not np.isnan(output_data[0, 0, 0]) else 'NaN'
            lon_val = output_data[1, 0, 0] if num_bands > 1 and not np.isnan(output_data[1, 0, 0]) else 'NaN'
            # logging.info(f"  Band 1 (lat): {lat_val}")
            # logging.info(f"  Band 2 (lon): {lon_val}")

    def process_all(self, batch_size=1, num_workers=0, max_files=None):
        logging.info(f"Starting generation of {self.output_modality} from {self.input_modalities}")
        
        def simple_collate_fn(batch):
            batch = [item for item in batch if item is not None]
            if not batch:
                return None, None
            
            tif_path, input_dict = batch[0]  
            return tif_path, input_dict

        ground_truth_data = {}
        
        if self.output_modality == "coords":
            coords_gt_file = self.root / "terramind_inputs" / "coords" / "tile_to_coords.json"
            if coords_gt_file.exists():
                import json
                with open(coords_gt_file, 'r') as f:
                    coords_data = json.load(f)
                    for tile, coord_str in coords_data.items():
                        # Parse "lat=-22.25 lon=-70.25" format
                        try:
                            parts = coord_str.replace('lat=', '').replace('lon=', '').split()
                            if len(parts) == 2:
                                lat, lon = float(parts[0]), float(parts[1])
                                ground_truth_data[tile] = (lat, lon)
                        except (ValueError, IndexError):
                            logging.warning(f"Could not parse coordinates for {tile}: {coord_str}")
                logging.info(f"Loaded ground truth coordinates for {len(ground_truth_data)} tiles")
        
        elif self.output_modality in ["DEM", "S1RTC", "S2L2A", "S2L1C", "LULC"]:
            gt_dir = self.root / "terramind_inputs" / self.output_modality
            if gt_dir.exists():
                for gt_file in gt_dir.glob("*.tif"):
                    tile_name = gt_file.stem
                    try:
                        with rasterio.open(gt_file) as src:
                            gt_data = src.read()
                            ground_truth_data[tile_name] = torch.tensor(gt_data, dtype=torch.float32)
                    except Exception as e:
                        logging.warning(f"Could not load ground truth for {tile_name}: {e}")
                logging.info(f"Loaded ground truth images for {len(ground_truth_data)} tiles")

        # Handle case where coords is the only modality
        non_coords_modalities = [mod for mod in self.input_modalities if mod != "coords"]
        if non_coords_modalities:
            # Use the first non-coords modality to get file list
            sample_modality = non_coords_modalities[0]
            sample_files = sorted(self.input_dirs[sample_modality].glob("*.tif"))
        else:
            # Only coords modality - use coordinate file keys to determine which files to process
            if self.coords_file and self.coords_file.exists():
                import json
                with open(self.coords_file, 'r') as f:
                    coords_data = json.load(f)
                # Create dummy file paths based on coordinate keys
                sample_files = [Path(f"{key}.tif") for key in coords_data.keys()]
            else:
                logging.error("No coordinate file found and only 'coords' modality specified")
                raise ValueError("No coordinate file found and only 'coords' modality specified")
        
        if max_files:
            sample_files = sample_files[:max_files]
            logging.info(f"Limited to {max_files} files for processing")

        dataset = CopGenDataset(
            sample_files, 
            self.input_modalities, 
            self.input_dirs,
            self.crop_size, 
            self.device,
            coords_file=self.coords_file,
            input_overrides=self.input_overrides
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=simple_collate_fn)
        
        processed_count = 0
        metrics_updated = False 
        
        for tif_path, input_dict in tqdm(loader, desc=f"Generating {self.output_modality}"):
            if tif_path is None: 
                continue
                
            with torch.no_grad():
                generated = self.model(input_dict, timesteps=self.timesteps, verbose=False)
            
            tile_name = Path(tif_path).stem
            ground_truth = ground_truth_data.get(tile_name, None)
            
            if ground_truth is not None:
                metrics_updated = True
            
            self.save_output(tif_path, generated, ground_truth)
            processed_count += 1

        if self.metrics is not None and processed_count > 0 and metrics_updated:
            logging.info("=" * 60)
            logging.info("FINAL EVALUATION METRICS")
            logging.info("=" * 60)
            
            metrics_summary = self.metrics.summary()
            
            if self.output_modality == "coords":
                logging.info(f"Processed {processed_count} coordinate predictions")
                logging.info(f"Mean distance error: {metrics_summary['mean_km']:.2f} km")
                logging.info(f"Median distance error: {metrics_summary['median_km']:.2f} km")
                logging.info(f"90th percentile error: {metrics_summary['p90_km']:.2f} km")
                logging.info(f"95th percentile error: {metrics_summary['p95_km']:.2f} km")
                
                # Accuracy at different radii
                for radius in [1, 10, 50, 100, 500]:
                    acc_key = f'acc@{radius}km'
                    if acc_key in metrics_summary:
                        logging.info(f"Accuracy within {radius}km: {metrics_summary[acc_key]:.1%}")
            elif self.output_modality == "LULC":
                logging.info(f"Processed {processed_count} LULC predictions")
                for metric_key in metrics_summary.keys():
                    if 'overall_top' in metric_key:
                        logging.info(f"{str(metric_key).capitalize()} Accuracy: {metrics_summary[metric_key]:.3f}")
                logging.info(f"Mean IoU: {metrics_summary['mean_iou']:.3f}")
                logging.info(f"Mean F1-Score: {metrics_summary['mean_f1']:.3f}")
                logging.info(f"Frequency-Weighted IoU: {metrics_summary['fw_iou']:.3f}")
            else:
                logging.info(f"Processed {processed_count} image predictions")
                logging.info(f"Mean Absolute Error: {metrics_summary['mae']:.4f}")
                logging.info(f"Root Mean Square Error: {metrics_summary['rmse']:.4f}")
                logging.info(f"Structural Similarity (SSIM): {metrics_summary['ssim']:.4f}")
                logging.info(f"Peak Signal-to-Noise Ratio: {metrics_summary['psnr']:.2f} dB")
            
            logging.info("=" * 60)
        elif processed_count > 0:
            logging.info(f"Processed {processed_count} predictions but no ground truth data was available for evaluation")
        else:
            logging.info("No files were processed")

# if __name__ == "__main__":
#     ROOT = "/your/path/here"
#     generator = TerraMindGenerator(
#         input_modalities=["S1RTC", "LULC"],
#         output_modality="S2L2A",
#         root=ROOT,
#         crop_size=(256, 256),
#         timesteps=10,
#         device="cuda",
#         pretrained=True,
#         standardize=True,
#     )
#     generator.process_all(batch_size=1, num_workers=4, max_files=10)