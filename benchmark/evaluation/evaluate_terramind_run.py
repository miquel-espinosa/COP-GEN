from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import logging
import csv

_DEPS = None

def _get_deps():
	"""Lazy import heavy deps so we can skip evaluation without importing torch/rasterio/etc."""
	global _DEPS
	if _DEPS is None:
		import numpy as np  # type: ignore
		import torch  # type: ignore
		import rasterio  # type: ignore
		from tqdm import tqdm  # type: ignore

		from benchmark.evaluation import ImageRegressionMetrics, SpatialMetrics, SegmentationMetrics  # type: ignore
		from ddm.pre_post_process_data import name_list  # type: ignore

		_DEPS = (np, torch, rasterio, tqdm, ImageRegressionMetrics, SpatialMetrics, SegmentationMetrics, name_list)
	return _DEPS


def _center_crop_to_match(gt: Any, pr: Any) -> Any:
	np, _, _, _, _, _, _, _ = _get_deps()
	if gt.shape == pr.shape:
		return gt
	_, h, w = pr.shape
	top = max(0, (gt.shape[1] - h) // 2)
	left = max(0, (gt.shape[2] - w) // 2)
	return gt[:, top:top + h, left:left + w]


def _load_coords_from_tif(tif_path: Path) -> Any | None:
	np, torch, rasterio, _, _, _, _, _ = _get_deps()
	try:
		with rasterio.open(tif_path) as src:
			arr = src.read().astype(np.float32)  # [C,H,W] expected C=2
			if arr.ndim != 3 or arr.shape[0] < 2:
				logging.error(f"Invalid array shape: {arr.shape}")
				return None
			lat = float(arr[0, 0, 0])
			lon = float(arr[1, 0, 0])
			return torch.tensor([[lat, lon]], dtype=torch.float32)
	except Exception:
		logging.error(f"Error loading coords from {tif_path}")
		return None


def _infer_modality_from_run_dir(run_root: Path) -> str:
	valid_modalities = {"DEM", "S1RTC", "S2L1C", "S2L2A", "LULC", "coords"}
	name = run_root.name
	if "output_" not in name:
		raise ValueError(f"Cannot infer modality: expected '_output_' in run directory name '{name}'")
	try:
		if "input_" in name:
			after_input = name.split("input_", 1)[1]
			_, after_output = after_input.split("_output_", 1)
		else:
			after_output = name.split("_output_", 1)[1]
		modality_part = after_output.split("_seed_", 1)[0]
	except Exception as e:
		raise ValueError(f"Failed to parse modality from run directory name '{name}': {e}")

	modality_key = modality_part.strip("_")
	normalized = {m.lower(): m for m in valid_modalities}.get(modality_key.lower())
	if normalized is None:
		raise ValueError(f"Parsed modality '{modality_key}' not in supported set {sorted(valid_modalities)}")
	return normalized

def _evaluate_images_stacked(gt_root: Path, gen_root: Path, modality: str, per_tile: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
	np, torch, rasterio, tqdm, ImageRegressionMetrics, _, _, _ = _get_deps()
	imgm = ImageRegressionMetrics()
	imgm_per_tile: Dict[str, ImageRegressionMetrics] = {} if per_tile else {}

	# Gather generated stacked TIFs
	pr_files: List[Path] = sorted((gen_root).glob("*.tif"))
	if not pr_files:
		logging.warning(f"No generated TIFs found in {gen_root}")

	for pr_file in tqdm(pr_files, desc="Evaluating images"):
		tile = pr_file.stem
		gt_file = gt_root / "terramind_inputs" / modality / f"{tile}.tif"
		if not gt_file.exists():
			# Skip if there is no GT for this tile
			logging.warning(f"No GT {modality} found for {tile}")
			continue
		try:
			with rasterio.open(pr_file) as pr_src, rasterio.open(gt_file) as gt_src:
				pr = pr_src.read().astype(np.float32)  # [C,H,W]
				gt = gt_src.read().astype(np.float32)  # [C,H,W] (may differ)

				# Check for NaN values
				if np.isnan(pr).any():
					logging.warning(f"NaN values detected in prediction for {tile} ({modality})")
				if np.isnan(gt).any():
					logging.warning(f"NaN values detected in ground truth for {tile} ({modality})")

				# Handle S2L1C B10 removal if GT has 13 bands and prediction has 12
				if modality == "S2L1C" and gt.shape[0] == 13 and pr.shape[0] == 12:
					# Remove band index 9 (B10)
					gt = np.concatenate([gt[:9, ...], gt[10:, ...]], axis=0)

				# Physical units conversion (match CopGen)
				if modality in ("S2L2A", "S2L1C"):
					pr = pr / 10000.0
					gt = gt / 10000.0
				elif modality == "S1RTC":
					# Files are stored in linear space. Convert to dB for metrics like in CopGen.
					pr = 10.0 * np.log10(pr)
					gt = 10.0 * np.log10(gt)

			# Center-crop GT to match PR spatially if needed
			gt = _center_crop_to_match(gt, pr)

			# Ensure [B,C,H,W]
			pr_t = torch.from_numpy(pr).unsqueeze(0).contiguous()
			gt_t = torch.from_numpy(gt).unsqueeze(0).contiguous()
			imgm.update(pr_t, gt_t)

			if per_tile:
				if tile not in imgm_per_tile:
					imgm_per_tile[tile] = ImageRegressionMetrics()
				imgm_per_tile[tile].update(pr_t, gt_t)
		except Exception as e:
			raise e

	if per_tile:
		return imgm.summary(), {t: m.summary() for t, m in imgm_per_tile.items()}
	return imgm.summary()


def _evaluate_lulc_stacked(gt_root: Path, gen_root: Path, per_tile: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
	np, torch, rasterio, _, _, _, SegmentationMetrics, _ = _get_deps()
	# Mirror generator behavior: treat labels as 0..9 index-coded; avoid LUT remapping
	num_classes = 10
	segm = SegmentationMetrics(num_classes=num_classes, topk=3)
	segm_per_tile: Dict[str, SegmentationMetrics] = {} if per_tile else {}

	pr_files: List[Path] = sorted((gen_root).glob("*.tif"))
	if not pr_files:
		logging.warning(f"No generated TIFs found in {gen_root}")

	for pr_file in pr_files:
		tile = pr_file.stem
		gt_file = gt_root / "terramind_inputs" / "LULC" / f"{tile}.tif"
		if not gt_file.exists():
			logging.warning(f"No GT LULC found for {tile}")
		try:
			with rasterio.open(pr_file) as pr_src, rasterio.open(gt_file) as gt_src:
				# PR may be stacked logits [C,H,W] or a single-band label map [1,H,W]
				pr = pr_src.read().astype(np.float32)  # [C,H,W]
				gt = gt_src.read().astype(np.float32)  # [C,H,W]

			# Check for NaN values
			if np.isnan(pr).any():
				logging.warning(f"NaN values detected in prediction for {tile} (LULC)")
			if np.isnan(gt).any():
				logging.warning(f"NaN values detected in ground truth for {tile} (LULC)")

			# If GT has multiple bands, use the first band (labels) for evaluation
			if gt.shape[0] > 1:
				gt = gt[:1, ...]
			# Align spatial dims
			gt = _center_crop_to_match(gt, pr)

			# Build GT one-hot directly from index-coded labels (0..9)
			gt_vals = gt[0].astype(np.int64)
			gt_vals = np.clip(gt_vals, 0, num_classes - 1)  # defensive clamp
			gt_onehot = torch.nn.functional.one_hot(torch.from_numpy(gt_vals), num_classes=num_classes).permute(2, 0, 1).unsqueeze(0).float()

			# Predictions:
			# - If multi-band: treat as logits [C,H,W]
			# - If single-band: treat as index-coded labels and convert to one-hot "logits"
			if pr.shape[0] > 1:
				pr_logits = torch.from_numpy(pr).unsqueeze(0).float()
			else:
				pr_vals = pr[0].astype(np.int64)
				pr_vals = np.clip(pr_vals, 0, num_classes - 1)  # defensive clamp
				pr_logits = torch.nn.functional.one_hot(torch.from_numpy(pr_vals), num_classes=num_classes).permute(2, 0, 1).unsqueeze(0).float()

			segm.update_from_logits(pr_logits, gt_onehot)

			if per_tile:
				if tile not in segm_per_tile:
					segm_per_tile[tile] = SegmentationMetrics(num_classes=num_classes, topk=3)
				segm_per_tile[tile].update_from_logits(pr_logits, gt_onehot)
		except Exception as e:
			raise e

	if per_tile:
		return segm.summary(), {t: m.summary() for t, m in segm_per_tile.items()}
	return segm.summary()


def _evaluate_coords(gt_root: Path, gen_root: Path, per_tile: bool = False) -> Dict[str, float] | Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
	_, torch, _, _, _, SpatialMetrics, _, _ = _get_deps()
	sm = SpatialMetrics(radii_km=(1, 10, 50, 100, 500))
	sm_per_tile: Dict[str, SpatialMetrics] = {} if per_tile else {}
	pr_files: List[Path] = sorted((gen_root).glob("*.tif"))
	if not pr_files:
		logging.warning(f"No generated TIFs found in {gen_root}")
	for pr_file in pr_files:
		tile = pr_file.stem
		gt_file = gt_root / "terramind_inputs" / "coords" / f"{tile}.tif"
		if not gt_file.exists():
			logging.warning(f"No GT coords found for {tile}")
		pred = _load_coords_from_tif(pr_file)
		target = _load_coords_from_tif(gt_file)
		if pred is None or target is None:
			logging.warning(f"No pred or target found for {tile}")
		else:
			# Check for NaN values
			if torch.isnan(pred).any():
				print(f"WARNING: NaN values detected in prediction coords for {tile}")
			if torch.isnan(target).any():
				print(f"WARNING: NaN values detected in ground truth coords for {tile}")
		sm.update(pred, target)
		if per_tile:
			if tile not in sm_per_tile:
				sm_per_tile[tile] = SpatialMetrics(radii_km=(1, 10, 50, 100, 500))
			sm_per_tile[tile].update(pred, target)
	if per_tile:
		return sm.summary(), {t: m.summary() for t, m in sm_per_tile.items()}
	return sm.summary()


def _print_metrics(metrics: Dict[str, Dict[str, float]]):
	_, _, _, _, _, _, _, name_list = _get_deps()
	print("=" * 60)
	print("FINAL EVALUATION METRICS")
	print("=" * 60)
	for modality, metrics_summary in metrics.items():
		print(f"\n[{modality}]")
		if modality == "lat_lon":
			print(f"  Median distance error: {metrics_summary['median_km']:.2f} km")
			print(f"  Mean distance error: {metrics_summary['mean_km']:.2f} km")
			print(f"  Std distance error: {metrics_summary['std_km']:.2f} km")
			print(f"  RMSE distance error: {metrics_summary['rmse_km']:.2f} km")
			print(f"  90th percentile error: {metrics_summary['p90_km']:.2f} km")
			print(f"  95th percentile error: {metrics_summary['p95_km']:.2f} km")
			print(f"  99th percentile error: {metrics_summary['p99_km']:.2f} km")
			for k in list(metrics_summary.keys()):
				if 'acc' in k:
					print(f"  Accuracy within {k}: {metrics_summary[k]:.1%}")
		elif modality == "LULC":
			for metric_key in metrics_summary.keys():
				if 'overall_top' in metric_key:
					print(f"  {str(metric_key).capitalize()} Accuracy: {metrics_summary[metric_key]:.3f}")
			print(f"  Mean IoU: {metrics_summary['mean_iou']:.3f}")
			print(f"  Mean F1-Score: {metrics_summary['mean_f1']:.3f}")
			print(f"  Frequency-Weighted IoU: {metrics_summary['fw_iou']:.3f}")
			print(f"  Per-class metrics:")
			num_classes = len(metrics_summary['per_class_iou'])
			for i in range(num_classes):
				class_name = name_list["LULC"][i]
				iou = metrics_summary['per_class_iou'][i]
				precision = metrics_summary['per_class_precision'][i]
				recall = metrics_summary['per_class_recall'][i]
				f1 = metrics_summary['per_class_f1'][i]
				print(f"    {class_name:20s}: IoU={iou:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
		else:
			print(f"  Mean Absolute Error: {metrics_summary['mae']:.4f}")
			print(f"  Root Mean Square Error: {metrics_summary['rmse']:.4f}")
			print(f"  Structural Similarity (SSIM): {metrics_summary['ssim']:.4f}")
			print(f"  Peak Signal-to-Noise Ratio: {metrics_summary['psnr']:.2f} dB")
			if 'per_band' in metrics_summary and isinstance(metrics_summary['per_band'], dict):
				print("  Per-band metrics:")
				for bname in sorted(metrics_summary['per_band'].keys()):
					b = metrics_summary['per_band'][bname]
					print(f"    {bname:10s}: MAE={b.get('mae', 0.0):.4f}, RMSE={b.get('rmse', 0.0):.4f}, SSIM={b.get('ssim', 0.0):.4f}, PSNR={b.get('psnr', 0.0):.2f}")
	print("=" * 60)


def evaluate_terramind_run(root: Path, run_root: Path, modality: str | None = None, verbose: bool = False, per_tile: bool = False) -> Dict[str, Dict[str, float]]:
	"""
	Evaluate a TerraMind run outputs against ground truth found in root/terramind_inputs.

	Args:
	    root: Path to TerraMind project root (contains terramind_inputs/)
	    run_root: Path to run folder (contains generations/)
	    modality: Optional override; if None, inferred from run_root name (expects pattern with '_output_<MODALITY>_')
	    per_tile: If True, also compute and write per-tile metrics
	Returns:
	    Dict of modality -> metrics summary (uses 'lat_lon' key for coords)
	Side-effects:
	    Writes a textual report to run_root / 'output_metrics.txt'
	    If per_tile, also writes run_root / 'output_metrics_per_tile.csv'
	"""
	gen_root = run_root / "generations"
	if not gen_root.exists():
		raise FileNotFoundError(f"Missing generations folder: {gen_root}")

	out_txt = run_root / "output_metrics.txt"
	out_csv = run_root / "output_metrics_per_tile.csv"
	if out_txt.exists() and out_csv.exists():
		msg = f"Skipping evaluation; metrics already exist at {out_txt} and {out_csv}"
		logging.info(msg)
		if verbose:
			print(msg)
		return {}

	valid_modalities = {"DEM", "S1RTC", "S2L1C", "S2L2A", "LULC", "coords"}
	if modality is None:
		modality = _infer_modality_from_run_dir(run_root)
	else:
		modality = modality.strip()
	if modality not in valid_modalities:
		raise ValueError(f"Unsupported modality '{modality}' (expected one of {sorted(valid_modalities)})")

	per_tile_maps: Dict[str, Dict[str, Dict[str, float]]] = {}

	if modality == "coords":
		if per_tile:
			metrics, per_tile_map = _evaluate_coords(root, gen_root, per_tile=True)
			per_tile_maps["lat_lon"] = per_tile_map
		else:
			metrics = _evaluate_coords(root, gen_root, per_tile=False)
		all_metrics: Dict[str, Dict[str, float]] = {"lat_lon": metrics}
	elif modality == "LULC":
		if per_tile:
			metrics, per_tile_map = _evaluate_lulc_stacked(root, gen_root, per_tile=True)
			per_tile_maps["LULC"] = per_tile_map
		else:
			metrics = _evaluate_lulc_stacked(root, gen_root, per_tile=False)
		all_metrics = {"LULC": metrics}
	else:
		if per_tile:
			metrics, per_tile_map = _evaluate_images_stacked(root, gen_root, modality, per_tile=True)
			per_tile_maps[modality] = per_tile_map
		else:
			metrics = _evaluate_images_stacked(root, gen_root, modality, per_tile=False)
		all_metrics = {modality: metrics}

	# Write report
	with open(out_txt, "w") as f:
		f.write("Evaluation metrics for generated modalities (TerraMind)\n")
		for mod, s in all_metrics.items():
			f.write(f"[{mod}] ")
			items: List[str] = []
			for k, v in s.items():
				if k == 'per_band' and isinstance(v, dict):
					for bname, bd in v.items():
						items.append(f"band:{bname}:mae={bd.get('mae', 0.0):.4f}")
						items.append(f"band:{bname}:rmse={bd.get('rmse', 0.0):.4f}")
						items.append(f"band:{bname}:ssim={bd.get('ssim', 0.0):.4f}")
						items.append(f"band:{bname}:psnr={bd.get('psnr', 0.0):.2f}")
				else:
					items.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
			f.write(" ".join(items) + "\n")

	# Optional per-tile CSV (mirror CopGen behavior)
	if per_tile and per_tile_maps:
		rows: List[Dict[str, object]] = []

		def _flatten_metrics_dict(d: Dict[str, object]) -> Dict[str, object]:
			flat: Dict[str, object] = {}
			for k, v in d.items():
				if k == 'per_band' and isinstance(v, dict):
					for bname, bd in v.items():
						flat[f"band:{bname}:mae"] = float(bd.get('mae', 0.0))
						flat[f"band:{bname}:rmse"] = float(bd.get('rmse', 0.0))
						flat[f"band:{bname}:ssim"] = float(bd.get('ssim', 0.0))
						flat[f"band:{bname}:psnr"] = float(bd.get('psnr', 0.0))
				else:
					flat[k] = v
			return flat

		for modality_key, per_tile_map in per_tile_maps.items():
			for tile_id, md in per_tile_map.items():
				row = {'tile_id': tile_id, 'modality': modality_key}
				row.update(_flatten_metrics_dict(md))
				rows.append(row)

		if rows:
			all_keys: List[str] = ['tile_id', 'modality']
			dynamic_keys = sorted({k for r in rows for k in r.keys()} - set(all_keys))
			header = all_keys + dynamic_keys
			out_csv = run_root / "output_metrics_per_tile.csv"
			with open(out_csv, "w", newline="") as fcsv:
				writer = csv.DictWriter(fcsv, fieldnames=header)
				writer.writeheader()
				for r in rows:
					writer.writerow({k: r.get(k, "") for k in header})

	if verbose:
		_print_metrics(all_metrics)
	return all_metrics


if __name__ == "__main__":
	import argparse
	p = argparse.ArgumentParser(description="Evaluate TerraMind run outputs against ground truth in terramind_inputs")
	p.add_argument("--dataset-dir", type=str, required=True, help="Path to TerraMind root (contains terramind_inputs/)")
	p.add_argument("--run-dir", type=str, required=True, help="Path to run root (with generations/)")
	p.add_argument("--modality", type=str, required=False, choices=["DEM", "S1RTC", "S2L1C", "S2L2A", "LULC", "coords"], help="Optional override for output modality; if omitted it is inferred from the run directory name")
	p.add_argument("--per-tile", action="store_true", help="Also compute and write per-tile metrics (default: False)")
	p.add_argument("--verbose", action="store_true")
	args = p.parse_args()
	evaluate_terramind_run(Path(args.dataset_dir), Path(args.run_dir), args.modality, verbose=args.verbose, per_tile=args.per_tile)


