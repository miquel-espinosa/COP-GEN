from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch

from visualisations.visualise_bands import visualise_bands
from visualisations.merge_visualisations import merge_visualisations
from importlib.machinery import SourceFileLoader


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(
		description="Render visualisations for an existing COP-GEN sample from saved generations.\n"
		            "Example: python plot_sample.py --sample-root /path/to/tile/samples/sample_1",
	)
	p.add_argument(
		"--sample-root",
		type=str,
		required=True,
		help="Path to a sample root directory (contains generations/ and visualisations/)",
	)
	p.add_argument(
		"--config",
		type=str,
		required=True,
		help="Path to Copgen model config .py (provides get_config())",
	)
	p.add_argument(
		"--milestone",
		type=int,
		default=0,
		help="Milestone/index to use for visualisation subfolder naming (visualisations-<milestone>)",
	)
	p.add_argument(
		"--view-histogram",
		action="store_true",
		help="If set, overlay per-band histograms (only effective when GT inputs are provided; here recon-only)",
	)
	p.add_argument(
		"--resize",
		type=int,
		default=None,
		help="Optional square resize for visual outputs. If not set, inferred as max tile size across modalities.",
	)
	return p.parse_args()


def _read_single_band(path: Path) -> np.ndarray:
	with rasterio.open(path) as src:
		arr = src.read(1)  # single-band
	return arr.astype(np.float32)


def _find_s2_product_dir(core_dir: Path) -> Optional[Path]:
	candidates = list(core_dir.rglob("B*.tif"))
	if not candidates:
		return None
	# choose directory with most S2 band files
	parent_to_files: Dict[Path, List[Path]] = {}
	for p in candidates:
		parent_to_files.setdefault(p.parent, []).append(p)
	product_dir = max(parent_to_files.items(), key=lambda kv: len(kv[1]))[0]
	return product_dir


def _find_s1_product_dir(core_dir: Path) -> Optional[Path]:
	# Find vv/vh anywhere and return their parent directory
	for name in ("vv.tif", "VV.tif", "vh.tif", "VH.tif"):
		found = list(core_dir.rglob(name))
		if found:
			return found[0].parent
	return None


def _load_dem(core_dir: Path) -> Optional[torch.Tensor]:
	dem_files = list(core_dir.rglob("DEM.tif"))
	if not dem_files:
		return None
	arr = _read_single_band(dem_files[0])[np.newaxis, ...]  # [1,H,W]
	return torch.from_numpy(arr).unsqueeze(0).float()


def _load_lulc(core_dir: Path) -> Optional[torch.Tensor]:
	# Pick first tif (there should be a single .tif product)
	tifs = [p for p in core_dir.rglob("*.tif")]
	if not tifs:
		return None
	arr = _read_single_band(sorted(tifs)[0])[np.newaxis, ...]  # [1,H,W]
	return torch.from_numpy(arr).unsqueeze(0).float()


def _infer_resize(modality_to_tensor: Dict[str, torch.Tensor]) -> Optional[int]:
	if not modality_to_tensor:
		return None
	try:
		return max(int(t.shape[-1]) for t in modality_to_tensor.values() if t is not None)
	except Exception:
		return None


def _modality_keys_for_base(base: str, all_keys: List[str]) -> List[str]:
	# Mirror CopgenBatchGenerator._modality_keys_for_base behaviour for relevant bases
	base_norm = base
	if base.lower() == "lat_lon":
		base_norm = "3d_cartesian_lat_lon"
	elif base.lower() == "timestamps":
		base_norm = "mean_timestamps"
	elif base.lower() == "cloud_mask":
		base_norm = "S2L1C_cloud_mask"
	# include keys containing this base_norm
	keys = [k for k in all_keys if base_norm in k]
	# Keep cloud_mask separate; include only when base explicitly matches S2L1C (not used here)
	if base_norm == "S2L1C":
		return [k for k in keys if "cloud_mask" not in k]
	return keys


def _load_band_arrays_from_product(product_dir: Path) -> Dict[str, np.ndarray]:
	# Map band name (e.g., "B02", "B8A", "vv", "vh", "DEM", "LULC") to 2D array
	band_map: Dict[str, np.ndarray] = {}
	for tif in sorted(product_dir.glob("*.tif")):
		name = tif.stem
		try:
			arr = _read_single_band(tif)
		except Exception:
			continue
		band_map[name] = arr
	return band_map


def main() -> None:
	args = parse_args()
	sample_root = Path(args.sample_root).expanduser().resolve()
	gen_root = sample_root / "generations"
	vis_root = sample_root / "visualisations"
	vis_root.mkdir(parents=True, exist_ok=True)

	if not gen_root.exists():
		raise FileNotFoundError(f"generations/ not found under sample root: {gen_root}")

	# Load model config to discover modality keys and band groupings
	cfg_mod = SourceFileLoader("copgen_cfg", str(Path(args.config))).load_module()
	model_config = cfg_mod.get_config()
	all_keys: List[str] = list(model_config.all_modality_configs.keys())
	# Access bands and img_input_resolution for each key via model_config
	def bands_for_key(k: str) -> List[str]:
		return list(model_config.all_modality_configs[k].bands)
	def size_for_key(k: str) -> Tuple[int, int]:
		h, w = model_config.all_modality_configs[k].img_input_resolution
		return int(h), int(w)

	# Build reconstructions per expanded key (as [1,C,H,W] tensors)
	key_to_tensor: Dict[str, torch.Tensor] = {}

	core_s2l1c = gen_root / "Core-S2L1C"
	if core_s2l1c.exists():
		product_dir = _find_s2_product_dir(core_s2l1c)
		if product_dir is not None:
			band_arrays = _load_band_arrays_from_product(product_dir)
			for key in _modality_keys_for_base("S2L1C", all_keys):
				bnames = bands_for_key(key)
				if not all(b in band_arrays for b in bnames):
					continue
				arrs = [band_arrays[b] for b in bnames]
				stack = np.stack(arrs, axis=0)
				key_to_tensor[key] = torch.from_numpy(stack).unsqueeze(0).float()

	core_s2l2a = gen_root / "Core-S2L2A"
	if core_s2l2a.exists():
		product_dir = _find_s2_product_dir(core_s2l2a)
		if product_dir is not None:
			band_arrays = _load_band_arrays_from_product(product_dir)
			for key in _modality_keys_for_base("S2L2A", all_keys):
				bnames = bands_for_key(key)
				if not all(b in band_arrays for b in bnames):
					continue
				arrs = [band_arrays[b] for b in bnames]
				stack = np.stack(arrs, axis=0)
				key_to_tensor[key] = torch.from_numpy(stack).unsqueeze(0).float()

	core_s1rtc = gen_root / "Core-S1RTC"
	if core_s1rtc.exists():
		product_dir = _find_s1_product_dir(core_s1rtc)
		if product_dir is not None:
			band_arrays = _load_band_arrays_from_product(product_dir)
			for key in _modality_keys_for_base("S1RTC", all_keys):
				bnames = bands_for_key(key)
				if not all(b in band_arrays for b in bnames):
					continue
				arrs = [band_arrays[b] for b in bnames]
				stack = np.stack(arrs, axis=0)
				key_to_tensor[key] = torch.from_numpy(stack).unsqueeze(0).float()

	core_dem = gen_root / "Core-DEM"
	if core_dem.exists():
		# DEM is usually a single key; still follow config grouping for consistency
		# Load actual DEM array from file directly
		dem_t = _load_dem(core_dem)
		if dem_t is not None:
			for key in _modality_keys_for_base("DEM", all_keys):
				# Expect single band named "DEM" in config
				key_to_tensor[key] = dem_t

	core_lulc = gen_root / "Core-LULC"
	if core_lulc.exists():
		lulc_t = _load_lulc(core_lulc)
		if lulc_t is not None:
			for key in _modality_keys_for_base("LULC", all_keys):
				key_to_tensor[key] = lulc_t

	if not key_to_tensor:
		print(f"No supported modalities found under {gen_root}")
		return

	# Decide resize
	if args.resize is not None:
		resize = int(args.resize)
	else:
		# Use config's max img_input_resolution across keys we will visualise
		resize = max(int(size_for_key(k)[1]) for k in key_to_tensor.keys())

	# Render visualisations per expanded key (reconstruction-only mode)
	for key, tensor in key_to_tensor.items():
		save_dir = vis_root / key
		save_dir.mkdir(parents=True, exist_ok=True)
		visualise_bands(
			inputs=None,
			reconstructions=tensor,
			save_dir=save_dir,
			milestone=int(args.milestone),
			satellite_type=key,
			view_histogram=bool(args.view_histogram),
			resize=resize,
			input_frame_color=None,
			recon_frame_color=(1.0, 0.3, 0.2),
		)

	# Optionally create merged visualisations grid
	try:
		merge_visualisations(results_path=str(vis_root), verbose=False, overwrite=False)
	except Exception:
		# Non-fatal if mapping doesn't match expected modality names
		pass


if __name__ == "__main__":
	main()


