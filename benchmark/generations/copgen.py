from __future__ import annotations

import shutil
import logging
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
from tqdm import tqdm
import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from benchmark.io.tiff import write_tif
from visualisations.visualise_bands import visualise_bands
from visualisations.merge_visualisations import merge_visualisations
from utils import set_seed
from datasets import _LatLonMemmapCache
import csv


def _center_window(src: rasterio.io.DatasetReader, size_hw: Tuple[int, int]) -> Window:
    h, w = src.height, src.width
    ch, cw = size_hw
    top = max(0, (h - ch) // 2)
    left = max(0, (w - cw) // 2)
    return Window(left, top, min(cw, w), min(ch, h))


def _read_band_window(path: Path, size_hw: Tuple[int, int]) -> np.ndarray:
    with rasterio.open(path) as src:
        win = _center_window(src, size_hw)
        arr = src.read(window=win)
        return arr.astype(np.float32)

@dataclass
class CopgenPaths:
    dataset_root: Path

    @property
    def core(self) -> Dict[str, Path]:
        base = self.dataset_root
        return {
            "S2L2A": base / "Core-S2L2A",
            "S2L1C": base / "Core-S2L1C",
            "S1RTC": base / "Core-S1RTC",
            "DEM": base / "Core-DEM",
            "LULC": base / "Core-LULC",
        }



class CopgenBatchGenerator:
    def __init__(
        self,
        model_path: Path,
        config_path: Path,
        dataset_root: Optional[Path] = None,
        device: Optional[str] = None,
        seed: int = 1234,
        input_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.paths = CopgenPaths(dataset_root=dataset_root)
        self.seed = int(seed)
        # Map of base modality -> override specification (string). See CLI help for formats.
        self.input_overrides: Dict[str, str] = {k: v for k, v in (input_overrides or {}).items()}
        print(f"Input overrides: {self.input_overrides}")

        # Load CopGen config to learn modality groupings and input sizes
        self.cfg_mod = SourceFileLoader("copgen_cfg", str(config_path)).load_module()
        self.model_config = self.cfg_mod.get_config()

        # Lazy import model to avoid hard dep if env missing
        from libs.copgen import CopgenModel  # type: ignore
        set_seed(self.seed)
        self.model = CopgenModel(model_path=model_path, config_path=config_path, seed=self.seed)

        # Optional pre/post
        try:
            from ddm.pre_post_process_data import pre_process_data, post_process_data  # type: ignore
            self.pre_process_data = pre_process_data
            self.post_process_data = post_process_data
        except Exception:
            logging.warning("ddm.pre_post_process_data not available")
            raise Exception("ddm.pre_post_process_data not available")

        # Load tile->product indices for each modality base from pickle files located in Core-{base}
        self.indices: Dict[str, Dict[str, str]] = {}
        for base in ("S2L2A", "S2L1C", "S1RTC", "DEM", "LULC"):
            try:
                self.indices[base] = self._load_index(self.paths.core[base])
            except Exception as e:
                logging.warning(f"Could not load index for {base}: {e}")
                self.indices[base] = {}

        # Load scalar caches (already encoded vectors)
        lat_dir = self.paths.dataset_root / "3d_cartesian_lat_lon_cache"
        logging.info(f"Loading lat/lon cache from {lat_dir}")
        self.latlon_cache = _LatLonMemmapCache(str(lat_dir))
        ts_pkl = self.paths.dataset_root / "mean_timestamps_cache.pkl"
        self.timestamps_cache = None
        if ts_pkl.exists():
            try:
                logging.info(f"Loading timestamps cache from {ts_pkl}")
                with open(ts_pkl, "rb") as f:
                    self.timestamps_cache = pickle.load(f)
            except Exception as e:
                logging.error(f"Failed loading timestamps cache: {e}")
                raise Exception(f"Failed loading timestamps cache: {e}")

    def _norm_base(self, base: str) -> str:
        return base.strip().lower()

    def _has_override_for_base(self, base: str) -> bool:
        b = self._norm_base(base)
        return any(self._norm_base(k) == b for k in self.input_overrides.keys())

    def _get_override_for_base(self, base: str) -> Optional[str]:
        b = self._norm_base(base)
        for k, v in self.input_overrides.items():
            if self._norm_base(k) == b:
                return v
        return None

    def _load_index(self, core_dir: Path) -> Dict[str, str]:
        """
        Load a pickle file in the given Core-<MODALITY> folder mapping tile keys to product ids.
        Expected entry structure examples:
          ('626U_18L', 'S2A_MSIL2A_..._Txx') for S2
          ('626U_18L', 'S1B_IW_GRDH_..._rtc') for S1
          ('626U_18L', 'id') for DEM
          ('626U_17L', '626U_17L_2020') for LULC
        Returns a dict: tile_key -> product_id
        """
        pkls = list(core_dir.glob("*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"No pickle index found in {core_dir}")
        # Prefer a file named "metadata" or similar if present
        pkl_path = None
        for cand in pkls:
            if "meta" in cand.stem.lower() or "index" in cand.stem.lower():
                pkl_path = cand
                break
        if pkl_path is None:
            pkl_path = pkls[0]
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # Convert list of tuples to dict if needed
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        if isinstance(data, list):
            return {str(k): str(v) for (k, v) in data}
        raise ValueError(f"Unsupported index format in {pkl_path}")

    def _modality_keys_for_base(self, base: str) -> List[str]:
        # Normalize base names for convenience
        base_norm = base
        if base.lower() == "lat_lon":
            base_norm = "3d_cartesian_lat_lon"
        elif base.lower() == "timestamps":
            base_norm = "mean_timestamps"
        elif base.lower() == "cloud_mask":
            base_norm = "S2L1C_cloud_mask"

        keys = [k for k in self.model_config.all_modality_configs.keys() if base_norm in k]
        # Keep cloud_mask separate; include only when base explicitly matches
        if base_norm == "S2L1C":
            return [k for k in keys if "cloud_mask" not in k]
        return keys

    def _bands_for_key(self, key: str) -> List[str]:
        return list(self.model_config.all_modality_configs[key].bands)

    def _size_for_key(self, key: str) -> Tuple[int, int]:
        h, w = self.model_config.all_modality_configs[key].img_input_resolution
        return int(h), int(w)

    def _prepare_conditions_for_tile(self, tile_key: str, condition_bases: List[str]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Return (raw_conditions, preprocessed_conditions) for a single tile.
        Raw tensors are suitable for visualisation; preprocessed tensors feed the model.
        Shapes are [1, C, H, W] for image-like, [1,1,1,3] for scalar keys.
        """
        conditions_raw: Dict[str, torch.Tensor] = {}
        conditions_pre: Dict[str, torch.Tensor] = {}
        for base in condition_bases:
            keys_for_base = self._modality_keys_for_base(base)
            override_spec = self._get_override_for_base(base) if self._has_override_for_base(base) else None
            # If an override is provided for an unsupported base, fail fast
            if override_spec is not None and self._norm_base(base) not in ("lat_lon", "timestamps", "lulc", "dem", "cloud_mask", "s1rtc", "s2l1c", "s2l2a"):
                raise ValueError(f"Input override for base '{base}' is not supported. Supported with overrides: lat_lon, timestamps, DEM, LULC, cloud_mask, s1rtc, s2l1c, s2l2a.")
            for key in keys_for_base:
                # Handle scalar modality overrides (lat_lon, timestamps)
                if override_spec is not None and self._norm_base(base) in ("lat_lon", "timestamps"):
                    if "3d_cartesian_lat_lon" in key:
                        # Expect "lat,lon"
                        try:
                            lat_str, lon_str = [s.strip() for s in override_spec.split(",")]
                            lat_val = float(lat_str)
                            lon_val = float(lon_str)
                        except Exception:
                            raise ValueError(f"Invalid lat_lon override '{override_spec}'. Expected 'lat,lon'.")
                        # Pre-process from lat/lon degrees to encoded vector
                        enc = self.pre_process_data(np.array([lat_val, lon_val], dtype=np.float32), key, self.model_config.dataset, already_encoded=False)  # type: ignore
                        enc_t = torch.from_numpy(enc) if not torch.is_tensor(enc) else enc
                        # Ensure [1,1,1,3]
                        if enc_t.ndim == 1:
                            enc_t = enc_t.view(1, 1, 1, -1)
                        conditions_pre[key] = enc_t.to(self.device)
                        # For raw, decode back to lat/lon degrees
                        conditions_raw[key] = self.post_process_data(conditions_pre[key], key, self.model_config.dataset)  # type: ignore
                        continue
                    if "mean_timestamps" in key:
                        # Expect "dd-mm-yyyy"
                        from datetime import datetime
                        try:
                            date = datetime.strptime(override_spec.strip(), "%d-%m-%Y")
                        except Exception:
                            raise ValueError(f"Invalid timestamps override '{override_spec}'. Expected 'dd-mm-yyyy'.")
                        ts = np.array(date.timestamp(), dtype=np.float32)
                        enc = self.pre_process_data(ts, key, self.model_config.dataset, already_encoded=False)  # type: ignore
                        enc_t = torch.from_numpy(enc) if not torch.is_tensor(enc) else enc
                        if enc_t.ndim == 1:
                            enc_t = enc_t.view(1, 1, 1, -1)
                        conditions_pre[key] = enc_t.to(self.device)
                        conditions_raw[key] = self.post_process_data(conditions_pre[key], key, self.model_config.dataset)  # type: ignore
                        continue

                # Handle image-like overrides for DEM, LULC, cloud_mask (S2L1C_cloud_mask)
                if override_spec is not None and self._norm_base(base) in ("dem", "lulc", "cloud_mask", "s1rtc", "s2l1c", "s2l2a"):
                    # Determine desired size and (if needed) construct raw tensor appropriately
                    size_hw = self._size_for_key(key)
                    # LULC / cloud_mask: allow class-name OR tif path
                    if self._norm_base(base) == "lulc" and not str(override_spec).lower().endswith(".tif"):
                        # Create a constant-tile with the requested class name
                        try:
                            from ddm.pre_post_process_data import name_list, value_to_index_mapping
                        except Exception:
                            raise RuntimeError("ddm.pre_post_process_data is required for LULC class overrides")
                        target_name = override_spec.strip().lower()
                        # Build mapping from normalized name -> index
                        lulc_names = [n.strip().lower() for n in name_list["LULC"]]
                        if target_name not in lulc_names:
                            raise ValueError(f"Unknown LULC class '{override_spec}'. Available: {', '.join(name_list['LULC'])}")
                        class_index = lulc_names.index(target_name)
                        # Map one-hot index back to original raster code
                        index_to_orig = {idx: orig for orig, idx in value_to_index_mapping["LULC"].items()}
                        orig_val = index_to_orig[class_index]
                        h, w = size_hw
                        arr = np.full((1, h, w), orig_val, dtype=np.int64)
                        t_raw = torch.from_numpy(arr).to(torch.float32)
                        t_pre = self.pre_process_data(t_raw, key, self.model_config.dataset)  # type: ignore
                        t_pre = t_pre if torch.is_tensor(t_pre) else torch.from_numpy(t_pre)
                        conditions_raw[key] = t_raw.unsqueeze(0).to(self.device)
                        conditions_pre[key] = t_pre.unsqueeze(0).to(self.device)
                        continue
                    elif self._norm_base(base) == "cloud_mask" and not str(override_spec).lower().endswith(".tif"):
                        # Create a constant-tile with the requested cloud class name
                        try:
                            from ddm.pre_post_process_data import name_list, value_to_index_mapping
                        except Exception:
                            raise RuntimeError("ddm.pre_post_process_data is required for cloud_mask class overrides")
                        target_name = override_spec.strip().lower()
                        cloud_names = [n.strip().lower() for n in name_list["cloud_mask"]]
                        if target_name not in cloud_names:
                            raise ValueError(f"Unknown cloud_mask class '{override_spec}'. Available: {', '.join(name_list['cloud_mask'])}")
                        class_index = cloud_names.index(target_name)
                        # Map one-hot index back to original raster code
                        index_to_orig = {idx: orig for orig, idx in value_to_index_mapping["cloud_mask"].items()}
                        orig_val = index_to_orig[class_index]
                        h, w = size_hw
                        arr = np.full((1, h, w), orig_val, dtype=np.int64)
                        t_raw = torch.from_numpy(arr).to(torch.float32)
                        t_pre = self.pre_process_data(t_raw, key, self.model_config.dataset)  # type: ignore
                        t_pre = t_pre if torch.is_tensor(t_pre) else torch.from_numpy(t_pre)
                        conditions_raw[key] = t_raw.unsqueeze(0).to(self.device)
                        conditions_pre[key] = t_pre.unsqueeze(0).to(self.device)
                        continue
                    else:
                        # tif path provided for DEM / LULC / cloud_mask / s1rtc / s2l1c / s2l2a
                        tif_path = Path(override_spec)
                        if not tif_path.exists():
                            raise FileNotFoundError(f"Override path not found for {base}: {tif_path}")
                        arr = _read_band_window(tif_path, size_hw)
                        t_raw = torch.from_numpy(arr).float()
                        t_pre = self.pre_process_data(t_raw, key, self.model_config.dataset)  # type: ignore
                        t_pre = t_pre if torch.is_tensor(t_pre) else torch.from_numpy(t_pre)
                        conditions_raw[key] = t_raw.unsqueeze(0).to(self.device)
                        conditions_pre[key] = t_pre.unsqueeze(0).to(self.device)
                        continue

                # No overrides: fall back to dataset-driven behaviour
                # Scalar modalities: pre-post is deterministic and always direct
                if "3d_cartesian_lat_lon" in key:
                    if self.latlon_cache is None:
                        raise FileNotFoundError("lat/lon cache not available")
                    xyz = self.latlon_cache[(tile_key, -1, -1)]  # shape (3,)
                    xyz = np.asarray(xyz, dtype=np.float32).reshape(1, 1, 1, 3)
                    t_pre = self.pre_process_data(xyz, key, self.model_config.dataset, already_encoded=True)  # type: ignore
                    t_pre = torch.from_numpy(t_pre) if not torch.is_tensor(t_pre) else t_pre
                    conditions_pre[key] = t_pre.to(self.device)
                    # For raw visualisation, decode from 3D cartesian to lat/lon
                    conditions_raw[key] = self.post_process_data(conditions_pre[key], key, self.model_config.dataset)
                    continue
                if "mean_timestamps" in key:
                    if self.timestamps_cache is None:
                        raise FileNotFoundError("timestamps cache not available")
                    ts_vec = self.timestamps_cache.get(tile_key)
                    if ts_vec is None:
                        raise KeyError(f"No timestamp for {tile_key}")
                    ts = np.asarray(ts_vec, dtype=np.float32).reshape(1, 1, 1, 3)
                    t_pre = self.pre_process_data(ts, key, self.model_config.dataset, already_encoded=True)  # type: ignore
                    t_pre = torch.from_numpy(t_pre) if not torch.is_tensor(t_pre) else t_pre
                    conditions_pre[key] = t_pre.to(self.device)
                    # For raw visualisation, decode from timestamp vector to day, month, year
                    conditions_raw[key] = self.post_process_data(conditions_pre[key], key, self.model_config.dataset)
                    continue

                size_hw = self._size_for_key(key)
                bands = self._bands_for_key(key)
                raw_bands: List[torch.Tensor] = []
                pre_bands: List[torch.Tensor] = []
                for band in bands:
                    band_file = self._resolve_band_path_for_tile(tile_key, base, band)
                    arr = _read_band_window(band_file, size_hw)
                    t_raw = torch.from_numpy(arr).float()
                    raw_bands.append(t_raw)
                    t_pre = self.pre_process_data(t_raw, key, self.model_config.dataset)  # type: ignore
                    pre_bands.append(t_pre if torch.is_tensor(t_pre) else torch.from_numpy(t_pre))
                conditions_raw[key] = torch.cat(raw_bands, dim=0).unsqueeze(0).to(self.device)
                conditions_pre[key] = torch.cat(pre_bands, dim=0).unsqueeze(0).to(self.device)
        return conditions_raw, conditions_pre

    def _resolve_band_path_for_tile(self, tile_key: str, base: str, band: str) -> Path:
        """Resolve the input band path for a given tile and base modality following MajorTOM layout using indices."""
        # Normalize modality key (e.g., "S2L1C_B02_B03_B04_B08") to base modality ("S2L1C")
        base = self._base_from_key(base)
        # Treat cloud_mask as part of S2L1C layout/index
        if base.lower() == "cloud_mask":
            base = "S2L1C"
        vstrip = tile_key.split("_")[0]
        core = self.paths.core[base]
        index = self.indices.get(base, {})
        if base.startswith("S2L"):
            prod = index.get(tile_key)
            if prod is None:
                raise FileNotFoundError(f"No {base} product for {tile_key}")
            return core / vstrip / tile_key / prod / f"{band}.tif"
        if base == "S1RTC":
            prod = index.get(tile_key)
            if prod is None:
                raise FileNotFoundError(f"No S1RTC product for {tile_key}")
            return core / vstrip / tile_key / prod / f"{band}.tif"  # band in ['vv','vh']
        if base == "DEM":
            prod = index.get(tile_key, "id")
            return core / vstrip / tile_key / prod / "DEM.tif"
        if base == "LULC":
            # LULC has path Core-LULC/<vstrip>/<L_part>/<product>.tif e.g. 618U/16L/618U_16L_2022.tif
            prod = index.get(tile_key)
            if prod is None:
                raise FileNotFoundError(f"No LULC product for {tile_key}")
            l_part = tile_key.split("_")[1]
            return core / vstrip / l_part / f"{prod}.tif"
        raise ValueError(f"Unsupported base modality {base}")

    def _save_generated_for_tile(self, tile_key: str, key: str, tensor: torch.Tensor, run_gen_root: Path, suffix: Optional[str] = None) -> None:
        base = self._base_from_key(key)
        vstrip = tile_key.split("_")[0]
        index = self.indices.get(base, {})
        prod = index.get(tile_key)
        if base in ("S2L2A", "S2L1C", "S1RTC"):
            if prod is None:
                raise FileNotFoundError(f"Missing product id for {base} {tile_key}")
            in_prod_dir = self.paths.core[base] / vstrip / tile_key / prod
            out_root = run_gen_root / f"Core-{base}" / in_prod_dir.relative_to(self.paths.dataset_root / f"Core-{base}")
            out_root.mkdir(parents=True, exist_ok=True)
        elif base == "DEM":
            prod = prod or "id"
            in_prod_dir = self.paths.core[base] / vstrip / tile_key / prod
            out_root = run_gen_root / f"Core-{base}" / in_prod_dir.relative_to(self.paths.dataset_root / f"Core-{base}")
            out_root.mkdir(parents=True, exist_ok=True)
        elif base == "LULC":
            if prod is None:
                raise FileNotFoundError(f"Missing product id for LULC {tile_key}")
            l_part = tile_key.split("_")[1]
            out_root = run_gen_root / "Core-LULC" / vstrip / l_part
            out_root.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"Unsupported base modality {base}")

        arr = tensor.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if base == "LULC":
            # Convert logits/one-hot to class indices if needed
            # if arr.shape[0] > 1:
            #     label = np.argmax(arr, axis=0).astype(np.float32)
            # else:
            #     label = arr[0]
            label = arr[0]
            out_path = out_root / f"{prod}{(suffix or '')}.tif"
            # Use LULC input as reference if available
            try:
                ref = self._resolve_band_path_for_tile(tile_key, base, band="LULC")
            except Exception:
                ref = None
            write_tif(out_path, label[np.newaxis, ...], ref_meta_or_path=ref, dtype=np.float32, nodata=None)
            return

        # Multiband (S2*, S1) or single DEM
        bands = self._bands_for_key(key)
        for i, band in enumerate(bands):
            out_path = out_root / (f"DEM{(suffix or '')}.tif" if base == "DEM" else f"{band}{(suffix or '')}.tif")
            try:
                ref = self._resolve_band_path_for_tile(tile_key, base, band if base != "DEM" else "DEM")
            except Exception:
                ref = None
            EPSILON = 1e-12
            if base == "S1RTC":
                arr = np.clip(arr, EPSILON, None)
            write_tif(out_path, arr[i:i+1], ref_meta_or_path=ref, dtype=np.float32, nodata=None)

    def _base_from_key(self, key: str) -> str:
        for base in ("S2L2A", "S2L1C", "S1RTC", "DEM", "LULC"):
            if base in key:
                return base
        return key

    def _load_gt_for_tile(self, tile_key: str, base: str, desired_hw: Tuple[int, int], bands: List[str]) -> np.ndarray:
        # Normalize modality key to base modality
        base = self._base_from_key(base)
        vstrip = tile_key.split("_")[0]
        if base in ("S2L2A", "S2L1C"):
            prod = self.indices[base].get(tile_key)
            if prod is None:
                raise FileNotFoundError(f"Missing product id for {base} {tile_key}")
            core_dir = self.paths.core[base] / vstrip / tile_key / prod
            arrays = []
            for bname in bands:
                with rasterio.open(core_dir / f"{bname}.tif") as src:
                    arrays.append(src.read(1).astype(np.float32))
            gt = np.stack(arrays, axis=0)
        elif base == "S1RTC":
            prod = self.indices[base].get(tile_key)
            if prod is None:
                raise FileNotFoundError(f"Missing product id for {base} {tile_key}")
            core_dir = self.paths.core[base] / vstrip / tile_key / prod
            arrays = []
            for pol in bands:
                with rasterio.open(core_dir / f"{pol}.tif") as src:
                    arrays.append(src.read(1).astype(np.float32))
            gt = np.stack(arrays, axis=0)
        elif base == "DEM":
            with rasterio.open(self.paths.core[base] / vstrip / tile_key / "id" / "DEM.tif") as src:
                gt = src.read().astype(np.float32)
        elif base == "LULC":
            prod = self.indices[base].get(tile_key)
            if prod is None:
                raise FileNotFoundError(f"Missing product id for {base} {tile_key}")
            l_part = tile_key.split("_")[1]
            with rasterio.open(self.paths.core[base] / vstrip / l_part / f"{prod}.tif") as src:
                gt1 = src.read(1).astype(np.float32)
            gt = gt1[None, ...]
        else:
            raise ValueError(f"Unsupported base {base}")
        # center crop
        c, h, w = gt.shape
        dh, dw = desired_hw
        if (h, w) != (dh, dw):
            top = max(0, (h - dh) // 2)
            left = max(0, (w - dw) // 2)
            gt = gt[:, top:top+dh, left:left+dw]
        return gt

    def process_all(
        self,
        condition_bases: List[str],
        generate_bases: List[str],
        max_products: Optional[int] = None,
        batch_size: int = 4,
        vis_every: Optional[int] = None,
        vis_comparison: bool = False,
        visualise_histograms: bool = False,
        samples_per_condition: int = 1,
    ) -> None:
        # Determine anchor tile keys using pickle index rather than scanning the fs
        anchor = "S2L2A" #if "S2L2A" in condition_bases else condition_bases[0]
        tile_keys = sorted(self.indices.get(anchor, {}).keys())
        if max_products is not None:
            tile_keys = tile_keys[:max_products]
        logging.info(f"Found {len(tile_keys)} tiles to process")

        # Resolve modality keys to generate (expand S2 groupings into keys)
        generate_keys: List[str] = []
        for base in tqdm(generate_bases, desc=f"Resolving modality keys", total=len(generate_bases)):
            generate_keys.extend(self._modality_keys_for_base(base))
        
        print(f"Generating {len(generate_keys)} keys")
        print(generate_keys)

        # Build run output root: <dataset_root>/outputs/input_..._output_..._seed_<seed>
        run_name = f"input_{'_'.join(sorted(condition_bases))}_output_{'_'.join(sorted(generate_bases))}_seed_{self.seed}"
        run_root = self.paths.dataset_root / "outputs" / "copgen" / run_name
        # Delete the run_root if it exists
        if run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        vis_root = run_root / "visualisations"
        vis_root.mkdir(parents=True, exist_ok=True)
        gen_root = run_root / "generations"
        gen_root.mkdir(parents=True, exist_ok=True)

        # Optional CSVs for scalar outputs
        want_latlon_csv = any("3d_cartesian_lat_lon" in k for k in generate_keys)
        want_time_csv = any("mean_timestamps" in k for k in generate_keys)
        latlon_csv_path = run_root / "lat_lon.csv"
        time_csv_path = run_root / "timestamp.csv"
        if want_latlon_csv and not latlon_csv_path.exists():
            with open(latlon_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["tile_id", "lat", "lon"])
        if want_time_csv and not time_csv_path.exists():
            with open(time_csv_path, "w", newline="") as f:
                csv.writer(f).writerow(["tile_id", "day", "month", "year"])

        # Process in batches
        def batched(seq, n):
            for i in range(0, len(seq), n):
                yield seq[i:i+n]

        for batch_idx, batch_tiles in enumerate(tqdm(list(batched(tile_keys, batch_size)), desc=f"Processing tiles", total=(len(tile_keys)+batch_size-1)//batch_size)):
            try:
                # Build batched conditions: raw for visualisation, preprocessed for the model
                batched_raw: Dict[str, List[torch.Tensor]] = {}
                batched_pre: Dict[str, List[torch.Tensor]] = {}
                for tile_key in batch_tiles:
                    cond_raw, cond_pre = self._prepare_conditions_for_tile(tile_key, condition_bases)
                    for k, v in cond_raw.items():
                        batched_raw.setdefault(k, []).append(v)
                    for k, v in cond_pre.items():
                        batched_pre.setdefault(k, []).append(v)
                conditions_raw = {k: torch.cat(v, dim=0) for k, v in batched_raw.items()}
                conditions_pre = {k: torch.cat(v, dim=0) for k, v in batched_pre.items()}

                generated = self.model.generate(modalities=generate_keys, conditions=conditions_pre, n_samples=int(samples_per_condition))  # type: ignore

                # Post-process generated modalities only
                generated_post: Dict[str, torch.Tensor] = {}
                for key, val in generated.items():
                    generated_post[key] = self.post_process_data(val, key, self.model_config.dataset)  # type: ignore
                # For visualisation, merge raw conditions with post-processed generated
                all_samples_for_vis = {**conditions_raw, **generated_post}

                # Save only generated modalities for each sample in batch
                for i, tile_key in enumerate(batch_tiles):
                    for key, tensor in generated_post.items():
                        # Determine slice indices for this tile across potential multiple samples
                        if tensor.ndim == 4:
                            if samples_per_condition > 1:
                                start = i * samples_per_condition
                                end = start + samples_per_condition
                                tile_samples = tensor[start:end]  # [S, C, H, W]
                            else:
                                tile_samples = tensor[i:i+1]  # [1, C, H, W]
                        else:
                            raise ValueError(f"Unexpected shape {tensor.shape} for modality {key}")

                        # Scalars to CSV (save per-sample and mean)
                        if "3d_cartesian_lat_lon" in key and want_latlon_csv:
                            # Save each sample
                            if samples_per_condition > 1:
                                for s in range(tile_samples.shape[0]):
                                    arr = tile_samples[s].detach().cpu().numpy().reshape(-1)
                                    lat, lon = float(arr[0]), float(arr[1])
                                    with open(latlon_csv_path, "a", newline="") as f:
                                        csv.writer(f).writerow([f"{tile_key}_sample_{s}", lat, lon])
                                # Save mean
                                mean_arr = tile_samples.to(torch.float32).mean(dim=0).detach().cpu().numpy().reshape(-1)
                                lat, lon = float(mean_arr[0]), float(mean_arr[1])
                                with open(latlon_csv_path, "a", newline="") as f:
                                    csv.writer(f).writerow([tile_key, lat, lon])
                            else:
                                arr = tile_samples[0].detach().cpu().numpy().reshape(-1)
                                lat, lon = float(arr[0]), float(arr[1])
                                with open(latlon_csv_path, "a", newline="") as f:
                                    csv.writer(f).writerow([tile_key, lat, lon])
                            continue
                        if "mean_timestamps" in key and want_time_csv:
                            # Save each sample
                            if samples_per_condition > 1:
                                for s in range(tile_samples.shape[0]):
                                    arr = tile_samples[s].detach().cpu().numpy().reshape(-1)
                                    day, month, year = int(arr[0]), int(arr[1]), int(arr[2])
                                    with open(time_csv_path, "a", newline="") as f:
                                        csv.writer(f).writerow([f"{tile_key}_sample_{s}", day, month, year])
                                # Save mean
                                mean_arr = tile_samples.to(torch.float32).mean(dim=0).detach().cpu().numpy().reshape(-1)
                                day, month, year = int(mean_arr[0]), int(mean_arr[1]), int(mean_arr[2])
                                with open(time_csv_path, "a", newline="") as f:
                                    csv.writer(f).writerow([tile_key, day, month, year])
                            else:
                                arr = tile_samples[0].detach().cpu().numpy().reshape(-1)
                                day, month, year = int(arr[0]), int(arr[1]), int(arr[2])
                                with open(time_csv_path, "a", newline="") as f:
                                    csv.writer(f).writerow([tile_key, day, month, year])
                            continue

                        # Image-like modalities: save per-sample and averaged
                        if samples_per_condition > 1:
                            for s in range(tile_samples.shape[0]):
                                self._save_generated_for_tile(tile_key, key, tile_samples[s], gen_root, suffix=f"_sample_{s}")
                            # For LULC (categorical) and cloud_mask, use mode instead of mean
                            base = self._base_from_key(key)
                            if base == "LULC" or 'cloud_mask' in key:
                                # Use mode (most frequent value) along sample axis
                                from scipy import stats
                                # tile_samples: [S, C, H, W]
                                tile_samples_np = tile_samples.detach().cpu().numpy()
                                mode_result = stats.mode(tile_samples_np, axis=0, keepdims=False)
                                avg_t = torch.from_numpy(mode_result.mode).to(tile_samples.device)  # [C, H, W]
                            else:
                                avg_t = tile_samples.to(torch.float32).mean(dim=0)  # [C, H, W]
                            
                            self._save_generated_for_tile(tile_key, key, avg_t, gen_root, suffix=None)
                        else:
                            self._save_generated_for_tile(tile_key, key, tile_samples[0], gen_root, suffix=None)

                # Visualisations every N batches (all modalities)
                if vis_every is not None and vis_every > 0 and (batch_idx % vis_every == 0):
                    gt_for_vis: Dict[str, torch.Tensor] = {}
                    
                    # Load ground truth for visualisation if comparison is enabled
                    if vis_comparison:
                        for key, tensor in all_samples_for_vis.items(): # for all modalities
                            if key in generated_post.keys(): # if modality is generated, load the ground truth
                                base = self._base_from_key(key)
                                # Image-like modalities: load from Core-*
                                if base in ("DEM", "S1RTC", "S2L1C", "S2L2A", "LULC"):
                                    bands = self._bands_for_key(key)
                                    dh, dw = int(tensor.shape[-2]), int(tensor.shape[-1])
                                    batch_arrays: List[np.ndarray] = []
                                    for i, tile_key in enumerate(batch_tiles):
                                        arr = self._load_gt_for_tile(tile_key, base, (dh, dw), bands if base in ("S2L2A", "S2L1C", "S1RTC") else [])
                                        batch_arrays.append(arr)
                                    gt_stack = torch.from_numpy(np.stack(batch_arrays, axis=0)).float()
                                    gt_for_vis[key] = gt_stack
                                # Scalar modalities: lat_lon and timestamps
                                elif "3d_cartesian_lat_lon" in key:
                                    # Build encoded GT xyz and decode via post_process_data -> lat/lon
                                    batch_xyz: List[torch.Tensor] = []
                                    for tile_key in batch_tiles:
                                        xyz = self.latlon_cache[(tile_key, -1, -1)]  # (3,)
                                        xyz_t = torch.tensor(np.asarray(xyz, dtype=np.float32)).view(1,1,1,3)
                                        batch_xyz.append(xyz_t)
                                    enc = torch.cat(batch_xyz, dim=0)
                                    gt_decoded = self.post_process_data(enc, key, self.model_config.dataset)
                                    gt_for_vis[key] = gt_decoded.float()
                                elif "mean_timestamps" in key:
                                    batch_ts: List[torch.Tensor] = []
                                    for tile_key in batch_tiles:
                                        vec = self.timestamps_cache[tile_key]
                                        ts_t = torch.tensor(np.asarray(vec, dtype=np.float32)).view(1,1,1,3)
                                        batch_ts.append(ts_t)
                                    enc = torch.cat(batch_ts, dim=0)
                                    gt_decoded = self.post_process_data(enc, key, self.model_config.dataset)
                                    gt_for_vis[key] = gt_decoded.float()
                                else:
                                    raise ValueError(f"Unsupported modality {key} for vis_comparison")
                            elif key in conditions_raw.keys(): # if modality is condition, load the ground truth
                                # gt_for_vis[key] = tensor
                                pass
                            else:
                                raise ValueError(f"Something went wrong for modality {key} for vis_comparison")
                                
                    max_img_input_resolution = max(self.model_config.all_modality_configs[modality].img_input_resolution[1] for modality in all_samples_for_vis.keys())
                    for modality, tensor in all_samples_for_vis.items():
                        save_dir = vis_root / modality
                        save_dir.mkdir(parents=True, exist_ok=True)
                        inputs = gt_for_vis.get(modality) if vis_comparison else None
                        # Resize should be the max img_input_resolution of all modalities
                        if inputs is not None:
                            input_frame_color = (0.2, 0.9, 0.3) # inputs are gt (softer green frame)
                            recon_frame_color = (0.3, 0.5, 1.0) # reconstructions are generated (softer blue frame)
                        else:
                            input_frame_color = None # inputs are none
                            recon_frame_color = (1.0, 0.3, 0.2) # reconstructions are the initial conditions (softer red frame)
                        if visualise_histograms:
                            view_histogram = True if modality.split('_')[0] in ['S2L2A', 'S2L1C', 'DEM', 'S1RTC'] else False
                        else:
                            view_histogram = False
                        AVERAGED_ACROSS_SAMPLES = True
                        if AVERAGED_ACROSS_SAMPLES and samples_per_condition > 1 and modality in generated_post.keys():
                            # tensor shape: [batch_size * samples_per_condition, C, H, W]
                            # Reshape to [batch_size, samples_per_condition, C, H, W] and average
                            C, H, W = tensor.shape[1], tensor.shape[2], tensor.shape[3]
                            tensor_reshaped = tensor.view(batch_size, samples_per_condition, C, H, W)
                            
                            # For LULC (categorical) and cloud_mask, use mode instead of mean
                            base = self._base_from_key(modality)
                            if base == "LULC" or 'cloud_mask' in modality:
                                # Use mode (most frequent value) along sample axis
                                from scipy import stats
                                tensor_np = tensor_reshaped.detach().cpu().numpy()
                                mode_result = stats.mode(tensor_np, axis=1, keepdims=False)
                                tensor = torch.from_numpy(mode_result.mode).to(tensor.device)  # [batch_size, C, H, W]
                            else:
                                tensor = tensor_reshaped.to(torch.float32).mean(dim=1)  # [batch_size, C, H, W]
                        visualise_bands(inputs=inputs, reconstructions=tensor, save_dir=save_dir,
                                        milestone=batch_idx, repeat_n_times=None, satellite_type=modality,
                                        view_histogram=view_histogram,
                                        resize=max_img_input_resolution, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color)
                    merge_visualisations(results_path=str(vis_root), verbose=False, overwrite=False)

            except Exception as e:
                logging.warning(f"Skipping batch starting at {batch_tiles[0]}: {e}")
                raise e


