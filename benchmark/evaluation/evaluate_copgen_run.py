from __future__ import annotations

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, TYPE_CHECKING
import csv
import pickle

if TYPE_CHECKING:
    from benchmark.evaluation import ImageRegressionMetrics, SpatialMetrics, SegmentationMetrics
    from benchmark.evaluation.regression import TemporalMetrics

_DEPS = None


def _get_deps():
    """Lazy import heavy dependencies to avoid loading them unnecessarily."""
    global _DEPS
    if _DEPS is None:
        import numpy as np  # type: ignore
        import torch  # type: ignore
        import rasterio  # type: ignore
        from tqdm import tqdm  # type: ignore

        from benchmark.evaluation import ImageRegressionMetrics, SpatialMetrics, SegmentationMetrics  # type: ignore
        from benchmark.evaluation.regression import TemporalMetrics  # type: ignore
        from ddm.pre_post_process_data import decode_lat_lon, decode_date, get_value_to_index, name_list  # type: ignore

        _DEPS = (
            np,
            torch,
            rasterio,
            tqdm,
            ImageRegressionMetrics,
            SpatialMetrics,
            SegmentationMetrics,
            TemporalMetrics,
            decode_lat_lon,
            decode_date,
            get_value_to_index,
            name_list,
        )
    return _DEPS

def _center_crop_to_match(gt: Any, pr: Any) -> Any:
    np, *_ = _get_deps()
    if gt.shape == pr.shape:
        return gt
    _, h, w = pr.shape
    top = max(0, (gt.shape[1] - h) // 2)
    left = max(0, (gt.shape[2] - w) // 2)
    return gt[:, top:top + h, left:left + w]


def _gather_generated_tifs(gen_root: Path) -> Dict[str, List[Path]]:
    per_mod: Dict[str, List[Path]] = {}
    for base in ["DEM", "S1RTC", "S2L1C", "S2L2A", "LULC"]:
        core_dir = gen_root / f"Core-{base}"
        if not core_dir.exists():
            continue
        if base == "S2L1C":
            # Separate cloud_mask.tif from other bands
            all_tifs = sorted(core_dir.rglob("*.tif"))
            cloud_mask_tifs = [f for f in all_tifs if f.name == "cloud_mask.tif"]
            other_tifs = [f for f in all_tifs if f.name != "cloud_mask.tif"]
            per_mod[base] = other_tifs
            if cloud_mask_tifs:
                per_mod["cloud_mask"] = cloud_mask_tifs
        else:
            per_mod[base] = sorted(core_dir.rglob("*.tif"))
    # Remove modalities that have empty lists
    per_mod = {k: v for k, v in per_mod.items() if v}
    return per_mod


def _match_gt_path(dataset_root: Path, base: str, gen_file: Path) -> Path:
    # Replace generations/Core-<base>/... with Core-<base>/...
    # gen_file: <run>/generations/Core-<base>/<vstrip>/<tile>/<prod>/Bxx.tif (or DEM.tif, LULC layout)
    # For cloud_mask, the generated file is in Core-S2L1C but we need to map to Core-S2L1C
    parts = list(gen_file.parts)
    
    # For cloud_mask, the generated file is in Core-S2L1C
    if base == "cloud_mask":
        if "Core-S2L1C" not in parts:
            raise ValueError(f"Unexpected generated path for cloud_mask: {gen_file}")
        idx = parts.index("Core-S2L1C")
        rel = Path(*parts[idx:])  # Core-S2L1C/...
        return dataset_root / rel
    
    if f"Core-{base}" not in parts:
        raise ValueError(f"Unexpected generated path for {base}: {gen_file}")
    idx = parts.index(f"Core-{base}")
    rel = Path(*parts[idx:])  # Core-<base>/...
    return dataset_root / rel


def _load_and_prepare_pair(dataset_root: Path, base: str, pr_file: Path):
    """Load a prediction/GT pair and apply modality-specific preprocessing.

    Returns (pr, gt, pr_file) or None if GT is missing.
    """
    np, _, rasterio, *_ = _get_deps()
    gt_file = _match_gt_path(dataset_root, base, pr_file)
    if not gt_file.exists():
        return None
    with rasterio.open(pr_file) as pr_src, rasterio.open(gt_file) as gt_src:
        pr = pr_src.read().astype(np.float32)
        gt = gt_src.read().astype(np.float32)
        # Physical units
        if base == "S2L2A" or base == "S2L1C":
            pr = pr / 10000.0
            gt = gt / 10000.0
        elif base == "S1RTC":
            # Convert linear backscatter to dB; guard against non-positive values
            # which would yield -inf and break MAE/RMSE.
            eps = 1e-12
            pr = 10.0 * np.log10(np.clip(pr, eps, None))
            gt = 10.0 * np.log10(np.clip(gt, eps, None))
            pr = np.clip(pr, -50.0, 10.0)
            gt = np.clip(gt, -50.0, 10.0)

    gt = _center_crop_to_match(gt, pr)
    return pr, gt, pr_file


def _extract_tile_id_from_gen_path(base: str, gen_file: Path) -> str:
    """Best-effort extraction of tile id from a generated file path.
    Expected layout:
      generations/Core-<base>/<vstrip>/<tile>/<product>/file.tif
    For cloud_mask, the files are under Core-S2L1C.
    """
    parts = list(gen_file.parts)
    key = "Core-S2L1C" if base == "cloud_mask" else f"Core-{base}"
    if key in parts:
        idx = parts.index(key)
        # tile is expected at idx+2: Core-<base>/<vstrip>/<tile>/...
        if len(parts) > idx + 2:
            return parts[idx + 2]
    # Fallbacks if structure is different
    try:
        return gen_file.parent.parent.name  # .../<tile>/<product>/file.tif
    except Exception:
        return gen_file.parent.name


def _evaluate_images(dataset_root: Path, gen_root: Path, per_tile: bool = False, num_workers: int | None = None) -> Dict[str, Dict[str, float]] | tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, Dict[str, float]]]]:
    (
        np,
        torch,
        _,
        tqdm,
        ImageRegressionMetrics,
        _,
        SegmentationMetrics,
        _,
        _,
        _,
        get_value_to_index,
        _,
    ) = _get_deps()
    per_mod_files = _gather_generated_tifs(gen_root)
    summaries: Dict[str, Dict[str, float]] = {}
    per_tile_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}

    topk = 3
    
    for base, files in per_mod_files.items():
        if base == "LULC" or base == "cloud_mask":
            lut = get_value_to_index(base)
            segm = SegmentationMetrics(num_classes=len(lut), topk=topk)
            segm_per_tile: Dict[str, SegmentationMetrics] = {} if per_tile else {}
        else:
            imgm = ImageRegressionMetrics()
            imgm_per_tile: Dict[str, ImageRegressionMetrics] = {} if per_tile else {}

        worker_count = num_workers or min(64, os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = {ex.submit(_load_and_prepare_pair, dataset_root, base, pr_file): pr_file for pr_file in files}
            with tqdm(total=len(futures), desc=f"Evaluating {base}") as pbar:
                for fut in as_completed(futures):
                    pbar.update(1)
                    result = fut.result()
                    if result is None:
                        continue
                    pr, gt, pr_file = result
                    try:
                        if base == "LULC" or base == "cloud_mask":
                            # Map GT classes to 0..9 and one-hot both
                            gt_mapped = np.zeros_like(gt[0], dtype=np.int64)
                            pr_mapped = np.zeros_like(pr[0], dtype=np.int64)
                            for orig, idx in lut.items():
                                gt_mapped[gt[0] == orig] = idx
                                pr_mapped[pr[0] == orig] = idx
                            pr_onehot = torch.nn.functional.one_hot(torch.from_numpy(pr_mapped), num_classes=len(lut)).permute(2,0,1).unsqueeze(0).float()
                            gt_onehot = torch.nn.functional.one_hot(torch.from_numpy(gt_mapped), num_classes=len(lut)).permute(2,0,1).unsqueeze(0).float()
                            segm.update_from_logits(pr_onehot, gt_onehot)
                            if per_tile:
                                tile_id = _extract_tile_id_from_gen_path(base, pr_file)
                                if tile_id not in segm_per_tile:
                                    segm_per_tile[tile_id] = SegmentationMetrics(num_classes=len(lut), topk=topk)
                                segm_per_tile[tile_id].update_from_logits(pr_onehot, gt_onehot)
                        else:
                            # Align shapes to [B,C,H,W] and ensure contiguous memory layout
                            pr_t = torch.from_numpy(pr).unsqueeze(0).contiguous()
                            gt_t = torch.from_numpy(gt).unsqueeze(0).contiguous()
                            assert pr_t.shape[1] == 1, "Currently only single-band tifs are supported for cop-gen evaluation"
                            imgm.update(pr_t, gt_t, band_names=[pr_file.stem])
                            if per_tile:
                                tile_id = _extract_tile_id_from_gen_path(base, pr_file)
                                if tile_id not in imgm_per_tile:
                                    imgm_per_tile[tile_id] = ImageRegressionMetrics()
                                imgm_per_tile[tile_id].update(pr_t, gt_t, band_names=[pr_file.stem])
                    except Exception as e:
                        raise e

        if base == "LULC" or base == "cloud_mask":
            summaries[base] = segm.summary()
            if per_tile:
                per_tile_summaries[base] = {t: m.summary() for t, m in segm_per_tile.items()}
        else:
            summaries[base] = imgm.summary()
            if per_tile:
                per_tile_summaries[base] = {t: m.summary() for t, m in imgm_per_tile.items()}

    if per_tile:
        return summaries, per_tile_summaries
    else:
        return summaries


def _evaluate_scalars(dataset_root: Path, run_root: Path, per_tile: bool = False) -> Dict[str, Dict[str, float]] | tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    (
        np,
        torch,
        _,
        _,
        _,
        SpatialMetrics,
        _,
        TemporalMetrics,
        decode_lat_lon,
        decode_date,
        _,
        _,
    ) = _get_deps()
    out: Dict[str, Dict[str, float]] = {}
    per_tile_out: Dict[str, Dict[str, float]] = {}
    # lat/lon
    latlon_csv = run_root / "lat_lon.csv"
    if latlon_csv.exists():
        lat_dir = dataset_root / "3d_cartesian_lat_lon_cache"
        from datasets import _LatLonMemmapCache  # local import to avoid heavy dep otherwise
        lat_cache = _LatLonMemmapCache(str(lat_dir))
        sm = SpatialMetrics(radii_km=(1, 10, 50, 100, 500))
        sm_per_tile: Dict[str, SpatialMetrics] = {} if per_tile else {}
        with open(latlon_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tile = row['tile_id']
                pred = torch.tensor([[float(row['lat']), float(row['lon'])]], dtype=torch.float32)
                xyz = lat_cache[(tile, -1, -1)]
                xyz_t = torch.tensor(np.asarray(xyz, dtype=np.float32)).view(1,1,1,3)
                gt_latlon = decode_lat_lon(xyz_t, '3d_cartesian_lat_lon').view(1,2)
                sm.update(pred, gt_latlon)
                if per_tile:
                    if tile not in sm_per_tile:
                        sm_per_tile[tile] = SpatialMetrics(radii_km=(1, 10, 50, 100, 500))
                    sm_per_tile[tile].update(pred, gt_latlon)
        out['lat_lon'] = sm.summary()
        if per_tile:
            for t, m in sm_per_tile.items():
                per_tile_out.setdefault('lat_lon', {})
                per_tile_out['lat_lon'][t] = m.summary()

    # timestamps
    ts_csv = run_root / "timestamp.csv"
    ts_pkl = dataset_root / "mean_timestamps_cache.pkl"
    if ts_csv.exists() and ts_pkl.exists():
        with open(ts_pkl, "rb") as f:
            ts_cache = dict(pickle.load(f))  # tile -> encoded vec
        tm = TemporalMetrics()
        tm_per_tile: Dict[str, TemporalMetrics] = {} if per_tile else {}
        with open(ts_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tile = row['tile_id']
                pred = torch.tensor([[int(row['day']), int(row['month']), int(row['year'])]], dtype=torch.float32)
                enc = torch.tensor(np.asarray(ts_cache[tile], dtype=np.float32)).view(1,1,1,3)
                gt_ymd = decode_date(enc).view(1,3)
                tm.update(pred, gt_ymd)
                if per_tile:
                    if tile not in tm_per_tile:
                        tm_per_tile[tile] = TemporalMetrics()
                    tm_per_tile[tile].update(pred, gt_ymd)
        out['timestamps'] = tm.summary()
        if per_tile:
            for t, m in tm_per_tile.items():
                per_tile_out.setdefault('timestamps', {})
                per_tile_out['timestamps'][t] = m.summary()

    if per_tile:
        return out, per_tile_out
    else:
        return out

def _print_metrics(metrics: Dict[str, Dict[str, float]]):
    *_, name_list = _get_deps()
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
            
            # Accuracy at different radii
            for keys_lat_lon in list(metrics_summary.keys()):
                if 'acc' in keys_lat_lon:
                    print(f"  Accuracy within {keys_lat_lon}: {metrics_summary[keys_lat_lon]:.1%}")
        elif modality == "LULC" or modality == "cloud_mask":
            for metric_key in metrics_summary.keys():
                if 'overall_top' in metric_key:
                    print(f"  {str(metric_key).capitalize()} Accuracy: {metrics_summary[metric_key]:.3f}")
            print(f"  Mean IoU: {metrics_summary['mean_iou']:.3f}")
            print(f"  Mean F1-Score: {metrics_summary['mean_f1']:.3f}")
            print(f"  Frequency-Weighted IoU: {metrics_summary['fw_iou']:.3f}")
            
            # Per-class metrics
            print(f"  Per-class metrics:")
            num_classes = len(metrics_summary['per_class_iou'])
            for i in range(num_classes):
                class_name = name_list[modality][i]
                iou = metrics_summary['per_class_iou'][i]
                precision = metrics_summary['per_class_precision'][i]
                recall = metrics_summary['per_class_recall'][i]
                f1 = metrics_summary['per_class_f1'][i]
                print(f"    {class_name:20s}: IoU={iou:.3f}, Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}")
                
        elif modality == "timestamps":
            print(f"  Median error: {metrics_summary['median_days']:.2f} days")
            print(f"  Mean error: {metrics_summary['mean_days']:.2f} days")
            print(f"  Std error: {metrics_summary['std_days']:.2f} days")
            print(f"  RMSE error: {metrics_summary['rmse_days']:.2f} days")
            print(f"  90th percentile error: {metrics_summary['p90_days']:.2f} days")
            print(f"  95th percentile error: {metrics_summary['p95_days']:.2f} days")
            print(f"  99th percentile error: {metrics_summary['p99_days']:.2f} days")
            print(f"  Mean absolute months: {metrics_summary['mean_abs_months']:.2f}")
            print(f"  Median absolute months: {metrics_summary['median_abs_months']:.2f}")
            print(f"  90th percentile absolute months: {metrics_summary['p90_abs_months']:.2f}")
            print(f"  Month accuracy: {metrics_summary['month_acc']:.1%}")
            print(f"  Year accuracy: {metrics_summary['year_acc']:.1%}")
        else:
            # Image modalities (S2L2A, S2L1C, S1RTC, DEM)
            print(f"  Mean Absolute Error: {metrics_summary['mae']:.4f}")
            print(f"  Root Mean Square Error: {metrics_summary['rmse']:.4f}")
            print(f"  Structural Similarity (SSIM): {metrics_summary['ssim']:.4f}")
            print(f"  Peak Signal-to-Noise Ratio: {metrics_summary['psnr']:.2f} dB")
            # Optional per-band breakdown
            if 'per_band' in metrics_summary and isinstance(metrics_summary['per_band'], dict):
                print("  Per-band metrics:")
                for bname in sorted(metrics_summary['per_band'].keys()):
                    b = metrics_summary['per_band'][bname]
                    print(f"    {bname:10s}: MAE={b.get('mae', 0.0):.4f}, RMSE={b.get('rmse', 0.0):.4f}, SSIM={b.get('ssim', 0.0):.4f}, PSNR={b.get('psnr', 0.0):.2f}")
    
    print("=" * 60)


def evaluate_copgen_run(dataset_root: Path, run_root: Path, verbose: bool = False, per_tile: bool = False, num_workers: int | None = None) -> Dict[str, Dict[str, float]]:
    """Evaluate a COP-GEN run outputs against ground truth dataset.

    Args:
        dataset_root: Path to MajorTOM dataset root (contains Core-*/ and caches)
        run_root: Path to run folder (contains generations/, lat_lon.csv?, timestamp.csv?)
        per_tile: If True, also compute and write per-tile metrics
        num_workers: Optional override for parallel image loading/preprocessing
    Returns:
        Dict of modality -> metrics summary
    Side-effects:
        Writes a textual report to run_root / 'output_metrics.txt'
        If per_tile, also writes run_root / 'output_metrics_per_tile.csv'
    """
    gen_root = run_root / "generations"
    if not gen_root.exists():
        raise FileNotFoundError(f"Missing generations folder: {gen_root}")

    out_txt = run_root / "output_metrics.txt"
    out_csv = run_root / "output_metrics_per_tile.csv"
    if out_txt.exists() and (not per_tile or out_csv.exists()):
        msg = f"Skipping evaluation; metrics already exist at {out_txt}"
        if per_tile:
            msg += f" and {out_csv}"
        logging.info(msg)
        if verbose:
            print(msg)
        return {}

    # Images
    if per_tile:
        img_metrics, img_metrics_per_tile = _evaluate_images(dataset_root, gen_root, per_tile=True, num_workers=num_workers)
    else:
        img_metrics = _evaluate_images(dataset_root, gen_root, per_tile=False, num_workers=num_workers)
        img_metrics_per_tile = {}
    # Scalars
    if per_tile:
        scalar_metrics, scalar_metrics_per_tile = _evaluate_scalars(dataset_root, run_root, per_tile=True)
    else:
        scalar_metrics = _evaluate_scalars(dataset_root, run_root, per_tile=False)
        scalar_metrics_per_tile = {}

    # Combine
    all_metrics: Dict[str, Dict[str, float]] = {}
    all_metrics.update(img_metrics)
    all_metrics.update(scalar_metrics)

    # Write report
    out_txt = run_root / "output_metrics.txt"
    with open(out_txt, "w") as f:
        f.write("Evaluation metrics for generated modalities\n")
        for mod, s in all_metrics.items():
            f.write(f"[{mod}] ")
            # Flatten per-band for textual report for easier parsing
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

    # Optional per-tile CSV
    if per_tile:
        rows: List[Dict[str, object]] = []
        # helper to flatten dict (only per_band special-cased similar to text output)
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

        # collect from images
        for modality, per_tile_map in img_metrics_per_tile.items():
            for tile_id, md in per_tile_map.items():
                row = {'tile_id': tile_id, 'modality': modality}
                row.update(_flatten_metrics_dict(md))
                rows.append(row)
        # collect from scalars
        for modality, per_tile_map in scalar_metrics_per_tile.items():
            for tile_id, md in per_tile_map.items():
                row = {'tile_id': tile_id, 'modality': modality}
                row.update(_flatten_metrics_dict(md))
                rows.append(row)

        if rows:
            # union of all keys
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
    print("Imports done")
    import argparse
    p = argparse.ArgumentParser(description="Evaluate COP-GEN run outputs against MajorTOM ground truth")
    p.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root (Core-*/ present)")
    p.add_argument("--run-root", type=str, required=True, help="Path to run root (with generations/, csvs)")
    p.add_argument("--per-tile", action="store_true", default=False, help="Also compute and write per-tile metrics (default: False)")
    p.add_argument("--workers", type=int, default=None, help="Parallel workers for image loading/preprocessing (default: CPU count)")
    args = p.parse_args()
    metrics = evaluate_copgen_run(Path(args.dataset_root), Path(args.run_root), verbose=True, per_tile=args.per_tile, num_workers=args.workers)
