from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from benchmark.generations.copgen import CopgenBatchGenerator
from benchmark.evaluation.evaluate_copgen_run import evaluate_copgen_run


# -----------------------------
# Argument parsing
# -----------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run COP-GEN std experiment: multiple samples per tile to estimate output variability",
        epilog="Modalities: lat_lon, timestamps, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask",
    )
    p.add_argument("--dataset-root", type=str, required=True, help="Path to MajorTOM dataset root")
    p.add_argument("--model", type=str, required=True, help="Path to Copgen model checkpoint (nnet_ema.pth)")
    p.add_argument("--config", type=str, required=True, help="Path to Copgen model config .py (get_config())")
    p.add_argument(
        "--inputs",
        nargs="+",
        type=str,
        required=True,
        help="Condition base modalities (e.g. S2L2A S1RTC DEM LULC)."
             " Allowed: lat_lon, timestamps, DEM, LULC, S1RTC, S2L1C, S2L2A, cloud_mask",
    )
    p.add_argument("--seed", type=int, default=1234, help="Global seed for reproducibility")
    p.add_argument("--num-tiles", type=int, default=5, help="Number of tiles to evaluate")
    p.add_argument("--num-samples", type=int, default=20, help="Number of samples per tile")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of samples to generate per batch iteration (per tile). Final batch may be smaller.",
    )
    p.add_argument(
        "--vis-every",
        type=int,
        default=20,
        help="Visualise every N samples per tile (0 disables). E.g., 20 -> samples 0,20,40,...",
    )
    p.add_argument(
        "--cleanup",
        action="store_true",
        help="If set, delete per-sample generations after evaluation to save disk",
    )
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------


ALLOWED_MODALITIES = {"lat_lon", "timestamps", "DEM", "LULC", "S1RTC", "S2L1C", "S2L2A", "cloud_mask"}


def build_run_name(inputs: List[str], outputs: List[str], seed: int, num_tiles: int, num_samples: int) -> str:
    return (
        f"input_{'_'.join(sorted(inputs))}_"
        f"output_{'_'.join(sorted(outputs))}_"
        f"seed_{seed}_"
        f"tiles_{num_tiles}_"
        f"samples_{num_samples}"
    )


def generate_reproducible_seeds(global_seed: int, count: int) -> List[int]:
    rng = np.random.RandomState(global_seed)
    # Use 31-bit range to stay in signed 32-bit PyTorch safe space
    seeds = rng.randint(low=0, high=2 ** 31 - 1, size=count, dtype=np.int64)
    return [int(s) for s in seeds]


def select_tiles_deterministic(indices: Dict[str, Dict[str, str]], num_tiles: int, seed: int) -> List[str]:
    anchor = "S2L2A"
    all_tiles = sorted(indices.get(anchor, {}).keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(all_tiles)
    return all_tiles[:num_tiles]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def is_scalar_key(modality_key: str) -> bool:
    return ("3d_cartesian_lat_lon" in modality_key) or ("mean_timestamps" in modality_key)


def base_from_key(modality_key: str) -> str:
    # Local fallback to match CopgenBatchGenerator._base_from_key logic
    for base in ("S2L2A", "S2L1C", "S1RTC", "DEM", "LULC"):
        if base in modality_key:
            return base
    return modality_key


def write_json(path: Path, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


@dataclass
class SampleEvalPaths:
    sample_root: Path
    sample_generations_root: Path
    sample_visual_root: Path
    visual_root: Path
    metrics_root: Path
    latlon_csv: Optional[Path]
    time_csv: Optional[Path]


def prepare_sample_paths(tile_root: Path, sample_index: int, want_latlon_csv: bool, want_time_csv: bool) -> SampleEvalPaths:
    sample_root = tile_root / "samples" / f"sample_{sample_index}"
    sample_generations_root = sample_root / "generations"
    sample_visual_root = sample_root / "visualisations"
    visual_root = tile_root / "visualisations"
    metrics_root = tile_root / "metrics"
    ensure_dir(sample_generations_root)
    ensure_dir(sample_visual_root)
    ensure_dir(visual_root)
    ensure_dir(metrics_root)
    latlon_csv = sample_root / "lat_lon.csv" if want_latlon_csv else None
    time_csv = sample_root / "timestamp.csv" if want_time_csv else None
    if latlon_csv is not None and not latlon_csv.exists():
        with open(latlon_csv, "w") as f:
            f.write("tile_id,lat,lon\n")
    if time_csv is not None and not time_csv.exists():
        with open(time_csv, "w") as f:
            f.write("tile_id,day,month,year\n")
    return SampleEvalPaths(
        sample_root=sample_root,
        sample_generations_root=sample_generations_root,
        sample_visual_root=sample_visual_root,
        visual_root=visual_root,
        metrics_root=metrics_root,
        latlon_csv=latlon_csv,
        time_csv=time_csv,
    )


def append_scalar_csv(latlon_csv: Optional[Path], time_csv: Optional[Path], tile_key: str, key: str, tensor: torch.Tensor) -> bool:
    # tensor is [C,H,W] or [1,1,3] like in post-processed scalars
    if "3d_cartesian_lat_lon" in key and latlon_csv is not None:
        arr = tensor.detach().cpu().numpy().reshape(-1)
        lat, lon = float(arr[0]), float(arr[1])
        with open(latlon_csv, "a") as f:
            f.write(f"{tile_key},{lat},{lon}\n")
        return True
    if "mean_timestamps" in key and time_csv is not None:
        arr = tensor.detach().cpu().numpy().reshape(-1)
        # Round to nearest integer day/month/year
        day, month, year = int(round(float(arr[0]))), int(round(float(arr[1]))), int(round(float(arr[2])))
        with open(time_csv, "a") as f:
            f.write(f"{tile_key},{day},{month},{year}\n")
        return True
    return False


def save_std_map_png(std_arr: np.ndarray, out_path: Path, title: str) -> None:
    # std_arr is HxW
    ensure_dir(out_path.parent)
    plt.figure(figsize=(6, 6))
    im = plt.imshow(std_arr, cmap="plasma")
    plt.title(title)
    plt.axis("off")
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("std")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_histogram_png(values: np.ndarray, out_path: Path, title: str, bins: int = 100) -> None:
    ensure_dir(out_path.parent)
    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins, color="#3366cc", alpha=0.85)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def summarize_stats(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "mean": float(np.nanmean(arr)),
        "std": float(np.nanstd(arr)),
    }


def aggregate_per_key_metrics(base_metrics: Dict[str, float], per_band: Dict[str, Dict[str, float]], key_to_bands: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Build per-expanded-key metrics by averaging the per-band metrics across bands belonging to each key.
    Falls back to base metrics if per_band is not available.
    """
    out: Dict[str, Dict[str, float]] = {}
    if not per_band:
        # No per-band; assign base metrics to each key
        for k in key_to_bands.keys():
            out[k] = dict(base_metrics)
        return out
    # Collect per metric arrays per key
    # Identify numeric metrics in per_band entries
    metric_names: List[str] = []
    for _, md in per_band.items():
        metric_names = [n for n, v in md.items() if isinstance(v, (int, float))]
        break
    for key, bands in key_to_bands.items():
        acc: Dict[str, List[float]] = {m: [] for m in metric_names}
        for b in bands:
            if b in per_band:
                for m in metric_names:
                    v = per_band[b].get(m)
                    if isinstance(v, (int, float)):
                        acc[m].append(float(v))
        out[key] = {m: (float(np.mean(acc[m])) if acc[m] else math.nan) for m in metric_names}
    return out


# -----------------------------
# Main experiment
# -----------------------------


def main() -> None:
    args = parse_args()

    # Validate modalities
    invalid = [m for m in args.inputs if m not in ALLOWED_MODALITIES]
    if invalid:
        raise ValueError(f"Invalid input modalities: {invalid}. Allowed: {sorted(ALLOWED_MODALITIES)}")
    outputs = sorted(list(ALLOWED_MODALITIES - set(args.inputs)))

    # Init generator (reuses model, pre/post, indices, caches)
    gen = CopgenBatchGenerator(
        model_path=Path(args.model),
        config_path=Path(args.config),
        dataset_root=Path(args.dataset_root),
        seed=int(args.seed),
    )

    # Expand generate keys
    generate_keys: List[str] = []
    for base in outputs:
        generate_keys.extend(gen._modality_keys_for_base(base))  # type: ignore[attr-defined]

    # Determine tiles and seeds
    tiles = select_tiles_deterministic(gen.indices, num_tiles=int(args.num_tiles), seed=int(args.seed))
    sample_seeds = generate_reproducible_seeds(global_seed=int(args.seed), count=int(args.num_samples))

    # Run root
    run_name = build_run_name(inputs=args.inputs, outputs=outputs, seed=int(args.seed), num_tiles=int(args.num_tiles), num_samples=int(args.num_samples))
    run_root = Path(args.dataset_root) / "outputs" / "copgen_std" / run_name
    if run_root.exists():
        shutil.rmtree(run_root)
    ensure_dir(run_root)

    print("=" * 60)
    print("COP-GEN STD EXPERIMENT CONFIGURATION")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Global seed: {args.seed}")
    print(f"Num tiles: {args.num_tiles}")
    print(f"Num samples per tile: {args.num_samples}")
    print(f"Inputs: {', '.join(sorted(args.inputs))}")
    print(f"Outputs: {', '.join(sorted(outputs))}")
    print(f"Generate keys: {len(generate_keys)} -> {generate_keys}")
    print(f"Run root: {str(run_root)}")
    print("=" * 60)

    # Map: per expanded key -> bands
    key_to_bands: Dict[str, List[str]] = {}
    img_like_keys: List[str] = []
    for k in generate_keys:
        if is_scalar_key(k):
            continue
        img_like_keys.append(k)
        key_to_bands[k] = gen._bands_for_key(k)  # type: ignore[attr-defined]

    # Per-tile processing
    all_samples_metrics_base: Dict[str, List[float]] = {}  # modality(base) -> list of metric values per sample across tiles (for mean/aggregate of one representative metric)
    aggregate_metrics_by_modality: Dict[str, Dict[str, List[float]]] = {}  # modality(base) -> metric_name -> values
    aggregate_metrics_by_key: Dict[str, Dict[str, List[float]]] = {}  # expanded key -> metric_name -> values

    for tile_idx, tile_key in enumerate(tiles):
        print(f"\nProcessing tile {tile_idx+1}/{len(tiles)}: {tile_key}")
        tile_root = run_root / tile_key
        ensure_dir(tile_root)

        # Prepare conditions once per tile
        conditions_raw, conditions_pre = gen._prepare_conditions_for_tile(tile_key, args.inputs)  # type: ignore[attr-defined]

        # Scalars present?
        want_latlon_csv = any("3d_cartesian_lat_lon" in k for k in generate_keys)
        want_time_csv = any("mean_timestamps" in k for k in generate_keys)

        # Accumulate generated arrays for std/hist per expanded key
        # key -> list of np.ndarray [C,H,W] per sample
        per_key_arrays: Dict[str, List[np.ndarray]] = {k: [] for k in img_like_keys}

        # Collect per-sample metrics for this tile
        per_sample_metrics_base: List[Dict[str, Dict[str, float]]] = []
        per_sample_metrics_key: List[Dict[str, Dict[str, float]]] = []

        # Batched sampling over num_samples
        batch_size = max(1, int(args.batch_size))
        for batch_start in tqdm(range(0, len(sample_seeds), batch_size), total=(len(sample_seeds) + batch_size - 1) // batch_size, desc="Processing samples (batches)"):
            batch_end = min(batch_start + batch_size, len(sample_seeds))
            current_bs = batch_end - batch_start

            # Prepare sample dirs and CSVs for this batch
            paths_batch: List[SampleEvalPaths] = []
            for s_idx in range(batch_start, batch_end):
                paths_batch.append(prepare_sample_paths(tile_root, s_idx, want_latlon_csv, want_time_csv))

            # Set seed once per batch for reproducibility; internal sampling generates distinct samples
            gen.model.set_seed(int(sample_seeds[batch_start]))

            # Generate a batch of samples for this tile
            generated = gen.model.generate(modalities=generate_keys, conditions=conditions_pre, n_samples=int(current_bs))  # type: ignore[attr-defined]

            # Post-process only generated modalities
            generated_post: Dict[str, torch.Tensor] = {}
            for key, val in generated.items():
                generated_post[key] = gen.post_process_data(val, key, gen.model_config.dataset)  # type: ignore[attr-defined]

            # Iterate each sample within the batch
            for local_idx, s_idx in enumerate(range(batch_start, batch_end)):
                paths = paths_batch[local_idx]

                # Save outputs for this sample
                for key, tensor in generated_post.items():
                    # Slice [B=current_bs,C,H,W] -> [C,H,W] for this sample
                    if tensor.ndim != 4:
                        raise ValueError(f"Unexpected tensor shape for {key}: {tuple(tensor.shape)}")
                    t = tensor[local_idx]
                    if append_scalar_csv(paths.latlon_csv, paths.time_csv, tile_key, key, t):
                        continue
                    # Image-like: save into generations layout
                    gen._save_generated_for_tile(tile_key, key, t, paths.sample_generations_root)  # type: ignore[attr-defined]
                    # Accumulate for std/hist per key
                    if key in per_key_arrays:
                        per_key_arrays[key].append(t.detach().cpu().numpy().astype(np.float32))

                # Visualise selected samples (conditions + generated) with GT for generated modalities
                if args.vis_every > 0 and (s_idx % int(args.vis_every) == 0):
                    from visualisations.visualise_bands import visualise_bands  # import lazily
                    from visualisations.merge_visualisations import merge_visualisations  # import lazily

                    # Build per-sample dicts: reuse conditions_raw (B=1) and select this sample from generated_post
                    generated_post_single: Dict[str, torch.Tensor] = {k: v[local_idx:local_idx+1] for k, v in generated_post.items()}
                    all_samples_for_vis: Dict[str, torch.Tensor] = {**conditions_raw, **generated_post_single}

                    # Build ground truth for generated modalities (follow copgen.py logic)
                    gt_for_vis: Dict[str, torch.Tensor] = {}
                    for key, tensor in all_samples_for_vis.items():
                        if key in generated_post_single.keys():
                            base = base_from_key(key)
                            if base in ("DEM", "S1RTC", "S2L1C", "S2L2A", "LULC"):
                                bands = gen._bands_for_key(key)  # type: ignore[attr-defined]
                                dh, dw = int(tensor.shape[-2]), int(tensor.shape[-1])
                                arr = gen._load_gt_for_tile(tile_key, base, (dh, dw), bands if base in ("S2L2A", "S2L1C", "S1RTC") else [])  # type: ignore[attr-defined]
                                gt_stack = torch.from_numpy(arr).unsqueeze(0).float()
                                gt_for_vis[key] = gt_stack
                            elif "3d_cartesian_lat_lon" in key:
                                xyz = gen.latlon_cache[(tile_key, -1, -1)]  # (3,)
                                xyz_t = torch.tensor(np.asarray(xyz, dtype=np.float32)).view(1, 1, 1, 3)
                                gt_decoded = gen.post_process_data(xyz_t, key, gen.model_config.dataset)  # type: ignore[attr-defined]
                                gt_for_vis[key] = gt_decoded.float()
                            elif "mean_timestamps" in key and gen.timestamps_cache is not None:
                                vec = gen.timestamps_cache[tile_key]
                                ts_t = torch.tensor(np.asarray(vec, dtype=np.float32)).view(1, 1, 1, 3)
                                gt_decoded = gen.post_process_data(ts_t, key, gen.model_config.dataset)  # type: ignore[attr-defined]
                                gt_for_vis[key] = gt_decoded.float()
                            else:
                                pass
                        elif key in conditions_raw.keys():
                            # for conditions we don't add GT (already represented by conditions themselves)
                            pass
                        else:
                            raise ValueError(f"Unexpected modality {key} in visualisation stage")

                    max_img_input_resolution = max(gen.model_config.all_modality_configs[modality].img_input_resolution[1] for modality in all_samples_for_vis.keys())
                    for modality, tensor in all_samples_for_vis.items():
                        save_dir = paths.sample_visual_root / modality
                        ensure_dir(save_dir)
                        inputs = gt_for_vis.get(modality)
                        # Resize should be the max img_input_resolution of all modalities
                        if inputs is not None:
                            input_frame_color = (0.2, 0.9, 0.3) # inputs are gt (softer green frame)
                            recon_frame_color = (0.3, 0.5, 1.0) # reconstructions are generated (softer blue frame)
                        else:
                            input_frame_color = None # inputs are none
                            recon_frame_color = (1.0, 0.3, 0.2) # reconstructions are the initial conditions (softer red frame)
                        view_histogram = True if modality.split('_')[0] in ['S2L2A', 'S2L1C', 'DEM', 'S1RTC'] else False
                        visualise_bands(inputs=inputs, reconstructions=tensor, save_dir=save_dir, milestone=s_idx, repeat_n_times=None, satellite_type=modality,
                                        view_histogram=view_histogram,
                                        resize=max_img_input_resolution, input_frame_color=input_frame_color, recon_frame_color=recon_frame_color)
                    merge_visualisations(results_path=str(paths.sample_visual_root), verbose=False, overwrite=False)

                # Evaluate this sample using existing evaluator
                sample_metrics = evaluate_copgen_run(Path(args.dataset_root), paths.sample_root, verbose=False)
                # Persist per-sample metrics JSON
                per_sample_json = paths.metrics_root / f"sample_{s_idx}.json"
                write_json(per_sample_json, sample_metrics)

                # Record base metrics per modality
                per_sample_metrics_base.append(sample_metrics)

                # Derive per-expanded-key metrics from per-band when possible
                per_key_metrics_for_sample: Dict[str, Dict[str, float]] = {}
                for modality, md in sample_metrics.items():
                    if modality in ("lat_lon", "timestamps", "LULC", "cloud_mask"):
                        continue
                    # Extract per-band dict if available
                    per_band = md.get("per_band") if isinstance(md, dict) else None
                    base_md = {k: v for k, v in md.items() if k != "per_band" and isinstance(v, (int, float))}
                    # Only consider keys belonging to this base
                    relevant_keys = {k: v for k, v in key_to_bands.items() if base_from_key(k) == modality}
                    per_key_metrics = aggregate_per_key_metrics(base_md, per_band if isinstance(per_band, dict) else {}, relevant_keys)
                    per_key_metrics_for_sample.update(per_key_metrics)
                per_sample_metrics_key.append(per_key_metrics_for_sample)

                # Optional cleanup: delete generations for this sample after evaluation
                if args.cleanup:
                    if paths.sample_generations_root.exists():
                        shutil.rmtree(paths.sample_generations_root)

        # Save visualisations for this tile, and merged visualisations for this tile
        # Build averaged predictions per key across samples for this tile and save visualisations
        aggregated_preds: Dict[str, torch.Tensor] = {}
        for key, arr_list in per_key_arrays.items():
            if not arr_list:
                continue
            # [S,C,H,W] -> average over samples -> [C,H,W]
            stack = np.stack(arr_list, axis=0)
            # For LULC (categorical) and cloud_mask, use mode instead of mean
            if base_from_key(key) == "LULC" or 'cloud_mask' in key:
                # Use mode (most frequent value) along sample axis
                from scipy import stats
                mode_result = stats.mode(stack, axis=0, keepdims=False)
                avg = mode_result.mode
            else:
                avg = np.mean(stack, axis=0)
            aggregated_preds[key] = torch.from_numpy(avg).unsqueeze(0).float()

        if aggregated_preds:
            from visualisations.visualise_bands import visualise_bands  # import lazily
            from visualisations.merge_visualisations import merge_visualisations  # import lazily

            # Build ground truth for generated modalities (for comparison)
            gt_for_vis_tile: Dict[str, torch.Tensor] = {}
            for key in aggregated_preds.keys():
                base = base_from_key(key)
                recon_t = aggregated_preds[key]  # [1,C,H,W]
                dh, dw = int(recon_t.shape[-2]), int(recon_t.shape[-1])
                if base in ("DEM", "S1RTC", "S2L1C", "S2L2A", "LULC"):
                    bands = gen._bands_for_key(key)  # type: ignore[attr-defined]
                    arr = gen._load_gt_for_tile(tile_key, base, (dh, dw), bands if base in ("S2L2A", "S2L1C", "S1RTC") else [])  # type: ignore[attr-defined]
                    gt_stack = torch.from_numpy(arr).unsqueeze(0).float()
                    gt_for_vis_tile[key] = gt_stack

            # Resize should be the max img_input_resolution of all aggregated modalities
            max_img_input_resolution = max(gen.model_config.all_modality_configs[modality].img_input_resolution[1] for modality in aggregated_preds.keys())

            # Save per-modality aggregated visualisations
            tile_visual_root = tile_root / "visualisations"
            for modality, recon_t in aggregated_preds.items():
                save_dir = tile_visual_root / modality
                ensure_dir(save_dir)
                inputs = gt_for_vis_tile.get(modality)
                if inputs is not None:
                    input_frame_color = (0.2, 0.9, 0.3)  # gt (softer green)
                    recon_frame_color = (0.3, 0.5, 1.0)  # predictions (softer blue)
                else:
                    input_frame_color = None
                    recon_frame_color = (1.0, 0.3, 0.2)  # conditions/none (softer red)
                view_histogram = True if modality.split('_')[0] in ['S2L2A', 'S2L1C', 'DEM', 'S1RTC'] else False
                visualise_bands(
                    inputs=inputs,
                    reconstructions=recon_t,
                    save_dir=save_dir,
                    milestone=0,
                    repeat_n_times=None,
                    satellite_type=modality,
                    view_histogram=view_histogram,
                    resize=max_img_input_resolution,
                    input_frame_color=input_frame_color,
                    recon_frame_color=recon_frame_color,
                )
            merge_visualisations(results_path=str(tile_visual_root), verbose=False, overwrite=False)

        # After all samples for this tile: compute std maps and histograms
        std_root = tile_root / "std_maps"
        hist_root = tile_root / "histograms"
        for key, arr_list in per_key_arrays.items():
            if not arr_list:
                continue
            # Stack to [S,C,H,W]
            stack = np.stack(arr_list, axis=0)  # float32
            bands = key_to_bands.get(key, [f"band_{i}" for i in range(stack.shape[1])])
            for b_idx, band_name in enumerate(bands):
                # std over samples -> [H,W]
                std_map = np.std(stack[:, b_idx, :, :], axis=0)
                out_png = std_root / key / "visualisations-0" / f"{band_name}.png"
                save_std_map_png(std_map, out_png, title=f"STD {key}:{band_name}")
                # histogram over all pixels across samples
                vals = stack[:, b_idx, :, :].reshape(-1)
                hist_png = hist_root / key / "visualisations-0" / f"{band_name}.png"
                save_histogram_png(vals, hist_png, title=f"Histogram {key}:{band_name}")

            # Also save merged/combined per-modality summaries
            # Combined std map: mean of per-band std maps (std over samples, averaged across bands)
            # Shape after std over samples: [C,H,W] (or [1,H,W])
            std_over_samples = np.std(stack, axis=0)
            if std_over_samples.ndim == 3:
                merged_std_map = np.mean(std_over_samples, axis=0)
            else:
                merged_std_map = std_over_samples
            merged_std_png = std_root / key / "visualisations-0" / "merged.png"
            save_std_map_png(merged_std_map, merged_std_png, title=f"STD {key}:merged")

            # Combined histogram: aggregate values across all bands, pixels, and samples
            merged_vals = stack.reshape(-1)
            merged_hist_png = hist_root / key / "visualisations-0" / "merged.png"
            save_histogram_png(merged_vals, merged_hist_png, title=f"Histogram {key}:merged")

        # Per-tile summary metrics (min/max/mean/std across samples)
        # Base modalities
        per_tile_summary_base: Dict[str, Dict[str, Dict[str, float]]] = {}
        # Expanded keys
        per_tile_summary_keys: Dict[str, Dict[str, float]] = {}

        # Aggregate base metrics
        modalities_in_any_sample = set()
        for sm in per_sample_metrics_base:
            modalities_in_any_sample.update(sm.keys())
        for modality in sorted(modalities_in_any_sample):
            # Collect numeric metric series across samples
            metrics_acc: Dict[str, List[float]] = {}
            for sm in per_sample_metrics_base:
                md = sm.get(modality, {})
                if isinstance(md, dict):
                    for k, v in md.items():
                        if k == "per_band":
                            continue
                        if isinstance(v, (int, float)):
                            metrics_acc.setdefault(k, []).append(float(v))
            per_tile_summary_base[modality] = {m: summarize_stats(vals) for m, vals in metrics_acc.items()}
            # Also collect into global aggregate
            for m, vals in metrics_acc.items():
                aggregate_metrics_by_modality.setdefault(modality, {}).setdefault(m, []).extend(vals)
            # Save per-modality histogram grid of metrics across samples
            if metrics_acc:
                # Determine subplot grid
                metric_names = sorted(metrics_acc.keys())
                n_metrics = len(metric_names)
                cols = min(2, n_metrics)
                rows = int(math.ceil(n_metrics / cols))
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.0, rows * 3.2))
                if isinstance(axes, np.ndarray):
                    axes_list = axes.flatten()
                else:
                    axes_list = [axes]
                for idx, metric_name in enumerate(metric_names):
                    ax = axes_list[idx]
                    vals = np.asarray(metrics_acc[metric_name], dtype=np.float64)
                    ax.hist(vals, bins=min(50, max(5, len(vals))), color="#3366cc", alpha=0.85)
                    mu = float(np.nanmean(vals)) if vals.size > 0 else math.nan
                    sd = float(np.nanstd(vals)) if vals.size > 0 else math.nan
                    ax.axvline(mu, color="red", linestyle="--", linewidth=1)
                    ax.set_title(metric_name)
                    ax.set_xlabel("value")
                    ax.set_ylabel("count")
                    ax.grid(alpha=0.2, linestyle=":")
                    ax.text(0.98, 0.95, f"mean={mu:.3f}\nstd={sd:.3f}", transform=ax.transAxes,
                            ha="right", va="top", fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, lw=0.0))
                # Hide any unused subplots
                for j in range(n_metrics, len(axes_list)):
                    axes_list[j].axis("off")
                fig.suptitle(f"{tile_key} · {modality} metrics across {len(per_sample_metrics_base)} samples", fontsize=12)
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                out_png = tile_root / "metrics" / f"{modality}_metrics_hist.png"
                ensure_dir(out_png.parent)
                fig.savefig(out_png, dpi=150)
                plt.close(fig)

        # Aggregate per-key metrics
        keys_in_any_sample = set()
        for sm in per_sample_metrics_key:
            keys_in_any_sample.update(sm.keys())
        for key in sorted(keys_in_any_sample):
            # Collect numeric metrics across samples for this key
            metrics_acc_k: Dict[str, List[float]] = {}
            for sm in per_sample_metrics_key:
                md = sm.get(key, {})
                for mk, mv in md.items():
                    if isinstance(mv, (int, float)):
                        metrics_acc_k.setdefault(mk, []).append(float(mv))
            # For the per-key summary, compute stats per metric and keep mean as representative (store mean of means)
            # Here, to keep structure compact, we store only mean/std/min/max for each metric for this tile
            per_tile_summary_keys[key] = {f"{mk}_mean": float(np.mean(mv)) for mk, mv in metrics_acc_k.items()}
            # Also append to global aggregate
            for mk, mv in metrics_acc_k.items():
                aggregate_metrics_by_key.setdefault(key, {}).setdefault(mk, []).extend(mv)

        write_json(tile_root / "metrics" / "per_tile_base.json", per_tile_summary_base)
        write_json(tile_root / "metrics" / "per_tile_keys.json", per_tile_summary_keys)

        # Pretty print per-tile summary
        print(f"\nTile summary metrics for {tile_key}")
        for modality in sorted(per_tile_summary_base.keys()):
            print(f"  [{modality}]")
            for metric_name, stats in per_tile_summary_base[modality].items():
                mean_v = stats.get("mean", math.nan)
                std_v = stats.get("std", math.nan)
                print(f"    {metric_name}: mean={mean_v:.4f} std={std_v:.4f}")

    # Across-tiles aggregate summary
    aggregate_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for modality, md in aggregate_metrics_by_modality.items():
        aggregate_summary.setdefault(modality, {})
        for metric_name, vals in md.items():
            aggregate_summary[modality][metric_name] = summarize_stats(vals)

    aggregate_keys_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key, md in aggregate_metrics_by_key.items():
        aggregate_keys_summary.setdefault(key, {})
        for metric_name, vals in md.items():
            aggregate_keys_summary[key][metric_name] = summarize_stats(vals)

    metrics_root = run_root / "metrics"
    ensure_dir(metrics_root)
    write_json(metrics_root / "aggregate_base.json", aggregate_summary)
    write_json(metrics_root / "aggregate_keys.json", aggregate_keys_summary)

    print("\nSaved aggregate summaries:")
    print(str(metrics_root / "aggregate_base.json"))
    print(str(metrics_root / "aggregate_keys.json"))

    # Pretty print aggregate summary
    print("\nOVERALL AGGREGATED METRICS (across tiles and samples)")
    for modality in sorted(aggregate_summary.keys()):
        print(f"  [{modality}]")
        for metric_name, stats in aggregate_summary[modality].items():
            mean_v = stats.get("mean", math.nan)
            std_v = stats.get("std", math.nan)
            print(f"    {metric_name}: mean={mean_v:.4f} std={std_v:.4f}")


if __name__ == "__main__":
    main()


