import os
import time
import functools
import contextlib
from collections import defaultdict
import numpy as np
import torch
import einops
import ml_collections
from concurrent.futures import ThreadPoolExecutor

from absl import app
from absl import flags
from ml_collections import config_flags
from tqdm.auto import tqdm

import utils
from sample_n_copgen import CopgenSamplingContext, build_copgen_dataloader
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver

from metrics import (
    SegmentationMetrics,
    ImageRegressionMetrics,
    SpatialMetrics,
    TemporalMetrics,
)

# visualisation utilities
from ddm.pre_post_process_data import post_process_data
from ddm.data import SatelliteDataset
from visualisations.visualise_bands import visualise_bands
from visualisations.merge_visualisations import merge_visualisations


def set_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


class _BenchTimer:
    """Lightweight benchmarking helper for scoped timing with CUDA sync."""
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.totals_s = defaultdict(float)
        self.last_s = {}
        self.counts = defaultdict(int)

    @contextlib.contextmanager
    def time(self, key: str):
        if self.use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self.use_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            self.totals_s[key] += dt
            self.last_s[key] = dt

    def inc(self, key: str, n: int = 1):
        self.counts[key] += n

    def get_last(self, key: str, default: float = 0.0) -> float:
        return float(self.last_s.get(key, default))

    def summary_lines(self, total_batches: int):
        lines = []
        keys = [
            'setup_ctx', 'setup_dataloader', 'dataloader_wait', 'batch_move_to_device',
            'encode_cond', 'duplicate_cond', 'sample', 'decode_preds', 'decode_inputs',
            'gt_load', 'gt_compute', 'metrics', 'visualize', 'batch_total'
        ]
        extra_keys = sorted([k for k in self.totals_s.keys() if k.startswith('metrics/')])
        all_keys = [k for k in keys if k in self.totals_s] + extra_keys
        lines.append("=== Benchmark summary (seconds) ===")
        for k in all_keys:
            tot = self.totals_s[k]
            avg = tot / max(total_batches, 1)
            lines.append(f"{k:>18}: total={tot:.3f}  avg/batch={avg:.3f}")
        lines.append("=== Top hotspots by total time ===")
        top = sorted(self.totals_s.items(), key=lambda kv: kv[1], reverse=True)
        for k, tot in top[:10]:
            avg = tot / max(total_batches, 1)
            lines.append(f"{k:>18}: total={tot:.3f}  avg/batch={avg:.3f}")
        return lines


def evaluate(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(config.get('seed'))
    config = ml_collections.FrozenConfigDict(config)

    utils.set_logger(log_level='info')

    bench = _BenchTimer(use_cuda=(device == 'cuda'))

    # Reuse common context and dataloader
    with bench.time('setup_ctx'):
        ctx = CopgenSamplingContext(config)
    with bench.time('setup_dataloader'):
        dataloader = build_copgen_dataloader(config)

    # Ensure debug output path
    if config.get('debug') is True and config.get('output_path') is None:
        raise ValueError("Output path is required when debug is True")
    if config.get('debug') and config.get('output_path'):
        os.makedirs(config.output_path, exist_ok=True)
        # Pre-create per-modality folders for clearer organization
        # Use context modalities so we cover generated, conditioned, and others as needed
        try:
            # ctx may not be available yet on first pass; guard with try
            pass
        except Exception:
            pass

    # Targets to evaluate (support multiple). Backward-compatible: allow single value.
    targets = []
    if hasattr(config, 'target_modalities') and config.target_modalities:
        targets = list(config.target_modalities)
    elif hasattr(config, 'target_modality') and config.target_modality:
        targets = [config.target_modality]
    else:
        raise ValueError("No target modalities specified in config")

    # Metric setup and optional raw GT datasets per target
    target_types = {t: config.all_modality_configs[t].modality_type for t in targets}
    patches_per_side = config.dataset.patches_per_side if hasattr(config, 'dataset') else None

    gt_dataset_by_target = {}
    filename_to_baseidx_by_target = {}

    if hasattr(config, 'raw_data_path') and config.raw_data_path:
        for target in targets:
            if target_types[target] != 'image':
                gt_dataset_by_target[target] = None
                filename_to_baseidx_by_target[target] = None
                continue
            modality_name = target
            sat_type = modality_name.split('_')[0]
            if 'cloud_mask' in modality_name:
                tif_bands = ['cloud_mask']
            else:
                parts = modality_name.split('_', 1)
                if len(parts) == 1:
                    tif_bands = [parts[0]]
                else:
                    # Special cases where bands are expected to be in a different order
                    if parts[1] == 'B02_B03_B04_B08':
                        tif_bands = ['B04', 'B03', 'B02', 'B08']
                    elif parts[1] == 'B05_B06_B07_B11_B12_B8A':
                        tif_bands = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
                    elif parts[1] == 'B01_B09':
                        tif_bands = ['B01', 'B09']
                    elif parts[1] == 'vh_vv':
                        tif_bands = ['vv', 'vh']
                    else:
                        tif_bands = parts[1].split('_')
            img_h, img_w = config.all_modality_configs[target].img_input_resolution
            # One-hot for categorical modalities
            one_hot_encode = None
            if 'LULC' in modality_name:
                one_hot_encode = int(config.all_modality_configs[target].img_bands)
            if 'cloud_mask' in modality_name:
                one_hot_encode = int(config.all_modality_configs[target].img_bands)
            _min_db = config.dataset.min_db.get(target, None) if hasattr(config.dataset, 'min_db') else None
            _max_db = config.dataset.max_db.get(target, None) if hasattr(config.dataset, 'max_db') else None
            _min_positive = config.dataset.min_positive.get(target, None) if hasattr(config.dataset, 'min_positive') else None
            gt_ds = SatelliteDataset(
                data_dir=f"{config.raw_data_path}/Core-{target.split('_')[0]}",
                image_size=[img_h, img_w],
                tif_bands=tif_bands,
                augment_horizontal_flip=False,
                satellite_type=sat_type,
                center_crop=False,
                normalize_to_neg_one_to_one=True,
                img_list=None,
                preprocess_bands=True,
                min_db=_min_db,
                max_db=_max_db,
                min_positive=_min_positive,
                resize_to=None,
                one_hot_encode=one_hot_encode,
                patchify=True,
                patches_per_side=patches_per_side,
            )
            gt_dataset_by_target[target] = gt_ds
            filename_to_baseidx_by_target[target] = {gc: i for i, gc in enumerate(gt_ds.grid_cells)}
    else:
        # Raise error?
        for target in targets:
            gt_dataset_by_target[target] = None
            filename_to_baseidx_by_target[target] = None

    # Metric accumulators per target
    metric_state_by_target = {}
    for target in targets:
        if 'LULC' in target:
            num_classes = config.all_modality_configs['LULC_LULC'].img_bands
            seg_metrics = SegmentationMetrics(num_classes=num_classes, topk=config.topk, ignore_index=0)
            metric_state_by_target[target] = {'task': 'classification', 'seg': seg_metrics}
        elif 'cloud_mask' in target:
            num_classes = config.all_modality_configs['S2L1C_cloud_mask'].img_bands
            seg_metrics = SegmentationMetrics(num_classes=num_classes, topk=config.topk)
            metric_state_by_target[target] = {'task': 'classification', 'seg': seg_metrics}
        elif '3d_cartesian_lat_lon' in target:
            metric_state_by_target[target] = {'task': 'spatial', 'spatial': SpatialMetrics(radii_km=tuple(config.acc_radii_km))}
        elif 'mean_timestamps' in target:
            metric_state_by_target[target] = {'task': 'temporal', 'temporal': TemporalMetrics()}
        else:
            metric_state_by_target[target] = {'task': 'image_reg', 'imgreg': ImageRegressionMetrics()}

    # Choose metric accumulator
    # We don't need this i think
    if 'LULC' in target:
        num_classes = config.all_modality_configs['LULC_LULC'].img_bands
        seg_metrics = SegmentationMetrics(num_classes=num_classes, topk=config.topk, ignore_index=0)
        metric_state = {'task': 'classification', 'seg': seg_metrics}
    elif 'cloud_mask' in target:
        num_classes = config.all_modality_configs['S2L1C_cloud_mask'].img_bands
        seg_metrics = SegmentationMetrics(num_classes=num_classes, topk=config.topk)
        metric_state = {'task': 'classification', 'seg': seg_metrics}
    elif '3d_cartesian_lat_lon' in target:
        metric_state = {'task': 'spatial', 'spatial': SpatialMetrics(radii_km=tuple(config.acc_radii_km))}
    elif 'mean_timestamps' in target:
        metric_state = {'task': 'temporal', 'temporal': TemporalMetrics()}
    else:
        # Regression for image-like modalities: DEM, S1RTC, S2L1C, S2L2A
        metric_state = {'task': 'image_reg', 'imgreg': ImageRegressionMetrics()}

    # Online evaluation loop
    progress = tqdm(dataloader, desc='eval', total=None)
    num_batches = 0
    prev_end_time = time.perf_counter()
    for batch in progress:
        now = time.perf_counter()
        bench.totals_s['dataloader_wait'] += (now - prev_end_time)
        with bench.time('batch_total'):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                moments, filenames = batch
            else:
                moments = batch
                filenames = None

            with bench.time('batch_move_to_device'):
                moments = {k: v.to(device) for k, v in moments.items()}

        # Prepare conditional latents
        with torch.no_grad():
            z_cond_map = {}
            with bench.time('encode_cond'):
                for m in ctx.modality_names:
                    if m in ctx.condition_set:
                        z_lat = ctx.encode_for_condition(moments[m], m)
                        z_cond_map[m] = z_lat

        # Get batch size from either z_cond_map or moments
        if z_cond_map:
            B = next(iter(z_cond_map.values())).shape[0]
        else:
            B = next(iter(moments.values())).shape[0]
        # Duplicate condition latents for n_samples
        effective_batch_size = B * (config.n_samples if hasattr(config, 'n_samples') and config.n_samples and config.n_samples > 1 else 1)
        if hasattr(config, 'n_samples') and config.n_samples and config.n_samples > 1:
            with bench.time('duplicate_cond'):
                for m in list(z_cond_map.keys()):
                    z_cond_map[m] = z_cond_map[m].repeat_interleave(int(config.n_samples), dim=0)

        # Sample using shared context
        with bench.time('sample'):
            _zs = ctx.sample_batch(effective_batch_size, z_cond_map)
        # Decode from latent to -1,1 range
        with bench.time('decode_preds'):
            samples_unstacked = [ctx.decode(_z, m) for m, _z in zip(ctx.modality_names, _zs)]

        # Decode inputs for conditioned modalities (for visualisation)
        inputs_decoded = {}
        if config.get('debug') and config.get('output_path'):
            with bench.time('decode_inputs'):
                for m in ctx.modality_names:
                    if m in ctx.condition_set:
                        inputs_decoded[m] = ctx.decode(z_cond_map[m], m)
                    else:
                        inputs_decoded[m] = None

        # Map of predictions per modality
        pred_map = {m: samples_unstacked[m_idx] for m_idx, m in enumerate(ctx.modality_names)}

        # Compute GT in decoded space for target. Load them in parallel
        gt_decoded_by_target = {}
        with torch.no_grad():
            for target in targets:
                if target_types[target] == 'image' and gt_dataset_by_target[target] is not None and filenames is not None:
                    with bench.time('gt_load'):
                        p2 = patches_per_side * patches_per_side
                        indices = []
                        for fname in filenames:
                            fname_str = fname if isinstance(fname, str) else str(fname)
                            if '_patch_' in fname_str:
                                grid_cell, patch_str = fname_str.split('_patch_')
                                try:
                                    patch_id = int(patch_str)
                                except ValueError:
                                    raise ValueError(f"Patch ID {patch_str} incorrect for {fname_str}")
                            else:
                                grid_cell = fname_str
                                patch_id = 0
                            base_idx = filename_to_baseidx_by_target[target][grid_cell]
                            indices.append(base_idx * p2 + patch_id)

                        def _fetch_image(idx):
                            return gt_dataset_by_target[target][idx]['image']

                        max_workers = min(len(indices), 16)
                        if max_workers <= 1:
                            gt_list = [_fetch_image(i) for i in indices]
                        else:
                            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                                gt_list = list(ex.map(_fetch_image, indices))
                        gt_decoded_by_target[target] = torch.stack(gt_list, dim=0).to(device)
                        if hasattr(config, 'n_samples') and config.n_samples and config.n_samples > 1:
                            gt_decoded_by_target[target] = gt_decoded_by_target[target].repeat_interleave(int(config.n_samples), dim=0)
                else:
                    with bench.time('gt_compute'):
                        if config.all_modality_configs[target].modality_type == 'image':
                            tgt_lat = ctx.encode_for_condition(moments[target], target)
                            gt_decoded_by_target[target] = ctx.decode(tgt_lat, target)
                        else:
                            gt_decoded_by_target[target] = moments[target]
                    if hasattr(config, 'n_samples') and config.n_samples and config.n_samples > 1:
                        gt_decoded_by_target[target] = gt_decoded_by_target[target].repeat_interleave(int(config.n_samples), dim=0)

        # Update metrics per target
        postfix = {}
        with bench.time('metrics'):
            for target in targets:
                metric_state = metric_state_by_target[target]
                with bench.time(f'metrics/{target}'):
                    if metric_state['task'] == 'classification':
                        seg_metrics = metric_state['seg']
                        seg_metrics.update_from_logits(pred_map[target], gt_decoded_by_target[target])
                        postfix[f'{target}/top1'] = seg_metrics.overall_top1()
                        postfix[f'{target}/topk'] = seg_metrics.overall_topk()
                    elif metric_state['task'] == 'spatial':
                        pred_ll = post_process_data(pred_map[target], '3d_cartesian_lat_lon', config.dataset)
                        tgt_ll = post_process_data(gt_decoded_by_target[target], '3d_cartesian_lat_lon', config.dataset)
                        pred_ll = pred_ll.view(pred_ll.shape[0], -1)
                        tgt_ll = tgt_ll.view(tgt_ll.shape[0], -1)
                        metric_state['spatial'].update(pred_ll, tgt_ll)
                        postfix[f'{target}/median_km'] = metric_state['spatial'].summary()['median_km']
                    elif metric_state['task'] == 'temporal':
                        pred_dt = post_process_data(pred_map[target], 'mean_timestamps', config.dataset)
                        tgt_dt = post_process_data(gt_decoded_by_target[target], 'mean_timestamps', config.dataset)
                        pred_dt = pred_dt.view(pred_dt.shape[0], -1)
                        tgt_dt = tgt_dt.view(tgt_dt.shape[0], -1)
                        metric_state['temporal'].update(pred_dt, tgt_dt)
                        ts = metric_state['temporal'].summary()
                        postfix[f'{target}/median_days'] = ts['median_days']
                        postfix[f'{target}/median_abs_months'] = ts['median_abs_months']
                        postfix[f'{target}/month_acc'] = ts['month_acc']
                        postfix[f'{target}/year_acc'] = ts['year_acc']
                    else:
                        pred_phys = post_process_data(pred_map[target], target, config.dataset)
                        gt_phys = post_process_data(gt_decoded_by_target[target], target, config.dataset)

                        if 'S2L2A' in target or 'S2L1C' in target:  # Convert to reflectance
                            pred_phys = pred_phys * 0.0001
                            gt_phys = gt_phys * 0.0001
                        elif 'S1RTC' in target:  # Convert to dB
                            pred_phys = 10 * torch.log10(pred_phys)
                            gt_phys = 10 * torch.log10(gt_phys)
                        elif 'DEM' in target:  # Convert to meters
                            pass

                        metric_state['imgreg'].update(pred_phys, gt_phys)
                        s = metric_state['imgreg'].summary()
                        postfix[f'{target}/mae'] = s['mae']
                        postfix[f'{target}/rmse'] = s['rmse']
                        postfix[f'{target}/ssim'] = s['ssim']
                        postfix[f'{target}/psnr'] = s['psnr']

        t_post = {
            'wait': f"{bench.get_last('dataloader_wait'):.3f}",
            'enc': f"{bench.get_last('encode_cond'):.3f}",
            'samp': f"{bench.get_last('sample'):.3f}",
            'dec': f"{bench.get_last('decode_preds'):.3f}",
            'gt': f"{bench.get_last('gt_load') + bench.get_last('gt_compute'):.3f}",
            'met': f"{bench.get_last('metrics'):.3f}",
            'vis': f"{bench.get_last('visualize'):.3f}",
            'tot': f"{bench.get_last('batch_total'):.3f}",
        }
        progress.set_postfix({**postfix, **t_post})

        # Debug: visualize all modalities (conditions and predictions)
        if config.get('debug') and config.get('output_path'):
            with bench.time('visualize'):
                for m_idx, m in enumerate(ctx.modality_names):
                    save_dir = os.path.join(config.output_path, m)
                    os.makedirs(save_dir, exist_ok=True)
                    _min_db = config.dataset.min_db.get(m, None) if hasattr(config.dataset, 'min_db') else None
                    _max_db = config.dataset.max_db.get(m, None) if hasattr(config.dataset, 'max_db') else None
                    _min_positive = config.dataset.min_positive.get(m, None) if hasattr(config.dataset, 'min_positive') else None
                    if m in targets: # For targets, show the ground truth and the prediction
                        gt_post_processed = post_process_data(gt_decoded_by_target[m], m, config.dataset)
                        pred_post_processed = post_process_data(pred_map[m], m, config.dataset)
                        visualise_bands(
                            inputs=gt_post_processed,
                            reconstructions=pred_post_processed,
                            save_dir=save_dir,
                            n_images_to_log=min(8, effective_batch_size),
                            milestone=num_batches,
                            satellite_type=m,
                            view_histogram=True if m.split('_')[0] in ['S2L2A', 'S2L1C', 'DEM', 'S1RTC'] else False,
                        )
                    elif m in ctx.condition_set: # For conditioned modalities, show the decoded input
                        condition_post_processed = post_process_data(inputs_decoded[m], m, config.dataset)
                        visualise_bands(
                            inputs=None,
                            reconstructions=condition_post_processed,
                            save_dir=save_dir,
                            n_images_to_log=min(8, effective_batch_size),
                            milestone=num_batches,
                            satellite_type=m,
                        )
                    else: # For unconditioned modalities, show the generated sample
                        sample_post_processed = post_process_data(samples_unstacked[m_idx], m, config.dataset)
                        visualise_bands(
                            inputs=None,
                            reconstructions=sample_post_processed,
                            save_dir=save_dir,
                            n_images_to_log=min(8, effective_batch_size),
                            milestone=num_batches,
                            satellite_type=m,
                        )

                # Merge visualisations
                merge_visualisations(results_path=config.output_path, verbose=False)

        num_batches += 1
        if config.max_batches and num_batches >= config.max_batches:
            break
        prev_end_time = time.perf_counter()

    # Final metrics per target
    results_by_target = {}
    for target in targets:
        metric_state = metric_state_by_target[target]
        if metric_state['task'] == 'classification':
            res = metric_state['seg'].summary()
        elif metric_state['task'] == 'spatial':
            res = metric_state['spatial'].summary()
        elif metric_state['task'] == 'temporal':
            res = metric_state['temporal'].summary()
        else:
            res = metric_state['imgreg'].summary()
        results_by_target[target] = res

    print("Final metrics:", results_by_target)
    for line in bench.summary_lines(total_batches=num_batches):
        print(line)
    return results_by_target


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Configuration.", lock_config=False)
flags.DEFINE_string("data_path", None, "Path to the LMDB features")
flags.DEFINE_string("nnet_path", None, "The COPGEN model to evaluate.")
flags.DEFINE_string("target", None, "Comma-separated modalities to evaluate (e.g., LULC_LULC or S2L1C_cloud_mask,DEM_DEM)")
flags.DEFINE_string("generate", None, "Comma-separated list of modalities to generate (must include target)")
flags.DEFINE_string("condition", None, "Comma-separated list of modalities to condition on")
flags.DEFINE_integer("batch_size", 6, "Batch size")
flags.DEFINE_integer("n_samples", 1, "Number of stochastic samples per conditional input")
flags.DEFINE_integer("seed", None, "Random seed")
flags.DEFINE_integer("topk", 5, "Top-k for segmentation metrics")
flags.DEFINE_multi_integer("acc_radii_km", [1, 10, 100, 500], "Radii for Accuracy@R in km (lat/lon)")
flags.DEFINE_integer("max_batches", None, "Optional cap for number of batches during eval")
flags.DEFINE_string("output_path", None, "Directory to save debug outputs (optional)")
flags.DEFINE_boolean("debug", False, "If set, save predictions and ground-truth for debugging")
flags.DEFINE_string("raw_data_path", None, "Path to the raw pixel dataset (TIFFs) for GT evaluation (optional)")


def main(argv):
    cfg = FLAGS.config
    cfg.nnet_path = FLAGS.nnet_path
    cfg.data_path = FLAGS.data_path
    cfg.batch_size = FLAGS.batch_size
    cfg.n_samples = FLAGS.n_samples if FLAGS.n_samples else 1

    if FLAGS.seed is not None:
        cfg.seed = FLAGS.seed

    modality_names = list(cfg.z_shapes.keys())
    cfg.modalities = modality_names

    if FLAGS.generate is None:
        raise ValueError("--generate flag is mandatory")
    if FLAGS.target is None:
        raise ValueError("--target flag is mandatory (comma-separated list supported)")

    target_modalities = [m for m in FLAGS.target.split(',') if m]
    cfg.target_modalities = target_modalities
    # Keep backward compat for any downstream code expecting a single target
    cfg.target_modality = target_modalities[0] if target_modalities else None

    cfg.generate_modalities = [m for m in FLAGS.generate.split(',') if m]
    cfg.condition_modalities = [m for m in FLAGS.condition.split(',')] if FLAGS.condition else []

    order_index = {m: i for i, m in enumerate(modality_names)}
    cfg.generate_modalities = sorted(cfg.generate_modalities, key=lambda x: order_index[x])
    cfg.condition_modalities = sorted(cfg.condition_modalities, key=lambda x: order_index[x])

    unknown = [m for m in (cfg.generate_modalities + cfg.condition_modalities + target_modalities) if m not in modality_names]
    if unknown:
        raise ValueError(f"Unknown modality names: {unknown}. Available: {modality_names}")

    if set(cfg.generate_modalities) & set(cfg.condition_modalities):
        raise ValueError("Generate and condition modalities must be disjoint")

    not_in_generate = [m for m in target_modalities if m not in cfg.generate_modalities]
    if not_in_generate:
        raise ValueError(f"All targets must be included in --generate. Missing: {not_in_generate}")

    cfg.topk = FLAGS.topk
    cfg.acc_radii_km = FLAGS.acc_radii_km
    cfg.max_batches = FLAGS.max_batches
    cfg.output_path = FLAGS.output_path
    cfg.debug = FLAGS.debug
    cfg.raw_data_path = FLAGS.raw_data_path

    evaluate(cfg)


if __name__ == "__main__":
    app.run(main) 