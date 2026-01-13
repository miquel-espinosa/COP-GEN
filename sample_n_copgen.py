

"""

THIS SCRIPT NEEDS TO BE FIXED. Update with the new COPGEN inference script.

"""


import os
import time
import torch
import ml_collections

from absl import app
from absl import flags
from ml_collections import config_flags
from tqdm.auto import tqdm
import utils
from libs.copgen import CopgenModel, build_copgen_dataloader

from visualisations.visualise_bands import visualise_bands
from visualisations.merge_visualisations import merge_visualisations
from ddm.pre_post_process_data import post_process_data

def evaluate(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    utils.set_seed(config.get('seed'))
    config = ml_collections.FrozenConfigDict(config)
    utils.set_logger(log_level='info')

    os.makedirs(config.output_path, exist_ok=True)

    model_config = ml_collections.ConfigDict(config.to_dict()) # Convert config to mutable ConfigDict
    model = CopgenModel(model_path=model_config.nnet_path, config_path=model_config, device=device, seed=model_config.get('seed'))

    # Optional dataloader only when we actually condition on inputs
    dataloader = None
    if len(model_config.condition_modalities) > 0:
        dataloader = build_copgen_dataloader(model_config)

        def get_data_generator():
            for data in tqdm(dataloader, desc='epoch'):
                yield data

        data_generator = get_data_generator()
    else:
        data_generator = [None]  # single unconditional batch

    # Sampling steps (fallback to 50 if not present)
    sample_steps = int(model_config.sample.sample_steps) if hasattr(model_config.sample, 'sample_steps') else 50

    for idx_batch, batch in enumerate(data_generator):
        
        conditions = {} # Prepare conditions (if any)
        effective_batch_size = model_config.batch_size * (model_config.n_samples if model_config.n_samples and model_config.n_samples > 1 else 1)

        if batch is not None:
            # batch is (moments_dict, filename)
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                moments, filenames = batch
            else:
                moments = batch
                filenames = None

            # Move to device
            moments = {k: v.to(device) for k, v in moments.items()}

            # Collect only the modalities we condition on
            for m in model_config.condition_modalities:
                if m in moments:
                    conditions[m] = moments[m]

            # # Determine batch size from conditions
            # if len(conditions) > 0:
            #     B = next(iter(conditions.values())).shape[0]
            #     effective_batch_size = B * (model_config.n_samples if model_config.n_samples and model_config.n_samples > 1 else 1)

        # Generate
        start_time = time.time()
        generated = model.generate(
            modalities=model_config.generate_modalities,
            conditions=conditions if len(conditions) > 0 else None,
            n_samples=int(model_config.n_samples) if model_config.get('n_samples') else 1,
            batch_size=(next(iter(conditions.values())).shape[0] if len(conditions) > 0 else int(model_config.batch_size)),
            sample_steps=sample_steps,
            return_latents=False,
        )
        sample_time = time.time() - start_time

        # Normalize generated to dict for a unified code path
        if isinstance(generated, torch.Tensor):
            generated = {model_config.generate_modalities[0]: generated}

        # For conditioned modalities, also decode inputs for side-by-side visualisation
        inputs_decoded = {m: None for m in model.modality_names}
        if len(conditions) > 0:
            with torch.no_grad():
                z_cond_map = model.encode(conditions, n_samples=int(model_config.n_samples) if model_config.get('n_samples') else 1)
                decoded_cond, _ = model.decode(z_cond_map, modalities=list(z_cond_map.keys()), return_latents=False)
                for m in decoded_cond:
                    inputs_decoded[m] = decoded_cond[m]

        # Save visualisations (only generated and conditioned modalities)
        generate_set = set(model_config.generate_modalities)
        condition_set = set(model_config.condition_modalities)

        for m in model.modality_names:
            if (m not in generate_set) and (m not in condition_set):
                continue
            save_dir = os.path.join(config.output_path, m)
            os.makedirs(save_dir, exist_ok=True)
            _min_db = config.dataset.min_db.get(m, None) if hasattr(config.dataset, 'min_db') else None
            _max_db = config.dataset.max_db.get(m, None) if hasattr(config.dataset, 'max_db') else None
            _min_positive = config.dataset.min_positive.get(m, None) if hasattr(config.dataset, 'min_positive') else None
            recon = inputs_decoded[m] if m in condition_set else generated[m]
            recon_post_processed = post_process_data(recon, m, config.dataset)
            visualise_bands(
                inputs=None,
                reconstructions=recon_post_processed,
                save_dir=save_dir,
                n_images_to_log=min(8, effective_batch_size),
                milestone=idx_batch,
                satellite_type=m,
            )

        print(f"Batch {idx_batch}: sampled {effective_batch_size} with {sample_steps} steps in {sample_time:.2f}s")

        # Merge visualisations
        merge_visualisations(results_path=config.output_path, verbose=False)

# -------------------------------
# CLI
# -------------------------------

# Move flag definitions under main guard to allow safe import from other modules
if __name__ == "__main__":
    FLAGS = flags.FLAGS
    config_flags.DEFINE_config_file(
        "config", None, "Configuration.", lock_config=False)
    flags.DEFINE_string("data_path", None, "Path to the LMDB features")
    flags.DEFINE_string("nnet_path", None, "The COPGEN model to evaluate.")
    flags.DEFINE_string("output_path", None, "Directory to save the generated outputs")
    flags.DEFINE_integer("n_samples", 1, "Number of stochastic samples per conditional input")
    flags.DEFINE_string("generate", None, "Comma-separated list of modalities to generate")
    flags.DEFINE_string("condition", None, "Comma-separated list of modalities to condition on")
    flags.DEFINE_integer("batch_size", 6, "Batch size (number of condition items per step)")
    flags.DEFINE_integer("seed", None, "Random seed for reproducibility (overrides config seed)")

    def main(argv):
        cfg = FLAGS.config
        cfg.nnet_path = FLAGS.nnet_path
        cfg.data_path = FLAGS.data_path
        cfg.n_samples = FLAGS.n_samples if FLAGS.n_samples else 1
        cfg.batch_size = FLAGS.batch_size
        cfg.seed = FLAGS.seed if FLAGS.seed is not None else cfg.seed

        # Modalities canonical order: use z_shapes keys as in training
        modality_names = list(cfg.z_shapes.keys())
        cfg.modalities = modality_names

        if FLAGS.generate is None:
            raise ValueError("--generate flag is mandatory")

        # Parse generate and condition lists
        cfg.generate_modalities = [m for m in FLAGS.generate.split(',') if m]
        cfg.condition_modalities = [m for m in FLAGS.condition.split(',')] if FLAGS.condition else []

        # Sort by canonical order
        order_index = {m: i for i, m in enumerate(modality_names)}
        cfg.generate_modalities = sorted(cfg.generate_modalities, key=lambda x: order_index[x])
        cfg.condition_modalities = sorted(cfg.condition_modalities, key=lambda x: order_index[x])

        # Validate modalities
        unknown = [m for m in (cfg.generate_modalities + cfg.condition_modalities) if m not in modality_names]
        if unknown:
            raise ValueError(f"Unknown modality names: {unknown}. Available: {modality_names}")

        # Ensure no overlap
        if set(cfg.generate_modalities) & set(cfg.condition_modalities):
            raise ValueError("Generate and condition modalities must be disjoint")

        # Build masks
        cfg.generate_modalities_mask = [m in cfg.generate_modalities for m in modality_names]
        cfg.condition_modalities_mask = [m in cfg.condition_modalities for m in modality_names]

        # Output path formatting
        clean_generate = [m.replace('_', '') for m in cfg.generate_modalities]
        if cfg.condition_modalities:
            clean_condition = [m.replace('_', '') for m in cfg.condition_modalities]
            output_dir = f"condition_{'_'.join(clean_condition)}_generate_{'_'.join(clean_generate)}_{cfg.n_samples}samples"
        else:
            output_dir = f"generate_{'_'.join(clean_generate)}_{cfg.n_samples}samples"

        if FLAGS.output_path is None:
            raise ValueError("--output_path must be provided")
        cfg.output_path = os.path.join(FLAGS.output_path, output_dir)

        evaluate(cfg)

    app.run(main)
