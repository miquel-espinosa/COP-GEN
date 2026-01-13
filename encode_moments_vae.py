import yaml
import argparse
import torch
import os
import random
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name
from ddm.pre_post_process_data import post_process_data
from train_vae import load_conf
import numpy as np
from visualisations.visualise_bands import visualise_bands
import matplotlib.pyplot as plt
from accelerate import Accelerator
from concurrent.futures import ThreadPoolExecutor
import lmdb
import pickle
import shutil
import zarr
from zarr.core.buffer.cpu import Buffer
import zipfile
import io

SATELLITE_IMG_SIZES = {
    "S2L2A": {
        "B02_B03_B04_B08": 1068,     # 10m bands
        "B05_B06_B07_B11_B12_B8A": 534,     # 20m bands
        "B01_B09": 178,     # 60m bands
    },
    "S2L1C": {
        "B02_B03_B04_B08": 1068,     # 10m bands
        "B05_B06_B07_B11_B12_B8A": 534,     # 20m bands
        "B01_B09_B10": 178,     # 60m bands
        "cloud_mask": 1068,     # 10m bands
    },
    "S1RTC": {
        "vh_vv": 1068,     # 10m bands
    },
    "DEM": {
        "DEM": 356,     # 30m bands
    },
    "LULC": {
        "LULC": 1068,     # 10m bands
    },
    "GOOGLE_EMBEDS": {
        "GOOGLE_EMBEDS": 1068,     # 10m bands
    }
}


def calculate_patches_per_side(satellite_type, tif_bands, patch_size):
    """
    Calculate the number of patches per side based on satellite image size and patch size.
    Args:
        satellite_type: Type of satellite (e.g., "S2L2A")
        tif_bands: List of bands being used
        patch_size: Size of each patch
    Returns:
        Number of patches per side (integer)
    """
    # Create band key for lookup
    band_key = "_".join(sorted(tif_bands))
    
    # Get the satellite image size
    if satellite_type in SATELLITE_IMG_SIZES:
        if band_key in SATELLITE_IMG_SIZES[satellite_type]:
            img_size = SATELLITE_IMG_SIZES[satellite_type][band_key]
        else:
            raise ValueError(f"Band key {band_key} not found in SATELLITE_IMG_SIZES for satellite type {satellite_type}")
    else:
        raise ValueError(f"Satellite type {satellite_type} not found in SATELLITE_IMG_SIZES")
    
    # Calculate patches per side: (img_size // patch_size) + 1 to ensure full coverage
    patches_per_side = (img_size // patch_size) + 1
    
    return patches_per_side


def parse_args():
    parser = argparse.ArgumentParser(description="Encode and decode images using a pretrained VAE")
    parser.add_argument("--cfg", help="Config file path", type=str, required=True)
    parser.add_argument("--checkpoint_path", help="Path to VAE checkpoint (overrides config)", type=str)
    parser.add_argument("--output_dir", help="Output directory for reconstructed images", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size for processing", type=int, default=4)
    parser.add_argument("--num_workers", help="Number of workers for data loading", type=int, default=2)
    parser.add_argument("--patchify", help="Patchify the input images", action="store_true")
    parser.add_argument("--plot_latents", help="Plot latents (without scaling)", action="store_true")
    parser.add_argument("--latents_only", help="Only encode images and save latents without reconstruction", action="store_true")
    parser.add_argument("--train_test_ratio", help="Ratio for train/test split", type=float, default=0.9)
    parser.add_argument("--seed", help="Random seed for reproducibility", type=int, default=42)
    parser.add_argument("--create_train_test", help="Only create train/test split files", action="store_true")
    parser.add_argument("--save_workers", help="Thread workers for async npy saving", type=int, default=4)
    parser.add_argument("--flush_every_batches", help="Flush async save pool every N batches (0 disables)", type=int, default=100)
    parser.add_argument("--save_to_lmdb", help="Save latent arrays to LMDB instead of individual npy files", action="store_true")
    parser.add_argument("--lmdb_map_size", help="LMDB map size in bytes (default: 10TB)", type=int, default=10 * (1 << 40)) # 10TB
    parser.add_argument("--lmdb_fast", help="Use LMDB performance-oriented flags (map_async, sync=False, metasync=False)", action="store_true")
    parser.add_argument("--data_dir", help="Override data.data_dir from config", type=str)
    parser.add_argument("--lulc_align", help="Align with LULC available tiles", action="store_true")
    parser.add_argument("--resume", help="Resume from previous run without deleting existing LMDB or progress files", action="store_true")
    parser.add_argument("--save_to_zarr", help="Save latents as zipped Zarr in the LMDB", action="store_true")
    parser.add_argument("--save_common_grid_cells", help="Save common grid cells to a txt file", type=str, default=None)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    if args.data_dir:
        args.cfg.setdefault("data", {})["data_dir"] = args.data_dir
    else:
        args.data_dir = args.cfg["data"]["data_dir"]
    return args

# -----------------------------------------------------------------------------
# Zarr helpers
# -----------------------------------------------------------------------------

def save_zarr_zip_blob(arr):
    """Serialize *arr* into a Zarr ZipStore that lives entirely in memory and write it to disk."""

    store = zarr.storage.MemoryStore()
    zarr.save(store, arr)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for key, value in store._store_dict.items():
            zf.writestr(str(key), value.to_bytes())
    buffer.seek(0)
    return buffer.getvalue()


def load_zarr_zip_blob(blob):
    """Deserialize a Zarr ZipStore from an in-memory ZIP blob and return the Zarr array or group."""

    buffer = io.BytesIO(blob)

    # Read the ZIP content and populate the MemoryStore
    store = zarr.storage.MemoryStore()
    with zipfile.ZipFile(buffer, mode="r") as zf:
        store_dict = {
            name: Buffer.from_bytes(zf.read(name)) for name in zf.namelist()
        }

    # Populate the store
    store._store_dict.update(store_dict)

    # Open the store (Zarr v3 compatible)
    return zarr.open(store, mode='r')


def main(args):
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Patchify is mandatory for the moment
    if not args.patchify:
        raise ValueError("Patchify is mandatory for the moment. Set --patchify")
    
    if not args.lulc_align:
        raise ValueError("LULC alignment is mandatory for the moment. Set --lulc_align")
    
    # Construct PKL_PATHS from data_dir
    data_dir_path = Path(args.data_dir)
    if data_dir_path.name.startswith('Core-'):
        BASEDIR = str(data_dir_path.parent)
    else:
        raise ValueError(f"data_dir must be a specific modality directory with 'Core-*' pattern: {args.data_dir}")
    PKL_PATHS = {
        "DEM":   f'{BASEDIR}/Core-DEM/.cache_DEM.pkl',
        "LULC":  f'{BASEDIR}/Core-LULC/.cache_LULC.pkl',
        "S1RTC": f'{BASEDIR}/Core-S1RTC/.cache_S1RTC.pkl',
        "S2L2A": f'{BASEDIR}/Core-S2L2A/.cache_S2L2A.pkl',
        "S2L1C": f'{BASEDIR}/Core-S2L1C/.cache_S2L1C.pkl',
    }
    
    data_cfg = args.cfg["data"]
    bands = "_".join(sorted(data_cfg['tif_bands'])) if 'tif_bands' in data_cfg else "_".join([data_cfg['band_name']]) if 'band_name' in data_cfg else ''
    
    # ----- Find Common IDs -----
    # Since LULC was downloaded separately, we need to find the intersection of the LULC and the other satellite datasets
    id_sets = {}
    for name, path in PKL_PATHS.items():
        # Skip Google Embeds PKL if we're not processing Google Embeds data
        if name == 'GOOGLE_EMBEDS' and data_cfg['satellite_type'] != 'GOOGLE_EMBEDS':
            continue
            
        if not os.path.exists(path):
            raise FileNotFoundError(f"PKL file not found at path: {path}")
        with open(path, 'rb') as f:
            data = pickle.load(f)
        ids = {item[0] for item in data}  # Extract first item in each tuple
        id_sets[name] = ids
        print(f"{name}: {len(ids)} IDs loaded")
    common_ids = set.intersection(*id_sets.values())
    print(f"\nTotal common IDs across all datasets: {len(common_ids)}")
    
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save common grid cells to a txt file if requested
    if args.save_common_grid_cells:
        with open(output_dir / f'{args.save_common_grid_cells}', 'w') as f:
            f.write("\n".join(sorted(common_ids)))
            
        print(f"Saved {len(common_ids)} common IDs to {output_dir / f'{args.save_common_grid_cells}'}")
        exit()

    # --------------------------------------------
    # Progress tracking setup (for job resumption)
    # --------------------------------------------
    progress_dir = output_dir / 'progress' / f'{data_cfg["satellite_type"]}_{bands}'
    progress_dir.mkdir(exist_ok=True, parents=True)

    # Collect already processed filenames (keys) from any previous run
    done_entries: set[str] = set()
    for pf in progress_dir.glob('progress_rank*.txt'):
        try:
            with open(pf, 'r') as f:
                done_entries.update(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            # File might disappear if another job rotates it while we are reading – ignore safely
            pass

    print(f"Found {len(done_entries)} previously processed entries that will be skipped in this run.")
    
    # Only create visualisation folder if we're doing reconstruction
    if not args.latents_only and not args.create_train_test:
        vis_folder = output_dir / 'visualisations'
        vis_folder.mkdir(exist_ok=True)
    
    
    
    # Check if data_cfg['data_dir']/.cache_<satellite_type>.pkl exists, otherwise, copy it from PKL_PATHS
    if not os.path.exists(f"{data_cfg['data_dir']}/.cache_{data_cfg['satellite_type']}.pkl"):
        print(f"Cache file not found at {data_cfg['data_dir']}/.cache_{data_cfg['satellite_type']}.pkl. Copying from {PKL_PATHS[data_cfg['satellite_type']]}")
        shutil.copy(PKL_PATHS[data_cfg['satellite_type']], f"{data_cfg['data_dir']}/.cache_{data_cfg['satellite_type']}.pkl")
    
    # Mandatory configurations for data config when encoding latents
    data_cfg['common_id_set'] = common_ids
    data_cfg['augment_horizontal_flip'] = False
    data_cfg['center_crop'] = False
    data_cfg["patchify"] = args.patchify
    data_cfg["skip_done_entries"] = list(done_entries)
    data_cfg["patches_per_side"] = calculate_patches_per_side(data_cfg['satellite_type'], data_cfg['tif_bands'], data_cfg['image_size'][0])
    print(f"Calculated {data_cfg['patches_per_side']} patches per side for {data_cfg['satellite_type']} with patch size {data_cfg['image_size'][0]}")
    assert data_cfg['patches_per_side'] == 6, "Patches per side should be 6 with current settings"
    dataset = construct_class_by_name(**data_cfg)
    
    # Get normalization info from data config
    normalize_to_neg_one_to_one = data_cfg.get('normalize_to_neg_one_to_one', False)
    
    # Define train/test file paths
    train_file = output_dir / 'train.txt'
    test_file = output_dir / 'test.txt'
    
    # Handle train/test split creation
    if args.create_train_test:
        # Check if train/test files already exist
        if train_file.exists() or test_file.exists():
            raise ValueError(
                f"Train/test split files already exist in {output_dir}. "
                f"Delete them first if you want to recreate the splits."
            )
            
        # Create train/test split
        all_grid_cells = dataset.grid_cells
        random.shuffle(all_grid_cells)
        n_train = int(len(all_grid_cells) * args.train_test_ratio)
        train_grid_cells = set(all_grid_cells[:n_train])
        test_grid_cells = set(all_grid_cells[n_train:])
        
        # Save train/test split info to files
        with open(train_file, 'w') as f:
            f.write("\n".join(sorted(train_grid_cells)))
        
        with open(test_file, 'w') as f:
            f.write("\n".join(sorted(test_grid_cells)))
        
        print(f"Created train/test split:")
        print(f"- Train: {len(train_grid_cells)} grid cells, saved to {output_dir}/train.txt")
        print(f"- Test: {len(test_grid_cells)} grid cells, saved to {output_dir}/test.txt")
        return  # Exit after creating split files
    
    # For the main encoding workflow, check if train/test split files exist
    if not train_file.exists() or not test_file.exists():
        raise ValueError(
            f"Train/test split files not found in {output_dir}. "
            f"Run with --create_train_test flag first to create them."
        )
    
    # Load train/test splits from files
    with open(train_file, 'r') as f:
        train_grid_cells = set(f.read().splitlines())
    
    with open(test_file, 'r') as f:
        test_grid_cells = set(f.read().splitlines())
    
    print(f"Loaded train/test split:")
    print(f"- Train: {len(train_grid_cells)} grid cells")
    print(f"- Test: {len(test_grid_cells)} grid cells")
    
    accelerator = Accelerator()
    
    # For GOOGLE_EMBEDS, we don't need a model as the data is already embedded
    if data_cfg["satellite_type"] == "GOOGLE_EMBEDS":
        model = None
        accelerator.print("Using GOOGLE_EMBEDS - no model needed as data is already embedded")
    else:
        # Load model from config
        model_cfg = args.cfg['model']
        
        # Override checkpoint path if provided in CLI
        if args.checkpoint_path:
            model_cfg['ckpt_path'] = args.checkpoint_path
        
        # Ensure checkpoint path exists
        if 'ckpt_path' not in model_cfg or not os.path.exists(model_cfg['ckpt_path']):
            raise ValueError(f"Checkpoint path not found: {model_cfg.get('ckpt_path', 'Not set')}")
        
        accelerator.print(f"Loading model from checkpoint: {model_cfg['ckpt_path']}")
        model = construct_class_by_name(**model_cfg)
        model.eval()
    
    device = accelerator.device

    # Each process maintains its own progress log file so that no cross-process file contention occurs
    progress_file_path = progress_dir / f"progress_rank{accelerator.process_index}.txt"
    keys_to_log: list[str] = []  # filenames that were successfully flushed to LMDB since last write
    
    # =========================
    # Custom collate function
    # =========================
    def safe_collate_fn(batch):
        """Collate function that drops None items and stacks tensors safely."""
        valid_items = [item for item in batch if item is not None]
        if len(valid_items) == 0:
            # Return an empty placeholder batch that the loop can skip
            return {'image': torch.empty(0), 'filename': []}
        try:
            images = torch.stack([item['image'] for item in valid_items], dim=0)
            filenames = [item['filename'] for item in valid_items]
            return {'image': images, 'filename': filenames}
        except Exception as e:
            print(f"Error stacking batch with filenames: {[item['filename'] for item in valid_items]}")
            raise e

    # -------------------------
    # DataLoader construction
    # -------------------------
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=args.num_workers // accelerator.num_processes if accelerator.num_processes > 1 else args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=safe_collate_fn,
    )

    # Prepare objects for distributed execution
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Create directories
    train_dir = output_dir / 'train' / f'{data_cfg["satellite_type"]}_{bands}'
    test_dir = output_dir / 'test' / f'{data_cfg["satellite_type"]}_{bands}'
    train_dir.mkdir(exist_ok=True, parents=True)
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # =========================
    # Prepare saving back-ends
    # =========================

    save_executor = None
    pending_futures = []  # track async save operations

    if args.save_to_lmdb:
        
        # Unless we are explicitly resuming, remove any pre-existing LMDB databases so that we start clean.
        if (not args.resume) and accelerator.is_main_process:
            print("Main process: delete existing LMDB files (no --resume flag)")
            if train_dir.exists():
                try:
                    os.remove(f"{train_dir}/data.mdb")
                    os.remove(f"{train_dir}/lock.mdb")
                except Exception as e:
                    accelerator.print(f"==> Error deleting train LMDB file: {e}")
            if test_dir.exists():
                try:
                    os.remove(f"{test_dir}/data.mdb")
                    os.remove(f"{test_dir}/lock.mdb")
                except Exception as e:
                    accelerator.print(f"==> Error deleting test LMDB file: {e}")

        print("Thread num waits: ", accelerator.process_index, accelerator.is_main_process)
        accelerator.wait_for_everyone()
        print("Everyone is here")

        lmdb_kwargs = dict(
            map_size=args.lmdb_map_size,
            readonly=False,
            lock=True,
            readahead=False,
            meminit=False,
        )
        if args.lmdb_fast:
            lmdb_kwargs.update(dict(sync=False, metasync=False, map_async=True))

        train_lmdb_env = lmdb.open(
            str(train_dir),
            **lmdb_kwargs,
        )

        test_lmdb_env = lmdb.open(
            str(test_dir),
            **lmdb_kwargs,
        )
    else:
        # Thread pool for async npy saving
        save_executor = ThreadPoolExecutor(max_workers=args.save_workers)

    # -------------------------------------------------
    # Buffers for batched LMDB writes (one per process)
    # -------------------------------------------------
    if args.save_to_lmdb:
        train_buffer: list[tuple[bytes, bytes]] = []
        test_buffer: list[tuple[bytes, bytes]] = []

        def _flush_lmdb_buffers():
            """Write buffered key-value pairs to LMDB in a single txn and clear the buffers."""
            nonlocal train_buffer, test_buffer
            if train_buffer:
                with train_lmdb_env.begin(write=True) as txn:
                    for k, v in train_buffer:
                        txn.put(k, v)
                train_buffer.clear()
            if test_buffer:
                with test_lmdb_env.begin(write=True) as txn:
                    for k, v in test_buffer:
                        txn.put(k, v)
                test_buffer.clear()

    # Process all images
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches", disable=not accelerator.is_local_main_process)):
            
            images = batch['image'].to(device, non_blocking=True)
            filenames = batch['filename']

            # Skip empty batches (e.g., if all samples were None)
            if images.numel() == 0 or len(filenames) == 0:
                continue

            if data_cfg["satellite_type"] == "GOOGLE_EMBEDS":
                latents = images
            else:
                # Encode images -- use the original (unwrapped) model to avoid DDP attribute issues
                base_model = accelerator.unwrap_model(model)
                latents = base_model.encode_moments(images)

            # ----- Save latent representations -----
            train_entries = []
            test_entries = []

            for j in range(len(images)):
                filename = filenames[j]

                # Handle patchified data
                if args.patchify:
                    # Extract grid cell and patch ID from filename (format: grid_cell_patch_X)
                    if '_patch_' in filename:
                        grid_cell = filename.split('_patch_')[0]
                        patch_id = filename.split('_patch_')[1]
                    else:
                        raise ValueError(f"Patch info not found in filename: {filename}")
                else:
                    grid_cell = filename

                arr = latents[j].cpu().numpy()

                if args.save_to_lmdb:
                    # Serialize using pickle to preserve dtype and shape
                    if args.save_to_zarr:
                        arr_bytes = save_zarr_zip_blob(arr)
                    else:
                        arr_bytes = pickle.dumps(arr, protocol=-1)
                    if grid_cell in train_grid_cells:
                        train_buffer.append((filename.encode(), arr_bytes))
                    else:
                        test_buffer.append((filename.encode(), arr_bytes))
                    # Track for progress logging *after* successful flush
                    keys_to_log.append(filename)
                else:
                    # Save to appropriate directory based on split
                    if grid_cell in train_grid_cells:
                        latent_path = train_dir / f"{filename}.npy"
                    else:
                        latent_path = test_dir / f"{filename}.npy"
                    fut = save_executor.submit(np.save, latent_path, arr)
                    pending_futures.append(fut)
            
            if (not args.save_to_lmdb) and args.flush_every_batches and ((i + 1) % args.flush_every_batches == 0):
                before = len(pending_futures)
                pending_futures = [f for f in pending_futures if not f.done()]
                done_cnt = before - len(pending_futures)
                accelerator.print(
                    f"[Flush] Batch {i + 1}: just cleared {done_cnt} finished writes | still queued: {len(pending_futures)}"
                )

            if args.save_to_lmdb and args.flush_every_batches and ((i + 1) % args.flush_every_batches == 0):
                _flush_lmdb_buffers()

                # Ensure data is persisted to disk frequently to minimise work loss on timeout
                if accelerator.is_main_process:
                    train_lmdb_env.sync()
                    test_lmdb_env.sync()
                accelerator.wait_for_everyone()

                # Persist the list of successfully flushed filenames
                if keys_to_log:
                    with open(progress_file_path, 'a') as pf:
                        pf.write("\n".join(keys_to_log) + "\n")
                        pf.flush()
                        os.fsync(pf.fileno())
                    done_entries.update(keys_to_log)
                    keys_to_log.clear()

            if args.plot_latents:
                # Create directory for latent visualisations
                latent_vis_dir = output_dir / 'latent_visualisations'
                latent_vis_dir.mkdir(exist_ok=True, parents=True)
                
                for j in range(len(images)):
                    filename = filenames[j]
                    
                    # Get current latent and reshape if needed
                    curr_latent = latents[j]
                    latent_dim = curr_latent.shape[0] // 2
                    
                    # Split into means and stds
                    means = curr_latent[:latent_dim]
                    stds = curr_latent[latent_dim:]
                    
                    # Create figure for means
                    fig_means, axes_means = plt.subplots(1, latent_dim, figsize=(latent_dim*3, 3))
                    if latent_dim == 1:
                        axes_means = [axes_means]
                    for k, ax in enumerate(axes_means):
                        im = ax.imshow(means[k].cpu().numpy())
                        ax.set_title(f"Mean {k}")
                        fig_means.colorbar(im, ax=ax)
                    plt.tight_layout()
                    plt.savefig(latent_vis_dir / f"{filename}_means.png")
                    plt.close()
                    
                    # Plot stds
                    fig_stds, axes_stds = plt.subplots(1, latent_dim, figsize=(latent_dim*3, 3))
                    if latent_dim == 1:
                        axes_stds = [axes_stds]
                    for k, ax in enumerate(axes_stds):
                        im = ax.imshow(stds[k].cpu().numpy())
                        ax.set_title(f"Std {k}")
                        fig_stds.colorbar(im, ax=ax)
                    plt.tight_layout()
                    plt.savefig(latent_vis_dir / f"{filename}_stds.png")
                    plt.close()
                    
                    # Generate samples from the latent
                    samples = base_model.sample_moments(curr_latent.unsqueeze(0))
                    
                    # UNCOMMENT TO VISUALIZE SCALED LATENTS
                    # scale_factor = 0.331742 # For S2L2A_B04_B03_B02_B08
                    # samples = samples * scale_factor
                    
                    # Plot the sampled reconstructions
                    samples = samples.squeeze(0)  # Remove batch dimension
                    
                    # Determine number of channels in the samples
                    n_channels = samples.shape[0]
                    
                    # Plot samples
                    fig_samples, axes_samples = plt.subplots(1, n_channels, figsize=(n_channels*3, 3))
                    if n_channels == 1:
                        axes_samples = [axes_samples]
                    for k, ax in enumerate(axes_samples):
                        im = ax.imshow(samples[k].cpu().numpy())
                        ax.set_title(f"Sample Ch {k}")
                        fig_samples.colorbar(im, ax=ax)
                    plt.tight_layout()
                    plt.savefig(latent_vis_dir / f"{filename}_samples.png")
                    plt.close()
            
            # Skip reconstruction if only encoding
            if not args.latents_only:
            
                # DEBUG: Lets vertically flip the latents
                # latents = torch.flip(latents, [3])
                
                # Decode images
                sampled_latents = base_model.sample_moments(latents)
                reconstructions = base_model.decode(sampled_latents)
                
                full_sat_name = f'{data_cfg["satellite_type"]}_{"_".join(data_cfg["tif_bands"])}'
                images_post_processed = post_process_data(images, full_sat_name, data_cfg)
                reconstructions_post_processed = post_process_data(reconstructions, full_sat_name, data_cfg)
                # Visualize and save results
                visualisation_results = visualise_bands(
                    inputs=images_post_processed, 
                    reconstructions=reconstructions_post_processed, 
                    save_dir=output_dir, 
                    n_images_to_log=min(args.batch_size, len(images)),
                    milestone=i,
                    satellite_type=full_sat_name,
                )

            # Non-buffered LMDB path removed; buffered handled above

    if args.save_to_lmdb:
        # Flush any remaining buffered writes
        _flush_lmdb_buffers()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            train_lmdb_env.sync()
            test_lmdb_env.sync()
        accelerator.wait_for_everyone()

        # Persist any outstanding progress information
        if keys_to_log:
            with open(progress_file_path, 'a') as pf:
                pf.write("\n".join(keys_to_log) + "\n")
                pf.flush()
                os.fsync(pf.fileno())
            keys_to_log.clear()

        train_lmdb_env.close()
        test_lmdb_env.close()
    else:
        # Ensure all asynchronous saves are finished
        pending_futures = [f for f in pending_futures if not f.done()]
        accelerator.print(f"Waiting for {len(pending_futures)} remaining file writes to finish …")
        for f in pending_futures:
            f.result()  # join individually to propagate exceptions if any
        save_executor.shutdown(wait=True)

    if args.latents_only:
        accelerator.print(
            f"Finished processing. Latents saved to train ({len(train_grid_cells)} grid cells) and test ({len(test_grid_cells)} grid cells) directories."
        )
    else:
        accelerator.print(f"Finished processing. Reconstructed images saved to {output_dir}")
    
    # Clean up distributed resources
    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        raise
    finally:
        # Ensure cleanup even if there's an exception
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()