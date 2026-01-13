import argparse
import importlib.util
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np
import torch

import libs.autoencoder as autoencoder_lib
from visualisations.visualise_bands import visualise_bands


def load_config(config_path: str):
    """Dynamically import a config file and return the config object.

    The config file must expose a `get_config()` function that returns an
    `ml_collections.ConfigDict` (as in the training script).
    """
    config_path = Path(config_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("config_module", str(config_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    if not hasattr(module, "get_config"):
        raise AttributeError(f"Config file {config_path} does not define `get_config()`")
    return module.get_config()


def prepare_autoencoders(config, device):
    """Load (and share) autoencoders as defined in the training config."""
    loaded = {}
    aes = {}
    # Some modalities may be scalar and skip loading an image autoencoder
    for modality, mod_cfg in config.all_modality_configs.items():
        if mod_cfg.modality_type == "image":
            ae_cfg = config.autoencoders[modality]
            pretrained_path = ae_cfg.pretrained_path
            if pretrained_path in loaded:
                aes[modality] = loaded[pretrained_path]
            else:
                model = autoencoder_lib.get_model(**ae_cfg)
                model = model.to(device)
                model.eval()
                loaded[pretrained_path] = model
                aes[modality] = model
    return aes


def expected_latent_shape(config, modality):
    """Return (C, H, W) latent shape stored in LMDB for a modality."""
    # Training config stores this under dataset.z_input_shapes
    z_shape = config.dataset.z_input_shapes[modality]
    if len(z_shape) != 3:
        raise ValueError(f"Invalid z_input_shapes for {modality}: {z_shape}")
    return tuple(z_shape)


def decode_latent(tensor: torch.Tensor, modality: str, cfg_mod, aes, device):
    """Decode latent representation into pixel / visualisation tensor."""
    if cfg_mod.modality_type == "image":
        ae = aes[modality]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.is_autocast_enabled()):
            z = ae.sample(tensor)  # (B, z_channels, H, W)
            decoded = ae.decode(z)  # (B, bands, H_img, W_img)
        return decoded
    elif cfg_mod.modality_type == "scalar":
        # For scalar modalities the latent itself is the data to visualise
        return tensor
    else:
        raise NotImplementedError(f"Unknown modality_type {cfg_mod.modality_type} for {modality}")


def main():
    parser = argparse.ArgumentParser(description="Decode and visualise LMDB latent entries.")
    parser.add_argument("lmdb_path", help="Path to LMDB database containing latent entries")
    parser.add_argument("config_path", help="Path to training config .py file (must expose get_config())")
    parser.add_argument("output_dir", help="Directory to store output visualisations")
    parser.add_argument("--n-entries", "-n", type=int, default=3, help="Number of LMDB entries to decode (default: 3)")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device to use")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------------------------------------------------------------------
    # Load configuration and autoencoders
    # ---------------------------------------------------------------------
    print(f"Loading config from {args.config_path} …")
    config = load_config(args.config_path)

    print("Preparing autoencoders …")
    autoencoders = prepare_autoencoders(config, device)

    # ---------------------------------------------------------------------
    # Open LMDB
    # ---------------------------------------------------------------------
    print(f"Opening LMDB → {args.lmdb_path}")
    env = lmdb.open(args.lmdb_path, readonly=True, lock=False, readahead=False, max_readers=4)


    def _key_to_str(k: object) -> str:
        """Convert LMDB key (bytes / memoryview / other) to printable string."""
        if isinstance(k, (bytes, bytearray)):
            try:
                return k.decode("utf-8")
            except UnicodeDecodeError:
                return k.hex()
        if isinstance(k, memoryview):
            try:
                return k.tobytes().decode("utf-8")
            except UnicodeDecodeError:
                return k.tobytes().hex()
        return str(k)
    
    # ---------------------------------------------------------------------
    # Iterate through entries
    # ---------------------------------------------------------------------
    
    # SKIP FIRST 100 ENTRIES
    SKIP_FIRST = 300
    count = 0
    with env.begin(buffers=True) as txn:
        cursor = txn.cursor()
        for idx, (key, value) in enumerate(cursor):
            if idx < SKIP_FIRST:
                continue
            count += 1
            if count >= args.n_entries:
                break

            key_str = _key_to_str(key)
            print(f"\nEntry {idx} — key: {key_str}")

            try:
                entry = pickle.loads(value)
            except Exception as e:
                print(f"  !! Failed to unpickle entry: {e}")
                continue

            entry_out_dir = os.path.join(args.output_dir, f"entry_{idx}_{key_str}")
            os.makedirs(entry_out_dir, exist_ok=True)

            # Process each modality present in the entry
            for modality, byte_data in entry.items():
                if modality not in config.all_modality_configs:
                    print(f"  ?? Unknown modality '{modality}' — skipping")
                    continue

                cfg_mod = config.all_modality_configs[modality]
                C, H, W = expected_latent_shape(config, modality)
                expected_elems = C * H * W

                latent_np = np.frombuffer(byte_data, dtype=np.float32)
                if latent_np.size != expected_elems:
                    print(f"  !! Shape mismatch for {modality}: expected {expected_elems} elements (shape {C}x{H}x{W}), "
                          f"got {latent_np.size}")
                    continue
                latent_np = latent_np.reshape((C, H, W))

                latent_t = torch.from_numpy(latent_np).unsqueeze(0).to(device)

                # Decode / convert to pixel space
                try:
                    decoded = decode_latent(latent_t, modality, cfg_mod, autoencoders, device)
                except Exception as e:
                    print(f"  !! Failed to decode modality {modality}: {e}")
                    continue

                # Move to CPU for visualisation
                decoded_cpu = decoded.detach().cpu()

                # Visualise (inputs=None because we only have reconstructions)
                print(f"  ✔ Decoded {modality}: shape {decoded_cpu.shape}")
                vis_save_dir = os.path.join(entry_out_dir, modality)
                os.makedirs(vis_save_dir, exist_ok=True)

                visualise_bands(
                    inputs=None,
                    reconstructions=decoded_cpu,
                    save_dir=vis_save_dir,
                    n_images_to_log=1,
                    milestone=None,
                    satellite_type=modality,
                    tif_bands=["cloud_mask"] if "cloud_mask" in modality else [],
                    was_normalized_to_neg_one_to_one=True,
                    min_db=getattr(config.dataset, "min_db", {}).get(modality, None),
                    max_db=getattr(config.dataset, "max_db", {}).get(modality, None),
                    min_positive=getattr(config.dataset, "min_positive", {}).get(modality, None),
                )

            print(f"  → Visualisations saved to {entry_out_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main() 