import yaml
import argparse
import torch
import os
from tqdm.auto import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from ddm.utils import construct_class_by_name
from train_vae import load_conf
from visualisations.visualise_bands import visualise_bands
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser(description="Encode and decode images using a pretrained VAE")
    parser.add_argument("--cfg", help="Config file path", type=str, required=True)
    parser.add_argument("--checkpoint_path", help="Path to VAE checkpoint (overrides config)", type=str)
    parser.add_argument("--output_dir", help="Output directory for reconstructed images", type=str, required=True)
    parser.add_argument("--batch_size", help="Batch size for processing", type=int, default=4)
    parser.add_argument("--num_workers", help="Number of workers for data loading", type=int, default=2)
    parser.add_argument("--latents_only", help="Only encode images and save latents without reconstruction", action="store_true")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    small_output_dir = output_dir / 'small'
    small_output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create comparison directory for visualisation
    comparison_dir = output_dir / 'comparison'
    comparison_dir.mkdir(exist_ok=True, parents=True)
    
    # Create perturbation directory for visualisation
    perturbation_dir = output_dir / 'perturbation'
    perturbation_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model from config
    model_cfg = args.cfg['model']
    
    # Override checkpoint path if provided in CLI
    if args.checkpoint_path:
        model_cfg['ckpt_path'] = args.checkpoint_path
    
    # Ensure checkpoint path exists
    if 'ckpt_path' not in model_cfg or not os.path.exists(model_cfg['ckpt_path']):
        raise ValueError(f"Checkpoint path not found: {model_cfg.get('ckpt_path', 'Not set')}")
    
    print(f"Loading model from checkpoint: {model_cfg['ckpt_path']}")
    model = construct_class_by_name(**model_cfg)
    model.eval()
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load dataset from config
    data_cfg = args.cfg["data"]
    dataset = construct_class_by_name(**data_cfg)
    
    # Get satellite type if defined, otherwise use a default
    satellite_type = getattr(dataset, 'satellite_type', None)
    band_name = getattr(dataset, 'band_name', None)
    if band_name == 'thumbnail':
        satellite_type = f"{satellite_type}_thumbnail"
    # Get normalization info from data config
    normalize_to_neg_one_to_one = data_cfg.get('normalize_to_neg_one_to_one', False)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create latents directory
    latents_dir = output_dir / 'latents'
    latents_dir.mkdir(exist_ok=True)
    
    # Process all images
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Get images from batch (handle different batch formats)
            if isinstance(batch, dict):
                images = batch['image']
                
                # Save filenames if available in batch
                filenames = batch.get('filename', None)
            elif isinstance(batch, torch.Tensor):
                images = batch
                filenames = None
            else:
                # Try to get first element of tuple or list
                images = batch[0]
                # Check if second element might be filenames
                filenames = batch[1] if len(batch) > 1 and isinstance(batch[1], (list, tuple)) else None
            
            # Move to device
            images = images.to(device)
            
            # Save image
            for j in range(len(images)):
                if filenames is not None:
                    filename = filenames[j]
                    if isinstance(filename, bytes):
                        filename = filename.decode('utf-8')
                    image_path = output_dir / f"{filename}_original.png"
                else:
                    image_path = output_dir / f"image_{i}_{j}_original.png"
                save_image(images[j], image_path)
            
            # Save a histogram of the images
            import matplotlib.pyplot as plt
            import numpy as np
            for band in range(images.shape[1]):
                plt.hist(images[:, band, :, :].flatten().cpu().numpy(), bins=256, alpha=0.5, label='Original')
                plt.legend()
                plt.savefig(output_dir / f"before_vae_histogram_band_{band}.png")
                plt.close()

            # DEBUG: Set images to values between 0.3 and 0.5
            # images = torch.rand_like(images) * 0.2 + 0.3
            
            print(f"input: images min max mean std: {images.min()}, {images.max()}, {images.mean()}, {images.std()}")
            
            # Encode images
            posterior = model.encode(images)
            latents = posterior.sample()
            
            reconstructions = model.decode(latents)
            
            print(f"reconstructions: reconstructions min max mean std: {reconstructions.min()}, {reconstructions.max()}, {reconstructions.mean()}, {reconstructions.std()}")
            # Save the reconstructed images
            for j in range(len(images)):
                if filenames is not None:
                    filename = filenames[j]
                    if isinstance(filename, bytes):
                        filename = filename.decode('utf-8')
                    reconstruction_path = output_dir / f"{filename}_reconstructed.png"
                else:
                    reconstruction_path = output_dir / f"reconstruction_{i}_{j}_reconstructed.png"
                save_image(reconstructions[j], reconstruction_path)
            
            # Save a histogram of the reconstructions
            for band in range(reconstructions.shape[1]):
                plt.hist(reconstructions[:, band, :, :].flatten().cpu().numpy(), bins=256, alpha=0.5, label='Reconstructed')
                plt.legend()
                plt.savefig(output_dir / f"after_vae_histogram_band_{band}.png")
                plt.close()
            
            # Latents are of shape (batch_size, 8, 132, 132) given an input of shape (batch_size, 4, 1056, 1056)
            # Now we random crop the latents from shape (batch_size, 8, 132, 132) to shape (batch_size, 8, 30, 30)
            B, C, H, W = latents.shape
            target_size_latent = data_cfg['image_size'][0]//8 # 24  # 15  # Target size in latent space
            target_size_input = data_cfg['image_size'][0] # 192 # 120 # Equivalent size in input space
            
            # Calculate scaling factor between latent and input space
            scale_factor = images.shape[2] // latents.shape[2]  # Should be 8
            assert scale_factor == 8
            
            # Calculate valid range for crop start positions in latent space
            max_x_latent = H - target_size_latent
            max_y_latent = W - target_size_latent
            
            # Generate random crop coordinates in latent space
            start_x_latent = torch.randint(0, max_x_latent + 1, (1,)).item()
            start_y_latent = torch.randint(0, max_y_latent + 1, (1,)).item()
            
            # Calculate equivalent coordinates in input space
            start_x_input = start_x_latent * scale_factor
            start_y_input = start_y_latent * scale_factor
            
            # Perform the crops in both spaces
            small_latents = latents[:, :, start_x_latent:start_x_latent + target_size_latent, 
                                     start_y_latent:start_y_latent + target_size_latent]
            small_images = images[:, :, start_x_input:start_x_input + target_size_input,
                                   start_y_input:start_y_input + target_size_input]

                        
            # Save latent representations
            for j in range(len(images)):
                # Use filename if available, otherwise use batch and index
                if filenames is not None:
                    filename = filenames[j]
                    if isinstance(filename, bytes):
                        filename = filename.decode('utf-8')
                    latent_path = latents_dir / f"{filename}.pt"
                else:
                    latent_path = latents_dir / f"latent_{i}_{j}.pt"
                
                # Save the latent representation as a PyTorch tensor
                torch.save(latents[j], latent_path)
            
            # Skip reconstruction if only encoding
            if args.latents_only:
                continue
                
            # Decode images
            reconstructions = model.decode(latents)
            small_reconstructions = model.decode(small_latents)
            
            # Extract same crops from full reconstructions for comparison
            crops_from_full = reconstructions[:, :, start_x_input:start_x_input + target_size_input,
                                             start_y_input:start_y_input + target_size_input]
            
            # print(f"images shape: {images.shape}")  # images shape: torch.Size([1, 2, 1056, 1056])
            # print(f"reconstructions shape: {reconstructions.shape}")  # reconstructions shape: torch.Size([1, 2, 1056, 1056])
            
            # Save the histograms per band of the images and reconstructions
            import matplotlib.pyplot as plt
            import numpy as np
            for band in range(images.shape[1]):
                plt.hist(images[:, band, :, :].flatten().cpu().numpy(), bins=256, alpha=0.5, label='Original')
                plt.hist(reconstructions[:, band, :, :].flatten().cpu().numpy(), bins=256, alpha=0.5, label='Reconstructed')
                plt.legend()
                plt.savefig(output_dir / f"histogram_band_{band}.png")
                plt.close()
            
            # Visualize and save results
            # TODO: Needs update with post_processing and new visualise_bands function
            visualisation_results = visualise_bands(
                images, # First image is the full size original image
                reconstructions, # Second image is the reconstructed full size image
                output_dir, 
                n_images_to_log=min(args.batch_size, len(images)),
                milestone=i,
                satellite_type=satellite_type,
                tif_bands=data_cfg['tif_bands'],
                was_normalized_to_neg_one_to_one=normalize_to_neg_one_to_one,
                min_db=data_cfg.get('min_db', None),
                max_db=data_cfg.get('max_db', None),
                min_positive=data_cfg.get('min_positive', None)
            )
            
            # TODO: Needs update with post_processing and new visualise_bands function
            small_visualisation_results = visualise_bands(
                small_images, # First image is the original small cropped image
                small_reconstructions, # Second image is the reconstruction from the small cropped latent
                small_output_dir, 
                n_images_to_log=min(args.batch_size, len(images)),
                milestone=i,
                satellite_type=satellite_type,
                tif_bands=data_cfg['tif_bands'],
                was_normalized_to_neg_one_to_one=normalize_to_neg_one_to_one,
                min_db=data_cfg.get('min_db', None),
                max_db=data_cfg.get('max_db', None),
                min_positive=data_cfg.get('min_positive', None)
            )
            
            # Visualize comparison between the two approaches
            # TODO: Needs update with post_processing and new visualise_bands function
            comparison_visualisation = visualise_bands(
                crops_from_full, # First image: we crop from the reconstructed full size image
                small_reconstructions, # Second image: we show the reconstruction from the small cropped latent
                comparison_dir,
                n_images_to_log=min(args.batch_size, len(images)),
                milestone=i,
                satellite_type=satellite_type,
                tif_bands=data_cfg['tif_bands'],
                was_normalized_to_neg_one_to_one=normalize_to_neg_one_to_one,
                min_db=data_cfg.get('min_db', None),
                max_db=data_cfg.get('max_db', None),
                min_positive=data_cfg.get('min_positive', None)
            )
            
            # UNCOMMENT TO VISUALIZE PERTURBATION
            # perturbation_levels = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            # for perturbation_level in perturbation_levels:
            #     small_latents_perturbation = small_latents + torch.rand_like(small_latents) * perturbation_level
            #     small_reconstructions_perturbation = model.decode(small_latents_perturbation)
                
            #     # Visualize and save results
            #     # TODO: Needs update with post_processing and new visualise_bands function
            #     small_visualisation_results_perturbation = visualise_bands(
            #         small_images, # First image: we show the original small cropped image
            #         small_reconstructions_perturbation, # Second image: we show the reconstruction from the small cropped latent with perturbation
            #         perturbation_dir / f'perturbation_{perturbation_level}', 
            #         n_images_to_log=min(args.batch_size, len(images)),
            #         milestone=i,
            #         satellite_type=satellite_type,
            #         was_normalized_to_neg_one_to_one=normalize_to_neg_one_to_one
            #     )
            exit()
                
    
    if args.latents_only:
        print(f"Finished processing. Latent representations saved to {latents_dir}")
    else:
        print(f"Finished processing. Reconstructed images saved to {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args) 