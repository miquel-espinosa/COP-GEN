"""
COPGEN Inference Interface

A clean and simple interface for using COPGEN models for multimodal generation.

Example usage:
    # Simple generation
    model = CopgenModel("path/to/model.pt", "path/to/config.yaml")
    samples = model.generate(["S2L2A_B02_B03_B04_B08", "DEM_DEM"])
    
    # Conditional generation
    conditions = {"S2L1C_cloud_mask": cloud_mask_tensor}
    samples = model.generate(["S2L2A_B02_B03_B04_B08"], conditions=conditions)
    
    # Multiple samples
    samples = model.generate(["DEM_DEM"], n_samples=10)
"""

import torch
import numpy as np
import ml_collections
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import einops
from datasets import get_dataset

# Import the core components
import libs.autoencoder
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver


class CopgenModel:
    """
    High-level interface for COPGEN multimodal generation.
    
    This class provides a simple API for loading and using COPGEN models
    while hiding the complexity of the underlying implementation.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config_path: Union[str, Path, ml_collections.ConfigDict],
        device: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize a COPGEN model for inference.
        
        Args:
            model_path: Path to the trained model checkpoint (.pt file)
            config_path: Path to the model configuration file or ConfigDict object
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
            seed: Random seed for reproducibility
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        
        # Handle both path and direct config object
        if isinstance(config_path, ml_collections.ConfigDict):
            self.config = config_path
            self.config_path = None
        else:
            self.config_path = Path(config_path)
            self.config = self._load_config(config_path)
        
        # Set seed if provided
        self.set_seed(seed)
        
        # Initialize model components
        self._setup_model()
        
    def set_seed(self, seed: int):
        utils.set_seed(seed)
        
    def _load_config(self, config_path: Union[str, Path]) -> ml_collections.ConfigDict:
        """Load and validate configuration."""
        # This is a simplified version - in practice you'd load from YAML/JSON
        # For now, assuming the config is passed as a Python module
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        return config_module.get_config()
        
    def _stable_diffusion_beta_schedule(self, linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
        _betas = (
            torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )
        return _betas.numpy()
    
    def _setup_model(self):
        """Initialize model components."""
        self.betas = self._stable_diffusion_beta_schedule()
        self.N = len(self.betas)
        
        # Load neural network
        self.nnet = utils.get_nnet(**self.config.nnet)
        self.nnet.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        self.nnet.to(self.device)
        self.nnet.eval()
        
        # Load autoencoders
        self.autoencoders = {}
        loaded_autoencoders = {}
        
        for modality in self.config.autoencoders.keys():
            ae_path = self.config.autoencoders[modality].pretrained_path
            if ae_path in loaded_autoencoders:
                self.autoencoders[modality] = loaded_autoencoders[ae_path]
            else:
                print(f"Loading autoencoder from {ae_path}")
                ae = libs.autoencoder.get_model(**self.config.autoencoders[modality])
                ae.to(self.device)
                ae.eval()
                self.autoencoders[modality] = ae
                loaded_autoencoders[ae_path] = ae
        
        # Print model parameter breakdown
        PRINT_MODEL_PARAMS = False
        if PRINT_MODEL_PARAMS:
            print("Model structure:")
            # Print nnet (diffusion UNet) structure
            nnet_params = sum(p.numel() for p in self.nnet.parameters())
            print(f"  nnet (diffusion UNet): {nnet_params:,} params ({nnet_params/1e6:.2f}M)")
            for name, module in self.nnet.named_children():
                sub_params = sum(p.numel() for p in module.parameters())
                if sub_params > 0:
                    print(f"    └─ {name}: {sub_params:,} params ({sub_params/1e6:.2f}M)")
            
            # Print autoencoders
            total_ae_params = 0
            print(f"  autoencoders:")
            for modality, ae in self.autoencoders.items():
                ae_params = sum(p.numel() for p in ae.parameters())
                # Avoid double counting shared autoencoders
                if ae in list(self.autoencoders.values())[:list(self.autoencoders.keys()).index(modality)]:
                    print(f"    └─ {modality}: (shared, see above)")
                else:
                    total_ae_params += ae_params
                    print(f"    └─ {modality}: {ae_params:,} params ({ae_params/1e6:.2f}M)")
                    # Go one level deeper for autoencoder
                    if hasattr(ae, 'named_children'):
                        for sub_name, sub_module in ae.named_children():
                            sub_params = sum(p.numel() for p in sub_module.parameters())
                            if sub_params > 0:
                                print(f"      └─ {sub_name}: {sub_params:,} params ({sub_params/1e6:.2f}M)")
            
            # Total
            total_params = nnet_params + total_ae_params
            trainable_nnet = sum(p.numel() for p in self.nnet.parameters() if p.requires_grad)
            trainable_ae = sum(p.numel() for p in ae.parameters() if p.requires_grad for ae in loaded_autoencoders.values())
            print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"  nnet: {nnet_params:,} ({nnet_params/1e6:.2f}M), trainable: {trainable_nnet:,}")
            
            # Print memory allocation
            if self.device == 'cuda':
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
            exit(0)
        
        # Setup modality information
        self.modality_names = list(self.config.z_shapes.keys())
        self.z_shapes = dict(self.config.z_shapes)
        self.input_resolutions = dict(self.config.input_resolutions)
        self.z_dims = {m: int(np.prod(self.z_shapes[m])) for m in self.modality_names}
        
        # Noise schedule
        self.noise_schedule = NoiseScheduleVP(
            schedule='discrete',
            betas=torch.tensor(self.betas, device=self.device).float()
        )
    
    @property
    def available_modalities(self) -> List[str]:
        """Get list of all available modalities."""
        return self.modality_names.copy()
    
    def generate(
        self,
        modalities: Union[str, List[str]],
        conditions: Optional[Dict[str, torch.Tensor]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        sample_steps: int = 50,
        return_latents: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
        """
        Generate samples for specified modalities.
        
        Args:
            modalities: Single modality name or list of modalities to generate
            conditions: Dictionary mapping modality names to conditioning tensors
            n_samples: Number of samples to generate per condition
            batch_size: Batch size for generation (optional, will be inferred from conditions if not provided)
            sample_steps: Number of diffusion steps
            return_latents: If True, also return the latent representations
            
        Returns:
            If single modality: tensor of shape [batch_size * n_samples, C, H, W]
            If multiple modalities: dict mapping modality names to tensors
            If return_latents=True: tuple of (decoded_samples, latent_samples)
        """
        # Handle single modality case
        if isinstance(modalities, str):
            modalities = [modalities]
            single_modality = True
        else:
            single_modality = False
        
        # Validate modalities
        for m in modalities:
            if m not in self.modality_names:
                raise ValueError(f"Unknown modality: {m}. Available: {self.modality_names}")
        
        # Validate conditions
        if conditions:
            for m in conditions:
                if m not in self.modality_names:
                    raise ValueError(f"Unknown conditioning modality: {m}")
                if m in modalities:
                    raise ValueError(f"Cannot condition on modality being generated: {m}")
        
        # Prepare generation
        conditions = conditions or {}
        if batch_size is None:
            if conditions:
                batch_size = next(iter(conditions.values())).shape[0]
            else:
                raise ValueError("Batch size must be provided in `.generate()` for unconditional generation")
        effective_batch_size = batch_size * n_samples
        
        # Encode conditions
        z_cond_map = self.encode(conditions, n_samples=n_samples)
        
        # Setup generation masks
        generate_set = set(modalities)
        condition_set = set(conditions.keys())
        
        # Run generation
        with torch.no_grad():
            latents = self._sample_batch(
                effective_batch_size,
                z_cond_map,
                generate_set,
                condition_set,
                sample_steps
            )
        
        # Decode latents
        decoded, latent_dict = self.decode(latents, modalities, return_latents=return_latents)
                
        # Return results
        if single_modality:
            result = decoded[modalities[0]]
        else:
            result = decoded
        
        if return_latents:
            return result, latent_dict
        else:
            return result
    
    def encode(
        self,
        data: Dict[str, torch.Tensor],
        modalities: Optional[List[str]] = None,
        n_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Encode data into latent space.
        
        Args:
            data: Dictionary mapping modality names to data tensors
            modalities: List of modalities to encode (default: all in data)
            n_samples: Number of repeated samples to encode (default: 1)
        Returns:
            Dictionary mapping modality names to latent tensors
        """
        modalities = modalities or list(data.keys())
        
        encoded = {}
        with torch.no_grad():
            for m in modalities:
                if m not in data:
                    raise ValueError(f"Modality {m} not found in data")
                
                tensor = data[m].to(self.device)
                
                if self.config.all_modality_configs[m].modality_type == 'image':
                    with torch.amp.autocast('cuda'):
                        # encoded[m] = self.autoencoders[m].sample(tensor)
                        encoded[m] = self.autoencoders[m].encode(tensor)
                elif self.config.all_modality_configs[m].modality_type == 'scalar':  # scalar
                    encoded[m] = tensor
                else:
                    raise NotImplementedError(f"Modality type {self.config.all_modality_configs[m].modality_type} not supported in encode")

                if n_samples > 1:
                    encoded[m] = encoded[m].repeat_interleave(n_samples, dim=0)
        
        return encoded
    
    def decode(
        self,
        latents: Dict[str, torch.Tensor],
        modalities: Optional[List[str]] = None,
        return_latents: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latents back to data space.
        
        Args:
            latents: Dictionary mapping modality names to latent tensors
            modalities: List of modalities to decode (default: all in latents)
            return_latents: If True, also return the latent representations
        Returns:
            Dictionary mapping modality names to decoded tensors
        """
        modalities = modalities or list(latents.keys())
        
        decoded = {}
        latent_dict = {}
        with torch.no_grad():
            for m in modalities:
                if m not in latents:
                    raise ValueError(f"Modality {m} not found in latents")
                
                latent = latents[m].to(self.device)
                
                if self.config.all_modality_configs[m].modality_type == 'image':
                    with torch.amp.autocast('cuda'):
                        decoded[m] = self.autoencoders[m].decode(latent)
                elif self.config.all_modality_configs[m].modality_type == 'scalar':  # scalar
                    decoded[m] = latent
                else:
                    raise NotImplementedError(f"Modality type {self.config.all_modality_configs[m].modality_type} not supported in decode")
                
                if return_latents:
                    latent_dict[m] = latent
        
        return decoded, latent_dict
    
    def _combine_joint(self, z_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.concat([
            einops.rearrange(z_i, 'B C H W -> B (C H W)') 
            for z_i in z_list
        ], dim=-1)
        
    def _split_joint(self, x: torch.Tensor, z_cond_map: dict[str, torch.Tensor],
                     generate_set: set, condition_set: set) -> list[torch.Tensor]:
        B = x.shape[0] if x is not None else next(iter(z_cond_map.values())).shape[0]
        out = []
        offset = 0
        for m in self.modality_names:
            C, H, W = self.z_shapes[m]
            dim = self.z_dims[m]
            if m in generate_set:
                chunk = x[:, offset:offset + dim]
                offset += dim
                out.append(einops.rearrange(chunk, 'B (C H W) -> B C H W', C=C, H=H, W=W))
            elif m in condition_set:
                out.append(z_cond_map[m])
            else:
                out.append(torch.randn(B, C, H, W, device=self.device))
        return out
    
    def _sample_batch(
        self,
        batch_size: int,
        z_cond_map: Dict[str, torch.Tensor],
        generate_set: set,
        condition_set: set,
        sample_steps: int
    ) -> List[torch.Tensor]:
        """Internal sampling method."""
        
        # Initialize noise for modalities to generate
        z_init_list = [
            torch.randn(batch_size, *self.z_shapes[m], device=self.device)
            for m in self.modality_names if m in generate_set
        ]
        
        # Combine into flat vector
        x_init = self._combine_joint(z_init_list)
        
        # Define model function for DPM solver
        def model_fn(x, t_continuous):
            t = t_continuous * self.N
            
            # Split x back into modalities
            out = self._split_joint(x, z_cond_map, generate_set, condition_set)
            
            # Run through network
            t_imgs = [t if m in generate_set else torch.zeros_like(t) 
                     for m in self.modality_names]
            z_out = self.nnet(out, t_imgs=t_imgs)
            
            # Extract only generated modalities
            z_out_generated = [
                z_out[i] for i, m in enumerate(self.modality_names) 
                if m in generate_set
            ]
            
            # Combine back to flat vector
            return self._combine_joint(z_out_generated)
        
        # Run DPM solver
        dpm_solver = DPM_Solver(
            model_fn, 
            self.noise_schedule, 
            predict_x0=True, 
            thresholding=False
        )
        
        with torch.autocast(device_type=self.device):
            x = dpm_solver.sample(
                x_init, 
                steps=sample_steps, 
                eps=1. / self.N, 
                T=1.
            )
        
        # Split final x back into all modalities
        final_latents = self._split_joint(x, z_cond_map, generate_set, condition_set)
        
        final_out = {}
        # Replace the conditioned modalities with the original ones
        for i, m in enumerate(self.modality_names):
            if m in condition_set:
                final_out[m] = z_cond_map[m]
            else:
                final_out[m] = final_latents[i]
                
        return final_out
            
def build_copgen_dataloader(config: ml_collections.ConfigDict) -> torch.utils.data.DataLoader:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset_config = ml_collections.ConfigDict(config.to_dict())
    dataset_config.dataset.path = config.data_path
    dataset_config.dataset.return_filename = True
    dataset = get_dataset(**dataset_config.dataset)
    test_dataset = dataset.get_split(split='train', labeled=False)

    g = torch.Generator()
    if config.get('seed') is not None:
        g.manual_seed(int(config.seed))

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        generator=g,
    )
    return dataloader

