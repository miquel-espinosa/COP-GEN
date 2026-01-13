import random
import numpy as np
import torch
import ml_collections
import einops
import libs.autoencoder
import utils
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from datasets import get_dataset


# -------------------------------
# Helpers
# -------------------------------

def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()

# -------------------------------
# Reusable sampling utilities
# -------------------------------

class CopgenSamplingContext:
    def __init__(self, config: ml_collections.ConfigDict):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        utils.set_seed(config.get('seed'))
        self.config = ml_collections.FrozenConfigDict(config)
        utils.set_logger(log_level='info')

        # Network
        self.betas = stable_diffusion_beta_schedule()
        self.N = len(self.betas)
        self.nnet = utils.get_nnet(**self.config.nnet)
        self.nnet.load_state_dict(torch.load(self.config.nnet_path, map_location='cpu'))
        self.nnet.to(self.device)
        self.nnet.eval()

        # Autoencoders (per modality)
        loaded_autoencoders: dict[str, torch.nn.Module] = {}
        self.autoencoders: dict[str, torch.nn.Module] = {}
        for modality in self.config.autoencoders.keys():
            ae_path = self.config.autoencoders[modality].pretrained_path
            if ae_path in loaded_autoencoders:
                self.autoencoders[modality] = loaded_autoencoders[ae_path]
            else:
                self.autoencoders[modality] = libs.autoencoder.get_model(**self.config.autoencoders[modality]).to(self.device)
                self.autoencoders[modality].eval()
                loaded_autoencoders[ae_path] = self.autoencoders[modality]

        # Modality metadata
        self.modality_names: list[str] = list(self.config.z_shapes.keys())
        self.z_shapes: dict[str, tuple[int, int, int]] = self.config.z_shapes
        self.generate_set = set(self.config.generate_modalities)
        self.condition_set = set(self.config.condition_modalities)
        self.generate_mask = [m in self.generate_set for m in self.modality_names]
        self.z_dims = {m: int(np.prod(self.z_shapes[m])) for m in self.modality_names}

        # Noise schedule
        self.noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(self.betas, device=self.device).float())

    def combine_joint(self, z_list: list[torch.Tensor]) -> torch.Tensor:
        return torch.concat([einops.rearrange(z_i, 'B C H W -> B (C H W)') for z_i in z_list], dim=-1)

    def split_joint(self, x: torch.Tensor, z_cond_map: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        B = x.shape[0] if x is not None else next(iter(z_cond_map.values())).shape[0]
        out = []
        offset = 0
        for m in self.modality_names:
            C, H, W = self.z_shapes[m]
            dim = self.z_dims[m]
            if m in self.generate_set:
                chunk = x[:, offset:offset + dim]
                offset += dim
                out.append(einops.rearrange(chunk, 'B (C H W) -> B C H W', C=C, H=H, W=W))
            elif m in self.condition_set:
                out.append(z_cond_map[m])
            else:
                out.append(torch.randn(B, C, H, W, device=self.device))
        return out

    @torch.cuda.amp.autocast()
    def decode(self, _batch: torch.Tensor, modality_name: str) -> torch.Tensor:
        if self.config.all_modality_configs[modality_name].modality_type == 'image':
            return self.autoencoders[modality_name].decode(_batch)
        elif self.config.all_modality_configs[modality_name].modality_type == 'scalar':
            return _batch
        else:
            raise NotImplementedError

    @torch.cuda.amp.autocast()
    def encode_for_condition(self, _batch: torch.Tensor, modality_name: str) -> torch.Tensor:
        if self.config.all_modality_configs[modality_name].modality_type == 'image':
            return self.autoencoders[modality_name].sample(_batch)
        elif self.config.all_modality_configs[modality_name].modality_type == 'scalar':
            return _batch
        else:
            raise NotImplementedError

    def run_nnet(self, x_vec: torch.Tensor, t: torch.Tensor, z_cond_map: dict[str, torch.Tensor]) -> torch.Tensor:
        t_imgs = [t if mask else torch.zeros_like(t) for mask in self.generate_mask]
        z_full = self.split_joint(x_vec, z_cond_map)
        z_out = self.nnet(z_full, t_imgs=t_imgs)
        z_out_generated = [z_out[i] for i, m in enumerate(self.modality_names) if m in self.generate_set]
        return self.combine_joint(z_out_generated)

    def sample_batch(self, effective_batch_size: int, z_cond_map: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        _z_init_list = [
            torch.randn(effective_batch_size, *self.z_shapes[m], device=self.device)
            for m in self.modality_names if m in self.generate_set
        ]
        _x_init = self.combine_joint(_z_init_list)

        def model_fn(x, t_continuous):
            t = t_continuous * self.N
            return self.run_nnet(x, t, z_cond_map)

        dpm_solver = DPM_Solver(model_fn, self.noise_schedule, predict_x0=True, thresholding=False)
        with torch.no_grad():
            with torch.autocast(device_type=self.device):
                x = dpm_solver.sample(_x_init, steps=self.config.sample.sample_steps, eps=1. / self.N, T=1.)
        _zs = self.split_joint(x, z_cond_map)
        for i, m in enumerate(self.modality_names):
            if m in self.condition_set:
                _zs[i] = z_cond_map[m]
        return _zs


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

