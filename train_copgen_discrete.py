import ml_collections
import torch
from torch import multiprocessing as mp
from datasets import get_dataset
from torchvision.utils import make_grid, save_image
import utils
import einops
from torch.utils._pytree import tree_map
import accelerate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
import tempfile
from tools.fid_score import calculate_fid_given_paths
from absl import logging
import builtins
import os
import wandb
import libs.autoencoder
import numpy as np
from visualisations.merge_visualisations import merge_visualisations


def stable_diffusion_beta_schedule(linear_start=0.00085, linear_end=0.0120, n_timestep=1000):
    _betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    )
    return _betas.numpy()


def get_skip(alphas, betas):
    N = len(betas) - 1
    skip_alphas = np.ones([N + 1, N + 1], dtype=betas.dtype)
    for s in range(N + 1):
        skip_alphas[s, s + 1:] = alphas[s + 1:].cumprod()
    skip_betas = np.zeros([N + 1, N + 1], dtype=betas.dtype)
    for t in range(N + 1):
        prod = betas[1: t + 1] * skip_alphas[1: t + 1, t]
        skip_betas[:t, t] = (prod[::-1].cumsum())[::-1]
    return skip_alphas, skip_betas


def stp(s, ts: torch.Tensor):  # scalar tensor product
    if isinstance(s, np.ndarray):
        s = torch.from_numpy(s).type_as(ts)
    extra_dims = (1,) * (ts.dim() - 1)
    return s.view(-1, *extra_dims) * ts


def mos(a, start_dim=1):  # mean of square
    return a.pow(2).flatten(start_dim=start_dim).mean(dim=-1)


class Schedule(object):  # discrete time
    def __init__(self, _betas):
        r""" _betas[0...999] = betas[1...1000]
             for n>=1, betas[n] is the variance of q(xn|xn-1)
             for n=0,  betas[0]=0
        """

        self._betas = _betas
        self.betas = np.append(0., _betas)
        self.alphas = 1. - self.betas
        self.N = len(_betas)

        assert isinstance(self.betas, np.ndarray) and self.betas[0] == 0
        assert isinstance(self.alphas, np.ndarray) and self.alphas[0] == 1
        assert len(self.betas) == len(self.alphas)

        # skip_alphas[s, t] = alphas[s + 1: t + 1].prod()
        self.skip_alphas, self.skip_betas = get_skip(self.alphas, self.betas)
        self.cum_alphas = self.skip_alphas[0]  # cum_alphas = alphas.cumprod()
        self.cum_betas = self.skip_betas[0]
        self.snr = self.cum_alphas / self.cum_betas

    def tilde_beta(self, s, t):
        return self.skip_betas[s, t] * self.cum_betas[s] / self.cum_betas[t]

    def sample(self, x0, multi_modal=False):  # sample from q(xn|x0), where n is uniform
        if multi_modal:
            n_list = []
            eps_list = []
            xn_list = []
            for x0_i in x0:
                n = np.random.choice(list(range(1, self.N + 1)), (len(x0_i),))
                eps = torch.randn_like(x0_i)
                # In simple terms, this is equivalent to: xn = data_value + noise
                xn = stp(self.cum_alphas[n] ** 0.5, x0_i) + stp(self.cum_betas[n] ** 0.5, eps)
                n_list.append(torch.tensor(n, device=x0_i.device))
                eps_list.append(eps)
                xn_list.append(xn)
            return n_list, eps_list, xn_list
        else:
            n = np.random.choice(list(range(1, self.N + 1)), (len(x0),))
            eps = torch.randn_like(x0)
            xn = stp(self.cum_alphas[n] ** 0.5, x0) + stp(self.cum_betas[n] ** 0.5, eps)
            return torch.tensor(n, device=x0.device), eps, xn

    def __repr__(self):
        return f'Schedule({self.betas[:10]}..., {self.N})'


def LSimple(x0, nnet, schedule, multi_modal=False, **kwargs):
    if multi_modal:
        n_list, eps_list, xn_list = schedule.sample(x0, multi_modal=multi_modal)  # n in {1, ..., 1000}
        eps_pred = nnet(xn_list, n_list, **kwargs)
        return sum(mos(n - np_) for n, np_ in zip(eps_list, eps_pred))
        # loss_per_modality = [mos(n - np_) for n, np_ in zip(eps_list, eps_pred)]
        # if modality_order is not None:
            # loss_per_modality[modality_order.index('LULC')] *= 5
        # return sum(loss_per_modality)
    else:
        n, eps, xn = schedule.sample(x0)  # n in {1, ..., 1000}
        eps_pred = nnet(xn, n, **kwargs)
    return mos(eps - eps_pred)


def train(config):
    if config.get('benchmark', False):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # mp.set_start_method('spawn')
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=config.train.get('gradient_accumulation_steps', 1),
    )
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision
    config = ml_collections.FrozenConfigDict(config)

    assert config.train.batch_size % accelerator.num_processes == 0
    mini_batch_size = config.train.batch_size // accelerator.num_processes

    if accelerator.is_main_process:
        os.makedirs(config.ckpt_root, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        wandb.init(dir=os.path.abspath(config.workdir), project=f'uvit_{config.dataset.name}', config=config.to_dict(),
                   name=config.hparams, job_type='train', mode='offline')
        utils.set_logger(log_level='info', fname=os.path.join(config.workdir, 'output.log'))
        logging.info(config)
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')

    dataset = get_dataset(**config.dataset)
    # Evaluation is not supported while training at the moment, just visual inspection
    # assert os.path.exists(dataset.fid_stat)
    train_dataset = dataset.get_split(split='train', labeled=config.train.mode == 'cond')
    train_dataset_loader = DataLoader(train_dataset, batch_size=mini_batch_size, shuffle=True, drop_last=True,
                                      num_workers=config.train.get('num_workers'),
                                      pin_memory=True, persistent_workers=True,
                                      prefetch_factor=4)

    train_state = utils.initialize_train_state(config, device)
    nnet, nnet_ema, optimizer, train_dataset_loader = accelerator.prepare(
        train_state.nnet, train_state.nnet_ema, train_state.optimizer, train_dataset_loader)
    lr_scheduler = train_state.lr_scheduler
    train_state.resume(config.ckpt_root)

    # Keep track of already loaded autoencoders by path
    loaded_autoencoders = {}
    autoencoders = {}
    for modality in config.autoencoders.keys():
        ae_path = config.autoencoders[modality].pretrained_path
        if ae_path in loaded_autoencoders:
            # Reuse already loaded autoencoder
            autoencoders[modality] = loaded_autoencoders[ae_path]
        else:
            # Load new autoencoder and store it
            autoencoders[modality] = libs.autoencoder.get_model(**config.autoencoders[modality]).to(device)
            loaded_autoencoders[ae_path] = autoencoders[modality]

    # @ torch.cuda.amp.autocast()
    # def encode(_batch):
    #     return autoencoder.encode(_batch)

    @ torch.cuda.amp.autocast()
    def decode(_batch, modality_name):
        if config.all_modality_configs[modality_name].modality_type == 'image':
            return autoencoders[modality_name].decode(_batch)
        elif config.all_modality_configs[modality_name].modality_type == 'scalar':
            return _batch
        else:
            raise NotImplementedError(f"Modality type {config.all_modality_configs[modality_name].modality_type} not supported in decode")

    def get_data_generator():
        while True:
            for data in tqdm(train_dataset_loader, disable=not accelerator.is_main_process, desc='epoch'):
                yield data

    data_generator = get_data_generator()

    _betas = stable_diffusion_beta_schedule()
    _schedule = Schedule(_betas)
    logging.info(f'use {_schedule}')


    def train_step(_batch):
        
        # DEBUG --- SAVE UNCONDITIONAL SAMPLES AS NPY FILES ========================================================
        # if accelerator.is_main_process:
        #     save_dir = './babayetu'
        #     os.makedirs(save_dir, exist_ok=True)
        #     logging.info('Saving unconditional samples as numpy files...')
        #     torch.cuda.empty_cache()
        #     nnet.eval()
            
        #     _, latents = dpm_solver_sample_multi_modal(_n_modalities=len(config.nnet.modalities),
        #                                           _n_samples=config.sample.n_images_to_log,
        #                                           _sample_steps=50,
        #                                           return_latents=True)
            
        #     for modality_idx, (modality, latent) in enumerate(zip(config.z_shapes.keys(), latents)):
        #         print(f"modality: {modality}")
        #         print(f"latent.shape: {latent.shape}")
        #         sample_path = os.path.join(save_dir, f'{modality}_latents.npy')
        #         np.save(sample_path, latent.detach().cpu().numpy())
        #         logging.info(f'Saved {modality} latents to {sample_path}')
            
        #     torch.cuda.empty_cache()
        #     nnet.train()
            
        # accelerator.wait_for_everyone()
        # exit()
        # DEBUG --- END ===============================================================================================

        
        # DEBUG --- PLOT INFERENCE IMAGES BEFORE TRAINING STARTS ========================================================
        # if accelerator.is_main_process:
        #     logging.info('Generating initial samples before training...')
        #     torch.cuda.empty_cache()
        #     nnet.eval()
            
        #     if config.train.mode == 'uncond':
        #         if config.train.multi_modal:
        #             samples = dpm_solver_sample_multi_modal(_n_modalities=len(config.nnet.modalities), 
        #                                                 _n_samples=config.sample.n_images_to_log, 
        #                                                 _sample_steps=50)
        #         else:
        #             samples = dpm_solver_sample(_n_samples=config.sample.n_images_to_log, _sample_steps=50)
        #     elif config.train.mode == 'cond':
        #         raise NotImplementedError
        #     else:
        #         raise NotImplementedError
            
        #     if config.train.multi_modal:
        #         from visualisations.visualise_bands import visualise_bands
        #         from ddm.pre_post_process_data import post_process_data
        #         for modality, sample in zip(config.z_shapes.keys(), samples):
        #             visualise_bands(inputs=None, reconstructions=sample,
        #                         save_dir=f'{config.sample_dir}/{modality}',
        #                         n_images_to_log=config.sample.n_images_to_log, milestone=0,
        #                         satellite_type=modality, was_normalized_to_neg_one_to_one=True,
        #                         min_db=config.dataset.get('min_db', None),
        #                         max_db=config.dataset.get('max_db', None),
        #                         min_positive=config.dataset.get('min_positive', None))
        #     else:
        #         samples = make_grid(dataset.unpreprocess(samples), 10)
        #         save_image(samples, os.path.join(config.sample_dir, 'initial_samples.png'))
        #         wandb.log({'initial_samples': wandb.Image(samples)}, step=0)
            
        #     torch.cuda.empty_cache()
        #     logging.info('Initial samples generated and saved.')
        
        # accelerator.wait_for_everyone()
        # exit()
        # DEBUG --- END =================================================================
        
        
        _metrics = dict()

        # Wrap training step in accumulate context
        with accelerator.accumulate(nnet):
            if config.train.mode == 'uncond': # Multi-modal data. Sample each modality independently
                if config.train.multi_modal:
                    _zs = []
                    for modality_name, modality in _batch.items():
                        if 'feature' in config.dataset.name:
                            if config.all_modality_configs[modality_name].modality_type == 'image':
                                _zs.append(autoencoders[modality_name].sample(modality))
                            elif config.all_modality_configs[modality_name].modality_type == 'scalar':
                                _zs.append(modality)
                            else:
                                raise NotImplementedError(f"Modality type {config.all_modality_configs[modality_name].modality_type} not supported")
                        else:
                            raise NotImplementedError(f"Modality type {config.all_modality_configs[modality_name].modality_type} not supported")
                        
                    modalities = list(_batch.keys())
                    # Check that the modalities in _batch.keys() are the same as in config.z_shapes.keys()
                    # and that they are in the same order
                    assert set(modalities) == set(config.all_modality_configs.keys())
                    assert modalities == list(config.all_modality_configs.keys())
                    assert sorted(modalities) == sorted(list(config.nnet.modalities))
                    
                    # TO REMOVE ========================================================
                    # import PIL
                    # from PIL import Image
                    # from visualisations.visualise_bands import visualise_bands
                    # from ddm.pre_post_process_data import post_process_data

                    # n_list, eps_list, xn_list = _schedule.sample(_zs, multi_modal=config.train.multi_modal)  # n in {1, ..., 1000}
            
                    # for n, eps, xn, modality_name in zip(n_list, eps_list, xn_list, modalities):
                    #     print(f"n: {n}")
                    #     print(f"eps.shape: {eps.shape}")
                    #     print(f"xn.shape: {xn.shape}")
                    
                    #     # For each timestep in n, we have a eps (noise) and a xn (noisy image)
                    #     # We want to: decode the eps and xn, and plot them all in a grid
                        
                    #     decoded_xn = decode(xn, modality_name)
                    #     decoded_eps = decode(eps, modality_name)
                    #     debug_dir_xn = f"debug_latents/{modality_name}/xn"
                    #     debug_dir_eps = f"debug_latents/{modality_name}/eps"
                    #     os.makedirs(debug_dir_xn, exist_ok=True)
                    #     os.makedirs(debug_dir_eps, exist_ok=True)
                    #     visualise_bands(
                    #         inputs=None,
                    #         reconstructions=decoded_xn,
                    #         save_dir=debug_dir_xn,
                    #         n_images_to_log=100,
                    #         milestone=train_state.step,
                    #         satellite_type=modality_name,
                    #         was_normalized_to_neg_one_to_one=True
                    #     )
                    #     print(f"Saved debug images for {modality_name} in {debug_dir_xn}")
                    #     visualise_bands(
                    #         inputs=None,
                    #         reconstructions=decoded_eps,
                    #         save_dir=debug_dir_eps,
                    #         n_images_to_log=100,
                    #         milestone=train_state.step,
                    #         satellite_type=modality_name,
                    #         was_normalized_to_neg_one_to_one=True
                    #     )
                    #     print(f"Saved debug images for {modality_name} in {debug_dir_eps}")
                    # exit()
                    # TO REMOVE ========================================================
                    
            
                    # DEBUG --- DECODE SAMPLED LATENTS TO CHECK QUALITY OF TRAINED VAEs ==============================================
                    # For debugging, let's decode and save the images we are using for training
                    # if accelerator.is_main_process:
                    #     from visualisations.visualise_bands import visualise_bands
                    #     from ddm.pre_post_process_data import post_process_data
                    #     import glob
                    #     import re

                    #     base_debug_dir = "debug_images/visualisations"
                    #     # Find all existing debug directories matching the pattern
                    #     existing_dirs = glob.glob(f"{base_debug_dir}_*")
                    #     # Extract numbers from the directory names
                    #     numbers = []
                    #     for d in existing_dirs:
                    #         match = re.search(rf"{re.escape(base_debug_dir)}_(\d+)", d)
                    #         if match:
                    #             numbers.append(int(match.group(1)))
                    #     next_num = max(numbers) + 1 if numbers else 0
                    #     debug_dir = f"{base_debug_dir}_{next_num}"
                    #     os.makedirs(debug_dir, exist_ok=True)
                    #     samples_unstacked = [decode(_z, modality_name) for modality_name, _z in zip(config.z_shapes.keys(), _zs)]
                    #     for modality, sample in zip(config.z_shapes.keys(), samples_unstacked):            
                    #         visualise_bands(inputs=None, reconstructions=sample,
                    #                         save_dir=debug_dir + f"/{modality}",
                    #                         n_images_to_log=config.sample.n_images_to_log, milestone=train_state.step,
                    #                         satellite_type=modality,
                    #                         tif_bands=['cloud_mask'] if 'cloud_mask' in modality else [],
                    #                         was_normalized_to_neg_one_to_one=True,
                    #                         min_db=config.dataset.min_db[modality],
                    #                         max_db=config.dataset.max_db[modality],
                    #                         min_positive=config.dataset.min_positive[modality])
                    # DEBUG --- END ================================================================================================== 
                    
                    
                    # For debugging, let's decode and save the images we are using for training
                    # from visualisations.visualise_bands import visualise_bands
                    # from ddm.pre_post_process_data import post_process_data
                    # debug_dir = f"debug_images"
                    # os.makedirs(debug_dir, exist_ok=True)
                    # samples_unstacked = [decode(_z, modality_name) for modality_name, _z in zip(config.z_shapes.keys(), _zs)]
                    # for modality, sample in zip(config.z_shapes.keys(), samples_unstacked):
                    #     visualise_bands(inputs=None, reconstructions=sample,
                    #                     save_dir=debug_dir + f"/{modality}",
                    #                     n_images_to_log=config.sample.n_images_to_log, milestone=train_state.step,
                    #                     satellite_type=modality,
                    #                     tif_bands=['cloud_mask'] if 'cloud_mask' in modality else [],
                    #                     was_normalized_to_neg_one_to_one=True,
                    #                     min_db=config.dataset.min_db[modality],
                    #                     max_db=config.dataset.max_db[modality],
                    #                     min_positive=config.dataset.min_positive[modality])
                    
                    # exit()
                    
                    
                    loss = LSimple(_zs, nnet, _schedule, multi_modal=config.train.multi_modal)
                else:
                    raise NotImplementedError("Never here")
                    _z = autoencoder.sample(_batch) if 'feature' in config.dataset.name else encode(_batch)
                    loss = LSimple(_z, nnet, _schedule)
            elif config.train.mode == 'cond':
                raise NotImplementedError("Never here")
                _z = autoencoder.sample(_batch[0]) if 'feature' in config.dataset.name else encode(_batch[0])
                loss = LSimple(_z, nnet, _schedule, y=_batch[1])
            else:
                raise NotImplementedError(config.train.mode)
            # Track loss
            _metrics['loss'] = accelerator.gather(loss.detach()).mean()
            
            # Backward pass
            accelerator.backward(loss.mean())
            
            # Optimizer / scheduler / EMA update only run when accumulation boundary is reached
            if accelerator.sync_gradients:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                train_state.ema_update(config.get('ema_rate', 0.9999))
                train_state.step += 1
        return dict(lr=train_state.optimizer.param_groups[0]['lr'], **_metrics)

    def dpm_solver_sample(_n_samples, _sample_steps, **kwargs):
        _z_init = torch.randn(_n_samples, *config.z_shape, device=device)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            eps_pre = nnet_ema(x, t, **kwargs)
            return eps_pre

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _z = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)
        return decode(_z)
    
    def combine_joint(z):
        z = torch.concat([einops.rearrange(z_i, 'B C H W -> B (C H W)') for z_i in z], dim=-1)
        return z
    
    def split_joint(x):
        # C, H, W = config.z_shape # old
        Cs = [config.z_shapes[modality][0] for modality in config.z_shapes.keys()]
        Hs = [config.z_shapes[modality][1] for modality in config.z_shapes.keys()]
        Ws = [config.z_shapes[modality][2] for modality in config.z_shapes.keys()]
        # z_dim = C * H * W # old
        z_dims = [C * H * W for C, H, W in zip(Cs, Hs, Ws)]
        z = x.split(z_dims, dim=1)
        # z = [einops.rearrange(z_i, 'B (C H W) -> B C H W', C=C, H=H, W=W) for z_i in z] # old
        z = [einops.rearrange(z_i, 'B (C H W) -> B C H W', C=Cs[i], H=Hs[i], W=Ws[i]) for i, z_i in enumerate(z)]
        return z
    
    def dpm_solver_sample_multi_modal(_n_modalities, _n_samples, _sample_steps, return_latents=False, **kwargs):
        _z_init = [torch.randn(_n_samples, *config.z_shapes[modality], device=device) for modality in config.z_shapes.keys()]
        _z_init = combine_joint(_z_init)
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.tensor(_betas, device=device).float())

        def model_fn(x, t_continuous):
            t = t_continuous * _schedule.N
            
            timesteps = [t] * _n_modalities
            z = split_joint(x)
            z_out = nnet_ema(z, t_imgs=timesteps)
            x_out = combine_joint(z_out)
            # eps_pre = nnet_ema(x, t, **kwargs)
            return x_out

        dpm_solver = DPM_Solver(model_fn, noise_schedule, predict_x0=True, thresholding=False)
        _zs = dpm_solver.sample(_z_init, steps=_sample_steps, eps=1. / _schedule.N, T=1.)
        _zs = split_joint(_zs)
        samples_unstacked = [decode(_z, modality_name) for modality_name, _z in zip(config.z_shapes.keys(), _zs)]
        if return_latents:
            return samples_unstacked, _zs
        else:
            return samples_unstacked

    def eval_step(n_samples, sample_steps):
        logging.info(f'eval_step: n_samples={n_samples}, sample_steps={sample_steps}'
                     f'mini_batch_size={config.sample.mini_batch_size}')

        def sample_fn(_n_samples):
            if config.train.mode == 'uncond':
                kwargs = dict()
            elif config.train.mode == 'cond':
                kwargs = dict(y=dataset.sample_label(_n_samples, device=device))
            else:
                raise NotImplementedError
            return dpm_solver_sample(_n_samples, sample_steps, **kwargs)


        with tempfile.TemporaryDirectory() as temp_path:
            path = config.sample.path or temp_path
            if accelerator.is_main_process:
                os.makedirs(path, exist_ok=True)
            utils.sample2dir(accelerator, path, n_samples, config.sample.mini_batch_size, sample_fn, dataset.unpreprocess)

            _fid = 0
            if accelerator.is_main_process:
                _fid = calculate_fid_given_paths((dataset.fid_stat, path))
                logging.info(f'step={train_state.step} fid{n_samples}={_fid}')
                with open(os.path.join(config.workdir, 'eval.log'), 'a') as f:
                    print(f'step={train_state.step} fid{n_samples}={_fid}', file=f)
                wandb.log({f'fid{n_samples}': _fid}, step=train_state.step)
            _fid = torch.tensor(_fid, device=device)
            _fid = accelerator.reduce(_fid, reduction='sum')

        return _fid.item()

    logging.info(f'Start fitting, step={train_state.step}, mixed_precision={config.mixed_precision}')

    step_fid = []
    while train_state.step < config.train.n_steps:
        nnet.train()
        # batch = tree_map(lambda x: x.to(device, non_blocking=True), next(data_generator))
        batch = next(data_generator)
        # batch_keys = list(batch.keys())
        # for modality in batch_keys:
        #     try:
        #         print(f"modality: {modality}, batch[modality].shape: {batch[modality].shape}")
        #     except Exception as e:
        #         print(f"batch[modality]: {batch[modality]}")
        #         print(f"batch keys: {batch_keys}")
        #         print(f"modality: {modality}, batch[modality] is None")
        #         raise e
        # exit()
        metrics = train_step(batch)

        nnet.eval()
        if accelerator.is_main_process and train_state.step % config.train.log_interval == 0:
            logging.info(utils.dct2str(dict(step=train_state.step, **metrics)))
            logging.info(config.workdir)
            wandb.log(metrics, step=train_state.step)

        if accelerator.is_main_process and train_state.step % config.train.eval_interval == 0:
            torch.cuda.empty_cache()
            logging.info('Save a grid of images...')
            if config.train.mode == 'uncond':
                if config.train.multi_modal:
                    samples = dpm_solver_sample_multi_modal(_n_modalities=len(config.nnet.modalities), _n_samples=config.sample.n_images_to_log, _sample_steps=50)
                else:
                    samples = dpm_solver_sample(_n_samples=config.sample.n_images_to_log, _sample_steps=50)
            elif config.train.mode == 'cond':
                raise NotImplementedError
                y = einops.repeat(torch.arange(config.sample.n_images_to_log, device=device) % dataset.K, 'nrow -> (nrow ncol)', ncol=10)
                samples = dpm_solver_sample(_n_samples=config.sample.n_images_to_log, _sample_steps=50, y=y)
            else:
                raise NotImplementedError
            
            if config.train.multi_modal:
                from visualisations.visualise_bands import visualise_bands
                from ddm.pre_post_process_data import post_process_data
                for modality, sample in zip(config.z_shapes.keys(), samples):
                    sample_post_processed = post_process_data(sample, modality, config.dataset)
                    visualise_bands(inputs=None, reconstructions=sample_post_processed,
                                    save_dir=f'{config.sample_dir}/{modality}',
                                    n_images_to_log=config.sample.n_images_to_log, milestone=train_state.step,
                                    satellite_type=modality)
                # Merge visualisations
                merge_visualisations(results_path=config.sample_dir, verbose=False)
                
                # samples = [post_process_data(inputs=None,
                #                              reconstructions=sample,
                #                              satellite_type=modality,
                #                              was_normalized_to_neg_one_to_one=True)
                #            for modality, sample in zip(config.z_shapes.keys(), samples)]
                # # samples = torch.stack([dataset.unpreprocess(sample) for sample in samples], dim=0)  # stack instead of cat
                # b = samples.shape[1]  # batch size
                # # Properly interleave samples from all modalities
                # # For each sample index, get all modalities before moving to next sample
                # samples = torch.stack([samples[j, i] for i in range(b) for j in range(config.nnet.num_modalities)]).view(-1, *samples.shape[2:])
                # # If the number of modalities is 3 then we plot in 9 columns
                # n_cols = 9 if config.nnet.num_modalities == 3 else 10
                # samples = make_grid(samples, n_cols)
            else:
                samples = make_grid(dataset.unpreprocess(samples), 10)
            # save_image(samples, os.path.join(config.sample_dir, f'{train_state.step}.png'))
            # wandb.log({'samples': wandb.Image(samples)}, step=train_state.step)
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

        if train_state.step % config.train.save_interval == 0 or train_state.step == config.train.n_steps:
            torch.cuda.empty_cache()
            logging.info(f'Save and eval checkpoint {train_state.step}...')
            if accelerator.is_main_process:
                try:
                    train_state.save(os.path.join(config.ckpt_root, f'{train_state.step}.ckpt'))
                except Exception as e:
                    logging.error(f" ==> Failed to save checkpoint: {e}!!!")
            accelerator.wait_for_everyone()
            # TODO: Skip FID for now
            # fid = eval_step(n_samples=10000, sample_steps=50)  # calculate fid of the saved checkpoint
            # step_fid.append((train_state.step, fid))
            torch.cuda.empty_cache()
        accelerator.wait_for_everyone()

    logging.info(f'Finish fitting, step={train_state.step}')
    logging.info(f'step_fid: {step_fid}')
    step_best = sorted(step_fid, key=lambda x: x[1])[0][0]
    logging.info(f'step_best: {step_best}')
    train_state.load(os.path.join(config.ckpt_root, f'{step_best}.ckpt'))
    del metrics
    accelerator.wait_for_everyone()
    eval_step(n_samples=config.sample.n_samples, sample_steps=config.sample.sample_steps)



from absl import flags
from absl import app
from ml_collections import config_flags
import sys
from pathlib import Path


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("workdir", None, "Work unit directory.")


def get_config_name():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            return Path(argv[i].split('=')[-1]).stem

def get_config_path():
    argv = sys.argv
    for i in range(1, len(argv)):
        if argv[i].startswith('--config='):
            path = argv[i].split('=')[-1]
            if path.startswith('configs/'):
                path = path[len('configs/'):]
            return path

def get_hparams():
    argv = sys.argv
    lst = []
    for i in range(1, len(argv)):
        assert '=' in argv[i]
        if argv[i].startswith('--config.') and not argv[i].startswith('--config.dataset.path'):
            hparam, val = argv[i].split('=')
            hparam = hparam.split('.')[-1]
            if hparam.endswith('path'):
                val = Path(val).stem
            lst.append(f'{hparam}={val}')
    hparams = '-'.join(lst)
    if hparams == '':
        hparams = 'default'
    return hparams


def main(argv):
    config = FLAGS.config
    # config.config_name = get_config_name()
    config.config_name = get_config_path().strip('.py')
    config.hparams = get_hparams()
    config.workdir = FLAGS.workdir or os.path.join('workdir', config.config_name, config.hparams)
    config.ckpt_root = os.path.join(config.workdir, 'ckpts')
    config.sample_dir = os.path.join(config.workdir, 'samples')
    train(config)


if __name__ == "__main__":
    app.run(main)
