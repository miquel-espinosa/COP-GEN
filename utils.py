import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from torchvision.utils import save_image
from absl import logging
from PIL import Image, ImageDraw, ImageFont

def set_logger(log_level='info', fname=None):
    import logging as _logging
    handler = logging.get_absl_handler()
    formatter = _logging.Formatter('%(asctime)s - %(filename)s - %(message)s')
    handler.setFormatter(formatter)
    logging.set_verbosity(log_level)
    if fname is not None:
        handler = _logging.FileHandler(fname)
        handler.setFormatter(formatter)
        logging.get_absl_logger().addHandler(handler)


def dct2str(dct):
    return str({k: f'{v:.6g}' for k, v in dct.items()})


def get_nnet(name, **kwargs):
    if name == 'uvit':
        from libs.uvit import UViT
        return UViT(**kwargs)
    elif name == 'uvit_t2i':
        from libs.uvit_t2i import UViT
        return UViT(**kwargs)
    elif name == 'uvit_multi_post_ln':
        from libs.uvit_multi_post_ln import UViT
        return UViT(**kwargs)
    elif name == 'uvit_multi_post_ln_v1':
        from libs.uvit_multi_post_ln_v1 import UViT
        return UViT(**kwargs)
    elif name == 'triffuser_multi_post_ln':
        from libs.triffuser_multi_post_ln import Triffuser
        return Triffuser(**kwargs)
    elif name == 'copgen_multi_post_ln':
        from libs.copgen_multi_post_ln import COPGEN
        return COPGEN(**kwargs)
    else:
        raise NotImplementedError(name)

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def get_optimizer(params, name, **kwargs):
    if name == 'adam':
        from torch.optim import Adam
        return Adam(params, **kwargs)
    elif name == 'adamw':
        from torch.optim import AdamW
        return AdamW(params, **kwargs)
    else:
        raise NotImplementedError(name)


def customized_lr_scheduler(optimizer, warmup_steps=-1):
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'customized':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)


def ema(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


class TrainState(object):
    def __init__(self, optimizer, lr_scheduler, step, nnet=None, nnet_ema=None):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.step = step
        self.nnet = nnet
        self.nnet_ema = nnet_ema

    def ema_update(self, rate=0.9999):
        if self.nnet_ema is not None:
            ema(self.nnet_ema, self.nnet, rate)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.step, os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                torch.save(val.state_dict(), os.path.join(path, f'{key}.pth'))

    def load_old_version(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        # print("We start lr scheduler from scratch so that we have a WARM UP phase")
        # self.lr_scheduler.load_state_dict(torch.load(os.path.join(path, 'lr_scheduler.pth'), map_location='cpu'))
        # for key, val in self.__dict__.items():
        #     if key != 'step' and val is not None:
        #         print("we are here and the key is ", key)
        #         print("the value is ", val)
        #         val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))

        # self.nnet.load_state_dict(torch.load(os.path.join(path, 'nnet.pth'), map_location='cpu'))
        # self.nnet_ema.load_state_dict(torch.load(os.path.join(path, 'nnet_ema.pth'), map_location='cpu'))
        # self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth'), map_location='cpu'))

        self._load_network_with_mismatch(self.nnet, os.path.join(path, 'nnet.pth'))
        self._load_network_with_mismatch(self.nnet_ema, os.path.join(path, 'nnet_ema.pth'))
        # self._load_optimizer_with_mismatch(os.path.join(path, 'optimizer.pth'))
        
    def load(self, path):
        logging.info(f'load from {path}')
        self.step = torch.load(os.path.join(path, 'step.pth'))
        for key, val in self.__dict__.items():
            if key != 'step' and val is not None:
                val.load_state_dict(torch.load(os.path.join(path, f'{key}.pth'), map_location='cpu'))
        


    def _load_network_with_mismatch(self, network, checkpoint_path):
        # Initialize network with zeros
        for param in network.parameters():
            param.data.zero_()
        checkpoint_state_dict = torch.load(checkpoint_path, map_location='cpu')
        current_model_state_dict = {name: torch.zeros_like(param) for name, param in network.state_dict().items()}
        new_state_dict = current_model_state_dict.copy()

        architecture_changed = False

        for name, model_param in current_model_state_dict.items():
            if name in checkpoint_state_dict:
                checkpoint_param = checkpoint_state_dict[name]

                if model_param.shape == checkpoint_param.shape:
                    new_state_dict[name] = checkpoint_param
                else:
                    architecture_changed = True
                    logging.info(f"Shape mismatch for {name}: model shape {model_param.shape}, checkpoint shape {checkpoint_param.shape}. Adapting.")
                    
                    temp_param = model_param.clone() # Start with model's current init for this param

                    try:
                        if name == 'pos_embed':
                            # model: [B, N_new, D], checkpoint: [B, N_old, D]
                            # Handles cases where N_new != N_old
                            if model_param.dim() == 3 and checkpoint_param.dim() == 3 and \
                               model_param.shape[0] == checkpoint_param.shape[0] and \
                               model_param.shape[2] == checkpoint_param.shape[2]:
                                min_N = min(model_param.shape[1], checkpoint_param.shape[1])
                                temp_param[:, :min_N, :] = checkpoint_param[:, :min_N, :]
                                new_state_dict[name] = temp_param
                                logging.info(f"Adapted {name}: Copied embeddings for the first {min_N} tokens.")
                            else:
                                logging.warning(f"Cannot adapt {name} due to incompatible dimensions beyond sequence length. Keeping model's init.")
                                new_state_dict[name] = model_param # Keep original model param
                        
                        elif name.startswith('patch_embeds.') and name.endswith('.proj.weight'):
                            # model: [O, C_new, K_h, K_w], checkpoint: [O, C_old, K_h, K_w]
                            # Handles cases where C_new != C_old (input channels)
                            if model_param.dim() == 4 and checkpoint_param.dim() == 4 and \
                               model_param.shape[0] == checkpoint_param.shape[0] and \
                               model_param.shape[2:] == checkpoint_param.shape[2:]:
                                min_C_in = min(model_param.shape[1], checkpoint_param.shape[1])
                                temp_param[:, :min_C_in, :, :] = checkpoint_param[:, :min_C_in, :, :]
                                new_state_dict[name] = temp_param
                                logging.info(f"Adapted {name}: Copied weights for the first {min_C_in} input channels.")
                            else:
                                logging.warning(f"Cannot adapt {name} due to incompatible dimensions. Keeping model's init.")
                                new_state_dict[name] = model_param

                        elif name.startswith('decoder_preds.') and name.endswith('.weight'):
                            # model: [O_new, I], checkpoint: [O_old, I]
                            # Handles cases where O_new != O_old (output features)
                            if model_param.dim() == 2 and checkpoint_param.dim() == 2 and \
                               model_param.shape[1] == checkpoint_param.shape[1]:
                                min_O_out = min(model_param.shape[0], checkpoint_param.shape[0])
                                temp_param[:min_O_out, :] = checkpoint_param[:min_O_out, :]
                                new_state_dict[name] = temp_param
                                logging.info(f"Adapted {name}: Copied weights for the first {min_O_out} output features.")
                            else:
                                logging.warning(f"Cannot adapt {name} due to incompatible dimensions. Keeping model's init.")
                                new_state_dict[name] = model_param

                        elif name.startswith('decoder_preds.') and name.endswith('.bias'):
                            # model: [O_new], checkpoint: [O_old]
                            # Handles cases where O_new != O_old (output features for bias)
                            if model_param.dim() == 1 and checkpoint_param.dim() == 1:
                                min_O_out = min(model_param.shape[0], checkpoint_param.shape[0])
                                temp_param[:min_O_out] = checkpoint_param[:min_O_out]
                                new_state_dict[name] = temp_param
                                logging.info(f"Adapted {name} bias: Copied for the first {min_O_out} features.")
                            else:
                                logging.warning(f"Cannot adapt {name} bias. Keeping model's init.")
                                new_state_dict[name] = model_param
                        else:
                            logging.warning(f"Unhandled shape mismatch for {name}. Parameter will keep its initial value in the new model.")
                            new_state_dict[name] = model_param # Keep original model param
                    except Exception as e:
                        logging.error(f"Error adapting {name}: {e}. Parameter will keep its initial value.")
                        new_state_dict[name] = model_param # Fallback to original model param on error

            else:
                logging.info(f"Parameter {name} not found in checkpoint. It will keep its initial value in the new model.")
                # new_state_dict[name] already contains model_param's initial value from the copy

        network.load_state_dict(new_state_dict)
        logging.info("Finished loading nnet state dict with adaptations.")

        if architecture_changed:
            logging.info("\033[91mModel architecture changed during loading. Optimizer state might need to be reset if it was tied to old parameter shapes.\033[0m")
        else:
            logging.info("\033[92mModel architecture did not change during loading. Optimizer state is not affected.\033[0m")
        

    def resume(self, ckpt_root, step=None):
        if not os.path.exists(ckpt_root):
            return
        if step is None:
            ckpts = list(filter(lambda x: '.ckpt' in x, os.listdir(ckpt_root)))
            if not ckpts:
                return
            steps = map(lambda x: int(x.split(".")[0]), ckpts)
            step = max(steps)
        ckpt_path = os.path.join(ckpt_root, f'{step}.ckpt')
        logging.info(f'resume from {ckpt_path}')
        self.load(ckpt_path)

    def to(self, device):
        for key, val in self.__dict__.items():
            if isinstance(val, nn.Module):
                val.to(device)


def cnt_params(model):
    return sum(param.numel() for param in model.parameters())


def initialize_train_state(config, device):
    params = []

    nnet = get_nnet(**config.nnet)
    params += nnet.parameters()
    nnet_ema = get_nnet(**config.nnet)
    nnet_ema.eval()
    logging.info(f'nnet has {cnt_params(nnet)} parameters')

    optimizer = get_optimizer(params, **config.optimizer)
    lr_scheduler = get_lr_scheduler(optimizer, **config.lr_scheduler)

    train_state = TrainState(optimizer=optimizer, lr_scheduler=lr_scheduler, step=0,
                             nnet=nnet, nnet_ema=nnet_ema)
    train_state.ema_update(0)
    train_state.to(device)
    return train_state


def amortize(n_samples, batch_size):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


def sample2dir(accelerator, path, n_samples, mini_batch_size, sample_fn, unpreprocess_fn=None):
    os.makedirs(path, exist_ok=True)
    idx = 0
    batch_size = mini_batch_size * accelerator.num_processes

    for _batch_size in tqdm(amortize(n_samples, batch_size), disable=not accelerator.is_main_process, desc='sample2dir'):
        samples = unpreprocess_fn(sample_fn(mini_batch_size))
        samples = accelerator.gather(samples.contiguous())[:_batch_size]
        if accelerator.is_main_process:
            for sample in samples:
                save_image(sample, os.path.join(path, f"{idx}.png"))
                idx += 1


def grad_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm



def center_crop(width, height, img):
    resample = {'box': Image.BOX, 'lanczos': Image.LANCZOS}['lanczos']
    crop = np.min(img.shape[:2])
    img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
          (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]  # center crop
    try:
        img = Image.fromarray(img, 'RGB')
    except:
        img = Image.fromarray(img)
    img = img.resize((width, height), resample)  # resize the center crop from [crop, crop] to [width, height]

    return np.array(img).astype(np.uint8)


def drawRoundRec(draw, color, x, y, w, h, r):
    drawObject = draw

    '''Rounds'''
    drawObject.ellipse((x, y, x + r, y + r), fill=color)
    drawObject.ellipse((x + w - r, y, x + w, y + r), fill=color)
    drawObject.ellipse((x, y + h - r, x + r, y + h), fill=color)
    drawObject.ellipse((x + w - r, y + h - r, x + w, y + h), fill=color)

    '''rec.s'''
    drawObject.rectangle((x + r / 2, y, x + w - (r / 2), y + h), fill=color)
    drawObject.rectangle((x, y + r / 2, x + w, y + h - (r / 2)), fill=color)


def add_water(img, text='UniDiffuser', pos=3):
    width, height = img.size
    scale = 4
    scale_size = 0.5
    img = img.resize((width * scale, height * scale), Image.LANCZOS)
    result = Image.new(img.mode, (width * scale, height * scale), color=(255, 255, 255))
    result.paste(img, box=(0, 0))

    delta_w = int(width * scale * 0.27 * scale_size)  # text width
    delta_h = width * scale * 0.05 * scale_size  # text height
    postions = np.array([[0, 0], [0, height * scale - delta_h], [width * scale - delta_w, 0],
                         [width * scale - delta_w, height * scale - delta_h]])
    postion = postions[pos]
    # 文本
    draw = ImageDraw.Draw(result)
    fillColor = (107, 92, 231)
    setFont = ImageFont.truetype("assets/ArialBoldMT.ttf", int(width * scale * 0.05 * scale_size))
    delta = 20 * scale_size
    padding = 15 * scale_size
    drawRoundRec(draw, (223, 230, 233), postion[0] - delta - padding, postion[1] - delta - padding,
                 w=delta_w + 2 * padding, h=delta_h + 2 * padding, r=50 * scale_size)
    draw.text((postion[0] - delta, postion[1] - delta), text, font=setFont, fill=fillColor)

    return result.resize((width, height), Image.LANCZOS)
