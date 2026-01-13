"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

import torch
import torch.nn as nn
from torchvision import models
from collections import namedtuple

from .util import get_ckpt_path

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "taming/modules/autoencoder/lpips")
        self.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")), strict=False)
        return model

    def forward(self, input, target):
        if input.shape[1] != 3 or target.shape[1] != 3:
            # Handle multi-channel inputs by grouping them in sets of 3
            return self.forward_multi_channel(input, target)
        
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val
    
    def forward_multi_channel(self, input, target):
        """
        Handle inputs with more than 3 channels by grouping them in sets of 3
        and computing the mean of LPIPS losses across all groups.
        For remaining channels less than 3, create an overlapping group using previous bands.
        
        Special cases:
        - 1 channel: duplicate the channel to create a 3-channel input
        - 2 channels: add a third zero-filled channel
        """
        batch_size, in_channels, height, width = input.shape
        
        # Special handling for inputs with fewer than 3 channels
        if in_channels == 1:
            # Duplicate the single channel to create a 3-channel input
            input_processed = input.repeat(1, 3, 1, 1)
            target_processed = target.repeat(1, 3, 1, 1)
            return self.forward(input_processed, target_processed)
        elif in_channels == 2:
            # Add a third zero-filled channel
            zeros = torch.zeros_like(input[:, :1, :, :])
            input_processed = torch.cat([input, zeros], dim=1)
            target_processed = torch.cat([target, zeros], dim=1)
            return self.forward(input_processed, target_processed)
        
        # Process complete groups of 3 channels
        group_losses = []
        for i in range(0, in_channels - (in_channels % 3), 3):
            in_group = input[:, i:i+3, :, :]
            target_group = target[:, i:i+3, :, :]
            group_loss = self.forward(in_group, target_group)
            group_losses.append(group_loss)
        
        # Handle remaining channels (if any) by creating an overlapping group
        remaining = in_channels % 3
        if remaining > 0:
            # Get the last 3 channels by using previous bands if needed
            last_in_group = input[:, -3:, :, :]
            last_target_group = target[:, -3:, :, :]
            
            group_loss = self.forward(last_in_group, last_target_group)
            group_losses.append(group_loss)
        
        # Return the mean of all group losses
        return torch.mean(torch.stack(group_losses, dim=0), dim=0)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        # Check if input has exactly 3 channels (RGB)
        if inp.shape[1] == 3:
            return (inp - self.shift) / self.scale
        else:
            # For inputs with other channel counts, forward_multi_channel in LPIPS will handle them
            raise ValueError(f"ScalingLayer expects 3-channel input, got {inp.shape[1]} channels. Use LPIPS.forward_multi_channel for multi-channel inputs.")


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x,eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2,dim=1,keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2,3],keepdim=keepdim)

