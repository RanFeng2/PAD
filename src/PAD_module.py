import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmengine.logging import MessageHub
from mmengine.model.weight_init import trunc_normal_init, constant_init, normal_init, kaiming_init
import math


def build_activation_layer(activation_cfg=dict(type='GELU')):
    if activation_cfg['type'] == 'GELU':
        return nn.GELU()
    elif activation_cfg['type'] == 'ReLU':
        return nn.ReLU()
    elif activation_cfg['type'] == 'LeakyReLU':
        negative_slope = activation_cfg['negative_slope'] if 'negative_slope' in activation_cfg else 0.01
        return nn.LeakyReLU(negative_slope=negative_slope)
    else:
        raise ValueError(f"Unsupported activation function type: {activation_cfg['type']}")


class PSC(nn.Module):
    def __init__(self, 
                 dim, 
                 reduction=8, 
                 activation_cfg=dict(type='GELU'),
                 ):
        super().__init__()
        self.hidden_dim = dim // reduction if dim > reduction else 1

        self.attention = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, 1),
            build_activation_layer(activation_cfg),
            nn.Conv2d(self.hidden_dim, 1, 1))
        
    def forward(self, phase):
        mask = torch.sigmoid(self.attention(phase))
        return phase * (1 + mask)


class MLP(nn.Module):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 hidden_list,
                 activation_cfg=dict(type='ReLU')
                 ):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(build_activation_layer(activation_cfg))
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class PAD_module(nn.Module):
    def __init__(self, 
                 dim: int,
                 penalty_weight: float = 0.1, 
                 reduction: int = 8,
                 radius: float = 0.1,
                 activation_cfg: dict = dict(type='LeakyReLU')
                 ):
        super().__init__()
        self.dim = dim
        self.penalty_weight = penalty_weight
        self.activation_cfg = activation_cfg

        self.residual = nn.Conv2d(2 * self.dim, self.dim, 1)
        self.fc = nn.Conv2d(2 * self.dim, 1, kernel_size=1)
        self.catconv1 = nn.Conv2d(2 * self.dim, self.dim, kernel_size=1)

        # for phase
        self.psc = PSC(self.dim, reduction=reduction, activation_cfg=activation_cfg)

        # for amplitude
        self.temperature = 10
        self.radius = nn.Parameter(torch.tensor(radius))
        self.asf = MLP(self.dim, self.dim, [self.dim, self.dim*2], activation_cfg=activation_cfg)
        self.mix_conv = nn.Conv2d(2 * self.dim, self.dim, 1)

    def frequency_mlp(self, x, mask, mlp):
        b, c, h, w = x.shape
        x = x * mask
        x = x.view(b, c, -1).permute(0, 2, 1)
        x = mlp(x)
        x = x.permute(0, 2, 1).view(b, c, h, w)
        return x
    
    def forward(self, x1, x2):
        H, W = x1.shape[-2:]
        device = x1.device
        
        x = torch.cat([x1, x2], dim=1)
        residual = self.residual(x)                                                             # (B, C, H, W)
        attention = torch.sigmoid(self.fc(x))
        x = self.catconv1(torch.cat([x1 * attention, x2 * (1 - attention)], dim=1))             # (B, C, H, W)

        x = x + residual

        #! Fourier transform
        x = torch.fft.rfft2(x, dim=(-2, -1), norm='ortho')                                      # (B, C, H, W//2+1)
        x = torch.fft.fftshift(x, dim=(-2, -1))

        #! for amplitude
        x_mag = torch.abs(x)
        x_mag_residual = x_mag
        # generate the frequency spectrum mask
        center = (x.shape[-2] // 2, x.shape[-1] // 2)
        Y, X = torch.meshgrid(torch.arange(x.shape[-2], device=device), 
                            torch.arange(x.shape[-1], device=device))
        dist = ((X - center[1]) ** 2 + (Y - center[0]) ** 2).sqrt()
        normalized_dist = dist / dist.max()
        # divide the frequency spectrum into high and low frequencies        
        radius = torch.sigmoid(self.radius)
        # generate the mask, used to distinguish high frequencies
        high_mask = torch.sigmoid((normalized_dist - radius) * self.temperature)
        x_mag_high = self.frequency_mlp(x_mag, high_mask, self.asf)
        x_mag = torch.cat([x_mag, x_mag_high], dim=1)
        x_mag = self.mix_conv(x_mag)
        # residual
        x_mag = x_mag + x_mag_residual

        self.penalty_loss = F.mse_loss(x_mag, x_mag_residual) * self.penalty_weight

        #! for phase
        x_pha = torch.angle(x)
        x_pha = self.psc(x_pha)

        #! inverse Fourier transform
        real = x_mag * torch.cos(x_pha)
        imag = x_mag * torch.sin(x_pha)
        x = torch.complex(real, imag)
        x = torch.fft.ifftshift(x, dim=(-2, -1))
        x = torch.fft.irfft2(x, s=(H, W), dim=(-2, -1), norm='ortho')

        x = x + residual

        return x
    

if __name__ == '__main__':
    model = PAD_module(dim=128, penalty_weight=0.1)
    x1 = torch.randn(2, 128, 64, 64)
    x2 = torch.randn(2, 128, 64, 64)
    y = model(x1, x2)
    print(y.shape)
    print(model.penalty_loss)
    