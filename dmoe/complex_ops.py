import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        scale = 1.0 / math.sqrt(2.0 * in_features)
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * scale)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_real = F.linear(z_real, self.weight_real) - F.linear(z_imag, self.weight_imag)
        out_imag = F.linear(z_real, self.weight_imag) + F.linear(z_imag, self.weight_real)

        if self.has_bias:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag

        return out_real, out_imag


class ModReLU(nn.Module):
    def __init__(self, features: int, eps: float = 1e-7):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(z_real ** 2 + z_imag ** 2 + self.eps)
        activated_mag = F.relu(mag + self.b)
        scale = activated_mag / mag
        return z_real * scale, z_imag * scale


class ComplexLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma_real = nn.Parameter(torch.ones(d_model))
        self.gamma_imag = nn.Parameter(torch.zeros(d_model))
        self.beta_real = nn.Parameter(torch.zeros(d_model))
        self.beta_imag = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_r = z_real.mean(dim=-1, keepdim=True)
        mean_i = z_imag.mean(dim=-1, keepdim=True)
        z_cent_r = z_real - mean_r
        z_cent_i = z_imag - mean_i

        var = (z_cent_r ** 2 + z_cent_i ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        norm_r = z_cent_r / std
        norm_i = z_cent_i / std

        out_r = norm_r * self.gamma_real - norm_i * self.gamma_imag + self.beta_real
        out_i = norm_r * self.gamma_imag + norm_i * self.gamma_real + self.beta_imag
        return out_r, out_i

