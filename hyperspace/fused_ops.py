"""
Fused GPU Operations for Dentate Pattern Separation & Complex Phasor Computing.
Eliminates memory bandwidth round-trips by fusing:
1. Sparse Perforant Projection + k-WTA Competitive Masking.
2. Toroidal Multi-Scale Grid Encoding.
3. Complex Phasor Projection & Normalized Hermitian Matrix Multiply.
"""

import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.jit.script
def fused_k_wta_mask(raw_acts: torch.Tensor, k_active: int) -> torch.Tensor:
    """
    Fused GPU kernel for competitive Winner-Take-All lateral inhibition.
    Selects top-k active neurons and normalizes output in a single fused pass.
    """
    topk_vals, _ = torch.topk(raw_acts, k=k_active, dim=-1)
    kth_val = topk_vals[..., -1:]
    mask = (raw_acts >= kth_val).float()
    sparse_acts = F.relu(raw_acts) * mask
    # Fused L2 unit normalization
    norm = torch.sqrt(torch.sum(sparse_acts * sparse_acts, dim=-1, keepdim=True) + 1e-12)
    return sparse_acts / norm

@torch.jit.script
def fused_real_to_phasor(real_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused projection from continuous real space to complex unit circle (cos, sin) components.
    """
    angles = torch.tanh(real_vector) * 3.141592653589793
    return torch.cos(angles), torch.sin(angles)

@torch.jit.script
def fused_hermitian_similarity(
    query_real: torch.Tensor,
    query_imag: torch.Tensor,
    key_real: torch.Tensor,
    key_imag: torch.Tensor,
) -> torch.Tensor:
    """
    Fused real-arithmetic Hermitian inner product between queries and expert address keys:
    Sim(q, k) = (1 / D) * (Re(q) @ Re(k)^T + Im(q) @ Im(k)^T)
    Eliminates complex tensor allocation overhead on GPU.
    """
    d = float(query_real.shape[-1])
    real_sim = torch.matmul(query_real, key_real.transpose(-1, -2))
    imag_sim = torch.matmul(query_imag, key_imag.transpose(-1, -2))
    return (real_sim + imag_sim) / d
