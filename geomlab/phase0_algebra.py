"""
Phase 0: Algebraic group learning, symmetry discovery, and geometric models.
Implements the tasks and models specified in research/geomlab_results/phase0/PREREG.md:
- Z_113 modular addition dataset builder
- S_5 permutation group dataset builder
- BaselineMLP, BaselineTransformer
- TorusPhaseModel (Abelian flat Clifford torus T^D)
- BlockUnitaryModel (Non-abelian Lie group U(b)^{D/b})
- Spectral participation ratio and geometry diagnostics
"""

import itertools
import math
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Group Data Builders
# ---------------------------------------------------------------------------

def build_z113_dataset(
    p: int = 113,
    train_fraction: float = 0.3,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates all pairs (a, b) in Z_p x Z_p with target (a + b) % p.
    Returns: (train_inputs, train_targets, test_inputs, test_targets)
    """
    rng = np.random.RandomState(seed)
    pairs = []
    targets = []
    for a in range(p):
        for b in range(p):
            pairs.append((a, b))
            targets.append((a + b) % p)

    pairs = np.array(pairs, dtype=np.int64)
    targets = np.array(targets, dtype=np.int64)
    n_total = len(pairs)
    n_train = int(n_total * train_fraction)

    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_inputs = torch.from_numpy(pairs[train_idx])
    train_targets = torch.from_numpy(targets[train_idx])
    test_inputs = torch.from_numpy(pairs[test_idx])
    test_targets = torch.from_numpy(targets[test_idx])

    return train_inputs, train_targets, test_inputs, test_targets


def build_s5_dataset(
    train_fraction: float = 0.3,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, ...]]]:
    """
    Generates all pairs (sigma_1, sigma_2) in S_5 x S_5 with target sigma_1 o sigma_2.
    |S_5| = 5! = 120. Total pairs = 14,400.
    Returns: (train_inputs, train_targets, test_inputs, test_targets, all_permutations)
    """
    rng = np.random.RandomState(seed)
    perms = list(itertools.permutations(range(5)))  # 120 perms
    perm_to_idx = {p: i for i, p in enumerate(perms)}

    # Precompute multiplication table
    n_perms = len(perms)
    pairs = []
    targets = []
    for i in range(n_perms):
        p1 = perms[i]
        for j in range(n_perms):
            p2 = perms[j]
            # Composition: (p1 o p2)[k] = p1[p2[k]]
            composed = tuple(p1[p2[k]] for k in range(5))
            pairs.append((i, j))
            targets.append(perm_to_idx[composed])

    pairs = np.array(pairs, dtype=np.int64)
    targets = np.array(targets, dtype=np.int64)
    n_total = len(pairs)
    n_train = int(n_total * train_fraction)

    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_inputs = torch.from_numpy(pairs[train_idx])
    train_targets = torch.from_numpy(targets[train_idx])
    test_inputs = torch.from_numpy(pairs[test_idx])
    test_targets = torch.from_numpy(targets[test_idx])

    return train_inputs, train_targets, test_inputs, test_targets, perms


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BaselineMLP(nn.Module):
    """
    Standard unconstrained 2-layer MLP control with LayerNorm.
    """
    def __init__(self, num_tokens: int, d_model: int = 128):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.fc1 = nn.Linear(2 * d_model, 2 * d_model)
        self.norm = nn.LayerNorm(2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, num_tokens)

        # Initialize
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 2)
        emb_a = self.embedding(x[:, 0])  # (B, D)
        emb_b = self.embedding(x[:, 1])  # (B, D)
        h = torch.cat([emb_a, emb_b], dim=-1)  # (B, 2D)
        h = F.gelu(self.fc1(h))
        h = self.norm(h)
        logits = self.fc2(h)  # (B, num_tokens)
        return logits


class BaselineTransformer(nn.Module):
    """
    Standard 1-layer Transformer control with causal attention.
    """
    def __init__(self, num_tokens: int, d_model: int = 128, n_heads: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_emb = nn.Parameter(torch.randn(2, d_model) * 0.02)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model)
        )
        self.head = nn.Linear(d_model, num_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2)
        h = self.embedding(x) + self.pos_emb.unsqueeze(0)  # (B, 2, D)
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)
        h = h + attn_out
        h = h + self.mlp(self.norm2(h))
        # Pool or take final token:
        out_token = h[:, 1, :]  # (B, D)
        logits = self.head(out_token)
        return logits


class TorusPhaseModel(nn.Module):
    """
    Abelian Clifford Torus T^D Model.
    Each token k has a learned phase vector theta_k in [-pi, pi)^D.
    Group composition is point-wise phase addition: theta_pred = theta_a + theta_b.
    Readout is cosine / inner product with dictionary:
      logits_k = beta/D * sum_j cos(theta_a_j + theta_b_j - theta_k_j)
    """
    def __init__(self, num_tokens: int, d_model: int = 128):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        # Phases initialized uniformly on circle
        self.phases = nn.Parameter(torch.empty(num_tokens, d_model).uniform_(-math.pi, math.pi))
        self.beta = nn.Parameter(torch.tensor(12.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2)
        theta_a = self.phases[x[:, 0]]  # (B, D)
        theta_b = self.phases[x[:, 1]]  # (B, D)
        theta_sum = theta_a + theta_b   # (B, D)

        # Logits: cos(theta_sum - theta_k)
        # theta_sum: (B, 1, D), self.phases: (1, N, D)
        diff = theta_sum.unsqueeze(1) - self.phases.unsqueeze(0)  # (B, N, D)
        cos_sim = torch.cos(diff).mean(dim=-1)                   # (B, N)
        logits = self.beta * cos_sim                             # (B, N)
        return logits


class BlockUnitaryModel(nn.Module):
    """
    Non-Abelian Lie Group U(b)^{D/b} Model.
    Each token k has M = D/b unitary blocks of size b x b.
    Blocks are parameterized via matrix exponential of anti-Hermitian matrices:
      U = exp(A - A^H)
    Composition is block-matrix multiplication: U_pred = U_a @ U_b.
    Readout is normalized trace similarity with codebook:
      logits_k = beta/D * Re Tr(U_k^H @ U_pred)
    """
    def __init__(self, num_tokens: int, d_model: int = 128, b: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.b = b
        self.num_blocks = d_model // b
        
        # Real and imaginary generator parameters: (num_tokens, num_blocks, b, b)
        # Skew-Hermitian matrix: H = (R - R^T) + i (S + S^T)
        self.real_gen = nn.Parameter(torch.randn(num_tokens, self.num_blocks, b, b) * 0.05)
        self.imag_gen = nn.Parameter(torch.randn(num_tokens, self.num_blocks, b, b) * 0.05)
        self.beta = nn.Parameter(torch.tensor(12.0))

    def get_unitary(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns (batch_or_all, num_blocks, b, b) complex unitary matrices.
        """
        if indices is not None:
            r = self.real_gen[indices]
            img = self.imag_gen[indices]
        else:
            r = self.real_gen
            img = self.imag_gen

        # Make anti-Hermitian: A^H = -A
        # Real part: skew-symmetric: A_real = r - r^T
        a_real = r - r.transpose(-1, -2)
        # Imag part: symmetric: A_imag = img + img^T
        a_imag = img + img.transpose(-1, -2)
        # Combine: A = a_real + 1j * a_imag
        A = torch.complex(a_real, a_imag)

        # Matrix exponential is unitary: U = exp(A)
        U = torch.linalg.matrix_exp(A)
        return U

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2)
        U_a = self.get_unitary(x[:, 0])  # (B, M, b, b)
        U_b = self.get_unitary(x[:, 1])  # (B, M, b, b)
        U_pred = torch.matmul(U_a, U_b)  # (B, M, b, b)

        # Codebook of all tokens: (N, M, b, b)
        U_all = self.get_unitary()

        # Compute Re Tr(U_k^H @ U_pred) via Frobenius inner product:
        # Tr(A^H B) = vec(B) @ vec(A)^H
        U_pred_flat = U_pred.reshape(x.shape[0], -1)              # (B, M * b * b)
        U_all_flat = U_all.reshape(self.num_tokens, -1)           # (N, M * b * b)
        # Dot product: (B, N)
        sim = (U_pred_flat @ U_all_flat.conj().transpose(0, 1)).real / self.d_model

        logits = self.beta * sim
        return logits


# ---------------------------------------------------------------------------
# Diagnostics & Spectral Analysis
# ---------------------------------------------------------------------------

def compute_fourier_participation_ratio(phases: torch.Tensor, p: int = 113) -> float:
    """
    Computes the spectral participation ratio of learned token phases under the discrete
    Fourier transform along the group coordinate:
      F(k, j) = sum_{a=0}^{p-1} exp(-2pi i k a / p) * exp(i theta_{a, j})
    Measures how concentrated the learned solution is on a few group frequencies.
    A small participation ratio indicates sparse harmonic representations (grokking mechanics).
    """
    # phases: (p, D)
    phasors = torch.exp(1j * phases.detach().cpu())  # (p, D)
    # 1D FFT along group dimension (dim 0)
    fft_vals = torch.fft.fft(phasors, dim=0)         # (p, D)
    power = (fft_vals.abs() ** 2).sum(dim=-1)        # (p,) power per frequency k
    
    total_power = power.sum()
    if total_power <= 0:
        return float(p)
    probs = power / total_power
    # Participation ratio = 1 / sum(p_k^2)
    pr = 1.0 / (probs ** 2).sum().item()
    return float(pr)
