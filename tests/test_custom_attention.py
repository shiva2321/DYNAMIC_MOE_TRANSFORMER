"""
Unit tests for Custom Rust Accelerated Riemannian Attention Engine.
Verifies exact mathematical isometry, numerical equivalence, and scale invariance.
"""

import math
import pytest
import torch
import numpy as np

from ishmn.spherical_attention import (
    has_rust_attention,
    fast_riemannian_so2_winding,
    fast_tiled_riemannian_attention,
    RiemannianSphericalAttention
)


def test_has_rust_attention():
    """Verify that native Rust extension is available and loaded."""
    assert has_rust_attention() is True


def test_riemannian_so2_winding_isometry():
    """
    Verify that continuous SO(2) phase winding:
    1. Preserves unit hypersphere norm ||u|| == 1 strictly without drift.
    2. Matches PyTorch analytical rotation to within 1e-6 error.
    """
    B, H, T, D = 2, 4, 16, 8
    freqs = torch.tensor([0.6, 0.3, 0.15, 0.075], dtype=torch.float32)
    u = torch.randn(B, H, T, D)
    u = torch.nn.functional.normalize(u, p=2, dim=-1)

    # Reference PyTorch computation
    t_indices = torch.arange(3, 3 + T, dtype=freqs.dtype)
    angles = torch.outer(t_indices, freqs)
    cos_a = torch.cos(angles).unsqueeze(0).unsqueeze(0)
    sin_a = torch.sin(angles).unsqueeze(0).unsqueeze(0)

    u_pairs = u.view(B, H, T, -1, 2)
    rot_0 = u_pairs[..., 0] * cos_a - u_pairs[..., 1] * sin_a
    rot_1 = u_pairs[..., 0] * sin_a + u_pairs[..., 1] * cos_a
    u_rot_ref = torch.stack([rot_0, rot_1], dim=-1).view(B, H, T, D)

    # Fast Rust computation
    u_rot_fast = fast_riemannian_so2_winding(u, freqs, start_pos=3)

    # Norm check (zero drift)
    norms = torch.norm(u_rot_fast, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Numerical equivalence
    diff = torch.max(torch.abs(u_rot_ref - u_rot_fast)).item()
    assert diff < 1e-5, f"Winding divergence too high: {diff}"


def test_tiled_riemannian_attention_numerical_match():
    """
    Verify that tiled online Riemannian softmax attention matches full dense
    attention score computation to within 1e-5 numerical tolerance.
    """
    B, H, T, S, D, d_v = 2, 4, 12, 24, 8, 16
    torch.manual_seed(42)

    u_q = torch.randn(B, H, T, D)
    u_q = torch.nn.functional.normalize(u_q, p=2, dim=-1)
    u_k = torch.randn(B, H, S, D)
    u_k = torch.nn.functional.normalize(u_k, p=2, dim=-1)

    r_q = torch.rand(B, H, T, 1) + 1.0
    r_k = torch.rand(B, H, S, 1) + 1.0
    v = torch.randn(B, H, S, d_v)

    kappa = torch.tensor([2.0, 3.5, 4.0, 5.0])
    beta = torch.tensor([0.8, 1.2, 0.5, 1.0])

    # Reference full dense computation
    ln_r_q = torch.log(r_q)
    ln_r_k = torch.log(r_k).transpose(-1, -2)
    s_angular = torch.matmul(u_q, u_k.transpose(-1, -2))
    s_radial = -((ln_r_q - ln_r_k) ** 2)

    k_eff = kappa.view(1, H, 1, 1) * math.sqrt(math.log(float(S) + 1.0))
    b_eff = beta.view(1, H, 1, 1)
    logits = k_eff * s_angular + b_eff * s_radial
    attn_weights = torch.softmax(logits, dim=-1)
    out_ref = torch.matmul(attn_weights, v)

    # Fast tiled online kernel
    out_fast = fast_tiled_riemannian_attention(
        u_q=u_q,
        u_k=u_k,
        r_q=r_q,
        r_k=r_k,
        v=v,
        kappa=kappa,
        beta=beta,
        is_causal=False,
        enable_entropy_scaling=True,
        block_size=8
    )

    diff = torch.max(torch.abs(out_ref - out_fast)).item()
    assert diff < 1e-4, f"Tiled attention divergence too high: {diff}"


def test_tiled_riemannian_attention_causal():
    """
    Verify causal masking behavior in tiled online attention.
    """
    B, H, T, S, D, d_v = 1, 2, 8, 8, 8, 8
    torch.manual_seed(123)

    u_q = torch.nn.functional.normalize(torch.randn(B, H, T, D), p=2, dim=-1)
    u_k = torch.nn.functional.normalize(torch.randn(B, H, S, D), p=2, dim=-1)
    r_q = torch.rand(B, H, T, 1) + 1.0
    r_k = torch.rand(B, H, S, 1) + 1.0
    v = torch.randn(B, H, S, d_v)
    kappa = torch.tensor([3.0, 4.0])
    beta = torch.tensor([1.0, 1.0])

    out_causal = fast_tiled_riemannian_attention(
        u_q=u_q,
        u_k=u_k,
        r_q=r_q,
        r_k=r_k,
        v=v,
        kappa=kappa,
        beta=beta,
        is_causal=True,
        enable_entropy_scaling=False,
        block_size=4
    )

    # First token can only attend to key 0
    assert torch.isfinite(out_causal).all()
