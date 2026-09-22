"""
Unit tests for RiemannianSphericalAttention (RSA) and RiemannianSphericalTransformerLayer.
"""

import pytest
import torch
import torch.nn.functional as F
from ishmn.spherical_attention import RiemannianSphericalAttention, RiemannianSphericalTransformerLayer, LoopedRiemannianTransformer

def test_rsa_norm_preservation():
    """Verify that SO(2) geodesic phase winding preserves ||u|| == 1 strictly."""
    attn = RiemannianSphericalAttention(d_model=32, num_heads=4, hyperspace_dim=8)
    u = torch.randn(2, 4, 10, 8)
    u_norm = F.normalize(u, p=2, dim=-1)

    rotated = attn._apply_so2_geodesic_winding(u_norm)
    norms = torch.norm(rotated, p=2, dim=-1)

    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "SO(2) phase winding violated unit sphere norm!"

def test_rsa_forward_backward():
    """Verify forward execution and gradient backpropagation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer = RiemannianSphericalTransformerLayer(
        d_model=32,
        num_heads=4,
        hyperspace_dim=8,
        d_ff=64,
        device=device
    ).to(device)

    x = torch.randn(4, 16, 32, device=device, requires_grad=True)
    out = layer(x)

    assert out.shape == (4, 16, 32)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_looped_transformer_dynamic_depth():
    """Verify that LoopedRiemannianTransformer iteratively executes with ACT."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    looped = LoopedRiemannianTransformer(
        d_model=32,
        num_heads=4,
        hyperspace_dim=8,
        d_ff=64,
        max_loops=6,
        device=device
    ).to(device)

    x = torch.randn(2, 8, 32, device=device, requires_grad=True)
    out, meta = looped(x)

    assert out.shape == (2, 8, 32)
    assert 1 <= meta["final_loops"] <= 6
    assert "ponder_cost" in meta
    assert len(meta["intermediate_states"]) == meta["final_loops"]

    loss = out.sum() + meta["ponder_cost"]
    loss.backward()
    assert x.grad is not None


def test_looped_transformer_with_dynamic_moe_and_memory():
    """Verify that LoopedRiemannianTransformer executes with D=16, dynamic MoE, and working memory."""
    from ishmn.memory import InfiniteSphericalMemory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    memory = InfiniteSphericalMemory(
        num_slots=16,
        hyperspace_dim=16,
        payload_dim=32,
        initial_radius_shells=4
    ).to(device)
    mem_state = memory.init_memory(batch_size=2, device=device)

    looped = LoopedRiemannianTransformer(
        d_model=32,
        num_heads=4,
        hyperspace_dim=16,
        d_ff=64,
        max_loops=4,
        use_dynamic_moe=True,
        moe_initial_experts=2,
        moe_max_experts=8,
        use_working_memory=True,
        device=device
    ).to(device)

    x = torch.randn(2, 6, 32, device=device, requires_grad=True)
    out, meta = looped(x, memory=memory, mem_state=mem_state)

    assert out.shape == (2, 6, 32)
    assert meta["num_experts"] >= 2
    assert "mem_state" in meta
    assert meta["mem_state"] is not None

    loss = out.sum() + meta["ponder_cost"]
    loss.backward()
    assert x.grad is not None


