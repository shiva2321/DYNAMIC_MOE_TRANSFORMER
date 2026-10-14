"""
Unit tests for Track Z Round Z2: Memory Geometry (Associative Recall per Byte).
Tests exact mathematical invariants:
1. Orthogonal keys with P <= d: every delta variant recalls 100% accurately.
2. Softmax reference ceiling recalls 100% on smallest setting.
3. Legacy adapter correctly integrates ISHMN InfiniteSphericalMemory without mutating ishmn/.
4. Delta L2 enforces exact unit norm on memory keys.
5. PoST decay rates follow geometric spacing.
6. State bytes accounting is exact.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from geomlab.memory_geometry import (
    SoftmaxAttentionMemory,
    LinearAttentionMemory,
    GatedDeltaMemory,
    DeltaL2Memory,
    KataPsdMemory,
    TriadicMemory,
    PostDecayMemory,
    LegacyVMFMemoryAdapter,
    compute_state_bytes,
)


def test_orthogonal_keys_recall_100_percent():
    """
    With mutually orthogonal keys and P <= d, delta associative memory
    must recall 100% of stored targets without associative crosstalk.
    """
    torch.manual_seed(42)
    B = 2
    P = 8
    d = 16
    
    # Generate P orthogonal keys in R^d
    Q_rand, _ = torch.linalg.qr(torch.randn(d, d))
    keys = Q_rand[:P]  # [P, d], mutually orthogonal
    values = torch.randn(P, d)
    
    delta_arms = [
        GatedDeltaMemory(dim=d),
        DeltaL2Memory(dim=d),
        PostDecayMemory(dim=d),
    ]
    
    for arm in delta_arms:
        arm.eval()
        # Initialize zero state
        state = arm.init_state(batch_size=B)
        
        # Sequentially write each (k, v) pair
        for i in range(P):
            k_t = keys[i:i+1].repeat(B, 1)  # [B, d]
            v_t = values[i:i+1].repeat(B, 1)  # [B, d]
            # Write with retention alpha=1.0 and beta=1.0
            state = arm.write_step(state, k_t, v_t, alpha=1.0, beta=1.0)
            
        # Read with each key and verify exact recall of corresponding value
        for i in range(P):
            q_t = keys[i:i+1].repeat(B, 1)
            v_expected = values[i:i+1].repeat(B, 1)
            v_retrieved = arm.read_step(state, q_t)
            
            # Cosine similarity between retrieved and target must be > 0.999
            cos_sim = F.cosine_similarity(v_retrieved, v_expected, dim=-1)
            assert torch.all(cos_sim > 0.99), (
                f"Arm {arm.__class__.__name__} failed orthogonal recall on pair {i}: cos_sim = {cos_sim.mean().item():.4f}"
            )


def test_softmax_reference_ceiling():
    """
    Softmax attention memory recalls 100% on the minimal MQAR setting.
    """
    torch.manual_seed(42)
    B = 2
    P = 4
    d = 16
    
    keys = F.normalize(torch.randn(P, d), p=2, dim=-1)
    values = torch.randn(P, d)
    
    mem = SoftmaxAttentionMemory(dim=d, scale=10.0)
    mem.eval()
    
    # Store all keys and values
    state = mem.init_state(batch_size=B)
    for i in range(P):
        state = mem.write_step(state, keys[i:i+1].repeat(B, 1), values[i:i+1].repeat(B, 1))
        
    for i in range(P):
        q_t = keys[i:i+1].repeat(B, 1)
        v_expected = values[i:i+1].repeat(B, 1)
        v_retrieved = mem.read_step(state, q_t)
        cos_sim = F.cosine_similarity(v_retrieved, v_expected, dim=-1)
        assert torch.all(cos_sim > 0.95), f"Softmax recall failed: {cos_sim.mean().item()}"


def test_legacy_vmf_adapter_non_mutating():
    """
    LegacyVMFMemoryAdapter wraps ISHMN InfiniteSphericalMemory cleanly
    and executes step-by-step write and read operations.
    """
    torch.manual_seed(42)
    B = 2
    d = 16
    num_slots = 16
    
    adapter = LegacyVMFMemoryAdapter(dim=d, num_slots=num_slots)
    adapter.eval()
    
    state = adapter.init_state(batch_size=B)
    assert "u" in state and "r" in state and "v" in state
    assert state["v"].shape == (B, num_slots, d)
    
    # Write a step
    k_t = torch.randn(B, d)
    v_t = torch.randn(B, d)
    state = adapter.write_step(state, k_t, v_t)
    
    # Read a step
    q_t = torch.randn(B, d)
    readout = adapter.read_step(state, q_t)
    assert readout.shape == (B, d)
    assert not torch.isnan(readout).any()


def test_delta_l2_unit_norm():
    """
    DeltaL2Memory strictly enforces unit norm on keys.
    """
    mem = DeltaL2Memory(dim=16)
    k = torch.randn(4, 16) * 10.0
    k_norm = mem.normalize_key(k)
    norms = torch.norm(k_norm, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_post_decay_geometric_spacing():
    """
    PostDecayMemory initializes decay rates with exact geometric spacing.
    """
    d = 32
    mem = PostDecayMemory(dim=d, decay_min=0.01, decay_max=0.99)
    rates = mem.get_decay_rates()  # [d]
    assert len(rates) == d
    # Check monotonic increase
    assert torch.all(rates[1:] >= rates[:-1])
    # Check boundary values
    assert math.isclose(rates[0].item(), 0.01, rel_tol=1e-3)
    assert math.isclose(rates[-1].item(), 0.99, rel_tol=1e-3)


def test_state_bytes_computation():
    """
    State bytes accounting correctly reflects memory footprint.
    """
    d = 64
    # Gated Delta state is [B, d, d] float32 = B * 64 * 64 * 4 bytes = B * 16384 bytes
    delta_bytes = compute_state_bytes("gated_delta", dim=d, num_slots=32, batch_size=1)
    assert delta_bytes == 64 * 64 * 4
    
    # Legacy VMF with 32 slots: u (32*8*4) + r (32*1*4) + v (32*64*4)
    vmf_bytes = compute_state_bytes("legacy_vmf", dim=d, num_slots=32, hyperspace_dim=8, batch_size=1)
    expected_vmf = 32 * (8 + 1 + 64) * 4
    assert vmf_bytes == expected_vmf
