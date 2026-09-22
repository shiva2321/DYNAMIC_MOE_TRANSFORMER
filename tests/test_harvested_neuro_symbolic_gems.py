"""
Unit Tests for Harvested Neuro-Symbolic General Intelligence Primitives.
Validates:
1. Chunked Gated DeltaNet fast-weight memory.
2. Dentate Gyrus 2% k-WTA pattern separation and Toroidal Grid Encoders.
3. Tolman-Eichenbaum Cognitive Map, Relational Reflection Coefficient, and Successor Representation.
4. Continuous Modern Hopfield Network attractor denoising.
5. Phasor superposition Global Workspace Bus and GWT arbitration.
6. Self-Organized Criticality (SOC) homeostatic controller.
"""

import pytest
import torch
import torch.nn.functional as F

from ishmn.neuro import (
    ChunkedGatedDeltaRuleLayer,
    DentateGyrusPatternSeparator,
    ToroidalGridEncoder,
    CognitiveMapGraph,
    ModernHopfieldNetwork,
    HyperspaceGlobalBus,
    PredictiveGlobalWorkspace,
    GWTTicket,
    CriticalityController
)


def test_chunked_gated_delta_net():
    B, L, d_in, d_k, d_v = 2, 32, 16, 8, 8
    layer = ChunkedGatedDeltaRuleLayer(d_in=d_in, d_k=d_k, d_v=d_v, chunk_size=16)

    x = torch.randn(B, L, d_in, requires_grad=True)
    out, final_state = layer(x)

    assert out.shape == (B, L, d_v)
    assert final_state.shape == (B, d_v, d_k)
    assert not torch.isnan(out).any()
    assert not torch.isnan(final_state).any()

    # Verify clean finite gradient propagation
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_dentate_gyrus_pattern_separation():
    d_in, d_sparse = 32, 1024
    dg = DentateGyrusPatternSeparator(d_in=d_in, d_sparse=d_sparse, sparsity_ratio=0.02)

    x1 = torch.randn(2, d_in)
    x2 = x1 + 0.1 * torch.randn(2, d_in)  # Overlapping inputs

    s1 = dg(x1)
    s2 = dg(x2)

    # Check exact top-k active sparsity (2% of 1024 = 20 active neurons)
    expected_active = max(1, int(d_sparse * 0.02))
    non_zeros_1 = (s1[0] > 0).sum().item()
    assert non_zeros_1 == expected_active

    # Check that pattern separation reduces similarity compared to raw inputs
    raw_sim = F.cosine_similarity(x1, x2, dim=-1).mean().item()
    sparse_sim = F.cosine_similarity(s1, s2, dim=-1).mean().item()
    assert sparse_sim < raw_sim


def test_toroidal_grid_encoder():
    d_in, d_out = 8, 128
    encoder = ToroidalGridEncoder(d_in=d_in, d_out=d_out, num_scales=4)

    x = torch.randn(4, d_in)
    grid_rep = encoder(x)

    assert grid_rep.shape == (4, d_out)
    norms = torch.norm(grid_rep, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_cognitive_map_successor_representation():
    cog_map = CognitiveMapGraph(d_model=16, gamma=0.85)

    # Build relational chain: A -> B -> C -> D
    cog_map.add_transition("sun", "light")
    cog_map.add_transition("light", "photosynthesis")
    cog_map.add_transition("photosynthesis", "glucose")

    assert len(cog_map) == 4
    assert len(cog_map.edges) == 3

    # Closed-form Successor Representation computation (M = (I - gamma * T)^{-1})
    M = cog_map.compute_successor_representation()
    assert M.shape == (4, 4)

    # Simulate multi-step mental rollout from "sun" in 0.15ms
    rollout = cog_map.simulate_future_rollout("sun", top_k=3)
    assert len(rollout) > 0
    downstream_names = [name for name, _ in rollout]
    assert "light" in downstream_names

    # Check relational reflection coefficient
    gamma_known = cog_map.reflection_coefficient("sun", "light")
    assert gamma_known < 0.90  # Known operator has low reflection coefficient (impedance match)


def test_modern_hopfield_attractor_denoising():
    mhn = ModernHopfieldNetwork(d_model=16, beta_init=1.0, beta_max=8.0, n_iters=5)

    v_cat = torch.randn(16)
    v_dog = torch.randn(16)
    mhn.store("cat", v_cat)
    mhn.store("dog", v_dog)

    assert len(mhn) == 2

    # Query with 25% noise corrupted cat vector
    v_noisy = v_cat + 0.25 * torch.randn(16)
    res = mhn.snap(v_noisy)

    assert res.best_concept == "cat"
    assert res.confidence > 0.80
    assert res.converged_vec.shape == (16,)


def test_hyperspace_global_bus_and_gwt():
    B, K, d_model, d_hyper = 2, 4, 16, 32
    bus = HyperspaceGlobalBus(d_model=d_model, d_hyper=d_hyper)

    outputs = torch.randn(B, K, d_model)
    keys = F.normalize(torch.randn(B, K, d_hyper), p=2, dim=-1)

    context = bus.broadcast_and_listen(outputs, keys)
    assert context.shape == (B, K, d_model)
    assert not torch.isnan(context).any()

    # Test Predictive GWT Ticket arbitration
    gwt = PredictiveGlobalWorkspace()
    ticket_a = GWTTicket(processor_id="logic", interpretation=torch.randn(16), resonance=0.85, confidence=0.90)
    ticket_b = GWTTicket(processor_id="memory", interpretation=torch.randn(16), resonance=0.40, confidence=0.70)

    winner = gwt.arbitrate([ticket_a, ticket_b])
    assert winner is not None
    assert winner.processor_id == "logic"


def test_criticality_controller():
    controller = CriticalityController(target_event_rate=0.05, window=10, gain=0.10)

    # Initially at default
    init_val = controller.current_value

    # Simulate excessive reset/spawn events (rate = 1.0 > 0.05)
    for _ in range(12):
        controller.step(was_event=True)

    # Controller should have decreased threshold to restore criticality
    assert controller.current_value < init_val
    assert controller.branching_ratio > 1.0
