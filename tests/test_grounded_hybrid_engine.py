"""
Unit & Integration Tests for Grounded Hybrid Engine & SLM-Manifold Adapter.
"""

import pytest
import torch
import torch.nn as nn

from ishmn.slm_adapter import SLMManifoldAdapter
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.grounded_hybrid_engine import LatentManifoldSandwich, DeliberativeLogitsProcessor


def test_slm_manifold_adapter_dimensions_and_steering():
    device = torch.device("cpu")
    adapter = SLMManifoldAdapter(
        hidden_size=1536,
        d_model=32,
        hyperspace_dim=8,
        mlp_dim=256,
        alpha=0.25,
        device=device
    )

    B, L, H = 2, 5, 1536
    h_mock = torch.randn(B, L, H, device=device) * 2000.0 # realistic LLM hidden norm

    # Test encoding
    z = adapter.encode(h_mock)
    assert z.shape == (B, L, 32)
    assert not torch.isnan(z).any()

    # Test decoding
    v = adapter.decode(z)
    assert v.shape == (B, L, 1536)
    assert not torch.isnan(v).any()

    # Test norm-scaled steering
    z_settled = torch.randn(B, 1, 32, device=device)
    h_ref = h_mock[:, -1:, :]
    delta_h = adapter.compute_steering_vector(z_settled, h_ref, gain=0.20)

    assert delta_h.shape == (B, 1, 1536)
    delta_norm = torch.norm(delta_h.float(), dim=-1)
    ref_norm = torch.norm(h_ref.float(), dim=-1)

    # Relative magnitude should match gain (0.20) within float precision
    rel_mag = (delta_norm / ref_norm).mean().item()
    assert abs(rel_mag - 0.20) < 1e-4


def test_deliberative_logits_processor_active_and_shield():
    device = torch.device("cpu")
    uce = UniversalCognitiveEngine(d_model=32, hyperspace_dim=8, device=device)
    sandwich = LatentManifoldSandwich(uce=uce, hidden_size=64, d_model=32, adapter_path=None, device=device)

    # Mock tokenizer
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            if "theta" in text.lower():
                return [101]
            if "spatial" in text.lower():
                return [202]
            return [999]

    tok = MockTokenizer()
    processor = DeliberativeLogitsProcessor(sandwich, tok, boost_weight=4.0)

    vocab_size = 1000
    scores = torch.zeros(1, vocab_size)

    # Case 1: Sandwich inactive
    sandwich.is_active = False
    out_inactive = processor(torch.tensor([[1]]), scores.clone())
    assert torch.equal(out_inactive, scores)

    # Case 2: Active with settled concepts
    sandwich.is_active = True
    sandwich.last_system_used = "System 2 (ZPD Deliberation)"
    sandwich.last_settled_concepts = [
        {"word": "theta", "semantic_similarity": 0.90},
        {"word": "spatial", "semantic_similarity": 0.85}
    ]
    out_active = processor(torch.tensor([[1]]), scores.clone())
    assert out_active[0, 101] > 0.0
    assert out_active[0, 202] > 0.0
    assert out_active[0, 500] == 0.0

    # Case 3: Cognitive Protection Shield active -> should suppress boost
    sandwich.last_system_used = "Cognitive Protection Gating (ZPD Chaotic Shield)"
    scores_shield = torch.zeros(1, vocab_size)
    out_shield = processor(torch.tensor([[1]]), scores_shield.clone())
    assert out_shield[0, 101] == 0.0
    assert out_shield[0, 202] == 0.0


def test_transient_ttt_offset_zero_forgetting():
    """Verify test-time adaptation respects geodesic trust region and leaves weights untouched."""
    from ishmn.geometry import geodesic_distance_sphere

    device = torch.device("cpu")
    uce = UniversalCognitiveEngine(d_model=32, hyperspace_dim=8, device=device)
    sandwich = LatentManifoldSandwich(uce=uce, hidden_size=64, d_model=32, adapter_path=None, device=device)

    # Record baseline weights
    orig_w_fwd = uce.world_model.w_fwd[0].weight.clone()

    z = torch.randn(1, 32, device=device)
    c = torch.randn(1, 32, device=device)

    z_adapted = sandwich.compute_transient_ttt_offset(z, c, steps=3, lr=0.05, trust_radius=0.20)
    assert z_adapted.shape == (1, 32)

    # Trust region check: d_S <= 0.20 rad
    u0, _ = uce.world_model.project_manifold(z)
    u_adapt, _ = uce.world_model.project_manifold(z_adapted)
    drift = geodesic_distance_sphere(u0, u_adapt).item()
    assert drift <= 0.2001, f"Geodesic trust region violated: {drift:.4f} rad > 0.20 rad"

    # Zero-forgetting check: weights completely unaltered
    assert torch.equal(orig_w_fwd, uce.world_model.w_fwd[0].weight), "Permanent weights were modified!"


def test_pondernet_probabilistic_halting():
    """Verify PonderNet cumulative hazard halting and trajectory telemetry."""
    from ishmn.universal_world_model import UniversalWorldModel

    device = torch.device("cpu")
    wm = UniversalWorldModel(d_model=32, hyperspace_dim=8, device=device)

    z = torch.randn(1, 32, device=device)
    c = torch.randn(1, 32, device=device)
    delib = wm.deliberate_and_halting_loop(z_init=z, context=c)

    assert "cum_halting_prob" in delib
    assert 0.0 <= delib["cum_halting_prob"] <= 1.0
    assert len(delib["thinking_trajectory"]) >= 1
    assert "cum_halting_prob" in delib["thinking_trajectory"][0]

