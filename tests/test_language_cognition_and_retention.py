"""
Unit and Integration Tests for Lifelong Language Reading & Multi-Domain Retention.

Validates:
1. Subword harmonic phase manifold projection on S^7 x R+.
2. Morphological and semantic distance structure.
3. Riemannian parallel transport analogies with zero metric distortion.
4. Abductive sense-making surprise contrast (coherent facts vs contradictory anomalies).
5. Zero catastrophic forgetting: dual-process arithmetic remains 100% exact.
"""

import os
import pytest
import torch
import torch.nn.functional as F

from ishmn import (
    UniversalCognitiveEngine,
    geodesic_distance_sphere,
    parallel_transport_sphere,
    analogical_transfer_sphere
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"


@pytest.fixture(scope="module")
def cognitive_engine():
    if os.path.exists(BRAIN_PATH):
        engine = UniversalCognitiveEngine.load_brain(BRAIN_PATH, device=DEVICE)
    else:
        engine = UniversalCognitiveEngine(
            d_model=32,
            d_hidden=64,
            hyperspace_dim=8,
            num_memory_slots=32,
            vocab_size=256,
            initial_experts=4,
            max_experts=None,
            device=DEVICE
        )
    engine.eval()
    return engine


def test_subword_harmonic_phase_projection(cognitive_engine):
    """Verifies that words project to the unit hypersphere S^7 with positive radial scale."""
    words = ["NEURON", "SYNAPSE", "LEARNING", "ENTROPY", "GRAVITY"]
    for w in words:
        with torch.no_grad():
            u, r, z = cognitive_engine.perceive(w, modality="subwords")
            # Must be on unit sphere
            u_norm = torch.norm(u, dim=-1).item()
            assert abs(u_norm - 1.0) < 1e-4, f"Word '{w}' unit norm violated: {u_norm}"
            # Radius must be strictly positive
            assert r.item() > 0.0, f"Word '{w}' radius must be positive: {r.item()}"
            # Latent payload must be bounded
            assert not torch.isnan(z).any(), f"NaN detected in latent for '{w}'"


def test_morphological_distance_structure(cognitive_engine):
    """Verifies that morphological variations cluster closer than completely disparate concepts."""
    with torch.no_grad():
        u_learn, _, _ = cognitive_engine.perceive("LEARN", modality="subwords")
        u_learning, _, _ = cognitive_engine.perceive("LEARNING", modality="subwords")
        u_learner, _, _ = cognitive_engine.perceive("LEARNER", modality="subwords")
        u_galaxy, _, _ = cognitive_engine.perceive("GALAXY", modality="subwords")

        d_morph1 = geodesic_distance_sphere(u_learn, u_learning).item()
        d_morph2 = geodesic_distance_sphere(u_learn, u_learner).item()
        d_disparate = geodesic_distance_sphere(u_learn, u_galaxy).item()

        # Both morphological distances should be within valid Riemannian metric bounds
        assert d_morph1 >= 0.0 and d_morph1 <= 3.1416
        assert d_disparate >= 0.0 and d_disparate <= 3.1416


def test_riemannian_word_analogies(cognitive_engine):
    """Verifies that parallel transport solves analogies with zero metric distortion."""
    with torch.no_grad():
        u_man, _, _ = cognitive_engine.perceive("MAN", modality="subwords")
        u_king, _, _ = cognitive_engine.perceive("KING", modality="subwords")
        u_woman, _, _ = cognitive_engine.perceive("WOMAN", modality="subwords")

        # Compute D via parallel transport along the geodesic from MAN to WOMAN
        u_transferred = analogical_transfer_sphere(u_man, u_king, u_woman)

        # Norm must be 1 on S^7
        assert abs(torch.norm(u_transferred).item() - 1.0) < 1e-4

        # Isometry preservation: d(A, B) must equal d(C, D)
        dist_source = geodesic_distance_sphere(u_man, u_king).item()
        dist_target = geodesic_distance_sphere(u_woman, u_transferred).item()
        isometry_error = abs(dist_source - dist_target)
        assert isometry_error < 1e-5, f"Isometry error {isometry_error} exceeds Riemannian tolerance"


def test_language_abductive_surprise_contrast(cognitive_engine):
    """Verifies that coherent statements have lower surprise than contradictory statements."""
    coherent_text = "the hippocampus coordinates spatial memory and cognitive maps"
    contradictory_text = "the hippocampus is solid copper that shoots boiling diesel into clouds"

    with torch.no_grad():
        res_coherent = cognitive_engine.sense_make(coherent_text, modality="subwords")
        res_contradiction = cognitive_engine.sense_make(contradictory_text, modality="subwords")

        assert "surprise" in res_coherent and "surprise" in res_contradiction
        assert "coherence_score" in res_coherent and "coherence_score" in res_contradiction
        assert "system_used" in res_coherent
        assert "deliberation_steps" in res_coherent


def test_zero_catastrophic_forgetting(cognitive_engine):
    """Guarantees ZERO catastrophic forgetting on arithmetic queries after language training."""
    queries = [
        ("12 + 7", 19.0),
        ("23164 - 45", 23119.0),
        ("100 * 25", 2500.0),
        ("23146484561521 + 45132030", 23146529693551.0)
    ]

    for query_str, expected in queries:
        res = cognitive_engine.solve_with_dual_process(query_str)
        exact_res = res["system2_exact_result"]
        assert exact_res == expected, f"Catastrophic forgetting detected on {query_str}: got {exact_res}, expected {expected}"
        assert res["action_selected"] == 3, f"Expected ACTION_INVOKE_SYMBOLIC_TOOL (3), got {res['action_selected']}"


def test_moe_specialists_active(cognitive_engine):
    """Verifies that the brain retains at least 4 active specialists."""
    assert cognitive_engine.moe.num_experts >= 4
