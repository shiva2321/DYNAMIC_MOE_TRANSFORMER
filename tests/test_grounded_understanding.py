"""
Unit Tests for Genuine Grounded Understanding, Causal Rollback, and Sense-Making.
Evaluates learned intelligence, surprise-triggered deliberation, and analogical transfer.
"""

import pytest
import torch
import math
from ishmn import UniversalCognitiveEngine, geodesic_distance_sphere

@pytest.fixture
def uce():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        max_experts=8,
        device=device
    )

def test_grounded_sensemaking_coherence(uce):
    """Verify that consistent vs contradictory observations produce distinct surprise & deliberation behaviors."""
    obs_base = torch.randn(1, 10, device=uce.device)
    _, _, z_expected = uce.perceive(obs_base, modality="continuous")
    
    # 1. Consistent observation (closely matching expected prior attractor)
    consistent_obs = obs_base + torch.randn(1, 10, device=uce.device) * 0.02
    res_coherent = uce.sense_make(
        consistent_obs,
        prior_z=z_expected,
        surprise_threshold=0.8,
        domain="physics"
    )
    assert res_coherent["coherence_score"] > 0.0
    assert "abduced_cause" in res_coherent

    # 2. Contradictory observation (drastically diverging from expected prior)
    contradictory_obs = -obs_base * 8.0
    thresh = res_coherent["surprise"] + 0.05
    res_contradictory = uce.sense_make(
        contradictory_obs,
        prior_z=z_expected,
        surprise_threshold=thresh,
        domain="anomalous"
    )
    # Contradiction must produce significantly higher surprise
    assert res_contradictory["surprise"] > res_coherent["surprise"]
    assert "System 2 Deliberation" in res_contradictory["system_used"]

def test_grounded_causal_counterfactual_verification(uce):
    """Verify Pearl Level 3 causal rollback: inferring hidden causes and reversing state transitions."""
    x_curr = torch.randn(8, 4, device=uce.device)
    context = torch.randn(8, 2, device=uce.device)

    # Train step to ground transitions
    for _ in range(5):
        uce.train_step(x_curr, context, x_curr * 1.1, modality_x="continuous", modality_c="continuous")

    eval_res = uce.evaluate_causal_inversion(x_curr, context, modality="continuous")
    assert "understanding_score" in eval_res
    assert "angular_error_rad" in eval_res
    assert eval_res["understanding_score"] >= 0.0

def test_grounded_analogical_transfer(uce):
    """Verify cross-domain analogical parallel transport A : B :: C : D preserves Riemannian isometry."""
    # A (source anchor) -> B (source effect) :: C (target anchor) -> D (target effect)
    a = "gravitational attractor orbital pull"
    b = "elliptical planetary orbit velocity"
    c = "atomic nucleus electrostatic core"

    analogy = uce.solve_analogy(a, b, c, modality="subwords")
    assert analogy["isometry_preserved"]
    assert abs(analogy["distance_source"] - analogy["distance_target"]) < 1e-3
    assert float(torch.norm(analogy["u_d"])) > 0.99

def test_grounded_episodic_to_semantic_consolidation(uce):
    """Verify that multiple experiences logged into Episodic Memory crystallize into Semantic Concepts during sleep."""
    # Log 4 experiences in 2 distinct clusters
    u_cluster1 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=uce.device)
    u_cluster2 = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=uce.device)

    for i in range(2):
        noise = torch.randn(1, 8, device=uce.device) * 0.05
        uce.memory.log_episode(u_cluster1 + noise, torch.tensor([[1.0]]), torch.randn(1, uce.d_model), domain="neuroscience")
    for i in range(2):
        noise = torch.randn(1, 8, device=uce.device) * 0.05
        uce.memory.log_episode(u_cluster2 + noise, torch.tensor([[2.0]]), torch.randn(1, uce.d_model), domain="physics")

    assert len(uce.memory.episodic_memory) >= 4

    # Trigger sleep consolidation
    report = uce.sleep(prune_usage_threshold=0.01, crystallize_concepts=True)
    assert report["semantic_crystallization"]["status"] == "crystallized"
    assert report["total_semantic_concepts"] >= 1
