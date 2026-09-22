"""
Unit tests for Universal Cognitive Engine (domain-agnostic JEPA, ISHM, SphericalDynamicMoE, ACT).
"""

import pytest
import numpy as np
import torch
from ishmn import UniversalCognitiveEngine


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
        consensus_threshold=0.70,
        max_thinking_steps=4,
        device=device
    )


def test_uce_initialization(uce):
    assert uce.moe.num_experts >= 2
    assert uce.mem_state["u"].size(1) == 16
    assert uce.world_model.hyperspace_dim == 8


def test_uce_perceive_continuous(uce):
    x = torch.randn(4, 10, device=uce.device)
    u, r, v = uce.perceive(x, modality="continuous")
    assert u.shape == (4, 8)
    assert r.shape == (4, 1)
    assert v.shape == (4, 32)
    # Verify unit norm on sphere
    norms = torch.norm(u, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_uce_forward_deliberation(uce):
    x = torch.randn(1, 10, device=uce.device)
    c = torch.randn(1, 10, device=uce.device)
    _, _, z_x = uce.perceive(x, modality="continuous")
    _, _, z_c = uce.perceive(c, modality="continuous")

    # Deliberation
    res = uce.forward_deliberation(z_x, z_c, enable_system2=True)
    assert "z_final" in res
    assert "steps_taken" in res
    assert 1 <= res["steps_taken"] <= 4
    assert "consensus" in res
    assert res["z_final"].shape == (1, 32)


def test_uce_train_step(uce):
    x_curr = torch.randn(8, 4, device=uce.device)
    context = torch.randn(8, 2, device=uce.device)
    x_next = torch.randn(8, 4, device=uce.device)

    metrics = uce.train_step(x_curr, context, x_next, modality_x="continuous", modality_c="continuous")
    assert "total_loss" in metrics
    assert "loss_forward" in metrics
    assert "loss_causal" in metrics
    assert "loss_backward" in metrics
    assert not torch.isnan(torch.tensor(metrics["total_loss"]))


def test_uce_causal_inversion_evaluation(uce):
    x_curr = torch.randn(10, 4, device=uce.device)
    context = torch.randn(10, 2, device=uce.device)

    eval_res = uce.evaluate_causal_inversion(x_curr, context, modality="continuous")
    assert "understanding_score" in eval_res
    assert "angular_error_rad" in eval_res
    assert 0.0 <= eval_res["understanding_score"] <= 1.0


def test_uce_rsa_memory_cross_attention(uce):
    """Verify that multi-head RSA cross-attention operates cleanly across memory payloads."""
    z_state = torch.randn(2, 32, device=uce.device, requires_grad=True)
    mem_payloads = uce.mem_state["v"].repeat(2, 1, 1)

    out, attn_weights = uce.memory_cross_attn(
        x=z_state.unsqueeze(1),
        context=mem_payloads
    )
    assert out.shape == (2, 1, 32)
    assert attn_weights.shape == (2, 4, 1, mem_payloads.size(1))
    # Row sums equal 1
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones_like(attn_weights.sum(dim=-1)), atol=1e-5)

    loss = out.sum()
    loss.backward()
    assert z_state.grad is not None


def test_uce_rsa_deliberation_trajectory(uce):
    """Verify that System 2 deliberation tracks RSA attention weights over thinking steps."""
    z_x = torch.randn(1, 32, device=uce.device)
    z_c = torch.randn(1, 32, device=uce.device)

    res = uce.forward_deliberation(z_x, z_c, enable_system2=True)
    trajectory = res["thinking_trajectory"]
    assert len(trajectory) >= 1

    for step_info in trajectory:
        assert "attn_weights" in step_info
        weights = step_info["attn_weights"]
        assert weights.ndim == 4  # [B, H, T, S]
        assert np.allclose(weights.sum(axis=-1), 1.0, atol=1e-4)


def test_uce_token_sequence_rsa(uce):
    """Verify that UniversalPerceptor uses RSA with SO(2) phase winding for discrete sequences."""
    tokens = torch.randint(0, 100, (4, 12), device=uce.device)
    u, r, v = uce.perceive(tokens, modality="tokens")

    assert u.shape == (4, 8)
    assert r.shape == (4, 1)
    assert v.shape == (4, 32)

    # Unit norm invariant
    norms = torch.norm(u, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_uce_stream_automaton(uce):
    """Verify that solve_stream_automaton and train_stream_automaton_step execute cleanly."""
    stream_inp = torch.randint(0, 10, (2, 5, 2), device=uce.device)
    res = uce.solve_stream_automaton(stream_inp, max_loops=3)
    assert res["predicted_tokens"].shape == (2, 5)
    assert res["logits"].shape == (2, 5, uce.vocab_size)

    targets = torch.randint(0, 10, (2, 5), device=uce.device)
    train_res = uce.train_stream_automaton_step(stream_inp, targets, max_loops=3)
    assert "loss" in train_res
    assert "token_accuracy" in train_res
    assert "exact_sequence_match" in train_res


def test_uce_sequence_autoregressive(uce):
    """Verify that solve_sequence_autoregressive generates tokens step-by-step."""
    prompt = torch.randint(0, 10, (2, 4), device=uce.device)
    res = uce.solve_sequence_autoregressive(prompt, max_gen_tokens=3, max_loops=2)
    assert res["generated_tokens"].shape == (2, 3)
    assert res["full_sequence"].shape == (2, 7)


def test_uce_predictive_coding_perceive_and_infer(uce):
    """Verify that perceive_and_infer executes hierarchical state relaxation and reduces free energy."""
    data = torch.randn(2, 32, device=uce.device)
    res = uce.perceive_and_infer(data, modality="continuous", num_inference_steps=5)

    assert "u_relaxed" in res
    assert "r_relaxed" in res
    assert "continuous_out" in res
    assert "symbolic_logits" in res
    assert len(res["relaxed_states"]) == 3
    assert res["final_free_energy"] <= res["initial_free_energy"]


def test_uce_predict_trajectory(uce):
    """Verify continuous manifold trajectory rollout."""
    init_state = torch.randn(2, uce.d_model, device=uce.device)
    rollout = uce.predict_trajectory(init_state, steps=5)

    assert len(rollout) == 5
    for item in rollout:
        assert "z" in item
        assert "u" in item
        assert "r" in item
        assert "continuous_pred" in item


def test_uce_abductive_sensemaking(uce):
    """Verify that UCE executes abductive sense-making, evaluates surprise, and logs episodic traces."""
    x_familiar = torch.randn(1, 10, device=uce.device) * 0.5
    
    # 1. Sense make with prior expectation
    prior_z = torch.randn(1, uce.d_model, device=uce.device)
    res_familiar = uce.sense_make(x_familiar, prior_z=prior_z, surprise_threshold=0.5, domain="physics")
    
    assert "surprise" in res_familiar
    assert "coherence_score" in res_familiar
    assert "abduced_cause" in res_familiar
    assert "recalled_episodes" in res_familiar
    assert "recalled_semantics" in res_familiar
    assert res_familiar["latent_state"].shape == (1, uce.d_model)
    assert 0.0 <= res_familiar["coherence_score"] <= 1.0

    # 2. Test high surprise trigger (novel / contradictory input)
    x_surprising = torch.randn(1, 10, device=uce.device) * 8.0
    res_surprising = uce.sense_make(x_surprising, prior_z=prior_z, surprise_threshold=0.3, domain="anomalous")
    assert res_surprising["surprise"] > 0.0
    assert "deliberation_steps" in res_surprising
    assert len(uce.memory.episodic_memory) >= 2


def test_uce_riemannian_analogy_transfer(uce):
    """Verify that UCE solves analogies A:B :: C:D using geodesic parallel transport."""
    a = torch.randn(1, 10, device=uce.device)
    b = torch.randn(1, 10, device=uce.device)
    c = torch.randn(1, 10, device=uce.device)

    analogy_res = uce.solve_analogy(a, b, c, modality="continuous")
    assert "u_d" in analogy_res
    assert "distance_source" in analogy_res
    assert "distance_target" in analogy_res
    assert analogy_res["isometry_preserved"]

    # D must be on unit sphere
    norm_d = torch.norm(analogy_res["u_d"], dim=-1)
    assert torch.allclose(norm_d, torch.ones_like(norm_d), atol=1e-5)


def test_uce_sleep_consolidation(uce):
    """Verify that uce.sleep executes MoE barycentric merging, apoptosis, and semantic concept crystallization."""
    initial_experts = uce.moe.num_experts

    # Trigger sleep consolidation
    report = uce.sleep(prune_usage_threshold=0.05, crystallize_concepts=True)
    assert "moe_consolidation" in report
    assert "semantic_crystallization" in report
    assert "working_memory_norm" in report
    assert report["status"] == "consolidated"
    assert uce.moe.num_experts >= uce.moe.min_experts

    # Post-sleep forward pass
    x = torch.randn(2, 10, device=uce.device)
    c = torch.randn(2, 10, device=uce.device)
    _, _, z_x = uce.perceive(x, modality="continuous")
    _, _, z_c = uce.perceive(c, modality="continuous")
    fwd_res = uce.forward_deliberation(z_x, z_c, enable_system2=False)
    assert fwd_res["z_final"].shape == (2, uce.d_model)


def test_uce_epistemic_curiosity(uce):
    """Verify that Active Inference computes Free Energy reduction and ranks epistemic value."""
    cand1 = torch.zeros(1, 10, device=uce.device)
    cand2 = torch.randn(1, 10, device=uce.device) * 2.0
    cand3 = torch.randn(1, 10, device=uce.device) * 5.0

    rankings = uce.evaluate_epistemic_curiosity([cand1, cand2, cand3], modality="continuous")
    assert len(rankings) == 3
    for r in rankings:
        assert "info_gain" in r
        assert "free_energy_reduction" in r
        assert "novelty_score" in r

    # Verify descending sort
    assert rankings[0]["info_gain"] >= rankings[1]["info_gain"]
    assert rankings[1]["info_gain"] >= rankings[2]["info_gain"]


def test_uce_subword_perception(uce):
    """Verify that subword n-gram phase encoding maps text strings onto S^{D-1} x R+."""
    u, r, v = uce.perceive("spherical geometry", modality="subwords")
    assert u.shape == (1, uce.hyperspace_dim)
    assert r.shape == (1, 1)
    assert v.shape == (1, uce.d_model)
    # Unit norm invariant
    norm = torch.norm(u, dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


def test_collective_hyperspace_bus(uce):
    """Verify that CollectiveHyperspaceBus synchronizes multiple UCEs via Riemannian cross-resonance."""
    from ishmn import CollectiveHyperspaceBus
    uce2 = UniversalCognitiveEngine(
        d_model=uce.d_model,
        d_hidden=uce.d_hidden,
        hyperspace_dim=uce.hyperspace_dim,
        initial_experts=1,
        max_experts=None,
        device=uce.device
    )

    bus = CollectiveHyperspaceBus(
        hyperspace_dim=uce.hyperspace_dim,
        d_model=uce.d_model,
        device=uce.device
    )
    bus.register_agent("agent_phys", uce)
    bus.register_agent("agent_sym", uce2)

    phys_in = torch.randn(1, 10, device=uce.device)
    res = bus.solve_coupled_task(
        agent_phys_id="agent_phys",
        agent_sym_id="agent_sym",
        physical_input=phys_in,
        symbolic_input="riemannian consensus manifold",
        symbolic_modality="subwords"
    )

    assert "phys_output" in res
    assert "sym_surprise" in res
    assert "sym_coherence" in res
    assert "mean_angular_distance" in res
    assert res["mean_angular_distance"] >= 0.0
    assert res["collective_entropy"] >= 0.0




