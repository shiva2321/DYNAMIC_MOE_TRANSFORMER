"""
Unit tests for Human-Inspired Developmental Learning Architecture (HIDLA).
Verifies ontogenetic stages, ZPD filtering, active causal play, and operational reversibility.
"""

import pytest
import torch

from ishmn import (
    UniversalCognitiveEngine,
    DevelopmentalStage,
    ZoneOfProximalDevelopment,
    ActiveCausalPlaySimulator,
    OperationalReversibilityAudit,
    DevelopmentalManifoldLearner
)


@pytest.fixture
def uce_engine():
    device = torch.device("cpu")
    engine = UniversalCognitiveEngine(
        d_model=32,
        hyperspace_dim=8,
        device=device
    )
    engine.eval()
    return engine


def test_zpd_evaluation_and_adaptation():
    """Verify Vygotskian Zone of Proximal Development gating."""
    zpd = ZoneOfProximalDevelopment(tau_lower=0.20, tau_upper=0.80)

    # 1. Mastered / Too Easy
    verdict, should_learn = zpd.evaluate(0.10)
    assert verdict == "mastered"
    assert should_learn is False

    # 2. Fertile ZPD
    verdict, should_learn = zpd.evaluate(0.50)
    assert verdict == "zpd_fertile"
    assert should_learn is True

    # 3. Chaotic / Out of ZPD
    verdict, should_learn = zpd.evaluate(0.95)
    assert verdict == "chaotic"
    assert should_learn is False

    # 4. Adaptive expansion
    zpd.adapt_window(mastery_rate=0.75)
    assert zpd.tau_upper >= 0.80


def test_active_causal_play_simulation(uce_engine):
    """Verify Alison Gopnik's 'Scientist in the Crib' active play simulator."""
    simulator = ActiveCausalPlaySimulator(uce_engine)
    z_state = torch.randn(1, uce_engine.d_model)

    candidates = [
        torch.randn(1, uce_engine.d_model) * 0.1,
        torch.randn(1, uce_engine.d_model) * 0.5,
        torch.zeros(1, uce_engine.d_model)
    ]

    res = simulator.simulate_interventions(z_state, candidates)
    assert "selected_action" in res
    assert res["selected_action"] is not None
    assert len(res["simulations"]) == 3
    for s in res["simulations"]:
        assert "geodesic_shift" in s
        assert "causal_error" in s
        assert "info_gain" in s
        assert torch.isfinite(s["z_predicted"]).all()


def test_operational_reversibility_audit(uce_engine):
    """Verify Piagetian Operational Reversibility of thought."""
    auditor = OperationalReversibilityAudit(uce_engine)
    z_state = torch.randn(1, uce_engine.d_model)

    audit = auditor.audit_reversibility(z_state)
    assert "angular_error" in audit
    assert "radial_error" in audit
    assert "total_reversibility_error" in audit
    assert isinstance(audit["is_conserved"], bool)
    assert audit["total_reversibility_error"] >= 0.0


def test_stage_0_sensorimotor_intake(uce_engine):
    """Verify Stage 0 sensorimotor coordinate calibration."""
    learner = DevelopmentalManifoldLearner(
        uce_engine,
        initial_stage=DevelopmentalStage.STAGE_0_SENSORIMOTOR
    )
    obs = torch.randn(1, 32)
    res = learner.step_intake(obs, modality="continuous")

    assert res["stage"] == DevelopmentalStage.STAGE_0_SENSORIMOTOR.value
    assert res["action_taken"] == "sensorimotor_calibrated"
    assert "System 1" in res["system_used"]


def test_stage_1_causal_play_intake(uce_engine):
    """Verify Stage 1 active intervention play and episodic buffer logging."""
    learner = DevelopmentalManifoldLearner(
        uce_engine,
        initial_stage=DevelopmentalStage.STAGE_1_CAUSAL_PLAY
    )
    initial_episodes = len(uce_engine.memory.episodic_memory)

    obs = torch.randn(1, 32)
    res = learner.step_intake(obs, modality="continuous", domain="physics_play")

    assert res["stage"] == DevelopmentalStage.STAGE_1_CAUSAL_PLAY.value
    assert res["action_taken"] == "causal_play_executed"
    assert "max_info_gain" in res
    # Should have logged an episodic play trace
    assert len(uce_engine.memory.episodic_memory) == initial_episodes + 1


def test_stage_2_zpd_scaffolded_intake(uce_engine):
    """Verify Stage 2 Zone of Proximal Development filtering."""
    learner = DevelopmentalManifoldLearner(
        uce_engine,
        initial_stage=DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD
    )
    obs = torch.randn(1, 32)
    res = learner.step_intake(obs, modality="continuous", domain="language_learning")

    assert res["stage"] == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD.value
    assert "zpd_verdict" in res
    assert res["action_taken"] in ["intuitive_bypass", "zpd_assimilated", "chaotic_deferred"]


def test_stage_3_concrete_operations_intake(uce_engine):
    """Verify Stage 3 operational reversibility & conservation verification."""
    learner = DevelopmentalManifoldLearner(
        uce_engine,
        initial_stage=DevelopmentalStage.STAGE_3_CONCRETE_OPERATIONS
    )
    obs = torch.randn(1, 32)
    res = learner.step_intake(obs, modality="continuous", domain="relational_math")

    assert res["stage"] == DevelopmentalStage.STAGE_3_CONCRETE_OPERATIONS.value
    assert res["action_taken"] == "concrete_operation_verified"
    assert "conserved" in res


def test_stage_4_formal_deliberation_intake(uce_engine):
    """Verify Stage 4 formal metacognitive reflection and knowledge boundary detection."""
    learner = DevelopmentalManifoldLearner(
        uce_engine,
        initial_stage=DevelopmentalStage.STAGE_4_FORMAL_DELIBERATION
    )
    obs = torch.randn(1, 32)
    res = learner.step_intake(obs, modality="continuous", domain="abstract_counterfactual")

    assert res["stage"] == DevelopmentalStage.STAGE_4_FORMAL_DELIBERATION.value
    assert res["action_taken"] == "formal_reasoning_evaluated"
    assert res["knowledge_boundary"] in ["known_territory", "metacognitive_unknown_boundary"]


def test_developmental_milestone_progression(uce_engine):
    """Verify automatic ontogenetic milestone checking and stage advancement."""
    learner = DevelopmentalManifoldLearner(
        uce_engine,
        initial_stage=DevelopmentalStage.STAGE_0_SENSORIMOTOR
    )

    # Ingest several stable sensorimotor steps to reach milestone
    for _ in range(6):
        obs = torch.randn(1, 32) * 0.1
        learner.step_intake(obs, modality="continuous")

    advanced, new_stage = learner.check_and_advance_milestones()
    if advanced:
        assert new_stage == DevelopmentalStage.STAGE_1_CAUSAL_PLAY.value
        assert learner.stage == DevelopmentalStage.STAGE_1_CAUSAL_PLAY


def test_developmental_profile_and_sleep(uce_engine):
    """Verify comprehensive developmental profile generation and sleep consolidation."""
    learner = DevelopmentalManifoldLearner(uce_engine)
    profile = learner.get_developmental_profile()

    assert "current_stage" in profile
    assert "zpd_telemetry" in profile
    assert "stage_transition_history" in profile

    # Perform nocturnal sleep
    sleep_res = learner.nocturnal_sleep(rem_steps=2)
    assert sleep_res["status"] == "developmental_sleep_completed"
