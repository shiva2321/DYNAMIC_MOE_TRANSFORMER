"""
Integration tests for Human-Inspired Developmental Learning Architecture (HIDLA)
integrated across UniversalCognitiveEngine, LatentManifoldSandwich, and Brain Persistence.
"""

import os
import pytest
import torch
import torch.nn as nn

from ishmn import (
    UniversalCognitiveEngine,
    DevelopmentalStage,
    ZoneOfProximalDevelopment,
    ActiveCausalPlaySimulator,
    OperationalReversibilityAudit,
    DevelopmentalManifoldLearner
)
from ishmn.grounded_hybrid_engine import LatentManifoldSandwich


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


def test_uce_developmental_engine_initialization(uce_engine):
    """Verify developmental learner is natively mounted on UniversalCognitiveEngine."""
    assert hasattr(uce_engine, "developmental_learner")
    assert isinstance(uce_engine.developmental_learner, DevelopmentalManifoldLearner)

    profile = uce_engine.get_developmental_profile()
    assert "current_stage" in profile
    assert profile["current_stage"] == DevelopmentalStage.STAGE_0_SENSORIMOTOR.value
    assert "developmental_ticks" in profile
    assert profile["developmental_ticks"] == 0
    assert "zpd_thresholds" in profile
    assert profile["zpd_thresholds"]["tau_lower"] > 0
    assert profile["zpd_thresholds"]["tau_upper"] > profile["zpd_thresholds"]["tau_lower"]


def test_uce_developmental_step_execution(uce_engine):
    """Verify developmental_step operates smoothly across continuous and subword modalities."""
    # 1. Continuous vector observation
    obs_vec = torch.randn(1, 32)
    res_vec = uce_engine.developmental_step(obs_vec, modality="continuous", domain="physics")
    assert res_vec["stage"] == DevelopmentalStage.STAGE_0_SENSORIMOTOR.value
    assert "manifold_surprise" in res_vec
    assert "causal_error" in res_vec
    assert "action_taken" in res_vec
    assert res_vec["action_taken"] == "sensorimotor_calibrated"

    # 2. Subwords / Symbolic observation
    res_sym = uce_engine.developmental_step("spherical hyperspace manifold", modality="subwords", domain="geometry")
    assert res_sym["stage"] == DevelopmentalStage.STAGE_0_SENSORIMOTOR.value
    assert res_sym["action_taken"] == "sensorimotor_calibrated"
    assert uce_engine.get_developmental_profile()["developmental_ticks"] == 2


def test_ontogenetic_stage_progression(uce_engine):
    """Verify full developmental progression across all 5 ontogenetic stages."""
    # Stage 0 -> Stage 1 (Active Causal Play)
    advanced, msg = uce_engine.advance_developmental_stage(force=True, reason="sensorimotor milestone met")
    assert advanced is True
    assert uce_engine.developmental_learner.stage == DevelopmentalStage.STAGE_1_CAUSAL_PLAY

    res_play = uce_engine.developmental_step("curiosity experiment", modality="subwords")
    assert res_play["action_taken"] == "causal_play_executed"
    assert "max_info_gain" in res_play

    # Stage 1 -> Stage 2 (Scaffolded ZPD)
    advanced, msg = uce_engine.advance_developmental_stage(force=True, reason="causal affordances mapped")
    assert advanced is True
    assert uce_engine.developmental_learner.stage == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD

    res_zpd = uce_engine.developmental_step("scaffolded learning step", modality="subwords")
    assert "zpd_verdict" in res_zpd
    assert res_zpd["zpd_verdict"] in ("mastered", "zpd_fertile", "chaotic")

    # Stage 2 -> Stage 3 (Concrete Operations & Conservation)
    advanced, msg = uce_engine.advance_developmental_stage(force=True, reason="fast-mapping completed")
    assert advanced is True
    assert uce_engine.developmental_learner.stage == DevelopmentalStage.STAGE_3_CONCRETE_OPERATIONS

    res_conc = uce_engine.developmental_step(torch.randn(1, 32), modality="continuous")
    assert res_conc["action_taken"] == "concrete_operation_verified"
    assert "conserved" in res_conc

    # Stage 3 -> Stage 4 (Formal Metacognitive Deliberation)
    advanced, msg = uce_engine.advance_developmental_stage(force=True, reason="operational reversibility verified")
    assert advanced is True
    assert uce_engine.developmental_learner.stage == DevelopmentalStage.STAGE_4_FORMAL_DELIBERATION

    res_form = uce_engine.developmental_step("formal mathematical logic", modality="subwords")
    assert res_form["stage"] == DevelopmentalStage.STAGE_4_FORMAL_DELIBERATION.value


def test_brain_save_load_developmental_profile_roundtrip(uce_engine, tmp_path):
    """Verify complete serialization and restoration of developmental profile during brain checkpointing."""
    # Advance to Stage 2 and execute steps to log history
    uce_engine.advance_developmental_stage(force=True, reason="testing save/load")
    uce_engine.advance_developmental_stage(force=True, reason="transition to stage 2")
    assert uce_engine.developmental_learner.stage == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD

    uce_engine.developmental_step("learning token 1", modality="subwords")
    uce_engine.developmental_step("learning token 2", modality="subwords")

    orig_profile = uce_engine.get_developmental_profile()
    checkpoint_path = str(tmp_path / "test_dev_brain.pt")
    uce_engine.save_brain(checkpoint_path)

    # Load into fresh instance
    loaded_uce = UniversalCognitiveEngine.load_brain(checkpoint_path, device=torch.device("cpu"))
    loaded_profile = loaded_uce.get_developmental_profile()

    assert loaded_profile["current_stage"] == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD.value
    assert loaded_uce.developmental_learner.stage == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD
    assert loaded_profile["developmental_ticks"] == orig_profile["developmental_ticks"]
    assert len(loaded_profile["stage_transition_history"]) == len(orig_profile["stage_transition_history"])


def test_latent_manifold_sandwich_developmental_scaffolding(uce_engine):
    """Verify LatentManifoldSandwich utilizes developmental ZPD routing and protection gating."""
    sandwich = LatentManifoldSandwich(
        uce=uce_engine,
        hidden_size=64,
        d_model=32,
        use_developmental_scaffolding=True
    )

    # Simulate hidden state output from transformer layer: [B=1, L=4, H=64]
    mock_hidden = torch.randn(1, 4, 64)
    out = sandwich.hook_fn(None, None, mock_hidden)

    # Verify tensor shape preserved
    assert out.shape == mock_hidden.shape

    # Verify telemetry was recorded with developmental metadata
    assert len(sandwich.deliberation_telemetry) >= 1
    telem = sandwich.deliberation_telemetry[-1]
    assert "zpd_verdict" in telem
    assert telem["zpd_verdict"] in ("mastered", "zpd_fertile", "chaotic")
    assert "system" in telem
    assert isinstance(telem["coherence"], float)
    assert telem["coherence"] > 0.0


def test_cognitive_protection_gating_behavior(uce_engine):
    """Verify Cognitive Protection Gating prevents catastrophic drift when stimulus is chaotic."""
    uce_engine.advance_developmental_stage(force=True, reason="scaffold test")
    uce_engine.advance_developmental_stage(force=True, reason="to stage 2")
    assert uce_engine.developmental_learner.stage == DevelopmentalStage.STAGE_2_SCAFFOLDED_ZPD

    # Set ZPD upper threshold very low to force any stimulus to be chaotic
    uce_engine.developmental_learner.zpd.tau_upper = 0.01

    sandwich = LatentManifoldSandwich(
        uce=uce_engine,
        hidden_size=64,
        d_model=32,
        use_developmental_scaffolding=True
    )

    mock_hidden = torch.randn(1, 2, 64)
    out = sandwich.hook_fn(None, None, mock_hidden)

    assert len(sandwich.deliberation_telemetry) >= 1
    telem = sandwich.deliberation_telemetry[-1]
    assert telem["zpd_verdict"] == "chaotic"
    assert "Cognitive Protection Gating" in telem["system"]
    assert telem["step_count"] == 0
