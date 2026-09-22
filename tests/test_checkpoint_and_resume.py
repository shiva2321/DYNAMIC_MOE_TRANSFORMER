"""
Unit Tests for Model Sizing Profiles, Time Budget Control, and Checkpoint Auto-Resume.
"""

import os
import shutil
import pytest
import torch
import torch.nn as nn

from ishmn.training_engine import (
    MODEL_PROFILES,
    TimeBudgetController,
    CheckpointManager,
    GracefulInterruptHandler
)
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.muon import HybridMuonAdamW


TEST_CKPT_DIR = "artifacts/test_checkpoints"


@pytest.fixture(autouse=True)
def clean_test_dir():
    os.makedirs(TEST_CKPT_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_CKPT_DIR):
        shutil.rmtree(TEST_CKPT_DIR)


def test_model_profiles_integrity():
    """Verify that all hardware-tailored profiles are valid and have required configurations."""
    for profile in ["nano", "micro", "standard"]:
        assert profile in MODEL_PROFILES
        cfg = MODEL_PROFILES[profile]
        for key in ["d_model", "num_layers", "num_heads", "hyperspace_dim", "d_ff", "num_experts", "top_k"]:
            assert key in cfg, f"Missing key {key} in profile {profile}"
            assert cfg[key] > 0


def test_time_budget_controller():
    """Verify that TimeBudgetController tracks throughput and halts upon budget expiration."""
    controller = TimeBudgetController(max_steps=5, max_time_minutes=None)
    assert not controller.is_over_budget()

    for _ in range(4):
        res = controller.step_tick(batch_tokens=128)
        assert res["tokens_per_sec"] >= 0.0
        assert not res["is_over_budget"]

    res5 = controller.step_tick(batch_tokens=128)
    assert res5["is_over_budget"]
    assert controller.is_over_budget()


def test_checkpoint_atomic_save_and_load():
    """Verify atomic checkpoint writing, absence of temporary debris, and load fidelity."""
    torch.manual_seed(42)
    model = NativeSphericalTransformer(
        vocab_size=32,
        d_model=16,
        num_layers=1,
        num_heads=2,
        hyperspace_dim=8,
        d_ff=32,
        max_seq_len=32,
        num_experts=2,
        top_k=1
    )
    optimizer = HybridMuonAdamW(model, muon_lr=0.01, adamw_lr=1e-3)
    mgr = CheckpointManager(checkpoint_dir=TEST_CKPT_DIR)

    # Save checkpoint
    saved_path = mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        step=42,
        loss=2.15,
        profile_name="test_nano",
        loss_history=[3.5, 2.8, 2.15]
    )

    assert os.path.exists(saved_path)
    assert not os.path.exists(saved_path + ".tmp"), "Temporary file was not cleanly replaced!"
    assert mgr.has_checkpoint()

    # Create new fresh instance and load checkpoint
    model2 = NativeSphericalTransformer(
        vocab_size=32,
        d_model=16,
        num_layers=1,
        num_heads=2,
        hyperspace_dim=8,
        d_ff=32,
        max_seq_len=32,
        num_experts=2,
        top_k=1
    )
    optimizer2 = HybridMuonAdamW(model2, muon_lr=0.01, adamw_lr=1e-3)

    ckpt = mgr.load_latest_checkpoint(model=model2, optimizer=optimizer2)
    assert ckpt["step"] == 42
    assert ckpt["loss"] == 2.15
    assert len(ckpt["loss_history"]) == 3

    # Check weight equality
    p1 = list(model.parameters())[0]
    p2 = list(model2.parameters())[0]
    assert torch.equal(p1, p2), "Model parameters did not load identically!"


def test_500m_peak_checkpoint_strict_load():
    """Verify that the 500M peak checkpoint loads with strict=True under standard MODEL_PROFILES['500m']."""
    peak_path = "artifacts/checkpoints/native_moe_500m_peak.pt"
    if not os.path.exists(peak_path):
        pytest.skip(f"Peak checkpoint {peak_path} not found.")

    cfg = MODEL_PROFILES["500m"]
    from ishmn.data_loader import SovereignTokenizer
    tokenizer = SovereignTokenizer("gpt2")

    # Instantiate with standard profile defaults (max_seq_len=256, enable_world_model=True)
    model = NativeSphericalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        hyperspace_dim=cfg["hyperspace_dim"],
        d_ff=cfg["d_ff"],
        max_seq_len=cfg["max_seq_len"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
        max_experts=cfg["num_experts"],
        enable_shared_expert=cfg["enable_shared_expert"],
        enable_aux_loss_free_balancing=cfg["enable_aux_loss_free_balancing"],
        enable_vsa_memory=True,
        enable_world_model=True,
        tie_weights=True,
        device="cpu"
    )

    ckpt = torch.load(peak_path, map_location="cpu", weights_only=False)
    # Assert strict loading succeeds without throwing RuntimeError
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    assert model.pos_embed.weight.shape[0] == cfg["max_seq_len"]
