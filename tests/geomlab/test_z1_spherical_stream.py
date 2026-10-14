"""
Tests for Round Z1: Spherical Residual Stream (nGPT family).
Verifies:
1. Unit norm of weight vectors after optimizer step (to 1e-5).
2. Stream vectors unit norm at every layer (to 1e-6).
3. Deterministic initialization.
4. Speedup computation on synthetic loss curves (including smoothing and edge cases).
5. Parameter matching for isolating control arm (base_scaled vs ngpt).
6. Salience channel mechanics (Atlas P-1).
"""

import math
import pytest
import torch
import torch.nn as nn
from geomlab.spherical_stream import (
    SphericalStreamLM,
    BaselineLM,
    SphericalStreamWithSalienceLM,
    normalize_weights_,
    compute_speedup,
    l2_norm,
)


def test_weight_unit_norm_after_step():
    """Every linear layer weight vector and embedding vector has unit norm to 1e-5 after optimizer step."""
    torch.manual_seed(42)
    model = SphericalStreamLM(
        vocab_size=96,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_mlp=128,
        max_seq_len=64,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Initial check
    normalize_weights_(model)
    for name, p in model.named_parameters():
        if "weight" in name and p.ndim == 2 and "norm" not in name:
            norms = torch.linalg.norm(p.data, dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), f"Init: {name} not unit norm"

    # Take a synthetic optimization step
    x = torch.randint(0, 96, (4, 16))
    y = torch.randint(0, 96, (4, 16))
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, 96), y.view(-1))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Re-normalize as required by nGPT recipe
    normalize_weights_(model)

    for name, p in model.named_parameters():
        if "weight" in name and p.ndim == 2 and "norm" not in name:
            norms = torch.linalg.norm(p.data, dim=-1)
            max_dev = (norms - 1.0).abs().max().item()
            assert max_dev < 1e-5, f"Post-step: {name} maximum norm deviation {max_dev} exceeds 1e-5"


def test_stream_vectors_unit_norm_at_every_layer():
    """Stream vectors have unit norm (1e-6) at input and after every attention and MLP block."""
    torch.manual_seed(123)
    model = SphericalStreamLM(
        vocab_size=96,
        d_model=64,
        n_layers=3,
        n_heads=2,
        d_mlp=128,
        max_seq_len=64,
    )
    normalize_weights_(model)

    x = torch.randint(0, 96, (2, 32))
    stream_states = model.forward_with_intermediates(x)

    # stream_states contains input embedding and output after each block
    for layer_idx, h in enumerate(stream_states):
        norms = torch.linalg.norm(h, dim=-1)
        max_dev = (norms - 1.0).abs().max().item()
        assert max_dev < 1e-6, f"Layer {layer_idx} stream norm deviation {max_dev} exceeds 1e-6"


def test_deterministic_initialization():
    """Normalized model initialized with the same seed produces identical outputs."""
    torch.manual_seed(999)
    m1 = SphericalStreamLM(vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_mlp=64)
    normalize_weights_(m1)

    torch.manual_seed(999)
    m2 = SphericalStreamLM(vocab_size=64, d_model=32, n_layers=2, n_heads=2, d_mlp=64)
    normalize_weights_(m2)

    x = torch.randint(0, 64, (2, 10))
    out1 = m1(x)
    out2 = m2(x)
    assert torch.equal(out1, out2), "Model outputs differ across identical initialization seeds"


def test_speedup_computation_synthetic_curves():
    """Verifies speedup calculation with 3-point smoothing, token ratio, and wallclock ratio."""
    eval_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    tokens_per_step = 8192

    # Baseline loss decreases slowly from 4.0 to 2.5
    # Steps 800, 900, 1000: [2.52, 2.50, 2.48] -> mean = 2.50
    val_losses_base = [4.0, 3.5, 3.2, 3.0, 2.8, 2.7, 2.6, 2.52, 2.50, 2.48]
    step_times_base = [i * 0.1 for i in eval_steps]  # 100s at step 1000

    # Fast arm reaches smoothed loss <= 2.50 by step 400:
    # Steps 200, 300, 400: [2.6, 2.45, 2.40] -> mean = 2.483 <= 2.50
    val_losses_fast = [3.8, 2.6, 2.45, 2.40, 2.30, 2.20, 2.10, 2.05, 2.02, 2.00]
    step_times_fast = [i * 0.11 for i in eval_steps]  # 10% overhead per step

    res_fast = compute_speedup(
        eval_steps=eval_steps,
        val_losses_base=val_losses_base,
        val_losses_arm=val_losses_fast,
        step_times_base=step_times_base,
        step_times_arm=step_times_fast,
        tokens_per_step=tokens_per_step,
    )

    assert res_fast["reached_target"] is True
    assert res_fast["target_step"] == 400
    # Token speedup: 1000 / 400 = 2.5x
    assert math.isclose(res_fast["token_speedup"], 2.5, rel_tol=1e-3)
    # Wallclock speedup: 100.0 / (400 * 0.11) = 100.0 / 44.0 = 2.2727x
    assert math.isclose(res_fast["wallclock_speedup"], 100.0 / 44.0, rel_tol=1e-3)

    # Slow arm never reaches target (stays above 2.50)
    val_losses_slow = [4.0, 3.8, 3.6, 3.4, 3.2, 3.0, 2.9, 2.8, 2.7, 2.6]
    res_slow = compute_speedup(
        eval_steps=eval_steps,
        val_losses_base=val_losses_base,
        val_losses_arm=val_losses_slow,
        step_times_base=step_times_base,
        step_times_arm=step_times_base,
        tokens_per_step=tokens_per_step,
    )
    assert res_slow["reached_target"] is False
    assert res_slow["target_step"] == 1000
    assert math.isclose(res_slow["token_speedup"], 1.0, rel_tol=1e-3)
    assert math.isclose(res_slow["wallclock_speedup"], 1.0, rel_tol=1e-3)


def test_isolating_control_parameter_matching():
    """base_scaled has identical learnable scale parameters (alpha_A, alpha_M, s_qk, etc.) to ngpt."""
    torch.manual_seed(42)
    ngpt = SphericalStreamLM(
        vocab_size=96,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_mlp=128,
        max_seq_len=64,
        mode="ngpt",
    )
    base_scaled = BaselineLM(
        vocab_size=96,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_mlp=128,
        max_seq_len=64,
        mode="base_scaled",
    )

    # Collect scale parameter shapes
    ngpt_scales = {
        name: p.shape
        for name, p in ngpt.named_parameters()
        if "alpha" in name or name.startswith("s_") or "scale" in name
    }
    scaled_scales = {
        name: p.shape
        for name, p in base_scaled.named_parameters()
        if "alpha" in name or name.startswith("s_") or "scale" in name
    }

    assert len(ngpt_scales) > 0, "nGPT must have learnable scale parameters"
    assert ngpt_scales == scaled_scales, "base_scaled must mirror all learnable scale parameters of ngpt"

    # base_scaled has standard RMSNorms while ngpt does not
    assert any("norm" in name for name, _ in base_scaled.named_parameters())
    assert not any("norm" in name for name, _ in ngpt.named_parameters())


def test_salience_channel_p1():
    """ngpt_salience produces directional unit vectors on S^{d-1} and explicit positive scalar salience."""
    torch.manual_seed(42)
    model = SphericalStreamWithSalienceLM(
        vocab_size=96,
        d_model=64,
        n_layers=2,
        n_heads=2,
        d_mlp=128,
        max_seq_len=64,
    )
    normalize_weights_(model)

    x = torch.randint(0, 96, (2, 16))
    logits, (sphere_state, salience) = model.forward_with_salience(x)

    assert logits.shape == (2, 16, 96)
    assert sphere_state.shape == (2, 16, 64)
    assert salience.shape == (2, 16, 1)

    # Sphere state must be unit-norm
    norms = torch.linalg.norm(sphere_state, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

    # Salience channel must be strictly non-negative
    assert (salience >= 0.0).all().item(), "Salience channel must be non-negative"
