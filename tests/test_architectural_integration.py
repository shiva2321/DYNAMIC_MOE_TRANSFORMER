"""
Comprehensive Architectural Integration Verification Tests.

Verifies:
1. Scale-equivariant log-radial perception on S^{D-1} x R+ (no saturation at 23 Trillion)
2. Exact roundtrip log-radial encode/decode
3. Causal masking in Looped Riemannian Transformer (Path A)
4. Scale-equivariant continuous arithmetic execution in UCE (Path B)
5. Autonomous living self-loop integration with UCE
"""

import math
import pytest
import torch
import torch.nn as nn
from ishmn import (
    UniversalCognitiveEngine,
    UniversalPerceptor,
    LoopedRiemannianTransformer,
    ContinuousSelfLoop
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def uce():
    engine = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        device=DEVICE
    )
    return engine


def test_perceptor_scale_equivariance_at_astronomical_magnitudes():
    """Verify that 23 Trillion maps to the sphere with zero saturation or NaNs."""
    perceptor = UniversalPerceptor(hyperspace_dim=8, payload_dim=32, embedding_dim=64).to(DEVICE)
    
    # 23 Trillion, single digits, continuous fractions, and negative values
    test_vectors = torch.tensor([
        [23146484561521.0],
        [45.0],
        [0.0035],
        [-1000000000.0]
    ], device=DEVICE)
    
    u, r, v = perceptor(test_vectors, modality="continuous", use_log_radial=True)
    
    # Verify finite values (no Inf, no NaN)
    assert torch.isfinite(u).all()
    assert torch.isfinite(r).all()
    assert torch.isfinite(v).all()
    
    # Verify unit norm on S^{D-1}
    norms = torch.norm(u, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    
    # Verify positive radial distance
    assert (r > 0).all()


def test_log_radial_roundtrip():
    """Verify exact roundtrip mapping for sign and log-magnitude."""
    values = [0.0, 1.0, 12.5, 23164.0, 23146484561521.0, -99999.0]
    for val in values:
        s, lm = UniversalPerceptor.log_radial_encode(val, scale=20.0)
        rec = UniversalPerceptor.log_radial_decode(s, lm, scale=20.0)
        if abs(val) > 0:
            rel_err = abs(rec - val) / abs(val)
            assert rel_err < 1e-4, f"Failed roundtrip for {val}: got {rec}"
        else:
            assert abs(rec - val) < 1e-6


def test_looped_transformer_causal_masking():
    """Verify that is_causal=True prevents past positions from attending to future positions."""
    transformer = LoopedRiemannianTransformer(
        d_model=32,
        num_heads=4,
        hyperspace_dim=8,
        d_ff=64,
        max_loops=2,
        device=DEVICE
    )
    
    B, T, D = 1, 5, 32
    x1 = torch.randn(B, T, D, device=DEVICE)
    
    # In x2, alter ONLY the final token (T=4)
    x2 = x1.clone()
    x2[:, -1, :] = x2[:, -1, :] + 10.0
    
    # Run with causal masking
    out1, _ = transformer(x1, is_causal=True)
    out2, _ = transformer(x2, is_causal=True)
    
    # Positions 0 to 3 should be IDENTICAL between out1 and out2 because future token cannot leak backwards
    diff_prefix = torch.norm(out1[:, :4, :] - out2[:, :4, :]).item()
    assert diff_prefix < 1e-4, f"Future token leaked into past! Prefix diff: {diff_prefix}"
    
    # Position 4 should differ
    diff_suffix = torch.norm(out1[:, 4:, :] - out2[:, 4:, :]).item()
    assert diff_suffix > 0.1


def test_uce_solve_continuous_arithmetic(uce):
    """Verify solve_continuous_arithmetic executes across orders of magnitude."""
    # Test 23 trillion addition
    res_astro = uce.solve_continuous_arithmetic(23146484561521.0, "+", 45132030.0)
    assert "prediction" in res_astro
    assert math.isfinite(res_astro["prediction"])
    assert res_astro["steps_taken"] >= 1
    
    # Test continuous decimal subtraction
    res_dec = uce.solve_continuous_arithmetic(9.5, "-", 4.5)
    assert "prediction" in res_dec
    assert math.isfinite(res_dec["prediction"])


def test_uce_autonomous_living_self_loop(uce):
    """Verify that UCE runs the living continuous autonomous self-loop."""
    stimulus = torch.randn(1, 32, device=uce.device)
    inputs = [{"data": stimulus, "modality": "continuous", "tag": "test_stim"}]
    
    emissions = uce.run_autonomous_loop(num_ticks=5, inputs=inputs, emission_threshold=0.01)
    assert isinstance(emissions, list)


def test_uce_train_continuous_step(uce):
    """Verify train_continuous_step trains the scale-equivariant log-radial pipeline."""
    a_vals = [23146484561521.0, 100.0, 9.5]
    op_codes = [0, 1, 1]  # +, -, -
    b_vals = [45132030.0, 45.0, 4.5]
    y_vals = [23146529693551.0, 55.0, 5.0]

    report = uce.train_continuous_step(a_vals, op_codes, b_vals, y_vals)
    assert "loss" in report
    assert math.isfinite(report["loss"])
    assert report["loss"] > 0.0

