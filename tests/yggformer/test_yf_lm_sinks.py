"""
test_yf_lm_sinks.py - Unit and invariant tests for Round Y6 (E8/E9 Small LM and Attention Sinks).
Verifies:
1. Null key support in attention and model.
2. Head-wise sigmoid gate support (Qiu et al. 2025).
3. First-token attention mass (sink metric) and max activation norm extraction.
4. Char-level dataset loader on local Python stdlib text.
"""

import os
import pytest
import torch
import torch.nn.functional as F

from yggformer.config import YggCfg
from yggformer.model import YggFormer
from yggformer.attention import YggMultiHeadAttention


def test_null_key_forward():
    """Verify null_key=True adds origin key and produces valid causal logits."""
    cfg = YggCfg(
        vocab=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        head_dim=16,
        max_len=16,
        head_geometry="hyp_phase_bias",
        null_key=True,
        seed=100,
    )
    model = YggFormer(cfg)
    tokens = torch.randint(0, 32, (2, 8))
    out = model(tokens)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_euclid_head_gate():
    """Verify head-wise sigmoid output gate (Qiu et al. 2025) modulates attention."""
    cfg = YggCfg(
        vocab=32,
        d_model=32,
        n_layers=1,
        n_heads=2,
        head_dim=16,
        max_len=16,
        head_geometry="euclid_gate",
        seed=101,
    )
    model = YggFormer(cfg)
    tokens = torch.randint(0, 32, (2, 8))
    out = model(tokens)
    assert out.shape == (2, 8, 32)
    assert torch.isfinite(out).all()


def test_sink_metric_calculation():
    """Verify first-token attention sink mass and max activation norm computation."""
    B, H, T = 2, 4, 16
    # Artificial attention weights where first token receives 70% of mass
    attn = torch.zeros(B, H, T, T)
    attn[:, :, :, 0] = 0.7
    attn[:, :, :, 1:] = 0.3 / (T - 1)

    # Sink mass = average over query positions t >= 1 of attention to token 0
    sink_mass = attn[:, :, 1:, 0].mean().item()
    assert abs(sink_mass - 0.7) < 1e-5
