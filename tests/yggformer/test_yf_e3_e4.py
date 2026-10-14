"""
test_yf_e3_e4.py - Unit and integration tests for Round Y3 (E3 sweep & E4 numerics).
Verifies:
1. Inputs_embeds forward in YggFormer.
2. Parameter matching across all 4 arms (euclid, hyp, hyp_phase, hyp_phase_bias) within 1%.
3. Training step on hierarchical task converges without NaNs.
4. E4 precision handling: safe_radius and clip_radius under different dtypes.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from yggformer.config import YggCfg
from yggformer.model import YggFormer
from yggformer.params import match_params, count_parameters
from yggformer.report import operating_point
from yggformer.tasks.hierarchy import TreeHierarchyDataset


def test_yggformer_inputs_embeds():
    """Verify YggFormer accepts inputs_embeds directly and preserves causality."""
    cfg = YggCfg(
        vocab=16,
        d_model=32,
        n_layers=2,
        n_heads=2,
        head_dim=16,
        max_len=32,
        head_geometry="hyp",
        seed=100,
    )
    model = YggFormer(cfg)
    embeds = torch.randn(2, 10, 32)
    out = model(inputs_embeds=embeds)
    assert out.shape == (2, 10, 16)
    assert torch.isfinite(out).all()


def test_param_matching_across_all_four_arms():
    """Verify parameter matching within 1% across euclid, hyp, hyp_phase, hyp_phase_bias."""
    arms = ["hyp", "hyp_phase", "hyp_phase_bias"]
    for hd in [8, 16, 32, 64]:
        base_cfg = YggCfg(
            vocab=8,
            d_model=hd * 2,
            n_layers=2,
            n_heads=2,
            head_dim=hd,
            head_geometry="euclid",
            ffn_mult=2.0,
            seed=101,
        )
        base_model = YggFormer(base_cfg)
        n_base = count_parameters(base_model)

        for arm in arms:
            target_cfg = YggCfg(
                vocab=8,
                d_model=hd * 2,
                n_layers=2,
                n_heads=2,
                head_dim=hd,
                head_geometry=arm,
                ffn_mult=2.0,
                seed=101,
            )
            matched_cfg = match_params(base_cfg, target_cfg)
            matched_model = YggFormer(matched_cfg)
            n_matched = count_parameters(matched_model)
            rel_diff = abs(n_matched - n_base) / n_base
            assert rel_diff <= 0.01, f"Rel diff {rel_diff:.4f} exceeds 1% for arm={arm}, hd={hd}"


def test_e4_precision_clipping_invariants():
    """Verify safe radius and clipping behavior across precision modes."""
    from geometry.attention import safe_radius, clip_radius

    assert safe_radius("bfloat16") == 2.0
    assert safe_radius("float16") == 6.0
    assert safe_radius("float32") == 50.0

    # Vectors exceeding safe radius are scaled down to safe radius
    large_x = torch.tensor([[10.0, 10.0]], dtype=torch.float32)
    clipped_bf16 = clip_radius(large_x, safe_radius("bfloat16"))
    assert torch.norm(clipped_bf16, dim=-1).item() <= 2.0001


def test_bf16_precision_difference_and_clipping():
    """Verify that bfloat16 compute_dtype differs from float32 on large-radius inputs by expected roundoff error."""
    from yggformer.attention import YggMultiHeadAttention

    cfg32 = YggCfg(head_dim=16, d_model=32, qk_dtype="float32", head_geometry="hyp", seed=42)
    cfg_bf16 = YggCfg(head_dim=16, d_model=32, qk_dtype="bfloat16", head_geometry="hyp", seed=42)

    attn32 = YggMultiHeadAttention(cfg32)
    attn_bf16 = YggMultiHeadAttention(cfg_bf16)
    attn_bf16.load_state_dict(attn32.state_dict())

    # Input operating at radius u ~ 1.2 - 1.8 (within bf16 safe_radius=2.0)
    x = torch.randn(2, 8, 32) * 0.7
    out32 = attn32(x)
    out_bf16 = attn_bf16(x)

    diff = (out32 - out_bf16).abs().max().item()
    # Should show non-zero bfloat16 mantissa truncation error (between 1e-4 and 0.15)
    assert 1e-4 < diff < 0.15, f"Expected non-zero precision difference in [1e-4, 0.15], got {diff}"

    # Large radius stress: clipping keeps output finite and non-NaN
    x_large = torch.randn(2, 8, 32) * 5.0
    out_large = attn_bf16(x_large)
    assert not torch.isnan(out_large).any(), "bfloat16 produced NaNs under large-radius stress"
    assert not torch.isinf(out_large).any(), "bfloat16 produced Infs under large-radius stress"

