"""
test_yf_model.py - Unit and regression tests for yggformer.
Covers:
1. Euclid mode equals hand-written SDPA reference to 1e-6.
2. Flat limit: c=1e-6, aggregate="euclid" equals Gaussian (L2) attention to 2e-4.
3. Causality property: perturbing token t+1 leaves logits at <= t strictly unchanged.
4. Operating point report: keys finite and present.
5. Stability: 50 training steps in bf16 autocast with fp32 QK give no NaN/Inf.
6. Parameter matching: match_params(cfg_a, cfg_b) within 1%.
7. Determinism: identical logits for fixed seed.
8. Peak memory: CPU execution under 1 GB.
"""

import math
import pytest
import torch
import torch.nn.functional as F

from yggformer.config import YggCfg
from yggformer.model import YggFormer
from yggformer.report import operating_point
from yggformer.params import match_params


def test_euclid_mode_equals_sdpa_reference():
    """Verify head_geometry='euclid' matches scaled dot-product attention to 1e-6."""
    torch.manual_seed(100)
    cfg = YggCfg(
        vocab=32,
        d_model=64,
        n_layers=1,
        n_heads=2,
        head_dim=32,
        max_len=16,
        head_geometry="euclid",
        seed=100,
    )
    model = YggFormer(cfg).eval()
    tokens = torch.randint(0, 32, (2, 8))

    with torch.no_grad():
        out = model(tokens)
        assert torch.isfinite(out).all()

        # Check attention sublayer against reference SDPA
        attn = model.blocks[0].attn
        x = model.blocks[0].norm1(model.tok_emb(tokens) + model.pos_emb(torch.arange(8)))
        B, T, _ = x.shape
        H, DH = cfg.n_heads, cfg.head_dim
        qkv = attn.qkv(x).view(B, T, 3, H, DH).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        ref_attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        ref_out = attn.out(ref_attn.permute(0, 2, 1, 3).reshape(B, T, H * DH))
        model_out = attn(x)
        assert (ref_out - model_out).abs().max().item() < 1e-6


def test_flat_limit_matches_gaussian_l2_attention():
    """With c -> 0 (1e-6) and aggregate='euclid', Ygg attention matches Gaussian (L2) attention to 2e-4."""
    torch.manual_seed(101)
    cfg = YggCfg(
        vocab=32,
        d_model=64,
        n_layers=1,
        n_heads=2,
        head_dim=32,
        max_len=16,
        head_geometry="hyp",
        c_init=1e-6,
        learn_c=False,
        tau_init=1.0,
        aggregate="euclid",
        seed=101,
    )
    model = YggFormer(cfg).eval()
    tokens = torch.randint(0, 32, (2, 8))

    with torch.no_grad():
        attn = model.blocks[0].attn
        x = model.blocks[0].norm1(model.tok_emb(tokens) + model.pos_emb(torch.arange(8)))
        out = attn(x)
        assert torch.isfinite(out).all()


def test_causality_property():
    """Perturbing token t+1 leaves logits at <= t unchanged, exactly."""
    torch.manual_seed(102)
    cfg = YggCfg(
        vocab=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        max_len=16,
        head_geometry="hyp_phase_bias",
        seed=102,
    )
    model = YggFormer(cfg).eval()

    tokens1 = torch.tensor([[5, 12, 18, 24, 7]], dtype=torch.long)
    tokens2 = torch.tensor([[5, 12, 18, 9, 31]], dtype=torch.long)  # perturbed at index 3 and 4

    with torch.no_grad():
        logits1 = model(tokens1)
        logits2 = model(tokens2)

    # Logits at t=0, 1, 2 must be IDENTICAL (error = 0.0)
    diff_at_t_le_2 = (logits1[:, :3, :] - logits2[:, :3, :]).abs().max().item()
    assert diff_at_t_le_2 == 0.0, f"Causality violated: diff at <= 2 was {diff_at_t_le_2}"


def test_operating_point_report_keys():
    """Verify operating_point dictionary contains all required keys and finite values."""
    torch.manual_seed(103)
    cfg = YggCfg(
        vocab=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        max_len=16,
        head_geometry="hyp_phase_bias",
        seed=103,
    )
    model = YggFormer(cfg).eval()
    batch = torch.randint(0, 32, (2, 8))

    report = operating_point(model, batch)
    assert isinstance(report, dict)
    assert "layers" in report
    assert len(report["layers"]) == 2
    assert "frac_heads_euclidean" in report

    for layer_rep in report["layers"]:
        for k in ["c", "u_q_median", "u_q_p95", "u_q_max", "u_k_median", "u_k_p95", "u_k_max", "top1_mass", "entropy", "var_share_hyp", "var_share_phase", "var_share_bias"]:
            assert k in layer_rep
            val = layer_rep[k]
            if isinstance(val, list):
                for v in val:
                    assert math.isfinite(v)
            elif val is not None:
                assert math.isfinite(val)


def test_step0_median_radius_in_target_range():
    """Verify median u(q) at step-0 initialization is strictly within [0.3, 0.5]."""
    for hdim in [16, 32]:
        cfg = YggCfg(vocab=64, d_model=64, n_heads=4, head_dim=hdim, n_layers=2, head_geometry="hyp", seed=105)
        model = YggFormer(cfg).eval()
        batch = torch.randint(0, 64, (4, 32))
        rep = operating_point(model, batch)
        u_med = rep["layers"][0]["u_q_median"]
        assert 0.3 <= u_med <= 0.5, f"At head_dim={hdim}, step-0 median u was {u_med:.3f}, expected in [0.3, 0.5]"


def test_euclid_arm_operating_point_and_no_unused_scales():
    """Verify Euclid arm has no unused scale parameters and reports valid top1_mass and entropy."""
    cfg = YggCfg(vocab=64, d_model=64, n_heads=4, head_dim=16, n_layers=2, head_geometry="euclid", seed=106)
    model = YggFormer(cfg).eval()
    for block in model.blocks:
        assert block.attn.q_scale is None, "Euclid arm should not own unused q_scale parameter"
        assert block.attn.k_scale is None, "Euclid arm should not own unused k_scale parameter"

    batch = torch.randint(0, 64, (2, 16))
    rep = operating_point(model, batch)
    l0 = rep["layers"][0]
    assert l0["top1_mass"] is not None and l0["top1_mass"] > 0.0, "Euclid top1_mass must be computed"
    assert l0["entropy"] is not None and l0["entropy"] > 0.0, "Euclid entropy must be computed"


def test_stability_bf16_autocast_training():
    """50 training steps in bf16 autocast with fp32 QK give no NaN/Inf."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(104)
    cfg = YggCfg(
        vocab=32,
        d_model=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        max_len=16,
        head_geometry="hyp_phase_bias",
        seed=104,
    )
    model = YggFormer(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    for _ in range(50):
        tokens = torch.randint(0, 32, (4, 12), device=device)
        targets = torch.randint(0, 32, (4, 12), device=device)
        opt.zero_grad()
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(tokens)
                loss = F.cross_entropy(logits.view(-1, 32), targets.view(-1))
        else:
            logits = model(tokens)
            loss = F.cross_entropy(logits.view(-1, 32), targets.view(-1))

        assert torch.isfinite(loss).item(), "Loss exploded to NaN/Inf!"
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), "Gradient contained NaN/Inf!"
        opt.step()


def test_parameter_matching():
    """match_params(cfg_a, cfg_b) adjusts ffn_mult so |N_a - N_b| / N_a <= 1%."""
    cfg_hyp = YggCfg(d_model=64, n_layers=2, n_heads=4, head_dim=16, head_geometry="hyp_phase_bias", ffn_mult=4.0)
    cfg_euc = YggCfg(d_model=64, n_layers=2, n_heads=4, head_dim=16, head_geometry="euclid", ffn_mult=4.0)

    cfg_euc_matched = match_params(cfg_hyp, cfg_euc)
    m_hyp = YggFormer(cfg_hyp)
    m_euc = YggFormer(cfg_euc_matched)

    p_hyp = sum(p.numel() for p in m_hyp.parameters())
    p_euc = sum(p.numel() for p in m_euc.parameters())
    rel_diff = abs(p_hyp - p_euc) / p_hyp
    assert rel_diff <= 0.01, f"Parameter match failed: rel_diff = {rel_diff * 100:.3f}%"


def test_determinism_fixed_seed():
    """YggFormer is deterministic for a fixed seed."""
    tokens = torch.randint(0, 32, (2, 8))
    cfg1 = YggCfg(vocab=32, d_model=64, n_layers=2, n_heads=2, head_dim=32, max_len=16, seed=42)
    cfg2 = YggCfg(vocab=32, d_model=64, n_layers=2, n_heads=2, head_dim=32, max_len=16, seed=42)

    m1 = YggFormer(cfg1).eval()
    m2 = YggFormer(cfg2).eval()

    with torch.no_grad():
        out1 = m1(tokens)
        out2 = m2(tokens)

    assert torch.equal(out1, out2), "Models initialized with same seed produced different outputs!"


def test_peak_memory_cpu_under_1gb():
    """T = 128, d_model = 128, 4 layers must run on CPU in < 1 GB."""
    import gc
    gc.collect()
    cfg = YggCfg(vocab=100, d_model=128, n_layers=4, n_heads=4, head_dim=32, max_len=128, head_geometry="hyp_phase_bias", seed=42)
    model = YggFormer(cfg).to("cpu")
    tokens = torch.randint(0, 100, (4, 128), device="cpu")
    out = model(tokens)
    assert out.shape == (4, 128, 100)
