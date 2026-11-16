"""
Unit tests for Stage 2 World-Model Uncertainty Calibration, Predictive-Coding
Conflict/OOD Signals, Strict Causality, and Stage 2 Baseline Sanity (§7).
"""

import torch
import torch.nn.functional as F

from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.stage2_baselines import (
    CausalDenseRoPEBaseline,
    CausalLoopedTransformerBaseline,
    GRUBaseline,
)
from dmoe.world_model import LatentWorldModel


def test_world_model_uncertainty_head_trains_and_calibrates():
    torch.manual_seed(0)
    B, L, D = 8, 12, 24
    wm = LatentWorldModel(d_model=D, d_ff=48, max_rollout_horizon=2)
    emb = torch.nn.Embedding(64, D)
    proj = torch.nn.Linear(D, D)
    pos = torch.nn.Parameter(torch.zeros(1, 32, D))
    wm.bind_encoder(emb, proj, pos, chunk_size=1)

    opt = torch.optim.AdamW(wm.parameters(), lr=5e-3)
    unc_losses = []
    for st in range(30):
        # Deterministic cyclic transitions 0 -> 1 -> ... -> 15 -> 0
        starts = torch.randint(0, 16, (B, 1))
        seq = (starts + torch.arange(L).unsqueeze(0)) % 16
        z_seq = proj(emb(seq)) + pos[:, :L, :]
        tgt = wm.encode_target(seq)
        opt.zero_grad()
        ld = wm.compute_world_model_loss(z_seq, tgt)
        ld["loss"].backward()
        if st == 0:
            for name, p in wm.named_parameters():
                if "target_" not in name and p.requires_grad:
                    assert p.grad is not None, f"Dead parameter in LatentWorldModel: {name}"
        opt.step()
        wm.update_target_encoder(emb, proj, pos)
        unc_losses.append(float(ld["unc_loss"].item()))

    assert unc_losses[-1] < unc_losses[0], (
        f"Expected uncertainty calibration loss to decrease: {unc_losses[0]:.4f} -> {unc_losses[-1]:.4f}"
    )


def test_stage2_full_causality_and_predictive_coding_separation():
    torch.manual_seed(1)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=64,
        initial_layers=1,
        chunk_size=1,
        enable_memory=True,
        num_ltm_slots=16,
        enable_three_level_attn=True,
        local_window=4,
        chunk_stride=4,
        enable_layer_stm=True,
        signed_beta=True,
        use_rope=True,
        causal_attn=True,
        auto_write_ltm=False,
        enable_world_model=True,
        wm_rollout_horizon=2,
        enable_thinking=True,
        max_thinking_steps=3,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()
    for _ in range(40):
        starts = torch.randint(32, 48, (12, 1))
        seq = 32 + ((starts - 32 + torch.arange(12).unsqueeze(0)) % 16)
        x = seq[:, :-1]
        y = seq[:, 1:]
        opt.zero_grad()
        out = model(x, force_thinking_steps=2, targets=y)
        loss_d = model.compute_total_loss(out, y, wm_weight=0.25)
        loss_d["loss"].backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        starts = torch.randint(32, 48, (16, 1))
        det_seq = 32 + ((starts - 32 + torch.arange(12).unsqueeze(0)) % 16)
        x_det = det_seq[:, :-1]
        x_det_pert = x_det.clone()
        x_det_pert[:, 6:] = 32 + ((x_det_pert[:, 6:] - 32 + 5) % 16)

        o1 = model(x_det, force_thinking_steps=2)
        o2 = model(x_det_pert, force_thinking_steps=2)
        u1 = model.assess_epistemic_uncertainty(o1)
        u2 = model.assess_epistemic_uncertainty(o2)

        assert (o1["output"][:, :6] - o2["output"][:, :6]).abs().max().item() <= 1e-5
        assert (u1["uncertainty"][:, :6] - u2["uncertainty"][:, :6]).abs().max().item() <= 1e-5

        # Same vocabulary [32..47], corrupted / unpredictable transitions vs coherent sequence
        x_corrupt = torch.randint(32, 48, x_det.shape)
        o_corrupt = model(x_corrupt, force_thinking_steps=2)
        u_corrupt = model.assess_epistemic_uncertainty(o_corrupt)

        assert u_corrupt["pc_error"][:, 1:].mean().item() > u1["pc_error"][:, 1:].mean().item() + 0.15
        assert u_corrupt["uncertainty"][:, 1:].mean().item() > u1["uncertainty"][:, 1:].mean().item() + 0.10


def test_stage2_baselines_causal_and_solve_known_task():
    torch.manual_seed(2)
    x = torch.randint(0, 32, (4, 12))
    x_pert = x.clone()
    x_pert[:, 6:] = (x_pert[:, 6:] + 7) % 32

    for cls, kwargs in [
        (CausalLoopedTransformerBaseline, dict(d_model=32, num_heads=2, d_ff=64, max_steps=3, vocab_size=64, out_vocab=64)),
        (CausalDenseRoPEBaseline, dict(d_model=32, num_heads=2, d_ff=64, num_layers=3, vocab_size=64, out_vocab=64)),
        (GRUBaseline, dict(d_model=32, num_layers=2, vocab_size=64, out_vocab=64)),
    ]:
        m = cls(**kwargs)
        m.eval()
        with torch.no_grad():
            o1 = m(x)["output"]
            o2 = m(x_pert)["output"]
            assert (o1[:, :6] - o2[:, :6]).abs().max().item() <= 1e-5

        # Verify each baseline learns simple next-token rule y_t = (x_t + 3) % 32 (non-strawman check)
        opt = torch.optim.AdamW(m.parameters(), lr=8e-3)
        m.train()
        for _ in range(45):
            xb = torch.randint(0, 32, (16, 10))
            yb = (xb + 3) % 32
            opt.zero_grad()
            out = m(xb)
            if isinstance(m, CausalLoopedTransformerBaseline):
                loss = m.compute_deep_loss(out, yb)
            else:
                loss = F.cross_entropy(out["output"].reshape(-1, 64), yb.reshape(-1))
            loss.backward()
            opt.step()

        m.eval()
        with torch.no_grad():
            xe = torch.randint(0, 32, (32, 10))
            ye = (xe + 3) % 32
            acc = (m(xe)["output"].argmax(dim=-1) == ye).float().mean().item()
            assert acc > 0.85, f"{cls.__name__} failed known-answer sanity task: acc={acc:.3f}"
