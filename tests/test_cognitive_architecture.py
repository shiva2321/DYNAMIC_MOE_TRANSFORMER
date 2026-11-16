import torch
import torch.nn.functional as F

from dmoe.hierarchical_attention import ThreeLevelAttention
from dmoe.memory import DualTimescaleMemory, GatedDeltaWorkingMemory, LongTermSlotMemory
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.world_model import LatentWorldModel


def test_dual_timescale_memory_short_and_long_term():
    torch.manual_seed(11)
    B, L, D = 4, 12, 16
    mem = DualTimescaleMemory(d_model=D, num_heads=2, num_ltm_slots=16, write_surprise_threshold=0.35)

    x = torch.randn(B, L, D)
    out1 = mem(x, update_ltm=False)
    assert out1["fused"].shape == (B, L, D)
    assert out1["stm_state"].shape == (B, 2, D // 2, D // 2)
    assert int(mem.long_term.slot_used.sum().item()) == 0

    # High-surprise write should populate LTM slots
    high_surp = torch.full((B, L), 0.90)
    writes = mem.long_term.write(x, surprise=high_surp)
    assert writes > 0
    assert int(mem.long_term.slot_used.sum().item()) > 0

    # Subsequent read with a query matching written vectors should produce non-zero ltm_out
    out2 = mem(x, update_ltm=False)
    assert out2["ltm_out"].abs().sum().item() > 0.0


def test_three_level_attention_causal_no_future_leak():
    torch.manual_seed(22)
    B, L, D = 2, 16, 16
    attn = ThreeLevelAttention(d_model=D, num_heads=2, local_window=4, chunk_stride=4)
    attn.eval()

    x1 = torch.randn(B, L, D)
    x2 = x1.clone()
    # Perturb only future tokens from t=9 onward
    x2[:, 9:, :] = torch.randn(B, L - 9, D)
    mem_ctx = torch.randn(B, 5, D)

    out1, w1 = attn(x1, memory_context=mem_ctx)
    out2, w2 = attn(x2, memory_context=mem_ctx)

    assert out1.shape == (B, L, D)
    assert w1.shape == (B, L, 3)
    assert torch.allclose(w1.sum(dim=-1), torch.ones(B, L), atol=1e-5)

    # Strictly causal: positions 0..8 must be identical despite perturbation at 9..15
    assert torch.allclose(out1[:, :9, :], out2[:, :9, :], atol=1e-5)

    # Empty/None memory_context suppresses Global branch weight while keeping causality
    out_nomem1, w_nomem1 = attn(x1, memory_context=None)
    out_nomem2, _ = attn(x2, memory_context=None)
    assert w_nomem1[..., 2].max().item() < 1e-6
    assert torch.allclose(w_nomem1.sum(dim=-1), torch.ones(B, L), atol=1e-5)
    assert torch.allclose(out_nomem1[:, :9, :], out_nomem2[:, :9, :], atol=1e-5)


def test_world_model_imagination_and_training():
    torch.manual_seed(33)
    B, L, D = 8, 12, 16
    wm = LatentWorldModel(d_model=D, d_ff=32, max_rollout_horizon=3)
    opt = torch.optim.AdamW(wm.parameters(), lr=1e-2)

    # Deterministic rotating latent trajectory
    t = torch.arange(L, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    base = torch.randn(B, 1, D)
    z_seq = torch.sin(t * 0.4 + base)
    targets = z_seq.detach()

    img0 = wm.imagine(z_seq, horizon=3)
    assert img0["imagined_latents"].shape == (B, L, 3, D)
    assert img0["imagined_uncertainty"].shape == (B, L, 3)

    init_loss = float(wm.compute_world_model_loss(z_seq, targets)["loss"].item())
    for _ in range(40):
        opt.zero_grad()
        ld = wm.compute_world_model_loss(z_seq, targets)
        ld["loss"].backward()
        opt.step()

    final_ld = wm.compute_world_model_loss(z_seq, targets)
    assert float(final_ld["loss"].item()) < init_loss
    assert float(final_ld["cos_sim_1step"].item()) > 0.70


def test_full_cognitive_dmoe_forward_and_training_step():
    torch.manual_seed(44)
    model = RealSelfStabilizingDMoE(
        d_model=24,
        num_heads=2,
        d_ff=48,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        auto_lifecycle=True,
        enable_memory=True,
        num_ltm_slots=16,
        enable_three_level_attn=True,
        local_window=4,
        chunk_stride=4,
        enable_world_model=True,
        wm_rollout_horizon=2,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()

    xb = torch.randint(32, 48, (4, 16))
    yb = ((xb - 32 + 1) % 16) + 32

    opt.zero_grad()
    out = model(xb, force_thinking_steps=2, targets=yb)
    loss_dict = model.compute_total_loss(out, yb, wm_weight=0.25)
    loss_dict["loss"].backward()
    opt.step()

    assert torch.isfinite(loss_dict["loss"])
    assert out["memory_out"] is not None
    assert len(out["level_weights"]) == 1
    assert out["world_model_imagine"] is not None
    assert out["world_model_loss"] is not None
    assert out["novelty_verdict"] is not None

