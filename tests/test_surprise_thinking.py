"""
Unit and behavioral verification tests for Surprise-Driven Dynamic Latent Thinking
(dmoe/thinking.py and RealSelfStabilizingDMoE with enable_thinking=True).
"""

import torch
import torch.nn.functional as F

from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.thinking import SurpriseDrivenLatentThinker


def test_thinking_disabled_by_default():
    model = RealSelfStabilizingDMoE(
        d_model=16, num_heads=2, d_ff=32, initial_layers=2, chunk_size=1
    )
    assert model.enable_thinking is False
    assert model.thinker is None

    model.eval()
    x = torch.randint(32, 48, (2, 8), dtype=torch.long)
    out = model(x)
    assert out["thinking_steps_used"] == 0
    assert out["active_token_fraction"] == 0.0


def test_fast_path_skip_when_surprise_below_skip_threshold():
    # skip_threshold=1.0 guarantees initial surprise (in [0, 1]) is <= skip_threshold
    model = RealSelfStabilizingDMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=2,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=5,
        skip_threshold=1.0,
    )
    model.eval()
    x = torch.randint(32, 48, (2, 8), dtype=torch.long)
    out = model(x)
    assert out["thinking_steps_used"] == 0
    assert out["active_token_fraction"] == 0.0
    assert torch.allclose(out["latent"], out["anchor_latent"])


def test_runs_to_max_steps_when_thresholds_unreachable():
    model = RealSelfStabilizingDMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=2,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=4,
        skip_threshold=-1.0,
        theta_halt=-1.0,
    )
    model.eval()
    x = torch.randint(32, 48, (2, 8), dtype=torch.long)
    out = model(x)
    assert out["thinking_steps_used"] == 4
    assert out["active_token_fraction"] == 1.0
    assert len(out["step_logits"]) == 5  # step 0 + 4 thinking steps
    assert out["token_steps"].shape == (2, 8)
    assert torch.all(out["token_steps"] == 4.0)


def test_per_token_selective_halting_allocates_variable_steps():
    torch.manual_seed(7)
    thinker = SurpriseDrivenLatentThinker(
        d_model=16,
        num_heads=2,
        d_ff=32,
        max_thinking_steps=4,
        skip_threshold=0.0,
        theta_halt=0.50,
    )
    thinker.eval()

    # Create a mock decoder where position 0 has near-zero entropy (peaked logit)
    # and position 1 has maximum entropy (uniform logits), and set energy head weights
    # so position 0 has low energy and position 1 has high energy.
    with torch.no_grad():
        for p in thinker.energy_head.parameters():
            p.zero_()

    def mock_decoder(z: torch.Tensor) -> torch.Tensor:
        B, L, _ = z.shape
        logits = torch.zeros(B, L, 256, device=z.device)
        # Position 0: peaked logit -> entropy ~ 0
        logits[:, 0, 32] = 50.0
        # Position 1: uniform logits -> entropy = 1.0
        return logits

    # Make think_block produce zero delta on position 0 and large delta on position 1
    h_anchor = torch.zeros(1, 2, 16)
    h_anchor[:, 1, :] = 5.0

    # Override compute_surprise to give pos 0 surprise=0.10 (<0.50) and pos 1 surprise=0.80 (>0.50)
    def controlled_surprise(h_a, z_s, z_c, lg):
        s = torch.tensor([[0.10, 0.80]], device=h_a.device)
        return s, s

    thinker.compute_surprise = controlled_surprise  # type: ignore[method-assign]
    res = thinker(h_anchor, mock_decoder)

    # Position 0 should halt immediately (0 steps), Position 1 should run all 4 steps
    assert res["token_steps"][0, 0].item() == 0.0
    assert res["token_steps"][0, 1].item() == 4.0
    assert 0.0 < res["active_fraction"] < 1.0


def test_latent_state_bounded_under_20_thinking_steps():
    torch.manual_seed(42)
    model = RealSelfStabilizingDMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=2,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=20,
        skip_threshold=-1.0,
        theta_halt=-1.0,
    )
    model.eval()
    x = torch.randint(32, 48, (2, 8), dtype=torch.long)
    out = model(x)
    assert out["thinking_steps_used"] == 20
    assert torch.isfinite(out["latent"]).all()
    assert torch.isfinite(out["output"]).all()
    assert out["latent"].norm(dim=-1).max().item() < 100.0


def test_deep_supervision_improves_step_k_over_step_0_on_multihop_task():
    """
    Verify that training with compute_thinking_loss on a variable-depth state-chain
    task (depths D in 0..3 during training):
    1. Strictly and monotonically improves accuracy across thinking steps k = 0..5.
    2. Generalizes zero-shot to out-of-distribution reasoning depths D = 4, 5.
    3. Calibrates token-level surprise so adaptive halting allocates ~d steps to
       depth-d tokens, saving >50% of active token compute while maintaining >=95% accuracy.
    """
    torch.manual_seed(13)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 8 states (0..7) and 8 transition operators (8..15)
    gen = torch.Generator().manual_seed(99)
    trans = torch.stack([torch.randperm(8, generator=gen) for _ in range(8)])

    def make_chain_batch(batch_size: int, seq_len: int, max_depth: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        x = torch.zeros(batch_size, seq_len, dtype=torch.long)
        y = torch.zeros(batch_size, seq_len, dtype=torch.long)
        depths = torch.zeros(batch_size, seq_len, dtype=torch.long)
        cur_s = torch.randint(0, 8, (batch_size,), generator=g)
        x[:, 0] = cur_s
        y[:, 0] = cur_s
        cur_d = torch.zeros(batch_size, dtype=torch.long)
        for t in range(1, seq_len):
            force_reset = cur_d >= max_depth
            rand_reset = torch.rand(batch_size, generator=g) < 0.25
            do_reset = force_reset | rand_reset
            new_s = torch.randint(0, 8, (batch_size,), generator=g)
            op = torch.randint(0, 8, (batch_size,), generator=g)
            x[:, t] = torch.where(do_reset, new_s, op + 8)
            cur_s = torch.where(do_reset, new_s, trans[op, cur_s])
            cur_d = torch.where(do_reset, torch.zeros_like(cur_d), cur_d + 1)
            y[:, t] = cur_s
            depths[:, t] = cur_d
        return x.to(device) + 32, y.to(device) + 32, depths.to(device)

    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=5,
        skip_threshold=0.05,
        theta_halt=0.18,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    model.train()
    for st in range(300):
        xb, yb, _ = make_chain_batch(64, 16, max_depth=3, seed=100 + st)
        opt.zero_grad()
        out = model(xb, force_thinking_steps=3)
        loss_dict = model.thinker.compute_thinking_loss(out["thinking_out"], yb)
        loss_dict["loss"].backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        x_test, y_test, d_test = make_chain_batch(128, 16, max_depth=5, seed=9999)
        # 1. Fixed 5-step rollout to check monotonic step-by-step improvement and OOD depth 4,5
        out_full = model(x_test, force_thinking_steps=5)
        step_logits = out_full["step_logits"]
        step_accs = [
            (lg.squeeze(2).argmax(dim=-1) == y_test).float().mean().item()
            for lg in step_logits
        ]
        ce_0 = F.cross_entropy(step_logits[0].reshape(-1, 256), y_test.reshape(-1)).item()
        ce_final = F.cross_entropy(step_logits[-1].reshape(-1, 256), y_test.reshape(-1)).item()

        # OOD depth-4 and depth-5 accuracy at step 5
        preds_5 = step_logits[-1].squeeze(2).argmax(dim=-1)
        ood_mask = d_test >= 4
        ood_acc = (preds_5[ood_mask] == y_test[ood_mask]).float().mean().item()

        # 2. Adaptive surprise-gated rollout (no oracle depth hints)
        out_adapt = model(x_test)
        preds_adapt = out_adapt["output"].squeeze(2).argmax(dim=-1)
        acc_adapt = (preds_adapt == y_test).float().mean().item()
        tsteps = out_adapt["token_steps"]
        steps_d0 = tsteps[d_test == 0].mean().item()
        steps_d1 = tsteps[d_test == 1].mean().item()
        steps_d3 = tsteps[d_test == 3].mean().item()
        steps_d5 = tsteps[d_test == 5].mean().item()

    assert ce_final < ce_0
    assert step_accs[0] < step_accs[1] < step_accs[2] < step_accs[3] < step_accs[4] < step_accs[5]
    assert step_accs[-1] >= 0.95
    assert ood_acc >= 0.90
    assert acc_adapt >= 0.95
    assert out_adapt["active_token_fraction"] < 0.50
    assert steps_d0 < steps_d1 < steps_d3 < steps_d5


def test_joint_thinker_and_upkeeper_multihop_belief_revision():
    """
    Verify that combining SurpriseDrivenLatentThinker + EvidenceAccumulatingUpkeeper:
    1. Learns the initial multi-hop transition rules across depths D = 0..5.
    2. Resists a 5-batch single-source contradictory burst (K2 post-burst >= 0.95).
    3. Revises its multi-hop transitions under persistent multi-source counter-evidence (K3)
       and generalizes the revised rules zero-shot to OOD depths D = 4, 5 (>= 0.90).
    """
    torch.manual_seed(21)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gen = torch.Generator().manual_seed(99)
    trans_init = torch.stack([torch.randperm(8, generator=gen) for _ in range(8)])
    trans_burst = (trans_init + 3) % 8
    trans_rev = (trans_init + 5) % 8

    def make_chain_batch(trans_table: torch.Tensor, batch_size: int, seq_len: int, max_depth: int, seed: int):
        g = torch.Generator().manual_seed(seed)
        x = torch.zeros(batch_size, seq_len, dtype=torch.long)
        y = torch.zeros(batch_size, seq_len, dtype=torch.long)
        depths = torch.zeros(batch_size, seq_len, dtype=torch.long)
        cur_s = torch.randint(0, 8, (batch_size,), generator=g)
        x[:, 0] = cur_s
        y[:, 0] = cur_s
        cur_d = torch.zeros(batch_size, dtype=torch.long)
        for t in range(1, seq_len):
            force_reset = cur_d >= max_depth
            rand_reset = torch.rand(batch_size, generator=g) < 0.25
            do_reset = force_reset | rand_reset
            new_s = torch.randint(0, 8, (batch_size,), generator=g)
            op = torch.randint(0, 8, (batch_size,), generator=g)
            x[:, t] = torch.where(do_reset, new_s, op + 8)
            cur_s = torch.where(do_reset, new_s, trans_table[op, cur_s])
            cur_d = torch.where(do_reset, torch.zeros_like(cur_d), cur_d + 1)
            y[:, t] = cur_s
            depths[:, t] = cur_d
        return x.to(device) + 32, y.to(device) + 32, depths.to(device)

    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=5,
        skip_threshold=0.05,
        theta_halt=0.18,
    ).to(device)

    # Phase 1: Initial learning on trans_init (depths D <= 3)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    model.train()
    for st in range(240):
        xb, yb, _ = make_chain_batch(trans_init, 64, 16, max_depth=3, seed=1000 + st)
        opt.zero_grad()
        out = model(xb, force_thinking_steps=3)
        preds = out["output"].squeeze(2).argmax(dim=-1)
        gated_y, mask = model.upkeeper.filter_and_gate_targets(
            xb, yb, preds, source_id="trusted_source", state_chain_mode=True
        )
        if bool(mask.any()):
            loss_dict = model.thinker.compute_thinking_loss(
                out["thinking_out"], gated_y, valid_mask=mask
            )
            loss_dict["loss"].backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        x_eval, y_eval, _ = make_chain_batch(trans_init, 128, 16, max_depth=5, seed=8888)
        acc_init = (model(x_eval)["output"].squeeze(2).argmax(dim=-1) == y_eval).float().mean().item()

    # Phase 2: K2 single-source contradictory burst (5 batches from noisy_source_1)
    model.train()
    opt_burst = torch.optim.AdamW(model.parameters(), lr=4e-3)
    for st in range(5):
        xb, yb, _ = make_chain_batch(trans_burst, 64, 16, max_depth=3, seed=2000 + st)
        opt_burst.zero_grad()
        out = model(xb, force_thinking_steps=3)
        preds = out["output"].squeeze(2).argmax(dim=-1)
        gated_y, mask = model.upkeeper.filter_and_gate_targets(
            xb, yb, preds, source_id="noisy_source_1", state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            loss_dict = model.thinker.compute_thinking_loss(
                out["thinking_out"], gated_y, valid_mask=mask
            )
            loss_dict["loss"].backward()
            opt_burst.step()

    model.eval()
    with torch.no_grad():
        acc_post_burst = (model(x_eval)["output"].squeeze(2).argmax(dim=-1) == y_eval).float().mean().item()

    # Phase 3: K3 multi-source genuine belief revision (trans_rev on depths D <= 3)
    model.train()
    opt_rev = torch.optim.AdamW(model.parameters(), lr=4e-3)
    sources = ["source_alpha", "source_beta", "source_gamma"]
    for st in range(250):
        xb, yb, _ = make_chain_batch(trans_rev, 64, 16, max_depth=3, seed=3000 + st)
        opt_rev.zero_grad()
        out = model(xb, force_thinking_steps=3)
        preds = out["output"].squeeze(2).argmax(dim=-1)
        src = sources[st % len(sources)]
        gated_y, mask = model.upkeeper.filter_and_gate_targets(
            xb, yb, preds, source_id=src, state_chain_mode=True
        )
        if int(mask.sum().item()) > 16:
            loss_dict = model.thinker.compute_thinking_loss(
                out["thinking_out"], gated_y, valid_mask=mask
            )
            loss_dict["loss"].backward()
            opt_rev.step()

    model.eval()
    with torch.no_grad():
        x_rev, y_rev, d_rev = make_chain_batch(trans_rev, 128, 16, max_depth=5, seed=9999)
        out_rev = model(x_rev)
        preds_rev = out_rev["output"].squeeze(2).argmax(dim=-1)
        acc_post_rev = (preds_rev == y_rev).float().mean().item()
        ood_rev_mask = d_rev >= 4
        acc_ood_rev = (preds_rev[ood_rev_mask] == y_rev[ood_rev_mask]).float().mean().item()

    assert acc_init >= 0.95
    assert acc_post_burst >= 0.95
    assert acc_post_rev >= 0.95
    assert acc_ood_rev >= 0.90
    assert model.upkeeper.tentative_store_bytes() <= 4096
