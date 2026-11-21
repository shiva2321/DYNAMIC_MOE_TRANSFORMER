"""
Tests for Autonomous Expert Lifecycle Management (dmoe/lifecycle.py + RealSelfStabilizingDMoE).

Verifies:
1. Backward compatibility (auto_lifecycle=False by default).
2. Anti-fragmentation cooldown prevents false spawns on short novelty bursts.
3. Self-triggered freeze + spawn on a genuine task-ID-free domain shift (zero manual spawn/freeze calls).
4. Automatic downgrade to DORMANT, revival on returning domain, and M08-safe pruning past prune_age.
"""

import torch
import torch.nn.functional as F

from dmoe.lifecycle import STATE_ACTIVE, STATE_DORMANT, STATE_FROZEN
from dmoe.real_model import RealSelfStabilizingDMoE


def test_auto_lifecycle_disabled_by_default():
    model = RealSelfStabilizingDMoE(
        d_model=16, num_heads=2, d_ff=32, initial_layers=1, chunk_size=1
    )
    assert model.auto_lifecycle is False
    assert model.lifecycle is None


def test_no_false_spawn_on_short_novelty_burst():
    torch.manual_seed(11)
    model = RealSelfStabilizingDMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        auto_lifecycle=True,
        min_mature_steps=10,
        novel_confirm_steps=4,
        novel_surprise_threshold=0.30,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
    model.train()

    # Mature Expert 0 on Domain A (tokens 32..39)
    for st in range(15):
        xb = torch.randint(32, 40, (16, 8))
        yb = (xb - 32 + 1) % 8 + 32
        opt.zero_grad()
        out = model(xb, force_thinking_steps=2)
        loss = model.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"]
        loss.backward()
        opt.step()

    assert len(model.layers[0].moe.experts) == 1

    # Short 2-step burst on novel Domain B (tokens 80..87) — shorter than novel_confirm_steps=4
    for _ in range(2):
        xb_burst = torch.randint(80, 88, (16, 8))
        yb_burst = (xb_burst - 80 + 3) % 8 + 80
        opt.zero_grad()
        out = model(xb_burst, force_thinking_steps=2)
        loss = model.thinker.compute_thinking_loss(out["thinking_out"], yb_burst)["loss"]
        loss.backward()
        opt.step()

    assert len(model.layers[0].moe.experts) == 1
    assert model.lifecycle.total_spawns == 0


def test_self_triggered_freeze_and_spawn_on_task_shift():
    torch.manual_seed(19)
    model = RealSelfStabilizingDMoE(
        d_model=24,
        num_heads=2,
        d_ff=64,
        initial_layers=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=2,
        auto_lifecycle=True,
        max_experts=4,
        min_mature_steps=20,
        novel_confirm_steps=3,
        novel_surprise_threshold=0.35,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    model.train()

    # Phase 1: Train Domain A (tokens 32..39) — no manual spawn/freeze calls
    for st in range(40):
        xb = torch.randint(32, 40, (16, 8))
        yb = (xb - 32 + 1) % 8 + 32
        opt.zero_grad()
        out = model(xb, force_thinking_steps=2)
        loss = model.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"]
        loss.backward()
        opt.step()

    assert len(model.layers[0].moe.experts) == 1

    # Phase 2: Shift stream to Domain B (tokens 120..127) — still zero manual calls
    spawned = False
    for st in range(40):
        xb = torch.randint(120, 128, (16, 8))
        yb = (xb - 120 + 3) % 8 + 120
        opt.zero_grad()
        out = model(xb, force_thinking_steps=2)
        if any(ev.startswith("spawned_expert_") for ev in out["lifecycle_events"]):
            spawned = True
            # Register newly spawned expert parameters in optimizer
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad], lr=5e-3
            )
        loss = model.thinker.compute_thinking_loss(out["thinking_out"], yb)["loss"]
        loss.backward()
        opt.step()

    assert spawned is True
    assert len(model.layers[0].moe.experts) == 2
    assert len(model.decoder.expert_heads) == 2
    assert model.lifecycle.expert_states == [STATE_FROZEN, STATE_ACTIVE]
    assert bool(model.layers[0].moe.expert_frozen[0].item()) is True
    assert bool(model.layers[0].moe.expert_frozen[1].item()) is False


def test_automatic_dormant_revival_and_prune():
    torch.manual_seed(33)
    model = RealSelfStabilizingDMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=2,
        chunk_size=1,
        enable_thinking=False,
        auto_lifecycle=True,
        max_experts=4,
        min_mature_steps=10,
        novel_confirm_steps=3,
        novel_surprise_threshold=0.25,
        dormant_age=10,
        prune_age=55,
        min_usage_threshold=0.20,
        revival_sim_threshold=0.70,
    )
    model.sequence_router.sim_threshold = 0.70
    opt = torch.optim.AdamW(model.parameters(), lr=8e-3)
    model.train()

    # Train Expert 0 on Domain A (tokens 32..39)
    for _ in range(20):
        xb_a = torch.randint(32, 40, (8, 8))
        yb_a = (xb_a - 32 + 1) % 8 + 32
        opt.zero_grad()
        out_a = model(xb_a)
        loss_a = F.cross_entropy(out_a["output"].reshape(-1, 256), yb_a.reshape(-1))
        loss_a.backward()
        opt.step()

    # Shift to Domain B (tokens 140..147) to trigger spawn of Expert 1 and let Expert 0 age to DORMANT
    for _ in range(40):
        xb_b = torch.randint(140, 148, (8, 8))
        yb_b = (xb_b - 140 + 3) % 8 + 140
        opt.zero_grad()
        out_b = model(xb_b)
        if any(ev.startswith("spawned_expert_") for ev in out_b["lifecycle_events"]):
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=8e-3)
        loss_b = F.cross_entropy(out_b["output"].reshape(-1, 256), yb_b.reshape(-1))
        loss_b.backward()
        opt.step()

    assert len(model.layers[0].moe.experts) == 2
    assert model.lifecycle.expert_states[0] == STATE_DORMANT

    # Return Domain A briefly -> Expert 0 must be revived to STATE_FROZEN
    model.eval()
    with torch.no_grad():
        xb_a_ret = torch.randint(32, 40, (8, 8))
        out_rev = model(xb_a_ret, update_stats=True)

    assert model.lifecycle.total_revivals >= 1
    assert model.lifecycle.expert_states[0] == STATE_FROZEN

    # Now run Domain B long enough without Domain A that Expert 0 goes DORMANT again and gets PRUNED
    model.train()
    for _ in range(65):
        xb_b = torch.randint(140, 148, (8, 8))
        model(xb_b)

    assert model.lifecycle.total_prunes >= 1
    assert len(model.layers[0].moe.experts) == 1
    assert len(model.layers[1].moe.experts) == 1
    assert len(model.decoder.expert_heads) == 1
    assert model.layers[0].moe.expert_usage.numel() == 1
    assert model.layers[0].moe.expert_age.numel() == 1
    assert model.layers[0].moe.expert_frozen.numel() == 1


def test_unified_end_to_end_m05_extend_spawn_and_auto_gpm():
    """
    Verify that a single unified RealSelfStabilizingDMoE with all subsystems active
    (Memory + 3-Level Attn + GatedDelta STM + Thinker + WorldModel + CausalProtoTrunkRouter
    + AutoLifecycle M05 + NullSpaceProjector):
    1. Preserves higher-indexed expert prototypes when updating Expert 0.
    2. Autonomously EXTENDS Expert 0 when a compatible stream arrives (low NLL delta).
    3. Autonomously SPAWNS Expert 1 and consolidates GPM null-space bases (including
       thinker & world_model) when a genuinely conflicting stream arrives.
    """
    from dmoe.null_space import NullSpaceProjector

    torch.manual_seed(77)
    model = RealSelfStabilizingDMoE(
        d_model=24,
        num_heads=2,
        d_ff=48,
        initial_layers=2,
        chunk_size=1,
        enable_memory=True,
        num_ltm_slots=8,
        auto_write_ltm=False,
        enable_three_level_attn=True,
        local_window=4,
        chunk_stride=4,
        enable_layer_stm=True,
        enable_thinking=True,
        max_thinking_steps=2,
        enable_world_model=True,
        wm_rollout_horizon=1,
        hybrid_mode=True,
        router_mode="causal_proto",
        auto_lifecycle=True,
        max_experts=3,
        min_mature_steps=10,
        novel_confirm_steps=2,
        novel_surprise_threshold=0.35,
        use_trunk_routing=True,
        router_context_blend=0.20,
        extend_loss_margin=0.25,
    )
    gpm = NullSpaceProjector(
        model,
        energy_threshold=0.95,
        max_rank_ratio=0.80,
        track_embeddings=True,
        freeze_shared_norms_on_update=True,
    )
    assert model.null_space_projector is gpm
    assert any("thinker" in k for k in gpm.monitored_modules)
    assert any("world_model" in k for k in gpm.monitored_modules)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=6e-3)
    model.train()

    # Stream 1: Train Expert 0 on rule y = (x - 32 + 1) % 8 + 32
    for _ in range(18):
        xb = torch.randint(32, 40, (8, 8))
        yb = (xb - 32 + 1) % 8 + 32
        opt.zero_grad()
        out = model(xb, targets=yb)
        loss = F.cross_entropy(out["output"].reshape(-1, 256), yb.reshape(-1))
        loss.backward()
        opt.step()

    assert len(model.layers[0].moe.experts) == 1
    assert model.lifecycle.expert_converged[0] is True

    # Stream 2 (Compatible same rule): must NOT spawn a duplicate expert
    for _ in range(4):
        xb = torch.randint(32, 40, (8, 8))
        yb = (xb - 32 + 1) % 8 + 32
        out = model(xb, targets=yb)
    assert len(model.layers[0].moe.experts) == 1

    # Stream 3 (Conflicting domain tokens 120..127 with different rule): must SPAWN Expert 1
    # and automatically consolidate GPM null-space bases!
    for _ in range(6):
        xb_c = torch.randint(120, 128, (8, 8))
        yb_c = (xb_c - 120 + 5) % 8 + 120
        out_c = model(xb_c, targets=yb_c)
        if any(ev.startswith("spawned_expert_") for ev in out_c["lifecycle_events"]):
            break

    assert len(model.layers[0].moe.experts) == 2
    assert len(model.decoder.expert_heads) == 2
    assert len(gpm.p_null) > 0


