import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.adapters import RoutedAdapterBank
from dmoe.lifecycle import STATE_ACTIVE, STATE_FROZEN
from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.routing import CausalPrototypeRouter
from dmoe.thinking import SurpriseDrivenLatentThinker


def test_routed_adapter_bank_zero_init_freeze_prune_and_gpm_exclusion():
    torch.manual_seed(0)
    bank = RoutedAdapterBank(d_model=32, rank=8, initial_experts=1)
    x = torch.randn(4, 10, 32)
    # Zero-initialized out_proj preserves parent computation bit-for-bit
    out0 = bank(x, force_idx=0)
    assert torch.allclose(out0, torch.zeros_like(out0), atol=1e-7)

    with torch.no_grad():
        bank.adapters[0].up.weight.fill_(0.05)
    bank.freeze(0)
    assert not any(p.requires_grad for p in bank.adapters[0].parameters())

    idx1 = bank.spawn(source_idx=0, fresh_init=True)
    assert idx1 == 1
    assert torch.allclose(bank(x, force_idx=1), torch.zeros_like(x), atol=1e-7)

    with torch.no_grad():
        bank.adapters[1].up.weight.normal_(std=0.05)

    assigned = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    out_mixed = bank(x, assigned_experts=assigned)
    loss = out_mixed.pow(2).sum()
    loss.backward()
    assert bank.adapters[0].up.weight.grad is None
    assert bank.adapters[1].up.weight.grad is not None
    assert float(bank.adapters[1].up.weight.grad.abs().sum().item()) > 0.0

    # Clone-spawn preserves parent delta and prunes cleanly
    idx2 = bank.spawn(source_idx=0, fresh_init=False)
    assert idx2 == 2
    assert torch.allclose(bank(x[:1], force_idx=2), bank(x[:1], force_idx=0), atol=1e-6)
    # Verify 2D flattened sparse-mask inputs [N, D] match 3D per-token results cleanly
    x_2d = x[:, :1, :].squeeze(1)
    out_2d = bank(x_2d, assigned_experts=torch.tensor([0, 1, 0, 1], dtype=torch.long))
    assert out_2d.shape == (4, 32)
    assert bank.prune(1) is True
    assert len(bank.adapters) == 2

    # Verify full-stack adapters are excluded from GPM shared-parameter monitors
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=True,
        enable_layer_stm=True,
        enable_full_stack_adapters=True,
        adapter_rank=8,
        auto_lifecycle=False,
    )
    gpm = NullSpaceProjector(model, energy_threshold=0.95, max_rank_ratio=0.80, max_step_rank_ratio=0.25)
    for monitored in gpm.monitored_modules.keys():
        assert "stm_adapters" not in monitored
        assert "attn_adapters" not in monitored
        assert "expert_adapters" not in monitored
        assert ".experts." not in monitored
    assert len(gpm.monitored_modules) > 0


def test_null_space_step_budget_and_deferred_cov_isolation():
    torch.manual_seed(1)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=False,
        enable_world_model=False,
        auto_lifecycle=False,
    )
    gpm = NullSpaceProjector(
        model,
        energy_threshold=0.99,
        max_rank_ratio=0.80,
        max_step_rank_ratio=0.25,
        defer_cov_commit=True,
    )
    model.train()
    gpm.is_tracking = True

    # Novel/deferred batch discarded -> zero covariance leak into covariances
    x_ood = torch.randint(0, 256, (4, 16))
    _ = model(x_ood, update_stats=False)
    assert len(gpm._pending_cov) > 0
    gpm.discard_pending()
    assert len(gpm._pending_cov) == 0
    assert len(gpm.covariances) == 0

    # Committed domain updates accumulate and consolidate within per-step rank budget
    ref_layer = None
    dim = None
    max_step_cap = None
    prev_rank = 0
    ranks = []
    for dom in range(4):
        lo, hi = dom * 60, (dom + 1) * 60
        for _ in range(3):
            x_d = torch.randint(lo, hi, (4, 16))
            _ = model(x_d, update_stats=False)
            gpm.commit_pending()
        if ref_layer is None:
            ref_layer = sorted(gpm.covariances.keys())[0]
            dim = gpm.covariances[ref_layer].size(0)
            max_step_cap = max(1, int(dim * 0.25))
        gpm.update_basis_for_domain(dom)
        cur_rank = int(gpm.bases[ref_layer].size(1))
        assert cur_rank - prev_rank <= max_step_cap + 1
        assert cur_rank >= prev_rank
        ranks.append(cur_rank)
        prev_rank = cur_rank

    assert ranks[0] < ranks[-1]
    assert ranks[-1] <= int(dim * 0.80)


def test_causal_router_idf_and_multi_expert_centering():
    torch.manual_seed(2)
    emb = nn.Embedding(258, 32)
    router = CausalPrototypeRouter(
        d_model=32,
        max_protos=32,
        sim_threshold=0.55,
        use_idf_weights=True,
        center_multi_expert=True,
    )
    # 75% shared English filler prefix + 25% distinctive domain suffix
    filler = torch.full((16, 48), ord(" "), dtype=torch.long)
    filler[:, 1::2] = ord("e")
    dom_a = torch.cat([filler, torch.randint(ord("a"), ord("z") + 1, (16, 16))], dim=1)
    dom_b = torch.cat([filler, torch.randint(ord("0"), ord("9") + 1, (16, 16))], dim=1)

    for _ in range(4):
        sig_a = router.extract_signature(dom_a, emb)
        sig_b = router.extract_signature(dom_b, emb)
        router.update_prototypes(0, sig_a)
        router.update_prototypes(1, sig_b)

    # Fresh test split
    test_a = torch.cat([filler[:8], torch.randint(ord("a"), ord("z") + 1, (8, 16))], dim=1)
    test_b = torch.cat([filler[:8], torch.randint(ord("0"), ord("9") + 1, (8, 16))], dim=1)
    asg_a, _, _ = router.route_sequences(router.extract_signature(test_a, emb))
    asg_b, _, _ = router.route_sequences(router.extract_signature(test_b, emb))
    acc = 0.5 * (float((asg_a == 0).float().mean().item()) + float((asg_b == 1).float().mean().item()))
    assert acc >= 0.875
    assert router.last_proto_sims is not None and router.last_proto_sims.shape == (8, 2)
    assert router.last_thr_per_exp is not None and router.last_thr_per_exp.numel() == 2


def test_thinker_marginal_halt_energy_only_and_expert_adapters():
    torch.manual_seed(3)
    thinker = SurpriseDrivenLatentThinker(
        d_model=32,
        num_heads=4,
        d_ff=64,
        max_thinking_steps=4,
        skip_threshold=0.35,
        theta_halt=0.05,
        min_surprise_drop=0.05,
        use_marginal_drop_halt=True,
        halt_use_energy_only=True,
        skip_final_lookahead=True,
        enable_expert_adapters=True,
        adapter_rank=8,
        initial_experts=2,
    )
    dec = nn.Linear(32, 32)
    h = torch.randn(2, 8, 32)
    eids = torch.tensor([0, 1], dtype=torch.long)

    # Force low initial energy bias -> early halting strictly below max_thinking_steps (in eval mode)
    thinker.eval()
    with torch.no_grad():
        for p in thinker.energy_head.net.parameters():
            p.zero_()
        thinker.energy_head.net[-1].bias.fill_(-5.0)  # sigmoid ~ 0.0067 < skip_threshold

    out_fast = thinker(h, dec, expert_ids=eids)
    assert out_fast["steps_used"] == 0
    assert float(out_fast["token_steps"].max().item()) == 0.0

    # With moderate energy (> skip_threshold) and small marginal energy change across steps,
    # marginal-drop halting must exit early (< max_thinking_steps) instead of maxing all 4 steps
    with torch.no_grad():
        thinker.energy_head.net[-1].bias.fill_(0.0)  # sigmoid(0)=0.5 > 0.35, constant drop=0 < 0.05
    out_halt = thinker(h, dec, expert_ids=eids)
    assert 1 <= out_halt["steps_used"] < 4
    assert float(out_halt["token_steps"].max().item()) <= 2.0

    # Per-expert adapter isolation + gradients into active adapter
    thinker.train()
    thinker.freeze_expert_adapter(0)
    with torch.no_grad():
        thinker.think_block.expert_adapters.adapters[0].up.weight.fill_(0.1)
        thinker.think_block.expert_adapters.adapters[1].up.weight.normal_(std=0.05)
    w0_before = thinker.think_block.expert_adapters.adapters[0].up.weight.clone()

    out_train = thinker(h, dec, force_steps=2, expert_ids=eids)
    tgt = torch.randint(0, 32, (2, 8))
    loss = thinker.compute_thinking_loss(out_train, tgt)["loss"]
    loss.backward()
    assert thinker.think_block.expert_adapters.adapters[0].up.weight.grad is None
    assert thinker.think_block.expert_adapters.adapters[1].up.weight.grad is not None
    assert torch.allclose(thinker.think_block.expert_adapters.adapters[0].up.weight, w0_before)


def test_lifecycle_zero_contamination_and_mixed_split_mitosis():
    # Part A: Sequential zero-contamination deferral + spawn on novel streak
    torch.manual_seed(4)
    m_seq = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=False,
        enable_world_model=False,
        auto_lifecycle=True,
        max_experts=4,
        min_mature_steps=4,
        novel_confirm_steps=3,
        novel_surprise_threshold=0.95,
        extend_loss_margin=0.05,
        preserve_input_novel_on_extend=True,
        defer_on_novel_streak=True,
        router_mode="causal_proto",
        router_sim_threshold=0.75,
        router_use_idf=False,
    )
    opt = torch.optim.AdamW(m_seq.parameters(), lr=1e-2)
    m_seq.train()
    x_a = torch.full((4, 16), ord("a"), dtype=torch.long)
    y_a = torch.full((4, 16), ord("a"), dtype=torch.long)
    for _ in range(6):
        out = m_seq(x_a, update_stats=True, targets=y_a)
        loss = m_seq.compute_total_loss(out, y_a)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert m_seq.lifecycle is not None
    assert len(m_seq.layers[0].moe.experts) == 1
    m_seq.lifecycle.expert_converged[0] = True

    x_b = torch.randint(180, 250, (4, 16))
    y_b = torch.randint(180, 250, (4, 16))
    deferred_seen = False
    spawned = False
    for _ in range(5):
        _ = m_seq(x_b, update_stats=True, targets=y_b)
        if m_seq.lifecycle.should_defer_update:
            deferred_seen = True
        if any(ev.startswith("spawned_expert_1") for ev in m_seq.last_lifecycle_events):
            spawned = True
            break
    assert deferred_seen is True
    assert spawned is True
    assert len(m_seq.layers[0].moe.experts) == 2

    # Part B: Mixed-stream co-active mitosis splits without freezing parent
    torch.manual_seed(5)
    m_mix = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=False,
        enable_world_model=False,
        auto_lifecycle=True,
        max_experts=4,
        min_mature_steps=2,
        novel_confirm_steps=2,
        novel_surprise_threshold=0.99,
        router_mode="causal_proto",
        router_sim_threshold=0.20,
        router_use_idf=False,
        router_center_multi=True,
        enable_mixed_split=True,
        mixed_split_gap=0.10,
        max_coactive_experts=2,
    )
    m_mix.train()
    xa = torch.randint(10, 40, (6, 16))
    xb = torch.randint(190, 230, (6, 16))
    x_mix = torch.cat([xa, xb], dim=0)
    y_mix = x_mix.clone()
    events_all = []
    for _ in range(8):
        _ = m_mix(x_mix, update_stats=True, targets=y_mix)
        events_all.extend(m_mix.last_lifecycle_events)
    assert any(ev.startswith("mixed_split_coactive_expert_1") for ev in events_all)
    assert m_mix.lifecycle is not None
    assert m_mix.lifecycle.expert_states == [STATE_ACTIVE, STATE_ACTIVE]
    # Next forward routes the two sub-clusters to distinct co-active experts simultaneously
    out_next = m_mix(x_mix, update_stats=False)
    asg = out_next["assigned_experts"]
    assert asg[:6].mode().values.item() != asg[6:].mode().values.item()


def test_pc_adaptive_rel_contradiction_selective_firing():
    torch.manual_seed(6)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_thinking=True,
        max_thinking_steps=1,
        enable_world_model=True,
        wm_rollout_horizon=1,
        enable_pc_rectification=True,
        pc_mode="adaptive_rel",
        pc_thresh=0.15,
        auto_lifecycle=False,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    pat = torch.tensor([list(range(65, 73)) * 4], dtype=torch.long)  # b"ABCDEFGH" * 4 -> len 32
    x = pat[:, :-1]
    y = pat[:, 1:]
    model.train()
    for _ in range(30):
        out = model(x, update_stats=True, targets=y)
        loss = model.compute_total_loss(out, y)["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        out_clean = model(x, update_stats=False)
        x_contra = x.clone()
        x_contra[:, 20] = 200  # abrupt focused contradiction inside high-confidence prefix
        out_contra = model(x_contra, update_stats=False)

    err_clean = float(out_clean["pre_pc_error"][0, 20].item())
    err_contra = float(out_contra["pre_pc_error"][0, 20].item())
    gate_clean = float(out_clean["pc_write_gate"][0, 20].item())
    gate_contra = float(out_contra["pc_write_gate"][0, 20].item())
    assert err_contra > err_clean + 0.15
    assert err_contra > 0.15
    assert gate_contra < gate_clean


def test_train_revisit_conf_margin_and_episode_kv_grad():
    torch.manual_seed(7)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        enable_memory=True,
        enable_thinking=False,
        enable_world_model=False,
        enable_full_stack_adapters=True,
        adapter_rank=8,
        route_during_train=True,
        train_revisit_conf_margin=0.12,
        router_mode="causal_proto",
        router_sim_threshold=0.40,
        router_use_idf=False,
        auto_lifecycle=True,
        max_experts=3,
        min_mature_steps=2,
        novel_confirm_steps=1,
    )
    # Seed Expert 0 prototypes on domain A, freeze Expert 0, spawn Expert 1 on domain B
    xa = torch.full((4, 16), ord("a"), dtype=torch.long)
    xb = torch.full((4, 16), ord("b"), dtype=torch.long)
    assert model.causal_router is not None
    assert model.lifecycle is not None
    sig_a = model.causal_router.extract_signature(xa, model.byte_embed)
    model.causal_router.update_prototypes(0, sig_a)
    for _ in range(4):
        model.causal_router.update_prototypes(0, sig_a)
    for layer in model.layers:
        layer.moe.freeze_expert(0)
        layer.freeze_expert_adapters(0)
        layer.moe.spawn_expert(source_idx=0, fresh_init=True)
        layer.spawn_expert_adapters(source_idx=0, fresh_init=True)
    model.decoder.freeze_expert_head(0)
    model.decoder.spawn_expert_head()
    model.lifecycle.sync_num_experts(2, model.layers[0].moe.expert_frozen)
    model.lifecycle.expert_states = [STATE_FROZEN, STATE_ACTIVE]
    model.lifecycle.cooldown_remaining = 0
    model.lifecycle.novel_streak = 0

    sig_b = model.causal_router.extract_signature(xb, model.byte_embed)
    for _ in range(5):
        model.causal_router.update_prototypes(1, sig_b)

    # Clear Domain B batch must stay on active Expert 1 (not stolen by Expert 0)
    model.train()
    out_b = model(xb, update_stats=False)
    assert bool((out_b["assigned_experts"] == 1).all().item())

    # Exact Domain A revisit batch with high confidence beats Expert 1 and routes to Expert 0
    out_a = model(xa, update_stats=False)
    assert bool((out_a["assigned_experts"] == 0).all().item())

    # Differentiable intra-window episodic KV training produces LTM gradients
    x_full = torch.randint(0, 256, (2, 32))
    ep_mem = model.extract_episode_kv(x_full[:, :16], top_k=4, min_nll=0.0)
    assert ep_mem is not None
    out_ep = model(x_full[:, 16:-1], update_stats=False, episode_memory=ep_mem, targets=x_full[:, 17:])
    loss_ep = F.cross_entropy(out_ep["output"].reshape(-1, 256), x_full[:, 17:].reshape(-1))
    model.zero_grad()
    loss_ep.backward()
    ltm_grad = sum(
        float(p.grad.abs().sum().item())
        for p in model.memory.long_term.parameters()
        if p.grad is not None
    )
    assert ltm_grad > 0.0

    # Verify vectorized mixed-batch MoE + Decoder keeps frozen expert/head grads strictly None
    for layer in model.layers:
        layer.moe.freeze_expert(0)
    model.decoder.freeze_expert_head(0)
    x_mix = torch.cat([xa[:2], xb[:2]], dim=0)
    asg = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    for layer in model.layers:
        layer.moe.assigned_experts = asg
        layer.moe.force_routed_expert = None
    h_mix = torch.randn(4, 16, 32)
    h_m_out, _ = model.layers[1].forward_moe(h_mix)
    lg_mix = model.decoder(h_m_out, assigned_experts=asg)
    model.zero_grad()
    lg_mix.pow(2).sum().backward()
    assert model.layers[1].moe.experts[0].linear1.weight.grad is None
    assert model.layers[1].moe.experts[1].linear1.weight.grad is not None
    assert float(model.layers[1].moe.experts[1].linear1.weight.grad.abs().sum().item()) > 0.0
    assert model.decoder.expert_heads[0].weight.grad is None
    assert model.decoder.expert_heads[1].weight.grad is not None


def test_null_space_batched_projection_matches_reference():
    torch.manual_seed(11)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=3,
        initial_experts=1,
        chunk_size=1,
        enable_memory=True,
        enable_three_level_attn=True,
        enable_layer_stm=True,
        hybrid_mode=True,
        enable_thinking=True,
        max_thinking_steps=1,
        enable_world_model=True,
        wm_rollout_horizon=1,
        enable_full_stack_adapters=True,
        adapter_rank=8,
        auto_lifecycle=False,
    )
    gpm = NullSpaceProjector(model, energy_threshold=0.95, max_rank_ratio=0.80, defer_cov_commit=True)
    model.train()
    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))
    out = model(x, update_stats=False, targets=y)
    gpm.commit_pending()
    gpm.update_basis_for_domain(0)
    assert len(gpm.p_null) > 0

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    opt.zero_grad(set_to_none=True)
    out2 = model(x, update_stats=False, targets=y)
    loss2 = model.compute_total_loss(out2, y)["loss"]
    loss2.backward()

    ref_grads = {
        name: gpm._project_right(mod.weight.grad.detach().clone(), name)
        for name, mod in gpm.monitored_modules.items()
        if name in gpm.p_null and hasattr(mod, "weight") and mod.weight.grad is not None
    }
    gpm.project_gradients()
    for name, ref_g in ref_grads.items():
        diff = (gpm.monitored_modules[name].weight.grad - ref_g).abs().max().item()
        assert diff <= 1e-5, f"grad mismatch {name}: {diff}"

    saved = gpm.snapshot_monitored_weights()
    ref_w0 = {
        name: mod.weight.detach().clone()
        for name, mod in gpm.monitored_modules.items()
        if name in gpm.p_null and hasattr(mod, "weight")
    }
    opt.step()
    ref_proj = {
        name: w0 + gpm._project_right(gpm.monitored_modules[name].weight.detach() - w0, name)
        for name, w0 in ref_w0.items()
    }
    gpm.project_parameter_deltas(saved)
    for name, ref_w in ref_proj.items():
        diff = (gpm.monitored_modules[name].weight - ref_w).abs().max().item()
        assert diff <= 1e-5, f"delta mismatch {name}: {diff}"


def test_amp_bfloat16_safe_and_tier1_tier2_cadence():
    torch.manual_seed(12)
    cfg = dict(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=6,
        initial_experts=1,
        chunk_size=1,
        enable_memory=True,
        enable_three_level_attn=True,
        enable_layer_stm=True,
        hybrid_mode=True,
        enable_thinking=True,
        max_thinking_steps=2,
        enable_world_model=True,
        wm_rollout_horizon=1,
        enable_pc_rectification=True,
        pc_mode="adaptive_rel",
        enable_full_stack_adapters=True,
        adapter_rank=8,
        auto_lifecycle=False,
    )
    m_all = RealSelfStabilizingDMoE(**cfg)
    assert [i for i, l in enumerate(m_all.layers) if l.layer_stm is not None] == [0, 1, 2, 4]
    m_e2 = RealSelfStabilizingDMoE(**dict(cfg, layer_stm_every=2))
    assert [i for i, l in enumerate(m_e2.layers) if l.layer_stm is not None] == [0, 2, 4]
    m_e4 = RealSelfStabilizingDMoE(**dict(cfg, layer_stm_every=4))
    assert [i for i, l in enumerate(m_e4.layers) if l.layer_stm is not None] == [0, 4]

    # Force vs uniform equivalence with 2 experts
    for layer in m_e4.layers:
        layer.moe.spawn_expert(source_idx=0, fresh_init=True)
        layer.spawn_expert_adapters(source_idx=0, fresh_init=True)
    m_e4.decoder.spawn_expert_head()
    m_e4.thinker.spawn_expert_adapter(source_idx=0, fresh_init=True)
    m_e4.eval()
    x = torch.randint(0, 256, (4, 16))
    y = torch.randint(0, 256, (4, 16))
    with torch.no_grad():
        out_f = m_e4(x, update_stats=False, force_expert=0)["output"]
        for l in m_e4.layers:
            l.moe.force_routed_expert = None
        out_u = m_e4(x, update_stats=False)["output"]
    assert torch.allclose(out_f, out_u, atol=1e-5)

    # bfloat16 autocast train forward+backward+GPM executes cleanly without runtime error
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_amp = RealSelfStabilizingDMoE(**dict(cfg, layer_stm_every=4)).to(dev)
    gpm = NullSpaceProjector(m_amp, energy_threshold=0.95, max_rank_ratio=0.80, defer_cov_commit=True)
    m_amp.train()
    xi = x.to(dev)
    yi = y.to(dev)
    for step_i in range(2):
        gpm.is_tracking = True
        with torch.amp.autocast(dev, dtype=torch.bfloat16):
            o = m_amp(xi, update_stats=False, targets=yi)
            loss = m_amp.compute_total_loss(o, yi)["loss"]
        loss.backward()
        gpm.commit_pending()
        if step_i == 0:
            gpm.update_basis_for_domain(0)
        gpm.project_gradients()
        m_amp.zero_grad(set_to_none=True)
