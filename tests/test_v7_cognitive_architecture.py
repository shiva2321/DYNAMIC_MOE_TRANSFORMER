import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.lifecycle import STATE_ACTIVE, STATE_FROZEN
from dmoe.null_space import NullSpaceProjector
from dmoe.novelty import NuanceClusterBuffer
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.routing import CausalContextTokenRouter, build_token_dispatch_plan
from dmoe.spgc import MuonBONGOptimizer, classify_v7_spgc_param_groups


def _make_tiny_v7(
    *,
    initial_layers: int = 4,
    router_layer: int = 2,
    initial_experts: int = 1,
    d_model: int = 48,
    num_heads: int = 4,
    auto_lifecycle: bool = False,
    enable_memory: bool = True,
    enable_thinking: bool = True,
    enable_world_model: bool = True,
    use_multiscale_chunker: bool = True,
    detach_decoder: bool = True,
) -> RealSelfStabilizingDMoE:
    return RealSelfStabilizingDMoE(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=96,
        d_ff_shared=96,
        d_ff_spec=64,
        initial_layers=initial_layers,
        router_layer=router_layer,
        initial_experts=initial_experts,
        chunk_size=4,
        use_v7_core=True,
        use_multiscale_chunker=use_multiscale_chunker,
        router_mode="causal_context",
        d_route=32,
        router_num_keys=4,
        enable_memory=enable_memory,
        num_ltm_slots=8,
        enable_three_level_attn=True,
        local_window=4,
        chunk_stride=4,
        enable_world_model=enable_world_model,
        wm_rollout_horizon=2,
        causal_attn=True,
        use_rope=True,
        enable_layer_stm=True,
        hybrid_mode=True,
        layer_stm_every=2,
        enable_full_stack_adapters=True,
        adapter_rank=8,
        enable_thinking=enable_thinking,
        max_thinking_steps=2,
        skip_threshold=0.05,
        theta_halt=0.10,
        auto_lifecycle=auto_lifecycle,
        max_experts=5,
        min_mature_steps=4,
        nuance_cluster_min=8,
        nuance_cluster_cos=0.75,
        detach_decoder=detach_decoder,
        lora_decoder_rank=8,
    )


def test_v7_strict_chunk_causality():
    torch.manual_seed(7)
    model = _make_tiny_v7(
        initial_layers=4,
        router_layer=2,
        initial_experts=2,
        enable_memory=True,
        enable_thinking=True,
        enable_world_model=True,
    )
    model.eval()
    x1 = torch.randint(0, 256, (2, 32))  # 32 bytes -> 8 concept chunks
    x2 = x1.clone()
    t_cut_chunks = 4
    x2[:, t_cut_chunks * 4 :] = (x2[:, t_cut_chunks * 4 :] + 73) % 256

    with torch.no_grad():
        o1 = model(x1, update_stats=False)
        o2 = model(x2, update_stats=False)

    for key in ("z_expr", "z_ctx", "router_q", "d_ctx", "top_w", "latent"):
        diff = (o1[key][:, :t_cut_chunks] - o2[key][:, :t_cut_chunks]).abs().max().item()
        assert diff < 1e-5, f"Future chunk leak detected in `{key}`: max_diff={diff}"


def test_v7_gradients_frozen_keys_gpm_and_detached_renderer():
    torch.manual_seed(11)
    model = _make_tiny_v7(
        initial_layers=4,
        router_layer=2,
        initial_experts=2,
        auto_lifecycle=False,
        detach_decoder=True,
    )
    model.train()

    # Check GPM collects trunk, shared_expert, context_router.{q_proj,ctx_proj}, expr_head, ctx_head, fsp_head
    # while strictly excluding private experts, adapters, need_head, and detached decoder.
    gpm = NullSpaceProjector(model, energy_threshold=0.95, max_rank_ratio=0.80, track_embeddings=True)
    mon_keys = set(gpm.monitored_modules.keys())
    assert "layers.0.moe.shared_expert.linear1" in mon_keys
    assert "layers.2.moe.shared_expert.linear2" in mon_keys
    assert "context_router.q_proj" in mon_keys
    assert "context_router.ctx_proj" in mon_keys
    assert "expr_head" in mon_keys
    assert "ctx_head" in mon_keys
    assert "fsp_head" in mon_keys
    assert "ms_chunker.chunk_proj" in mon_keys
    for k in mon_keys:
        assert ".experts." not in k
        assert "adapters" not in k
        assert "decoder" not in k
        assert "need_head" not in k

    x = torch.randint(0, 256, (6, 24))
    y_eq = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    out = model(x, update_stats=True)
    losses = model.compute_v7_non_token_loss(
        out,
        x=x,
        equiv_labels=y_eq,
        contra_pair_idx=(torch.tensor([0, 2]), torch.tensor([1, 3])),
        nuance_targets=torch.tensor([1.0, 0.0]),
    )
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()

    assert model.context_router.q_proj.weight.grad is not None
    assert float(model.context_router.q_proj.weight.grad.abs().sum().item()) > 0.0
    assert model.context_router.ctx_proj.weight.grad is not None
    assert float(model.context_router.ctx_proj.weight.grad.abs().sum().item()) > 0.0
    assert model.context_router.need_head[0].weight.grad is not None
    assert model.expr_head.weight.grad is not None
    assert model.ctx_head.weight.grad is not None
    assert model.fsp_head.weight.grad is not None
    assert model.abstain_head.weight.grad is not None
    assert model.thinker.think_block.q_proj.weight.grad is not None
    assert model.wm_imagine_proj.weight.grad is not None
    assert model.layers[0].moe.shared_expert.linear1.weight.grad is not None
    assert model.layers[2].moe.shared_expert.linear1.weight.grad is not None
    assert model.layers[2].moe.experts[0].linear1.weight.grad is not None

    # Detached decoder must receive ZERO core gradients!
    for p in model.decoder.parameters():
        assert p.grad is None

    # Training detached renderer separately must send ZERO gradients back into core weights!
    model.zero_grad(set_to_none=True)
    ren_logits = model.render_text_detached(out)  # [B, T, 4, 256]
    ren_loss = F.cross_entropy(ren_logits.reshape(-1, 256), x.reshape(-1))
    ren_loss.backward()
    assert model.decoder.shared_base.weight.grad is not None
    assert float(model.decoder.shared_base.weight.grad.abs().sum().item()) > 0.0
    assert model.layers[2].moe.shared_expert.linear1.weight.grad is None
    assert model.layers[2].moe.experts[0].linear1.weight.grad is None
    assert model.context_router.q_proj.weight.grad is None

    # Freeze expert 0 and verify its parameters and router keys stay immutable
    keys_before = model.context_router.expert_keys[0].detach().clone()
    model.layers[2].moe.freeze_expert(0)
    model.context_router.freeze_expert(0)
    assert all(not p.requires_grad for p in model.layers[2].moe.experts[0].parameters())
    model.context_router.update_keys(
        torch.randn(4, 6, 32),
        torch.zeros(4, 6, 2, dtype=torch.long),
        torch.ones(4, 6, 2),
        expert_states=[STATE_FROZEN],
    )
    assert torch.allclose(model.context_router.expert_keys[0], keys_before, atol=1e-7)


def test_v7_dynamic_topk_nuance_escalation_and_cluster_buffer():
    torch.manual_seed(23)
    router = CausalContextTokenRouter(
        d_model=24,
        d_route=16,
        max_experts=4,
        initial_experts=2,
        num_keys=4,
        tau_skip=0.35,
        tau_top2=0.75,
        delta_mix=0.25,
        warmup_steps=0,
    )
    router.eval()
    h = torch.randn(2, 8, 24)
    q_state = router.compute_queries(h)

    # Craft deterministic scenarios across chunk positions via controlled need & probs
    probs = torch.zeros(1, 3, 2)
    probs[0, 0] = torch.tensor([0.95, 0.05])  # confident margin -> k=1 (when need=0.5)
    probs[0, 1] = torch.tensor([0.95, 0.05])  # low need -> k=0, unless nuanced!
    probs[0, 2] = torch.tensor([0.52, 0.48])  # ambiguous margin -> k=2
    q_stub = {
        "q_tok": torch.zeros(1, 3, 16),
        "need": torch.tensor([[0.50, 0.10, 0.50]]),
        "d_ctx": torch.tensor([[0.1, 0.1, 0.4]]),
        "probs": probs,
        "dormant_mask": torch.zeros(2, dtype=torch.bool),
    }
    res_clean = router.select_experts(q_stub, nuance_mask=torch.tensor([[False, False, False]]))
    assert res_clean["k_used"].tolist() == [[1, 0, 2]]
    assert float(res_clean["top_w"][0, 1].sum().item()) == 0.0

    # Escalating chunk 1 via nuance_mask forces k=2 immediately!
    res_esc = router.select_experts(q_stub, nuance_mask=torch.tensor([[False, True, False]]))
    assert res_esc["k_used"].tolist() == [[1, 2, 2]]
    assert float(res_esc["top_w"][0, 1, 0].item()) > 0.0
    assert float(res_esc["top_w"][0, 1, 1].item()) > 0.0

    # Verify NuanceClusterBuffer rejects noise and spawns on coherent clusters cleanly
    buf = NuanceClusterBuffer(cap=64, d_route=16)
    noise = F.normalize(torch.randn(18, 16), dim=-1)
    buf.push(noise)
    ready_n, *_ = buf.check_coherent_cluster(min_count=12, min_cos=0.78)
    assert not ready_n

    base = F.normalize(torch.randn(1, 16), dim=-1)
    coh = F.normalize(base.expand(16, -1) + 0.05 * torch.randn(16, 16), dim=-1)
    buf.push(coh)
    ready_y, cent, vecs, mcos, _ = buf.check_coherent_cluster(min_count=12, min_cos=0.78)
    assert ready_y and vecs is not None and mcos >= 0.78

    # End-to-end lifecycle cluster spawn on tiny v7 model
    model = _make_tiny_v7(initial_layers=4, router_layer=2, initial_experts=1, auto_lifecycle=True)
    model.train()
    for _ in range(5):
        out = model(torch.randint(0, 256, (2, 16)), update_stats=True)
    model.nuance_buffer.push(
        F.normalize(torch.ones(10, 32), dim=-1),
        err_vals=torch.ones(10) * 2.0,
        coarse_sims=torch.zeros(10),
    )
    events_all = []
    for _ in range(2):
        res_ev = model.lifecycle.step_contextual_nuance_lifecycle(
            model,
            nuance_mask=torch.ones(2, 4, dtype=torch.bool),
            q_tok=F.normalize(torch.ones(2, 4, 32), dim=-1),
            nuance_score=torch.ones(2, 4) * 2.0,
            top_ids=torch.zeros(2, 4, 2, dtype=torch.long),
            top_w=torch.ones(2, 4, 2) * 0.5,
            nuance_buffer=model.nuance_buffer,
            is_training=True,
            cluster_min_count=8,
            cluster_min_cos=0.75,
        )[1]
        events_all.extend(res_ev)
    assert any(ev.startswith("spawned_") for ev in events_all)
    assert len(model.layers[2].moe.experts) == 2
    assert len(model.layers[0].moe.experts) == 0  # shared lower trunk stays unbranched!


def test_v7_per_group_spgc_muon_and_bong_coexistence():
    torch.manual_seed(31)
    model = _make_tiny_v7(initial_layers=4, router_layer=2, initial_experts=1)
    groups = classify_v7_spgc_param_groups(
        model,
        shared_mode="consolidate",
        specialist_mode="acquire",
        lr_muon=5e-3,
        lr_adam=2e-3,
        lr_consolidate=2e-3,
    )
    by_name = {g["name"]: g for g in groups}
    assert by_name["v7_shared_muon"]["mode"] == "consolidate"
    assert by_name["v7_specialist_muon"]["mode"] == "acquire"

    opt = MuonBONGOptimizer(groups, mode="acquire", bong_prior_pull=0.05)
    # Check active trunk shared weights (`layers[0].moe.shared_expert`) vs specialist (`layers[2].moe.experts[0]`)
    trunk_p = model.layers[0].moe.shared_expert.linear1.weight
    spec_p = model.layers[2].moe.experts[0].linear1.weight
    trunk_before = trunk_p.detach().clone()
    spec_before = spec_p.detach().clone()
    prec_before = opt.precision[id(trunk_p)].detach().clone()

    x = torch.randint(0, 256, (4, 16))
    out = model(x, update_stats=True)
    losses = model.compute_v7_non_token_loss(out, equiv_labels=torch.tensor([0, 0, 1, 1]))
    opt.zero_grad()
    losses["loss"].backward()
    opt.step()

    assert not torch.allclose(trunk_p.detach(), trunk_before)
    assert not torch.allclose(spec_p.detach(), spec_before)
    # Confirm consolidate group updated its diagonal precision during step()
    assert not torch.allclose(opt.precision[id(trunk_p)], prec_before)


def test_v7_cuda_bf16_smoke():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(42)
    dev = torch.device("cuda")
    model = _make_tiny_v7(initial_layers=6, router_layer=3, initial_experts=2, d_model=64, num_heads=4).to(dev)
    gpm = NullSpaceProjector(model, energy_threshold=0.95, max_rank_ratio=0.80, defer_cov_commit=True)
    gpm.is_tracking = True
    groups = classify_v7_spgc_param_groups(model, shared_mode="acquire", specialist_mode="acquire")
    opt = MuonBONGOptimizer(groups, mode="acquire")

    x = torch.randint(0, 256, (4, 32), device=dev)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(x, update_stats=True)
        losses = model.compute_v7_non_token_loss(
            out,
            equiv_labels=torch.tensor([0, 0, 1, 1], device=dev),
            contra_pair_idx=(torch.tensor([0], device=dev), torch.tensor([2], device=dev)),
            nuance_targets=torch.tensor([1.0], device=dev),
        )
        assert torch.isfinite(losses["loss"])
        opt.zero_grad()
        losses["loss"].backward()
        gpm.commit_pending()
        saved_pre = gpm.snapshot_monitored_weights()
        gpm.update_basis_for_domain(0)
        gpm.project_gradients()
    opt.step()
    gpm.project_parameter_deltas(saved_pre)

    # Second step forces consolidation mid-step between snapshot and project_parameter_deltas
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out2 = model(x, update_stats=True)
        losses2 = model.compute_v7_non_token_loss(out2, equiv_labels=torch.tensor([0, 0, 1, 1], device=dev))
    saved_pre2 = gpm.snapshot_monitored_weights()
    opt.zero_grad()
    losses2["loss"].backward()
    gpm.commit_pending()
    gpm.update_basis_for_domain(1)
    gpm.project_gradients()
    opt.step()
    gpm.project_parameter_deltas(saved_pre2)


def test_v7_gradient_coverage_gate_I1():
    """I1 Gate: every enabled active v7 module receives non-zero gradient by step 2; detached modules do not."""
    torch.manual_seed(55)
    model = RealSelfStabilizingDMoE(
        d_model=64,
        num_heads=4,
        d_ff=96,
        d_ff_shared=96,
        d_ff_spec=64,
        initial_layers=6,
        router_layer=3,
        initial_experts=1,
        chunk_size=4,
        use_v7_core=True,
        use_multiscale_chunker=True,
        router_mode="causal_context",
        d_route=32,
        router_num_keys=4,
        enable_memory=True,
        auto_write_ltm=True,
        num_ltm_slots=8,
        enable_three_level_attn=True,
        local_window=8,
        chunk_stride=4,
        enable_world_model=True,
        wm_rollout_horizon=2,
        causal_attn=True,
        use_rope=True,
        enable_layer_stm=True,
        hybrid_mode=True,
        layer_stm_every=2,
        enable_full_stack_adapters=True,
        adapter_rank=8,
        enable_thinking=True,
        max_thinking_steps=2,
        skip_threshold=0.05,
        theta_halt=0.10,
        auto_lifecycle=False,
        detach_decoder=True,
    )
    model.train()
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for step in range(2):
        x = torch.randint(0, 256, (4, 32))
        out = model(x, update_stats=True)
        losses = model.compute_v7_non_token_loss(
            out,
            x=x,
            equiv_labels=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            contra_pair_idx=(torch.tensor([0]), torch.tensor([2])),
            nuance_targets=torch.tensor([1.0]),
        )
        opt.zero_grad()
        losses["loss"].backward()
        if step == 0:
            opt.step()

    intentional_no_grad = {
        "concept_quantizer.atoms",  # Track-1 EMA cluster codebook
        "layers.3.hier_attn.glob_fallback",  # Fallback only used when LTM has 0 written slots
        "layers.5.hier_attn.glob_fallback",
    }
    missing = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("decoder.") or name.startswith("memory.long_term.k_proj") or name.startswith("memory.long_term.v_proj") or name.startswith("memory.long_term.salience_head"):
            if name.startswith("decoder.") and p.grad is not None and float(p.grad.abs().sum()) > 0.0:
                raise AssertionError(f"Detached decoder received unexpected grad: {name}")
            continue
        if name in intentional_no_grad:
            continue
        if p.grad is None or float(p.grad.abs().sum().item()) == 0.0:
            missing.append(name)
    assert not missing, f"Active enabled v7 parameters received zero gradient: {missing}"


def test_v7_edit_families_verified_disjoint():
    import random
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from experiments.v7_edit_families import (
        HELDOUT_FAMILIES,
        TRAIN_FAMILIES,
        generate_verified_triple,
    )

    assert set(TRAIN_FAMILIES).isdisjoint(set(HELDOUT_FAMILIES))
    rng = random.Random(123)
    raw = b"The system processes valid packets in order while the threshold is >= 4 for each queue item in flight. " * 2
    seen_tr = set()
    for fam in TRAIN_FAMILIES:
        o, p, f, tag = generate_verified_triple(raw, rng, allowed_families=(fam,), seq_len=128)
        assert len(o) == 128 and len(p) == 128 and len(f) == 128
        assert o[:64] == p[:64] == f[:64]
        assert o != p and o != f
        seen_tr.add(tag.split(":")[0])
    seen_ho = set()
    for fam in HELDOUT_FAMILIES:
        o, p, f, tag = generate_verified_triple(raw, rng, allowed_families=(fam,), seq_len=128)
        assert len(o) == 128 and len(p) == 128 and len(f) == 128
        assert o[:64] == p[:64] == f[:64]
        assert o != p and o != f
        seen_ho.add(tag)
    assert seen_ho == set(HELDOUT_FAMILIES)
