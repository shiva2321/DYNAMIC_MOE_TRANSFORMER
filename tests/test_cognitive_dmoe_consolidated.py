"""
Unit and invariant tests for the consolidated `CognitiveDMoE` architecture (v11).

Tests every core invariant against independent references (`research-protocol.md` §8):
  1. Zero decoder CE gradient into cognitive core + working detached byte rendering & generation.
  2. Zero dead subsystems & exact parameter accounting in `architecture_summary()`.
  3. Shift-resilient causal BlurPool chunker causality (0.0 future leak) and shift stability.
  4. Continual-safe router: `k <= 1` cap (zero biased-margin leakage), frozen-expert bias
     exemption, padding `k=0`, and content-masked causal EMA.
  5. Upgraded Dual-Head Closed-Form Ridge (`op_rank=48`, `op_rank_low=12`, MLP value head):
     strict causality + 4-shot unseen-meaning cyclic-shift induction on permuted answer encodings.
  6. `CognitiveTrainer` end-to-end continual stack: GPM + MuonBONG + Grokfast-EMA + Quarantine
     gate (<= 16 KB) + state-preserving optimizer rebuild on autonomous specialist spawn.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.chunker import BYTE_PAD_ID, MultiScaleByteChunker
from dmoe.cognitive_dmoe import CognitiveDMoE, CognitiveDMoEConfig, CognitiveTrainer
from dmoe.memory import LongTermSlotMemory
from dmoe.routing import CausalContextTokenRouter


def _small_cfg(**overrides) -> CognitiveDMoEConfig:
    base = dict(
        d_model=64,
        num_heads=4,
        num_layers=4,
        router_layer=2,
        d_ff_shared=128,
        d_ff_spec=96,
        chunk_size=4,
        d_route=32,
        router_num_keys=4,
        max_experts=4,
        initial_experts=1,
        dual_ridge_op_rank=24,
        dual_ridge_op_rank_low=8,
        max_thinking_steps=2,
        min_mature_steps=4,
        nuance_cluster_min=4,
        nuance_cluster_cos=0.30,
    )
    base.update(overrides)
    return CognitiveDMoEConfig(**base)


def test_zero_decoder_ce_grad_and_detached_generation():
    torch.manual_seed(0)
    model = CognitiveDMoE(_small_cfg())
    trainer = CognitiveTrainer(model, seed=0, decoder_lr=1e-2)

    texts = [
        "alpha beta gamma delta epsilon",
        "alpha beta gamma delta epsilon",
        "neural cognitive dynamic moe!!",
        "neural cognitive dynamic moe!!",
    ]
    x = model.encode_text(texts, seq_len=32)

    # Snapshot every cognitive core parameter before decoder training
    core_before = {
        n: p.detach().clone()
        for n, p in model.named_parameters()
        if not n.startswith("decoder.")
    }

    losses = []
    model.train()
    for _ in range(18):
        out = model(x, update_stats=False)
        loss_val = trainer.step_decoder(out, x)
        losses.append(loss_val)

    # 1. Decoder loss must genuinely decrease on real UTF-8 byte prediction
    assert losses[-1] < losses[0] - 0.5, f"Detached decoder failed to learn: {losses[0]:.3f} -> {losses[-1]:.3f}"

    # 2. ZERO gradient and ZERO weight change in the entire cognitive core
    for n, p in model.named_parameters():
        if not n.startswith("decoder."):
            if p.grad is not None:
                assert float(p.grad.abs().max().item()) == 0.0, f"Core param {n} received decoder grad!"
            delta = float((p.detach() - core_before[n]).abs().max().item())
            assert delta == 0.0, f"Core param {n} drifted by {delta} during decoder training!"

    # 3. Autoregressive generation works end-to-end and returns valid UTF-8 strings
    gen_out = model.generate(["alpha beta", "neural cog"], max_new_chunks=3, greedy=True)
    assert len(gen_out["texts"]) == 2
    assert all(isinstance(s, str) and len(s) > 0 for s in gen_out["texts"])
    assert gen_out["abstain_probs"].shape == (2, 3)
    assert torch.isfinite(gen_out["abstain_probs"]).all()
    assert gen_out["byte_ids"].size(1) == 12 + 3 * 4


def test_zero_dead_subsystems_and_architecture_summary():
    torch.manual_seed(1)
    model = CognitiveDMoE(_small_cfg())
    summary = model.architecture_summary()

    exact_total = sum(p.numel() for p in model.parameters())
    assert summary["total_params"] == exact_total
    assert summary["cognitive_core_params"] < summary["total_params"]
    assert summary["active_core_params_k1"] <= summary["cognitive_core_params"]
    assert all(v > 0 for v in summary["subsystems"].values()), f"Zero-size subsystem: {summary['subsystems']}"

    # Run cognitive + episodic + interchange + detached decoder losses and verify gradients reach every subsystem
    x = model.encode_text(
        [
            "card payment declined at atm",
            "atm rejected my card payment",
            "for i in range(10): print(i)",
            "solve x + 3 = 10 for x = 7!!",
        ],
        seq_len=32,
    )
    # Seed LTM with one slot so global memory cross-attention is live too
    model.consolidate_ltm(x[:2], top_k=2)

    model.train()
    out = model(x, update_stats=True)
    rq_pert, _ = model.extract_router_queries(x)
    loss_d = model.compute_cognitive_loss(
        out,
        x=x,
        equiv_labels=torch.tensor([0, 0, 1, 2]),
        contra_pair_idx=(torch.tensor([0]), torch.tensor([2])),
        hard_neg_pair_idx=(torch.tensor([0]), torch.tensor([3])),
        nuance_targets=torch.tensor([1.0]),
        struct_targets=torch.tensor([1, 1, 0, 2]),
        router_q_perturbed=rq_pert,
        router_q_pert_idx=torch.arange(4),
    )

    # Also exercise Dual Ridge episodic path and IIT interchange path
    ep_kv = model.extract_episode_kv(x[:2], top_k=2)
    assert ep_kv is not None and "op_keys" in ep_kv
    ep_mem = {
        k: v.unsqueeze(0).reshape(1, -1, v.size(-1)) if v.dim() == 3 else v.unsqueeze(0).reshape(1, -1)
        for k, v in ep_kv.items()
        if k in ("ep_keys", "ep_vals", "ep_mask", "op_keys", "op_vals", "op_mask")
    }
    out_ep = model(x[:1], update_stats=False, episode_memory=ep_mem)
    out_iit = model.forward_interchange(x[:2], x[2:4], intervene_chunks=(0, 3), dim_slice=(0, 32))
    dec_l = model.compute_detached_decoder_loss(out, x)

    total = loss_d["loss"] + out_ep["latent"].pow(2).mean() + out_iit["latent"].pow(2).mean() + dec_l
    total.backward()

    # Check every top-level subsystem has non-zero gradient magnitude
    grad_by_sub = {k: 0.0 for k in summary["subsystems"]}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = float(p.grad.abs().sum().item())
        if name.startswith("ms_chunker."):
            grad_by_sub["sensory_chunker"] += gn
        elif name.startswith("memory."):
            grad_by_sub["dual_memory_and_ridge"] += gn
        elif name.startswith("decoder."):
            grad_by_sub["detached_decoder"] += gn
        elif name.startswith("context_router."):
            grad_by_sub["router_and_lifecycle"] += gn
        elif name.startswith("world_model.") or name.startswith("wm_"):
            grad_by_sub["world_model_and_retrieval"] += gn
        elif name.startswith("thinker.") or name.startswith("abstain_head."):
            grad_by_sub["thinker_and_abstain"] += gn
        elif name.startswith(("expr_", "ctx_head.", "equiv_proto_bank.", "fsp_head.", "i7_struct_head.")):
            grad_by_sub["meaning_and_fsp_heads"] += gn
        elif ".moe.experts." in name or "_adapters." in name:
            grad_by_sub["specialist_moe_and_adapters"] += gn
        else:
            grad_by_sub["shared_trunk_layers"] += gn

    for sub, gn in grad_by_sub.items():
        assert gn > 0.0, f"Subsystem '{sub}' received ZERO gradient (dead module detected)!"


def test_shift_invariant_chunker_causality_and_shift_resilience():
    torch.manual_seed(2)
    ch_blur = MultiScaleByteChunker(d_model=64, chunk_size=4, shift_invariant_pool=True).eval()
    ch_raw = MultiScaleByteChunker(d_model=64, chunk_size=4, shift_invariant_pool=False).eval()
    # Share exact weights between both chunkers for a fair apples-to-apples comparison
    ch_raw.load_state_dict(
        {k: v for k, v in ch_blur.state_dict().items() if k != "blur_kernel"},
        strict=True,
    )

    # 1. Strict causality check: mutating bytes >= 16 must NOT change chunks 0..3
    x1 = MultiScaleByteChunker.text_to_bytes(["alpha beta gamma delta epsilon!!"])
    x2 = x1.clone()
    x2[:, 16:] = ord("Z")
    with torch.no_grad():
        h1 = ch_blur(x1)
        h2 = ch_blur(x2)
    max_leak = float((h1[:, :4, :] - h2[:, :4, :]).abs().max().item())
    assert max_leak == 0.0, f"Shift-invariant chunker leaked future bytes: {max_leak}"

    # 2. Shift resilience under 1..3 leading spaces (boundary shift)
    base_texts = [
        "Where is my replacement debit card delivery?",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)",
        "The derivative of sin(x) is cos(x) for all real x.",
    ]
    xb_base = MultiScaleByteChunker.text_to_bytes(base_texts)
    xb_shift = MultiScaleByteChunker.text_to_bytes(["  " + t for t in base_texts])
    L = min(xb_base.size(1), xb_shift.size(1))
    L = (L // 4) * 4
    with torch.no_grad():
        hb_raw = F.normalize(ch_raw(xb_base[:, :L]).mean(dim=1), dim=-1)
        hs_raw = F.normalize(ch_raw(xb_shift[:, :L]).mean(dim=1), dim=-1)
        cos_raw = float((hb_raw * hs_raw).sum(dim=-1).mean().item())

        hb_blur = F.normalize(ch_blur(xb_base[:, :L]).mean(dim=1), dim=-1)
        hs_blur = F.normalize(ch_blur(xb_shift[:, :L]).mean(dim=1), dim=-1)
        cos_blur = float((hb_blur * hs_blur).sum(dim=-1).mean().item())

    assert cos_blur > cos_raw, f"BlurPool did not improve 2-byte shift cosine: blur={cos_blur:.4f} vs raw={cos_raw:.4f}"


def test_continual_safe_router_k_cap_and_frozen_exempt_balance():
    torch.manual_seed(3)
    router = CausalContextTokenRouter(
        d_model=64,
        d_route=32,
        max_experts=4,
        initial_experts=3,
        delta_mix=0.0,
        tau_top2=1.05,
        use_loss_free_balance=True,
        frozen_exempt_balance=True,
        residual_ortho_router=True,
        enable_surface_whitening=True,
        surface_dim=6,
    )
    # Freeze expert 0 and keep experts 1 and 2 active
    router.freeze_expert(0)
    assert float(router.expert_bias[0].item()) == 0.0

    # Push 15 steps of synthetic traffic routed strictly to expert 1; expert 0 bias MUST stay 0.0!
    expert_states = [1, 0, 0]  # 0=FROZEN, 1=ACTIVE, 2=ACTIVE
    for _ in range(15):
        h = torch.randn(4, 8, 64)
        cm = torch.ones(4, 8, dtype=torch.bool)
        cm[:, 6:] = False  # trailing padding chunks
        q_st = router.compute_queries(h, expert_states=expert_states, content_mask=cm)
        res = router.select_experts(q_st, nuance_mask=None, content_mask=cm)

        # 1. Strict k <= 1 cap (no biased-margin k=2 leakage!)
        assert int(res["k_used"].max().item()) <= 1, f"k=2 leaked! max k={res['k_used'].max().item()}"
        # 2. Trailing padding chunks strictly get k=0
        assert int(res["k_used"][:, 6:].max().item()) == 0

        # Simulate all active tokens going to expert 1
        fake_ids = res["top_ids"].clone()
        fake_ids[..., 0] = 1
        fake_w = res["top_w"].clone()
        fake_w[..., 0] = cm.float()
        router.update_keys(res["q_tok"], fake_ids, fake_w, expert_states=expert_states)

    assert float(router.expert_bias[0].item()) == 0.0, f"Frozen expert 0 bias drifted to {router.expert_bias[0].item()}"
    # Co-active plastic experts 1 and 2 SHOULD still balance against each other
    assert float(router.expert_bias[2].item()) > float(router.expert_bias[1].item())

    # 3. Causality check on content-masked causal EMA: mutating future chunk t=7 cannot affect t=0..6
    router.eval()
    h_a = torch.randn(2, 8, 64)
    h_b = h_a.clone()
    h_b[:, 7, :] += 5.0
    cm_a = torch.tensor([[False, True, True, True, True, True, True, True]] * 2)
    with torch.no_grad():
        qa = router.compute_queries(h_a, content_mask=cm_a)["q_tok"]
        qb = router.compute_queries(h_b, content_mask=cm_a)["q_tok"]
    assert float((qa[:, :7, :] - qb[:, :7, :]).abs().max().item()) < 1e-6


def test_dual_ridge_mlp_value_head_causality_and_unseen_shift_induction():
    torch.manual_seed(4)
    # 1. Verify strict causality of extract_episode_kv on CognitiveDMoE
    model = CognitiveDMoE(_small_cfg()).eval()
    x_ep1 = model.encode_text(["Q0 C01 C02 C00 C00 ? Y05 ", "Q0 C03 C04 C00 C00 ? Y11 "], seq_len=28)
    x_ep2 = model.encode_text(["Q0 C01 C02 C00 C00 ? Y14 ", "Q0 C03 C04 C00 C00 ? Y02 "], seq_len=28)
    with torch.no_grad():
        kv1 = model.extract_episode_kv(x_ep1, top_k=4)
        kv2 = model.extract_episode_kv(x_ep2, top_k=4)
    assert kv1 is not None and kv2 is not None
    # Keys depend ONLY on prompt (bytes 0..23), NEVER on answer bytes 24..27!
    key_leak = float((kv1["op_keys"] - kv2["op_keys"]).abs().max().item())
    val_diff = float((kv1["op_vals"] - kv2["op_vals"]).abs().max().item())
    assert key_leak < 1e-6, f"op_keys leaked future answer bytes: {key_leak}"
    assert val_diff > 1e-3, "op_vals did not reflect distinct answers"

    # 2. Synthetic 4-shot unseen-meaning shift induction with permuted surface answer embeddings
    #    (proving op_v_low 2-layer MLP maps arbitrary surface answers onto circular rotation space!)
    K, D = 16, 48
    angles = torch.arange(K, dtype=torch.float32) * (2.0 * math.pi / K)
    # Clean circular key embeddings (what a grokked trunk produces)
    circ = torch.stack([torch.cos(angles), torch.sin(angles), torch.cos(2 * angles), torch.sin(2 * angles)], dim=-1)
    key_proj = torch.linalg.qr(torch.randn(D, 4))[0]
    key_table = F.normalize(circ @ key_proj.t(), dim=-1)  # [16, D]
    # Arbitrary random surface encodings for answer strings "Y00 ".."Y15 " (NOT aligned with circle!)
    ans_surface_table = F.normalize(torch.randn(K, D), dim=-1)

    ltm = LongTermSlotMemory(d_model=D, enable_dual_ridge_op=True, op_rank=24, op_rank_low=12, op_v_low_mlp=True)
    cls_head = nn.Linear(D, K)
    opt = torch.optim.AdamW(list(ltm.parameters()) + list(cls_head.parameters()), lr=6e-3)

    gen = torch.Generator().manual_seed(99)
    for _ in range(300):
        B_ep = 32
        shifts = torch.randint(1, K, (B_ep,), generator=gen)
        sup_m = torch.stack([torch.randperm(K, generator=gen)[:6] for _ in range(B_ep)], dim=0)  # [B, 6]
        unseen_m = torch.tensor(
            [
                int([c for c in range(K) if c not in set(sup_m[b].tolist())][0])
                for b in range(B_ep)
            ],
            dtype=torch.long,
        )
        sup_y = (sup_m + shifts.unsqueeze(1)) % K
        q_y = (unseen_m + shifts) % K

        raw_k = key_table[sup_m]  # [B, 6, D]
        raw_v = ans_surface_table[sup_y]  # [B, 6, D]
        op_k, op_v = ltm.encode_op_kv(raw_k, raw_v)
        q_lat = key_table[unseen_m].unsqueeze(1)  # [B, 1, D]
        pred_lat = ltm.read_dual_operator(q_lat, op_k, op_v).squeeze(1)
        loss = F.cross_entropy(cls_head(pred_lat), q_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Evaluate zero-weight-update 4-shot unseen-meaning shift vs shuffled null
    ltm.eval()
    cls_head.eval()
    with torch.no_grad():
        B_ev = 128
        shifts = torch.randint(1, K, (B_ev,), generator=gen)
        sup_m = torch.stack([torch.randperm(K, generator=gen)[:4] for _ in range(B_ev)], dim=0)
        unseen_m = torch.tensor(
            [int([c for c in range(K) if c not in set(sup_m[b].tolist())][0]) for b in range(B_ev)],
            dtype=torch.long,
        )
        sup_y = (sup_m + shifts.unsqueeze(1)) % K
        q_y = (unseen_m + shifts) % K
        op_k, op_v = ltm.encode_op_kv(key_table[sup_m], ans_surface_table[sup_y])
        q_lat = key_table[unseen_m].unsqueeze(1)
        pred_acc = float((cls_head(ltm.read_dual_operator(q_lat, op_k, op_v).squeeze(1)).argmax(-1) == q_y).float().mean().item())
        shuff_acc = float(
            (cls_head(ltm.read_dual_operator(q_lat, op_k.roll(1, 0), op_v.roll(1, 0)).squeeze(1)).argmax(-1) == q_y).float().mean().item()
        )
    assert pred_acc >= 0.85, f"Upgraded Dual Ridge 4-shot unseen shift acc too low: {pred_acc:.3f}"
    assert shuff_acc <= 0.20, f"Shuffled null exceeded chance: {shuff_acc:.3f}"


def test_cognitive_trainer_continual_spawn_gpm_quarantine():
    torch.manual_seed(5)
    cfg = _small_cfg(min_mature_steps=3, nuance_cluster_min=4, nuance_cluster_cos=0.20, z_nuance_thresh=0.1, d_sem_nuance_thresh=0.1)
    model = CognitiveDMoE(cfg)
    trainer = CognitiveTrainer(model, seed=5, grokfast_alpha=0.90, grokfast_lamb=1.0)

    x_dom0 = model.encode_text(
        ["def add(a, b): return a + b", "for x in items: total += x", "class Node: pass", "import math; math.sqrt(9)"],
        seq_len=32,
    )
    spawned = False
    for step in range(12):
        model.train()
        saved = trainer.gpm.snapshot_monitored_weights() if (trainer.gpm is not None and trainer.gpm.p_null) else None
        out = model(x_dom0, update_stats=True)
        ld = model.compute_cognitive_loss(out, x=x_dom0)
        if trainer.maybe_rebuild_on_spawn(out["lifecycle_events"]):
            spawned = True
        trainer.step_loss(ld["loss"], lifecycle_events=(), saved_weights=saved)

    assert spawned or model.layers[model.router_layer].moe.num_experts >= 2, "Lifecycle failed to spawn specialist"
    assert trainer.total_spawns >= 1

    # Consolidate domain 0 -> GPM basis populated & trunk shifted to BONG consolidate
    trainer.consolidate_domain(0)
    assert trainer.gpm is not None and len(trainer.gpm.p_null) > 0
    assert any(g.get("role") == "trunk" and g.get("mode") == "consolidate" for g in trainer.opt.param_groups)

    # Quarantine gate stays strictly within 16 KB and blocks single-source burst contradictions
    z_c = F.normalize(torch.randn(8, cfg.d_model), dim=-1)
    y_true = torch.zeros(8, dtype=torch.long)
    logits_conf = torch.zeros(8, 16)
    logits_conf[:, 0] = 6.0
    for _ in range(5):
        trainer.gate.triage_batch(x_dom0.repeat(2, 1), y_true, logits_conf, z_c, source_id="trusted", skip_familiar=False, enable_quarantine=False)
    # Single-source adversarial flip attempt
    y_adv = torch.ones(8, dtype=torch.long)
    x_act, _, flips = trainer.gate.triage_batch(x_dom0.repeat(2, 1), y_adv, logits_conf, z_c, source_id="attacker", skip_familiar=False, enable_quarantine=True)
    assert x_act is None and flips == 0
    assert trainer.gate.storage_bytes() <= 16384

