import random
import torch
import torch.nn.functional as F

from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import MuonBONGOptimizer, classify_v7_spgc_param_groups
from experiments.run_v8_cognitive_pilot import encode_texts_v8, roc_auc_pairs
from experiments.v8_edit_families import (
    HELDOUT_FAMILIES,
    HELDOUT_FORBIDDEN_PHRASES,
    TRAIN_FAMILIES,
    generate_i7_procedural_batch,
    generate_v7_triple,
    generate_v8_train_triple,
)


def _make_tiny_v8(
    *,
    initial_layers: int = 4,
    router_layer: int = 2,
    initial_experts: int = 2,
    d_model: int = 48,
    num_heads: int = 4,
    halt_use_energy_only: bool = True,
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
        use_v8_core=True,
        use_multiscale_chunker=True,
        router_mode="causal_context",
        d_route=32,
        router_num_keys=4,
        enable_memory=True,
        auto_write_ltm=True,
        num_ltm_slots=8,
        enable_three_level_attn=True,
        local_window=4,
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
        halt_use_energy_only=halt_use_energy_only,
        use_marginal_drop_halt=True,
        skip_final_lookahead=True,
        auto_lifecycle=False,
        max_experts=5,
        detach_decoder=detach_decoder,
        lora_decoder_rank=8,
    )


def test_v8_strict_causality_and_padding_mask_invariance():
    torch.manual_seed(101)
    model = _make_tiny_v8(initial_experts=2)
    model.eval()
    x1 = torch.randint(33, 122, (2, 32))
    x2 = x1.clone()
    t_cut = 4
    x2[:, t_cut * 4 :] = (x2[:, t_cut * 4 :] + 17) % 89 + 33

    with torch.no_grad():
        o1 = model(x1, update_stats=False)
        o2 = model(x2, update_stats=False)

    for key in ("z_expr", "z_ctx", "router_q", "d_ctx", "top_w", "latent"):
        diff = (o1[key][:, :t_cut] - o2[key][:, :t_cut]).abs().max().item()
        assert diff < 1e-5, f"Future leak in {key}: {diff}"

    # Verify BYTE_PAD_ID=257 trailing padding is masked from pooling vs different padding lengths
    s = b"why was my card declined today?"
    xp_short = encode_texts_v8([s], 48, torch.device("cpu"), pad_id=257)
    xp_long = encode_texts_v8([s], 64, torch.device("cpu"), pad_id=257)
    with torch.no_grad():
        os_ = model(xp_short, update_stats=False)
        ol_ = model(xp_long, update_stats=False)
    cos_pad = F.cosine_similarity(os_["z_expr_pool"], ol_["z_expr_pool"], dim=-1).item()
    assert cos_pad > 0.99, f"BYTE_PAD_ID masking drifted under pad extension: cos={cos_pad}"


def test_v8_relational_order_sensitivity():
    torch.manual_seed(102)
    model = _make_tiny_v8(initial_experts=1)
    model.eval()
    z = torch.randn(1, 6, 48)
    z_rev = torch.flip(z, dims=[1])
    mask = torch.ones(1, 6, dtype=torch.bool)
    _, p1, d1 = model._valid_relational_pool(z, mask=mask, use_rel=True)
    _, p2, d2 = model._valid_relational_pool(z_rev, mask=mask, use_rel=True)
    # Direction vector reverses cleanly while plain mean would collapse to zero diff!
    assert torch.allclose(d1, -d2, atol=1e-5)
    assert float((p1 - p2).abs().max().item()) > 1e-2


def test_v8_gradient_coverage_and_detached_decoder_gate_I1():
    torch.manual_seed(103)
    model = _make_tiny_v8(initial_experts=2, detach_decoder=True, halt_use_energy_only=True)
    gpm = NullSpaceProjector(model, energy_threshold=0.95, max_rank_ratio=0.80, track_embeddings=True)
    assert "rel_pool_proj" in gpm.monitored_modules
    assert "wm_ret_proj" in gpm.monitored_modules
    assert "i7_struct_head" in gpm.monitored_modules

    model.train()
    x = torch.randint(33, 122, (6, 32))
    out = model(x, update_stats=True)
    losses = model.compute_v8_non_token_loss(
        out,
        x=x,
        equiv_labels=torch.tensor([0, 0, 1, 2, 100, 100]),
        contra_pair_idx=(torch.tensor([0, 4]), torch.tensor([1, 5])),
        hard_neg_pair_idx=(torch.tensor([0, 4]), torch.tensor([1, 5])),
        nuance_targets=torch.tensor([1.0, 0.0]),
        teacher_gram=torch.eye(6),
        teacher_weight=0.25,
    )
    model.zero_grad(set_to_none=True)
    losses["loss"].backward()

    for nm_mod in (
        model.expr_proj[0].weight,
        model.rel_pool_proj.weight,
        model.wm_ret_proj.weight,
        model.i7_struct_head.weight,
        model.fsp_head.weight,
        model.abstain_head.weight,
        model.wm_imagine_proj.weight,
        model.thinker.think_block.q_proj.weight,
        model.layers[2].moe.experts[0].linear1.weight,
    ):
        assert nm_mod.grad is not None and float(nm_mod.grad.abs().sum().item()) > 0.0

    # Decoder MUST remain strictly zero-grad under cognitive core loss
    for p in model.decoder.parameters():
        assert p.grad is None

    # Training detached verbalizer MUST leak ZERO gradients back into core!
    model.zero_grad(set_to_none=True)
    rlg = model.render_text_detached(out)[:, :-1, :, :]
    dloss = F.cross_entropy(rlg.reshape(-1, 256), x[:, 4:].reshape(-1))
    dloss.backward()
    assert float(model.decoder.shared_base.weight.grad.abs().sum().item()) > 0.0
    assert model.rel_pool_proj.weight.grad is None
    assert model.thinker.think_block.q_proj.weight.grad is None


def test_v8_interchange_intervention_and_chunked_ltm_I5_I6():
    torch.manual_seed(104)
    model = _make_tiny_v8(initial_experts=2)
    model.train()
    xb = torch.randint(33, 122, (4, 24))
    xd = torch.randint(33, 122, (4, 24))
    out_cf = model.forward_interchange(xb, xd, intervene_chunks=(0, 3), dim_slice=(0, 24))
    loss_cf = out_cf["latent"][:, -1, :].pow(2).mean()
    model.zero_grad(set_to_none=True)
    loss_cf.backward()
    assert model.layers[0].q_proj.weight.grad is not None
    assert float(model.layers[0].q_proj.weight.grad.abs().sum().item()) > 0.0
    assert model.layers[2].moe.experts[0].linear1.weight.grad is not None

    # Test chunked I6 episode KV extraction on chunk_size=4 (> 1) & LTM consumption
    model.eval()
    x_ep = torch.randint(33, 122, (4, 64))
    kv = model.extract_episode_kv(x_ep, top_k=4, min_nll=0.0)
    assert kv is not None and kv["ep_keys"].shape[:2] == (4, 4)
    assert bool(kv["ep_mask"].any().item())
    o_plain = model(xb, update_stats=False)
    o_mem = model(xb, update_stats=False, episode_memory=kv)
    assert (o_plain["latent"] - o_mem["latent"]).abs().max().item() > 1e-6


def test_v8_spgc_i10_fisher_scaling_and_disjoint_families():
    torch.manual_seed(105)
    model = _make_tiny_v8(initial_experts=1)
    groups = classify_v7_spgc_param_groups(model, shared_mode="consolidate", specialist_mode="acquire", trunk_prec_scale=0.85)
    opt = MuonBONGOptimizer(groups, mode="consolidate", bong_max_prec=25.0)
    p0 = model.rel_pool_proj.weight
    p_before = opt.precision[id(p0)].clone()
    p0.grad = torch.ones_like(p0) * 0.5
    opt.step(fisher_scale=0.1)
    delta_low = (opt.precision[id(p0)] - p_before).abs().mean().item()
    opt.precision[id(p0)].copy_(p_before)
    p0.grad = torch.ones_like(p0) * 0.5
    opt.step(fisher_scale=1.5)
    delta_high = (opt.precision[id(p0)] - p_before).abs().mean().item()
    assert delta_high > delta_low

    # Verify I7 procedural + TRAIN_FAMILIES never emit HELDOUT_FAMILIES phrases
    rng = random.Random(7)
    x_i7, *_ = generate_i7_procedural_batch(32, rng, seq_len=128)
    for row in x_i7.tolist():
        s = bytes([b for b in row if b < 256]).decode("ascii", "ignore").lower()
        for bad in HELDOUT_FORBIDDEN_PHRASES:
            assert bad not in s, f"Disjointness violation in I7: {bad} in {s}"
    raw = b"alpha beta gamma delta " * 6
    for _ in range(40):
        o, p, f, fam = generate_v8_train_triple(raw[:128], rng, allowed_families=TRAIN_FAMILIES, seq_len=128)
        assert fam in TRAIN_FAMILIES and fam not in HELDOUT_FAMILIES
        for chunk_b in (o[64:], p[64:], f[64:]):
            txt = chunk_b.decode("ascii", "ignore").lower()
            if fam in ("code_alpha_var", "algebra_sympy", "unit_scale"):
                for bad in HELDOUT_FORBIDDEN_PHRASES:
                    assert bad not in txt, f"Disjointness violation in {fam}: {bad} in {txt}"

    # Verify vectorized roc_auc_pairs equals exact O(N^2) formula
    pos = [0.9, 0.4, 0.5]
    neg = [0.5, 0.2]
    brute = sum((1.0 if a > b else (0.5 if a == b else 0.0)) for a in pos for b in neg) / (len(pos) * len(neg))
    assert abs(roc_auc_pairs(pos, neg) - brute) < 1e-12
    # Quick check held-out generator produces disjoint E-F cleanly
    _, _, _, tf = generate_v7_triple(raw[:128], rng, allowed_families=HELDOUT_FAMILIES, seq_len=128)
    assert tf in HELDOUT_FAMILIES
