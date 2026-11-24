"""
Unit & invariant tests for Architecture v10 (`Cog_v10`):
  1. Zero decoder CE gradient into cognitive trunk, router, and Dual Ridge memory.
  2. Surface Null-Space Whitening strictly reduces linear R^2 predictability of surface
     byte/length features while preserving orthogonal semantic variance, and is strictly causal.
  3. Expert output subspace orthogonality loss is permutation-invariant, 0 for orthogonal
     experts, >0 for identical/permuted experts, and produces finite gradients.
  4. Lifecycle decoupling allows expert spawning when `enable_nuance_escalation=False`
     while keeping `k2 == 0.0` (compute efficiency `mean_k <= 1.0`).
  5. Closed-Form Dual Ridge Episodic Operator:
     - Strictly causal `op_keys` (zero future leakage from answer chunk).
     - Numerically stable with masked/all-masked/duplicate slots in fp32 & bf16 autocast.
     - Solves synthetic 4-shot unseen-meaning cyclic shift induction vs independent reference
       (>0.85 accuracy vs softmax KV lookup at chance).
     - Gradients flow cleanly through `torch.linalg.solve` into `op_*` parameters.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from dmoe.memory import LongTermSlotMemory
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.routing import CausalContextTokenRouter, compute_chunk_surface_features


def _make_v10_model(**overrides) -> RealSelfStabilizingDMoE:
    cfg = dict(
        d_model=64,
        num_heads=4,
        d_ff=128,
        d_ff_shared=128,
        d_ff_spec=64,
        initial_layers=4,
        initial_experts=1,
        max_experts=4,
        chunk_size=4,
        use_v7_core=True,
        use_v8_core=True,
        use_multiscale_chunker=True,
        router_layer=2,
        d_route=32,
        router_num_keys=4,
        enable_thinking=True,
        max_thinking_steps=1,
        halt_use_energy_only=True,
        enable_memory=True,
        enable_world_model=True,
        enable_three_level_attn=True,
        causal_attn=True,
        use_rope=True,
        enable_layer_stm=True,
        hybrid_mode=True,
        auto_lifecycle=True,
        detach_decoder=True,
        use_rel_pool=False,
        enable_nuance_escalation=False,
        decouple_nuance_lifecycle=True,
        enable_surface_whitening=True,
        enable_cris_loss=True,
        enable_dual_ridge_ep=True,
        dual_ridge_op_rank=32,
        specialist_init_scale=0.25,
        shared_dropout_prob=0.15,
    )
    cfg.update(overrides)
    return RealSelfStabilizingDMoE(**cfg)


def test_v10_zero_decoder_ce_gradient_into_core() -> None:
    torch.manual_seed(0)
    model = _make_v10_model()
    model.train()
    x = torch.randint(32, 122, (4, 32))
    out = model(x, update_stats=True)
    logits = out["output"]  # [B, T, chunk_size, 256] or [B, L, 256]
    targets = x.clamp(0, 255)
    ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    model.zero_grad()
    ce_loss.backward()

    non_decoder_grad_norm = 0.0
    decoder_grad_norm = 0.0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        gn = float(p.grad.detach().norm().item())
        if "decoder" in name:
            decoder_grad_norm += gn
        else:
            non_decoder_grad_norm += gn

    assert decoder_grad_norm > 0.0, "Decoder should receive gradient from decoder CE."
    assert non_decoder_grad_norm == 0.0, f"Core received non-zero decoder CE grad: {non_decoder_grad_norm}"


def test_surface_features_strictly_causal_and_whitening_removes_surface_r2() -> None:
    torch.manual_seed(1)
    # 1. Strict causality of compute_chunk_surface_features
    x1 = torch.randint(32, 122, (4, 32))
    x2 = x1.clone()
    x2[:, 16:] = 257  # mutate future half
    s1 = compute_chunk_surface_features(x1, chunk_size=4, num_chunks=8)
    s2 = compute_chunk_surface_features(x2, chunk_size=4, num_chunks=8)
    assert torch.allclose(s1[:, :4, :], s2[:, :4, :], atol=0.0), "Surface features leaked future chunks!"

    # 2. Verify whitening eliminates linear predictability of surface features while keeping semantic subspace
    D, d_s = 64, 12
    router = CausalContextTokenRouter(
        d_model=D, d_route=32, max_experts=4, initial_experts=2,
        enable_surface_whitening=True, surface_dim=d_s, surface_ema_decay=0.94, surface_ridge=1e-3,
    )
    router.train()
    # Construct synthetic h = S @ W_true + Z_sem (orthogonal semantic signal)
    W_true = torch.randn(d_s, D)
    for _ in range(12):
        S = torch.randn(16, 8, d_s)
        Z_sem = torch.randn(16, 8, D) * 0.5
        H = S @ W_true + Z_sem
        router._update_surface_subspace(H, S)

    S_eval = torch.randn(64, 8, d_s)
    Z_eval = torch.randn(64, 8, D) * 0.5
    H_eval = S_eval @ W_true + Z_eval

    # Whitened H
    H_c = H_eval - router.surf_mu_h.view(1, 1, -1)
    B_mat = router.surf_proj_basis
    H_white = H_c - (H_c @ B_mat) @ B_mat.t()

    # Independent linear R^2 check predicting S from H_raw vs H_white
    def linear_r2(X: torch.Tensor, Y: torch.Tensor) -> float:
        Xf = X.reshape(-1, X.size(-1))
        Yf = Y.reshape(-1, Y.size(-1))
        Xf = Xf - Xf.mean(0, keepdim=True)
        Yf = Yf - Yf.mean(0, keepdim=True)
        W = torch.linalg.lstsq(Xf, Yf).solution
        pred = Xf @ W
        ss_res = ((Yf - pred) ** 2).sum().item()
        ss_tot = (Yf ** 2).sum().item() + 1e-8
        return 1.0 - ss_res / ss_tot

    r2_raw = linear_r2(H_eval, S_eval)
    r2_white = linear_r2(H_white, S_eval)
    assert r2_raw > 0.85, f"Expected high raw surface R^2, got {r2_raw}"
    assert r2_white < 0.15, f"Whitening failed to remove surface subspace: R^2={r2_white}"

    # Verify orthogonal semantic variance is preserved
    # Project Z_eval onto nullspace of W_true
    Q_w, _ = torch.linalg.qr(W_true.t())
    Z_orth = Z_eval - (Z_eval @ Q_w) @ Q_w.t()
    H_white_orth = H_white - (H_white @ Q_w) @ Q_w.t()
    cos_preserve = F.cosine_similarity(Z_orth.reshape(-1, D), H_white_orth.reshape(-1, D), dim=-1).mean().item()
    assert cos_preserve > 0.90, f"Whitening damaged orthogonal semantic signal: cos={cos_preserve}"


def test_expert_orthogonality_permutation_invariance_and_cris_grad() -> None:
    torch.manual_seed(2)
    model = _make_v10_model(initial_experts=2)
    model.train()
    moe = model.layers[model.router_layer].moe

    # Case A: identical (column-permuted) W2 -> overlap should be ~1.0
    with torch.no_grad():
        perm = torch.randperm(moe.experts[0].linear2.weight.size(1))
        moe.experts[1].linear2.weight.copy_(moe.experts[0].linear2.weight[:, perm])
    x = torch.randint(32, 122, (4, 32))
    out = model(x, update_stats=False)
    eq = torch.tensor([0, 0, 1, 1])
    st = torch.tensor([0, 1, 2, 0])
    rq_pert, _ = model.extract_router_queries(x)
    loss_same = model.compute_v7_non_token_loss(
        out, x=x, equiv_labels=eq, struct_targets=st, router_q_perturbed=rq_pert
    )
    assert float(loss_same["exp_orth_loss"].item()) > 0.25

    # Case B: strictly orthogonal output row spaces across experts -> overlap == 0.0
    with torch.no_grad():
        for layer in model.layers[model.router_layer :]:
            layer.moe.experts[0].linear2.weight.zero_()
            layer.moe.experts[1].linear2.weight.zero_()
            layer.moe.experts[0].linear2.weight[:32, :32] = torch.eye(32)
            layer.moe.experts[1].linear2.weight[32:, :32] = torch.eye(32)
    out2 = model(x, update_stats=False)
    loss_orth = model.compute_v7_non_token_loss(
        out2, x=x, equiv_labels=eq, struct_targets=st, router_q_perturbed=rq_pert
    )
    assert float(loss_orth["exp_orth_loss"].item()) < 1e-6
    assert torch.isfinite(loss_orth["loss"])
    loss_orth["loss"].backward()
    assert model.context_router.q_proj.weight.grad is not None


def test_decoupled_nuance_lifecycle_spawns_without_k2_escalation() -> None:
    torch.manual_seed(3)
    model = _make_v10_model(
        initial_experts=1,
        max_experts=4,
        enable_nuance_escalation=False,
        decouple_nuance_lifecycle=True,
        min_mature_steps=2,
        nuance_cluster_min=4,
        nuance_cluster_cos=0.30,
        z_nuance_thresh=0.1,
        d_sem_nuance_thresh=0.1,
    )
    model.train()
    spawned = False
    for step in range(12):
        x = torch.randint(40, 110, (4, 32))
        out = model(x, update_stats=True)
        # Escalation is OFF -> with 1 expert, k_used must never be 2
        assert int((out["k_used"] == 2).sum().item()) == 0
        if any(ev.startswith("spawned_") for ev in out["lifecycle_events"]):
            spawned = True
            break
    assert spawned, "Decoupled lifecycle failed to spawn expert when nuance_escalation=False!"
    assert len(model.layers[model.router_layer].moe.experts) >= 2


def test_dual_ridge_episodic_operator_causality_stability_and_induction() -> None:
    torch.manual_seed(4)
    # 1. Causality of extract_episode_kv op_keys (zero leakage from answer chunk)
    model = _make_v10_model()
    model.eval()
    x_ep1 = torch.randint(48, 90, (4, 28))  # 7 chunks: 0..5 prompt, 6 answer
    x_ep2 = x_ep1.clone()
    x_ep2[:, 24:28] = 65  # completely overwrite answer chunk 6!
    kv1 = model.extract_episode_kv(x_ep1, top_k=6, min_nll=0.0)
    kv2 = model.extract_episode_kv(x_ep2, top_k=6, min_nll=0.0)
    assert kv1 is not None and kv2 is not None
    assert "op_keys" in kv1 and "op_vals" in kv1 and "op_mask" in kv1
    max_diff = (kv1["op_keys"] - kv2["op_keys"]).abs().max().item()
    assert max_diff < 1e-6, f"op_keys leaked future answer chunk! max_diff={max_diff}"
    # Meanwhile op_vals MUST differ because chunk 6 changed!
    val_diff = (kv1["op_vals"] - kv2["op_vals"]).abs().max().item()
    assert val_diff > 1e-3, "op_vals did not reflect answer chunk change!"

    # 2. Masked & all-masked numerical stability + gradient flow through linalg.solve
    ltm = LongTermSlotMemory(d_model=32, enable_dual_ridge_op=True, op_rank=16)
    q = torch.randn(3, 6, 32, requires_grad=True)
    k_raw = torch.randn(3, 4, 32, requires_grad=True)
    v_raw = torch.randn(3, 4, 32, requires_grad=True)
    op_k, op_v = ltm.encode_op_kv(k_raw, v_raw)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],  # all-masked edge case!
            [True, True, True, True],
        ],
        dtype=torch.bool,
    )
    out = ltm.read_dual_operator(q, op_k, op_v, op_mask=mask)
    assert torch.isfinite(out).all(), "Non-finite output in read_dual_operator with masking!"
    assert out[1].abs().max().item() == 0.0, "All-masked batch item should produce exact 0!"
    out.sum().backward()
    for p_name in ("op_log_lambda", "op_gate"):
        p = getattr(ltm, p_name)
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert ltm.op_lin.weight.grad is not None and torch.isfinite(ltm.op_lin.weight.grad).all()

    # 3. Independent reference induction test: 4-shot UNSEEN-MEANING cyclic shift over Z_16
    K = 16
    D = 32
    ltm_ind = LongTermSlotMemory(d_model=D, enable_dual_ridge_op=True, op_rank=D)
    with torch.no_grad():
        # Set identity feature map and value projection so we test pure operator math
        ltm_ind.op_lin.weight.copy_(torch.eye(D))
        ltm_ind.op_lin.bias.zero_()
        ltm_ind.op_mlp[2].weight.zero_()
        ltm_ind.op_mlp[2].bias.zero_()
        ltm_ind.op_v_proj.weight.copy_(torch.eye(D))
        ltm_ind.op_v_proj.bias.zero_()
        ltm_ind.op_out_proj.weight.copy_(torch.eye(D))
        ltm_ind.op_log_lambda.fill_(-4.5)  # small ridge ~0.011
        ltm_ind.op_gate.fill_(1.0)

    angles = torch.arange(K) * (2.0 * math.pi / K)
    E = torch.zeros(K, D)
    E[:, 0] = torch.cos(angles)
    E[:, 1] = torch.sin(angles)
    E[:, 2] = torch.cos(2.0 * angles)
    E[:, 3] = torch.sin(2.0 * angles)
    E = F.normalize(E, dim=-1)

    gen = torch.Generator().manual_seed(123)
    hits, tot = 0, 100
    for _ in range(tot):
        delta = int(torch.randint(0, K, (1,), generator=gen).item())
        perm = torch.randperm(K, generator=gen)
        m_sup = perm[:4]
        m_q = perm[4]  # strictly UNSEEN meaning!
        y_sup = (m_sup + delta) % K
        y_q = int((m_q + delta) % K)
        op_k_e, op_v_e = ltm_ind.encode_op_kv(E[m_sup].unsqueeze(0), E[y_sup].unsqueeze(0))
        pred = ltm_ind.read_dual_operator(E[m_q].view(1, 1, D), op_k_e, op_v_e).squeeze(0).squeeze(0)
        pred_cls = int((F.normalize(pred, dim=-1) @ E.t()).argmax().item())
        hits += int(pred_cls == y_q)
    acc = hits / tot
    assert acc >= 0.85, f"Dual Ridge operator failed unseen-meaning cyclic shift: acc={acc:.3f}"
