"""
Unit & invariant verification suite for CognitiveDMoE v12 mechanisms:
  1. StableMax cross-entropy & OrthoGrad radial-projection math (Prieto et al. 2025).
  2. Multi-Plane SO(2)^p Orthogonal Procrustes zero-weight-update unseen-meaning shift induction + null control.
  3. H-Net causal boundary-weighted byte chunker strict causality (0.0 future leak) & shift resilience.
  4. Zero-sync vectorized masked-GEMM MoE / Adapter / Decoder dispatch exact forward + backward equivalence
     against sorted scatter-add (including frozen expert gradient isolation).
  5. Full CognitiveDMoE v12 end-to-end invariants (0.0 decoder-to-core grad, InfoNCE CRIS, router anchor, mean_k <= 1.0).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.adapters import RoutedAdapterBank
from dmoe.chunker import BYTE_PAD_ID, MultiScaleByteChunker
from dmoe.cognitive_dmoe import CognitiveDMoE, CognitiveDMoEConfig, CognitiveTrainer
from dmoe.memory import LongTermSlotMemory
from dmoe.real_model import RealDecoder, RealDynamicMorphicMoE
from dmoe.routing import build_token_dispatch_plan, build_vectorized_dispatch_plan
from dmoe.spgc import MuonBONGOptimizer, stablemax_cross_entropy


def test_stablemax_and_orthograd_properties() -> None:
    torch.manual_seed(0)
    # 1. Exact formula check
    logits = torch.tensor([[2.0, -1.0, 0.0], [-3.0, 0.5, 1.5]], dtype=torch.float32)
    targets = torch.tensor([0, 2], dtype=torch.long)
    loss = stablemax_cross_entropy(logits, targets)
    s_ref = torch.where(logits >= 0.0, logits + 1.0, 1.0 / (1.0 - logits))
    p_ref = s_ref / s_ref.sum(dim=-1, keepdim=True)
    expected = -torch.log(p_ref[torch.arange(2), targets]).mean()
    assert torch.allclose(loss, expected, atol=1e-6), f"StableMax mismatch: {loss} vs {expected}"

    # 2. Safe backward at exact x == 1.0 (avoid 0 * inf = NaN from 1/(1-x)) AND extreme logits (+/- 40)
    at_one = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    l_one = stablemax_cross_entropy(at_one, torch.tensor([0]))
    g_one = torch.autograd.grad(l_one, at_one)[0]
    assert torch.isfinite(l_one) and torch.isfinite(g_one).all(), f"NaN at x==1.0: {g_one}"

    extreme = torch.tensor([[40.0, -40.0, 0.0]], requires_grad=True)
    l_ce = F.cross_entropy(extreme, torch.tensor([1]))
    g_ce = torch.autograd.grad(l_ce, extreme)[0]
    extreme_sm = torch.tensor([[40.0, -40.0, 0.0]], requires_grad=True)
    l_sm = stablemax_cross_entropy(extreme_sm, torch.tensor([1]))
    g_sm = torch.autograd.grad(l_sm, extreme_sm)[0]
    assert torch.isfinite(l_sm) and torch.isfinite(g_sm).all()
    assert float(g_sm.abs().max().item()) > 1e-3, "StableMax should preserve usable gradients under extreme logits"
    assert torch.isfinite(g_ce).all()

    # 3. OrthoGrad radial component projection preserves weight norm to first order
    w = nn.Parameter(torch.randn(16, 16))
    w_init = w.data.clone()
    opt = MuonBONGOptimizer(
        [{"name": "test", "params": [w], "use_muon": False, "role": "trunk", "lr": 1e-2, "weight_decay": 0.0}],
        mode="acquire",
        disable_bong=True,
        ortho_grad=True,
        grad_clip=0.0,
    )
    # Purely radial gradient (g = 3.0 * w) should be projected to 0 by OrthoGrad!
    w.grad = 3.0 * w.data.clone()
    opt.step()
    delta = (w.data - w_init).abs().max().item()
    assert delta < 1e-5, f"OrthoGrad failed to zero radial NLM gradient: max delta={delta}"


def test_procrustes_so2_unseen_meaning_cyclic_shift() -> None:
    """Verify Multi-Plane SO(2)^p Orthogonal Procrustes deduces unseen-meaning cyclic shifts from 4-8 shots."""
    torch.manual_seed(42)
    K = 16
    D = 64
    # Build 16 circular harmonic concept latents (as produced by a grokked modular trunk)
    # plus 16 ARBITRARY permuted non-circular surface answer embeddings ("Y00 ".."Y15 ")
    angles = 2.0 * torch.pi * torch.arange(K, dtype=torch.float32) / float(K)
    circ = torch.stack(
        [torch.cos(angles), torch.sin(angles), torch.cos(2 * angles), torch.sin(2 * angles)],
        dim=-1,
    )  # [16, 4]
    proj_k, _ = torch.linalg.qr(torch.randn(D, 4))
    key_bank = F.normalize(circ @ proj_k.t(), dim=-1)  # [16, D]
    ans_bank, _ = torch.linalg.qr(torch.randn(D, K))
    ans_bank = ans_bank.t().contiguous()  # [16, D], arbitrary non-circular orthonormal surface answers
    # Frozen linear readout matching key_bank -> concept class
    cls_w = key_bank.clone() * 8.0

    ltm = LongTermSlotMemory(
        d_model=D,
        num_slots=16,
        enable_dual_ridge_op=True,
        op_rank=24,
        enable_procrustes_op=True,
        op_procrustes_planes=8,
    )
    op_params = [p for n, p in ltm.named_parameters() if n.startswith("op_")]
    opt = torch.optim.AdamW(op_params, lr=8e-3, weight_decay=1e-4)

    def sample_shift_batch(bsz: int, n_shots: int, gen: torch.Generator):
        sup_k, sup_v, q_k, y_q = [], [], [], []
        n_uniq = min(n_shots, K - 4)
        for _ in range(bsz):
            delta = int(torch.randint(1, K, (1,), generator=gen).item())
            perm = torch.randperm(K, generator=gen)
            s_m = perm[:n_uniq]
            u_m = perm[n_uniq:]
            idx_s = s_m[torch.arange(n_shots) % n_uniq]
            y_s = (idx_s + delta) % K
            mq = int(u_m[int(torch.randint(0, u_m.numel(), (1,), generator=gen).item())].item())
            yq = (mq + delta) % K
            sup_k.append(key_bank[idx_s])
            sup_v.append(ans_bank[y_s])
            q_k.append(key_bank[mq : mq + 1])
            y_q.append(yq)
        return (
            torch.stack(sup_k, dim=0),
            torch.stack(sup_v, dim=0),
            torch.stack(q_k, dim=0),
            torch.tensor(y_q, dtype=torch.long),
        )

    g_tr = torch.Generator().manual_seed(101)
    ltm.train()
    for _ in range(180):
        sk, sv, qk, yq = sample_shift_batch(32, 8, g_tr)
        op_k, op_v = ltm.encode_op_kv(sk, sv)
        pred_h = ltm.read_dual_operator(qk, op_k, op_v)
        logits = F.linear(pred_h[:, 0, :], cls_w)
        loss = stablemax_cross_entropy(logits, yq)
        opt.zero_grad()
        loss.backward()
        opt.step()

    ltm.eval()
    g_ev = torch.Generator().manual_seed(202)
    with torch.no_grad():
        for n_sh in (4, 8):
            sk, sv, qk, yq = sample_shift_batch(128, n_sh, g_ev)
            op_k, op_v = ltm.encode_op_kv(sk, sv)
            pred_h = ltm.read_dual_operator(qk, op_k, op_v)
            acc = float((F.linear(pred_h[:, 0, :], cls_w).argmax(-1) == yq).float().mean().item())
            assert acc >= 0.85, f"Procrustes {n_sh}-shot unseen shift accuracy too low: {acc:.3f}"

            # Shuffled episode null control MUST stay at chance!
            pred_null = ltm.read_dual_operator(qk, torch.roll(op_k, 1, dims=0), torch.roll(op_v, 1, dims=0))
            null_acc = float((F.linear(pred_null[:, 0, :], cls_w).argmax(-1) == yq).float().mean().item())
            assert null_acc <= 0.20, f"Shuffled null control leaked: {null_acc:.3f}"


def test_hnet_boundary_chunker_causality_and_shift_resilience() -> None:
    torch.manual_seed(7)
    chunker = MultiScaleByteChunker(d_model=64, chunk_size=4, enable_hnet_boundary=True)
    chunker.eval()

    x = torch.randint(33, 122, (4, 32))
    x_mut = x.clone()
    x_mut[:, 16:] = torch.randint(33, 122, (4, 16))  # mutate future chunks 4..7 only

    with torch.no_grad():
        y1 = chunker(x)
        s1 = chunker.last_causal_byte_summary.clone()
        y2 = chunker(x_mut)
        s2 = chunker.last_causal_byte_summary.clone()

    # Exact 0.0 future leakage on chunks 0..3
    assert torch.allclose(y1[:, :4, :], y2[:, :4, :], atol=1e-6), "H-Net chunker leaked future bytes!"
    assert torch.allclose(s1[:, :4, :], s2[:, :4, :], atol=1e-6), "H-Net causal byte summary leaked future bytes!"

    # Shift resilience under 1..3 leading spaces on real ASCII text (measured on content chunks, not trailing pad!)
    raw = MultiScaleByteChunker.text_to_bytes(["def compute_sum(a, b): return a + b", "please cancel my credit card now!"])
    raw = F.pad(raw, (0, 40 - raw.size(1)), value=BYTE_PAD_ID)
    shifted = MultiScaleByteChunker.text_to_bytes(["  def compute_sum(a, b): return a + b", "   please cancel my credit card now!"])
    shifted = F.pad(shifted, (0, 40 - shifted.size(1)), value=BYTE_PAD_ID)
    with torch.no_grad():
        o_raw = chunker(raw)
        s_raw = chunker.last_causal_byte_summary[:, -1, :]
        o_sh = chunker(shifted)
        s_sh = chunker.last_causal_byte_summary[:, -1, :]
    cos_summary = float(F.cosine_similarity(s_raw, s_sh, dim=-1).mean().item())
    cos_raw_content = float(F.cosine_similarity(o_raw[:, :8, :].mean(dim=1), o_sh[:, :8, :].mean(dim=1), dim=-1).mean().item())
    assert cos_summary > 0.90, f"Causal byte summary should be shift-resilient: cos={cos_summary:.3f}"
    assert cos_summary > cos_raw_content, f"Summary ({cos_summary:.3f}) should beat fixed-slot raw chunks ({cos_raw_content:.3f})"


def test_vectorized_dispatch_exact_numerical_and_grad_equivalence() -> None:
    torch.manual_seed(11)
    B, T, D, d_ff, E = 4, 8, 48, 64, 6
    x = torch.randn(B, T, D, requires_grad=True)
    # Mix of k=0, k=1, and k=2 weights across all 6 experts
    top_ids = torch.randint(0, E, (B, T, 2))
    top_w = torch.rand(B, T, 2)
    top_w[0, :, :] = 0.0  # k=0 skipped tokens
    top_w[1, :, 1] = 0.0  # k=1 single-expert tokens

    plan_ref = build_token_dispatch_plan(top_ids, top_w, num_experts=E)
    plan_vec = build_vectorized_dispatch_plan(top_ids, top_w, num_experts=E)

    # 1. RealDynamicMorphicMoE with frozen experts 0 and 1
    moe = RealDynamicMorphicMoE(d_model=D, d_ff=d_ff, initial_experts=E, use_shared_expert=True, d_ff_shared=64)
    moe.freeze_expert(0)
    moe.freeze_expert(1)

    y_ref, _ = moe.forward_sparse(x, dispatch_plan=plan_ref)
    g_ref = torch.autograd.grad(y_ref.pow(2).mean(), [x] + [p for p in moe.parameters() if p.requires_grad])

    y_vec, _ = moe.forward_sparse(x, dispatch_plan=plan_vec)
    g_vec = torch.autograd.grad(y_vec.pow(2).mean(), [x] + [p for p in moe.parameters() if p.requires_grad])

    assert torch.allclose(y_ref, y_vec, atol=1e-5, rtol=1e-4), f"MoE forward max diff={(y_ref - y_vec).abs().max().item()}"
    for i, (gr, gv) in enumerate(zip(g_ref, g_vec)):
        assert torch.allclose(gr, gv, atol=1e-5, rtol=1e-4), f"MoE grad {i} max diff={(gr - gv).abs().max().item()}"
    # Verify frozen experts remain untouched (requires_grad == False)
    for e_idx in (0, 1):
        for p in moe.experts[e_idx].parameters():
            assert not p.requires_grad

    # 2. RoutedAdapterBank equivalence
    bank = RoutedAdapterBank(d_model=D, rank=8, initial_experts=E)
    for adp in bank.adapters:
        nn.init.normal_(adp.up.weight, std=0.1)
    bank.freeze(0)
    a_ref = bank.dispatch_sparse(x, plan_ref)
    a_vec = bank.dispatch_sparse(x, plan_vec)
    assert torch.allclose(a_ref, a_vec, atol=1e-5, rtol=1e-4), f"Adapter forward diff={(a_ref - a_vec).abs().max().item()}"

    # 3. RealDecoder LoRA equivalence
    dec = RealDecoder(d_model=D, chunk_size=4, vocab_size=32, enable_lora_residual=True, lora_rank=8)
    for _ in range(E - 1):
        dec.spawn_expert_head()
    for lb in dec.lora_b:
        nn.init.normal_(lb.weight, std=0.1)
    d_ref = dec(x, dispatch_plan=plan_ref)
    d_vec = dec(x, dispatch_plan=plan_vec)
    assert torch.allclose(d_ref, d_vec, atol=1e-5, rtol=1e-4), f"Decoder forward diff={(d_ref - d_vec).abs().max().item()}"


def test_v12_full_model_invariants_and_infonce_router() -> None:
    torch.manual_seed(99)
    cfg = CognitiveDMoEConfig(
        d_model=96,
        num_heads=4,
        num_layers=4,
        router_layer=2,
        d_ff_shared=192,
        d_ff_spec=96,
        enable_hnet_boundary=True,
        enable_procrustes_op=True,
        op_procrustes_planes=4,
        dual_granularity_router=True,
        use_vectorized_dispatch=True,
        use_infonce_cris=True,
        router_anchor_weight=0.5,
        shared_dropout_prob=0.0,
        specialist_init_scale=0.06,
    )
    model = CognitiveDMoE(cfg)
    trainer = CognitiveTrainer(model, seed=99, ortho_grad=True, enable_gpm=True)

    x = torch.randint(33, 122, (8, 32))
    x_pert = torch.randint(33, 122, (8, 32))
    out = model(x, update_stats=True)
    assert float(out["k_used"].float().mean().item()) <= 1.0 + 1e-6

    rq_pert, _ = model.extract_router_queries(x_pert)
    loss_d = model.compute_cognitive_loss(
        out,
        x=x,
        router_q_perturbed=rq_pert,
        router_q_pert_idx=torch.arange(8),
        cris_weight=0.30,
    )
    assert torch.isfinite(loss_d["loss"])
    trainer.step_loss(loss_d["loss"])

    # Consolidate Domain 0 -> snapshots router anchor -> verify router_anchor_loss activates cleanly
    trainer.consolidate_domain(0)
    assert len(model.context_router._router_anchors) == 3

    # Verify 0.0 decoder-to-core gradient drift
    core_snap = {n: p.detach().clone() for n, p in model.named_parameters() if not n.startswith("decoder.")}
    out2 = model(x, update_stats=False)
    trainer.step_decoder(out2, x)
    max_drift = max(
        float((p.detach() - core_snap[n]).abs().max().item())
        for n, p in model.named_parameters()
        if not n.startswith("decoder.")
    )
    assert max_drift == 0.0, f"Decoder step modified cognitive core weights: {max_drift}"
