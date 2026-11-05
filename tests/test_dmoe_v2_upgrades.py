import torch
import torch.nn.functional as F

from dmoe.attention import ComplexMultiHeadAttention
from dmoe.complex_ops import ComplexLinear
from dmoe.continual import ContinualLearner
from dmoe.experts import DeltaComplexLinear, DynamicMorphicMoE, WithinBatchGapMonitor
from dmoe.fast_memory import ExactLdaFastMemory, MultiSlotExactLdaMemory
from dmoe.model import SelfStabilizingDynamicMorphicMoE


def test_token_sorted_topk_dispatch_matches_reference_and_skips_inactive_experts():
    torch.manual_seed(42)
    moe = DynamicMorphicMoE(d_model=16, d_ff=32, initial_experts=6, top_k=2)
    z_r = torch.randn(3, 8, 16)
    z_i = torch.randn(3, 8, 16)

    # Force expert #5 to have very negative router bias so 0 tokens route to it
    with torch.no_grad():
        moe.router_linear.bias[5] = -100.0

    # Replace expert #5's forward with a spy that raises if called
    def forbidden_forward(*args, **kwargs):
        raise RuntimeError("Inactive expert #5 should never be executed under sorted top-k dispatch")

    moe.experts[5].forward = forbidden_forward
    out_r, out_i, entropy = moe(z_r, z_i)
    assert out_r.shape == (3, 8, 16)
    assert out_i.shape == (3, 8, 16)
    assert torch.isfinite(entropy)

    # Verify mathematical equivalence against explicit reference top-k sum
    flat_in = torch.cat([z_r, z_i], dim=-1)
    probs = F.softmax(moe.router_linear(flat_in), dim=-1)
    top_v, top_idx = torch.topk(probs, k=2, dim=-1)
    top_w = top_v / (top_v.sum(dim=-1, keepdim=True) + 1e-9)

    ref_r = torch.zeros_like(z_r)
    ref_i = torch.zeros_like(z_i)
    for slot in range(2):
        for eid in range(5):
            mask = (top_idx[:, :, slot] == eid).unsqueeze(-1).float()
            if mask.any():
                er, ei = moe.experts[eid](z_r, z_i)
                ref_r = ref_r + mask * top_w[:, :, slot : slot + 1] * er
                ref_i = ref_i + mask * top_w[:, :, slot : slot + 1] * ei

    assert torch.allclose(out_r, ref_r, atol=1e-5)
    assert torch.allclose(out_i, ref_i, atol=1e-5)


def test_delta_complex_linear_svd_compression_reduces_params_and_preserves_output():
    torch.manual_seed(7)
    base = ComplexLinear(32, 64)
    for p in base.parameters():
        p.requires_grad_(False)

    delta_layer = DeltaComplexLinear(base, mutation_std=0.02)
    # Inject a known rank-2 delta so 99% energy SVD compression is exact
    with torch.no_grad():
        u_r = torch.randn(64, 2)
        v_r = torch.randn(2, 32)
        u_i = torch.randn(64, 2)
        v_i = torch.randn(2, 32)
        delta_layer.delta_real.copy_(u_r @ v_r)
        delta_layer.delta_imag.copy_(u_i @ v_i)

    x_r = torch.randn(4, 10, 32)
    x_i = torch.randn(4, 10, 32)
    params_before = sum(p.numel() for p in delta_layer.parameters())
    before_r, before_i = delta_layer(x_r, x_i)

    delta_layer.compress_svd(energy_keep=0.999, max_rank=4)
    params_after = sum(p.numel() for p in delta_layer.parameters())
    after_r, after_i = delta_layer(x_r, x_i)

    assert params_after < params_before
    assert torch.allclose(before_r, after_r, atol=1e-4)
    assert torch.allclose(before_i, after_i, atol=1e-4)


def test_gap_monitor_and_patience_and_dual_loss_gate():
    mon = WithinBatchGapMonitor(window=3)
    assert mon.gap() is None
    mon.add(2.0, 1.5)
    mon.add(2.0, 1.5)
    mon.add(2.0, 1.5)
    assert abs(mon.gap() - 0.25) < 1e-6

    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=2, tau_split=10.0, tau_novelty=10.0, entropy_floor=0.1, spawn_patience=2
    )
    # First high-loss call builds streak=1 (does not spawn yet because spawn_patience=2)
    s1, _, _ = moe.mutate_topology(last_entropy=0.5, novelty_score=0.1, task_loss=3.0, tau_loss=2.0)
    assert not s1
    assert moe.num_experts == 2

    # Second consecutive high-loss call reaches streak=2 and spawns!
    s2, _, log2 = moe.mutate_topology(last_entropy=0.5, novelty_score=0.1, task_loss=3.0, tau_loss=2.0)
    assert s2
    assert moe.num_experts == 3
    assert "DualGate TaskLoss" in log2


def test_causal_attention_and_complex_kv_cache_autoregressive_equivalence():
    torch.manual_seed(11)
    attn = ComplexMultiHeadAttention(d_model=16, num_heads=4)
    attn.eval()

    z_r = torch.randn(2, 6, 16)
    z_i = torch.randn(2, 6, 16)

    with torch.no_grad():
        full_r, full_i = attn(z_r, z_i, causal=True)

        kv_cache = {}
        step_r_list, step_i_list = [], []
        for t in range(6):
            sr, si = attn(z_r[:, t : t + 1, :], z_i[:, t : t + 1, :], causal=True, kv_cache=kv_cache)
            step_r_list.append(sr)
            step_i_list.append(si)
        cached_r = torch.cat(step_r_list, dim=1)
        cached_i = torch.cat(step_i_list, dim=1)

    assert torch.allclose(full_r, cached_r, atol=1e-5)
    assert torch.allclose(full_i, cached_i, atol=1e-5)


def test_exact_lda_fast_memory_absorb_and_bit_exact_downdate_unlearning():
    torch.manual_seed(19)
    d_in, n_classes = 16, 12
    x_a = torch.randn(20, d_in) + 1.0
    y_a = torch.randint(0, 4, (20,))
    x_b = torch.randn(20, d_in) - 1.0
    y_b = torch.randint(4, 8, (20,))
    x_c = torch.randn(20, d_in) + 0.5
    y_c = torch.randint(8, 12, (20,))

    mem_all_then_unlearn = ExactLdaFastMemory(d_in=d_in, num_classes=n_classes, reg=0.5)
    mem_all_then_unlearn.absorb(x_a, y_a)
    mem_all_then_unlearn.absorb(x_b, y_b)
    mem_all_then_unlearn.absorb(x_c, y_c)
    mem_all_then_unlearn.downdate(x_b, y_b)

    mem_retain_only = ExactLdaFastMemory(d_in=d_in, num_classes=n_classes, reg=0.5)
    mem_retain_only.absorb(x_a, y_a)
    mem_retain_only.absorb(x_c, y_c)

    q = torch.randn(10, d_in)
    logits_unlearned = mem_all_then_unlearn.predict_logits(q)
    logits_retained = mem_retain_only.predict_logits(q)

    max_diff = (logits_unlearned.double() - logits_retained.double()).abs().max().item()
    assert max_diff < 1e-10
    assert torch.equal(logits_unlearned.argmax(dim=-1), logits_retained.argmax(dim=-1))


def test_continual_learner_with_fast_memory_and_selective_simulation_halting():
    torch.manual_seed(23)
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=1,
        initial_experts=2,
        enable_simulation_loop=True,
        max_simulation_steps=3,
        theta_converge=0.5,
        skip_threshold=-100.0,
        tokenizer_free=True,
        chunk_size=4,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    learner = ContinualLearner(
        model, opt, replay_buffer_capacity=32, enable_fast_memory=True, fast_memory_blend=0.5
    )

    x_a = torch.randint(0, 64, (8, 12), dtype=torch.long)
    y_a = (x_a + 3) % 64
    for _ in range(10):
        learner.step(x_a, y_a, output_head="byte_chunk")

    assert isinstance(learner.fast_memory, MultiSlotExactLdaMemory)
    logits = learner.predict_logits(x_a, output_head="byte_chunk")
    assert logits.shape == (8, 3, 4, 256)
    assert hasattr(model, "last_active_token_fraction")
    assert 0.0 <= model.last_active_token_fraction <= 1.0

