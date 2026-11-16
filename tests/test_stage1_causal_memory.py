import torch
import torch.nn.functional as F

from dmoe.memory import GatedDeltaWorkingMemory, LongTermSlotMemory
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.stage1_baselines import CausalDenseRoPEBaseline, GRUBaseline


def _measure_causality_diff(
    model: torch.nn.Module, seq_len: int = 32, cut: int = 16, vocab: int = 64
) -> tuple[float, float]:
    model.eval()
    torch.manual_seed(123)
    x1 = torch.randint(32, 32 + vocab, (2, seq_len))
    x2 = x1.clone()
    x2[:, cut:] = torch.randint(32, 32 + vocab, (2, seq_len - cut))

    with torch.no_grad():
        y1 = model(x1)["output"].reshape(2, seq_len, -1)
        y2 = model(x2)["output"].reshape(2, seq_len, -1)

    prefix_diff = (y1[:, :cut, :] - y2[:, :cut, :]).abs().max().item()
    suffix_diff = (y1[:, cut:, :] - y2[:, cut:, :]).abs().max().item()
    return prefix_diff, suffix_diff


def test_stage1_all_models_strictly_causal():
    torch.manual_seed(7)
    cfg = dict(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        initial_experts=1,
        chunk_size=1,
        causal_attn=True,
        use_rope=True,
        enable_layer_stm=True,
        signed_beta=True,
    )
    m_flat = RealSelfStabilizingDMoE(**cfg, enable_memory=False, enable_three_level_attn=False)
    m_cog = RealSelfStabilizingDMoE(
        **cfg,
        enable_memory=True,
        num_ltm_slots=16,
        enable_three_level_attn=True,
        local_window=4,
        chunk_stride=4,
    )
    # Even when LTM slots already contain prior-episode vectors, intra-sequence causality holds
    m_cog.memory.long_term.write_kv(torch.randn(4, 32), torch.randn(4, 32))
    m_dense = CausalDenseRoPEBaseline(d_model=32, num_heads=4, d_ff=64, num_layers=2)
    m_gru = GRUBaseline(d_model=32, num_layers=2)

    for name, mod in [("m_flat", m_flat), ("m_cog", m_cog), ("m_dense", m_dense), ("m_gru", m_gru)]:
        prefix_diff, suffix_diff = _measure_causality_diff(mod)
        assert prefix_diff <= 1e-5, f"{name}: Future token leaked into prefix: max_diff={prefix_diff}"
        assert suffix_diff > 1e-3, f"{name}: Model ignored perturbed suffix: max_diff={suffix_diff}"


def test_gated_delta_memory_contraction_and_null_silence():
    torch.manual_seed(19)
    B, L, D = 2, 512, 32
    mem = GatedDeltaWorkingMemory(d_model=D, num_heads=4, signed_beta=True)
    mem.eval()

    x = torch.randn(B, L, D)
    with torch.no_grad():
        out, state = mem(x)
        zero_out, _ = mem(torch.zeros(B, 16, D))

    assert torch.isfinite(out).all()
    assert torch.isfinite(state).all()
    assert state.abs().max().item() < 200.0
    # Unretrieved / zero input should remain quiet rather than LayerNorm-amplifying zeros
    assert zero_out.abs().max().item() < 0.25


def test_gated_delta_memory_signed_eigenvalue_tracking():
    # Single-step Householder reflection: S_t = (I - beta * k k^T) S_{t-1} can reach -S_{t-1}
    # when signed_beta=True allows beta up to ~2.0, enabling odd/even parity sign inversion,
    # whereas signed_beta=False (beta <= 1.0) cannot invert stored sign in one step.
    torch.manual_seed(23)
    D, H = 16, 2
    mem_signed = GatedDeltaWorkingMemory(d_model=D, num_heads=H, conv_kernel=1, signed_beta=True)
    mem_pos = GatedDeltaWorkingMemory(d_model=D, num_heads=H, conv_kernel=1, signed_beta=False)
    mem_pos.load_state_dict(mem_signed.state_dict())

    with torch.no_grad():
        # Force high forget retention (alpha ~ 1) and peak step rate (beta -> 2 vs 1)
        mem_signed.w_alpha.bias.fill_(8.0)
        mem_pos.w_alpha.bias.fill_(8.0)
        mem_signed.w_beta.weight.zero_()
        mem_pos.w_beta.weight.zero_()
        mem_signed.w_beta.bias.fill_(6.0)
        mem_pos.w_beta.bias.fill_(6.0)
        # Identity-like out/v weights and matching x so step 1 carries zero new v along k
        x_cue = torch.randn(1, 1, D)
        # Step 0 has residual contribution, Step 1 uses zero value input after key normalization?
        # Simpler: inspect state transition directly from known non-zero init_state under same cue!
        init_S = torch.randn(1, H, D // H, D // H)
        mem_signed.w_v.weight.zero_()
        mem_pos.w_v.weight.zero_()

    # When w_v == 0, v_t == 0, so S_1 = a * (I - b k k^T) S_0 -> negative projection iff b > 1!
    _, s_signed = mem_signed(x_cue, init_state=init_S)
    _, s_pos = mem_pos(x_cue, init_state=init_S)

    loss = s_signed.pow(2).mean()
    loss.backward()
    assert mem_signed.w_beta.weight.grad is not None

    with torch.no_grad():
        # Extract active normalized k direction used at t=0
        k_vec = F.normalize(
            F.silu(mem_signed.conv_k(mem_signed.w_k(x_cue))).view(1, H, D // H),
            p=2, dim=-1, eps=1e-6,
        )
        before = torch.einsum("bhij,bhj->bhi", init_S, k_vec)
        after_s = torch.einsum("bhij,bhj->bhi", s_signed, k_vec)
        after_p = torch.einsum("bhij,bhj->bhi", s_pos, k_vec)
        cos_s = (before * after_s).sum()
        cos_p = (before * after_p).sum()
        assert cos_s.item() < -0.5 * before.pow(2).sum().item(), f"Expected sign flip, got {cos_s.item()}"
        assert cos_p.item() >= -1e-4, f"Positive-only beta should not flip sign, got {cos_p.item()}"


def test_ltm_null_sink_invariance_and_no_auto_bleed():
    torch.manual_seed(31)
    ltm = LongTermSlotMemory(d_model=32, num_slots=64)
    ltm.eval()

    # Fixed query direction orthogonal to stored random noise slots
    q_lat = torch.randn(1, 8, 32)
    with torch.no_grad():
        empty_out, empty_p = ltm.read(q_lat)
        assert empty_out.abs().max().item() == 0.0
        assert empty_p.numel() == 0

        # Populate M=1 vs M=64 orthogonal slots
        ltm.write_kv(torch.randn(1, 32), torch.ones(1, 32))
        _, p_1 = ltm.read(q_lat)
        ltm.write_kv(torch.randn(63, 32), torch.ones(63, 32))
        _, p_64 = ltm.read(q_lat)

    mass_1 = float(p_1.sum(dim=-1).mean().item())
    mass_64 = float(p_64.sum(dim=-1).mean().item())
    assert mass_1 < 0.15, f"M=1 noise mass too high: {mass_1}"
    assert mass_64 < 0.25, f"M=64 noise mass exploded with slot count: {mass_64}"

    # Verify default RealSelfStabilizingDMoE does NOT silently leak training batches into LTM
    model = RealSelfStabilizingDMoE(
        d_model=24, num_heads=4, d_ff=48, initial_layers=1, chunk_size=1,
        enable_memory=True, num_ltm_slots=16, use_rope=True, auto_write_ltm=False,
    )
    model.train()
    x_tr = torch.randint(32, 64, (4, 16))
    _ = model(x_tr)
    assert int(model.memory.long_term.slot_used.sum().item()) == 0


def test_rope_arbitrary_length_above_64():
    torch.manual_seed(41)
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=64,
        initial_layers=2,
        chunk_size=1,
        enable_memory=True,
        enable_three_level_attn=True,
        enable_layer_stm=True,
        local_window=4,
        chunk_stride=4,
        use_rope=True,
    )
    model.eval()
    x_long = torch.randint(32, 64, (2, 128))
    with torch.no_grad():
        out = model(x_long)["output"]
    assert out.shape == (2, 128, 1, 256)
    assert torch.isfinite(out).all()


def test_stage1_baselines_learn_simple_causal_task():
    # Unit test baselines on a trivial 1-step copy delay task to verify optimization & wiring.
    torch.manual_seed(55)
    for cls, kwargs in [
        (CausalDenseRoPEBaseline, dict(d_model=32, num_heads=4, d_ff=64, num_layers=2)),
        (GRUBaseline, dict(d_model=32, num_layers=2)),
    ]:
        model = cls(**kwargs)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(120):
            x = torch.randint(32, 48, (16, 16))
            logits = model(x)["output"][:, 1:, :]
            targets = x[:, :-1]  # predict delayed previous token (purely causal)
            loss = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            x_te = torch.randint(32, 48, (32, 16))
            preds = model(x_te)["output"][:, 1:, :].argmax(dim=-1)
            acc = float((preds == x_te[:, :-1]).float().mean().item())
        assert acc > 0.90, f"{cls.__name__} failed sanity learnability check: acc={acc:.3f}"


def test_cross_episode_ltm_gradient_flow_and_consolidation():
    torch.manual_seed(77)
    model = RealSelfStabilizingDMoE(
        d_model=24,
        num_heads=4,
        d_ff=48,
        initial_layers=1,
        chunk_size=1,
        enable_memory=True,
        num_ltm_slots=16,
        enable_three_level_attn=True,
        use_rope=True,
        auto_write_ltm=False,
    )
    model.train()
    x_ep1 = torch.randint(32, 64, (2, 10))
    x_ep2 = torch.randint(32, 64, (2, 10))

    ep_mem = model.extract_episode_kv(x_ep1, top_k=4, min_nll=0.0)
    assert ep_mem is not None
    out = model(x_ep2, update_stats=False, episode_memory=ep_mem)
    loss = out["output"].pow(2).mean()
    loss.backward()

    ltm = model.memory.long_term
    assert ltm.k_proj.weight.grad is not None and ltm.k_proj.weight.grad.abs().sum().item() > 0
    assert ltm.v_proj.weight.grad is not None and ltm.v_proj.weight.grad.abs().sum().item() > 0
    assert ltm.q_proj.weight.grad is not None and ltm.q_proj.weight.grad.abs().sum().item() > 0
    assert ltm.out_proj.weight.grad is not None and ltm.out_proj.weight.grad.abs().sum().item() > 0

    # Zero-grad inference consolidation + reset check
    model.zero_grad()
    written = model.consolidate_ltm(x_ep1, top_k=4, min_nll=0.0)
    assert written > 0
    assert int(ltm.slot_used.sum().item()) > 0
    model.reset_ltm()
    assert int(ltm.slot_used.sum().item()) == 0


def test_chunked_delta_scan_exact_equivalence_padded_length():
    from dmoe.memory import _chunked_gated_delta_scan

    torch.manual_seed(91)
    B, L, H, dh = 3, 19, 4, 8
    k = F.normalize(torch.randn(B, L, H, dh), dim=-1).requires_grad_(True)
    q = F.normalize(torch.randn(B, L, H, dh), dim=-1).requires_grad_(True)
    v = torch.randn(B, L, H, dh, requires_grad=True)
    a = torch.sigmoid(torch.randn(B, L, H) + 2.5).requires_grad_(True)
    b = (2.0 * torch.sigmoid(torch.randn(B, L, H))).requires_grad_(True)
    init_S = torch.randn(B, H, dh, dh, requires_grad=True)

    # Sequential reference
    S_ref = init_S
    outs_ref = []
    for t in range(L):
        kt = k[:, t].unsqueeze(-1)
        qt = q[:, t].unsqueeze(-1)
        at = a[:, t].view(B, H, 1, 1)
        bt = b[:, t].view(B, H, 1, 1)
        v_old = torch.matmul(S_ref, kt).squeeze(-1)
        err = v[:, t] - at.squeeze(-1) * v_old
        S_ref = at * S_ref + bt * torch.matmul(err.unsqueeze(-1), kt.transpose(-2, -1))
        outs_ref.append(torch.matmul(S_ref, qt).squeeze(-1))
    y_ref = torch.stack(outs_ref, dim=1)
    grads_ref = torch.autograd.grad(y_ref.sum() + S_ref.sum(), (k, q, v, a, b, init_S), retain_graph=True)

    y_chk, S_chk = _chunked_gated_delta_scan(k, q, v, a, b, init_state=init_S, chunk_size=16)
    grads_chk = torch.autograd.grad(y_chk.sum() + S_chk.sum(), (k, q, v, a, b, init_S))

    assert (y_ref - y_chk).abs().max().item() < 1e-5
    assert (S_ref - S_chk).abs().max().item() < 1e-5
    for g1, g2 in zip(grads_ref, grads_chk):
        assert (g1 - g2).abs().max().item() < 1e-4


def test_multihead_signed_beta_heterogeneous_init():
    from dmoe.memory import GatedDeltaWorkingMemory

    mem_s = GatedDeltaWorkingMemory(d_model=48, num_heads=4, signed_beta=True)
    biases_s = mem_s.w_beta.bias.detach()
    assert bool((biases_s < 0).any() and (biases_s > 0).any()), (
        f"Expected signed_beta multi-head init spanning beta<1 and beta>1, got {biases_s.tolist()}"
    )

    mem_p = GatedDeltaWorkingMemory(d_model=48, num_heads=4, signed_beta=False)
    assert float(mem_p.w_beta.bias.detach().abs().max().item()) == 0.0


def test_delta_product_step2_causality_null_silence_and_equivalence():
    from dmoe.memory import _chunked_gated_delta_scan, _sequential_gated_delta_scan

    torch.manual_seed(83)
    mem = GatedDeltaWorkingMemory(d_model=32, num_heads=4, signed_beta=True, delta_steps=2)
    mem.eval()

    # Strict causality on GatedDeltaWorkingMemory(delta_steps=2) and full model
    model = RealSelfStabilizingDMoE(
        d_model=32,
        num_heads=4,
        d_ff=40,
        initial_layers=2,
        chunk_size=1,
        enable_memory=True,
        enable_three_level_attn=True,
        enable_layer_stm=True,
        layer_stm_delta_steps=2,
        use_rope=True,
    )
    model.eval()

    x1 = torch.randint(32, 64, (2, 24))
    x2 = x1.clone()
    x2[:, 12:] = torch.randint(64, 96, (2, 12))
    with torch.no_grad():
        o1 = model(x1)["output"][:, :12]
        o2 = model(x2)["output"][:, :12]
    assert (o1 - o2).abs().max().item() <= 1e-5

    # Null-zero silence and zero-safe RMSNorm (zero state + zero writes => exact zero output)
    with torch.no_grad():
        z_in = torch.zeros(2, 16, 32)
        out_z, _ = mem(z_in)
        assert torch.isfinite(out_z).all()
        assert out_z.abs().max().item() < 0.25

        mem_zero_v = GatedDeltaWorkingMemory(d_model=32, num_heads=4, signed_beta=True, delta_steps=2)
        mem_zero_v.eval()
        mem_zero_v.w_v.weight.zero_()
        mem_zero_v.conv_v.conv.bias.zero_()
        mem_zero_v.w_v2.weight.zero_()
        out_zv, S_zv = mem_zero_v(torch.randn(2, 10, 32))
        assert out_zv.abs().max().item() == 0.0
        assert S_zv.abs().max().item() == 0.0

    # Exact sequential vs chunked (chunk_size=32) equivalence for padded lengths (L=4 -> 8, L=19 -> 38)
    for L in (4, 19):
        seq_len = 2 * L
        k = F.normalize(torch.randn(2, seq_len, 4, 8), p=2, dim=-1, eps=1e-6)
        q = F.normalize(torch.randn(2, seq_len, 4, 8), p=2, dim=-1, eps=1e-6)
        v = torch.randn(2, seq_len, 4, 8)
        alpha = torch.sigmoid(torch.randn(2, seq_len, 4) + 3.0)
        beta = 2.0 * torch.sigmoid(torch.randn(2, seq_len, 4))
        S0 = torch.randn(2, 4, 8, 8)
        with torch.no_grad():
            o_seq, S_seq = _sequential_gated_delta_scan(k, q, v, alpha, beta, S0)
            o_chk, S_chk = _chunked_gated_delta_scan(k, q, v, alpha, beta, S0, chunk_size=32)
        assert (o_seq - o_chk).abs().max().item() < 1e-5, f"L={L} out diff={(o_seq - o_chk).abs().max().item()}"
        assert (S_seq - S_chk).abs().max().item() < 1e-5, f"L={L} S diff={(S_seq - S_chk).abs().max().item()}"

    # End-to-end forward shapes at L=4 (<=8 sequential path) and L=19 (>8 chunked path)
    with torch.no_grad():
        for L in (4, 19):
            o_m, S_m = mem(torch.randn(2, L, 32))
            assert o_m.shape == (2, L, 32)
            assert S_m.shape == (2, 4, 8, 8)
            assert torch.isfinite(o_m).all() and torch.isfinite(S_m).all()


def test_delta_product_signed_beta_head_init_and_grad_flow():
    torch.manual_seed(91)
    mem_signed = GatedDeltaWorkingMemory(d_model=64, num_heads=4, signed_beta=True, delta_steps=2)
    mem_pos = GatedDeltaWorkingMemory(d_model=64, num_heads=4, signed_beta=False, delta_steps=2)

    assert torch.allclose(mem_signed.w_alpha.bias, torch.tensor([6.0, 5.5, 6.0, 6.0]), atol=1e-6)
    assert torch.allclose(mem_signed.w_beta.bias, torch.tensor([-2.0, -1.0, 1.0, 1.5]), atol=1e-6)
    assert torch.allclose(mem_signed.w_beta2.bias, torch.tensor([-4.0, -3.5, 1.0, 1.5]), atol=1e-6)
    assert torch.allclose(mem_pos.w_beta.bias, torch.zeros(4), atol=1e-6)
    assert torch.allclose(mem_pos.w_beta2.bias, torch.zeros(4), atol=1e-6)

    for mem in (mem_signed, mem_pos):
        mem.train()
        x = torch.randn(2, 12, 64, requires_grad=True)
        init_S = torch.randn(2, 4, 16, 16)
        out, final_S = mem(x, init_state=init_S)
        loss = out.pow(2).mean() + 0.1 * final_S.pow(2).mean()
        loss.backward()

        assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum().item() > 0.0
        for pname in ("o_norm_w", "w_k.weight", "w_v.weight", "w_k2.weight", "w_v2.weight", "w_beta.weight", "w_beta2.weight", "w_alpha.weight"):
            p = mem.get_parameter(pname)
            assert p.grad is not None and torch.isfinite(p.grad).all(), f"Bad grad on {pname}"
            assert p.grad.abs().sum().item() > 0.0, f"Zero grad on {pname}"
