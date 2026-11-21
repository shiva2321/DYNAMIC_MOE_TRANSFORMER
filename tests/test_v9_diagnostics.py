import torch
import torch.nn.functional as F

from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.spgc import MuonBONGOptimizer, classify_v7_spgc_param_groups


def _make_tiny_v9(
    *,
    use_rel_pool: bool = True,
    enable_nuance_escalation: bool = True,
    is_dense: bool = False,
) -> RealSelfStabilizingDMoE:
    return RealSelfStabilizingDMoE(
        d_model=48,
        num_heads=4,
        d_ff=96,
        d_ff_shared=96 if not is_dense else 192,
        d_ff_spec=64 if not is_dense else 64,
        initial_layers=4,
        router_layer=2,
        initial_experts=1 if is_dense else 2,
        max_experts=1 if is_dense else 4,
        auto_lifecycle=not is_dense,
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
        halt_use_energy_only=True,
        use_marginal_drop_halt=True,
        skip_final_lookahead=True,
        detach_decoder=True,
        lora_decoder_rank=8,
        use_rel_pool=use_rel_pool,
        enable_nuance_escalation=enable_nuance_escalation,
    )


def test_rel_pool_unbundling():
    m_with = _make_tiny_v9(use_rel_pool=True)
    m_without = _make_tiny_v9(use_rel_pool=False)
    assert m_with.rel_pool_proj is not None
    assert m_without.rel_pool_proj is None

    x = torch.randint(33, 122, (2, 32))
    o_with = m_with(x, update_stats=False)
    o_without = m_without(x, update_stats=False)
    assert o_with["z_expr_pool"].shape == (2, 48)
    assert o_without["z_expr_pool"].shape == (2, 48)


def test_nuance_escalation_flag():
    m_no_esc = _make_tiny_v9(enable_nuance_escalation=False)
    assert m_no_esc.z_nuance_thresh >= 1e8
    assert m_no_esc.d_sem_nuance_thresh >= 1e8

    x = torch.randint(33, 122, (2, 32))
    o = m_no_esc(x, update_stats=False)
    assert "nuance_mask" in o
    assert o["nuance_mask"].sum().item() == 0


def test_matched_control_dense():
    m_dense = _make_tiny_v9(is_dense=True)
    assert m_dense.auto_lifecycle is False
    assert len(m_dense.layers[2].moe.experts) == 1

    groups = classify_v7_spgc_param_groups(
        m_dense,
        shared_mode="acquire",
        specialist_mode="acquire",
        lr_muon=8e-3,
        lr_adam=3e-3,
        lr_consolidate=2e-3,
        trunk_prec_scale=0.85,
    )
    opt = MuonBONGOptimizer(groups, mode="acquire", bong_prior_pull=0.02, bong_max_prec=100.0)
    assert opt is not None

    x = torch.randint(33, 122, (2, 32))
    out = m_dense(x, update_stats=True)
    loss = out["latent"].sum()
    loss.backward()
    opt.step(fisher_scale=1.0)
