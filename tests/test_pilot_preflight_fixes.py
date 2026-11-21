"""Regression tests for the two defects found by the real-data pilot preflight audit (C1, C5)."""
import torch

from dmoe.null_space import NullSpaceProjector
from dmoe.real_model import RealSelfStabilizingDMoE


def _model(**kw):
    cfg = dict(
        d_model=32, num_heads=4, d_ff=64, initial_layers=3, chunk_size=1, enable_memory=True,
        num_ltm_slots=8, enable_three_level_attn=True, local_window=4, chunk_stride=4,
        enable_layer_stm=True, use_rope=True, causal_attn=True, enable_world_model=True,
        wm_rollout_horizon=1, hybrid_mode=True,
    )
    cfg.update(kw)
    return RealSelfStabilizingDMoE(**cfg)


def test_c1_wm_target_matches_online_encoder_when_aligned():
    torch.manual_seed(1)
    m = _model(align_wm_target_with_encoder=True).eval()
    for L in (8, 32, 64, 96):
        x = torch.randint(0, 256, (2, L))
        with torch.no_grad():
            diff = (m.encode(x) - m.world_model.encode_target(x)).abs().max().item()
        assert diff <= 1e-6, f"L={L}: online/target encoder mismatch {diff}"


def test_c1_legacy_default_keeps_locked_behaviour():
    torch.manual_seed(1)
    m = _model().eval()  # align flag defaults to False so locked v4 results stay reproducible
    x = torch.randint(0, 256, (2, 16))
    with torch.no_grad():
        diff = (m.encode(x) - m.world_model.encode_target(x)).abs().max().item()
    assert diff > 1e-3, "legacy path expected to carry the positional-embedding offset"


def _train_one_pass(m):
    m.train()
    x = torch.randint(0, 256, (4, 16))
    m(x, update_stats=False, targets=x)


def _unprojected_trainable(m, gpm):
    covered = {id(p) for mod in gpm.monitored_modules.values() for p in mod.parameters()}
    norm_ids = {id(p) for n in gpm.shared_norms for p in n.parameters()}
    out = []
    for n, p in m.named_parameters():
        if "experts" in n or "expert_heads" in n:
            continue
        if id(p) in covered or id(p) in norm_ids:
            continue
        if p.requires_grad:
            out.append(n)
    return out


def test_c5_freeze_unprojected_shared_closes_the_gap():
    torch.manual_seed(2)
    m = _model()
    gpm = NullSpaceProjector(m, energy_threshold=0.95, max_rank_ratio=0.8, track_embeddings=True,
                             freeze_shared_norms_on_update=True, freeze_unprojected_shared=True)
    _train_one_pass(m)
    assert len(_unprojected_trainable(m, gpm)) > 0  # gap exists before consolidation
    gpm.update_basis_for_domain(0)
    assert _unprojected_trainable(m, gpm) == []


def test_c5_default_leaves_legacy_behaviour():
    torch.manual_seed(2)
    m = _model()
    gpm = NullSpaceProjector(m, energy_threshold=0.95, max_rank_ratio=0.8, track_embeddings=True,
                             freeze_shared_norms_on_update=True)
    _train_one_pass(m)
    gpm.update_basis_for_domain(0)
    assert len(_unprojected_trainable(m, gpm)) > 0
