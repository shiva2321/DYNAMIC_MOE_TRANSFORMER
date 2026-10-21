"""Adoption of stage 4 (2026-10-21; E-041 the run, E-042 its independent review, E-043 the adoption): the router and the transfer spawn rule are the defaults, growth from the fit gap is
validated but NOT the default, and the old behaviour is one line away.

The pre-registered run (RV2-D, seeds 85-99) passed all nine statements, so the decision rules adopt each component. The independent review found that `growth_rule="gap"` REPLACES the
shipped widen-and-deepen policy and, on operator-by-operator arithmetic, costs 1.2 to 1.9 points where the shipped policy deepens, so that default waits for the user's decision. The
router was tested only with a fast route rule, so it follows the route rule: on with `accept_then_logf` (what every tested pool used), off with the reference path "default".
Everything older is reproduced with `PRE_STAGE4`."""

import dataclasses

import pytest
import torch

from growmoe.config import PRE_STAGE4, PhyloCfg
from growmoe.pools.phylo import PhyloMoE


def test_the_router_and_the_transfer_rule_are_the_defaults_with_the_tested_constants():
    cfg = PhyloCfg(route_rule="accept_then_logf")
    assert (cfg.router, cfg.router_mode, cfg.router_margin) == (True, "free", 0.01)
    assert (cfg.spawn_rule, cfg.transfer_margin, cfg.transfer_alpha) == ("transfer", 0.25, 0.01)


def test_growth_from_the_fit_gap_is_validated_but_not_the_default():
    assert PhyloCfg().growth_rule == "probe" and PhyloCfg(route_rule="accept_then_logf").growth_rule == "probe"
    assert PhyloCfg(growth_rule="gap").grow_gap == 0.17                                  # the validated constant comes with the opt-in


def test_the_router_follows_the_route_rule_it_was_tested_with():
    assert PhyloCfg().router is False                                          # the reference path "default" is router-blind: the router stays off, nothing raises
    assert PhyloCfg(route_rule="accept_then_logf").router is True
    assert PhyloCfg(route_rule="accept_then_logf", router=False).router is False
    with pytest.raises(ValueError):
        PhyloCfg(router=True)                                                  # asking for the router explicitly on the router-blind path is still a mistake


def test_a_config_copy_keeps_what_was_resolved():
    cfg = PhyloCfg(route_rule="accept_then_logf")
    assert dataclasses.replace(cfg, q=0.02).router is True
    assert dataclasses.replace(cfg, router=False).router is False


def test_pre_stage_4_reproduces_the_old_defaults_field_by_field():
    old = PhyloCfg(**PRE_STAGE4)
    assert (old.router, old.spawn_rule, old.growth_rule) == (False, "probe", "probe")
    new = PhyloCfg(route_rule="accept_then_logf")
    changed = {f.name for f in dataclasses.fields(PhyloCfg) if getattr(PhyloCfg(**PRE_STAGE4, route_rule="accept_then_logf"), f.name) != getattr(new, f.name)}
    assert changed == {"router", "spawn_rule"}                                 # growth_rule's default is still "probe"; router_mode is inert while the router is off, so PRE_STAGE4 does not pin it


def test_the_phase_one_entry_point_pins_the_pre_stage_4_pool_so_recorded_results_reproduce():
    from growmoe import bench
    cfg = bench._phase1_cfg(seed=0, route_rule="accept_then_logf")
    assert (cfg.router, cfg.spawn_rule, cfg.growth_rule) == (False, "probe", "probe")
    assert bench._phase1_cfg(seed=0, route_rule="accept_then_logf", router=True).router is True          # callers can still override


def test_the_rv2d_arms_are_defined_against_the_pool_before_stage_4():
    """The arms list only what they turn ON; the S3 arm must stay 'stage 3 without stage 4' now that the router and the transfer rule are the defaults."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "research" / "growmoe_results" / "rv2d"))
    import rv2d_lib as R
    s3 = R.arm_flags(R.Cell("s3", "F-15", "S3", 100))
    assert (s3["router"], s3["spawn_rule"], s3["growth_rule"]) == (False, "probe", "probe")
    both = R.arm_flags(R.Cell("s3", "F-15", "S3+T+G", 100, gap=0.17, tmargin=0.25))
    assert (both["router"], both["spawn_rule"], both["growth_rule"], both["grow_gap"], both["transfer_margin"]) == (False, "transfer", "gap", 0.17, 0.25)


def _small_stream_pool(**kw):
    torch.manual_seed(0)
    cfg = PhyloCfg(d=16, hidden=16, n_classes=4, seed=3, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=3, route_rule="accept_then_logf", **kw)
    pool = PhyloMoE(cfg)
    g = torch.Generator().manual_seed(1)
    centre = torch.nn.functional.normalize(torch.randn(2, 16, generator=g), dim=1)
    W = torch.randn(2, 16, 4, generator=g)
    for step in range(16):
        r = step // 8
        x = torch.nn.functional.normalize(8.0 * centre[r] + torch.randn(64, 16, generator=g), dim=1)
        y = (x @ W[r]).argmax(1)
        pool.observe(x, y)
    return pool, x


def test_the_adopted_defaults_run_end_to_end_on_a_small_stream():
    pool, x = _small_stream_pool()
    pred = pool.predict(x)
    assert pred.probs.shape == (64, 4) and len(pool.experts) >= 2
    assert pool.cfg.router and pool.cfg.spawn_rule == "transfer" and pool.cfg.growth_rule == "probe"


def test_the_growth_opt_in_runs_end_to_end_on_the_same_stream():
    pool, x = _small_stream_pool(growth_rule="gap")
    assert pool.predict(x).probs.shape == (64, 4) and pool.cfg.growth_rule == "gap"
