"""RV4 scaffolding: every stage-4 flag exists, carries the declared constants and is validated (spec sections 4.3, 5.3, 6.5).

The flags were OFF until the pre-registered run (RV2-D); the router (with a fast route rule) and the transfer rule are ON by default since its adoption (2026-10-21, E-041 to E-043,
test_gm_adopt_stage4.py), growth from the fit gap is validated but not the default; `PRE_STAGE4` restores the old behaviour."""

import importlib.util
from pathlib import Path

import pytest

from growmoe.config import PRE_STAGE4, PhyloCfg

ROOT = Path(__file__).resolve().parents[2]


def _golden():
    spec = importlib.util.spec_from_file_location("golden", ROOT / "research" / "growmoe_results" / "rv2c" / "golden.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_router_and_the_transfer_rule_are_on_by_default_growth_is_not_and_pre_stage_4_turns_all_three_off():
    cfg = PhyloCfg(route_rule="accept_then_logf")
    assert cfg.router is True and cfg.spawn_rule == "transfer" and cfg.growth_rule == "probe"
    old = PhyloCfg(route_rule="accept_then_logf", **PRE_STAGE4)
    assert old.router is False and old.spawn_rule == "probe" and old.growth_rule == "probe"
    assert set(PRE_STAGE4) == {"router", "spawn_rule", "growth_rule"}                  # the three switches, nothing else


def test_the_stage_4_constants_have_the_declared_defaults():
    cfg = PhyloCfg(route_rule="accept_then_logf")
    assert (cfg.router_candidates, cfg.router_margin, cfg.router_folds, cfg.router_refit_every, cfg.router_min_protos, cfg.router_mode,
            cfg.router_softmax_max_experts) == (("density", "lda", "softmax", "knn5"), 0.01, 5, 50, 8, "free", 64)
    assert (cfg.transfer_alpha, cfg.transfer_margin, cfg.transfer_perms, cfg.transfer_ref, cfg.transfer_min_ref, cfg.spawn_pretrain_steps) == (0.01, 0.25, 999, 512, 64, 60)
    assert (cfg.grow_window, cfg.grow_plateau, cfg.grow_min_age, cfg.grow_cooldown, cfg.grow_factor, cfg.grow_guard_window, cfg.grow_guard_eps,
            cfg.grow_budget_params) == (100, 0.95, 300, 200, 2.0, 100, 0.02, None)
    assert cfg.grow_gap == 0.17                                                         # chosen on dev seeds by the declared rule, frozen in the pre-registration


def test_valid_stage_4_settings_are_accepted():
    cfg = PhyloCfg(router=True, route_rule="accept_then_logf", spawn_rule="transfer", growth_rule="gap", router_mode="free", grow_budget_params=10_000)
    assert cfg.router and cfg.spawn_rule == "transfer" and cfg.growth_rule == "gap"


@pytest.mark.parametrize("kw", [
    dict(router_mode="anywhere"),
    dict(router_candidates=("lda",)),                                    # density is the fallback and must be a candidate
    dict(router_candidates=("density", "svm")),
    dict(router_folds=1),
    dict(router_min_protos=3),                                           # fewer than the folds
    dict(router_refit_every=0),
    dict(router_margin=-0.1),
    dict(router_softmax_max_experts=1),
    dict(router=True),                                                   # route_rule defaults to "default": the reference path is router-blind, so refuse instead of silently doing nothing
    dict(router=True, route_rule="default"),
    dict(spawn_rule="oracle"),
    dict(transfer_alpha=0.0), dict(transfer_alpha=1.0),
    dict(transfer_alpha=0.001, transfer_perms=99),                       # 1/(1+99) > 0.001: the test could never reject
    dict(transfer_margin=-0.01),
    dict(transfer_perms=0),
    dict(transfer_ref=32, transfer_min_ref=64),
    dict(transfer_min_ref=4),
    dict(spawn_pretrain_steps=-1),
    dict(growth_rule="always"),
    dict(grow_gap=0.0), dict(grow_gap=1.0),
    dict(grow_window=5),
    dict(grow_plateau=0.0), dict(grow_plateau=1.5),
    dict(grow_min_age=0),
    dict(grow_cooldown=-1),
    dict(grow_factor=1.0),
    dict(grow_guard_window=0),
    dict(grow_guard_eps=-0.1),
    dict(grow_budget_params=0),
])
def test_invalid_stage_4_settings_are_refused(kw):
    with pytest.raises(ValueError):
        PhyloCfg(**kw)


def test_the_golden_file_pins_every_stage_4_flag_off():
    golden = _golden()
    assert golden.BASE["router"] is False and golden.BASE["spawn_rule"] == "probe" and golden.BASE["growth_rule"] == "probe"


def test_with_all_stage_4_flags_off_the_pool_is_bit_identical_to_the_recorded_golden_run():
    golden = _golden()
    assert golden.differences(golden.QUICK) == []
