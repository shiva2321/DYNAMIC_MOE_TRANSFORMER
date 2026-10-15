"""RV4 stage 4c inside PhyloMoE: the controller, the guard, caps, budget, accounting (spec sections 6.2-6.6).

The synthetic task is a random two-layer teacher on unit inputs (16 inputs, 8 classes). The growth threshold of the growth tests is NOT a magic number: it is calibrated inside
the module from the measured gap of a starved and an adequate expert (`gamma`), so the tests check the mechanism, not a constant."""

import copy
import pickle

import pytest
import torch
import torch.nn.functional as F

from growmoe.accounting import pool_report
from growmoe.config import GrowthCfg, PhyloCfg
from growmoe.gapgrow import GapMonitor, GrowthGuard
from growmoe.pools.phylo import PhyloMoE
from rv4_helpers import regime_batch, trained_pool

D, C = 16, 8


def _batches(seed=0, scale=3.0, hidden=32, random_labels=False):
    g = torch.Generator().manual_seed(seed)
    w1, w2 = torch.randn(D, hidden, generator=g), torch.randn(hidden, C, generator=g)

    def batch(k):
        gk = torch.Generator().manual_seed(50_000 + 31 * seed + k)
        x = F.normalize(torch.randn(64, D, generator=gk), dim=1)
        if random_labels:
            return x, torch.randint(0, C, (64,), generator=gk)
        return x, (torch.tanh(scale * (x @ w1)) @ w2).argmax(1)
    return batch


def _gap_pool(hidden, max_hidden=1024, **kw):
    base = dict(d=D, hidden=hidden, n_classes=C, seed=1, q=0.02, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False,
                growth_rule="gap", grow_gap=0.2, grow_window=20, grow_min_age=40, grow_cooldown=20, grow_guard_window=20)
    base.update(kw)
    return PhyloMoE(PhyloCfg(**base), grow_cfg=GrowthCfg(max_hidden=max_hidden))


def _run(pool, batch, n, start=0):
    for k in range(start, start + n):
        pool.observe(*batch(k))


@pytest.fixture(scope="module")
def gaps():
    batch = _batches()
    out = {}
    for h in (4, 16, 64):
        pool = _gap_pool(h, grow_min_age=10 ** 9)                                       # the monitor runs, nothing may grow
        _run(pool, batch, 150)
        out[h] = pool.experts[0].meta["gap"].gap()
    return out


@pytest.fixture(scope="module")
def gamma(gaps):
    return 0.5 * (gaps[16] + gaps[64])


# ------------------------------------------------------------------------------------------------ the signal
def test_the_fit_gap_rises_with_width(gaps):
    """Premise measured in spike C (E-032) on the arithmetic stream; here on a teacher stream with widths 4, 16, 64."""
    assert 0.0 < gaps[4] < gaps[16] < gaps[64] < 1.0


def test_the_post_loss_costs_one_forward_of_the_trained_rows_and_no_more():
    pool = _gap_pool(8, grow_min_age=10 ** 9)
    _run(pool, _batches(), 30)
    s = pool.grow_stats
    assert s["post_rows"] == s["train_rows"] > 0                                        # at most 64 rows per batch, and a batch has 64


# ------------------------------------------------------------------------------------------------ the controller
def test_a_starved_expert_grows_at_least_twice_and_then_stops(gamma):
    pool = _gap_pool(4, grow_gap=gamma)
    _run(pool, _batches(), 700)
    grows = pool.events.of_kind("grow")
    assert len(grows) >= 2 and all(e.fields["rule"] == "gap" and e.fields["kind"] == "widen" for e in grows)
    assert all(e.fields["gap"] < gamma and e.fields["hidden_after"] == 2 * e.fields["hidden_before"] for e in grows)
    assert pool.experts[0].module.hidden >= 16
    assert grows[-1].step < 700 - 150                                                   # it stopped by itself


def test_an_adequate_expert_does_not_grow(gamma):
    pool = _gap_pool(64, grow_gap=gamma)
    _run(pool, _batches(), 400)
    assert not pool.events.of_kind("grow") and pool.experts[0].module.hidden == 64


def test_maturity_blocks_growth_early_and_for_ever_when_the_age_is_out_of_reach():
    never = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_min_age=10 ** 6)
    _run(never, _batches(), 200)
    assert not never.events.of_kind("grow")
    early = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_min_age=60)
    _run(early, _batches(), 300)
    grows = early.events.of_kind("grow")
    assert grows and grows[0].step >= 60


def test_growth_stops_at_the_cap_and_at_the_budget_without_error():
    capped = _gap_pool(4, max_hidden=16, grow_gap=0.9, grow_plateau=1.0)
    _run(capped, _batches(), 400)
    assert capped.experts[0].module.hidden <= 16 and capped.grow_stats["blocked_cap"] > 0
    poor = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0)
    _run(poor, _batches(), 1)
    poor.cfg.grow_budget_params = sum(r.module.num_params() for r in poor.experts.values()) + 10
    _run(poor, _batches(), 300, start=1)
    assert not poor.events.of_kind("grow") and poor.grow_stats["blocked_budget"] > 0


def test_pure_noise_cannot_grow_an_expert_past_its_cap():
    pool = _gap_pool(4, max_hidden=32, grow_gap=0.9, grow_plateau=1.0)
    _run(pool, _batches(random_labels=True), 400)
    assert pool.experts[0].module.hidden <= 32 and len(pool.events.of_kind("grow")) <= 3


def test_the_growth_rule_needs_the_master_switch_and_the_probe_rule_keeps_no_monitor():
    off = _gap_pool(4, grow=False, grow_gap=0.9, grow_plateau=1.0)
    _run(off, _batches(), 120)
    assert "gap" not in off.experts[0].meta and not off.events.of_kind("grow")
    probe = _gap_pool(4, growth_rule="probe")
    _run(probe, _batches(), 60)
    assert "gap" not in probe.experts[0].meta and probe.grow_stats["post_rows"] == 0


# ------------------------------------------------------------------------------------------------ the guard
def _until_first_growth(pool, batch, limit=400):
    for k in range(limit):
        pool.observe(*batch(k))
        if pool.events.of_kind("grow"):
            return k + 1
    raise AssertionError("no growth within the limit: change the setting of the test")


def test_the_guard_reverts_a_growth_whose_window_loss_is_worse_than_before():
    """The loss before the growth is set unreachably low, as if the grown expert had come out much worse. (Wrecking the weights instead also trips the competence monitor, whose alarm
    skips the training step and with it the guard's bookkeeping: that interplay is the monitor's, not the guard's.)"""
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0), _batches()
    k = _until_first_growth(pool, batch)
    rec = pool.experts[0]
    grown = rec.module.hidden
    assert "guard" in rec.meta and grown > 4
    rec.meta["guard"].before = 1e-6
    _run(pool, batch, 22, start=k)
    assert pool.events.of_kind("grow_revert") and rec.module.hidden < grown and "guard" not in rec.meta
    assert pool.grow_stats["reverts"] == 1 and rec.meta["grow_cooldown"] == 40


def test_a_healthy_growth_is_kept_when_the_guard_window_ends():
    """grow_guard_eps is 0.5 here: with the product value 0.02 and a 20-batch window the noise of two window means alone (about 3 %) would revert a healthy growth now and then;
    this test is about the KEEP path. The revert rate at the product settings is a dev-review item (task 6)."""
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_cooldown=10_000, grow_guard_eps=0.5), _batches()
    k = _until_first_growth(pool, batch)
    rec = pool.experts[0]
    grown = rec.module.hidden
    _run(pool, batch, 22, start=k)
    assert "guard" not in rec.meta and rec.module.hidden == grown and not pool.events.of_kind("grow_revert")


def test_the_guard_copy_is_counted_while_it_exists():
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_guard_window=30, grow_cooldown=10_000), _batches()
    k = _until_first_growth(pool, batch)
    rec = pool.experts[0]
    rep = pool_report(pool, D)
    assert rep.params == rec.module.num_params() + rec.meta["guard"].module.num_params()
    _run(pool, batch, 35, start=k)
    assert "guard" not in rec.meta and pool_report(pool, D).params == rec.module.num_params()


def test_a_pool_with_an_active_guard_can_be_pickled():
    pool, batch = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0, grow_cooldown=10_000), _batches()
    _until_first_growth(pool, batch)
    assert "guard" in pool.experts[0].meta
    for clone in (pickle.loads(pickle.dumps(pool)), copy.deepcopy(pool)):
        x, _ = batch(999)
        assert torch.allclose(clone.predict(x).probs, pool.predict(x).probs, atol=1e-5)


def test_a_merge_drops_the_guard_and_the_stale_windows():
    pool, W = trained_pool(n_batches=3, growth_rule="gap")
    rec = pool.experts[0]
    rec.meta["guard"] = GrowthGuard(module=rec.module.clone(), opt=rec.opt, before=1.0, hidden_before=rec.module.hidden)
    rec.meta["gap"] = GapMonitor(20)
    pool._forget_growth_state(rec)
    assert "guard" not in rec.meta and "gap" not in rec.meta


# ------------------------------------------------------------------------------------------------ determinism and the flag off
def test_two_identical_runs_grow_identically():
    runs = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(2):
            pool = _gap_pool(4, grow_gap=0.9, grow_plateau=1.0)
            _run(pool, _batches(), 150)
            runs.append([(e.step, e.fields["increment"], e.fields["hidden_after"]) for e in pool.events.of_kind("grow")])
    finally:
        torch.set_num_threads(threads)
    assert runs[0] == runs[1] and runs[0]


def test_with_the_probe_rule_the_pool_is_bit_identical_to_the_recorded_golden_run():
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("golden", root / "research" / "growmoe_results" / "rv2c" / "golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    assert golden.differences(golden.QUICK) == []
