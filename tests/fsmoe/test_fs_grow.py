import numpy as np
import pytest
import torch
import torch.nn.functional as F

from growmoe.growth import GrowthDecision
from fsmoe.grow.governor import BruteGovernor, GovCfg, Governor, pick_size
from fsmoe.grow.system import Expert, GrowSystem, Reservoir, SysCfg, Tier, evaluate
from fsmoe.grow.world import FAMILIES, K, GrowthWorld, Regime, need_from_acc


# ------------------------------------------------------------------ world
def test_every_family_gives_balanced_classes_and_is_deterministic():
    for kind in FAMILIES:
        a, b = Regime(0, kind, np.random.default_rng(3)), Regime(0, kind, np.random.default_rng(3))
        xa, ya = a.sample(2000, np.random.default_rng(9))
        xb, yb = b.sample(2000, np.random.default_rng(9))
        assert np.array_equal(ya, yb) and np.allclose(xa, xb)
        assert np.bincount(ya, minlength=K).min() >= 0.03 * 2000
        assert np.allclose(np.linalg.norm(xa, axis=1), 1.0, atol=1e-5)


def test_unknown_family_is_refused():
    with pytest.raises(ValueError):
        Regime(0, "nonsense", np.random.default_rng(0))


def test_regime_centres_are_kept_apart_and_the_stream_brings_regimes_phase_by_phase():
    w = GrowthWorld(5, ["lin"] * 8, per_phase=4, phase_batches=6, batch=16)
    C = np.stack([r.centre for r in w.regimes])
    assert np.abs(C @ C.T - np.eye(8)).max() <= 0.3 + 1e-9
    st = w.stream()
    assert len(st) == 12 and all(b.regime < 4 for b in st[:6]) and any(b.regime >= 4 for b in st[6:]) and any(b.regime < 4 for b in st[6:])


def test_need_from_accuracies_covers_every_branch():
    assert need_from_acc({"small": 0.6, "wide": 0.75, "deep": 0.7}) == "width"
    assert need_from_acc({"small": 0.6, "wide": 0.75, "deep": 0.75}) == "width"          # a tie goes to width
    assert need_from_acc({"small": 0.6, "wide": 0.7, "deep": 0.75}) == "depth"
    assert need_from_acc({"small": 0.6, "wide": 0.62, "deep": 0.64}) == "none"           # gains below the margin
    assert need_from_acc({"small": 0.6, "wide": 0.65, "deep": 0.6}, margin=0.05) == "width"
    assert need_from_acc({"small": 0.6, "wide": 0.64, "deep": 0.6}, margin=0.05) == "none"


# ------------------------------------------------------------------ system
def test_reservoir_is_bounded_and_samples_the_whole_stream():
    rng = np.random.default_rng(0)
    r = Reservoir(2, 10)
    for i in range(200):
        r.add(torch.tensor([[float(i), 0.0]]), torch.tensor([i % 3]), rng)
    assert len(r) == 10 and r.x.shape == (10, 2) and float(r.x[:, 0].max()) > 60


def _unit(i, d=32):
    v = torch.zeros(d)
    v[i] = 1.0
    return v


def _blob(centre, n, seed, noise=0.8, d=32):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(centre + noise * torch.randn(n, d, generator=g) / np.sqrt(d), dim=1)


def _two_expert_system(**kw):
    s = GrowSystem(SysCfg(**kw))
    for e_id, i in enumerate((0, 1)):
        e = Expert(e_id, s.cfg)
        x = _blob(_unit(i), 200, e_id)
        e.csum, e.n = x.sum(0), 200
        e.cos.extend((x @ e.centroid).tolist())
        s.experts[e_id] = e
    s.next_id = 2
    return s


def test_threshold_is_the_default_until_enough_samples_then_a_floored_quantile():
    cfg = SysCfg()
    e = Expert(0, cfg)
    assert e.threshold(cfg) == cfg.accept_default
    e.cos.extend([0.9] * 100)
    assert e.threshold(cfg) == pytest.approx(0.9)
    e.cos.extend([0.0] * 100)
    assert e.threshold(cfg) == cfg.accept_floor                                           # the quantile is 0, the floor holds it up


def test_routing_sends_a_sample_to_the_accepting_expert_and_calls_a_far_one_novel():
    s = _two_expert_system()
    x = torch.cat([_blob(_unit(0), 3, 5), _blob(_unit(1), 3, 6), _blob(_unit(7), 3, 7)])
    routed, nearest, gate = s.route(x)
    assert routed.tolist()[:6] == [0, 0, 0, 1, 1, 1] and routed.tolist()[6:] == [-1, -1, -1]
    assert nearest.shape == (9,) and gate == 2 * 32


def test_first_batch_of_a_regime_founds_an_expert_and_the_next_one_trains_it():
    s = GrowSystem(SysCfg(net_at_birth=True, seed=1))
    y = torch.randint(0, 6, (128,))
    s.observe(_blob(_unit(0), 128, 1), y)
    assert len(s.experts) == 1 and s.experts[0].net is not None
    s.observe(_blob(_unit(0), 128, 2), y)
    assert len(s.experts) == 1 and s.experts[0].age == 2
    s.observe(_blob(_unit(9), 128, 3), y)                                                 # a different direction: a second expert
    assert len(s.experts) == 2


def test_a_notebook_first_newborn_has_no_network_and_still_answers():
    s = GrowSystem(SysCfg(net_at_birth=False))
    x = _blob(_unit(0), 128, 1)
    y = (x[:, 1] > x[:, 1].median()).long()
    s.observe(x, y)
    e = s.experts[0]
    assert e.net is None and e.state == "newborn" and e.params()[0] == 0
    probs, who, _ = s.predict(_blob(_unit(0), 64, 4))
    assert probs.shape == (64, 6) and bool((who == 0).all()) and torch.allclose(probs.sum(1), torch.ones(64), atol=1e-4)


def test_a_group_that_is_an_existing_experts_own_outliers_is_absorbed_not_founded():
    s = GrowSystem(SysCfg(net_at_birth=True, seed=2))
    y = torch.randint(0, 6, (128,))
    s.observe(_blob(_unit(0), 128, 1), y)
    out = _blob(_unit(0), 150, 9, noise=1.8)                                               # the loose tail of the same direction (cosine about 0.5 to its centre, pairs about 0.25)
    before = len(s.experts)
    s.room_x, s.room_y, s.room_reg = [out], [torch.randint(0, 6, (150,))], [0] * 150
    s._room_step()
    assert len(s.experts) == before and s.summary()["events"].get("absorb") == 1


def test_a_group_far_from_every_expert_founds_a_new_one():
    s = GrowSystem(SysCfg(net_at_birth=True, seed=2))
    s.observe(_blob(_unit(0), 128, 1), torch.randint(0, 6, (128,)))
    s.room_x, s.room_y, s.room_reg = [_blob(_unit(11), 90, 5)], [torch.randint(0, 6, (90,))], [1] * 90
    s._room_step()
    assert len(s.experts) == 2 and s.summary()["events"].get("spawn") == 2 and not s.room_x


def test_params_count_networks_and_notebooks_separately():
    s = GrowSystem(SysCfg(net_at_birth=True))
    s.observe(_blob(_unit(0), 128, 1), torch.randint(0, 6, (128,)))
    p = s.params()
    e = s.experts[0]
    assert p["net"] == e.net.num_params() and p["notebook"] == e.nb.n_scalars() + 32 and p["total"] == p["net"] + p["notebook"]


def test_a_tier_cuts_the_gate_cost_and_keeps_the_routing():
    cfg = SysCfg(seed=3)
    s = GrowSystem(cfg)
    rng = np.random.default_rng(0)
    for i in range(24):
        e = Expert(i, cfg)
        x = _blob(_unit(i), 80, i)
        e.csum, e.n = x.sum(0), 80
        e.cos.extend((x @ e.centroid).tolist())
        e.hold.add(x[:40], torch.zeros(40, dtype=torch.long), rng)
        s.experts[i] = e
    s.dirty = True
    val = torch.cat([_blob(_unit(i), 20, 100 + i) for i in range(24)])
    flat, _, gate_flat = s.route(val)
    s.cfg.gate_budget = 100.0
    Governor(GovCfg())._tier(s)
    assert s.tier is not None
    tiered, _, gate_tier = s.route(val)
    assert gate_flat == 24 * 32 and gate_tier <= 0.8 * gate_flat and float((tiered == flat).float().mean()) >= 0.95


def test_no_tier_is_built_when_the_flat_gate_is_within_budget():
    s = _two_expert_system()
    s.cfg.gate_budget = 1000.0
    Governor(GovCfg())._tier(s)
    assert s.tier is None


def test_evaluate_reports_per_regime_accuracy_and_cost():
    w = GrowthWorld(2, ["lin"] * 2, per_phase=2, phase_batches=30, batch=64, n_test=100)
    s = GrowSystem(SysCfg(net_at_birth=True, seed=2))
    for b in w.stream():
        s.observe(b.x, b.y, torch.full((b.x.shape[0],), b.regime))
    ev = evaluate(s, w)
    assert set(ev["acc"]) == {0, 1} and 0.0 <= ev["min"] <= ev["mean"] <= 1.0 and ev["macs_answer"] > 0 and ev["mean"] > 0.5


# ------------------------------------------------------------------ governor
class _FakePolicy:
    """Stands in for the growth trial: records calls and commits a widening of 8 units."""

    def __init__(self, kind="widen"):
        self.calls, self.kind = 0, kind

    def step(self, net, x_tr, y_tr, x_va, y_va, rng):
        self.calls += 1
        before = net.num_params()
        net.widen(8) if self.kind == "widen" else net.deepen()
        return GrowthDecision(kind=self.kind, gain=0.1, params_added=net.num_params() - before, p_value=0.001, reason="committed", tried={"a": {}, "b": {}})


def _trained_expert(cfg, seed=0, n=1200, hold=300):
    """An expert with a network that has seen a hard (wide) regime, plenty of data, and a held-out set."""
    w = GrowthWorld(seed, ["wide"], per_phase=1, phase_batches=1)
    rng = np.random.default_rng(seed)
    x, y = (torch.from_numpy(a) for a in w.regimes[0].sample(n + hold, rng))
    e = Expert(0, cfg)
    e.csum, e.n, e.age = x.sum(0), x.shape[0], 50
    e.train.add(x[:n], y[:n], rng)
    e.hold.add(x[n:], y[n:], rng)
    e.nb.update(x[:n], y[:n])
    e.new_net(8, 1, seed=1)
    for _ in range(60):
        idx = torch.randint(0, n, (128,))
        e.net.train_step(x[idx], y[idx], e.opt)
    return e


def _system_with(e, **gov_kw):
    s = GrowSystem(SysCfg(seed=0))
    s.experts[e.id] = e
    return s, Governor(GovCfg(**gov_kw))


def test_growth_is_asked_about_only_for_an_expert_with_error_age_and_data():
    cfg = SysCfg()
    e = _trained_expert(cfg)
    e.policy = _FakePolicy()
    s, gov = _system_with(e)
    e.age = 2
    gov._grow(s, e)
    assert e.policy.calls == 0                                                            # too young
    e.age = 50
    gov._grow(s, e)
    assert e.policy.calls == 1 and e.guard is not None and s.events[-1]["kind"] == "grow"


def test_an_expert_that_already_fits_its_held_out_data_is_not_asked():
    cfg = SysCfg()
    e = _trained_expert(cfg)
    e.policy = _FakePolicy()
    s, gov = _system_with(e, err_trigger=1.1)                                             # no held-out error can exceed this
    gov._grow(s, e)
    assert e.policy.calls == 0


def test_the_trial_budget_stops_growth_questions():
    cfg = SysCfg()
    e = _trained_expert(cfg)
    e.policy = _FakePolicy()
    s, gov = _system_with(e, trial_ratio=1.0)
    s.macs_train, s.macs_trial_grow = 10.0, 11.0
    gov._grow(s, e)
    assert e.policy.calls == 0
    s.macs_trial_grow = 5.0
    gov._grow(s, e)
    assert e.policy.calls == 1


def test_growth_over_the_parameter_budget_is_undone():
    cfg = SysCfg()
    e = _trained_expert(cfg)
    e.policy = _FakePolicy()
    s, gov = _system_with(e, max_params=1)
    h = e.net.hidden
    gov._grow(s, e)
    assert e.net.hidden == h and e.guard is None and s.events[-1]["kind"] == "refused"


def test_the_guard_undoes_a_growth_that_made_the_held_out_loss_worse_and_keeps_a_good_one():
    cfg = SysCfg()
    e = _trained_expert(cfg)
    s, gov = _system_with(e, guard_checks=2)
    seen = e.net.seen[e.hold.y]
    now = float(F.cross_entropy(e.net(e.hold.x[seen]), e.hold.y[seen]))
    snap = e.net.clone()
    e.net.widen(8)
    e.guard = {"snap": snap, "before": now * 0.5, "looks": 0}                              # the loss before was much lower: the growth is a regression
    gov._guard(s, e)
    assert e.guard is not None                                                            # first look: still waiting
    gov._guard(s, e)
    assert e.guard is None and e.net is snap and s.events[-1]["kind"] == "revert"
    e.guard = {"snap": e.net.clone(), "before": now * 10, "looks": 1}                      # the loss before was much higher: keep
    gov._guard(s, e)
    assert e.guard is None and s.events[-1]["kind"] == "keep"


def test_a_notebook_gets_a_network_only_when_it_wins_on_held_out_data_and_the_size_is_right_sized():
    cfg = SysCfg()
    w = GrowthWorld(4, ["wide", "lin"], per_phase=2, phase_batches=1)
    out = {}
    for rid in (0, 1):
        rng = np.random.default_rng(rid)
        x, y = (torch.from_numpy(a) for a in w.regimes[rid].sample(12000, rng))
        e = Expert(rid, cfg)
        e.state, e.age = "newborn", 30
        e.csum, e.n = x.sum(0), 12000
        e.train.add(x[:10800], y[:10800], rng)
        e.hold.add(x[10800:], y[10800:], rng)
        e.nb.update(x[:10800], y[:10800])
        s, gov = _system_with(e, promote_steps=400, promote_sizes=((8, 1), (64, 1)))
        gov._promote(s, e)
        out[rid] = (e.state, None if e.net is None else e.net.hidden, s.events[-1]["kind"])
    assert out[0][0] == "active" and out[0][1] == 64 and out[0][2] == "promote"           # the wide rule needs the bigger candidate
    assert out[1][0] == "notebook" and out[1][1] is None and out[1][2] == "keep_notebook"  # the linear rule is already the notebook's
    assert e.next_decision == 2 * len(e.train)                                            # judged again when the data has doubled


def test_a_kept_notebook_is_judged_again_only_when_its_data_has_doubled():
    cfg = SysCfg()
    w = GrowthWorld(4, ["lin"], per_phase=1, phase_batches=1)
    rng = np.random.default_rng(0)
    x, y = (torch.from_numpy(a) for a in w.regimes[0].sample(1600, rng))
    e = Expert(0, cfg)
    e.state, e.age, e.next_decision = "notebook", 30, 1400
    e.train.add(x[:1300], y[:1300], rng)
    e.hold.add(x[1300:], y[1300:], rng)
    e.nb.update(x[:1300], y[:1300])
    s, gov = _system_with(e, promote_steps=20, promote_sizes=((8, 1),))
    gov._promote(s, e)
    assert not s.events and s.macs_trial == 0.0                                           # 1,300 < 1,400: not yet


def test_brute_force_alternates_wider_and_deeper_up_to_its_caps_with_no_test():
    s = GrowSystem(SysCfg(net_at_birth=True, seed=1))
    s.observe(_blob(_unit(0), 128, 1), torch.randint(0, 6, (128,)))
    e = s.experts[0]
    e.age = 10
    gov = BruteGovernor(check_every=1, age_min=1, max_hidden=32, max_depth=3)
    s.batches = 1
    seen = []
    for _ in range(8):
        gov.step(s)
        seen.append((e.net.hidden, len(e.net.blocks)))
    assert seen[0] == (16, 1) and seen[1] == (16, 2) and seen[2] == (32, 2) and seen[3] == (32, 3) and seen[-1] == (32, 3)
    assert all(ev["kind"] == "grow" for ev in s.events if ev["kind"] != "spawn")


def test_the_governor_only_looks_every_check_every_batches():
    s = GrowSystem(SysCfg(net_at_birth=True))
    gov = Governor(GovCfg(check_every=10))
    called = []
    gov._guard = lambda *a: called.append(1)
    s.experts[0] = Expert(0, s.cfg)
    s.batches = 7
    gov.step(s)
    assert not called
    s.batches = 20
    gov.step(s)
    assert called


def test_pick_size_takes_the_cheapest_candidate_within_the_knee_of_the_best_and_none_below_the_margin():
    tried = [(900, 0.70), (3000, 0.79), (5000, 0.80)]
    assert pick_size(tried, 0.60, margin=0.02, knee=0.015) == 1                           # 0.79 is within 0.015 of 0.80 and cheaper
    assert pick_size(tried, 0.60, margin=0.02, knee=0.15) == 0                            # a wide knee takes the smallest
    assert pick_size(tried, 0.79, margin=0.02, knee=0.015) is None                        # the best gains only 0.01 over the notebook
    assert pick_size([(900, 0.7)], 0.70 - 0.02, margin=0.02, knee=0.0) == 0               # exactly the margin passes


def test_held_out_samples_are_never_trained_on():
    s = GrowSystem(SysCfg(net_at_birth=False, seed=3))
    x = _blob(_unit(0), 128, 1)
    s.observe(x, torch.randint(0, 6, (128,)))
    s.observe(_blob(_unit(0), 128, 2), torch.randint(0, 6, (128,)))
    e = s.experts[0]
    assert len(e.train) + len(e.hold) == e.n and len(e.hold) > 0
    both = torch.cat([e.train.x, e.hold.x])
    assert torch.unique(both, dim=0).shape[0] == e.n                                      # no row is in both
    assert e.nb.total == len(e.train)                                                     # the notebook learned from the training part only


def _four_experts(centres):
    cfg = SysCfg(seed=1)
    s = GrowSystem(cfg)
    rng = np.random.default_rng(0)
    for i, c in enumerate(centres):
        e = Expert(i, cfg)
        x = _blob(c, 100, i)
        e.csum, e.n = x.sum(0), 100
        e.cos.extend((x @ e.centroid).tolist())
        e.hold.add(x[:50], torch.zeros(50, dtype=torch.long), rng)
        s.experts[i] = e
    s.dirty = True
    return s


def test_the_tier_falls_back_to_a_full_scan_when_the_best_groups_hold_no_accepting_expert():
    s = _four_experts([_unit(i) for i in range(4)])
    cs = torch.stack([_unit(2), _unit(0)])                                                # group 0 = experts {0, 1} but pointed at the wrong place, group 1 = {2, 3}
    s.tier = Tier([[0, 1], [2, 3]], cs, 1, 4)
    x = _blob(_unit(0), 20, 9)
    routed, _, gate = s.route(x)
    s.tier = None
    flat, _, _ = s.route(x)
    assert routed.tolist() == flat.tolist() and int((routed == 0).sum()) >= 15 and gate > 2 * 32 + 2 * 32     # the same experts as the flat scan; the full scan was paid for


def test_a_tier_that_is_not_cheap_enough_or_does_not_agree_enough_is_refused_and_a_refusal_is_not_repeated():
    s = _four_experts([_unit(i) for i in range(4)])
    s.cfg.gate_budget = 10.0
    gov = Governor(GovCfg())
    gov._tier(s)                                                                          # 4 experts: 2 groups of 2, top-1 costs 64 + 64 = 128 = flat, not under 80 %
    assert s.tier is None and s.events[-1]["kind"] == "tier_refused"
    n = len(s.events)
    gov._tier(s)
    assert len(s.events) == n                                                             # not asked again until the pool has grown by 25 %
    s2 = _four_experts([_unit(i) for i in range(4)])
    s2.cfg.gate_budget = 10.0
    Governor(GovCfg(tier_agree=1.01))._tier(s2)
    assert s2.tier is None


def test_a_built_tier_is_rebuilt_only_when_the_pool_has_grown():
    cfg = SysCfg(seed=3, gate_budget=100.0)
    s = GrowSystem(cfg)
    rng = np.random.default_rng(0)
    for i in range(24):
        e = Expert(i, cfg)
        x = _blob(_unit(i), 80, i)
        e.csum, e.n = x.sum(0), 80
        e.cos.extend((x @ e.centroid).tolist())
        e.hold.add(x[:40], torch.zeros(40, dtype=torch.long), rng)
        s.experts[i] = e
    s.dirty = True
    gov = Governor(GovCfg())
    gov._tier(s)
    first = s.tier
    assert first is not None and first.built_for == 24
    gov._tier(s)
    assert s.tier is first                                                                # 24 < 1.25 x 24: left alone


def test_training_cost_is_counted_for_networks_and_not_for_notebooks():
    a, b = GrowSystem(SysCfg(net_at_birth=True, seed=1)), GrowSystem(SysCfg(net_at_birth=False, seed=1))
    for s in (a, b):
        s.observe(_blob(_unit(0), 128, 1), torch.randint(0, 6, (128,)))
        s.observe(_blob(_unit(0), 128, 2), torch.randint(0, 6, (128,)))
    assert a.macs_train > 0 and b.macs_train == 0.0


def test_a_cheap_tier_is_still_refused_when_it_does_not_agree_with_the_flat_routing_enough():
    cfg = SysCfg(seed=3, gate_budget=100.0)
    s = GrowSystem(cfg)
    rng = np.random.default_rng(0)
    for i in range(24):
        e = Expert(i, cfg)
        x = _blob(_unit(i), 80, i)
        e.csum, e.n = x.sum(0), 80
        e.cos.extend((x @ e.centroid).tolist())
        e.hold.add(x[:40], torch.zeros(40, dtype=torch.long), rng)
        s.experts[i] = e
    s.dirty = True
    Governor(GovCfg(tier_agree=1.01))._tier(s)                                            # no routing can agree on more than all of the samples
    assert s.tier is None and s.events[-1]["kind"] == "tier_refused"
