"""The stream suite (unimoe/suite.py): domains turned into one global label space and three schedules (sequential, mixed, drifting) with test sets and out-of-distribution sets."""

import numpy as np
import pytest
import torch

from unimoe import suite as SU
from unimoe.domains import Domain


def toy_domain(name, kind, n_classes, regimes_of_class, n_train, n_test, seed, regime_by_example=None):
    rng = np.random.default_rng(seed)
    def make(n):
        y = rng.integers(0, n_classes, n)
        x = torch.nn.functional.normalize(torch.from_numpy(rng.standard_normal((n, 128)).astype(np.float32)), dim=1)
        r = torch.from_numpy(np.asarray(regimes_of_class)[y]) if regime_by_example is None else torch.from_numpy(rng.integers(0, regime_by_example, n))
        return x, torch.from_numpy(y.astype(np.int64)), r.long()
    xt, yt, rt = make(n_train)
    xe, ye, re_ = make(n_test)
    n_reg = int(max(rt.max(), re_.max())) + 1
    return Domain(name, kind, xt, yt, rt, xe, ye, re_, n_classes, [f"c{i}" for i in range(n_classes)], [f"{name}_r{i}" for i in range(n_reg)], {})


def toy_domains():
    return {"banking": toy_domain("banking", "text", 6, [0, 0, 1, 1, 2, 2], 900, 300, 1),
            "agnews": toy_domain("agnews", "text", 4, [0, 0, 1, 1], 700, 200, 2),
            "svamp": toy_domain("svamp", "text", 3, [0, 0, 0], 60, 40, 3),                   # fewer examples than a visit asks for
            "arith": toy_domain("arith", "digits", 12, [0] * 12, 1200, 400, 4, regime_by_example=3)}


def toy_ood():
    return {"stories": torch.nn.functional.normalize(torch.randn(20, 128), dim=1)}


def build(schedule, seed=0, **kw):
    kw = {"n_visit": 100, "n_visit_arith": 300, "n_test": 40, "revisit_k": 2, "n_phases_mix": 4, "n_phases_drift": 6, **kw}
    return SU.build_suite(schedule, toy_domains(), seed, toy_ood(), **kw)


def test_global_label_blocks_are_disjoint_and_every_regime_keeps_inside_its_domains_block():
    s = build("seq")
    assert s.n_classes == 6 + 4 + 3 + 12 and s.d == 128
    blocks = {d: (o, o + n) for d, (o, n) in s.label_blocks.items()}
    spans = sorted(blocks.values())
    assert all(spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))
    assert len(s.regimes) == 3 + 2 + 1 + 3
    for r in s.regimes.values():
        lo, hi = blocks[r.domain]
        assert r.y_te.min() >= lo and r.y_te.max() < hi and 0 < len(r.y_te) <= 40
    for ph in s.phases:
        for rid in ph.regime.unique().tolist():
            lo, hi = blocks[s.regimes[rid].domain]
            assert ph.y[ph.regime == rid].min() >= lo and ph.y[ph.regime == rid].max() < hi


def test_sequential_phases_are_one_regime_each_first_visits_then_revisits_with_the_domains_in_a_seed_dependent_order():
    s = build("seq")
    assert len(s.phases) == len(s.regimes) + 2
    assert all(len(ph.regime.unique()) == 1 for ph in s.phases)
    firsts = [bool(ph.first.all()) for ph in s.phases]
    assert firsts == [True] * len(s.regimes) + [False, False]
    order = [int(ph.regime[0]) for ph in s.phases[:len(s.regimes)]]
    assert sorted(order) == sorted(s.regimes)
    dom = [s.regimes[r].domain for r in order]
    for name in set(dom):
        pos = [i for i, d in enumerate(dom) if d == name]
        assert pos == list(range(pos[0], pos[0] + len(pos)))                                         # a domain's regimes are consecutive
    orders = {tuple(s2.regimes[int(p.regime[0])].domain for p in s2.phases[:len(s2.regimes)]) for s2 in (build("seq", seed=i) for i in range(6))}
    assert len(orders) > 1                                                                           # the domain order depends on the seed


def test_a_small_domain_gives_its_first_visit_without_duplicates():
    s = build("seq")
    rid = [r for r, reg in s.regimes.items() if reg.domain == "svamp"][0]
    first = [ph for ph in s.phases if ph.first.all() and int(ph.regime[0]) == rid][0]
    assert len(first.x) == 60                                                                        # all it has (60), not the 100 a visit asks for
    assert len({tuple(v.tolist()[:6]) for v in first.x}) == len(first.x)


def test_the_same_seed_gives_the_same_stream_and_another_seed_another_order_of_data():
    a, b, c = build("seq", seed=3), build("seq", seed=3), build("seq", seed=4)
    assert all(torch.equal(p.x, q.x) and torch.equal(p.y, q.y) for p, q in zip(a.phases, b.phases))
    assert not all(p.x.shape == q.x.shape and torch.equal(p.x, q.x) for p, q in zip(a.phases, c.phases))


def test_the_mixed_schedule_interleaves_every_regime_in_every_phase_and_spends_the_same_data_as_the_first_visits():
    m, q = build("mix"), build("seq")
    assert len(m.phases) == 4
    assert all(len(ph.regime.unique()) >= 4 for ph in m.phases)
    assert sum(len(p.x) for p in m.phases) == sum(len(p.x) for p in q.phases if p.first.all())
    assert bool(m.phases[0].first.all()) and not bool(m.phases[-1].first.any())


def test_the_drifting_schedule_moves_the_weight_from_regime_to_regime_over_time():
    d = build("drift")
    assert len(d.phases) == 6
    dominant = [int(torch.bincount(ph.regime, minlength=len(d.regimes)).argmax()) for ph in d.phases]
    assert len(set(dominant)) >= 3
    share = lambda ph: torch.bincount(ph.regime, minlength=len(d.regimes)).float() / len(ph.regime)
    assert float(share(d.phases[0]).max()) > 1.5 / len(d.regimes)


def test_the_algorithm_sees_only_inputs_and_labels_and_the_benchmark_keeps_the_rest():
    s = build("seq")
    batch = next(s.algo_batches(0, 32, seed=0))
    assert len(batch) == 2 and batch[0].shape == (32, 128)
    x, y, gt = next(s.batches(0, 32, seed=0))
    assert set(gt) == {"regime", "first", "phase"} and len(gt["regime"]) == len(y) == 32
    assert sum(len(x) for x, _ in s.algo_batches(0, 32, seed=0)) == len(s.phases[0].x)              # a phase is consumed completely, once
    ts = s.test_sets()
    assert set(ts) == set(s.regimes) and all(v[0].shape[1] == 128 for v in ts.values())
    assert set(s.ood) == {"stories"}


def test_unknown_schedule_is_refused():
    with pytest.raises(ValueError):
        build("zigzag")


def test_every_regime_remembers_which_test_examples_of_its_domain_it_holds():
    doms = toy_domains()
    s = SU.build_suite("seq", doms, 0, toy_ood(), n_visit=100, n_visit_arith=300, n_test=40, revisit_k=1)
    for r in s.regimes.values():
        idx = r.meta["test_index"]
        d, (off, _) = doms[r.domain], s.label_blocks[r.domain]
        assert len(idx) == len(r.y_te) and torch.equal(d.y_test[torch.as_tensor(idx)] + off, r.y_te) and torch.equal(d.x_test[torch.as_tensor(idx)], r.x_te)
