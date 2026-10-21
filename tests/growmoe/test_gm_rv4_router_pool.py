"""RV4 stage 4a inside PhyloMoE: lazy refit, restrict/free, accounting, pickling, flag off (spec sections 4.2, 4.4, 4.5)."""

import copy
import importlib.util
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from growmoe.accounting import pool_report
from growmoe.config import PRE_STAGE4, PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from growmoe.streams_m1 import function_regimes

from rv4_helpers import aniso, antipodal, unit

ROOT = Path(__file__).resolve().parents[2]
K = 4                                                                              # label space of the hand-built experts (they are never trained)
# These tests compare "the router off" with "the router on", so the base configuration keeps the router OFF (and the restrict mode these tests were written for) until a test asks
# otherwise; since the adoption (E-041) a plain PhyloCfg with a fast route rule has the router on and the mode "free".
BASE = {**PRE_STAGE4, "router_mode": "restrict"}


def _hand_pool(Xs, **kw):
    """A pool whose experts are built by hand from the given prototype matrices (one per expert). The experts are untrained: these tests look at WHICH expert answers."""
    cfg = dict(q=0.02, set_gate=False, group_vote=False, route_rule="accept_then_logf", **BASE)
    cfg.update(kw)
    pool = PhyloMoE(PhyloCfg(d=Xs[0].shape[1], hidden=8, n_classes=K, seed=1, **cfg))
    for c, X in enumerate(Xs):
        mod = MLPExpert(d_in=X.shape[1], hidden=8, depth=1, n_classes=K, seed=c)
        mod.mark_seen(torch.arange(K))
        sig = VmfSignature(d=X.shape[1], max_protos=256, seed=c)
        sig.update(X)
        sig.fit()
        pool.add_expert(mod, parent=None, state="active", meta={"sig": sig})
    return pool


def _split(X, y, k):
    return [X[y == c] for c in range(k)]


class _Prefer:
    """A scorer that ranks the highest-numbered expert first for every input (tests how the pool USES a scorer, whatever was fitted)."""

    def __init__(self, k):
        self.k = k

    def __call__(self, Z):
        s = torch.zeros(Z.shape[0], self.k)
        s[:, -1] = 1.0
        return s


def _two_blob_pool(**kw):
    """Two far-apart blobs, 80 stored prototypes each (a conformal p-value cannot fall below 1/(n+1), so fewer than 50 prototypes could never flag anything at q = 0.02),
    and 20 FRESH test inputs from each blob."""
    g = np.random.default_rng(0)
    d = 12
    draw = lambda i, n: torch.from_numpy(unit(np.eye(d)[i] * 8 + g.standard_normal((n, d))))              # noqa: E731
    pool = _hand_pool([draw(0, 80), draw(1, 80)], **kw)
    return pool, draw(0, 20), draw(1, 20)


def _stream_pool(**kw):
    st = function_regimes(100, overlap=0.6, n_regimes=4, n_train=128, n_test=64)
    cfg = PhyloCfg(seed=7, d=st.d, n_classes=st.n_classes, inner_steps=1, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", set_gate=False, group_vote=False, **{**BASE, **kw})
    return st, PhyloMoE(cfg)


# ------------------------------------------------------------------------------------------------ the flag off
def test_with_the_router_off_the_pool_is_bit_identical_to_the_recorded_golden_run():
    spec = importlib.util.spec_from_file_location("golden", ROOT / "research" / "growmoe_results" / "rv2c" / "golden.py")
    golden = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(golden)
    assert golden.differences(golden.QUICK) == []


def test_the_router_never_influences_training():
    st, off = _stream_pool(router=False)
    _, on = _stream_pool(router=True)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)                                                                       # one thread, so equal means bit-equal (as the golden check does)
    try:
        for x, y, ctx in st.algo_batches(64, 7):
            ro, rn = off.observe(x, y, ctx), on.observe(x, y, ctx)
            assert ro.assigned.tolist() == rn.assigned.tolist() and ro.novel.tolist() == rn.novel.tolist()
    finally:
        torch.set_num_threads(threads)
    assert sorted(off.experts) == sorted(on.experts)
    for e in off.experts:
        for a, b in zip(off.experts[e].module.parameters(), on.experts[e].module.parameters()):
            assert torch.equal(a, b)
    assert [(e.kind, e.step) for e in off.events] == [(e.kind, e.step) for e in on.events]          # nothing is fitted inside observe, so not even an event differs


# ------------------------------------------------------------------------------------------------ selection and routing
def test_with_the_router_on_and_density_selected_the_answers_equal_the_shipped_path():
    X, y = antipodal(seed=0)
    Xs = _split(X, y, 4)
    off, on = _hand_pool(Xs), _hand_pool(Xs, router=True)
    Xt, _ = antipodal(seed=1)
    a, b = off.predict(Xt), on.predict(Xt)
    assert on._current_route_rule().name == "density"
    assert torch.allclose(a.probs, b.probs, atol=1e-6) and torch.equal(a.expert, b.expert) and torch.equal(a.unknown, b.unknown) and torch.allclose(a.p_nov, b.p_nov, atol=1e-6)


def test_with_lda_selected_the_router_routes_better_than_density_and_leaves_novelty_alone():
    X, y = aniso(seed=0)
    Xs = _split(X, y, 6)
    off, on = _hand_pool(Xs), _hand_pool(Xs, router=True)
    Xt, yt = aniso(seed=1)
    a, b = off.predict(Xt), on.predict(Xt)
    assert on._current_route_rule().name == "lda"
    assert float((b.expert == yt).float().mean()) > float((a.expert == yt).float().mean()) + 0.2
    assert torch.equal(a.unknown, b.unknown) and torch.allclose(a.p_nov, b.p_nov, atol=1e-6)         # the density gate still decides known or unknown
    assert on.events.of_kind("router_refit")[-1].fields["selected"] == "lda"
    assert on.events.of_kind("router_switch")[-1].fields == {"from": "density", "to": "lda"}


def test_restrict_never_routes_to_an_expert_the_gate_rejects_and_free_does():
    pool, a, _ = _two_blob_pool(router=True)
    pool._current_route_rule().set("lda", _Prefer(2), [0, 1], {}, 80)                              # force a rule that always prefers expert 1
    xa = a[:20]
    pool.cfg.router_mode = "restrict"
    assert pool.predict(xa).expert.tolist() == [0] * 20                                           # blob A is accepted by expert 0 only
    pool.cfg.router_mode = "free"
    assert pool.predict(xa).expert.tolist() == [1] * 20


def test_restrict_falls_back_to_the_best_rule_score_when_the_gate_accepts_nobody():
    pool, _, _ = _two_blob_pool(router=True)
    pool._current_route_rule().set("lda", _Prefer(2), [0, 1], {}, 80)
    far = torch.from_numpy(unit(-np.eye(12)[5][None, :] + 0.01 * np.random.default_rng(1).standard_normal((6, 12))))
    out = pool.predict(far)
    assert bool(out.unknown.all()) and out.expert.tolist() == [1] * 6


def test_a_pool_with_one_expert_or_a_young_expert_keeps_the_density_rule():
    X, y = aniso(seed=0)
    one = _hand_pool(_split(X, y, 6)[:1], router=True)
    young = _hand_pool(_split(X, y, 6), router=True, router_min_protos=1000)
    Xt, _ = aniso(seed=1)
    for pool in (one, young):
        assert pool._current_route_rule().name == "density"
        assert pool.predict(Xt).expert.shape == (len(Xt),)


def test_an_empty_batch_is_handled_exactly_like_the_shipped_path():
    X, y = aniso(seed=0)
    Xs = _split(X, y, 6)
    results = []
    for pool in (_hand_pool(Xs), _hand_pool(Xs, router=True)):
        try:
            results.append(("ok", tuple(pool.predict(torch.empty(0, 12)).expert.shape)))
        except Exception as exc:                                                                  # noqa: BLE001 - the point is that the router adds no NEW way to fail
            results.append(("error", type(exc)))
    assert results[0] == results[1]


def test_a_router_with_a_reference_route_rule_is_refused_at_predict_time():
    pool, a, _ = _two_blob_pool(router=True)
    pool.cfg.route_rule = "default"                                                                # changed after construction: PhyloCfg cannot catch it
    with pytest.raises(ValueError):
        pool.predict(a[:4])


def test_the_lazy_refit_works_when_predict_is_called_under_no_grad():
    """Found by the arithmetic smoke run: L.Pool.predict is decorated with torch.no_grad(), and the refit inside predict trains a softmax candidate."""
    X, y = aniso(seed=0)
    pool = _hand_pool(_split(X, y, 6), router=True)
    Xt, yt = aniso(seed=1)
    with torch.no_grad():
        out = pool.predict(Xt)
    assert pool._route.name == "lda" and float((out.expert == yt).float().mean()) > 0.8


def test_group_answering_follows_the_router():
    pool, a, _ = _two_blob_pool(router=True, router_mode="free", group_vote=True, group_veto=False, group_min=4)
    pool._current_route_rule().set("lda", _Prefer(2), [0, 1], {}, 80)
    out = pool.predict_grouped(a[:10], torch.zeros(10, dtype=torch.long))
    assert out.expert.tolist() == [1] * 10


# ------------------------------------------------------------------------------------------------ lazy refit
def test_staleness_follows_the_expert_set_the_dirty_mark_and_the_age():
    X, y = aniso(seed=0)
    pool = _hand_pool(_split(X, y, 6)[:3], router=True, router_refit_every=5)
    assert pool._route_is_stale()                                                                  # never fitted
    pool._current_route_rule()
    assert not pool._route_is_stale() and pool._route.eids == [0, 1, 2]
    n = len(pool.events.of_kind("router_refit"))
    pool._current_route_rule()
    assert len(pool.events.of_kind("router_refit")) == n                                           # nothing changed: no refit
    pool._route_age = 4
    assert not pool._route_is_stale()
    pool._route_age = 5
    assert pool._route_is_stale()
    pool._current_route_rule()
    assert pool._route_age == 0 and len(pool.events.of_kind("router_refit")) == n + 1
    pool._route_dirty = True
    assert pool._route_is_stale()
    pool._current_route_rule()
    assert not pool._route_dirty
    mod = MLPExpert(d_in=12, hidden=8, depth=1, n_classes=K, seed=9)
    sig = VmfSignature(d=12, max_protos=256, seed=9)
    sig.update(_split(X, y, 6)[3])
    sig.fit()
    pool.add_expert(mod, parent=None, state="active", meta={"sig": sig})
    assert pool._route_is_stale()                                                                  # a new expert
    pool._current_route_rule()
    assert pool._route.eids == [0, 1, 2, 3]


def test_a_spawn_marks_the_router_stale_and_the_next_predict_refits_it():
    st, pool = _stream_pool(router=True)
    x_t = next(iter(st.test_sets().values()))[0]
    seen_spawn = False
    for x, y, ctx in st.algo_batches(64, 7):
        before = len(pool.experts)
        pool.observe(x, y, ctx)
        if len(pool.experts) > before >= 1:
            seen_spawn = True
            assert pool._route_dirty
            pool.predict(x_t)
            assert not pool._route_dirty and pool._route.eids == sorted(pool.experts)
            assert pool.events.of_kind("router_refit")[-1].fields["experts"] == len(pool.experts)
    assert seen_spawn


# ------------------------------------------------------------------------------------------------ accounting, determinism, pickling
def test_accounting_counts_the_parameters_and_multiply_adds_of_the_selected_rule():
    X, y = aniso(seed=0)
    Xs = _split(X, y, 6)
    on, off = _hand_pool(Xs, router=True), _hand_pool(Xs)
    on._current_route_rule()
    assert on._route.name == "lda"
    r_on, r_off = pool_report(on, 12), pool_report(off, 12)
    k, d = 6, 12
    assert r_on.params - r_off.params == k * d + k
    assert r_on.gate_macs - r_off.gate_macs == k * d
    assert r_on.memory_bytes - r_off.memory_bytes == 4 * (k * d + k)
    X2, y2 = antipodal(seed=0)
    dense = _hand_pool(_split(X2, y2, 4), router=True)
    dense._current_route_rule()
    ref = _hand_pool(_split(X2, y2, 4))
    assert pool_report(dense, 16) == pool_report(ref, 16)                                          # density selected: not a byte more


def test_two_identical_pools_select_and_route_identically():
    X, y = aniso(seed=0)
    a, b = _hand_pool(_split(X, y, 6), router=True), _hand_pool(_split(X, y, 6), router=True)
    Xt, _ = aniso(seed=1)
    pa, pb = a.predict(Xt), b.predict(Xt)
    assert a._route.name == b._route.name and a._route.cv == pytest.approx(b._route.cv)
    assert torch.equal(pa.expert, pb.expert) and torch.allclose(pa.probs, pb.probs, atol=1e-6)


def test_a_pool_that_holds_a_fitted_router_can_be_pickled_and_copied():
    X, y = aniso(seed=0)
    pool = _hand_pool(_split(X, y, 6), router=True)
    Xt, _ = aniso(seed=1)
    ref = pool.predict(Xt).expert
    assert pool._route.name == "lda"
    for clone in (pickle.loads(pickle.dumps(pool)), copy.deepcopy(pool)):
        assert torch.equal(clone.predict(Xt).expert, ref)

# ------------------------------------------------------------------------------------------------ the boundaries of the decision
def test_restrict_accepts_an_expert_whose_p_value_equals_q():
    """q = 0.25 is a dyadic number, so a p-value of exactly q can be written down in float32: the gate's rule is 'accepted when p >= q'."""
    pool, a, _ = _two_blob_pool(router=True, q=0.25)
    pool._current_route_rule().set("lda", _Prefer(2), [0, 1], {}, 80)               # always prefers expert 1
    P = torch.tensor([[0.25, 0.0]] * 3)                                              # expert 0 sits exactly at q, expert 1 far below
    out = pool._route_by_router(a[:3], [0, 1], P, torch.zeros(3, 2), torch.zeros(3, dtype=torch.long))
    assert out.tolist() == [0, 0, 0]


def test_an_expert_with_exactly_router_min_protos_prototypes_is_not_young():
    X, y = aniso(seed=0)                                                             # 16 prototypes per expert
    assert _hand_pool(_split(X, y, 6), router=True, router_min_protos=16)._current_route_rule().name == "lda"
    assert _hand_pool(_split(X, y, 6), router=True, router_min_protos=17)._current_route_rule().name == "density"
