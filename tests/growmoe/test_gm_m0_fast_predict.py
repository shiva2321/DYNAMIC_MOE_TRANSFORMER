"""M0: the vectorised ``PhyloMoE.predict`` must return exactly what the original per-sample loop returned.

The original loop is kept as ``_predict_reference``; this file is the proof that the speed-up changes nothing a result depends on.
"""

import time

import numpy as np
import pytest
import torch

from growmoe import streams as S
from growmoe.config import PRE_STAGE4, PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature

D = 32
RULES = ["accept_then_logf", "argmax_logf", "argmax_p", "zscore_logf", "nearest_proto", "default"]


def _unit(n, seed, d=D):
    z = np.random.default_rng(seed).standard_normal((n, d))
    return torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))


def _signature(seed, n=128, centre=None, d=D):
    x = _unit(n, seed, d)
    if centre is not None:
        x = torch.nn.functional.normalize(x * 0.6 + centre, dim=1)
    sig = VmfSignature(d, 256, seed=seed)
    sig.update(x)
    sig.fit()
    return sig


def _fake_pool(n_experts, rule, no_conformal=False, share_first_two=False):
    """Experts with random signatures, every fifth one an infant (relaxed gate); optionally two identical experts (exact ties)."""
    pool = PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=4, seed=0, route_rule=rule, no_conformal=no_conformal, max_experts=4096, router=False))   # the reference loop is router-blind
    centres = _unit(n_experts, seed=99)
    for e in range(n_experts):
        sig = _signature(e, centre=centres[e])
        if share_first_two and e == 1:
            sig = _signature(0, centre=centres[0])                                      # identical prototypes to expert 0
        pool.add_expert(MLPExpert(D, 16, 1, 4, seed=e), parent=None, state="infant" if e % 5 == 0 else "active", meta={"sig": sig})
    return pool


def _same(fast, ref):
    assert torch.equal(fast.expert, ref.expert)
    assert torch.equal(fast.unknown, ref.unknown)
    assert torch.equal(fast.p_nov, ref.p_nov)
    assert torch.equal(fast.probs, ref.probs)


@pytest.mark.parametrize("rule", RULES)
def test_vectorised_predict_matches_the_reference_loop_on_a_trained_pool(rule):
    st = S.s2_permuted(0, n_regimes=3, schedule=[0, 1, 2, 0], n_train=200, n_test=60)
    pool = PhyloMoE(PhyloCfg(seed=0, q=0.02, route_rule=rule, **PRE_STAGE4))                  # the reference loop is router-blind, and the pool must grow to >= 2 experts on this short stream
    for x, y, _ in st.algo_batches(64, seed=0):
        pool.observe(x, y)
    assert len(pool.experts) >= 2
    x = torch.cat([reg.x_te for reg in st.regimes.values()])
    _same(pool.predict(x), pool._predict_reference(x))


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("no_conformal", [False, True])
def test_vectorised_predict_matches_the_reference_loop_with_infants_and_no_conformal_mode(rule, no_conformal):
    pool = _fake_pool(12, rule, no_conformal=no_conformal)
    x = torch.cat([_unit(150, seed=5), _unit(50, seed=6) * 0 + _unit(1, seed=7)])      # random points and exact duplicates
    _same(pool.predict(x), pool._predict_reference(x))


@pytest.mark.parametrize("rule", ["accept_then_logf", "argmax_logf", "argmax_p"])
def test_exact_ties_go_to_the_same_expert_as_in_the_reference_loop(rule):
    pool = _fake_pool(6, rule, share_first_two=True)
    x = _unit(80, seed=3)
    _same(pool.predict(x), pool._predict_reference(x))


def test_vectorised_predict_handles_an_empty_pool_like_the_reference():
    pool = PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=4, seed=0))
    x = _unit(5, seed=1)
    _same(pool.predict(x), pool._predict_reference(x))


def test_vectorised_predict_is_at_least_twice_as_fast_as_the_reference_loop():
    pool = _fake_pool(20, "accept_then_logf")
    x = _unit(1500, seed=1)

    def best_of(fn, reps=3):
        fn()
        times = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    t_ref = best_of(lambda: pool._predict_reference(x))
    t_fast = best_of(lambda: pool.predict(x))
    assert t_fast * 2.0 <= t_ref, f"fast {t_fast:.3f}s vs reference {t_ref:.3f}s"
