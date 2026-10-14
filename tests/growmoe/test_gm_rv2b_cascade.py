"""RV2-B: the two-stage gate (spec section 5). Fake pools with random signatures, every seventh expert an infant, as in test_gm_m0_fast_predict."""

import numpy as np
import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.conformal import benjamini_hochberg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature

D = 32


def _unit(n, seed, d=D):
    z = np.random.default_rng(seed).standard_normal((n, d))
    return torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))


def _signature(seed, n=128, centre=None):
    x = _unit(n, seed)
    if centre is not None:
        x = torch.nn.functional.normalize(x * 0.6 + centre, dim=1)
    sig = VmfSignature(D, 256, seed=seed)
    sig.update(x)
    sig.fit()
    return sig


def _fake_pool(n_experts, infants=True, **cfg):
    pool = PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=4, seed=0, q=0.02, max_experts=4096, route_rule="accept_then_logf", **cfg))
    centres = _unit(n_experts, seed=99)
    for e in range(n_experts):
        state = "infant" if infants and e % 7 == 0 else "active"
        pool.add_expert(MLPExpert(D, 16, 1, 4, seed=e), parent=None, state=state, meta={"sig": _signature(e, centre=centres[e])})
    return pool, centres


def _near(centres, n, seed):
    idx = torch.from_numpy(np.random.default_rng(seed).integers(0, len(centres), n))
    return torch.nn.functional.normalize(centres[idx] * 0.6 + _unit(n, seed + 1), dim=1)


def _old_observe_loop(pool, x, eids):
    """The per-expert loop ``observe`` used before it was routed through ``_gate_matrices``."""
    p_list, logf_list = [], []
    for eid in eids:
        rec = pool.experts[eid]
        sig = rec.meta["sig"]
        p_k = sig.pvalue(x)
        if rec.state == "infant":
            p_k = torch.clamp(p_k * pool.cfg.infant_relax, max=1.0)
        p_list.append(p_k)
        logf_list.append(sig.logpdf(x))
    return torch.stack(p_list, dim=1), torch.stack(logf_list, dim=1)


def test_gate_matrices_equal_the_original_observe_loop_bit_for_bit():
    pool, centres = _fake_pool(12)
    x = _near(centres, 64, 5)
    eids = list(pool.experts)
    P, F_ = pool._gate_matrices(x, eids, phase="observe")
    P0, F0 = _old_observe_loop(pool, x, eids)
    assert torch.equal(P, P0) and torch.equal(F_, F0)


def test_observe_counts_its_gate_work_under_its_own_phase():
    pool, centres = _fake_pool(6, infants=False)
    x = _near(centres, 64, 7)
    stored_before = sum(rec.meta["sig"].n for rec in pool.experts.values())             # the scan happens before training or a spawn changes anything
    pool.observe(x, torch.zeros(64, dtype=torch.long))
    s = pool.gate_stats["observe"]
    assert s == {"samples": 64, "stage1": 0, "stage2": 64 * stored_before * D, "second_look": 0}
    assert pool.gate_stats["predict"]["samples"] == 0


def test_predict_counts_the_exact_scan_per_input():
    pool, centres = _fake_pool(6, infants=False)
    x = _near(centres, 50, 8)
    pool.predict(x)
    stored = sum(rec.meta["sig"].n for rec in pool.experts.values())
    assert pool.gate_stats["predict"] == {"samples": 50, "stage1": 0, "stage2": 50 * stored * D, "second_look": 0}


def test_reset_gate_stats_zeroes_the_counters():
    pool, centres = _fake_pool(4, infants=False)
    pool.predict(_near(centres, 10, 9))
    pool.reset_gate_stats()
    assert all(v == 0 for ph in pool.gate_stats.values() for v in ph.values())


def _cascade_pool(n, shortlist=2, k=4, **kw):
    return _fake_pool(n, gate_shortlist=shortlist, gate_fingerprint_k=k, gate_shortlist_min_experts=4, **kw)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("shortlist", [1, 2, 3])
def test_novelty_flags_are_identical_to_the_exact_gate(seed, shortlist):
    pool, centres = _cascade_pool(20, shortlist)
    x = torch.cat([_near(centres, 60, seed), _unit(30, seed + 50)])                     # known-ish inputs plus pure noise (mostly novel)
    eids = list(pool.experts)
    P_e, F_e = pool._gate_exact(x, eids)
    P_c, F_c = pool._gate_matrices(x, eids)
    q = pool.cfg.q
    pe, pc = P_e.max(1).values, P_c.max(1).values
    assert torch.equal(benjamini_hochberg(pe, q), benjamini_hochberg(pc, q))           # THE guarantee
    small = pe <= q
    assert torch.allclose(pc[small], pe[small], atol=0.01)                              # at or below q the p-value is exact (one conformal step of slack for float noise)
    assert bool((pc[~small] <= pe[~small] + 0.01).all())                                # elsewhere a lower bound
    seen = F_c > -float("inf")
    assert torch.allclose(F_c[seen], F_e[seen], atol=1e-3) and torch.allclose(P_c[seen], P_e[seen], atol=0.01)   # every scored pair carries the exact value


def test_a_shortlist_covering_the_pool_is_the_exact_path_bit_for_bit():
    pool, centres = _fake_pool(6, gate_shortlist=6, gate_shortlist_min_experts=2)
    x = _near(centres, 40, 3)
    eids = list(pool.experts)
    P_c, F_c = pool._gate_matrices(x, eids)
    P_e, F_e = pool._gate_exact(x, eids)
    assert torch.equal(P_c, P_e) and torch.equal(F_c, F_e) and pool.gate_stats["predict"]["stage1"] == 0


def test_below_the_minimum_number_of_experts_the_exact_path_is_used():
    pool, centres = _fake_pool(10, gate_shortlist=2, gate_shortlist_min_experts=16)
    pool._gate_matrices(_near(centres, 20, 3), list(pool.experts))
    assert pool.gate_stats["predict"]["stage1"] == 0


def test_the_cascade_is_used_only_on_the_conformal_path():
    pool, centres = _cascade_pool(12, no_conformal=True)
    pool._gate_matrices(_near(centres, 20, 3), list(pool.experts))
    assert pool.gate_stats["predict"]["stage1"] == 0


def test_a_batch_in_which_every_sample_is_novel_gets_a_second_look_for_every_row():
    pool, centres = _cascade_pool(12, infants=False)
    x = -_near(centres, 30, 4)                                                          # antipodal to the experts: nothing accepts it
    eids = list(pool.experts)
    P_c, F_c = pool._gate_matrices(x, eids)
    P_e, F_e = pool._gate_exact(x, eids)
    stored = sum(pool.experts[e].meta["sig"].n for e in eids)
    assert pool.gate_stats["predict"]["second_look"] == 30 * stored * D
    assert torch.allclose(P_c, P_e, atol=0.01) and torch.allclose(F_c, F_e, atol=1e-3)


def test_an_all_infant_pool_is_scored_exactly():
    pool, centres = _cascade_pool(8, infants=False)
    for rec in pool.experts.values():
        pool.set_state(rec.id, "infant")
    x = _near(centres, 40, 6)
    eids = list(pool.experts)
    P_c, _ = pool._gate_matrices(x, eids)
    P_e, _ = pool._gate_exact(x, eids)
    assert torch.allclose(P_c, P_e, atol=0.01)
    stored = sum(pool.experts[e].meta["sig"].n for e in eids)
    assert pool.gate_stats["predict"]["stage2"] == 40 * stored * D


def test_a_single_sample_works_through_the_cascade():
    pool, centres = _cascade_pool(12)
    P, F_ = pool._gate_matrices(_near(centres, 1, 2), list(pool.experts))
    assert P.shape == (1, 12) and F_.shape == (1, 12)


def test_the_counters_follow_the_two_stages():
    pool, centres = _cascade_pool(20, shortlist=2, k=4, infants=False)
    x = _near(centres, 64, 11)
    eids = list(pool.experts)
    pool._gate_matrices(x, eids)
    s = pool.gate_stats["predict"]
    n_max = max(pool.experts[e].meta["sig"].n for e in eids)
    assert s["samples"] == 64 and s["stage1"] == 64 * 20 * 4 * D
    assert 64 * 2 * n_max * D >= s["stage2"] >= 64 * 2 * min(pool.experts[e].meta["sig"].n for e in eids) * D


def test_the_fingerprint_is_rebuilt_only_after_enough_new_points():
    pool, centres = _cascade_pool(6, gate_fingerprint_refresh=50, infants=False)
    eids = list(pool.experts)
    pool._gate_matrices(_near(centres, 10, 1), eids)
    sig = pool.experts[eids[0]].meta["sig"]
    first = pool.experts[eids[0]].meta["fp"]
    sig.update(_unit(20, 5))
    pool._gate_matrices(_near(centres, 10, 1), eids)
    assert pool.experts[eids[0]].meta["fp"] is first                                    # 20 < 50: not rebuilt
    sig.update(_unit(40, 6))
    pool._gate_matrices(_near(centres, 10, 1), eids)
    assert pool.experts[eids[0]].meta["fp"] is not first


def test_two_identical_pools_give_identical_cascade_results():
    a, centres = _cascade_pool(12)
    b, _ = _cascade_pool(12)
    x = _near(centres, 40, 4)
    Pa, Fa = a._gate_matrices(x, list(a.experts))
    Pb, Fb = b._gate_matrices(x, list(b.experts))
    assert torch.equal(Pa, Pb) and torch.equal(Fa, Fb)


def test_predict_unknown_flags_do_not_change_when_the_cascade_is_switched_on():
    pool, centres = _cascade_pool(16)
    x = torch.cat([_near(centres, 60, 2), _unit(30, 77)])
    on = pool.predict(x)
    pool.cfg.gate_shortlist = 0
    off = pool.predict(x)
    assert torch.equal(on.unknown, off.unknown)

