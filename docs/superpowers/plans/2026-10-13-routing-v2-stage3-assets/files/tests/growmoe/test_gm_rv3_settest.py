"""RV3: exact permutation tests of a batch against a reference sample (spec section 4.1)."""

import numpy as np
import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.settest import cap_rows, generator, max_exceedances, perm_pvalue, perm_reject

D = 16


def _cloud(n, centre, seed, spread=0.6):
    """n unit vectors scattered around the unit vector ``centre``."""
    g = np.random.default_rng(seed)
    z = centre[None, :] + spread * g.standard_normal((n, D)) / np.sqrt(D)
    return torch.nn.functional.normalize(torch.from_numpy(z.astype(np.float32)), dim=1)


def _axis(i):
    v = np.zeros(D)
    v[i] = 1.0
    return v


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def test_the_smallest_p_value_is_one_over_n_plus_one_and_a_far_batch_is_rejected():
    ref, far = _cloud(100, _axis(0), 1), _cloud(16, -_axis(0), 2)
    assert perm_pvalue(ref, far, n_perm=999, gen=_gen()) == pytest.approx(1 / 1000)
    assert perm_reject(ref, far, alpha=0.001, n_perm=999, gen=_gen()) == (True, 0, 999)


def test_a_batch_from_the_same_distribution_is_not_rejected_and_the_early_stop_saves_work():
    used = []
    for s in range(20):
        ref, batch = _cloud(100, _axis(0), 10 + s), _cloud(16, _axis(0), 100 + s)
        rejected, c, u = perm_reject(ref, batch, alpha=0.001, n_perm=999, gen=_gen(s))
        assert not rejected
        used.append(u)
    assert float(np.median(used)) <= 32                                                 # a known batch costs about one chunk


def test_the_early_stop_never_changes_the_decision():
    for s in range(60):
        shift = 0.0 if s % 3 == 0 else (0.25 if s % 3 == 1 else 0.6)
        centre = (1 - shift) * _axis(0) + shift * _axis(1)
        ref, batch = _cloud(80, _axis(0), s), _cloud(12, centre / np.linalg.norm(centre), 500 + s)
        for alpha, n_perm in ((0.05, 199), (0.001, 999)):
            early = perm_reject(ref, batch, alpha=alpha, n_perm=n_perm, gen=_gen(s))[0]
            full = perm_pvalue(ref, batch, n_perm=n_perm, gen=_gen(s)) <= alpha
            assert early == full, (s, alpha)


def test_the_false_alarm_rate_is_at_most_the_level_under_the_null():
    rejected = 0
    trials = 300
    for s in range(trials):
        ref, batch = _cloud(60, _axis(0), s), _cloud(12, _axis(0), 10_000 + s)
        rejected += perm_reject(ref, batch, alpha=0.05, n_perm=99, gen=_gen(s))[0]
    assert rejected / trials <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / trials)


def test_a_shifted_batch_is_found():
    shifted = (0.5 * _axis(0) + 0.87 * _axis(1))
    found = sum(perm_reject(_cloud(100, _axis(0), s), _cloud(16, shifted / np.linalg.norm(shifted), 900 + s), alpha=0.05, n_perm=199, gen=_gen(s))[0] for s in range(40))
    assert found >= 36


def test_mmd_detects_a_difference_in_spread_with_the_same_mean():
    g = np.random.default_rng(3)
    hits = 0
    trials = 40
    for s in range(trials):
        ref = torch.from_numpy(g.standard_normal((120, D)).astype(np.float32) * 0.5)
        batch = torch.from_numpy(g.standard_normal((40, D)).astype(np.float32) * 1.5)       # same mean, three times the spread
        hits += perm_reject(ref, batch, alpha=0.05, n_perm=199, gen=_gen(s), stat="mmd")[0]
    assert hits >= 0.9 * trials


def test_the_result_is_deterministic_for_a_seed_and_independent_of_the_chunk_size():
    ref, batch = _cloud(100, _axis(0), 4), _cloud(16, 0.9 * _axis(0) + 0.4 * _axis(1), 5)
    a = perm_reject(ref, batch, alpha=0.05, n_perm=199, gen=_gen(7))
    assert a == perm_reject(ref, batch, alpha=0.05, n_perm=199, gen=_gen(7))
    assert perm_pvalue(ref, batch, n_perm=199, gen=_gen(7), chunk=16) == perm_pvalue(ref, batch, n_perm=199, gen=_gen(7), chunk=64)


@pytest.mark.parametrize("stat", ["meandist", "mmd"])
def test_degenerate_inputs_do_not_produce_nan_or_a_false_rejection(stat):
    same = torch.ones(8, D) / np.sqrt(D)
    assert perm_pvalue(same, same[:4], n_perm=99, gen=_gen(), stat=stat) == pytest.approx(1.0)       # every point identical: every permutation ties
    assert perm_reject(same, same[:4], alpha=0.05, n_perm=99, gen=_gen(), stat=stat)[0] is False
    one = perm_reject(_cloud(50, _axis(0), 1), _cloud(1, -_axis(0), 2), alpha=0.05, n_perm=99, gen=_gen(), stat=stat)   # a batch of one row works
    assert isinstance(one[0], bool)


def test_bad_inputs_raise():
    ok = _cloud(10, _axis(0), 1)
    with pytest.raises(ValueError):
        perm_reject(ok, ok[:0])
    with pytest.raises(ValueError):
        perm_reject(ok[:0], ok)
    with pytest.raises(ValueError):
        perm_reject(ok, torch.ones(3, D + 1))
    with pytest.raises(ValueError):
        perm_reject(ok, ok[:3], stat="nope")


def test_a_test_that_could_never_reject_is_refused():
    ok = _cloud(20, _axis(0), 1)
    with pytest.raises(ValueError):
        perm_reject(ok, ok[:8], alpha=0.001, n_perm=99)                                  # smallest p-value 0.01 > 0.001: silently never rejecting is the worst failure
    assert perm_reject(ok, ok[:8], alpha=0.01, n_perm=99)[0] in (True, False)


def test_generator_is_a_pure_function_of_its_key():
    a, b, c = generator(1, 2, 3), generator(1, 2, 3), generator(1, 2, 4)
    assert torch.equal(torch.rand(5, generator=a), torch.rand(5, generator=b))
    assert not torch.equal(torch.rand(5, generator=generator(1, 2, 3)), torch.rand(5, generator=c))
    with pytest.raises(ValueError):
        generator(1, -2)


def test_cap_rows_keeps_small_inputs_and_subsamples_large_ones_reproducibly():
    z = torch.arange(100.0).reshape(50, 2)
    assert cap_rows(z, 50, generator(0)) is z
    cut = cap_rows(z, 10, generator(0))
    assert cut.shape == (10, 2) and torch.equal(cut, cap_rows(z, 10, generator(0)))
    assert (cut[1:, 0] > cut[:-1, 0]).all()                                              # original order kept
    with pytest.raises(ValueError):
        cap_rows(z, 0, generator(0))


def test_the_config_refuses_exactly_the_levels_the_permutation_count_cannot_reach():
    assert max_exceedances(0.001, 999) == 0 and max_exceedances(0.001, 99) == -1 and max_exceedances(0.05, 99) == 4 and max_exceedances(0.01, 199) == 1
    for alpha, split, perms in ((0.001, 0.01, 999), (0.001, 0.01, 99), (0.05, 0.01, 99), (0.05, 0.001, 999), (0.001, 0.01, 998), (0.02, 0.02, 49), (0.02, 0.02, 48)):
        refused = False
        try:
            PhyloCfg(set_gate_alpha=alpha, set_gate_split_alpha=split, set_gate_perms=perms)
        except ValueError:
            refused = True
        assert refused == (min(max_exceedances(alpha, perms), max_exceedances(split, perms)) < 0), (alpha, split, perms)


def test_a_slow_early_stop_is_caught_by_drawing_one_permutation_at_a_time():
    used = []
    for s in range(31):
        ref, batch = _cloud(100, _axis(0), 10 + s), _cloud(16, _axis(0), 100 + s)
        rejected, c, u = perm_reject(ref, batch, alpha=0.001, n_perm=999, gen=_gen(s), chunk=1)
        assert not rejected and c == 1                                                     # stopped at the very first exceedance
        used.append(u)
    assert float(np.median(used)) <= 2
