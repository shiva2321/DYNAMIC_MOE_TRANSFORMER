"""RV2-A: overlapping regimes and a mixed tail for the function-regime stream (spec section 4).

``overlap`` gives the regimes' centres a shared component so their pairwise cosine is about ``overlap``; at 0 the stream is bit-identical to
the pilot. ``mixed_tail`` appends 70/30 segments of regimes the pool already knows, built from their unused second halves.
"""

import pytest
import torch

from growmoe.streams import _balanced_teacher, _long, _rng, _tensor, _unit
from growmoe.streams_m1 import function_regimes


def _pilot(seed, n_regimes, n_train, n_test, d=128, n_classes=8):
    """function_regimes exactly as it was before ``overlap`` existed."""
    rng = _rng(seed, 0xF5)
    centres = _unit(rng.standard_normal((n_regimes, d)))
    pools, tests = {}, {}
    for r in range(n_regimes):
        teacher = _balanced_teacher(8, 1, d, n_classes, seed=seed * 1000 + r)

        def draw(n):
            z = rng.standard_normal((n, d))
            y = teacher(_tensor(_unit(z))).argmax(1).detach().numpy()
            return _unit(8.0 * centres[r] + z), y

        pools[r] = draw(2 * n_train)
        tests[r] = draw(n_test)
    segs, visits = [], {}
    for r in list(range(n_regimes)) + [0, n_regimes // 2]:
        v = visits.get(r, 0)
        visits[r] = v + 1
        sl = slice(v * n_train, (v + 1) * n_train)
        segs.append((_tensor(pools[r][0][sl]), _long(pools[r][1][sl])))
    return segs, {r: (_tensor(x), _long(y)) for r, (x, y) in tests.items()}


def _mean_pair_cosine(st):
    c = torch.as_tensor(st.meta["centres"], dtype=torch.float32)
    g = c @ c.T
    n = len(c)
    return float((g.sum() - g.diag().sum()) / (n * (n - 1)))


def test_overlap_zero_is_bit_identical_to_the_pilot_stream():
    st = function_regimes(5, n_regimes=4, n_train=64, n_test=32, overlap=0.0)
    segs, tests = _pilot(5, 4, 64, 32)
    assert len(st.segments) == len(segs)
    for seg, (x, y) in zip(st.segments, segs):
        assert torch.equal(seg.x, x) and torch.equal(seg.y, y)
    for r, (x, y) in tests.items():
        assert torch.equal(st.regimes[r].x_te, x) and torch.equal(st.regimes[r].y_te, y)


@pytest.mark.parametrize("rho", [0.3, 0.6])
def test_overlap_sets_the_mean_pairwise_cosine_of_the_centres(rho):
    st = function_regimes(1, n_regimes=10, n_train=32, n_test=16, overlap=rho)
    assert abs(_mean_pair_cosine(st) - rho) < 0.05
    assert st.meta["overlap"] == rho


def test_regimes_stay_separable_from_the_inputs_alone_at_moderate_overlap():
    st = function_regimes(2, n_regimes=6, n_train=64, n_test=200, overlap=0.3)
    centres = torch.as_tensor(st.meta["centres"], dtype=torch.float32)
    for r, reg in st.regimes.items():
        assert float(((reg.x_te @ centres.T).argmax(1) == r).float().mean()) >= 0.99


def test_mixed_tail_appends_seventy_thirty_segments_of_regimes_that_are_not_revisited():
    st = function_regimes(3, n_regimes=10, n_train=100, n_test=40, mixed_tail=2)
    assert len(st.segments) == 12 + 2
    tail = st.segments[12:]
    assert [s.regime for s in tail] == [1, 3]                                           # majors of the pairs (1, 2) and (3, 4)
    assert st.meta["mixed_pairs"] == [(1, 2), (3, 4)] and st.meta["minority_regimes"] == [2, 4]
    assert st.meta["schedule"] == [seg.regime for seg in st.segments]
    centres = torch.as_tensor(st.meta["centres"], dtype=torch.float32)
    for seg, (a, b) in zip(tail, st.meta["mixed_pairs"]):
        assert not seg.first_visit and len(seg.y) == 100
        nearest = (seg.x @ centres.T).argmax(1)
        assert abs(int((nearest == a).sum()) - 70) <= 1 and abs(int((nearest == b).sum()) - 30) <= 1


def test_the_tail_uses_fresh_samples():
    st = function_regimes(3, n_regimes=10, n_train=100, n_test=40, mixed_tail=1)
    first, tail = st.segments[1].x, st.segments[12].x                                  # regime 1's first visit, then the tail
    assert not any(bool((first == row).all(dim=1).any()) for row in tail[:20])


def test_the_default_stream_is_unchanged_by_the_new_options():
    st = function_regimes(3, n_regimes=10, n_train=100, n_test=40)
    assert len(st.segments) == 12 and st.meta["mixed_pairs"] == [] and st.meta["minority_regimes"] == []


def test_bad_arguments_are_refused():
    with pytest.raises(ValueError):
        function_regimes(0, n_regimes=10, overlap=1.0)
    with pytest.raises(ValueError):
        function_regimes(0, n_regimes=10, overlap=-0.1)
    with pytest.raises(ValueError):
        function_regimes(0, n_regimes=10, mixed_tail=5)                                 # only four disjoint pairs exist for 10 regimes
    with pytest.raises(ValueError):
        function_regimes(0, n_regimes=10, mixed_tail=1, minor_share=0.6)
