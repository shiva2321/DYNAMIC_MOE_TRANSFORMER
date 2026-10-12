from functools import lru_cache

import numpy as np
import pytest
import torch

from growmoe import streams as S
from growmoe.experts import MLPExpert
from growmoe.frontend import HashFrontend


# ------------------------------------------------------------------ cached builders (building is the slow part)
@lru_cache(maxsize=None)
def _s1(seed=0):
    return S.s1_types(seed)


@lru_cache(maxsize=None)
def _s2(seed=0, n_regimes=30):
    return S.s2_permuted(seed, n_regimes=n_regimes)


@lru_cache(maxsize=None)
def _s3(seed=0):
    return S.s3_drift(seed)


@lru_cache(maxsize=None)
def _s4(seed=0, outlier_rate=0.01, label_noise=0.1):
    return S.s4_outliers(seed, outlier_rate=outlier_rate, label_noise=label_noise)


@lru_cache(maxsize=None)
def _s6(seed=0, n_regimes=60):
    return S.s6_long(seed, n_regimes=n_regimes)


@lru_cache(maxsize=None)
def _s7(seed=0, with_ctx=False):
    return S.s7_mapping(seed, with_ctx=with_ctx)


def _train(stream):
    return torch.cat([s.x for s in stream.segments]), torch.cat([s.y for s in stream.segments])


def _class_means(x, y, n_classes):
    return torch.stack([torch.nn.functional.normalize(x[y == c].mean(0), dim=0) for c in range(n_classes)])


# ------------------------------------------------------------------------------- data model / iterators
def test_batches_chunk_each_segment_and_carry_ground_truth():
    st = S.s2_permuted(0, n_regimes=3, n_train=70, n_test=20)
    seen, n = [], 0
    for x, y, gt in st.batches(32, seed=1):
        assert x.shape[1] == st.d and x.shape[0] == y.shape[0] <= 32
        assert set(gt) >= {"regime", "segment", "first_visit", "is_outlier", "y_clean"}
        assert gt["is_outlier"].shape == (x.shape[0],) and gt["is_outlier"].dtype == torch.bool
        seen.append((gt["segment"], gt["regime"], gt["first_visit"]))
        n += x.shape[0]
    assert n == sum(len(s.y) for s in st.segments) == 3 * 70
    assert [s for s, _, _ in seen] == sorted(s for s, _, _ in seen)                 # segments are played in order
    assert {r for _, r, _ in seen} == {0, 1, 2}


def test_algorithm_facing_iterator_strips_ground_truth():
    st = S.s2_permuted(0, n_regimes=3, n_train=70, n_test=20)
    full, algo = list(st.batches(32, seed=1)), list(st.algo_batches(32, seed=1))
    assert len(full) == len(algo)
    for (x, y, _), item in zip(full, algo):
        assert len(item) == 3 and not isinstance(item[2], dict)
        assert torch.equal(x, item[0]) and torch.equal(y, item[1])
        assert item[2] is None                                                       # S2 has no context


def test_batch_seed_changes_the_order_but_not_the_content():
    st = S.s2_permuted(0, n_regimes=2, n_train=64, n_test=20)
    a = torch.cat([x for x, _, _ in st.batches(16, seed=1)])
    b = torch.cat([x for x, _, _ in st.batches(16, seed=1)])
    c = torch.cat([x for x, _, _ in st.batches(16, seed=2)])
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.allclose(a.sum(0), c.sum(0), atol=1e-3)


def test_test_sets_are_per_regime():
    st = _s2()
    tests = st.test_sets()
    assert set(tests) == set(range(30))
    assert all(x.shape == (500, st.d) and y.shape == (500,) for x, y in tests.values())


# ----------------------------------------------------------------------------------- generic properties
BUILDERS = {
    "s1": lambda seed: S.s1_types(seed, n_train=60, n_test=20, revisit_size=20),
    "s2": lambda seed: S.s2_permuted(seed, n_regimes=3, n_train=60, n_test=20),
    "s3": lambda seed: S.s3_drift(seed, n_samples=400),
    "s4": lambda seed: S.s4_outliers(seed, n_regimes=2, n_train=100, n_test=20),
    "s5": lambda seed: S.s5_ladder(seed, width=8, depth=2, n_train=100, n_test=20),
    "s6": lambda seed: S.s6_long(seed, n_regimes=6, n_train=40, n_test=20),
    "s7": lambda seed: S.s7_mapping(seed, with_ctx=True, n_train=40, n_test=20),
}


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_builders_are_deterministic_in_the_seed(name):
    a, b, c = BUILDERS[name](0), BUILDERS[name](0), BUILDERS[name](1)
    xa, ya = _train(a)
    xb, yb = _train(b)
    xc, _ = _train(c)
    assert torch.equal(xa, xb) and torch.equal(ya, yb)
    assert xa.shape != xc.shape or not torch.equal(xa, xc)
    for r in a.regimes:
        assert torch.equal(a.test_sets()[r][0], b.test_sets()[r][0])


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_every_input_is_a_unit_vector_and_labels_are_in_range(name):
    st = BUILDERS[name](0)
    xs = [_train(st)[0]] + [x for x, _ in st.test_sets().values()]
    for x in xs:
        assert x.shape[1] == st.d and torch.isfinite(x).all()
        assert torch.allclose(x.norm(dim=1), torch.ones(x.shape[0]), atol=1e-4)
    ys = [_train(st)[1]] + [y for _, y in st.test_sets().values()]
    assert all(int(y.min()) >= 0 and int(y.max()) < st.n_classes for y in ys)


@pytest.mark.parametrize("name", ["s1", "s2", "s4", "s6", "s7"])
def test_train_and_test_samples_come_from_disjoint_source_ids(name):
    st = BUILDERS[name](0)
    for r, regime in st.regimes.items():
        train_ids = set(torch.cat([s.src_ids for s in st.segments if s.regime == r]).tolist())
        assert train_ids and not (train_ids & set(regime.te_ids.tolist()))


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_first_visit_is_true_for_exactly_one_segment_per_regime(name):
    st = BUILDERS[name](0)
    firsts = [s.regime for s in st.segments if s.first_visit]
    assert sorted(firsts) == sorted(st.regimes) and len(set(firsts)) == len(firsts)
    seen = set()
    for s in st.segments:
        assert s.first_visit == (s.regime not in seen)
        seen.add(s.regime)


# ---------------------------------------------------------------------------------------------------- S1
def test_s1_has_seven_regimes_of_five_types_with_disjoint_label_blocks():
    st = _s1()
    assert len(st.regimes) == 7 and st.n_classes == 30
    assert len({r.meta["type"] for r in st.regimes.values()}) >= 5
    blocks = [set(r.meta["labels"]) for r in st.regimes.values()]
    assert sum(len(b) for b in blocks) == len(set().union(*blocks)) <= 30
    first = [s for s in st.segments if s.first_visit]
    assert all(len(s.y) == 600 for s in first)
    assert all(x.shape[0] == 200 for x, _ in st.test_sets().values())
    for s in st.segments:
        assert set(s.y.tolist()) <= set(st.regimes[s.regime].meta["labels"])


def test_s1_replays_earlier_types_after_the_first_pass():
    st = _s1()
    order = [s.regime for s in st.segments]
    assert order[:7] == list(range(7)) and len(order) > 7
    assert all(r in range(7) for r in order[7:]) and not any(s.first_visit for s in st.segments[7:])


def test_s1_regimes_occupy_distinguishable_parts_of_the_sphere():
    # each type must be recognisably itself: a held-out sample is closer to its own regime's training samples
    st = _s1()
    train = {r: torch.cat([s.x for s in st.segments if s.regime == r]) for r in st.regimes}
    hits = total = 0
    for r, (x, _) in st.test_sets().items():
        own = (x @ train[r].T).max(dim=1).values
        other = torch.stack([(x @ train[q].T).max(dim=1).values for q in st.regimes if q != r]).max(dim=0).values
        hits += int((own > other).sum())
        total += x.shape[0]
    assert hits / total > 0.9


# ---------------------------------------------------------------------------------------------------- S2
def test_s2_schedule_shared_labels_and_segment_sizes():
    st = _s2()
    expect = list(range(15)) + [3, 7, 0, 12] + list(range(15, 30)) + [5, 20, 1]
    assert [s.regime for s in st.segments] == expect
    assert st.n_classes == 10 and all(len(s.y) == 1000 for s in st.segments)
    assert all(set(s.y.tolist()) <= set(range(10)) for s in st.segments)


def test_s2_regimes_are_distinguishable_despite_sharing_a_common_direction():
    # Digit pixels are non-negative and the hash signs belong to output positions, so they cancel across regimes:
    # every permuted regime keeps a large common component (regime means have cos ~0.75 with the global mean and
    # class-mean |cos| across regimes is ~0.47). The regimes are NOT orthogonal, but they are cleanly separable.
    st = _s2()
    x, _ = _train(st)
    regime = torch.cat([torch.full((len(s.y),), s.regime) for s in st.segments])
    train = {r: x[regime == r] for r in range(10)}
    hits = total = 0
    for r in range(10):
        xt, _ = st.test_sets()[r]
        own = (xt @ train[r].T).max(1).values
        other = torch.stack([(xt @ train[q].T).max(1).values for q in range(10) if q != r]).max(0).values
        hits += int((own > other).sum())
        total += xt.shape[0]
    assert hits / total > 0.99                                                      # measured 1.000 over all 30 regimes
    u = torch.nn.functional.normalize(x.mean(0), dim=0)
    assert torch.nn.functional.normalize(train[0].mean(0), dim=0) @ u > 0.5          # the shared direction is real


def test_s2_digits_are_learnable_within_a_regime():
    st = _s2()
    x, y = _train(st)
    regime = torch.cat([torch.full((len(s.y),), s.regime) for s in st.segments])
    xr, yr = x[regime == 0], y[regime == 0]
    means = _class_means(xr, yr, 10)
    xt, yt = st.test_sets()[0]
    assert ((xt @ means.T).argmax(1) == yt).float().mean() > 0.8                   # nearest class-mean already works


# ---------------------------------------------------------------------------------------------------- S3
def test_s3_drift_is_gradual_and_has_five_stage_test_sets():
    st = _s3()
    assert len(st.regimes) == 5 and sum(len(s.y) for s in st.segments) == 4000
    x, y = _train(st)
    third = len(y) // 3
    m0, m1, m2 = (_class_means(x[i * third:(i + 1) * third], y[i * third:(i + 1) * third], 10) for i in range(3))
    near = (m0 * m1).sum(1).mean()
    far = (m0 * m2).sum(1).mean()
    assert near > far + 0.1                                                         # neighbours resemble each other more than distant stages


# ---------------------------------------------------------------------------------------------------- S4
def test_s4_injects_the_requested_outlier_and_label_noise_rates():
    st = _s4()
    flags = torch.cat([s.is_outlier for s in st.segments])
    y = torch.cat([s.y for s in st.segments])
    clean = torch.cat([s.y_clean for s in st.segments])
    assert abs(flags.float().mean().item() - 0.01) < 0.005
    noisy = (y[~flags] != clean[~flags]).float().mean().item()
    assert abs(noisy - 0.1) < 0.02
    assert all(set(y_te.tolist()) <= set(range(10)) for _, y_te in st.test_sets().values())


def test_s4_outliers_are_far_from_the_data_they_replace():
    st = _s4()
    ratios = []
    for r in st.regimes:
        segs = [s for s in st.segments if s.regime == r]
        x = torch.cat([s.x for s in segs])
        out = torch.cat([s.is_outlier for s in segs])
        if int(out.sum()) == 0:
            continue
        sim = x[out] @ x[~out].T
        inlier = x[~out][:200] @ x[~out].T
        inlier.fill_diagonal_(-1.0)
        ratios.append((sim.max(dim=1).values.mean().item(), inlier.max(dim=1).values.mean().item()))
    assert np.mean([o for o, _ in ratios]) < np.mean([i for _, i in ratios]) - 0.2


def test_s4_zero_rates_reproduce_the_clean_prefix():
    st = S.s4_outliers(0, outlier_rate=0.0, label_noise=0.0, n_regimes=2, n_train=100, n_test=20)
    assert not any(bool(s.is_outlier.any()) for s in st.segments)
    assert all(torch.equal(s.y, s.y_clean) for s in st.segments)


# ---------------------------------------------------------------------------------------------------- S5
@pytest.mark.parametrize("width,depth", [(8, 1), (32, 2), (128, 3)])
def test_s5_teacher_has_the_stated_shape_and_labels_are_its_own_argmax(width, depth):
    st = S.s5_ladder(0, width=width, depth=depth, n_train=300, n_test=100)
    teacher = st.regimes[0].meta["teacher"]
    assert teacher.hidden == width and len(teacher.blocks) == depth and st.d == 128 and st.n_classes == 8
    x, y = _train(st)
    xt, yt = st.test_sets()[0]
    with torch.no_grad():
        assert (teacher(x).argmax(1) == y).all() and (teacher(xt).argmax(1) == yt).all()


def test_s5_teacher_labels_are_roughly_balanced():
    st = S.s5_ladder(0, width=32, depth=2, n_train=2000, n_test=500)
    freq = torch.bincount(_train(st)[1], minlength=8).float() / 2000
    assert freq.max() < 0.3 and freq.min() > 0.04                                   # ideal 0.125; an unbalanced teacher can reach 0.8


def test_s5_default_size_is_large_enough_for_a_student_to_learn_the_teacher():
    # at the spec's 2,000 samples a width-8 teacher is data-limited (student accuracy ~0.55); at the default 20,000 it is not
    st = S.s5_ladder(0, width=8, depth=1)
    assert len(st.segments[0].y) == 20000
    x, y = _train(st)
    xt, yt = st.test_sets()[0]
    student = MLPExpert(d_in=128, hidden=32, depth=1, n_classes=8, seed=0)
    student.mark_seen(torch.arange(8))
    opt, g = torch.optim.Adam(student.parameters(), lr=3e-3), np.random.default_rng(0)
    for _ in range(1500):
        i = torch.from_numpy(g.integers(0, len(x), 64))
        student.train_step(x[i], y[i], opt)
    with torch.no_grad():
        assert (student(xt).argmax(1) == yt).float().mean().item() > 0.7             # measured 0.82 at 2,000 steps


def test_s5_epochs_repeat_the_training_segment_as_revisits():
    st = S.s5_ladder(0, width=8, depth=1, n_train=100, n_test=20, epochs=3)
    assert len(st.segments) == 3 and [s.first_visit for s in st.segments] == [True, False, False]
    assert torch.equal(st.segments[0].x, st.segments[2].x)


def test_s5_grid_has_the_nine_teachers():
    grid = S.s5_grid(0, n_train=50, n_test=20)
    assert set(grid) == {(w, d) for w in (8, 32, 128) for d in (1, 2, 3)}


# ---------------------------------------------------------------------------------------------------- S6
def test_s6_related_regimes_share_class_structure_with_their_parent_unrelated_do_not():
    st = _s6()
    assert len(st.regimes) == 60
    related = [r for r in st.regimes.values() if r.parent is not None]
    assert 0.2 <= len(related) / 60 <= 0.4
    x, y = _train(st)
    regime = torch.cat([torch.full((len(s.y),), s.regime) for s in st.segments])
    means = {r: _class_means(x[regime == r], y[regime == r], 8) for r in st.regimes}
    rel = torch.stack([(means[r.id] * means[r.parent]).sum(1).mean() for r in related])
    unrelated = [r.id for r in st.regimes.values() if r.parent is None]
    unr = torch.stack([(means[a] * means[b]).sum(1).abs().mean() for a in unrelated[:12] for b in unrelated[:12] if a < b])
    assert rel.mean() > 0.7 and unr.mean() < 0.25


def test_s6_sizes_and_revisits():
    st = _s6()
    first = [s for s in st.segments if s.first_visit]
    assert len(first) == 60 and all(len(s.y) == 500 for s in first)
    assert all(x.shape[0] == 200 for x, _ in st.test_sets().values())
    assert 0 < len(st.segments) - 60 <= 12
    assert st.regimes[0].parent is None and all(r.parent < r.id for r in st.regimes.values() if r.parent is not None)


def test_s6_within_regime_classes_are_separable():
    st = _s6()
    x, y = _train(st)
    regime = torch.cat([torch.full((len(s.y),), s.regime) for s in st.segments])
    xt, yt = st.test_sets()[5]
    means = _class_means(x[regime == 5], y[regime == 5], 8)
    assert ((xt @ means.T).argmax(1) == yt).float().mean() > 0.95


# ---------------------------------------------------------------------------------------------------- S7
def test_s7_has_identical_inputs_and_permuted_labels_across_regimes():
    st = _s7()
    segs = {s.regime: s for s in st.segments if s.first_visit}
    assert len(segs) >= 3
    x0, y0 = segs[0].x, segs[0].y
    for r, s in segs.items():
        assert torch.equal(s.x, x0)
        mapping = {}
        for a, b in zip(y0.tolist(), s.y.tolist()):
            assert mapping.setdefault(a, b) == b                                    # a function of the base label ...
        assert sorted(mapping.values()) == sorted(mapping)                          # ... and a permutation of the labels
    assert len({tuple(torch.stack([y0, s.y]).tolist()[1]) for s in segs.values()}) == len(segs)


def test_s7_context_is_present_only_when_requested_and_differs_by_regime():
    with_ctx, without = _s7(with_ctx=True), _s7(with_ctx=False)
    assert all(s.ctx is None for s in without.segments) and all(r.ctx is None for r in without.regimes.values())
    c = {r: reg.ctx for r, reg in with_ctx.regimes.items()}
    assert all(v is not None for v in c.values())
    assert not torch.allclose(c[0], c[1])
    seg = with_ctx.segments[0]
    assert seg.ctx.shape[0] == len(seg.y) and torch.allclose(seg.ctx[0], with_ctx.regimes[seg.regime].ctx)
    assert all(a[2] is not None for a in with_ctx.algo_batches(64))
