"""RV3: the guard stream with label-sorted batches and the group evaluation helpers (spec sections 5, 7, 11)."""

import numpy as np
import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.groupeval import evaluate_groups, make_groups
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes
from growmoe.streams_order import label_sorted


def _sets(sizes, d=8, seed=0):
    g = np.random.default_rng(seed)
    out = {}
    for r, n in sizes.items():
        x = torch.from_numpy(g.standard_normal((n, d)).astype(np.float32))
        y = torch.full((n,), r, dtype=torch.long)                                          # the label is the regime id, so a row's source is readable from y
        out[r] = (x, y)
    return out


# ------------------------------------------------------------------------------------------------ label-sorted stream
def test_a_label_sorted_stream_has_the_same_samples_and_batch_count_but_each_batch_is_ordered():
    st = function_regimes(100, n_regimes=4, n_train=128, n_test=64)
    srt = label_sorted(st)
    assert srt.name == st.name + "-SORTED" and srt.meta["ordered"] == "label" and srt.n_classes == st.n_classes
    plain, ordered = list(st._walk(64, 3)), list(srt._walk(64, 3))
    assert len(plain) == len(ordered)
    for (k1, s1, i1), (k2, s2, i2) in zip(plain, ordered):
        assert k1 == k2 and s1 is s2 and len(i1) == len(i2)
    for k, seg in enumerate(srt.segments):
        idx = torch.cat([i for kk, _, i in ordered if kk == k])
        assert sorted(idx.tolist()) == list(range(len(seg.y)))                              # every sample exactly once
        labels = seg.y[idx]
        assert (labels[1:] >= labels[:-1]).all()                                            # ascending label across the whole segment
    _, y_sorted, _ = next(iter(srt.batches(64, 3)))
    _, y_plain, _ = next(iter(st.batches(64, 3)))
    assert len(torch.unique(y_sorted)) <= 5 and len(torch.unique(y_plain)) >= 7              # 8 classes, 16 samples of each per segment: a sorted batch holds four, a shuffled one all


def test_the_sorted_stream_does_not_depend_on_the_seed_and_leaves_the_original_alone():
    st = function_regimes(100, n_regimes=3, n_train=64, n_test=32)
    srt = label_sorted(st)
    assert [i.tolist() for _, _, i in srt._walk(64, 1)] == [i.tolist() for _, _, i in srt._walk(64, 2)]
    assert st.name.endswith("-SORTED") is False and "ordered" not in st.meta


# ------------------------------------------------------------------------------------------------ make_groups
def test_pure_groups_hold_rows_of_one_regime_only_and_drop_the_remainder():
    ts = _sets({0: 150, 1: 70, 2: 63})
    x, y, g = make_groups(ts, size=64, minor=0, seed=1)
    assert len(y) == 3 * 64 and torch.unique(g).numel() == 3                                # regime 0: two groups of 64, regime 1: one, regime 2: none (63 < 64)
    for gid in torch.unique(g).tolist():
        assert len(torch.unique(y[g == gid])) == 1 and (g == gid).sum() == 64


def test_mixed_groups_have_the_requested_split_between_a_regime_and_its_next_regime():
    ts = _sets({0: 200, 1: 200, 2: 200})
    for size, minor in ((64, 32), (64, 6)):
        x, y, g = make_groups(ts, size=size, minor=minor, seed=2)
        for gid in torch.unique(g).tolist():
            labels = y[g == gid]
            main, partner = labels[: size - minor], labels[size - minor:]
            r = int(main[0])
            assert (main == r).all() and (partner == (r + 1) % 3).all() and len(labels) == size


def test_groups_are_reproducible_and_the_seed_changes_them():
    ts = _sets({0: 130, 1: 130})
    a, b, c = make_groups(ts, 64, 0, seed=1), make_groups(ts, 64, 0, seed=1), make_groups(ts, 64, 0, seed=2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2]) and not torch.equal(a[0], c[0])


def test_impossible_requests_raise():
    ts = _sets({0: 10, 1: 10})
    with pytest.raises(ValueError):
        make_groups(ts, size=64, minor=0)                                                  # not one group of 64 fits
    with pytest.raises(ValueError):
        make_groups(ts, size=64, minor=40)                                                 # the minority may not outnumber the majority
    with pytest.raises(ValueError):
        make_groups(ts, size=1, minor=0)
    with pytest.raises(ValueError):
        make_groups(_sets({0: 100}), size=64, minor=6)                                     # a single regime has no partner


# ------------------------------------------------------------------------------------------------ evaluate_groups
class _FakePool:
    """Answers every input by its own label unless grouped, in which case the group's first member decides: the two answers can differ."""

    def __init__(self, vote_on):
        self.cfg = PhyloCfg(group_vote=vote_on)

    def predict(self, x, ctx=None):
        n = x.shape[0]
        probs = torch.zeros(n, 4)
        probs[torch.arange(n), (x[:, 0] > 0).long()] = 1.0                                  # class = sign of the first coordinate
        return type("P", (), {"probs": probs})()

    def predict_grouped(self, x, groups, ctx=None):
        probs = self.predict(x).probs
        first = {}
        for i, gid in enumerate(groups.tolist()):
            first.setdefault(gid, i)
        for i, gid in enumerate(groups.tolist()):
            probs[i] = probs[first[gid]].clone()
        return type("P", (), {"probs": probs})()


def test_evaluate_groups_scores_the_same_rows_both_ways_and_counts_what_changed():
    x = torch.tensor([[1.0], [1.0], [-1.0], [1.0]])
    y = torch.tensor([1, 1, 0, 1])
    res = evaluate_groups(_FakePool(True), x, y, torch.tensor([0, 0, 1, 1]))               # group 1 = {-1, +1}: its first member decides class 0 for both
    assert res["single"] == 1.0 and res["group"] == 0.75 and res["n"] == 4 and res["n_groups"] == 2 and res["changed"] == 1
    y2 = torch.tensor([1, 1, 0, 0])                                                         # now the group's answer is the right one for the last row
    res2 = evaluate_groups(_FakePool(True), x, y2, torch.tensor([0, 0, 1, 1]))
    assert res2["single"] == 0.75 and res2["group"] == 1.0 and res2["changed"] == 1


def test_evaluate_groups_refuses_a_pool_whose_group_answering_is_off():
    with pytest.raises(ValueError):
        evaluate_groups(_FakePool(False), torch.zeros(4, 1), torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long))
