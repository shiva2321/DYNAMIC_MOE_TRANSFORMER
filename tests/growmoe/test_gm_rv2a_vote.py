"""RV2-A stage 1: the batch vote (spec section 4).

Per-sample routing lets a fraction of every batch train the wrong expert when regimes overlap (training purity 78 % at centre cosine 0.3, R5).
When one expert holds at least ``batch_vote_share`` of a batch's accepted samples, all of them train that expert.
"""

import numpy as np
import pytest
import torch

from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes

A_KW = dict(q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf")
SEEDS = (100, 101, 102, 103)


def _vote(assigned, **kw):
    pool = PhyloMoE(PhyloCfg(d=8, n_classes=4, batch_vote=True, **kw))
    return pool._batch_consistent(torch.tensor(assigned))


def _pool(st, steps=5, seed=100, **kw):
    return PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=steps, **A_KW, **kw))


def _purity(st, pool, seed):
    """Share of the samples each expert trained on that came from its dominant regime (summed over experts)."""
    counts = {}
    for (_, seg, _), (x, y, _) in zip(st._walk(64, seed), st.algo_batches(64, seed)):
        for e in pool.observe(x, y).assigned.tolist():
            if e >= 0:
                counts[(seg.regime, e)] = counts.get((seg.regime, e), 0) + 1
    per_expert = {}
    for (_, e), c in counts.items():
        per_expert.setdefault(e, []).append(c)
    return sum(max(v) for v in per_expert.values()) / max(1, sum(counts.values()))


def _purity_for(seed, vote, overlap):
    st = function_regimes(seed, n_regimes=3, n_train=256, n_test=100, overlap=overlap)
    return _purity(st, _pool(st, seed=seed, batch_vote=vote), seed)


# ------------------------------------------------------------------------------------------------------ the rule itself
def test_a_unanimous_batch_is_unchanged():
    assert _vote([3] * 20).tolist() == [3] * 20


def test_a_75_25_batch_goes_entirely_to_the_mode():
    assert _vote([3] * 15 + [5] * 5).tolist() == [3] * 20


def test_a_batch_that_mixes_two_known_regimes_is_pushed_into_the_majority_expert_by_design():
    """Documented behaviour (spec section 4 failure mode a): the 30 % minority trains the majority's expert. The G-MIX guard bounds the cost."""
    assert _vote([1] * 14 + [2] * 6).tolist() == [1] * 20


def test_novel_samples_are_untouched():
    assert _vote([3] * 12 + [5] * 4 + [-1] * 4).tolist() == [3] * 16 + [-1] * 4


def test_below_the_share_nothing_moves():
    a = [1] * 6 + [2] * 6 + [3] * 6                                                     # best share 1/3 < 0.5
    assert _vote(a).tolist() == a


def test_fewer_than_the_minimum_accepted_samples_nothing_moves():
    a = [1] * 5 + [2] * 2 + [-1] * 20                                                   # 7 accepted < 8
    assert _vote(a).tolist() == a


def test_an_all_novel_batch_is_left_alone():
    assert _vote([-1] * 30).tolist() == [-1] * 30


def test_ties_go_to_the_smallest_id():
    assert _vote([7] * 8 + [2] * 8).tolist() == [2] * 16


def test_the_input_tensor_is_not_modified():
    t = torch.tensor([3] * 15 + [5] * 5)
    before = t.clone()
    PhyloMoE(PhyloCfg(d=8, n_classes=4, batch_vote=True))._batch_consistent(t)
    assert torch.equal(t, before)


def test_share_and_minimum_are_configurable():
    a = [1] * 9 + [2] * 11                                                              # best share 0.55
    assert _vote(a, batch_vote_share=0.6).tolist() == a
    b = [1] * 5 + [2] * 3                                                               # 8 accepted
    assert _vote(b).tolist() == [1] * 8
    assert _vote(b, batch_vote_min=9).tolist() == b


# ------------------------------------------------------------------------------------------------------ in the pool
def test_the_flag_is_on_by_default_since_the_rv2a_test_was_confirmed_on_fresh_seeds():
    assert PhyloCfg().batch_vote is True


def test_with_the_flag_off_the_vote_is_never_called(monkeypatch):
    def boom(self, assigned):
        raise AssertionError("_batch_consistent must not run when batch_vote is False")

    monkeypatch.setattr(PhyloMoE, "_batch_consistent", boom)
    st = function_regimes(100, n_regimes=3, n_train=128, n_test=50)
    pool = _pool(st, batch_vote=False)                                                  # ONE pool: from the second batch on the gate and assignment code runs
    for x, y, _ in st.algo_batches(64, 100):
        pool.observe(x, y)
    assert len(pool.experts) >= 1


def test_the_vote_restores_training_purity_when_regimes_overlap():
    off = [_purity_for(s, False, 0.4) for s in SEEDS]
    on = [_purity_for(s, True, 0.4) for s in SEEDS]
    assert np.mean(off) <= 0.88, f"control: per-sample routing should contaminate the experts, got {off}"
    assert np.mean(on) >= 0.95, f"with the vote purity should be high, got {on}"
    assert all(b >= a - 0.02 for a, b in zip(off, on)), (off, on)


def test_with_no_overlap_the_vote_changes_almost_nothing():
    st = function_regimes(100, n_regimes=3, n_train=256, n_test=100)
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    acc = []
    for vote in (False, True):
        pool = _pool(st, batch_vote=vote)
        for bx, by, _ in st.algo_batches(64, 100):
            pool.observe(bx, by)
        acc.append(float((pool.predict(x).probs.argmax(1) == y).float().mean()))
    assert acc[1] >= acc[0] - 0.03, acc


def _trained(**kw):
    st = function_regimes(100, n_regimes=3, n_train=128, n_test=50)
    pool = _pool(st, **kw)
    x, y, _ = next(iter(st.algo_batches(64, 100)))
    pool.observe(x, y)                                                                  # founds expert 0
    return st, pool, x, y


def test_a_single_sample_and_unlabeled_batches_work_with_the_vote_on():
    st, pool, x, y = _trained(batch_vote=True)
    assert pool.observe(x[:1], y[:1]).assigned.shape == (1,)
    assert pool.observe(x).assigned.shape == (len(x),)                                  # y=None


def test_the_no_conformal_ablation_works_with_the_vote_on():
    st, pool, x, y = _trained(batch_vote=True, no_conformal=True)
    assert pool.observe(x, y).assigned.shape == (len(x),)
