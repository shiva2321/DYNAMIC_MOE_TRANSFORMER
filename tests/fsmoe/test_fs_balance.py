from collections import Counter

import numpy as np
import pytest
import torch

from fsmoe.balance import BalancedLearner, DomainHead


class Rec:
    def __init__(self):
        self.calls = []

    def __call__(self, x, y):
        self.calls.append((x.clone(), y.clone()))


def _x(i):
    return torch.full((3,), float(i))


def _learner(mode, known=(0, 1, 2, 3), **kw):
    rec = Rec()
    return rec, BalancedLearner(rec, mode, set(known), np.random.default_rng(0), **kw)


def test_sub_forms_a_chunk_only_when_every_domain_seen_holds_its_quota_and_splits_it_equally():
    rec, L = _learner("sub", chunk=6)                                     # two domains: quota 3 and 3
    for i in range(10):
        L.add(_x(i), 0, domain=1)                                         # only domain 1 seen so far: it forms chunks alone
    assert len(rec.calls) == 1 and (rec.calls[0][1] == 0).all() and len(rec.calls[0][1]) == 6
    rec.calls.clear()
    L.add(_x(0), 1, domain=2)                                             # a second domain appears: nothing flows until it holds 3 items
    L.add(_x(1), 1, domain=2)
    assert not rec.calls
    L.add(_x(2), 1, domain=2)
    assert len(rec.calls) == 1 and sorted(rec.calls[0][1].tolist()) == [0, 0, 0, 1, 1, 1]


def test_sub_never_uses_the_surplus_of_the_abundant_domain_beyond_the_scarce_pace():
    rec, L = _learner("sub", chunk=6)
    L.add(_x(0), 1, domain=2)
    for i in range(60):
        L.add(_x(i), 0, domain=1)
    L.add(_x(1), 1, domain=2)
    L.add(_x(2), 1, domain=2)
    L.finish()
    assert L.used[2] == 3 and L.used[1] <= 6 + 3 + 3                       # domain 2 gave 3 items; domain 1 was used at its pace, not 60 items


def test_new_class_items_come_first_and_are_balanced_over_their_classes():
    rec, L = _learner("sub", known=(0,), chunk=4)
    for i in range(6):
        L.add(_x(i), 0, domain=1)                                         # known class 0 ... but they wait: another domain not yet seen
    L.add(_x(0), 7, domain=1)
    L.add(_x(1), 8, domain=1)
    L.add(_x(2), 7, domain=1)
    labels = [l for _, y in rec.calls for l in y.tolist()]
    first_chunk = rec.calls[0][1].tolist() if rec.calls else []
    assert len(rec.calls) >= 1
    # the first chunk was formed from known items before any new class existed; later chunks put the new classes first
    after = [y.tolist() for _, y in rec.calls[1:]]
    assert after and 7 in after[0] and 8 in after[0]


def test_over_uses_every_abundant_item_once_and_fills_the_scarce_domain_by_replay():
    rec, L = _learner("over", chunk=6, replay_cap=50)
    for i in range(3):
        L.add(_x(i), 1, domain=2)                                         # three scarce items; one domain seen so far, its quota is the whole chunk (6), so nothing flows yet
    assert not rec.calls
    for i in range(12):
        L.add(_x(100 + i), 0, domain=1)
    L.finish()
    assert L.used[1] == 12 and L.used[2] == 3                              # abundant items used once, scarce ones fresh only once
    scarce_in_chunks = sum(int((y == 1).sum()) for _, y in rec.calls)
    assert scarce_in_chunks > 3                                            # but they were REPLAYED into later chunks
    assert all(len(y) <= 6 for _, y in rec.calls)


def test_over_replay_does_not_invent_items_for_a_domain_that_has_none():
    rec, L = _learner("over", chunk=6)
    for i in range(6):
        L.add(_x(i), 0, domain=1)
    assert all(set(y.tolist()) == {0} for _, y in rec.calls)


def test_reservoir_is_bounded():
    rec, L = _learner("over", chunk=4, replay_cap=5)
    for i in range(100):
        L.add(_x(i), 0, domain=1)
    assert len(L.reservoir[1]) == 5


def test_quotas_split_the_chunk_as_evenly_as_possible_and_always_sum_to_the_chunk():
    _, L = _learner("sub", chunk=16)
    assert L._quotas([0, 1, 2]) == {0: 6, 1: 5, 2: 5}
    assert L._quotas([0, 1]) == {0: 8, 1: 8} and L._quotas([4]) == {4: 16}
    rec, L = _learner("sub", chunk=16)
    for d in range(3):
        for i in range(6):
            L.add(_x(i), d, domain=d)                                    # classes 0..2 are known; three domains of 6 items each
    first = rec.calls[0][1].tolist()
    assert len(first) == 16 and sorted(Counter(first).values()) == [5, 5, 6]


def test_order_puts_new_classes_first_round_robin_then_known_items_oldest_first():
    _, L = _learner("sub", known=(0,), chunk=4)
    L.fresh[1] = [(_x(i), l) for i, l in enumerate([0, 0, 7, 8, 7, 0])]
    assert L._order(1) == [2, 3, 4, 0, 1, 5]


def test_over_forms_a_chunk_as_soon_as_one_domain_holds_its_quota_even_if_another_is_short():
    rec, L = _learner("over", chunk=6)
    for i in range(3):
        L.add(_x(i), 1, domain=2)                                         # alone, its quota is 6: nothing flows
    assert not rec.calls
    L.add(_x(9), 0, domain=1)                                             # a second domain appears (quota 3 each); domain 2 holds 3, domain 1 holds 1
    assert len(rec.calls) == 1 and sorted(rec.calls[0][1].tolist()) == [0, 1, 1, 1]
    rec2, S = _learner("sub", chunk=6)
    for i in range(3):
        S.add(_x(i), 1, domain=2)
    S.add(_x(9), 0, domain=1)
    assert not rec2.calls                                                 # `sub` waits for the scarce domain


def test_reservoir_is_a_sample_of_the_whole_stream_not_its_first_items():
    rec, L = _learner("over", chunk=4, replay_cap=10)
    for i in range(300):
        L.add(_x(i), 0, domain=1)
    assert max(float(x[0]) for x, _ in L.reservoir[1]) > 100


def test_hybrid_forms_a_chunk_from_the_new_queue_alone_and_never_before_it_holds_new_cap_items():
    rec, L = _learner("hybrid", chunk=8, new_cap=4)
    L.add(_x(0), 0, domain=1)                                             # one known item: its domain is seen, the known part (4 slots) is not full
    for i in range(3):
        L.add(_x(i), 7, domain=2)                                         # new-class items of ANOTHER domain
    assert not rec.calls
    L.add(_x(9), 7, domain=2)
    assert len(rec.calls) == 1 and sorted(rec.calls[0][1].tolist()) == [0, 7, 7, 7, 7]
    assert L.used_new[2] == 4 and L.used[2] == 4 and L.used[1] == 1


def test_hybrid_trains_new_items_in_order_of_arrival_whatever_their_domain_and_none_waits_for_another_domain():
    rec, L = _learner("hybrid", chunk=8, new_cap=4)
    for i, (lab, dom) in enumerate([(7, 1), (8, 2), (7, 1), (8, 2), (9, 1), (9, 1), (7, 2), (7, 2)]):
        L.add(_x(i), lab, domain=dom)
    assert [y.tolist() for _, y in rec.calls] == [[7, 8, 7, 8], [9, 9, 7, 7]]
    assert rec.calls[0][0][:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert L.used_new[1] == 4 and L.used_new[2] == 4 and not L.new_q


def test_hybrid_splits_the_known_part_equally_over_the_domains_and_waits_for_the_scarcest():
    rec, L = _learner("hybrid", chunk=16, new_cap=4)                      # known part: 12 slots, 4 per domain
    for d in range(3):
        for i in range(3):
            L.add(_x(i), d, domain=d)
    assert not rec.calls                                                  # every domain holds 3 of its 4
    for d in range(3):
        L.add(_x(9), d, domain=d)
    assert len(rec.calls) == 1 and sorted(Counter(rec.calls[0][1].tolist()).values()) == [4, 4, 4]


def test_hybrid_finish_flushes_the_new_queue_and_leaves_the_known_surplus_unused():
    rec, L = _learner("hybrid", known=(0, 1), chunk=8, new_cap=4)
    L.add(_x(0), 1, domain=2)                                             # scarce domain: one known item (its quota is 2 of the 4 known slots)
    for i in range(30):
        L.add(_x(i), 0, domain=1)                                         # abundant domain
    for i in range(3):
        L.add(_x(i), 9, domain=2)                                         # three new items: below new_cap
    n_before = len(rec.calls)
    L.finish()
    assert len(rec.calls) > n_before and L.new_q == []                    # the three new items were flushed
    assert L.used_new[2] == 3 and L.used[1] < 30                           # the abundant domain's surplus is not all used


def test_hybrid_finish_flushes_new_items_even_when_no_known_item_was_ever_seen():
    rec, L = _learner("hybrid", chunk=8, new_cap=4)
    for i in range(3):
        L.add(_x(i), 7, domain=1)
    assert not rec.calls
    L.finish()
    assert len(rec.calls) == 1 and rec.calls[0][1].tolist() == [7, 7, 7] and L.new_q == []


def test_hybrid_validates_new_cap_and_other_modes_count_new_items_trained():
    for bad in (0, 8):
        with pytest.raises(ValueError):
            BalancedLearner(Rec(), "hybrid", set(), np.random.default_rng(0), chunk=8, new_cap=bad)
    rec, L = _learner("sub", known=(0,), chunk=4)
    for i in range(4):
        L.add(_x(i), 0, domain=1)
    L.add(_x(5), 7, domain=1)
    for i in range(4):
        L.add(_x(i), 0, domain=1)
    L.finish()
    assert L.used_new[1] == 1 and L.used[1] == 9


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        BalancedLearner(Rec(), "nonsense", set(), np.random.default_rng(0))


def test_domain_head_assigns_clusters_to_their_domain():
    g = torch.Generator().manual_seed(0)
    centres = torch.nn.functional.normalize(torch.randn(3, 16, generator=g), dim=1)
    X = torch.nn.functional.normalize(centres.repeat_interleave(40, 0) + 0.05 * torch.randn(120, 16, generator=g), dim=1)
    dom = torch.arange(3).repeat_interleave(40)
    head = DomainHead(16).fit(X, dom)
    assert (head(X) == dom).float().mean() > 0.98 and head(X[:1]).shape == (1,)
