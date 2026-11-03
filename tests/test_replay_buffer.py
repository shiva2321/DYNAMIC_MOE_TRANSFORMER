import random
import pytest

from dmoe.replay_buffer import ReplayBuffer


def test_rejects_non_positive_capacity():
    with pytest.raises(ValueError):
        ReplayBuffer(capacity=0)


def test_buffer_keeps_all_items_under_capacity():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.add(i)
    assert len(buf) == 5


def test_buffer_caps_at_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(100):
        buf.add(i)
    assert len(buf) == 5


def test_sample_returns_requested_count_of_distinct_items():
    buf = ReplayBuffer(capacity=20)
    for i in range(20):
        buf.add(i)
    sampled = buf.sample(5)
    assert len(sampled) == 5
    assert len(set(sampled)) == 5


def test_sample_returns_all_items_when_k_exceeds_buffer_size():
    buf = ReplayBuffer(capacity=20)
    for i in range(3):
        buf.add(i)
    sampled = buf.sample(10)
    assert sorted(sampled) == [0, 1, 2]


def test_reservoir_sampling_is_not_biased_toward_recency():
    random.seed(0)
    trials = 2000
    capacity = 10
    stream_length = 100
    early_item_survival = 0
    late_item_survival = 0

    for _ in range(trials):
        buf = ReplayBuffer(capacity=capacity)
        for i in range(stream_length):
            buf.add(i)
        if 0 in buf._items:
            early_item_survival += 1
        if (stream_length - 1) in buf._items:
            late_item_survival += 1

    early_rate = early_item_survival / trials
    late_rate = late_item_survival / trials
    assert abs(early_rate - late_rate) < 0.05
    assert abs(early_rate - capacity / stream_length) < 0.05
