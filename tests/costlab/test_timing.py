"""Tests for costlab.timing: time_callable and latency_throughput."""

import time

from costlab.timing import TimingResult, latency_throughput, time_callable


def test_time_callable_structure():
    """Verify time_callable executes and produces valid TimingResult fields."""
    def fast_fn():
        time.sleep(0.001)

    res = time_callable(fast_fn, warmup=2, repeats=5, device="cpu")
    assert isinstance(res, TimingResult)
    assert len(res.samples) == 5
    assert res.repeats == 5
    assert res.warmup == 2
    assert res.median > 0.0
    assert res.mean > 0.0
    assert res.std >= 0.0
    assert res.cv >= 0.0
    assert res.p95 > 0.0
    assert res.cpu_seconds_per_call >= 0.0


def test_time_callable_min_duration():
    """Verify min_duration_s executes multiple inner calls for very short callables."""
    counter = {"calls": 0}

    def tiny_fn():
        counter["calls"] += 1

    res = time_callable(tiny_fn, warmup=1, repeats=3, min_duration_s=0.01)
    # warmup (1) + test_est (1) + repeats (3 * k where k >= 1)
    assert counter["calls"] > 4
    assert res.mean > 0.0


def test_latency_throughput():
    """Verify latency_throughput evaluates across batch sizes."""
    def make_fn(batch_size):
        def fn():
            time.sleep(0.001)
        return fn

    batch_sizes = (1, 4)
    res = latency_throughput(make_fn, batch_sizes=batch_sizes, warmup=1, repeats=3)
    assert set(res.keys()) == {1, 4}
    for b in batch_sizes:
        assert res[b]["batch_size"] == b
        assert isinstance(res[b]["timing"], TimingResult)
        assert res[b]["throughput_items_per_s"] > 0.0
