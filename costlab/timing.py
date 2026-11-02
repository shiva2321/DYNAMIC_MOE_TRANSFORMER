"""Callable execution timing and latency/throughput profiling."""

from __future__ import annotations

import dataclasses
import math
import statistics
import time
from typing import Any, Callable

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class TimingResult:
    """Statistical summary of execution timing across multiple repeats.

    Attributes:
        samples: Wall-clock duration per call in seconds for each repeat.
        median: Median wall-clock duration in seconds.
        mean: Arithmetic mean duration in seconds.
        std: Sample standard deviation in seconds.
        cv: Coefficient of variation (std / mean).
        p95: 95th percentile wall-clock duration in seconds.
        cpu_seconds_per_call: CPU process time per call in seconds.
        warmup: Number of warmup calls executed.
        repeats: Number of measurement repeats collected.
    """

    samples: list[float]
    median: float
    mean: float
    std: float
    cv: float
    p95: float
    cpu_seconds_per_call: float
    warmup: int
    repeats: int


def time_callable(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    repeats: int = 10,
    device: str = "cpu",
    min_duration_s: float = 0.0,
) -> TimingResult:
    """Measure the execution time of a zero-argument callable with high precision.

    Uses time.perf_counter for wall-clock timing and time.process_time for CPU time.
    Synchronizes CUDA before and after each measurement when device is 'cuda'.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup: Number of unmeasured warmup iterations before sampling.
        repeats: Number of timed measurement repeats.
        device: Device string ('cpu' or 'cuda').
        min_duration_s: If > 0, each repeat executes fn multiple times so each sample
            runs for at least this duration, dividing elapsed time by repeat count.

    Returns:
        TimingResult: Dataclass with samples, median, mean, std, cv, p95, and cpu time.
    """
    is_cuda = (device == "cuda" or device.startswith("cuda")) and torch.cuda.is_available()

    if repeats < 1:
        raise ValueError(f"repeats must be at least 1, got {repeats}")

    # Warmup
    for _ in range(warmup):
        if is_cuda:
            torch.cuda.synchronize()
        fn()
    if is_cuda:
        torch.cuda.synchronize()

    # Determine inner batch count if min_duration_s > 0
    k = 1
    if min_duration_s > 0.0:
        if is_cuda:
            torch.cuda.synchronize()
        t0_est = time.perf_counter()
        fn()
        if is_cuda:
            torch.cuda.synchronize()
        t_est = max(time.perf_counter() - t0_est, 1e-9)
        if t_est < min_duration_s:
            k = max(1, math.ceil(min_duration_s / t_est))

    samples: list[float] = []
    cpu_samples: list[float] = []

    for _ in range(repeats):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        c0 = time.process_time()

        for _ in range(k):
            fn()

        if is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        c1 = time.process_time()

        sample_wall = (t1 - t0) / k
        sample_cpu = (c1 - c0) / k
        samples.append(sample_wall)
        cpu_samples.append(sample_cpu)

    mean_val = float(statistics.mean(samples)) if samples else 0.0
    median_val = float(statistics.median(samples)) if samples else 0.0
    std_val = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    cv_val = float(std_val / mean_val) if mean_val > 0.0 else 0.0
    p95_val = float(np.percentile(samples, 95)) if samples else 0.0
    cpu_val = float(statistics.mean(cpu_samples)) if cpu_samples else 0.0

    return TimingResult(
        samples=samples,
        median=median_val,
        mean=mean_val,
        std=std_val,
        cv=cv_val,
        p95=p95_val,
        cpu_seconds_per_call=cpu_val,
        warmup=warmup,
        repeats=repeats,
    )


def latency_throughput(
    make_fn: Callable[[int], Callable[[], Any]],
    batch_sizes: tuple[int, ...] = (1, 32),
    **kwargs: Any,
) -> dict[int, dict[str, Any]]:
    """Benchmark latency and throughput across multiple batch sizes.

    Args:
        make_fn: Factory receiving batch size and returning a zero-argument callable.
        batch_sizes: Iterable of batch sizes to evaluate.
        **kwargs: Extra arguments forwarded to time_callable (e.g. warmup, repeats, device).

    Returns:
        dict: Mapping each batch size to its TimingResult and items_per_second throughput.
    """
    results: dict[int, dict[str, Any]] = {}
    for batch in batch_sizes:
        fn = make_fn(batch)
        timing = time_callable(fn, **kwargs)
        throughput = float(batch / timing.median) if timing.median > 0.0 else 0.0
        results[batch] = {
            "batch_size": batch,
            "timing": timing,
            "throughput_items_per_s": throughput,
        }
    return results
