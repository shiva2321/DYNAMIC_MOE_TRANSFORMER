"""Process RSS and GPU memory usage measurement."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

import psutil
import torch


def peak_memory(fn: Callable[[], Any], device: str = "cpu") -> dict[str, Any]:
    """Measure peak memory usage during the execution of fn.

    Runs a background sampler thread measuring process RSS every 10 ms.
    If device is 'cuda' (and CUDA is available), calls reset_peak_memory_stats()
    before execution and queries max_memory_allocated() afterward.

    Args:
        fn: Zero-argument callable to execute.
        device: Device string ('cpu' or 'cuda').

    Returns:
        dict: Memory metrics containing:
            - rss_start_bytes: Starting process RSS in bytes.
            - rss_peak_bytes: Maximum process RSS recorded during execution.
            - rss_diff_bytes: Difference (rss_peak - rss_start) in bytes.
            - rss_samples_count: Number of RSS samples taken.
            - rss_sampler_errors: Number of RSS reads that failed (those samples are missing).
            - cuda_start_bytes: Allocated CUDA memory before execution (or None).
            - cuda_peak_bytes: Peak allocated CUDA memory during execution (or None).
            - cuda_allocated_bytes: Alias for cuda_peak_bytes (or None).
            - cuda_diff_bytes: Difference (cuda_peak - cuda_start) (or None).
            - device: Device string passed.
    """
    proc = psutil.Process()
    rss_samples: list[int] = []
    sampler_errors: list[str] = []
    stop_event = threading.Event()

    rss_start = proc.memory_info().rss
    rss_samples.append(rss_start)

    def _sampler() -> None:
        while not stop_event.is_set():
            try:
                rss_samples.append(proc.memory_info().rss)
            except psutil.Error as exc:
                sampler_errors.append(repr(exc))
            time.sleep(0.01)

    th = threading.Thread(target=_sampler, daemon=True)

    is_cuda = (device == "cuda" or device.startswith("cuda")) and torch.cuda.is_available()
    cuda_start: int | None = None
    cuda_peak: int | None = None
    cuda_diff: int | None = None

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        cuda_start = int(torch.cuda.memory_allocated())

    th.start()
    try:
        fn()
        if is_cuda:
            torch.cuda.synchronize()
    finally:
        stop_event.set()
        th.join()

    rss_peak = max(rss_samples)
    rss_diff = max(0, rss_peak - rss_start)

    if is_cuda:
        cuda_peak = int(torch.cuda.max_memory_allocated())
        cuda_diff = max(0, cuda_peak - (cuda_start or 0))

    return {
        "rss_start_bytes": rss_start,
        "rss_peak_bytes": rss_peak,
        "rss_diff_bytes": rss_diff,
        "rss_samples_count": len(rss_samples),
        "rss_sampler_errors": len(sampler_errors),
        "cuda_start_bytes": cuda_start,
        "cuda_peak_bytes": cuda_peak,
        "cuda_allocated_bytes": cuda_peak,
        "cuda_diff_bytes": cuda_diff,
        "device": device,
    }
