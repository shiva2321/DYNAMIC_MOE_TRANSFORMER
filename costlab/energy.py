"""GPU power and energy consumption profiling using NVML."""

from __future__ import annotations

import statistics
import threading
import time
from typing import Any, Callable

import torch


class EnergyUnavailable(RuntimeError):
    """Raised when NVML or GPU energy counters are unavailable."""


def cpu_energy_note() -> str:
    """Return explicit notice that CPU energy is not measured."""
    return "CPU energy is not measured on this machine."


def _get_nvml_handle(device_index: int = 0) -> tuple[Any, Any]:
    """Initialize NVML and return (pynvml_module, device_handle)."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        return pynvml, handle
    except Exception as exc:
        raise EnergyUnavailable(f"NVML is not available: {exc}") from exc


def gpu_state(device_index: int = 0) -> dict[str, Any]:
    """Query current GPU state via NVML.

    Args:
        device_index: Integer GPU index.

    Returns:
        dict: Utilization rates, memory usage in bytes, and current power in watts.
    """
    pynvml, handle = _get_nvml_handle(device_index)
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        return {
            "utilization_gpu_pct": int(util.gpu),
            "utilization_mem_pct": int(util.memory),
            "memory_used_bytes": int(mem.used),
            "memory_total_bytes": int(mem.total),
            "power_watts": float(power_mw / 1000.0),
        }
    except Exception as exc:
        raise EnergyUnavailable(f"Failed to query GPU state: {exc}") from exc


def idle_baseline(
    seconds: float = 10.0,
    device_index: int = 0,
) -> dict[str, Any]:
    """Measure GPU idle power baseline over a specified duration.

    Args:
        seconds: Duration in seconds to sample power.
        device_index: Integer GPU index.

    Returns:
        dict: Summary containing mean watts at rest, sample counts, and min/max.
    """
    pynvml, handle = _get_nvml_handle(device_index)
    samples: list[float] = []
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < seconds:
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            samples.append(float(power_mw / 1000.0))
        except Exception as exc:
            raise EnergyUnavailable(f"Error reading power usage: {exc}") from exc
        time.sleep(0.05)

    if not samples:
        raise EnergyUnavailable("No power samples were collected during baseline.")

    dur = time.perf_counter() - t0
    return {
        "duration_s": float(dur),
        "mean_watts": float(statistics.mean(samples)),
        "min_watts": float(min(samples)),
        "max_watts": float(max(samples)),
        "samples_count": len(samples),
        "samples": samples,
    }


def measure_energy(
    fn: Callable[[], Any],
    min_duration_s: float = 3.0,
    device_index: int = 0,
) -> dict[str, Any]:
    """Measure GPU energy consumed by running fn repeatedly for at least min_duration_s.

    Records energy via two independent methods:
    1. NVML hardware total-energy counter (TotalEnergyConsumption in mJ).
    2. Numerical trapezoidal integration of power samples polled every 50 ms.

    Args:
        fn: Zero-argument callable to execute repeatedly.
        min_duration_s: Minimum duration in seconds to loop execution.
        device_index: Integer GPU index.

    Returns:
        dict: Measurement results containing counter_joules, integrated_joules,
            mean_watts, duration_s, calls, joules_per_call, and states before/after.
    """
    pynvml, handle = _get_nvml_handle(device_index)

    gpu_before = gpu_state(device_index)

    stop_event = threading.Event()
    power_samples: list[tuple[float, float]] = []
    sampler_errors: list[str] = []

    def _sampler() -> None:
        while not stop_event.is_set():
            try:
                p_w = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                power_samples.append((time.perf_counter(), p_w))
            except pynvml.NVMLError as exc:
                sampler_errors.append(repr(exc))
            time.sleep(0.05)

    sampler_thread = threading.Thread(target=_sampler, daemon=True)

    try:
        e_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    except Exception as exc:
        raise EnergyUnavailable(
            f"Hardware total-energy counter is not supported: {exc}"
        ) from exc

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    p_start_w = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
    power_samples.append((t_start, p_start_w))

    sampler_thread.start()

    calls = 0
    try:
        while time.perf_counter() - t_start < min_duration_s:
            fn()
            calls += 1

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        t_end = time.perf_counter()
        p_end_w = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        e_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        stop_event.set()
        sampler_thread.join()

    power_samples.append((t_end, p_end_w))
    power_samples.sort(key=lambda s: s[0])

    gpu_after = gpu_state(device_index)

    duration = t_end - t_start
    counter_joules = float(e_end_mj - e_start_mj) / 1000.0

    # Trapezoidal integration of (timestamp, power_watts)
    integrated_joules = 0.0
    for i in range(1, len(power_samples)):
        dt = power_samples[i][0] - power_samples[i - 1][0]
        p_avg = (power_samples[i][1] + power_samples[i - 1][1]) / 2.0
        integrated_joules += p_avg * dt

    mean_watts = counter_joules / duration if duration > 0 else 0.0
    joules_per_call = counter_joules / calls if calls > 0 else 0.0

    return {
        "counter_joules": counter_joules,
        "integrated_joules": integrated_joules,
        "mean_watts": mean_watts,
        "duration_s": duration,
        "calls": calls,
        "joules_per_call": joules_per_call,
        "power_samples_count": len(power_samples),
        "power_sample_errors": len(sampler_errors),
        "gpu_state_before": gpu_before,
        "gpu_state_after": gpu_after,
    }
