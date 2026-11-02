"""costlab: the cost meter for compute, memory, timing, and energy."""

from __future__ import annotations

from costlab.bytes import ActiveBytes, active_param_bytes, total_param_bytes
from costlab.count import Counter, MacCount, count_macs, macs_matmul
from costlab.energy import (
    EnergyUnavailable,
    cpu_energy_note,
    gpu_state,
    idle_baseline,
    measure_energy,
)
from costlab.env import environment, fixed_threads
from costlab.memory import peak_memory
from costlab.report import measure, to_json, to_markdown
from costlab.timing import TimingResult, latency_throughput, time_callable

__all__ = [
    "ActiveBytes",
    "Counter",
    "EnergyUnavailable",
    "MacCount",
    "TimingResult",
    "active_param_bytes",
    "count_macs",
    "cpu_energy_note",
    "environment",
    "fixed_threads",
    "gpu_state",
    "idle_baseline",
    "latency_throughput",
    "macs_matmul",
    "measure",
    "measure_energy",
    "peak_memory",
    "time_callable",
    "to_json",
    "to_markdown",
    "total_param_bytes",
]

__version__ = "0.1.0"
