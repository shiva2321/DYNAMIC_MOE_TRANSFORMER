"""Consolidated reporting, JSON serialization, and Markdown export."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

from costlab.bytes import active_param_bytes, total_param_bytes
from costlab.count import count_macs
from costlab.energy import cpu_energy_note, measure_energy, EnergyUnavailable
from costlab.env import environment
from costlab.memory import peak_memory
from costlab.timing import latency_throughput


class _CostlabJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling dataclasses, tuples, numpy types, and paths."""

    def default(self, o: Any) -> Any:
        if dataclasses.is_dataclass(o) and not isinstance(o, type):
            return dataclasses.asdict(o)
        if hasattr(o, "_asdict"):  # NamedTuple
            return o._asdict()
        if isinstance(o, (np.integer, np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.floating, np.float64, np.float32)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, set):
            return sorted(list(o))
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def measure(
    make_fn: Callable[[int], Callable[[], Any]],
    *,
    model: nn.Module | None = None,
    batch_sizes: tuple[int, ...] = (1, 32),
    device: str = "cpu",
) -> dict[str, Any]:
    """Perform comprehensive cost measurement across multiple batch sizes.

    Measures environment, parameter footprint (if model provided), MACs,
    timing/latency/throughput, peak memory, and energy (for CUDA).

    Args:
        make_fn: Factory mapping batch_size to a zero-argument callable.
        model: Optional PyTorch module to evaluate parameter footprint.
        batch_sizes: Batch sizes to measure.
        device: Target execution device ('cpu' or 'cuda').

    Returns:
        dict: Complete measurement report containing schema_version, environment,
            measurements summary, and detailed raw profiles.
    """
    env_info = environment()
    is_cuda = (device == "cuda" or device.startswith("cuda")) and torch.cuda.is_available()

    # Byte footprint
    bytes_info: dict[str, Any] = {}
    if model is not None:
        tot_bytes = total_param_bytes(model)
        bytes_info["total_param_bytes"] = tot_bytes
        bytes_info["active_by_batch"] = {}
        for b in batch_sizes:
            fn = make_fn(b)
            act_b = active_param_bytes(model, fn)
            bytes_info["active_by_batch"][b] = {
                "active_bytes": act_b.total,
                "by_module": act_b.by_module,
            }

    # MACs counting
    macs_info: dict[int, Any] = {}
    for b in batch_sizes:
        fn = make_fn(b)
        mac_res = count_macs(fn)
        macs_info[b] = dataclasses.asdict(mac_res)

    # Timing and throughput
    timing_info = latency_throughput(make_fn, batch_sizes=batch_sizes, device=device)

    # Memory profiling
    memory_info: dict[int, Any] = {}
    for b in batch_sizes:
        fn = make_fn(b)
        memory_info[b] = peak_memory(fn, device=device)

    # Energy profiling
    energy_info: dict[str, Any] = {}
    if is_cuda:
        try:
            # Measure on largest batch size
            largest_b = max(batch_sizes)
            fn_large = make_fn(largest_b)
            energy_info = measure_energy(fn_large, min_duration_s=3.0)
            energy_info["measured_batch"] = largest_b
        except EnergyUnavailable as exc:
            energy_info = {"unavailable_reason": str(exc)}
    else:
        energy_info = {"note": cpu_energy_note()}

    # High-level measurements summary
    measurements_summary: dict[str, Any] = {}
    for b in batch_sizes:
        t_res = timing_info[b]["timing"]
        m_res = memory_info[b]
        mac_val = macs_info[b]["macs"]
        measurements_summary[str(b)] = {
            "batch_size": b,
            "latency_median_ms": t_res.median * 1000.0,
            "latency_mean_ms": t_res.mean * 1000.0,
            "latency_cv_pct": t_res.cv * 100.0,
            "throughput_items_s": timing_info[b]["throughput_items_per_s"],
            "macs": mac_val,
            "rss_peak_mb": m_res["rss_peak_bytes"] / (1024 * 1024),
            "cuda_peak_mb": (
                (m_res["cuda_peak_bytes"] / (1024 * 1024))
                if m_res["cuda_peak_bytes"] is not None
                else None
            ),
        }

    return {
        "schema_version": "1.0.0",
        "environment": env_info,
        "device": device,
        "batch_sizes": list(batch_sizes),
        "bytes": bytes_info,
        "macs": macs_info,
        "timing": {
            b: {
                "batch_size": b,
                "timing": dataclasses.asdict(timing_info[b]["timing"]),
                "throughput_items_per_s": timing_info[b]["throughput_items_per_s"],
            }
            for b in batch_sizes
        },
        "memory": memory_info,
        "energy": energy_info,
        "measurements": measurements_summary,
    }


def to_json(result: dict[str, Any], path: str | Path) -> None:
    """Save measurement dictionary to a JSON file.

    Args:
        result: Dictionary returned by measure() or validation routines.
        path: Destination file path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=_CostlabJSONEncoder)


def to_markdown(result: dict[str, Any]) -> str:
    """Convert measurement results into a GitHub-Flavored Markdown summary.

    Args:
        result: Dictionary generated by measure().

    Returns:
        str: Markdown-formatted report tables.
    """
    env = result.get("environment", {})
    meas = result.get("measurements", {})
    energy = result.get("energy", {})

    lines = [
        "# Costlab Measurement Report",
        "",
        "## Environment",
        f"- **Python**: {env.get('python')}",
        f"- **PyTorch**: {env.get('torch')}",
        f"- **CUDA**: {env.get('cuda')} (GPU: {env.get('gpu_name', 'N/A')})",
        f"- **CPU**: {env.get('cpu_name')} ({env.get('logical_cores')} logical cores)",
        f"- **RAM**: {env.get('ram_bytes', 0) / (1024**3):.1f} GB",
        f"- **Timestamp**: {env.get('timestamp')}",
        "",
        "## Measurements Summary",
        "| Batch | Latency Median (ms) | CV (%) | Throughput (items/s) | MACs | Peak RSS (MB) | Peak CUDA (MB) |",
        "|------:|--------------------:|-------:|---------------------:|-----:|--------------:|---------------:|",
    ]

    for b_str, m in meas.items():
        cuda_mb_str = f"{m['cuda_peak_mb']:.1f}" if m.get("cuda_peak_mb") is not None else "-"
        lines.append(
            f"| {m['batch_size']} | {m['latency_median_ms']:.2f} | {m['latency_cv_pct']:.2f}% | "
            f"{m['throughput_items_s']:.1f} | {m['macs']:,} | {m['rss_peak_mb']:.1f} | {cuda_mb_str} |"
        )

    lines.append("")
    lines.append("## Energy Profile")
    if "counter_joules" in energy:
        lines.append(f"- **Counter Joules**: {energy['counter_joules']:.2f} J")
        lines.append(f"- **Integrated Joules**: {energy['integrated_joules']:.2f} J")
        lines.append(f"- **Mean Loaded Power**: {energy['mean_watts']:.2f} W")
        lines.append(f"- **Energy per call**: {energy['joules_per_call']:.4f} J")
    elif "note" in energy:
        lines.append(f"- {energy['note']}")
    elif "unavailable_reason" in energy:
        lines.append(f"- *Energy unavailable*: {energy['unavailable_reason']}")

    lines.append("")
    return "\n".join(lines)
