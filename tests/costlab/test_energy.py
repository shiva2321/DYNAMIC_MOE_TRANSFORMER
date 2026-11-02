"""Tests for costlab.energy: GPU state, idle baseline, and energy measurement."""

import pytest
import torch

from costlab.energy import (
    EnergyUnavailable,
    cpu_energy_note,
    gpu_state,
    idle_baseline,
    measure_energy,
)


def test_cpu_energy_note():
    """Verify cpu_energy_note returns the documented notice."""
    assert cpu_energy_note() == "CPU energy is not measured on this machine."


def _has_cuda_and_nvml() -> tuple[bool, str]:
    """Check whether both CUDA and NVML are functional."""
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    try:
        import pynvml
        pynvml.nvmlInit()
        return True, ""
    except Exception as exc:
        return False, f"NVML is not available: {exc}"


CUDA_NVML_OK, REASON = _has_cuda_and_nvml()


@pytest.mark.skipif(not CUDA_NVML_OK, reason=REASON)
def test_gpu_state():
    """Verify gpu_state returns utilization, memory, and power."""
    state = gpu_state()
    assert "utilization_gpu_pct" in state
    assert "utilization_mem_pct" in state
    assert "memory_used_bytes" in state
    assert "memory_total_bytes" in state
    assert "power_watts" in state
    assert state["power_watts"] > 0.0


@pytest.mark.skipif(not CUDA_NVML_OK, reason=REASON)
def test_idle_baseline_short():
    """Verify idle_baseline returns sampled power statistics."""
    res = idle_baseline(seconds=0.5)
    assert "mean_watts" in res
    assert "min_watts" in res
    assert "max_watts" in res
    assert "samples_count" in res
    assert res["mean_watts"] > 0.0


@pytest.mark.skipif(not CUDA_NVML_OK, reason=REASON)
def test_measure_energy_short():
    """Verify measure_energy executes and returns dual energy metrics."""
    x = torch.randn(500, 500, device="cuda")

    def workload():
        torch.mm(x, x)

    res = measure_energy(workload, min_duration_s=0.5)
    assert "counter_joules" in res
    assert "integrated_joules" in res
    assert "mean_watts" in res
    assert "calls" in res
    assert "joules_per_call" in res
    assert "gpu_state_before" in res
    assert "gpu_state_after" in res
    assert res["calls"] > 0
    assert res["counter_joules"] > 0.0
    assert "power_sample_errors" in res
