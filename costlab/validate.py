"""Validation suite enforcing bars V1 through V5."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
import torch.nn as nn

from costlab.bytes import active_param_bytes, total_param_bytes
from costlab.count import Counter, count_macs, macs_matmul
from costlab.energy import EnergyUnavailable, gpu_state, idle_baseline, measure_energy
from costlab.env import fixed_threads
from costlab.memory import peak_memory
from costlab.report import _CostlabJSONEncoder
from costlab.timing import time_callable


def _wait_for_idle_cpu(threshold_pct: float = 30.0, max_wait_s: float = 20.0) -> float:
    """Wait while system CPU usage is above threshold. Returns last measured CPU percent."""
    t0 = time.perf_counter()
    usage = psutil.cpu_percent(interval=1.0)
    while time.perf_counter() - t0 < max_wait_s:
        if usage <= threshold_pct:
            break
        usage = psutil.cpu_percent(interval=2.0)
    return usage


def _wait_for_idle_gpu(threshold_pct: int = 25, max_wait_s: float = 60.0) -> int:
    """Wait up to max_wait_s if initial GPU utilization exceeds threshold."""
    t0 = time.perf_counter()
    util = 0
    while time.perf_counter() - t0 < max_wait_s:
        try:
            state = gpu_state()
            util = state["utilization_gpu_pct"]
            if util <= threshold_pct:
                break
        except EnergyUnavailable:
            break
        time.sleep(1.0)
    return util


def validate_v1_macs() -> dict[str, Any]:
    """V1: MACs within 5% of analytic count for MLP, MoE, attention, plus Counter unit test."""
    results: dict[str, Any] = {"subchecks": {}}

    # (a) MLP 64-256-256-10, batch 32
    # Analytic: B * (64*256 + 256*256 + 256*10)
    B = 32
    mlp = nn.Sequential(
        nn.Linear(64, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 256, bias=False),
        nn.ReLU(),
        nn.Linear(256, 10, bias=False),
    )
    x_mlp = torch.randn(B, 64)
    res_mlp = count_macs(mlp, x_mlp)
    analytic_mlp = B * (64 * 256 + 256 * 256 + 256 * 10)
    err_mlp = abs(res_mlp.macs - analytic_mlp) / analytic_mlp
    mlp_pass = err_mlp <= 0.05
    results["subchecks"]["mlp"] = {
        "measured_macs": res_mlp.macs,
        "analytic_macs": analytic_mlp,
        "error_pct": err_mlp * 100.0,
        "tolerance_pct": 5.0,
        "pass": mlp_pass,
    }

    # (b) top-1 routed toy mixture of 8 experts (router Linear 64 to 8, experts Linear 64 to 64), batch 32
    # Grouped matrix products so only the executed products exist: B*64*8 + B*64*64
    router = nn.Linear(64, 8, bias=False)
    experts = nn.ModuleList([nn.Linear(64, 64, bias=False) for _ in range(8)])

    def run_moe(inp: torch.Tensor) -> torch.Tensor:
        logits = router(inp)
        chosen = torch.argmax(logits, dim=-1)
        out = torch.empty_like(inp)
        for e_idx in range(8):
            mask = chosen == e_idx
            if mask.any():
                x_e = inp[mask]
                out[mask] = experts[e_idx](x_e)
        return out

    x_moe = torch.randn(B, 64)
    res_moe = count_macs(run_moe, x_moe)
    analytic_moe = B * 64 * 8 + B * 64 * 64
    err_moe = abs(res_moe.macs - analytic_moe) / analytic_moe
    moe_pass = err_moe <= 0.05
    results["subchecks"]["moe"] = {
        "measured_macs": res_moe.macs,
        "analytic_macs": analytic_moe,
        "error_pct": err_moe * 100.0,
        "tolerance_pct": 5.0,
        "pass": moe_pass,
    }

    # (c) single-head attention with explicit matrix products: 4*B*T*d*d + 2*B*T*T*d
    B_attn, T_attn, d_attn = 2, 16, 32
    q_proj = nn.Linear(d_attn, d_attn, bias=False)
    k_proj = nn.Linear(d_attn, d_attn, bias=False)
    v_proj = nn.Linear(d_attn, d_attn, bias=False)
    out_proj = nn.Linear(d_attn, d_attn, bias=False)

    def run_attn(seq: torch.Tensor) -> torch.Tensor:
        q = q_proj(seq)
        k = k_proj(seq)
        v = v_proj(seq)
        scores = torch.bmm(q, k.transpose(-1, -2))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights, v)
        return out_proj(context)

    x_attn = torch.randn(B_attn, T_attn, d_attn)
    res_attn = count_macs(run_attn, x_attn)
    analytic_attn = (
        4 * B_attn * T_attn * d_attn * d_attn + 2 * B_attn * T_attn * T_attn * d_attn
    )
    err_attn = abs(res_attn.macs - analytic_attn) / analytic_attn
    attn_pass = err_attn <= 0.05
    results["subchecks"]["attention"] = {
        "measured_macs": res_attn.macs,
        "analytic_macs": analytic_attn,
        "error_pct": err_attn * 100.0,
        "tolerance_pct": 5.0,
        "pass": attn_pass,
    }

    # Unit test of Counter
    counter = Counter()
    mm_macs = macs_matmul(10, 20, 30)
    counter.add("matmul", mm_macs)
    counter.add("conv", 500)
    c_tot = counter.total()
    c_by = counter.by_name()
    counter_pass = (c_tot == 6500) and (c_by == {"matmul": 6000, "conv": 500})
    results["subchecks"]["counter"] = {
        "total": c_tot,
        "expected_total": 6500,
        "by_name": c_by,
        "pass": counter_pass,
    }

    all_pass = mlp_pass and moe_pass and attn_pass and counter_pass
    results["pass"] = all_pass
    return results


def validate_v2_timing() -> dict[str, Any]:
    """V2: Coefficient of variation <= 5% over 10 repeats at fixed threads (2) and affinity (logical processors 0 and 2, one per physical core; both SMT siblings of a core gave 10-20% jitter).

    For CPU workload >= 20 ms and GPU workload >= 20 ms.
    If a bar fails, retry at most twice, waiting while selected cores are busy.
    """
    results: dict[str, Any] = {
        "settings": {"threads": 2, "cpus": [0, 2]},
        "attempts_cpu": [],
        "attempts_gpu": [],
    }

    with fixed_threads(2, [0, 2]):
        # CPU workload: 1150 x 1150 numpy matmul (~23 ms >= 20 ms)
        np_a = np.random.randn(1150, 1150)
        np_b = np.random.randn(1150, 1150)

        def cpu_workload() -> np.ndarray:
            return np.dot(np_a, np_b)

        cpu_passed = False
        best_cpu_cv = float("inf")
        for attempt in range(3):
            if attempt > 0:
                _wait_for_idle_cpu(threshold_pct=30.0, max_wait_s=20.0)

            res_cpu = time_callable(
                cpu_workload, warmup=5, repeats=10, device="cpu"
            )
            cv_val = res_cpu.cv
            mean_val = res_cpu.mean
            attempt_info = {
                "attempt": attempt + 1,
                "mean_ms": mean_val * 1000.0,
                "std_ms": res_cpu.std * 1000.0,
                "cv_pct": cv_val * 100.0,
                "is_gte_20ms": mean_val >= 0.020,
                "cv_lte_5pct": cv_val <= 0.05,
            }
            results["attempts_cpu"].append(attempt_info)
            if cv_val < best_cpu_cv:
                best_cpu_cv = cv_val
            if mean_val >= 0.020 and cv_val <= 0.05:
                cpu_passed = True
                break

        # GPU workload: 5000 x 5000 torch.mm (~34 ms >= 20 ms)
        gpu_passed = False
        best_gpu_cv = float("inf")
        if torch.cuda.is_available():
            gpu_a = torch.randn(5000, 5000, device="cuda")
            gpu_b = torch.randn(5000, 5000, device="cuda")

            def gpu_workload() -> torch.Tensor:
                return torch.mm(gpu_a, gpu_b)

            for attempt in range(3):
                if attempt > 0:
                    _wait_for_idle_cpu(threshold_pct=30.0, max_wait_s=20.0)

                res_gpu = time_callable(
                    gpu_workload, warmup=5, repeats=10, device="cuda"
                )
                cv_val = res_gpu.cv
                mean_val = res_gpu.mean
                attempt_info = {
                    "attempt": attempt + 1,
                    "mean_ms": mean_val * 1000.0,
                    "std_ms": res_gpu.std * 1000.0,
                    "cv_pct": cv_val * 100.0,
                    "is_gte_20ms": mean_val >= 0.020,
                    "cv_lte_5pct": cv_val <= 0.05,
                }
                results["attempts_gpu"].append(attempt_info)
                if cv_val < best_gpu_cv:
                    best_gpu_cv = cv_val
                if mean_val >= 0.020 and cv_val <= 0.05:
                    gpu_passed = True
                    break

    results["cpu_passed"] = cpu_passed
    results["gpu_passed"] = gpu_passed
    results["best_cpu_cv_pct"] = best_cpu_cv * 100.0
    results["best_gpu_cv_pct"] = best_gpu_cv * 100.0
    results["pass"] = cpu_passed and gpu_passed
    return results


def validate_v3_energy() -> dict[str, Any]:
    """V3 GPU energy:

    (i) loaded mean power > idle baseline
    (ii) running for 2x calls gives joules within 25% of 2x the first value
    (iii) counter-based and integrated joules agree within 15%.
    """
    results: dict[str, Any] = {}
    if not torch.cuda.is_available():
        results["pass"] = False
        results["error"] = "CUDA is unavailable"
        return results

    # Wait up to 60s if initial GPU utilization > 25%
    initial_util = _wait_for_idle_gpu(threshold_pct=25, max_wait_s=60.0)
    results["initial_gpu_utilization_pct"] = initial_util

    # Idle baseline over 5.0 seconds
    idle_res = idle_baseline(seconds=5.0)
    idle_power_w = idle_res["mean_watts"]
    results["idle_mean_watts"] = idle_power_w

    gpu_a = torch.randn(4000, 4000, device="cuda")
    gpu_b = torch.randn(4000, 4000, device="cuda")

    def workload() -> torch.Tensor:
        return torch.mm(gpu_a, gpu_b)

    # Warmup
    for _ in range(15):
        workload()
    torch.cuda.synchronize()

    # (i) & (iii): measure energy for 3.0 s
    m_res = measure_energy(workload, min_duration_s=3.0)
    counter_j1 = m_res["counter_joules"]
    integ_j1 = m_res["integrated_joules"]
    loaded_power_w = m_res["mean_watts"]
    calls_1 = m_res["calls"]

    results["first_run"] = {
        "counter_joules": counter_j1,
        "integrated_joules": integ_j1,
        "mean_watts": loaded_power_w,
        "calls": calls_1,
        "duration_s": m_res["duration_s"],
    }

    # (i) loaded mean power > idle baseline
    power_above_idle = loaded_power_w > idle_power_w
    results["subchecks"] = {
        "loaded_gt_idle": {
            "loaded_power_w": loaded_power_w,
            "idle_power_w": idle_power_w,
            "pass": power_above_idle,
        }
    }

    # (iii) counter vs integrated within 15%
    ci_diff_pct = abs(counter_j1 - integ_j1) / counter_j1 * 100.0 if counter_j1 > 0 else float("inf")
    ci_pass = ci_diff_pct <= 15.0
    results["subchecks"]["counter_vs_integrated"] = {
        "counter_joules": counter_j1,
        "integrated_joules": integ_j1,
        "diff_pct": ci_diff_pct,
        "tolerance_pct": 15.0,
        "pass": ci_pass,
    }

    # (ii) running for 2x calls gives joules within 25% of 2x first value
    import pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    torch.cuda.synchronize()
    e_start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    for _ in range(2 * calls_1):
        workload()
    torch.cuda.synchronize()
    e_end_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    counter_j2 = float(e_end_mj - e_start_mj) / 1000.0

    expected_2x = 2.0 * counter_j1
    diff_2x_pct = (
        abs(counter_j2 - expected_2x) / expected_2x * 100.0 if expected_2x > 0 else float("inf")
    )
    linearity_pass = diff_2x_pct <= 25.0
    results["subchecks"]["two_x_calls_linearity"] = {
        "first_calls": calls_1,
        "first_joules": counter_j1,
        "two_x_calls": 2 * calls_1,
        "two_x_joules": counter_j2,
        "expected_joules": expected_2x,
        "diff_pct": diff_2x_pct,
        "tolerance_pct": 25.0,
        "pass": linearity_pass,
    }

    results["pass"] = power_above_idle and ci_pass and linearity_pass
    return results


def validate_v4_bytes() -> dict[str, Any]:
    """V4: Active bytes vs total parameter bytes for dense MLP and routed toy MoE."""
    results: dict[str, Any] = {"subchecks": {}}

    # Dense MLP
    mlp = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    x_mlp = torch.randn(32, 64)
    tot_mlp = total_param_bytes(mlp)
    act_mlp, _ = active_param_bytes(mlp, mlp, x_mlp)
    mlp_pass = act_mlp == tot_mlp
    results["subchecks"]["dense_mlp_exact_match"] = {
        "total_bytes": tot_mlp,
        "active_bytes": act_mlp,
        "pass": mlp_pass,
    }

    # Routed toy
    class ToyMoE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = nn.Linear(64, 8)
            self.experts = nn.ModuleList([nn.Linear(64, 64) for _ in range(8)])

        def forward(
            self, x: torch.Tensor, force_expert: int | None = None
        ) -> torch.Tensor:
            logits = self.router(x)
            if force_expert is not None:
                chosen = torch.full((x.shape[0],), force_expert, dtype=torch.long)
            else:
                chosen = torch.argmax(logits, dim=-1)
            out = torch.empty_like(x)
            for e_idx in range(8):
                mask = chosen == e_idx
                if mask.any():
                    x_e = x[mask]
                    out[mask] = self.experts[e_idx](x_e)
            return out

    moe = ToyMoE()
    router_bytes = total_param_bytes(moe.router)
    expert_bytes = total_param_bytes(moe.experts[0])
    expected_one_expert = router_bytes + expert_bytes

    # Batch 1
    x_1 = torch.randn(1, 64)
    act_b1, _ = active_param_bytes(moe, moe, x_1)
    b1_pass = act_b1 == expected_one_expert
    results["subchecks"]["moe_batch1_one_expert"] = {
        "expected_bytes": expected_one_expert,
        "active_bytes": act_b1,
        "pass": b1_pass,
    }

    # Forced expert 3
    x_batch = torch.randn(16, 64)
    act_forced, by_m_forced = active_param_bytes(
        moe, lambda inp: moe(inp, force_expert=3), x_batch
    )
    forced_pass = (act_forced == expected_one_expert) and ("experts.3" in by_m_forced)
    results["subchecks"]["moe_forced_expert_3"] = {
        "expected_bytes": expected_one_expert,
        "active_bytes": act_forced,
        "has_expert_3": "experts.3" in by_m_forced,
        "pass": forced_pass,
    }

    results["pass"] = mlp_pass and b1_pass and forced_pass
    return results


def validate_v5_memory() -> dict[str, Any]:
    """V5: 256 MB torch GPU allocation peak within bounds; 200 MB numpy RSS within 20%."""
    results: dict[str, Any] = {"subchecks": {}}

    # GPU 256 MB allocation
    gpu_pass = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        expected_gpu_bytes = 256 * 1024 * 1024

        def alloc_gpu() -> None:
            t = torch.empty((67108864,), dtype=torch.float32, device="cuda")
            time.sleep(0.02)
            del t

        mem_gpu = peak_memory(alloc_gpu, device="cuda")
        peak_cuda = mem_gpu["cuda_peak_bytes"] or 0
        min_bound = expected_gpu_bytes
        max_bound = 1.1 * expected_gpu_bytes + start_alloc

        gpu_pass = (peak_cuda >= min_bound) and (peak_cuda <= max_bound)
        results["subchecks"]["gpu_256mb"] = {
            "expected_bytes": expected_gpu_bytes,
            "measured_peak_bytes": peak_cuda,
            "min_bound": min_bound,
            "max_bound": max_bound,
            "pass": gpu_pass,
        }

    # CPU 200 MB numpy allocation
    expected_cpu_bytes = 200 * 1024 * 1024

    def alloc_cpu() -> None:
        arr = np.ones((52428800,), dtype=np.float32)
        time.sleep(0.05)
        del arr

    mem_cpu = peak_memory(alloc_cpu, device="cpu")
    rss_diff = mem_cpu["rss_diff_bytes"]
    rss_err = abs(rss_diff - expected_cpu_bytes) / expected_cpu_bytes
    cpu_pass = rss_err <= 0.20

    results["subchecks"]["cpu_200mb_numpy_rss"] = {
        "expected_bytes": expected_cpu_bytes,
        "measured_rss_diff_bytes": rss_diff,
        "error_pct": rss_err * 100.0,
        "tolerance_pct": 20.0,
        "pass": cpu_pass,
    }

    results["pass"] = gpu_pass and cpu_pass
    return results


def run_all_validations() -> dict[str, Any]:
    """Execute all validation bars V1 through V5 and print detailed summary."""
    print("=" * 65)
    print("COSTLAB VALIDATION SUITE (M0 Step 4)")
    print("=" * 65)

    print("\n[V1 MACs]")
    v1 = validate_v1_macs()
    mlp_chk = v1["subchecks"]["mlp"]
    moe_chk = v1["subchecks"]["moe"]
    attn_chk = v1["subchecks"]["attention"]
    cnt_chk = v1["subchecks"]["counter"]
    print(
        f"  (a) MLP 64-256-256-10: measured {mlp_chk['measured_macs']:,} MACs vs analytic {mlp_chk['analytic_macs']:,} (error {mlp_chk['error_pct']:.2f}% <= 5%) -> {'PASS' if mlp_chk['pass'] else 'FAIL'}"
    )
    print(
        f"  (b) MoE 8 experts:     measured {moe_chk['measured_macs']:,} MACs vs analytic {moe_chk['analytic_macs']:,} (error {moe_chk['error_pct']:.2f}% <= 5%) -> {'PASS' if moe_chk['pass'] else 'FAIL'}"
    )
    print(
        f"  (c) Self-Attention:    measured {attn_chk['measured_macs']:,} MACs vs analytic {attn_chk['analytic_macs']:,} (error {attn_chk['error_pct']:.2f}% <= 5%) -> {'PASS' if attn_chk['pass'] else 'FAIL'}"
    )
    print(
        f"  (d) Counter Unit Test: total {cnt_chk['total']} == {cnt_chk['expected_total']} -> {'PASS' if cnt_chk['pass'] else 'FAIL'}"
    )
    print(f"  Bar V1 Overall: {'PASS' if v1['pass'] else 'FAIL'}")

    print("\n[V2 Timing]")
    v2 = validate_v2_timing()
    cpu_last = v2["attempts_cpu"][-1]
    gpu_last = v2["attempts_gpu"][-1] if v2["attempts_gpu"] else {}
    print(
        f"  CPU Timing (pinned 0,2, 2 threads): mean {cpu_last['mean_ms']:.2f} ms (>= 20 ms), CV {cpu_last['cv_pct']:.2f}% (<= 5%, attempts {len(v2['attempts_cpu'])}) -> {'PASS' if v2['cpu_passed'] else 'FAIL'}"
    )
    if v2["attempts_gpu"]:
        print(
            f"  GPU Timing (CUDA synchronize):      mean {gpu_last['mean_ms']:.2f} ms (>= 20 ms), CV {gpu_last['cv_pct']:.2f}% (<= 5%, attempts {len(v2['attempts_gpu'])}) -> {'PASS' if v2['gpu_passed'] else 'FAIL'}"
        )
    print(f"  Bar V2 Overall: {'PASS' if v2['pass'] else 'FAIL'}")

    print("\n[V3 GPU Energy]")
    v3 = validate_v3_energy()
    p_chk = v3["subchecks"]["loaded_gt_idle"]
    ci_chk = v3["subchecks"]["counter_vs_integrated"]
    lin_chk = v3["subchecks"]["two_x_calls_linearity"]
    print(
        f"  (i)   Loaded Power > Idle:  {p_chk['loaded_power_w']:.2f} W > {p_chk['idle_power_w']:.2f} W -> {'PASS' if p_chk['pass'] else 'FAIL'}"
    )
    print(
        f"  (ii)  2x Calls Linearity:   {lin_chk['two_x_joules']:.2f} J vs expected {lin_chk['expected_joules']:.2f} J (diff {lin_chk['diff_pct']:.2f}% <= 25%) -> {'PASS' if lin_chk['pass'] else 'FAIL'}"
    )
    print(
        f"  (iii) Counter vs Integrated: counter {ci_chk['counter_joules']:.2f} J vs integrated {ci_chk['integrated_joules']:.2f} J (diff {ci_chk['diff_pct']:.2f}% <= 15%) -> {'PASS' if ci_chk['pass'] else 'FAIL'}"
    )
    print(f"  Bar V3 Overall: {'PASS' if v3['pass'] else 'FAIL'}")

    print("\n[V4 Bytes]")
    v4 = validate_v4_bytes()
    mlp_b = v4["subchecks"]["dense_mlp_exact_match"]
    moe_b1 = v4["subchecks"]["moe_batch1_one_expert"]
    moe_f3 = v4["subchecks"]["moe_forced_expert_3"]
    print(
        f"  Dense MLP:    active {mlp_b['active_bytes']:,} bytes == total {mlp_b['total_bytes']:,} bytes -> {'PASS' if mlp_b['pass'] else 'FAIL'}"
    )
    print(
        f"  MoE Batch 1:  active {moe_b1['active_bytes']:,} bytes == router + 1 expert ({moe_b1['expected_bytes']:,} bytes) -> {'PASS' if moe_b1['pass'] else 'FAIL'}"
    )
    print(
        f"  MoE Forced 3: active {moe_f3['active_bytes']:,} bytes == router + expert 3 ({moe_f3['expected_bytes']:,} bytes) -> {'PASS' if moe_f3['pass'] else 'FAIL'}"
    )
    print(f"  Bar V4 Overall: {'PASS' if v4['pass'] else 'FAIL'}")

    print("\n[V5 Memory]")
    v5 = validate_v5_memory()
    gpu_m = v5["subchecks"]["gpu_256mb"]
    cpu_m = v5["subchecks"]["cpu_200mb_numpy_rss"]
    print(
        f"  GPU 256 MB: measured peak {gpu_m['measured_peak_bytes']:,} bytes in [{gpu_m['min_bound']:,}, {int(gpu_m['max_bound']):,}] -> {'PASS' if gpu_m['pass'] else 'FAIL'}"
    )
    print(
        f"  CPU 200 MB: measured RSS diff {cpu_m['measured_rss_diff_bytes']:,} bytes vs expected {cpu_m['expected_bytes']:,} (error {cpu_m['error_pct']:.2f}% <= 20%) -> {'PASS' if cpu_m['pass'] else 'FAIL'}"
    )
    print(f"  Bar V5 Overall: {'PASS' if v5['pass'] else 'FAIL'}")

    overall_pass = v1["pass"] and v2["pass"] and v3["pass"] and v4["pass"] and v5["pass"]

    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overall_pass": overall_pass,
        "bars": {
            "V1_macs": v1,
            "V2_timing": v2,
            "V3_energy": v3,
            "V4_bytes": v4,
            "V5_memory": v5,
        },
    }


def main() -> None:
    """CLI entrypoint for costlab.validate."""
    parser = argparse.ArgumentParser(description="Validate costlab measurement bars.")
    parser.add_argument(
        "--out",
        default="costlab_validation.json",
        help="Output path for validation JSON results.",
    )
    args = parser.parse_args()

    results = run_all_validations()

    out_path = Path(args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, cls=_CostlabJSONEncoder)

    print("")
    print("=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    for bar_name, bar_res in results["bars"].items():
        status = "PASS" if bar_res.get("pass") else "FAIL"
        print(f"  {bar_name:12s}: {status}")
    print("-" * 65)
    print(f"  Overall     : {'PASS' if results['overall_pass'] else 'FAIL'}")
    print(f"  Saved JSON  : {out_path.resolve()}")
    print("=" * 65)

    if not results["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
