"""GPU repeat runner for milestone M1 wide baselines (Job G-1).

Repeats wide baselines (A3, A3-2, A4, A4-2, A6, A8) on GPU while pools stay on CPU.
Maintains exact schema 1.2.0 compatibility, determinism, and isolated CPU pinning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Section 2.2: Set determinism before CUDA initializes
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Disable TF32 and enforce deterministic algorithms on CUDA
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)

from costlab.bytes import active_param_bytes
from costlab.count import count_macs
from costlab.energy import EnergyUnavailable, gpu_state, idle_baseline, measure_energy
from costlab.env import environment, fixed_threads
from costlab.memory import peak_memory
from costlab.report import _CostlabJSONEncoder
from costlab.timing import time_callable
from growmoe.accounting import pool_report
from growmoe.baselines import JointOracle, ReplayNet
from growmoe.experts import MLPExpert
from growmoe.metrics import accuracy_stats
from growmoe.types import Prediction, StepRecord
from m1.arms import Arm, build_arm
from m1.baselines_extra import RandomFeatureRidge
from m1.beds import Bed, build_bed
from m1.cell import (
    SCHEMA_VERSION,
    accounting_block,
    cell_filename,
    compute_tuning_ref,
    is_valid_cell_record,
    load_cell_record,
    read_pool_budget,
    read_pool_macs,
    record_matches_config,
    save_cell_record,
)
from m1.config_m1 import DEV_SEEDS, PINNING_CPUS, PINNING_THREADS, REPORT_SEEDS
from m1.freeze import freeze_digest, require_frozen
from m1.queue import load_tuning_map, parse_seeds

REPEATED_ARMS = ("A3", "A3-2", "A4", "A4-2", "A6", "A8")
ENERGY_COOLDOWN_S = 5.0          # the card is still hot and clocked up right after a stream: wait before measuring its idle power
ENERGY_IDLE_MAX_SPREAD_W = 5.0   # an idle baseline whose power moves more than this over its second is not an idle baseline
EQUAL_MEMORY_ARMS_R1 = ("A3",)
EQUAL_MEMORY_ARMS_R2 = ("A3-2",)
COMPUTE_MATCHED_ARMS_R1 = ("A4",)
COMPUTE_MATCHED_ARMS_R2 = ("A4-2",)


def adapt_arm_for_device(arm: Arm, device: str = "cuda") -> Arm:
    """Move arm computation to target device and install host-device boundary wrappers.

    Host-to-device batch copy is placed inside the observe/predict calls so it is
    properly measured in timed regions, and predict returns probabilities on CPU.
    """
    if device == "cpu":
        return arm

    target_dev = torch.device(device)

    # 1. ReplayNet (A3, A3-2, A4, A4-2)
    if isinstance(getattr(arm, "model", None), ReplayNet):
        net = arm.model.net
        net.to(target_dev)
        for state in arm.model.opt.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(target_dev)
        arm.model.buffer.x = arm.model.buffer.x.to(target_dev)
        arm.model.buffer.y = arm.model.buffer.y.to(target_dev)

        orig_obs = arm.model.observe

        def observe_gpu(x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> Any:
            x_dev = x.to(target_dev) if isinstance(x, torch.Tensor) else x
            y_dev = y.to(target_dev) if isinstance(y, torch.Tensor) else y
            return orig_obs(x_dev, y=y_dev, ctx=ctx)

        arm.model.observe = observe_gpu

        orig_pred = arm.model.predict

        def predict_gpu(x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
            x_dev = x.to(target_dev) if isinstance(x, torch.Tensor) else x
            pred = orig_pred(x_dev, ctx=ctx)
            probs_cpu = pred.probs.to("cpu")
            if target_dev.type == "cuda":
                torch.cuda.synchronize()
            return Prediction(probs=probs_cpu, expert=pred.expert, p_nov=pred.p_nov, unknown=pred.unknown)

        arm.model.predict = predict_gpu

    # 2. JointOracle (A8)
    elif isinstance(getattr(arm, "model", None), JointOracle):
        net = arm.model.net
        net.to(target_dev)
        for state in arm.model.opt.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(target_dev)
        arm.model._x = arm.model._x.to(target_dev)
        arm.model._y = arm.model._y.to(target_dev)

        def _store_gpu(x: Tensor, y: Tensor) -> None:
            need = arm.model._n + x.shape[0]
            if need > arm.model._x.shape[0]:
                cap = max(need, 2 * arm.model._x.shape[0])
                arm.model._x = torch.cat(
                    [arm.model._x[: arm.model._n], torch.zeros(cap - arm.model._n, arm.model.cfg.d, device=target_dev)]
                )
                arm.model._y = torch.cat(
                    [
                        arm.model._y[: arm.model._n],
                        torch.zeros(cap - arm.model._n, dtype=torch.long, device=target_dev),
                    ]
                )
            arm.model._x[arm.model._n : need], arm.model._y[arm.model._n : need] = x.detach(), y.detach()
            arm.model._n = need

        arm.model._store = _store_gpu

        orig_obs = arm.model.observe

        def observe_gpu(x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> Any:
            x_dev = x.to(target_dev) if isinstance(x, torch.Tensor) else x
            y_dev = y.to(target_dev) if isinstance(y, torch.Tensor) else y
            return orig_obs(x_dev, y=y_dev, ctx=ctx)

        arm.model.observe = observe_gpu

        orig_pred = arm.model.predict

        def predict_gpu(x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
            x_dev = x.to(target_dev) if isinstance(x, torch.Tensor) else x
            pred = orig_pred(x_dev, ctx=ctx)
            probs_cpu = pred.probs.to("cpu")
            if target_dev.type == "cuda":
                torch.cuda.synchronize()
            return Prediction(probs=probs_cpu, expert=pred.expert, p_nov=pred.p_nov, unknown=pred.unknown)

        arm.model.predict = predict_gpu

    # 3. RandomFeatureRidge (A6)
    elif isinstance(getattr(arm, "model", None), RandomFeatureRidge):
        m = arm.model
        m.W_proj = m.W_proj.to(target_dev)
        m._G = m._G.to(target_dev)
        m._B = m._B.to(target_dev)
        m._cnt = m._cnt.to(target_dev)
        if m._W is not None:
            m._W = m._W.to(target_dev)

        def _solve_weights_gpu() -> Tensor:
            if m._W is None:
                A = m._G + m.ridge * torch.eye(m.D, dtype=torch.float64, device=target_dev)
                A = 0.5 * (A + A.T)
                try:
                    L = torch.linalg.cholesky(A)
                    m._W = torch.cholesky_solve(m._B, L)
                except torch.linalg.LinAlgError:
                    m._W = torch.linalg.solve(A, m._B)
            return m._W

        m._solve_weights = _solve_weights_gpu

        def observe_gpu(x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
            m.step += 1
            if y is not None:
                x_dev = x.to(target_dev)
                y_dev = y.to(target_dev)
                x_unit = m._unit(x_dev)
                phi = F.relu(x_unit @ m.W_proj).double()
                m._G += phi.T @ phi
                y_long = y_dev.long()
                Y = F.one_hot(y_long, m.n_classes).double()
                m._B += phi.T @ Y
                m._cnt.index_add_(0, y_long, torch.ones(len(y_dev), dtype=torch.long, device=target_dev))
                m._W = None
            return m._record(x.shape[0], -1 if y is None else 0, n_experts=1)

        m.observe = observe_gpu

        def predict_gpu(x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
            seen = m._cnt > 0
            if not bool(seen.any()):
                return m._empty_prediction(x.shape[0])
            W = _solve_weights_gpu()
            x_dev = x.to(target_dev)
            x_unit = m._unit(x_dev)
            phi = F.relu(x_unit @ m.W_proj).double()
            logits = (phi @ W).float() * m.scale
            logits = logits.masked_fill(~seen, float("-inf"))
            probs_cpu = torch.softmax(logits, dim=1).to("cpu")
            if target_dev.type == "cuda":
                torch.cuda.synchronize()
            return m._prediction(probs_cpu, 0)

        m.predict = predict_gpu

    return arm


def execute_gpu_cell(
    bed: Bed,
    arm_name: str,
    pool_memory_bytes: Optional[int] = None,
    pool_macs_budget: Optional[int] = None,
    split: str = "dev",
    pinning: Optional[Dict[str, Any]] = None,
    profile: str = "full",
    tuning: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    cpu_record: Optional[Dict[str, Any]] = None,
    reuse_cpu_update_macs: bool = False,
    **arm_kwargs: Any,
) -> Dict[str, Any]:
    """Execute one repeated benchmark cell on GPU (or CPU test) with costlab profiling.

    The update multiply-add count does not depend on the device. With ``reuse_cpu_update_macs`` and a CPU record of the same
    configuration it is taken from that record instead of being counted again by a second full CPU pass over the stream
    (which costs as much as the CPU cell itself); how it was obtained is stored in ``macs_check``.
    """
    if split == "report":
        require_frozen([bed.seed])

    if profile not in ("full", "det", "light"):
        raise ValueError(f"Unknown profile {profile!r}; must be 'full', 'det' or 'light'.")

    env_info = environment()
    commit = env_info.get("git_commit") or "unknown"
    is_cuda = (device == "cuda" or device.startswith("cuda")) and torch.cuda.is_available()

    # Capture initial NVML state if on GPU
    nvml_state_before: Optional[Dict[str, Any]] = None
    if is_cuda:
        try:
            nvml_state_before = gpu_state()
        except (EnergyUnavailable, Exception) as exc:
            nvml_state_before = {"error": str(exc)}

    # Build arm on CPU exactly as m1.arms.build_arm does
    arm = build_arm(
        arm_name,
        bed,
        pool_memory_bytes=pool_memory_bytes,
        pool_macs_budget=pool_macs_budget,
        tuning=tuning,
        **arm_kwargs,
    )

    # Compute device-independent counted multiply-adds on CPU copy
    macs_check: Dict[str, Any] = {}
    cpu_update_macs = cpu_record.get("update_cost", {}).get("counted_macs") if cpu_record is not None else None
    if profile in ("full", "det"):
        if arm_name in ("A7", "A7all", "A7-2"):
            update_macs = 0
        elif reuse_cpu_update_macs and cpu_update_macs is not None:
            update_macs = int(cpu_update_macs)
            macs_check["update"] = "reused_from_cpu_cell"
        else:

            def _mac_pass() -> None:
                extra = dict(arm_kwargs)
                if hasattr(arm, "inner_steps") and "inner_steps" not in extra:
                    extra["inner_steps"] = arm.inner_steps
                sh = build_arm(
                    arm_name,
                    bed,
                    pool_memory_bytes=pool_memory_bytes,
                    pool_macs_budget=pool_macs_budget,
                    tuning=tuning,
                    **extra,
                )
                for x, y, _ in bed:
                    sh.observe(x, y)

            update_macs = count_macs(_mac_pass).macs
    else:
        update_macs = None

    # Adapt arm for GPU computation
    if device != "cpu":
        adapt_arm_for_device(arm, device=device)

    # Warm-up (CUDA context, cuBLAS handle, one throw-away step on net of same width)
    warmup_s = 0.0
    if is_cuda:
        torch.cuda.synchronize()
        t_w0 = time.perf_counter()
        if hasattr(arm, "hidden"):
            w = getattr(arm, "hidden", 128)
            dummy_net = MLPExpert(bed.d, w, 1, bed.n_classes, seed=8888).to(device)
            dummy_opt = torch.optim.Adam(dummy_net.parameters(), lr=1e-3)
            dummy_x = torch.randn(64, bed.d, device=device)
            dummy_y = torch.randint(0, bed.n_classes, (64,), device=device)
            dummy_net.train_step(dummy_x, dummy_y, dummy_opt)
            del dummy_net, dummy_opt, dummy_x, dummy_y
        elif hasattr(arm, "D"):
            d_val = getattr(arm, "D", 1024)
            dummy_x = torch.randn(64, bed.d, device=device)
            dummy_y = torch.randint(0, bed.n_classes, (64,), device=device)
            dummy_w = torch.randn(bed.d, d_val, device=device)
            dummy_phi = F.relu(dummy_x @ dummy_w).double()
            dummy_G = dummy_phi.T @ dummy_phi + 0.1 * torch.eye(d_val, dtype=torch.float64, device=device)
            dummy_B = dummy_phi.T @ F.one_hot(dummy_y, bed.n_classes).double()
            dummy_L = torch.linalg.cholesky(dummy_G)
            torch.cholesky_solve(dummy_B, dummy_L)
            del dummy_x, dummy_y, dummy_w, dummy_phi, dummy_G, dummy_B, dummy_L
        torch.cuda.synchronize()
        warmup_s = float(time.perf_counter() - t_w0)

    obs_durations: List[float] = []
    eval_durations: List[float] = []
    reg_durations: Dict[str, float] = {}
    bytes_by_regime: Dict[str, int] = {}
    R_rows: List[List[float]] = []
    learned_at: Dict[int, int] = {}
    wall_clock_total = 0.0

    tests = bed.test_sets

    def _eval_row() -> List[float]:
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        row = [
            float((arm.predict(x).probs.argmax(dim=-1).cpu() == y.cpu()).float().mean().item())
            for x, y in (tests[r] for r in range(bed.n_regimes))
        ]
        if is_cuda:
            torch.cuda.synchronize()
        eval_durations.append(time.perf_counter() - t0)
        return row

    def _train_stream() -> None:
        nonlocal wall_clock_total
        cur_phase = -1
        last_regime = -1
        for x_b, y_b, regimes, first_regimes, phase_idx in bed.walk_full(64):
            regime_id = regimes[0]
            if phase_idx != cur_phase:
                if cur_phase >= 0:
                    if profile in ("full", "det"):
                        R_rows.append(_eval_row())
                    bytes_by_regime.setdefault(str(last_regime), arm.memory_bytes())
                cur_phase = phase_idx
                last_regime = regime_id
            for r_first in first_regimes:
                if r_first not in learned_at:
                    learned_at[r_first] = len(R_rows)

            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            arm.observe(x_b, y_b)
            if is_cuda:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            obs_durations.append(dt)
            r_str = str(regime_id)
            reg_durations[r_str] = reg_durations.get(r_str, 0.0) + dt
            wall_clock_total += dt

        R_rows.append(_eval_row())
        if last_regime >= 0:
            bytes_by_regime.setdefault(str(last_regime), arm.memory_bytes())

    if profile == "full":
        peak_mem_profile = peak_memory(_train_stream, device=device)
    else:
        _train_stream()
        peak_mem_profile = None

    R = np.array(R_rows, dtype=np.float64)
    if len(R_rows) > 1:
        never = [r for r in range(bed.n_regimes) if r not in learned_at]
        if never:
            raise ValueError(f"regimes {never} have no first-visit sample in stream of {bed.name}")
        la = [learned_at[r] for r in range(bed.n_regimes)]
        stats = accuracy_stats(R, learned_at=la, chance=1.0 / bed.n_classes)
        forgetting = None if np.isnan(stats["forgetting"]) else float(stats["forgetting"])
    else:
        forgetting = None

    test_x = bed.test_sets[0][0]
    answer_cost: Dict[str, Any] = {}
    if profile in ("full", "det"):
        for b in (1, 32):
            x_b = test_x[torch.arange(b) % len(test_x)]
            fn = lambda: arm.predict(x_b)
            mod = arm.get_torch_modules()
            act_b = active_param_bytes(mod, fn).total if mod is not None else 0
            entry: Dict[str, Any] = {
                "batch_size": b,
                "rows_timed": int(len(x_b)),
                "counted_macs": count_macs(fn).macs,
                "active_bytes": act_b,
            }
            if profile == "full":
                t_res = time_callable(fn, warmup=5, repeats=10, device=device)
                entry.update(
                    latency_median_s=t_res.median,
                    latency_mean_s=t_res.mean,
                    latency_std_s=t_res.std,
                    latency_cv=t_res.cv,
                    latency_p95_s=t_res.p95,
                    items_per_second=float(b / t_res.median) if t_res.median > 0 else 0.0,
                )
            answer_cost[str(b)] = entry

    # Verify counted multiply-adds equality against CPU cell if provided; the outcome is stored in the record, not only printed
    if cpu_record is None:
        macs_check.setdefault("update", "no_cpu_cell")
        macs_check["answer_1"] = macs_check["answer_32"] = "no_cpu_cell"
    else:
        cpu_macs = cpu_update_macs
        if "update" not in macs_check:
            if cpu_macs is None or update_macs is None:
                macs_check["update"] = "not_comparable"
            elif cpu_macs == update_macs:
                macs_check["update"] = "equal"
            else:
                macs_check["update"] = "mismatch"
                macs_check["cpu_update_macs"], macs_check["gpu_update_macs"] = int(cpu_macs), int(update_macs)
                print(
                    f"[MISMATCH] Update MACs mismatch for {bed.name} {arm_name} seed={bed.seed}: "
                    f"CPU {cpu_macs} vs GPU {update_macs}"
                )
        for b in ("1", "32"):
            c_ans = cpu_record.get("answer_cost", {}).get(b, {}).get("counted_macs")
            g_ans = answer_cost.get(b, {}).get("counted_macs")
            if c_ans is None or g_ans is None:
                macs_check[f"answer_{b}"] = "not_comparable"
            elif c_ans == g_ans:
                macs_check[f"answer_{b}"] = "equal"
            else:
                macs_check[f"answer_{b}"] = "mismatch"
                print(
                    f"[MISMATCH] Answer MACs (b={b}) mismatch for {bed.name} {arm_name} seed={bed.seed}: "
                    f"CPU {c_ans} vs GPU {g_ans}"
                )

    # Section 2.5: Energy profiling
    energy_info: Dict[str, Any] = {"available": False, "reason": "device is cpu" if device == "cpu" else "unmeasured"}
    if is_cuda and profile == "full":
        try:
            torch.cuda.synchronize()
            time.sleep(ENERGY_COOLDOWN_S)
            idle = idle_baseline(seconds=1.0)
            # Answer batch 32 energy
            x_32 = test_x[torch.arange(32) % len(test_x)]
            ans_fn = lambda: arm.predict(x_32)
            ans_energy = measure_energy(ans_fn, min_duration_s=1.0)

            # 100 update steps energy on a scratch arm
            scratch_arm = build_arm(
                arm_name,
                bed,
                pool_memory_bytes=pool_memory_bytes,
                pool_macs_budget=pool_macs_budget,
                tuning=tuning,
                **arm_kwargs,
            )
            adapt_arm_for_device(scratch_arm, device=device)

            def _100_updates() -> None:
                step_cnt = 0
                while step_cnt < 100:
                    for xb, yb, _, _, _ in bed.walk_full(64):
                        scratch_arm.observe(xb, yb)
                        step_cnt += 1
                        if step_cnt >= 100:
                            break

            up_energy = measure_energy(_100_updates, min_duration_s=1.0)
            del scratch_arm

            idle_w = float(idle["mean_watts"])
            idle_stable = float(idle["max_watts"]) - float(idle["min_watts"]) <= ENERGY_IDLE_MAX_SPREAD_W

            def _net_joules_per_call(m: Dict[str, Any]) -> Optional[float]:
                """Gross counter joules minus the idle power over the same time, per call; None when the baseline is not stable."""
                if not idle_stable or m["calls"] <= 0:
                    return None
                return (float(m["counter_joules"]) - idle_w * float(m["duration_s"])) / m["calls"]

            energy_info = {
                "available": True,
                "cooldown_s": ENERGY_COOLDOWN_S,
                "idle_baseline": idle,
                "idle_stable": idle_stable,
                "answer_batch_32": ans_energy,
                "update_100_steps": up_energy,
                "joules_per_answer_batch_32": ans_energy["joules_per_call"],
                "joules_per_100_updates": up_energy["joules_per_call"],
                "net_joules_per_answer_batch_32": _net_joules_per_call(ans_energy),
                "net_joules_per_100_updates": _net_joules_per_call(up_energy),
            }
        except (EnergyUnavailable, Exception) as exc:
            energy_info = {"available": False, "reason": str(exc)}

    bed_info = {
        "d": bed.d,
        "classes": bed.n_classes,
        "regimes": bed.n_regimes,
        "phases": bed.n_phases,
        "total_samples": bed.total_samples,
        "regime_names": [bed.regime_info[r]["name"] for r in range(bed.n_regimes)],
    }
    if getattr(bed, "label_blocks", None) is not None:
        bed_info["label_blocks"] = bed.label_blocks

    reference_pool: Optional[Dict[str, Any]] = None
    if arm_name in ("A3", "A4", "A7"):
        reference_pool = {"arm": "A1", "bytes": pool_memory_bytes, "macs": pool_macs_budget}
    elif arm_name in ("A3-2", "A4-2", "A7-2"):
        reference_pool = {"arm": "A2", "bytes": pool_memory_bytes, "macs": pool_macs_budget}

    tuning_ref = compute_tuning_ref(tuning)
    update_cost = (
        {"counted_macs": update_macs}
        if profile == "det"
        else {
            "wall_clock_per_observe": obs_durations,
            "wall_clock_per_regime": reg_durations,
            "wall_clock_total": wall_clock_total,
            "counted_macs": update_macs,
            "peak_memory": peak_mem_profile,
            "eval_wall_clock_per_regime_boundary": eval_durations,
        }
    )

    gpu_info: Optional[Dict[str, Any]] = None
    if is_cuda:
        gpu_info = {
            "name": env_info.get("gpu_name"),
            "driver": env_info.get("gpu_driver"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "state_before": nvml_state_before,
        }

    numerics = {
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "precision": "fp32_fp64",
    }

    vram_peak_bytes = (
        peak_mem_profile.get("cuda_peak_bytes")
        if (peak_mem_profile and is_cuda)
        else (int(torch.cuda.max_memory_allocated()) if is_cuda else None)
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "bed": bed.name,
        "n_regimes": bed.n_regimes,
        "arm": arm_name,
        "seed": bed.seed,
        "split": split,
        "freeze_sha256": freeze_digest() if split == "report" else None,
        "commit": commit,
        "environment": env_info,
        "pinning": pinning,
        "profile": profile,
        "bed_info": bed_info,
        "tuning_ref": tuning_ref,
        "reference_pool": reference_pool,
        "uncounted_note": getattr(arm, "uncounted_note", False),
        "pool_time_split": None,
        "config": json.loads(json.dumps(arm.config(), cls=_CostlabJSONEncoder)),
        "accuracy": float(R[-1].mean()),
        "per_regime_accuracy": [float(x) for x in R[-1]],
        "forgetting": forgetting,
        "update_cost": update_cost,
        "answer_cost": answer_cost,
        "memory_bytes": arm.memory_bytes(),
        "accounting": accounting_block(arm, bed.d),
        "memory_bytes_by_regime": bytes_by_regime,
        "describe": arm.describe(),
        "device": device,
        "gpu": gpu_info,
        "vram_peak_bytes": vram_peak_bytes,
        "numerics": numerics,
        "warmup_s": warmup_s,
        "energy": energy_info,
        "macs_check": macs_check,
    }


def run_gpu_repeat(
    split: str,
    beds: List[str],
    ns: List[int],
    arms: List[str],
    seeds: List[int],
    out: str | Path,
    tuning_dir: Optional[str] = None,
    budgets_from: Optional[str | Path] = None,
    profile: str = "full",
    cpus: Optional[List[int]] = None,
    threads: int = 2,
    device: str = "cuda",
    recount_macs: bool = False,
) -> None:
    """Execute queue of repeated baseline cells on GPU idempotently.

    On the report split the update multiply-adds are taken from the CPU cell of the same configuration (they do not depend on
    the device and counting them again costs a whole CPU pass per cell); ``recount_macs`` forces the count. Dev runs always count.
    """
    if split == "report":
        require_frozen(seeds)
        if not tuning_dir:
            raise ValueError("Stage B refuses --split report without --tuning: report seeds run tuned operating points only.")

    invalid_seeds = [s for s in seeds if s not in (REPORT_SEEDS if split == "report" else DEV_SEEDS)]
    if split == "report" and invalid_seeds:
        raise ValueError(f"Seeds {invalid_seeds} are outside report seeds ({REPORT_SEEDS[0]}-{REPORT_SEEDS[-1]}).")

    cpus_list = cpus if cpus is not None else [8, 10]
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    budgets_dir = Path(budgets_from) if budgets_from else out_dir

    cells_run = 0
    cells_skipped = 0

    with fixed_threads(n=threads, cpus=cpus_list) as pinning:
        for bed_name in beds:
            tuning_map = load_tuning_map(tuning_dir, bed_name)
            bed_ns = [6] if bed_name == "B2" else ([22] if bed_name.startswith("B3") else ns)
            for n in bed_ns:
                for seed in seeds:
                    bed = build_bed(bed_name, n, seed)
                    pool_bytes_r1 = read_pool_budget(budgets_dir, bed, pool_arm="A1", tuning=tuning_map.get("A1"))
                    pool_macs_r1 = read_pool_macs(budgets_dir, bed, pool_arm="A1", tuning=tuning_map.get("A1"))
                    pool_bytes_r2 = read_pool_budget(budgets_dir, bed, pool_arm="A2", tuning=tuning_map.get("A2"))
                    pool_macs_r2 = read_pool_macs(budgets_dir, bed, pool_arm="A2", tuning=tuning_map.get("A2"))

                    for arm in arms:
                        if arm not in REPEATED_ARMS:
                            continue

                        target_file = out_dir / cell_filename(bed_name, n, arm, seed)
                        arm_tuning = tuning_map.get(arm)
                        if tuning_dir and arm_tuning is None:
                            raise RuntimeError(f"{arm} is not in tuning file of {bed_name}: run m1.tune first.")

                        mem_budget: Optional[int] = None
                        macs_budget: Optional[int] = None
                        if arm in EQUAL_MEMORY_ARMS_R1 or arm in COMPUTE_MATCHED_ARMS_R1:
                            mem_budget = pool_bytes_r1
                        elif arm in EQUAL_MEMORY_ARMS_R2 or arm in COMPUTE_MATCHED_ARMS_R2:
                            mem_budget = pool_bytes_r2

                        if arm in COMPUTE_MATCHED_ARMS_R1:
                            macs_budget = pool_macs_r1
                        elif arm in COMPUTE_MATCHED_ARMS_R2:
                            macs_budget = pool_macs_r2

                        needs_pool = {"A3": "A1", "A4": "A1", "A3-2": "A2", "A4-2": "A2"}.get(arm)
                        if needs_pool is not None and mem_budget is None:
                            missing_name = cell_filename(bed_name, n, needs_pool, seed)
                            raise RuntimeError(
                                f"{arm} on {bed_name} N={n} seed {seed} needs a current {needs_pool} record in {budgets_dir} "
                                f"(missing cell: {missing_name}): run {needs_pool} first."
                            )
                        if arm in COMPUTE_MATCHED_ARMS_R1 + COMPUTE_MATCHED_ARMS_R2 and macs_budget is None:
                            missing_name = cell_filename(bed_name, n, needs_pool, seed)
                            raise RuntimeError(
                                f"{arm} on {bed_name} N={n} seed {seed} needs the counted multiply-adds of a current {needs_pool} "
                                f"full-profile record in {budgets_dir} (missing cell: {missing_name})."
                            )

                        # Idempotent skip check
                        if target_file.exists():
                            try:
                                rec = load_cell_record(target_file)
                            except (OSError, ValueError) as exc:
                                rec = None
                                print(f"  [CORRUPT] {target_file.name} cannot be read as JSON ({exc})")

                            if (
                                rec is not None
                                and is_valid_cell_record(rec)
                                and rec.get("device") == device
                                and record_matches_config(
                                    rec,
                                    bed,
                                    arm,
                                    pool_memory_bytes=mem_budget,
                                    pool_macs_budget=macs_budget,
                                    tuning=arm_tuning,
                                    profile=profile,
                                )
                            ):
                                print(f"  [SKIP] {target_file.name} (complete, valid, current configuration, device={device})")
                                cells_skipped += 1
                                continue

                            print(
                                f"  [STALE] {target_file.name} is unreadable, invalid, or made by another configuration/device: rerunning"
                            )

                        # Load CPU reference cell if present
                        cpu_rec_file = budgets_dir / cell_filename(bed_name, n, arm, seed)
                        cpu_record = None
                        if cpu_rec_file.exists():
                            try:
                                cpu_record = load_cell_record(cpu_rec_file)
                            except Exception as exc:
                                print(f"  [NOTE] Could not load CPU cell {cpu_rec_file.name} for MACs verification ({exc})")

                        reuse_macs = (
                            split == "report"
                            and not recount_macs
                            and cpu_record is not None
                            and record_matches_config(
                                cpu_record,
                                bed,
                                arm,
                                pool_memory_bytes=mem_budget,
                                pool_macs_budget=macs_budget,
                                tuning=arm_tuning,
                            )
                        )
                        print(
                            f"  [RUN GPU] {bed_name} N={n} {arm} seed={seed} device={device} (profile={profile}, "
                            f"update macs {'from the CPU cell' if reuse_macs else 'counted'})...",
                            flush=True,
                        )
                        t0 = time.time()
                        record = execute_gpu_cell(
                            bed=bed,
                            arm_name=arm,
                            pool_memory_bytes=mem_budget,
                            pool_macs_budget=macs_budget,
                            split=split,
                            pinning=pinning,
                            profile=profile,
                            tuning=arm_tuning,
                            device=device,
                            cpu_record=cpu_record,
                            reuse_cpu_update_macs=reuse_macs,
                        )
                        save_cell_record(target_file, record)
                        dt = time.time() - t0
                        print(
                            f"  [DONE] {target_file.name}: acc {record['accuracy'] * 100.0:.2f}% | "
                            f"{record['memory_bytes'] / 1e6:.2f} MB | {dt:.1f}s",
                            flush=True,
                        )
                        cells_run += 1

    print(f"GPU Repeat finished: {cells_run} cells executed, {cells_skipped} cells skipped.")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPU repeat for wide baselines of milestone M1.")
    parser.add_argument("--split", default="dev", help="Split name ('dev' or 'report').")
    parser.add_argument("--beds", default="B1", help="Comma-separated bed names (default: B1).")
    parser.add_argument("--ns", default="10", help="Comma-separated regime counts (default: 10).")
    parser.add_argument("--arms", default=",".join(REPEATED_ARMS), help="Comma-separated arm names.")
    parser.add_argument("--seeds", default="800", help="Comma-separated seeds or ranges (default: 800).")
    parser.add_argument("--out", default=None, help="Output directory (default: results/m1/dev_gpu or scored_gpu).")
    parser.add_argument("--tuning", default="results/m1/tuning", help="Tuning directory.")
    parser.add_argument("--budgets-from", default=None, help="Directory containing CPU records for budgets.")
    parser.add_argument("--profile", default="full", choices=["full", "det", "light"], help="Profile to execute.")
    parser.add_argument("--cpus", default="8,10", help="Logical processors to pin (default: 8,10).")
    parser.add_argument("--threads", type=int, default=2, help="Thread count (default: 2).")
    parser.add_argument("--device", default="cuda", help="Target device (default: cuda).")
    parser.add_argument("--recount-macs", action="store_true", help="Count the update multiply-adds again on the report split instead of taking them from the CPU cell.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    beds = [b.strip() for b in args.beds.split(",") if b.strip()]
    ns = [int(n.strip()) for n in args.ns.split(",") if n.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    seeds = parse_seeds(args.seeds)
    cpus = [int(c.strip()) for c in args.cpus.split(",")] if args.cpus else [8, 10]
    out_dir = args.out or ("results/m1/scored_gpu" if args.split == "report" else "results/m1/dev_gpu")
    budgets_from = args.budgets_from or ("results/m1/scored" if args.split == "report" else "results/m1/dev")

    run_gpu_repeat(
        split=args.split,
        beds=beds,
        ns=ns,
        arms=arms,
        seeds=seeds,
        out=out_dir,
        tuning_dir=args.tuning,
        budgets_from=budgets_from,
        profile=args.profile,
        cpus=cpus,
        threads=args.threads,
        device=args.device,
        recount_macs=args.recount_macs,
    )


if __name__ == "__main__":
    main()
