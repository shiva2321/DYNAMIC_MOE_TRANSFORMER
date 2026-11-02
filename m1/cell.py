"""Cell execution, costlab profiling, schema validation, and atomic recording."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from costlab.bytes import active_param_bytes
from costlab.count import count_macs
from costlab.env import environment
from costlab.memory import peak_memory
from costlab.report import _CostlabJSONEncoder
from costlab.timing import time_callable
from growmoe.metrics import accuracy_stats
from m1.arms import build_arm, pool_budget_bytes
from m1.beds import Bed

SCHEMA_VERSION = "1.1.0"
_REQUIRED_ROOT = (
    "schema_version", "bed", "n_regimes", "arm", "seed", "split", "commit",
    "environment", "pinning", "config", "accuracy", "per_regime_accuracy", "forgetting", "update_cost",
    "answer_cost", "memory_bytes", "memory_bytes_by_regime",
)
_REQUIRED_UPDATE = (
    "wall_clock_per_observe", "wall_clock_per_regime", "wall_clock_total",
    "counted_macs", "peak_memory", "eval_wall_clock_per_regime_boundary",
)
_REQUIRED_ANSWER = (
    "batch_size", "rows_timed", "latency_median_s", "latency_mean_s", "latency_std_s",
    "latency_cv", "latency_p95_s", "items_per_second", "counted_macs", "active_bytes",
)


def cell_filename(bed: str, n_regimes: int, arm: str, seed: int) -> str:
    """Canonical filename for a benchmark cell record."""
    return f"{bed}_n{n_regimes}_{arm}__seed{seed}.json"


def is_valid_cell_record(rec: Dict[str, Any]) -> bool:
    """Validate that a cell record dictionary contains all required fields (structure only; see record_matches_config for the content)."""
    if not all(k in rec for k in _REQUIRED_ROOT):
        return False
    if not all(k in rec["update_cost"] for k in _REQUIRED_UPDATE):
        return False
    ans = rec["answer_cost"]
    return all(b in ans and all(k in ans[b] for k in _REQUIRED_ANSWER) for b in ("1", "32"))


def record_matches_config(rec: Dict[str, Any], bed: Bed, arm_name: str, pool_memory_bytes: Optional[int] = None) -> bool:
    """True when the record was produced by the arm as it is configured now.

    A structurally valid record can still be stale (the arm was changed after it was written); the queue must rerun such a cell. For the equal-memory arms
    (A3, A7) the budget is part of the configuration, so a budget that is not known here counts as a mismatch.
    """
    if arm_name in ("A3", "A7") and pool_memory_bytes is None:
        return False
    fresh = build_arm(arm_name, bed, pool_memory_bytes=pool_memory_bytes).config()
    return json.loads(json.dumps(fresh, cls=_CostlabJSONEncoder)) == rec.get("config")


def save_cell_record(path: Path | str, record: Dict[str, Any]) -> None:
    """Save cell record to disk atomically using a temp file and os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.{time.time_ns()}")
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, cls=_CostlabJSONEncoder)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_cell_record(path: Path | str) -> Dict[str, Any]:
    """Load cell record from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_pool_budget(out_dir: Path | str, bed: Bed) -> Optional[int]:
    """Final memory of the A1 pool of the same bed and seed, read from its record in out_dir.

    None when there is no usable record: missing, unreadable, structurally invalid, or stale (A1 was reconfigured since it was written).
    """
    path = Path(out_dir) / cell_filename(bed.name, bed.n_regimes, "A1", bed.seed)
    if not path.exists():
        return None
    try:
        rec = load_cell_record(path)
    except (OSError, ValueError):
        return None
    if not (is_valid_cell_record(rec) and record_matches_config(rec, bed, "A1")):
        return None
    return int(rec["memory_bytes"])


def execute_cell(
    bed: Bed,
    arm_name: str,
    pool_memory_bytes: Optional[int] = None,
    split: str = "dev",
    pinning: Optional[Dict[str, Any]] = None,
    **arm_kwargs: Any,
) -> Dict[str, Any]:
    """Execute one benchmark cell (bed x arm x seed) and gather all costlab metrics.

    ``pinning`` is the settings dict yielded by costlab.env.fixed_threads (threads, cpus, blas_limited); it is stored in the record (Q-29).
    """
    if split == "report":
        raise ValueError("Stage A refuses --split report; report seeds are reserved.")

    env_info = environment()
    commit = env_info.get("git_commit") or "unknown"
    if arm_name in ("A3", "A7") and pool_memory_bytes is None:
        pool_memory_bytes = pool_budget_bytes(bed)         # one budget for the measured arm and for the counting pass
    arm = build_arm(arm_name, bed, pool_memory_bytes=pool_memory_bytes, **arm_kwargs)

    obs_durations: List[float] = []
    eval_durations: List[float] = []
    reg_durations: Dict[str, float] = {}
    bytes_by_regime: Dict[str, int] = {}
    R_rows: List[List[float]] = []
    learned_at: Dict[int, int] = {}
    wall_clock_total = 0.0

    tests = bed.test_sets

    def _eval_row() -> List[float]:
        t0 = time.perf_counter()
        row = [float((arm.predict(x).probs.argmax(dim=-1).cpu() == y.cpu()).float().mean().item())
               for x, y in (tests[r] for r in range(bed.n_regimes))]
        eval_durations.append(time.perf_counter() - t0)      # includes any model refit that predict triggers (the router refits inside predict)
        return row

    def _train_stream() -> None:
        nonlocal wall_clock_total
        cur_reg = -1
        for _, seg, idx in bed.stream._walk(64, bed.seed):
            if seg.regime != cur_reg:
                if cur_reg >= 0:
                    R_rows.append(_eval_row())
                    bytes_by_regime[str(cur_reg)] = arm.memory_bytes()
                cur_reg = seg.regime
                if seg.first_visit and seg.regime not in learned_at:
                    learned_at[seg.regime] = len(R_rows)

            t0 = time.perf_counter()
            arm.observe(seg.x[idx], seg.y[idx])
            dt = time.perf_counter() - t0
            obs_durations.append(dt)
            r_str = str(seg.regime)
            reg_durations[r_str] = reg_durations.get(r_str, 0.0) + dt
            wall_clock_total += dt

        R_rows.append(_eval_row())
        if cur_reg >= 0:
            bytes_by_regime[str(cur_reg)] = arm.memory_bytes()

    peak_mem_profile = peak_memory(_train_stream, device="cpu")

    R = np.array(R_rows, dtype=np.float64)
    la = [learned_at.get(r, r) for r in range(bed.n_regimes)]
    stats = accuracy_stats(R, learned_at=la, chance=1.0 / bed.n_classes)
    forgetting = None if np.isnan(stats["forgetting"]) else float(stats["forgetting"])    # undefined stays undefined: no invented zero

    if arm_name == "A7":
        update_macs = 0
    else:
        def _mac_pass():
            sh = build_arm(arm_name, bed, pool_memory_bytes=pool_memory_bytes, **arm_kwargs)
            for x, y, _ in bed:
                sh.observe(x, y)
        update_macs = count_macs(_mac_pass).macs

    test_x = bed.test_sets[0][0]
    answer_cost: Dict[str, Any] = {}
    for b in (1, 32):
        x_b = test_x[torch.arange(b) % len(test_x)]            # exactly b rows, repeating the test set when it is shorter
        fn = lambda: arm.predict(x_b)
        t_res = time_callable(fn, warmup=5, repeats=10, device="cpu")
        mod = arm.get_torch_modules()
        act_b = active_param_bytes(mod, fn).total if mod is not None else 0
        answer_cost[str(b)] = {
            "batch_size": b, "rows_timed": int(len(x_b)), "latency_median_s": t_res.median, "latency_mean_s": t_res.mean,
            "latency_std_s": t_res.std, "latency_cv": t_res.cv, "latency_p95_s": t_res.p95,
            "items_per_second": float(b / t_res.median) if t_res.median > 0 else 0.0,
            "counted_macs": count_macs(fn).macs, "active_bytes": act_b,
        }

    return {
        "schema_version": SCHEMA_VERSION, "bed": bed.name, "n_regimes": bed.n_regimes,
        "arm": arm_name, "seed": bed.seed, "split": split, "commit": commit, "environment": env_info,
        "pinning": pinning, "config": json.loads(json.dumps(arm.config(), cls=_CostlabJSONEncoder)),
        "accuracy": float(R[-1].mean()), "per_regime_accuracy": [float(x) for x in R[-1]], "forgetting": forgetting,
        "update_cost": {
            "wall_clock_per_observe": obs_durations, "wall_clock_per_regime": reg_durations,
            "wall_clock_total": wall_clock_total, "counted_macs": update_macs, "peak_memory": peak_mem_profile,
            "eval_wall_clock_per_regime_boundary": eval_durations,
        },
        "answer_cost": answer_cost, "memory_bytes": arm.memory_bytes(),
        "memory_bytes_by_regime": bytes_by_regime, "describe": arm.describe(),
    }
