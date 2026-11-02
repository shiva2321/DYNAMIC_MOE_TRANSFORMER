"""Cell execution, costlab profiling, schema validation, and atomic recording."""

from __future__ import annotations

import hashlib
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
from growmoe.accounting import measured_gate_macs_per_input, pool_report
from growmoe.metrics import accuracy_stats
from m1.arms import build_arm, pool_budget_bytes, pool_budget_macs
from m1.beds import Bed

SCHEMA_VERSION = "1.2.0"
_REQUIRED_ROOT = (
    "schema_version",
    "bed",
    "n_regimes",
    "arm",
    "seed",
    "split",
    "commit",
    "environment",
    "pinning",
    "profile",
    "bed_info",
    "tuning_ref",
    "reference_pool",
    "uncounted_note",
    "config",
    "accuracy",
    "per_regime_accuracy",
    "forgetting",
    "update_cost",
    "memory_bytes",
    "memory_bytes_by_regime",
)
_REQUIRED_UPDATE_FULL = (
    "wall_clock_per_observe",
    "wall_clock_per_regime",
    "wall_clock_total",
    "counted_macs",
    "peak_memory",
    "eval_wall_clock_per_regime_boundary",
)
_REQUIRED_ANSWER_DET = ("batch_size", "rows_timed", "counted_macs", "active_bytes")      # profile det: no timing fields at all
_REQUIRED_ANSWER = (
    "batch_size",
    "rows_timed",
    "latency_median_s",
    "latency_mean_s",
    "latency_std_s",
    "latency_cv",
    "latency_p95_s",
    "items_per_second",
    "counted_macs",
    "active_bytes",
)


def cell_filename(bed: str, n_regimes: int, arm: str, seed: int) -> str:
    """Canonical filename for a benchmark cell record."""
    return f"{bed}_n{n_regimes}_{arm}__seed{seed}.json"


def compute_tuning_ref(tuning: Optional[Dict[str, Any]]) -> Optional[str]:
    """Compute deterministic SHA256 hex digest of a tuning configuration dict."""
    if not tuning:
        return None
    canonical = json.dumps(tuning, sort_keys=True, cls=_CostlabJSONEncoder)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_valid_cell_record(rec: Dict[str, Any]) -> bool:
    """Validate that a cell record dictionary contains all required fields for its profile and schema 1.2.0."""
    if rec.get("schema_version") != SCHEMA_VERSION:
        return False
    if not all(k in rec for k in _REQUIRED_ROOT):
        return False

    profile = rec.get("profile")
    if profile not in ("full", "det", "light"):
        return False

    update_cost = rec.get("update_cost", {})
    if profile == "det":
        ans = rec.get("answer_cost", {})
        return "counted_macs" in update_cost and all(b in ans and all(k in ans[b] for k in _REQUIRED_ANSWER_DET) for b in ("1", "32"))
    if profile == "full":
        if not all(k in update_cost for k in _REQUIRED_UPDATE_FULL):
            return False
        ans = rec.get("answer_cost", {})
        if not all(b in ans and all(k in ans[b] for k in _REQUIRED_ANSWER) for b in ("1", "32")):
            return False
    else:
        if "wall_clock_total" not in update_cost:
            return False

    return True


def record_matches_config(
    rec: Dict[str, Any],
    bed: Bed,
    arm_name: str,
    pool_memory_bytes: Optional[int] = None,
    pool_macs_budget: Optional[int] = None,
    tuning: Optional[Dict[str, Any]] = None,
    profile: Optional[str] = None,
) -> bool:
    """True when the record is structurally valid and was produced by the arm as configured now."""
    if not is_valid_cell_record(rec):
        return False

    if profile is not None and rec.get("profile") != profile:
        return False

    if tuning is not None:
        expected_ref = compute_tuning_ref(tuning)
        if rec.get("tuning_ref") != expected_ref:
            return False

    if arm_name in ("A3", "A7", "A3-2", "A7-2") and pool_memory_bytes is None:
        return False

    if arm_name in ("A4", "A4-2") and (pool_memory_bytes is None or pool_macs_budget is None):
        return False

    fresh = build_arm(
        arm_name,
        bed,
        pool_memory_bytes=pool_memory_bytes,
        pool_macs_budget=pool_macs_budget,
        tuning=tuning,
    ).config()

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


def read_pool_budget(out_dir: Path | str, bed: Bed, pool_arm: str = "A1", tuning: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Final memory of the reference pool of the same bed and seed, read from its record in out_dir."""
    path = Path(out_dir) / cell_filename(bed.name, bed.n_regimes, pool_arm, bed.seed)
    if not path.exists():
        return None
    try:
        rec = load_cell_record(path)
    except (OSError, ValueError):
        return None
    if not (is_valid_cell_record(rec) and record_matches_config(rec, bed, pool_arm, tuning=tuning)):
        return None
    return int(rec["memory_bytes"])


def read_pool_macs(out_dir: Path | str, bed: Bed, pool_arm: str = "A1", tuning: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Counted training multiply-adds of reference pool of same bed and seed from record in out_dir."""
    path = Path(out_dir) / cell_filename(bed.name, bed.n_regimes, pool_arm, bed.seed)
    if not path.exists():
        return None
    try:
        rec = load_cell_record(path)
    except (OSError, ValueError):
        return None
    if not (is_valid_cell_record(rec) and record_matches_config(rec, bed, pool_arm, tuning=tuning)):
        return None
    macs = rec.get("update_cost", {}).get("counted_macs")
    return int(macs) if macs is not None else None


def accounting_block(arm: Any, d: int) -> Dict[str, Any]:
    """What growmoe.accounting says about the trained arm, kept in the record so that the report can set it against the counted multiply-adds.

    ``predict_macs`` is the static count (every stored signature scanned once plus the weight matrices of one expert); ``gate_macs_measured_predict`` is what the pool's own
    gate counters spent per input when it answers (smaller when the shortlist is on); it is absent for arms without gate counters.
    """
    rep = pool_report(arm.model, d)
    block: Dict[str, Any] = {
        "n_experts": int(rep.n_experts),
        "params": int(rep.params),
        "stored_vectors": int(rep.stored_vectors),
        "memory_bytes": int(rep.memory_bytes),
        "gate_macs": int(rep.gate_macs),
        "expert_macs": int(rep.expert_macs),
        "predict_macs": int(rep.predict_macs),
    }
    stats = getattr(arm.model, "gate_stats", None)
    if isinstance(stats, dict) and stats.get("predict", {}).get("samples"):
        block["gate_macs_measured_predict"] = float(measured_gate_macs_per_input(arm.model, "predict"))
    return block


def execute_cell(
    bed: Bed,
    arm_name: str,
    pool_memory_bytes: Optional[int] = None,
    pool_macs_budget: Optional[int] = None,
    split: str = "dev",
    pinning: Optional[Dict[str, Any]] = None,
    profile: str = "full",
    tuning: Optional[Dict[str, Any]] = None,
    **arm_kwargs: Any,
) -> Dict[str, Any]:
    """Execute one benchmark cell (bed x arm x seed) and gather all costlab metrics."""
    if split == "report":
        raise ValueError("Stage B refuses --split report; report seeds are reserved.")

    if profile not in ("full", "det", "light"):
        raise ValueError(f"Unknown profile {profile!r}; must be 'full' (everything), 'det' (deterministic metrics only, no timing) or 'light' (tuning).")

    env_info = environment()
    commit = env_info.get("git_commit") or "unknown"

    # Budgets come from the reference pool's own record. Only A3 and A7 (reference A1) may fall back to the untuned A1 of stage A;
    # A4, A3-2, A4-2 and A7-2 raise in build_arm when their budget is missing.
    if arm_name in ("A3", "A7") and pool_memory_bytes is None:
        pool_memory_bytes = pool_budget_bytes(bed)

    arm = build_arm(
        arm_name,
        bed,
        pool_memory_bytes=pool_memory_bytes,
        pool_macs_budget=pool_macs_budget,
        tuning=tuning,
        **arm_kwargs,
    )

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
        row = [
            float((arm.predict(x).probs.argmax(dim=-1).cpu() == y.cpu()).float().mean().item())
            for x, y in (tests[r] for r in range(bed.n_regimes))
        ]
        eval_durations.append(time.perf_counter() - t0)
        return row

    def _train_stream() -> None:
        nonlocal wall_clock_total
        cur_phase = -1
        last_regime = -1
        for x_b, y_b, regimes, first_regimes, phase_idx in bed.walk_full(64):
            regime_id = regimes[0]                                  # the time of a batch of a mixed phase is booked on its first regime
            if phase_idx != cur_phase:
                if cur_phase >= 0:
                    if profile in ("full", "det"):              # the light profile (tuning) needs only the final accuracy: no evaluation, and so no router refit, at the phase boundaries
                        R_rows.append(_eval_row())
                    bytes_by_regime.setdefault(str(last_regime), arm.memory_bytes())      # the first visit only: a later visit of the same regime must not overwrite it (M1-8 reads this)
                cur_phase = phase_idx
                last_regime = regime_id
            for r_first in first_regimes:                           # every regime first seen in this batch is learned at the row that follows this phase
                if r_first not in learned_at:
                    learned_at[r_first] = len(R_rows)

            t0 = time.perf_counter()
            arm.observe(x_b, y_b)
            dt = time.perf_counter() - t0
            obs_durations.append(dt)
            r_str = str(regime_id)
            reg_durations[r_str] = reg_durations.get(r_str, 0.0) + dt
            wall_clock_total += dt

        R_rows.append(_eval_row())
        if last_regime >= 0:
            bytes_by_regime.setdefault(str(last_regime), arm.memory_bytes())

    if profile == "full":
        peak_mem_profile = peak_memory(_train_stream, device="cpu")
    else:
        _train_stream()
        peak_mem_profile = None

    R = np.array(R_rows, dtype=np.float64)
    if len(R_rows) > 1:
        never = [r for r in range(bed.n_regimes) if r not in learned_at]
        if never:
            raise ValueError(f"regimes {never} have no first-visit sample in the stream of {bed.name}: the row at which they were learned is unknown")
        la = [learned_at[r] for r in range(bed.n_regimes)]
        stats = accuracy_stats(R, learned_at=la, chance=1.0 / bed.n_classes)
        forgetting = None if np.isnan(stats["forgetting"]) else float(stats["forgetting"])
    else:
        forgetting = None                                       # one row only (light profile): forgetting is not defined

    if profile in ("full", "det"):
        if arm_name in ("A7", "A7all", "A7-2"):
            update_macs = 0                      # a lookup stores vectors at update time: nothing is multiplied
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

        test_x = bed.test_sets[0][0]
        answer_cost: Dict[str, Any] = {}
        for b in (1, 32):
            x_b = test_x[torch.arange(b) % len(test_x)]
            fn = lambda: arm.predict(x_b)
            mod = arm.get_torch_modules()
            act_b = active_param_bytes(mod, fn).total if mod is not None else 0
            entry: Dict[str, Any] = {"batch_size": b, "rows_timed": int(len(x_b)), "counted_macs": count_macs(fn).macs, "active_bytes": act_b}
            if profile == "full":                    # timing only in the isolated pass: other work on the machine would contaminate it
                t_res = time_callable(fn, warmup=5, repeats=10, device="cpu")
                entry.update(
                    latency_median_s=t_res.median,
                    latency_mean_s=t_res.mean,
                    latency_std_s=t_res.std,
                    latency_cv=t_res.cv,
                    latency_p95_s=t_res.p95,
                    items_per_second=float(b / t_res.median) if t_res.median > 0 else 0.0,
                )
            answer_cost[str(b)] = entry
    else:
        update_macs = None
        answer_cost = {}

    # Metadata & Schema 1.2.0 additions
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
    pool_time_split = getattr(arm, "time_split", None) if profile != "det" else None
    update_cost = {"counted_macs": update_macs} if profile == "det" else {
        "wall_clock_per_observe": obs_durations,
        "wall_clock_per_regime": reg_durations,
        "wall_clock_total": wall_clock_total,
        "counted_macs": update_macs,
        "peak_memory": peak_mem_profile,
        "eval_wall_clock_per_regime_boundary": eval_durations,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "bed": bed.name,
        "n_regimes": bed.n_regimes,
        "arm": arm_name,
        "seed": bed.seed,
        "split": split,
        "commit": commit,
        "environment": env_info,
        "pinning": pinning,
        "profile": profile,
        "bed_info": bed_info,
        "tuning_ref": tuning_ref,
        "reference_pool": reference_pool,
        "uncounted_note": getattr(arm, "uncounted_note", False),
        "pool_time_split": pool_time_split,
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
    }
