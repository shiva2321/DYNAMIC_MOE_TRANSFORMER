"""Tuning driver for M1 benchmark: tunes hyperparameter grids on development seeds.

Implements M1 Stage B Section 4.3 & 5.7:
- Evaluates arms on development seeds (800-804) using light profile.
- Selection rule: highest mean final accuracy over seeds; ties (< 0.001) broken
  by lower mean memory, then lower mean update wall-clock.
- Edge extension: if best value is at grid edge (steps, width, D), extends by factor
  (steps x 2.5 up to 200, width/D x 2 up to 8192) while new end wins.
- Idempotent: per-trial files saved under results/m1/tuning/trials/<bed>/.
- Enforces dependencies: A1, A1q, A2, then baselines; -2 baselines after A2.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from costlab.count import count_macs
from costlab.env import environment, fixed_threads
from costlab.report import _CostlabJSONEncoder
from m1.arms import build_arm, compute_a4_steps, pool_budget_bytes
from m1.beds import Bed, build_bed
from m1.cell import (
    compute_tuning_ref,
    execute_cell,
    load_cell_record,
    save_cell_record,
)
from m1.config_m1 import (
    A2_CANDIDATES,
    B3_UNIPOOL_CANDIDATES,
    DEV_SEEDS,
    EDGE_EXTEND_CAP_D,
    EDGE_EXTEND_CAP_STEPS,
    EDGE_EXTEND_CAP_WIDTH,
    EDGE_EXTEND_FACTOR_D,
    EDGE_EXTEND_FACTOR_STEPS,
    EDGE_EXTEND_FACTOR_WIDTH,
    GRID_A1_STEPS,
    GRID_A1Q_Q,
    GRID_A1Q_STEPS,
    GRID_A2_Q,
    GRID_A3_STEPS,
    GRID_A3_WIDTH,
    GRID_A4_WIDTH,
    GRID_A5A_SHRINKAGE,
    GRID_A5B_SHRINKAGE,
    GRID_A6_D,
    GRID_A6_RIDGE,
    GRID_A7_K,
    GRID_A7ALL_K,
    GRID_A8_STEPS,
    GRID_A8_WIDTH,
    PINNING_CPUS,
    PINNING_THREADS,
    SMOKE_A2_CANDIDATES,
    SMOKE_GRID_A1_STEPS,
    SMOKE_GRID_A1Q_Q,
    SMOKE_GRID_A1Q_STEPS,
    SMOKE_GRID_A2_Q,
    SMOKE_GRID_A3_STEPS,
    SMOKE_GRID_A3_WIDTH,
    SMOKE_GRID_A4_WIDTH,
    SMOKE_GRID_A5A_SHRINKAGE,
    SMOKE_GRID_A5B_SHRINKAGE,
    SMOKE_GRID_A6_D,
    SMOKE_GRID_A6_RIDGE,
    SMOKE_GRID_A7_K,
    SMOKE_GRID_A7ALL_K,
    SMOKE_GRID_A8_STEPS,
    SMOKE_GRID_A8_WIDTH,
    TOLERANCE_TIE_ACCURACY,
)

ALL_ARM_ORDER = (
    "A1",
    "A1q",
    "A2",
    "A3",
    "A4",
    "A5a",
    "A5b",
    "A6",
    "A7",
    "A7all",
    "A8",
    "A3-2",
    "A4-2",
    "A7-2",
)


def select_best_candidate(candidate_summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Select best candidate by M1 selection rule:

    Highest mean accuracy over seeds; ties (difference < 0.001) go to lower
    mean memory bytes, then lower mean update wall-clock time.
    """
    if not candidate_summaries:
        raise ValueError("Cannot select from empty candidate summaries.")

    max_acc = max(s["mean_accuracy"] for s in candidate_summaries)
    # Filter candidates within tie margin
    ties = [s for s in candidate_summaries if (max_acc - s["mean_accuracy"]) <= TOLERANCE_TIE_ACCURACY]

    # Break ties by lower mean memory, then lower mean update wall-clock
    ties.sort(key=lambda s: (s["mean_memory_bytes"], s["mean_wall_clock"]))
    return ties[0]


def is_candidate_better(new_cand: Dict[str, Any], old_cand: Dict[str, Any]) -> bool:
    """Return True if new_cand beats old_cand under the M1 selection rule."""
    diff = new_cand["mean_accuracy"] - old_cand["mean_accuracy"]
    if abs(diff) <= TOLERANCE_TIE_ACCURACY:
        if new_cand["mean_memory_bytes"] != old_cand["mean_memory_bytes"]:
            return new_cand["mean_memory_bytes"] < old_cand["mean_memory_bytes"]
        return new_cand["mean_wall_clock"] < old_cand["mean_wall_clock"]
    return diff > 0.0


def _run_plain(fn) -> None:
    """Run ``fn`` without counting (the memory-only reference pass); returns None so that the record stores no multiply-adds."""
    fn()
    return None


class TuningDriver:
    """Manages tuning runs, caching, and trial execution."""

    def __init__(
        self,
        bed_name: str,
        n_regimes: int,
        out_dir: Path | str,
        smoke: bool = False,
        seeds: Optional[List[int]] = None,
    ) -> None:
        self.bed_name = bed_name
        self.tune_bed_name = "B3seq" if bed_name in ("B3mix", "B3drift") else bed_name
        if self.bed_name == "B2":
            self.n_regimes = 6
        elif self.bed_name.startswith("B3"):
            self.n_regimes = 22
        else:
            self.n_regimes = n_regimes
        self.out_dir = Path(out_dir)
        self.smoke = smoke
        self.seeds = seeds if seeds is not None else list(DEV_SEEDS)

        self.trials_dir = self.out_dir / "trials" / bed_name
        self.macs_dir = self.out_dir / "macs"
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        self.macs_dir.mkdir(parents=True, exist_ok=True)

        self.tuning_file = self.out_dir / f"{bed_name}.json"
        self.state: Dict[str, Any] = self._load_tuning_state()

    def _load_tuning_state(self) -> Dict[str, Any]:
        if self.tuning_file.exists():
            try:
                return load_cell_record(self.tuning_file)
            except (OSError, ValueError) as exc:
                raise ValueError(f"The tuning file {self.tuning_file} cannot be read ({exc}); move it away to start the tuning again (finished trials are kept).") from exc
        return {
            "bed": self.bed_name,
            "tune_n": self.n_regimes,
            "selection_rule": (
                "highest mean final accuracy over dev seeds; ties (<0.001) broken "
                "by lower memory then lower update wall-clock"
            ),
            "code_stamp": environment().get("git_commit") or "unknown",
            "arms": {},
        }

    def save_tuning_state(self) -> None:
        save_cell_record(self.tuning_file, self.state)
        if self.bed_name == "B3seq":
            for other_bed in ("B3mix", "B3drift"):
                other_file = self.out_dir / f"{other_bed}.json"
                other_state = dict(self.state, bed=other_bed)
                save_cell_record(other_file, other_state)

    def pool_reference(self, pool_arm: str, seed: int, need_macs: bool = True) -> Dict[str, int]:
        """Final memory and (if ``need_macs``) whole-stream counted training multiply-adds of the tuned reference pool (A1 or A2) on this bed and seed.

        The memory comes from a plain pass; the multiply-adds need a counted pass, about five times as slow (and a pool that fragments is slow), so they are counted only for A4.
        The result is cached in ``macs/`` and is valid only for the tuning entry it was made with.
        """
        chosen = self.state["arms"].get(pool_arm, {}).get("chosen")
        if chosen is None:
            raise RuntimeError(f"{pool_arm} must be tuned before the arms sized from it: tune {pool_arm} first.")
        ref = compute_tuning_ref(chosen)
        macs_file = self.macs_dir / f"{self.bed_name}_n{self.n_regimes}_{pool_arm}__seed{seed}.json"
        cached: Dict[str, Any] = {}
        if macs_file.exists():
            try:
                rec = load_cell_record(macs_file)
                if rec.get("tuning_ref") == ref and "bytes" in rec:
                    cached = rec
                else:
                    print(f"  [STALE] {macs_file.name} was made for another tuning entry: recomputing", flush=True)
            except (OSError, ValueError):
                print(f"  [CORRUPT] {macs_file.name} cannot be read: recomputing", flush=True)
        if cached and (not need_macs or cached.get("macs") is not None):
            return {"bytes": int(cached["bytes"]), "macs": None if cached.get("macs") is None else int(cached["macs"])}

        bed = build_bed(self.tune_bed_name, self.n_regimes, seed)
        held: Dict[str, int] = {}

        def _run() -> None:
            pool = build_arm(pool_arm, bed, tuning=chosen)
            for x, y, _ in bed:
                pool.observe(x, y)
            held["bytes"] = pool.memory_bytes()

        total_macs = count_macs(_run).macs if need_macs else _run_plain(_run)
        record = {"bed": self.bed_name, "n": self.n_regimes, "arm": pool_arm, "seed": seed, "macs": total_macs, "bytes": held["bytes"], "tuning_ref": ref}
        save_cell_record(macs_file, record)
        return {"bytes": held["bytes"], "macs": total_macs}

    def get_pool_budget_bytes(self, pool_arm: str, seed: int) -> int:
        """Final memory of the tuned reference pool (cached; a plain pass)."""
        return self.pool_reference(pool_arm, seed, need_macs=False)["bytes"]

    def get_pool_budget_macs(self, pool_arm: str, seed: int) -> int:
        """Whole-stream counted training multiply-adds of the tuned reference pool (cached; a counted pass)."""
        return self.pool_reference(pool_arm, seed, need_macs=True)["macs"]

    def run_single_trial(
        self,
        arm_id: str,
        config: Dict[str, Any],
        seed: int,
        pool_memory_bytes: Optional[int] = None,
        pool_macs_budget: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run or load a single trial cell."""
        cfg_str = json.dumps(config, sort_keys=True, cls=_CostlabJSONEncoder)
        cfg_hash = hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()[:12]
        trial_file = self.trials_dir / f"trial__{arm_id}__{cfg_hash}__seed{seed}.json"

        if trial_file.exists():
            try:
                rec = load_cell_record(trial_file)
                if "accuracy" in rec and "memory_bytes" in rec:
                    print(
                        f"  [SKIP TRIAL] {arm_id} config={config} seed={seed}: "
                        f"acc={rec['accuracy']*100:.2f}% mem={rec['memory_bytes']/1e6:.2f}MB "
                        f"time={rec['update_cost']['wall_clock_total']:.2f}s",
                        flush=True,
                    )
                    return rec
            except (OSError, ValueError, KeyError):
                print(f"  [CORRUPT] {trial_file.name} is unreadable or incomplete: rerunning the trial", flush=True)

        bed = build_bed(self.tune_bed_name, self.n_regimes, seed)
        rec = execute_cell(
            bed=bed,
            arm_name=arm_id,
            pool_memory_bytes=pool_memory_bytes,
            pool_macs_budget=pool_macs_budget,
            split="dev",
            profile="light",
            tuning=config,
        )
        save_cell_record(trial_file, rec)
        print(
            f"  [TRIAL] {arm_id} config={config} seed={seed}: "
            f"acc={rec['accuracy']*100:.2f}% mem={rec['memory_bytes']/1e6:.2f}MB "
            f"time={rec['update_cost']['wall_clock_total']:.2f}s",
            flush=True,
        )
        return rec

    def evaluate_candidate(
        self,
        arm_id: str,
        config: Dict[str, Any],
        ref_pool: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a candidate configuration across all seeds."""
        per_seed_acc: Dict[int, float] = {}
        per_seed_mem: Dict[int, int] = {}
        per_seed_time: Dict[int, float] = {}

        for seed in self.seeds:
            pool_mem = self.get_pool_budget_bytes(ref_pool, seed) if ref_pool else None
            pool_macs = self.get_pool_budget_macs(ref_pool, seed) if ref_pool and "A4" in arm_id else None

            # For A4, compute steps per seed
            cfg_to_run = dict(config)
            if arm_id in ("A4", "A4-2"):
                bed = build_bed(self.tune_bed_name, self.n_regimes, seed)
                s_w = compute_a4_steps(bed, pool_macs, pool_mem, int(config["hidden"]))
                if s_w < 1:
                    return None  # Width exceeds budget on this seed
                cfg_to_run["inner_steps"] = s_w

            rec = self.run_single_trial(arm_id, cfg_to_run, seed, pool_memory_bytes=pool_mem, pool_macs_budget=pool_macs)
            per_seed_acc[seed] = rec["accuracy"]
            per_seed_mem[seed] = rec["memory_bytes"]
            per_seed_time[seed] = rec["update_cost"]["wall_clock_total"]

        mean_acc = float(sum(per_seed_acc.values()) / len(per_seed_acc))
        mean_mem = float(sum(per_seed_mem.values()) / len(per_seed_mem))
        mean_time = float(sum(per_seed_time.values()) / len(per_seed_time))

        return {
            "config": config,
            "mean_accuracy": mean_acc,
            "mean_memory_bytes": mean_mem,
            "mean_wall_clock": mean_time,
            "per_seed_accuracy": per_seed_acc,
            "per_seed_memory": per_seed_mem,
            "per_seed_wall_clock": per_seed_time,
        }

    def _extend_edge(
        self,
        arm_id: str,
        best_cand: Dict[str, Any],
        param_name: str,
        factor: float,
        cap: int,
        ref_pool: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], bool]:
        """Extend parameter at edge by factor up to cap while new end wins."""
        curr_best = best_cand
        val = int(curr_best["config"][param_name])
        at_edge = False

        while True:
            next_val = min(int(round(val * factor)), cap)
            if next_val == val:
                at_edge = True
                break

            extended_cfg = dict(curr_best["config"], **{param_name: next_val})
            print(f"  [EXTEND] {arm_id} testing edge extension {param_name}={next_val} (cap={cap})...", flush=True)
            res = self.evaluate_candidate(arm_id, extended_cfg, ref_pool=ref_pool)

            if res is not None and is_candidate_better(res, curr_best):
                curr_best = res
                val = next_val
                if val >= cap:
                    at_edge = True
                    break
            else:
                break

        return curr_best, at_edge

    def grid_ref(self, arm_id: str) -> str:
        """Digest of everything that decides the result of tuning ``arm_id``: grids, candidates, seeds, bed size, smoke flag and the tuning entry of its reference pool."""
        s = self.smoke
        grids = {
            "A1": [SMOKE_GRID_A1_STEPS if s else GRID_A1_STEPS],
            "A1q": [SMOKE_GRID_A1Q_STEPS if s else GRID_A1Q_STEPS, SMOKE_GRID_A1Q_Q if s else GRID_A1Q_Q],
            "A2": [
                SMOKE_A2_CANDIDATES if s else (A2_CANDIDATES, list(B3_UNIPOOL_CANDIDATES) if self.bed_name.startswith("B3") else []),
                SMOKE_GRID_A2_Q if s else GRID_A2_Q,
                self.state["arms"].get("A1q", {}).get("tuning_ref"),
            ],
            "A3": [SMOKE_GRID_A3_WIDTH if s else GRID_A3_WIDTH, SMOKE_GRID_A3_STEPS if s else GRID_A3_STEPS],
            "A4": [SMOKE_GRID_A4_WIDTH if s else GRID_A4_WIDTH],
            "A5a": [SMOKE_GRID_A5A_SHRINKAGE if s else GRID_A5A_SHRINKAGE],
            "A5b": [SMOKE_GRID_A5B_SHRINKAGE if s else GRID_A5B_SHRINKAGE],
            "A6": [SMOKE_GRID_A6_D if s else GRID_A6_D, SMOKE_GRID_A6_RIDGE if s else GRID_A6_RIDGE],
            "A7": [SMOKE_GRID_A7_K if s else GRID_A7_K],
            "A7all": [SMOKE_GRID_A7ALL_K if s else GRID_A7ALL_K],
            "A8": [SMOKE_GRID_A8_WIDTH if s else GRID_A8_WIDTH, SMOKE_GRID_A8_STEPS if s else GRID_A8_STEPS],
        }
        base = arm_id[:-2] if arm_id.endswith("-2") else arm_id
        ref_pool = "A2" if arm_id.endswith("-2") else {"A3": "A1", "A4": "A1", "A7": "A1"}.get(base)
        payload = {"arm": arm_id, "grid": grids[base], "seeds": self.seeds, "n": self.n_regimes, "smoke": s,
                   "ref": self.state["arms"].get(ref_pool, {}).get("tuning_ref") if ref_pool else None}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]

    def tune_arm(self, arm_id: str) -> Dict[str, Any]:
        """Tune a specific arm by running its grid and edge extensions."""
        print(f"\n--- Tuning Arm {arm_id} on Bed {self.bed_name} (N={self.n_regimes}) ---", flush=True)

        grid_ref = self.grid_ref(arm_id)
        if arm_id in self.state["arms"]:
            if self.state["arms"][arm_id].get("grid_ref") == grid_ref:
                print(f"  [ALREADY TUNED] Arm {arm_id} chosen config: {self.state['arms'][arm_id]['chosen']}")
                return self.state["arms"][arm_id]
            print(f"  [STALE] Arm {arm_id} was tuned with another grid, seed list or reference: tuning it again (finished trials are reused)", flush=True)

        ref_pool: Optional[str] = None
        if arm_id in ("A3", "A4", "A7"):
            ref_pool = "A1"
        elif arm_id in ("A3-2", "A4-2", "A7-2"):
            ref_pool = "A2"

        trials_summary: List[Dict[str, Any]] = []

        if arm_id == "A1":
            grid = SMOKE_GRID_A1_STEPS if self.smoke else GRID_A1_STEPS
            for s in grid:
                summary = self.evaluate_candidate(arm_id, {"inner_steps": s})
                if summary:
                    trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            # Edge extension on inner_steps
            if best["config"]["inner_steps"] == grid[-1] and not self.smoke:
                best, at_edge = self._extend_edge(arm_id, best, "inner_steps", EDGE_EXTEND_FACTOR_STEPS, EDGE_EXTEND_CAP_STEPS)
            else:
                at_edge = False

        elif arm_id == "A1q":
            grid_s = SMOKE_GRID_A1Q_STEPS if self.smoke else GRID_A1Q_STEPS
            grid_q = SMOKE_GRID_A1Q_Q if self.smoke else GRID_A1Q_Q
            for s in grid_s:
                for q in grid_q:
                    summary = self.evaluate_candidate(arm_id, {"inner_steps": s, "q": q})
                    if summary:
                        trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            if best["config"]["inner_steps"] == grid_s[-1] and not self.smoke:
                best, at_edge = self._extend_edge(arm_id, best, "inner_steps", EDGE_EXTEND_FACTOR_STEPS, EDGE_EXTEND_CAP_STEPS)
            else:
                at_edge = False

        elif arm_id == "A2":
            a1q_steps = self.state["arms"]["A1q"]["chosen"]["inner_steps"]
            if self.smoke:
                cands = list(SMOKE_A2_CANDIDATES.keys())
            else:
                cands = list(A2_CANDIDATES.keys())
                if self.bed_name.startswith("B3"):
                    cands.extend(B3_UNIPOOL_CANDIDATES)
            grid_q = SMOKE_GRID_A2_Q if self.smoke else GRID_A2_Q
            for cand_name in cands:
                for q in grid_q:
                    summary = self.evaluate_candidate(arm_id, {"candidate": cand_name, "inner_steps": a1q_steps, "q": q})
                    if summary:
                        trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False

        elif arm_id in ("A3", "A3-2"):
            grid_w = SMOKE_GRID_A3_WIDTH if self.smoke else GRID_A3_WIDTH
            grid_s = SMOKE_GRID_A3_STEPS if self.smoke else GRID_A3_STEPS
            for w in grid_w:
                for s in grid_s:
                    summary = self.evaluate_candidate(arm_id, {"hidden": w, "inner_steps": s}, ref_pool=ref_pool)
                    if summary:
                        trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False
            if best["config"]["hidden"] == grid_w[-1] and not self.smoke:
                best, at_edge = self._extend_edge(arm_id, best, "hidden", EDGE_EXTEND_FACTOR_WIDTH, EDGE_EXTEND_CAP_WIDTH, ref_pool=ref_pool)

        elif arm_id in ("A4", "A4-2"):
            grid_w = SMOKE_GRID_A4_WIDTH if self.smoke else GRID_A4_WIDTH
            for w in grid_w:
                summary = self.evaluate_candidate(arm_id, {"hidden": w}, ref_pool=ref_pool)
                if summary:
                    trials_summary.append(summary)
            if not trials_summary:
                # If all failed budget, fallback to smallest width with 1 step
                summary = self.evaluate_candidate(arm_id, {"hidden": grid_w[0]}, ref_pool=ref_pool)
                trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False
            if best["config"]["hidden"] == grid_w[-1] and not self.smoke:
                best, at_edge = self._extend_edge(arm_id, best, "hidden", EDGE_EXTEND_FACTOR_WIDTH, EDGE_EXTEND_CAP_WIDTH, ref_pool=ref_pool)

        elif arm_id == "A5a":
            grid = SMOKE_GRID_A5A_SHRINKAGE if self.smoke else GRID_A5A_SHRINKAGE
            for shr in grid:
                summary = self.evaluate_candidate(arm_id, {"shrinkage": shr})
                if summary:
                    trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False

        elif arm_id == "A5b":
            grid = SMOKE_GRID_A5B_SHRINKAGE if self.smoke else GRID_A5B_SHRINKAGE
            for shr in grid:
                summary = self.evaluate_candidate(arm_id, {"shrinkage": shr})
                if summary:
                    trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False

        elif arm_id == "A6":
            grid_d = SMOKE_GRID_A6_D if self.smoke else GRID_A6_D
            grid_r = SMOKE_GRID_A6_RIDGE if self.smoke else GRID_A6_RIDGE
            for d_val in grid_d:
                for r in grid_r:
                    summary = self.evaluate_candidate(arm_id, {"D": d_val, "ridge": r})
                    if summary:
                        trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False
            if best["config"]["D"] == grid_d[-1] and not self.smoke:
                best, at_edge = self._extend_edge(arm_id, best, "D", EDGE_EXTEND_FACTOR_D, EDGE_EXTEND_CAP_D)

        elif arm_id in ("A7", "A7-2"):
            grid = SMOKE_GRID_A7_K if self.smoke else GRID_A7_K
            for k in grid:
                summary = self.evaluate_candidate(arm_id, {"k": k}, ref_pool=ref_pool)
                if summary:
                    trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False

        elif arm_id == "A7all":
            grid = SMOKE_GRID_A7ALL_K if self.smoke else GRID_A7ALL_K
            for k in grid:
                summary = self.evaluate_candidate(arm_id, {"k": k})
                if summary:
                    trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False

        elif arm_id == "A8":
            grid_w = SMOKE_GRID_A8_WIDTH if self.smoke else GRID_A8_WIDTH
            grid_s = SMOKE_GRID_A8_STEPS if self.smoke else GRID_A8_STEPS
            for w in grid_w:
                for s in grid_s:
                    summary = self.evaluate_candidate(arm_id, {"hidden": w, "inner_steps": s})
                    if summary:
                        trials_summary.append(summary)
            best = select_best_candidate(trials_summary)
            at_edge = False
            if best["config"]["hidden"] == grid_w[-1] and not self.smoke:
                best, at_edge = self._extend_edge(arm_id, best, "hidden", EDGE_EXTEND_FACTOR_WIDTH, EDGE_EXTEND_CAP_WIDTH)

        else:
            raise ValueError(f"Unknown arm_id {arm_id!r} in tuning driver.")

        chosen_entry = {
            "grid_ref": grid_ref,
            "chosen": best["config"],
            "at_edge": at_edge,
            "tuning_ref": compute_tuning_ref(best["config"]),
            "best_summary": {
                "mean_accuracy": best["mean_accuracy"],
                "mean_memory_bytes": best["mean_memory_bytes"],
                "mean_wall_clock": best["mean_wall_clock"],
            },
            "trials": trials_summary,
        }
        self.state["arms"][arm_id] = chosen_entry
        self.save_tuning_state()
        print(f"  [CHOSEN] {arm_id}: {best['config']} (acc={best['mean_accuracy']*100:.2f}%, at_edge={at_edge})")
        return chosen_entry


def run_tuning(
    bed: str = "B1",
    tune_n: int = 30,
    arms: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    out: Path | str = "results/m1/tuning",
    cpus: Optional[List[int]] = None,
    threads: int = 2,
    smoke: bool = False,
) -> None:
    """Execute tuning driver for specified bed, arms, and seeds."""
    if bed == "B2":
        tune_n = 6
    elif bed.startswith("B3"):
        tune_n = 22
    target_arms = arms if arms is not None else list(ALL_ARM_ORDER)
    # Filter and preserve dependency order
    ordered_arms = [a for a in ALL_ARM_ORDER if a in target_arms]
    cpus_list = cpus if cpus is not None else list(PINNING_CPUS)

    with fixed_threads(n=threads, cpus=cpus_list):
        driver = TuningDriver(bed_name=bed, n_regimes=tune_n, out_dir=out, smoke=smoke, seeds=seeds)
        for arm in ordered_arms:
            driver.tune_arm(arm)


def parse_seeds(text: str) -> List[int]:
    """Parse comma-separated seed specifications and ranges (e.g. '800', '800-804')."""
    seeds: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_str, _, hi_str = part.partition("-")
            seeds.extend(range(int(lo_str), int(hi_str) + 1))
        else:
            seeds.append(int(part))
    return sorted(list(dict.fromkeys(seeds)))


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="M1 benchmark hyperparameter tuning driver.")
    parser.add_argument("--bed", default="B1", help="Bed identifier (default: B1).")
    parser.add_argument("--tune-n", type=int, default=30, help="Regime count for tuning (default: 30 for B1).")
    parser.add_argument("--arms", nargs="+", default=list(ALL_ARM_ORDER), help="Arms to tune.")
    parser.add_argument("--seeds", default="800-804", help="Seeds or range (default: 800-804).")
    parser.add_argument("--out", default="results/m1/tuning", help="Output directory for tuning files.")
    parser.add_argument("--cpus", default="0,2", help="Logical processors to pin (default: 0,2).")
    parser.add_argument("--threads", type=int, default=2, help="Thread count (default: 2).")
    parser.add_argument("--smoke", action="store_true", help="Use single-value smoke grids for quick verification.")
    parser.add_argument("--resume", action="store_true", default=True, help="Resume from existing trials (default: True).")

    args = parser.parse_args(argv)
    cpus = [int(c.strip()) for c in args.cpus.split(",")] if args.cpus else list(PINNING_CPUS)
    seeds = parse_seeds(args.seeds)

    # Flatten arms if space or comma separated
    arms: List[str] = []
    for a in args.arms:
        for sub in a.split(","):
            sub = sub.strip()
            if sub:
                arms.append(sub)

    run_tuning(
        bed=args.bed,
        tune_n=args.tune_n,
        arms=arms,
        seeds=seeds,
        out=args.out,
        cpus=cpus,
        threads=args.threads,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
