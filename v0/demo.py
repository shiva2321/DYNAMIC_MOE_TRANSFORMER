"""V0 Walking Skeleton Demonstration runner.

Runs V0 on a stream of real text topics along with honest comparators:
- Replay network (A3) matched at V0's final memory
- k-Nearest Neighbours (A7) matched at V0's final memory
- Streaming Linear Discriminant Analysis (A5a)
- Mahalanobis classifier (A5b)
- Random-feature ridge regression (A6)
- k-Nearest Neighbours storing all seen examples (A7all)
- Optional: Offline joint network reference ceiling (A8, via --with-reference)

Outputs: record.json, table.md, figure.png, summary.txt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from costlab.env import environment, fixed_threads
from costlab.report import _CostlabJSONEncoder
from costlab.timing import time_callable
from growmoe.metrics import accuracy_stats
from m1.arms import build_arm
from m1.beds import Bed, build_bed
from m1.queue import load_tuning_map
from m1.report import ARM_COLORS
from v0.auroc import rank_auroc
from v0.system import build_v0, describe_v0

DEMO_LABEL = "DEMONSTRATION, NOT A RESULT"

DEFAULT_COMPARATORS = ["A3", "A7", "A5a", "A5b", "A6", "A7all"]

ARM_DESCRIPTIONS: Dict[str, str] = {
    "V0": "V0 (UniPool)",
    "A3": "A3 (ReplayNet)",
    "A7": "A7 (kNN)",
    "A5a": "A5a (Streaming LDA)",
    "A5b": "A5b (Mahalanobis)",
    "A6": "A6 (Random-Feature Ridge)",
    "A7all": "A7all (kNN All)",
    "A8": "A8 (OFFLINE REFERENCE, NOT A METHOD)",
}


def measure_answer_latency(arm: Any, bed: Bed) -> Dict[str, Any]:
    """Measure answer latency for batch sizes 1 and 32 on test set inputs."""
    test_x = bed.test_sets[0][0]
    res: Dict[str, Any] = {}
    for b in (1, 32):
        x_b = test_x[torch.arange(b) % len(test_x)]
        fn = lambda: arm.predict(x_b)
        t_res = time_callable(fn, warmup=5, repeats=10, device="cpu")
        res[str(b)] = {
            "batch_size": b,
            "rows_timed": int(len(x_b)),
            "latency_median_s": float(t_res.median),
            "latency_mean_s": float(t_res.mean),
            "latency_std_s": float(t_res.std),
            "latency_cv": float(t_res.cv),
            "latency_p95_s": float(t_res.p95),
            "items_per_second": float(b / t_res.median) if t_res.median > 0 else 0.0,
        }
    return res


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via temporary file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via temporary file replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_CostlabJSONEncoder)
    tmp_path.replace(path)


def write_figure(path: Path, phases: List[Dict[str, Any]], final_data: Dict[str, Any]) -> None:
    """Render small multiples figure showing accuracy, experts, memory, and novelty flags."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    phase_indices = [p["phase"] for p in phases]

    # Active arm ids present in final_data
    arm_ids = [k for k in ["V0", "A3", "A7", "A5a", "A5b", "A6", "A7all", "A8"] if k in final_data]

    # Markers mapping
    markers = {
        "V0": "o",
        "A3": "s",
        "A7": "^",
        "A5a": "d",
        "A5b": "v",
        "A6": "p",
        "A7all": "x",
        "A8": "*",
    }

    # 1. Mean accuracy over seen regimes against phase
    ax1 = axes[0, 0]
    for arm_id in arm_ids:
        accs = [p[arm_id]["mean_accuracy"] for p in phases if arm_id in p]
        c = "#1f77b4" if arm_id == "V0" else ARM_COLORS.get(arm_id, "#555555")
        label = ARM_DESCRIPTIONS.get(arm_id, arm_id)
        ax1.plot(
            phase_indices[:len(accs)],
            accs,
            label=label,
            color=c,
            marker=markers.get(arm_id, "."),
            markersize=3,
            linewidth=1.3 if arm_id != "V0" else 1.8,
        )

    v0_final_acc = final_data["V0"]["final_accuracy"]
    ax1.set_title(f"Accuracy Progression (V0 Final: {v0_final_acc:.1%})", fontsize=10)
    ax1.set_xlabel("Phase", fontsize=9)
    ax1.set_ylabel("Mean Accuracy (Seen Regimes)", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=7)

    # 2. Experts of V0 against phase
    ax2 = axes[0, 1]
    v0_experts = [p["V0"]["n_experts"] for p in phases]
    ax2.plot(phase_indices, v0_experts, color="#1f77b4", marker="o", markersize=3, linewidth=1.5)
    ax2.set_title(f"V0 Experts: {v0_experts[-1]} experts at phase {phase_indices[-1]}", fontsize=10)
    ax2.set_xlabel("Phase", fontsize=9)
    ax2.set_ylabel("Expert Count", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Memory in MB against phase
    ax3 = axes[1, 0]
    for arm_id in arm_ids:
        mems = [p[arm_id]["memory_bytes"] / (1024 * 1024) for p in phases if arm_id in p]
        c = "#1f77b4" if arm_id == "V0" else ARM_COLORS.get(arm_id, "#555555")
        label = ARM_DESCRIPTIONS.get(arm_id, arm_id)
        ax3.plot(
            phase_indices[:len(mems)],
            mems,
            label=label,
            color=c,
            marker=markers.get(arm_id, "."),
            markersize=3,
            linewidth=1.3 if arm_id != "V0" else 1.8,
        )

    v0_final_mb = final_data["V0"]["final_memory_bytes"] / (1024 * 1024)
    ax3.set_title(f"Memory Footprint (V0 Final: {v0_final_mb:.2f} MB)", fontsize=10)
    ax3.set_xlabel("Phase", fontsize=9)
    ax3.set_ylabel("Memory (MB)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=7)

    # 4. Unknown-flag rates of V0 against phase
    ax4 = axes[1, 1]
    fnr_rates = [p["V0"]["unknown_flag"]["false_new_rate"] for p in phases]
    ood_rates = [p["V0"]["unknown_flag"]["ood_detection_rate"] for p in phases]
    ax4.plot(phase_indices, fnr_rates, label="Known Test Marked Unknown (FNR)", color="#d62728", marker="o", markersize=3, linewidth=1.5)
    ax4.plot(phase_indices, ood_rates, label="Out-of-Domain Marked Unknown", color="#2ca02c", marker="s", markersize=3, linewidth=1.5)
    mean_auc = final_data["V0"]["auroc"].get("mean", 0.5)
    ax4.set_title(f"Unknown Flag Rates (AUROC {mean_auc:.2f})", fontsize=10)
    ax4.set_xlabel("Phase", fontsize=9)
    ax4.set_ylabel("Flag Rate", fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    tmp_path = path.with_suffix(path.suffix + ".tmp.png")
    fig.savefig(tmp_path, dpi=120)
    plt.close(fig)
    tmp_path.replace(path)


def generate_tables_markdown(phases: List[Dict[str, Any]], final_data: Dict[str, Any], schedule: str, seed: int) -> str:
    """Generate Markdown representation with one table per phase boundary plus summary."""
    lines: List[str] = [
        f"# V0 Walking Skeleton Demonstration: Schedule `{schedule}`, Seed `{seed}`",
        f"**Status:** {DEMO_LABEL}",
        "",
        "## Final Performance Summary",
        "",
        "| Arm | Final Accuracy | Forgetting | Memory (Bytes) | Memory (MB) | Update Time (s) | Latency Batch 1 (ms) | Latency Batch 32 (ms) | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    arm_order = [k for k in ["V0", "A3", "A7", "A5a", "A5b", "A6", "A7all", "A8"] if k in final_data]

    for arm_id in arm_order:
        d = final_data[arm_id]
        acc_s = f"{d['final_accuracy']:.2%}"
        forg = d.get("forgetting")
        forg_s = f"{forg:.2%}" if forg is not None else "-"
        mem_b = d["final_memory_bytes"]
        mem_mb = f"{mem_b / (1024 * 1024):.2f}"
        up_s = f"{d.get('update_seconds_total', 0.0):.2f}"
        lat = d.get("answer_latency", {})
        l1 = f"{lat.get('1', {}).get('latency_median_s', 0.0) * 1000.0:.2f}"
        l32 = f"{lat.get('32', {}).get('latency_median_s', 0.0) * 1000.0:.2f}"

        notes = []
        if arm_id == "A8":
            notes.append("OFFLINE REFERENCE, NOT A METHOD")
        if d.get("capped_by_stream_size"):
            notes.append("Capped by stream size")
        note_str = "; ".join(notes) if notes else "-"

        label = ARM_DESCRIPTIONS.get(arm_id, arm_id)
        lines.append(f"| {label} | {acc_s} | {forg_s} | {mem_b:,} | {mem_mb} MB | {up_s} | {l1} ms | {l32} ms | {note_str} |")

    lines.extend([
        "",
        "## Novelty / Unknown Flag Diagnostics (V0)",
        "",
        f"- **Final False New Rate (Known Test Marked Unknown):** {phases[-1]['V0']['unknown_flag']['false_new_rate']:.2%}",
        f"- **Final OOD Detection Rate (Out-of-Domain Marked Unknown):** {phases[-1]['V0']['unknown_flag']['ood_detection_rate']:.2%}",
        f"- **Rank AUROC (Novelty Score 1 - p_nov):**",
    ])
    for ood_name, auc_v in final_data["V0"]["auroc"].get("by_dataset", {}).items():
        lines.append(f"  - `{ood_name}`: {auc_v:.4f}")
    lines.append(f"  - **Mean AUROC:** {final_data['V0']['auroc'].get('mean', 0.5):.4f}")

    lines.extend([
        "",
        "## Progression Tables (Per Phase Boundary)",
        "",
    ])

    for p in phases:
        ph_idx = p["phase"]
        n_reg = len(p["V0"]["regimes_seen"])
        lines.append(f"### Phase {ph_idx} Boundary (Regimes Seen: {n_reg})")

        header = "| Metric | " + " | ".join(ARM_DESCRIPTIONS.get(a, a) for a in arm_order) + " |"
        sep = "|---|" + "|".join("---" for _ in arm_order) + "|"
        lines.extend([header, sep])

        # Mean accuracy row
        acc_cells = []
        for a in arm_order:
            acc_cells.append(f"{p[a]['mean_accuracy']:.2%}" if a in p else "-")
        lines.append("| Mean Accuracy | " + " | ".join(acc_cells) + " |")

        # Memory row
        mem_cells = []
        for a in arm_order:
            mem_cells.append(f"{p[a]['memory_bytes'] / (1024*1024):.2f} MB" if a in p else "-")
        lines.append("| Memory Footprint | " + " | ".join(mem_cells) + " |")

        # Experts / details row
        exp_cells = []
        for a in arm_order:
            if a == "V0":
                exp_cells.append(str(p["V0"]["n_experts"]))
            elif a in ("A3", "A8"):
                exp_cells.append("1 (net)")
            else:
                exp_cells.append("-")
        lines.append("| Experts | " + " | ".join(exp_cells) + " |")
        lines.append("")

    return "\n".join(lines)


def generate_summary_text(
    final_data: Dict[str, Any],
    phases: List[Dict[str, Any]],
    schedule: str,
    seed: int,
    raw_stream_bytes: Optional[int] = None,
) -> str:
    """Generate plain English summary with all generated numbers."""
    v0_data = final_data["V0"]
    v0_acc = v0_data["final_accuracy"]
    v0_mem = v0_data["final_memory_bytes"]
    v0_exp = v0_data["final_experts"]
    v0_forg = v0_data.get("forgetting")
    v0_forg_str = f"{v0_forg:.2%}" if v0_forg is not None else "None"
    v0_time = v0_data["update_seconds_total"]
    lat_v0 = v0_data.get("answer_latency", {})
    l1_v0 = lat_v0.get("1", {}).get("latency_median_s", 0.0) * 1000.0
    l32_v0 = lat_v0.get("32", {}).get("latency_median_s", 0.0) * 1000.0

    lines: List[str] = [
        f"V0 WALKING SKELETON DEMONSTRATION (Schedule: {schedule}, Seed: {seed})",
        f"LABEL: {DEMO_LABEL}",
        "--------------------------------------------------------------------------------",
        "1. BEHAVIOUR AND ACCURACY:",
        f"V0 final mean accuracy across all seen regimes: {v0_acc:.2%}.",
        f"V0 forgetting up to stream end: {v0_forg_str}.",
    ]

    comp_ids = [k for k in ["A3", "A7", "A5a", "A5b", "A6", "A7all", "A8"] if k in final_data]

    if comp_ids:
        for c_id in comp_ids:
            c_d = final_data[c_id]
            c_acc = c_d["final_accuracy"]
            c_forg = c_d.get("forgetting")
            c_forg_str = f"{c_forg:.2%}" if c_forg is not None else "None"
            ref_note = " [OFFLINE REFERENCE, NOT A METHOD]" if c_id == "A8" else ""
            lines.append(
                f"Comparator {c_id}{ref_note} final accuracy: {c_acc:.2%}, forgetting: {c_forg_str}."
            )

        ahead = []
        for c_id in comp_ids:
            c_d = final_data[c_id]
            c_acc = c_d["final_accuracy"]
            if c_acc > v0_acc:
                diff = c_acc - v0_acc
                c_mem_mb = c_d["final_memory_bytes"] / (1024 * 1024)
                v0_mem_mb = v0_mem / (1024 * 1024)
                ahead.append(
                    f"{c_id} (diff {diff:+.2%}, memory {c_mem_mb:.2f} MB vs V0 {v0_mem_mb:.2f} MB)"
                )

        if ahead:
            comp_sentence = f"On this stream, comparator(s) ahead of V0 on final mean accuracy: {'; '.join(ahead)}."
        else:
            comp_sentence = "no comparator printed here is ahead on this stream"
        lines.append(comp_sentence)
    else:
        lines.append("No comparators were run for this configuration.")

    lines.extend([
        "",
        "2. COST NEXT TO BEHAVIOUR:",
        f"V0 spawned {v0_exp} experts with a final memory footprint of {v0_mem:,} bytes ({v0_mem / (1024*1024):.2f} MB).",
        f"V0 cumulative update time: {v0_time:.2f} seconds.",
        f"V0 answer latency: {l1_v0:.2f} ms (batch 1), {l32_v0:.2f} ms (batch 32).",
    ])

    if raw_stream_bytes is not None and raw_stream_bytes > 0:
        raw_mb = raw_stream_bytes / (1024 * 1024)
        v0_ratio = v0_mem / float(raw_stream_bytes)
        lines.append(
            f"Raw training stream size: {raw_stream_bytes:,} bytes ({raw_mb:.2f} MB); V0 final memory is {v0_mem / (1024*1024):.2f} MB ({v0_ratio:.2f}x raw stream size)."
        )

    for c_id in comp_ids:
        c_d = final_data[c_id]
        if c_d.get("capped_by_stream_size"):
            mem_mb = c_d["final_memory_bytes"] / (1024 * 1024)
            lines.append(
                f"{c_id} could not use the budget: it already holds the whole training stream, {mem_mb:.2f} MB"
            )

    for c_id in comp_ids:
        c_d = final_data[c_id]
        c_mem = c_d["final_memory_bytes"]
        c_up = c_d.get("update_seconds_total", 0.0)
        lat = c_d.get("answer_latency", {})
        l1 = lat.get("1", {}).get("latency_median_s", 0.0) * 1000.0
        l32 = lat.get("32", {}).get("latency_median_s", 0.0) * 1000.0
        ref_tag = " [OFFLINE REFERENCE, NOT A METHOD]" if c_id == "A8" else ""
        lines.append(
            f"{c_id}{ref_tag} memory: {c_mem:,} bytes ({c_mem / (1024*1024):.2f} MB), update: {c_up:.2f}s, latency: {l1:.2f} ms (b1), {l32:.2f} ms (b32)."
        )

    fnr = phases[-1]["V0"]["unknown_flag"]["false_new_rate"]
    ood_rate = phases[-1]["V0"]["unknown_flag"]["ood_detection_rate"]
    mean_auc = v0_data["auroc"].get("mean", 0.5)

    lines.extend([
        "",
        "3. UNKNOWN FLAG & NOVELTY DETECTION:",
        f"V0 false new rate on known test examples: {fnr:.2%}.",
        f"V0 out-of-domain detection rate: {ood_rate:.2%}.",
        f"V0 rank AUROC vs out-of-domain sets (1 - p_nov): {mean_auc:.4f}.",
    ])
    for ood_n, auc_val in v0_data["auroc"].get("by_dataset", {}).items():
        lines.append(f"  - vs {ood_n}: {auc_val:.4f}")

    if fnr == 0.0 and ood_rate == 0.0:
        lines.append(
            "The unknown flag never fired at its configured level (on known or on out-of-domain examples); the AUROC measures the novelty score, not the flag."
        )

    lines.extend([
        "A novelty AUROC value near 0.5 means the unknown flag cannot distinguish out-of-domain text from known topics and does not work.",
        "",
        "--------------------------------------------------------------------------------",
        "End of demonstration summary.",
    ])
    return "\n".join(lines)


def run_demo(
    schedule: str,
    seed: int,
    out_dir: Path | str,
    max_phases: Optional[int] = None,
    tuning_dir: str = "results/m1/tuning",
    comparators: bool = True,
    with_reference: bool = False,
    cpus: Optional[List[int]] = None,
    threads: int = 2,
) -> Dict[str, Any]:
    """Execute V0 demonstration run end to end with honest comparators.

    Args:
        schedule: Stream schedule ('seq', 'mix', or 'drift').
        seed: Experiment seed (dev seed 800-804).
        out_dir: Output directory for records and artifacts.
        max_phases: Maximum stream phases to run (for tests/fast smoke).
        tuning_dir: Directory containing tuning JSON files.
        comparators: Whether to evaluate comparators (A3, A7, A5a, A5b, A6, A7all).
        with_reference: Whether to include offline reference A8.
        cpus: Logical processors to pin.
        threads: Intra-op thread count.

    Returns:
        The complete record dictionary.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    rec_file = out_path / "record.json"
    tbl_file = out_path / "table.md"
    fig_file = out_path / "figure.png"
    sum_file = out_path / "summary.txt"

    # Idempotence check
    if rec_file.exists() and tbl_file.exists() and fig_file.exists() and sum_file.exists():
        try:
            existing = json.loads(rec_file.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "complete"
                and existing.get("schedule") == schedule
                and existing.get("seed") == seed
                and existing.get("max_phases") == max_phases
                and existing.get("label") == DEMO_LABEL
            ):
                print(f"[SKIP] Complete demo record already exists in {out_path} for schedule={schedule}, seed={seed}, max_phases={max_phases}")
                return existing
        except (OSError, ValueError):
            pass

    cpus_list = cpus if cpus is not None else [8, 10]

    with fixed_threads(n=threads, cpus=cpus_list) as pinning:
        bed_name = f"B3{schedule}"
        bed = build_bed(bed_name, 22, seed)

        env_info = environment()

        # Step 1: Run V0
        v0 = build_v0(bed, tuning_dir=tuning_dir)

        v0_batch_hasher = hashlib.sha256()
        cur_phase = -1
        last_regime = -1
        R_rows: List[List[float]] = []
        learned_at: Dict[int, int] = {}
        seen_regimes_set: set[int] = set()
        phase_records: List[Dict[str, Any]] = []
        v0_update_time = 0.0

        tests = bed.test_sets

        def _eval_row() -> List[float]:
            return [
                float((v0.predict(x).probs.argmax(dim=-1).cpu() == y.cpu()).float().mean().item())
                for x, y in (tests[r] for r in range(bed.n_regimes))
            ]

        def _record_phase_boundary(ph_idx: int) -> None:
            nonlocal v0_update_time
            eval_row = _eval_row()
            R_rows.append(eval_row)
            seen_list = sorted(list(seen_regimes_set))
            if not seen_list:
                seen_list = [0]
            mean_acc = float(np.mean([eval_row[r] for r in seen_list]))

            # Forgetting
            R_arr = np.array(R_rows, dtype=np.float64)
            if len(R_rows) > 1:
                la = [learned_at.get(r, 0) for r in seen_list]
                st = accuracy_stats(R_arr[:, seen_list], learned_at=la, chance=1.0 / bed.n_classes)
                forg = None if np.isnan(st["forgetting"]) else float(st["forgetting"])
            else:
                forg = 0.0

            # Unknown flags on known test
            x_known = torch.cat([tests[r][0] for r in seen_list])
            pred_known = v0.predict(x_known)
            fnr = float(pred_known.unknown.float().mean().item())

            # OOD flags
            ood_rates: Dict[str, float] = {}
            for ood_name, x_ood in bed.stream.ood.items():
                p_ood = v0.predict(x_ood)
                ood_rates[ood_name] = float(p_ood.unknown.float().mean().item())
            mean_ood = float(np.mean(list(ood_rates.values()))) if ood_rates else 0.0

            phase_entry = {
                "phase": ph_idx,
                "V0": {
                    "regimes_seen": list(seen_list),
                    "mean_accuracy": mean_acc,
                    "per_regime_accuracy": {str(r): eval_row[r] for r in seen_list},
                    "forgetting": forg,
                    "n_experts": len(v0.model.experts),
                    "memory_bytes": v0.memory_bytes(),
                    "update_seconds_so_far": v0_update_time,
                    "unknown_flag": {
                        "false_new_rate": fnr,
                        "ood_detection_rate": mean_ood,
                        "ood_by_domain": ood_rates,
                    },
                },
            }
            # Backwards compatibility alias
            phase_entry["v0"] = phase_entry["V0"]
            phase_records.append(phase_entry)

        for x_b, y_b, regimes, first_regimes, phase_idx in bed.walk_full(64):
            if max_phases is not None and phase_idx >= max_phases:
                break
            v0_batch_hasher.update(y_b.cpu().numpy().tobytes())
            regime_id = regimes[0]
            if phase_idx != cur_phase:
                if cur_phase >= 0:
                    _record_phase_boundary(cur_phase)
                cur_phase = phase_idx
                last_regime = regime_id

            for r_first in first_regimes:
                if r_first not in learned_at:
                    learned_at[r_first] = len(R_rows)
                seen_regimes_set.add(r_first)
            for r in regimes:
                seen_regimes_set.add(r)

            t0 = time.perf_counter()
            v0.observe(x_b, y_b)
            v0_update_time += time.perf_counter() - t0

        if cur_phase >= 0:
            _record_phase_boundary(cur_phase)

        v0_final_memory = v0.memory_bytes()
        v0_batch_hash = v0_batch_hasher.hexdigest()

        # AUROC computation at stream end
        all_seen = sorted(list(seen_regimes_set)) if seen_regimes_set else [0]
        x_known_all = torch.cat([tests[r][0] for r in all_seen])
        pred_known_all = v0.predict(x_known_all)
        p_nov_known = pred_known_all.p_nov.detach().cpu().numpy()
        s_known = 1.0 - p_nov_known

        auroc_by_dataset: Dict[str, float] = {}
        for ood_name, x_ood in bed.stream.ood.items():
            pred_ood = v0.predict(x_ood)
            p_nov_ood = pred_ood.p_nov.detach().cpu().numpy()
            s_ood = 1.0 - p_nov_ood
            auroc_by_dataset[ood_name] = rank_auroc(neg=s_known, pos=s_ood)

        mean_auroc = float(np.mean(list(auroc_by_dataset.values()))) if auroc_by_dataset else 0.5
        v0_auroc = {
            "by_dataset": auroc_by_dataset,
            "mean": mean_auroc,
        }

        # Answer latency
        v0_latency = measure_answer_latency(v0, bed)

        v0_final_dict: Dict[str, Any] = {
            "config": v0.config(),
            "final_accuracy": float(R_rows[-1][0]) if len(R_rows[-1]) == 1 else float(np.mean([R_rows[-1][r] for r in all_seen])),
            "per_regime_accuracy": [float(x) for x in R_rows[-1]],
            "final_memory_bytes": v0_final_memory,
            "final_experts": len(v0.model.experts),
            "forgetting": phase_records[-1]["V0"]["forgetting"] if phase_records else 0.0,
            "update_seconds_total": v0_update_time,
            "answer_latency": v0_latency,
            "auroc": v0_auroc,
        }

        final_data: Dict[str, Any] = {
            "V0": v0_final_dict,
            "v0": v0_final_dict,  # backwards compatibility alias
        }

        batch_hashes = {"V0": v0_batch_hash}

        # Step 2: Run Comparators
        if comparators:
            tuning_map_b3 = load_tuning_map(tuning_dir, "B3seq")

            active_comp_ids = list(DEFAULT_COMPARATORS)
            if with_reference:
                active_comp_ids.append("A8")

            for arm_id in active_comp_ids:
                tuning_arm = tuning_map_b3.get(arm_id)
                if arm_id in ("A3", "A7"):
                    arm = build_arm(arm_id, bed, pool_memory_bytes=v0_final_memory, tuning=tuning_arm)
                else:
                    arm = build_arm(arm_id, bed, tuning=tuning_arm)

                c_hasher = hashlib.sha256()
                c_cur_phase = -1
                c_R_rows: List[List[float]] = []
                c_seen_set: set[int] = set()
                c_learned_at: Dict[int, int] = {}
                c_update_time = 0.0

                def _eval_comp_row(current_arm: Any) -> List[float]:
                    return [
                        float((current_arm.predict(x).probs.argmax(dim=-1).cpu() == y.cpu()).float().mean().item())
                        for x, y in (tests[r] for r in range(bed.n_regimes))
                    ]

                p_record_idx = 0

                for x_b, y_b, regimes, first_regimes, phase_idx in bed.walk_full(64):
                    if max_phases is not None and phase_idx >= max_phases:
                        break
                    c_hasher.update(y_b.cpu().numpy().tobytes())
                    if phase_idx != c_cur_phase:
                        if c_cur_phase >= 0:
                            row_c = _eval_comp_row(arm)
                            c_R_rows.append(row_c)
                            c_seen_list = sorted(list(c_seen_set)) if c_seen_set else [0]
                            c_mean_acc = float(np.mean([row_c[r] for r in c_seen_list]))
                            if p_record_idx < len(phase_records):
                                phase_records[p_record_idx][arm_id] = {
                                    "mean_accuracy": c_mean_acc,
                                    "memory_bytes": arm.memory_bytes(),
                                    "update_seconds_so_far": c_update_time,
                                }
                                # Lowercase alias
                                phase_records[p_record_idx][arm_id.lower()] = phase_records[p_record_idx][arm_id]
                            p_record_idx += 1
                        c_cur_phase = phase_idx

                    for r in regimes:
                        c_seen_set.add(r)
                    for r in first_regimes:
                        if r not in c_learned_at:
                            c_learned_at[r] = len(c_R_rows)
                        c_seen_set.add(r)

                    t0 = time.perf_counter()
                    arm.observe(x_b, y_b)
                    c_update_time += time.perf_counter() - t0

                if c_cur_phase >= 0:
                    row_c = _eval_comp_row(arm)
                    c_R_rows.append(row_c)
                    c_seen_list = sorted(list(c_seen_set)) if c_seen_set else [0]
                    c_mean_acc = float(np.mean([row_c[r] for r in c_seen_list]))
                    if p_record_idx < len(phase_records):
                        phase_records[p_record_idx][arm_id] = {
                            "mean_accuracy": c_mean_acc,
                            "memory_bytes": arm.memory_bytes(),
                            "update_seconds_so_far": c_update_time,
                        }
                        phase_records[p_record_idx][arm_id.lower()] = phase_records[p_record_idx][arm_id]

                c_final_memory = arm.memory_bytes()

                # Forgetting for comparator
                c_R_arr = np.array(c_R_rows, dtype=np.float64)
                if len(c_R_rows) > 1:
                    la_c = [c_learned_at.get(r, 0) for r in c_seen_list]
                    st_c = accuracy_stats(c_R_arr[:, c_seen_list], learned_at=la_c, chance=1.0 / bed.n_classes)
                    c_forg = None if np.isnan(st_c["forgetting"]) else float(st_c["forgetting"])
                else:
                    c_forg = 0.0

                c_lat = measure_answer_latency(arm, bed)

                c_record: Dict[str, Any] = {
                    "config": arm.config(),
                    "final_accuracy": float(np.mean([c_R_rows[-1][r] for r in c_seen_list])),
                    "per_regime_accuracy": [float(x) for x in c_R_rows[-1]],
                    "forgetting": c_forg,
                    "final_memory_bytes": c_final_memory,
                    "update_seconds_total": c_update_time,
                    "answer_latency": c_lat,
                }

                if arm_id in ("A3", "A7"):
                    c_record["memory_budget_bytes"] = v0_final_memory
                    c_record["capped_by_stream_size"] = bool(c_final_memory < 0.95 * v0_final_memory)

                if arm_id == "A8":
                    c_record["note"] = "OFFLINE REFERENCE, NOT A METHOD"

                final_data[arm_id] = c_record
                final_data[arm_id.lower()] = c_record
                batch_hashes[arm_id] = c_hasher.hexdigest()

        # Compute raw stream bytes
        raw_stream_bytes = int(bed.total_samples * bed.d * 4)

        # Generate output texts and figures
        table_md = generate_tables_markdown(phase_records, final_data, schedule, seed)
        summary_txt = generate_summary_text(final_data, phase_records, schedule, seed, raw_stream_bytes=raw_stream_bytes)

        full_record: Dict[str, Any] = {
            "label": DEMO_LABEL,
            "status": "complete",
            "bed": bed.name,
            "schedule": schedule,
            "seed": seed,
            "max_phases": max_phases,
            "commit": env_info.get("git_commit"),
            "environment": env_info,
            "pinning": {"threads": threads, "cpus": cpus_list},
            "parts_description": describe_v0(),
            "raw_stream_bytes": raw_stream_bytes,
            "arms": {
                "V0": final_data["V0"],
            },
            "phases": phase_records,
            "batch_hashes": batch_hashes,
        }

        if comparators:
            active_comp_ids = list(DEFAULT_COMPARATORS)
            if with_reference:
                active_comp_ids.append("A8")
            for arm_id in active_comp_ids:
                if arm_id in final_data:
                    full_record["arms"][arm_id] = final_data[arm_id]

        atomic_write_json(rec_file, full_record)
        atomic_write_text(tbl_file, table_md)
        atomic_write_text(sum_file, summary_txt)
        write_figure(fig_file, phase_records, final_data)

        return full_record


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V0 Walking Skeleton Demo runner.")
    parser.add_argument("--schedule", choices=["seq", "mix", "drift"], default="seq", help="Schedule (seq, mix, drift).")
    parser.add_argument("--seed", type=int, default=800, help="Development seed (800-804).")
    parser.add_argument("--out", type=str, required=True, help="Output directory.")
    parser.add_argument("--max-phases", type=int, default=None, help="Max phases to run (optional, for testing).")
    parser.add_argument("--tuning-dir", type=str, default="results/m1/tuning", help="Tuning directory.")
    parser.add_argument("--cpus", type=str, default="8,10", help="CPUs to pin.")
    parser.add_argument("--threads", type=int, default=2, help="Thread count.")
    parser.add_argument("--no-comparators", action="store_true", help="Omit comparators (for test fast paths).")
    parser.add_argument("--with-reference", action="store_true", help="Include A8 offline reference ceiling.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cpus = [int(c.strip()) for c in args.cpus.split(",")] if args.cpus else [8, 10]
    run_demo(
        schedule=args.schedule,
        seed=args.seed,
        out_dir=args.out,
        max_phases=args.max_phases,
        tuning_dir=args.tuning_dir,
        comparators=not args.no_comparators,
        with_reference=args.with_reference,
        cpus=cpus,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
