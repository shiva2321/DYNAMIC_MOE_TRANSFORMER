"""V0 Walking Skeleton Demonstration runner.

Runs V0 on a stream of real text topics along with three honest comparators:
- Replay network (A3) matched at V0's final memory
- k-Nearest Neighbours (A7) matched at V0's final memory
- Streaming Linear Discriminant Analysis (A5a)

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    phase_indices = [p["phase"] for p in phases]

    # Colors
    c_v0 = "#1f77b4"
    c_a3 = ARM_COLORS.get("A3", "#2ca02c")
    c_a7 = ARM_COLORS.get("A7", "#e377c2")
    c_a5a = ARM_COLORS.get("A5a", "#9467bd")

    # 1. Mean accuracy over seen regimes against phase
    ax1 = axes[0, 0]
    v0_accs = [p["v0"]["mean_accuracy"] for p in phases]
    ax1.plot(phase_indices, v0_accs, label="V0", color=c_v0, marker="o", markersize=3, linewidth=1.5)

    has_comparators = "a3" in phases[0]
    if has_comparators:
        a3_accs = [p["a3"]["mean_accuracy"] for p in phases]
        a7_accs = [p["a7"]["mean_accuracy"] for p in phases]
        a5a_accs = [p["a5a"]["mean_accuracy"] for p in phases]
        ax1.plot(phase_indices, a3_accs, label="A3 (ReplayNet)", color=c_a3, marker="s", markersize=3, linewidth=1.2)
        ax1.plot(phase_indices, a7_accs, label="A7 (kNN)", color=c_a7, marker="^", markersize=3, linewidth=1.2)
        ax1.plot(phase_indices, a5a_accs, label="A5a (Streaming LDA)", color=c_a5a, marker="d", markersize=3, linewidth=1.2)

    v0_final_acc = final_data["v0"]["final_accuracy"]
    if has_comparators:
        a3_final_acc = final_data["a3"]["final_accuracy"]
        a7_final_acc = final_data["a7"]["final_accuracy"]
        a5a_final_acc = final_data["a5a"]["final_accuracy"]
        ax1.set_title(f"Accuracy: V0 {v0_final_acc:.1%} vs A3 {a3_final_acc:.1%}, A7 {a7_final_acc:.1%}, A5a {a5a_final_acc:.1%}", fontsize=10)
    else:
        ax1.set_title(f"Accuracy: V0 Final {v0_final_acc:.1%}", fontsize=10)
    ax1.set_xlabel("Phase", fontsize=9)
    ax1.set_ylabel("Mean Accuracy", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=8)

    # 2. Experts of V0 against phase
    ax2 = axes[0, 1]
    v0_experts = [p["v0"]["n_experts"] for p in phases]
    ax2.plot(phase_indices, v0_experts, color=c_v0, marker="o", markersize=3, linewidth=1.5)
    ax2.set_title(f"V0 Experts: {v0_experts[-1]} experts at phase {phase_indices[-1]}", fontsize=10)
    ax2.set_xlabel("Phase", fontsize=9)
    ax2.set_ylabel("Expert Count", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)

    # 3. Memory in bytes against phase for all four
    ax3 = axes[1, 0]
    v0_mems = [p["v0"]["memory_bytes"] / (1024 * 1024) for p in phases]
    ax3.plot(phase_indices, v0_mems, label="V0", color=c_v0, marker="o", markersize=3, linewidth=1.5)
    if has_comparators:
        a3_mems = [p["a3"]["memory_bytes"] / (1024 * 1024) for p in phases]
        a7_mems = [p["a7"]["memory_bytes"] / (1024 * 1024) for p in phases]
        a5a_mems = [p["a5a"]["memory_bytes"] / (1024 * 1024) for p in phases]
        ax3.plot(phase_indices, a3_mems, label="A3", color=c_a3, marker="s", markersize=3, linewidth=1.2)
        ax3.plot(phase_indices, a7_mems, label="A7", color=c_a7, marker="^", markersize=3, linewidth=1.2)
        ax3.plot(phase_indices, a5a_mems, label="A5a", color=c_a5a, marker="d", markersize=3, linewidth=1.2)
        ax3.set_title(f"Memory: V0 {v0_mems[-1]:.2f} MB vs A3 {a3_mems[-1]:.2f} MB", fontsize=10)
    else:
        ax3.set_title(f"Memory: V0 Final {v0_mems[-1]:.2f} MB", fontsize=10)
    ax3.set_xlabel("Phase", fontsize=9)
    ax3.set_ylabel("Memory (MB)", fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=8)

    # 4. Unknown-flag rates of V0 against phase
    ax4 = axes[1, 1]
    fnr_rates = [p["v0"]["unknown_flag"]["false_new_rate"] for p in phases]
    ood_rates = [p["v0"]["unknown_flag"]["ood_detection_rate"] for p in phases]
    ax4.plot(phase_indices, fnr_rates, label="Known Test Marked Unknown (FNR)", color="#d62728", marker="o", markersize=3, linewidth=1.5)
    ax4.plot(phase_indices, ood_rates, label="Out-of-Domain Marked Unknown", color="#2ca02c", marker="s", markersize=3, linewidth=1.5)
    mean_auc = final_data["v0"]["auroc"].get("mean", 0.5)
    ax4.set_title(f"Unknown Flag: FNR {fnr_rates[-1]:.1%}, OOD {ood_rates[-1]:.1%} (AUROC {mean_auc:.2f})", fontsize=10)
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
        "| Arm | Final Accuracy | Memory (Bytes) | Memory (MB) | Latency Batch 1 (ms) | Latency Batch 32 (ms) |",
        "|---|---|---|---|---|---|",
    ]

    has_comp = "a3" in final_data
    arms_order = ["v0", "a3", "a7", "a5a"] if has_comp else ["v0"]
    arm_labels = {"v0": "V0 (UniPool)", "a3": "A3 (ReplayNet)", "a7": "A7 (kNN)", "a5a": "A5a (Streaming LDA)"}

    for arm_k in arms_order:
        d = final_data[arm_k]
        acc_s = f"{d['final_accuracy']:.2%}"
        mem_b = d["final_memory_bytes"]
        mem_mb = f"{mem_b / (1024 * 1024):.2f}"
        lat = d.get("answer_latency", {})
        l1 = f"{lat.get('1', {}).get('latency_median_s', 0.0) * 1000.0:.2f}"
        l32 = f"{lat.get('32', {}).get('latency_median_s', 0.0) * 1000.0:.2f}"
        lines.append(f"| {arm_labels[arm_k]} | {acc_s} | {mem_b:,} | {mem_mb} MB | {l1} ms | {l32} ms |")

    lines.extend([
        "",
        "## Novelty / Unknown Flag Diagnostics (V0)",
        "",
        f"- **Final False New Rate (Known Test Marked Unknown):** {phases[-1]['v0']['unknown_flag']['false_new_rate']:.2%}",
        f"- **Final OOD Detection Rate (Out-of-Domain Marked Unknown):** {phases[-1]['v0']['unknown_flag']['ood_detection_rate']:.2%}",
        f"- **Rank AUROC (Novelty Score 1 - p_nov):**",
    ])
    for ood_name, auc_v in final_data["v0"]["auroc"].get("by_dataset", {}).items():
        lines.append(f"  - `{ood_name}`: {auc_v:.4f}")
    lines.append(f"  - **Mean AUROC:** {final_data['v0']['auroc'].get('mean', 0.5):.4f}")

    lines.extend([
        "",
        "## Progression Tables (Per Phase Boundary)",
        "",
    ])

    for p in phases:
        ph_idx = p["phase"]
        n_reg = len(p["v0"]["regimes_seen"])
        v0_acc = f"{p['v0']['mean_accuracy']:.2%}"
        v0_exp = p["v0"]["n_experts"]
        v0_mem = f"{p['v0']['memory_bytes'] / (1024 * 1024):.2f} MB"
        fnr = f"{p['v0']['unknown_flag']['false_new_rate']:.2%}"
        ood_f = f"{p['v0']['unknown_flag']['ood_detection_rate']:.2%}"

        lines.append(f"### Phase {ph_idx} Boundary (Regimes Seen: {n_reg})")
        if has_comp:
            a3_acc = f"{p['a3']['mean_accuracy']:.2%}"
            a7_acc = f"{p['a7']['mean_accuracy']:.2%}"
            a5a_acc = f"{p['a5a']['mean_accuracy']:.2%}"
            lines.extend([
                "| Metric | V0 (UniPool) | A3 (ReplayNet) | A7 (kNN) | A5a (LDA) |",
                "|---|---|---|---|---|",
                f"| Mean Accuracy (Seen Regimes) | {v0_acc} | {a3_acc} | {a7_acc} | {a5a_acc} |",
                f"| Memory Footprint | {v0_mem} | {p['a3']['memory_bytes'] / (1024*1024):.2f} MB | {p['a7']['memory_bytes'] / (1024*1024):.2f} MB | {p['a5a']['memory_bytes'] / (1024*1024):.2f} MB |",
                f"| Experts | {v0_exp} | 1 | - | - |",
                f"| False New Rate (Known) | {fnr} | - | - | - |",
                f"| OOD Detection Rate | {ood_f} | - | - | - |",
                "",
            ])
        else:
            lines.extend([
                "| Metric | V0 (UniPool) |",
                "|---|---|",
                f"| Mean Accuracy (Seen Regimes) | {v0_acc} |",
                f"| Memory Footprint | {v0_mem} |",
                f"| Experts | {v0_exp} |",
                f"| False New Rate (Known) | {fnr} |",
                f"| OOD Detection Rate | {ood_f} |",
                "",
            ])

    return "\n".join(lines)


def generate_summary_text(final_data: Dict[str, Any], phases: List[Dict[str, Any]], schedule: str, seed: int) -> str:
    """Generate plain English summary with all generated numbers."""
    v0_data = final_data["v0"]
    has_comp = "a3" in final_data
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
        f"Forgetting up to stream end: {v0_forg_str}.",
    ]

    comp_sentence = ""
    if has_comp:
        a3_acc = final_data["a3"]["final_accuracy"]
        a7_acc = final_data["a7"]["final_accuracy"]
        a5a_acc = final_data["a5a"]["final_accuracy"]
        lines.extend([
            f"Comparator ReplayNet (A3) final accuracy: {a3_acc:.2%}.",
            f"Comparator kNN (A7) final accuracy: {a7_acc:.2%}.",
            f"Comparator Streaming LDA (A5a) final accuracy: {a5a_acc:.2%}.",
        ])
        winners = []
        if a3_acc > v0_acc:
            winners.append(f"A3 (ReplayNet by {a3_acc - v0_acc:+.2%})")
        if a7_acc > v0_acc:
            winners.append(f"A7 (kNN by {a7_acc - v0_acc:+.2%})")
        if a5a_acc > v0_acc:
            winners.append(f"A5a (Streaming LDA by {a5a_acc - v0_acc:+.2%})")

        if winners:
            comp_sentence = f"On this stream, comparator(s) {' and '.join(winners)} beat V0 on final mean accuracy."
        else:
            comp_sentence = "On this stream, V0 achieved higher final mean accuracy than all three honest comparators."
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

    if has_comp:
        a3_mem = final_data["a3"]["final_memory_bytes"]
        lat_a3 = final_data["a3"].get("answer_latency", {})
        l1_a3 = lat_a3.get("1", {}).get("latency_median_s", 0.0) * 1000.0
        lines.extend([
            f"A3 ReplayNet memory: {a3_mem:,} bytes ({a3_mem / (1024*1024):.2f} MB), answer latency: {l1_a3:.2f} ms (batch 1).",
            f"A7 kNN memory: {final_data['a7']['final_memory_bytes']:,} bytes.",
            f"A5a Streaming LDA memory: {final_data['a5a']['final_memory_bytes']:,} bytes.",
        ])

    fnr = phases[-1]["v0"]["unknown_flag"]["false_new_rate"]
    ood_rate = phases[-1]["v0"]["unknown_flag"]["ood_detection_rate"]
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
    cpus: Optional[List[int]] = None,
    threads: int = 2,
) -> Dict[str, Any]:
    """Execute V0 demonstration run end to end with honest comparators.

    Args:
        schedule: Stream schedule ('seq', 'mix', or 'drift').
        seed: Experiment seed (must be a dev seed 800-804).
        out_dir: Output directory for records and artifacts.
        max_phases: Maximum stream phases to run (for tests/fast smoke).
        tuning_dir: Directory containing tuning JSON files.
        comparators: Whether to evaluate A3, A7, A5a comparators.
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

            phase_records.append({
                "phase": ph_idx,
                "v0": {
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
            })

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

        final_data: Dict[str, Any] = {
            "v0": {
                "config": v0.config(),
                "final_accuracy": float(R_rows[-1][0]) if len(R_rows[-1]) == 1 else float(np.mean([R_rows[-1][r] for r in all_seen])),
                "per_regime_accuracy": [float(x) for x in R_rows[-1]],
                "final_memory_bytes": v0_final_memory,
                "final_experts": len(v0.model.experts),
                "forgetting": phase_records[-1]["v0"]["forgetting"] if phase_records else 0.0,
                "update_seconds_total": v0_update_time,
                "answer_latency": v0_latency,
                "auroc": v0_auroc,
            }
        }

        batch_hashes = {"V0": v0_batch_hash}

        # Step 2: Run Comparators
        if comparators:
            tuning_map_b3 = load_tuning_map(tuning_dir, "B3seq")
            tuning_a3 = tuning_map_b3.get("A3")
            tuning_a7 = tuning_map_b3.get("A7")
            tuning_a5a = tuning_map_b3.get("A5a")

            a3 = build_arm("A3", bed, pool_memory_bytes=v0_final_memory, tuning=tuning_a3)
            a7 = build_arm("A7", bed, pool_memory_bytes=v0_final_memory, tuning=tuning_a7)
            a5a = build_arm("A5a", bed, tuning=tuning_a5a)

            comp_arms = [("a3", "A3", a3), ("a7", "A7", a7), ("a5a", "A5a", a5a)]

            for key, arm_id, arm in comp_arms:
                c_hasher = hashlib.sha256()
                c_cur_phase = -1
                c_R_rows: List[List[float]] = []
                c_seen_set: set[int] = set()
                c_update_time = 0.0

                def _eval_comp_row() -> List[float]:
                    return [
                        float((arm.predict(x).probs.argmax(dim=-1).cpu() == y.cpu()).float().mean().item())
                        for x, y in (tests[r] for r in range(bed.n_regimes))
                    ]

                p_record_idx = 0

                for x_b, y_b, regimes, first_regimes, phase_idx in bed.walk_full(64):
                    if max_phases is not None and phase_idx >= max_phases:
                        break
                    c_hasher.update(y_b.cpu().numpy().tobytes())
                    if phase_idx != c_cur_phase:
                        if c_cur_phase >= 0:
                            row_c = _eval_comp_row()
                            c_R_rows.append(row_c)
                            c_seen_list = sorted(list(c_seen_set)) if c_seen_set else [0]
                            c_mean_acc = float(np.mean([row_c[r] for r in c_seen_list]))
                            if p_record_idx < len(phase_records):
                                phase_records[p_record_idx][key] = {
                                    "mean_accuracy": c_mean_acc,
                                    "memory_bytes": arm.memory_bytes(),
                                    "update_seconds_so_far": c_update_time,
                                }
                            p_record_idx += 1
                        c_cur_phase = phase_idx

                    for r in regimes:
                        c_seen_set.add(r)
                    for r in first_regimes:
                        c_seen_set.add(r)

                    t0 = time.perf_counter()
                    arm.observe(x_b, y_b)
                    c_update_time += time.perf_counter() - t0

                if c_cur_phase >= 0:
                    row_c = _eval_comp_row()
                    c_R_rows.append(row_c)
                    c_seen_list = sorted(list(c_seen_set)) if c_seen_set else [0]
                    c_mean_acc = float(np.mean([row_c[r] for r in c_seen_list]))
                    if p_record_idx < len(phase_records):
                        phase_records[p_record_idx][key] = {
                            "mean_accuracy": c_mean_acc,
                            "memory_bytes": arm.memory_bytes(),
                            "update_seconds_so_far": c_update_time,
                        }

                c_lat = measure_answer_latency(arm, bed)
                final_data[key] = {
                    "config": arm.config(),
                    "final_accuracy": float(np.mean([c_R_rows[-1][r] for r in c_seen_list])),
                    "per_regime_accuracy": [float(x) for x in c_R_rows[-1]],
                    "final_memory_bytes": arm.memory_bytes(),
                    "update_seconds_total": c_update_time,
                    "answer_latency": c_lat,
                }
                batch_hashes[arm_id] = c_hasher.hexdigest()

        # Generate output texts and figures
        table_md = generate_tables_markdown(phase_records, final_data, schedule, seed)
        summary_txt = generate_summary_text(final_data, phase_records, schedule, seed)

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
            "arms": {
                "V0": final_data["v0"],
            },
            "phases": phase_records,
            "batch_hashes": batch_hashes,
        }

        if comparators:
            full_record["arms"]["A3"] = final_data["a3"]
            full_record["arms"]["A7"] = final_data["a7"]
            full_record["arms"]["A5a"] = final_data["a5a"]

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
        cpus=cpus,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
