"""GPU repeat reporting tool for milestone M1 wide baselines (Job G-1).

Compares CPU and GPU performance across wide baselines (A3, A3-2, A4, A4-2, A6, A8),
verifies multiply-add counts, and evaluates restated Rule 6.1 in both hardware readings.
Generates REPORT_M1_GPU.md, gpu_verdicts.json, and gpu_speedup.png under --out.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from costlab.report import _CostlabJSONEncoder
from growmoe.metrics import mean_ci, paired_diff
from m1.cell import SCHEMA_VERSION, is_valid_cell_record, load_cell_record
from m1.config_m1 import ALL_ARMS
from m1.report import ARM_COLORS, load_records
from m1.verdict import (
    compute_arm_summary,
    evaluate_all_predictions,
    evaluate_rule_6_1,
    evaluate_rule_6_1_all,
    group_records,
    record_bed_key,
)

REPEATED_ARMS = ("A3", "A3-2", "A4", "A4-2", "A6", "A8")


def format_val_ci(mean: float, ci_lo: float, ci_hi: float, is_pct: bool = False, signed: bool = False) -> str:
    """Format mean with 95% confidence interval cleanly."""
    sign = "+" if signed and mean > 0 else ""
    if math.isinf(ci_lo) or math.isinf(ci_hi) or math.isnan(ci_lo):
        return f"{sign}{mean * 100.0:.2f}%" if is_pct else f"{sign}{mean:.4f}"
    if is_pct:
        return f"{sign}{mean * 100.0:.2f}% [{ci_lo * 100.0:.2f}%, {ci_hi * 100.0:.2f}%]"
    return f"{sign}{mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"


def _fmt_energy(value: Optional[float], digits: int) -> str:
    """Joules or 'n/a (idle baseline not stable or energy unavailable)'."""
    return f"{value:.{digits}f}" if value is not None else "n/a"


def compute_comparison_tables(
    cpu_records: Sequence[Dict[str, Any]],
    gpu_records: Sequence[Dict[str, Any]],
    config_mismatches: Optional[List[str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build per-bed comparison table for repeated arms across common seeds.

    A CPU cell and a GPU cell are paired only when their stored arm configurations are equal (same width, steps, buffer, budgets);
    pairs that differ are left out and named in ``config_mismatches`` (a difference between two configurations is not a hardware effect).
    """
    cpu_grouped = group_records(cpu_records)
    gpu_grouped = group_records(gpu_records)

    all_beds = sorted(set(cpu_grouped.keys()) | set(gpu_grouped.keys()))
    comparison: Dict[str, List[Dict[str, Any]]] = {}

    for bed in all_beds:
        bed_rows: List[Dict[str, Any]] = []
        cpu_arms = cpu_grouped.get(bed, {})
        gpu_arms = gpu_grouped.get(bed, {})

        for arm in REPEATED_ARMS:
            c_recs = cpu_arms.get(arm, [])
            g_recs = gpu_arms.get(arm, [])
            if not c_recs or not g_recs:
                continue

            c_by_seed = {r["seed"]: r for r in c_recs if "seed" in r}
            g_by_seed = {r["seed"]: r for r in g_recs if "seed" in r}
            both = sorted(set(c_by_seed.keys()) & set(g_by_seed.keys()))
            common_seeds = [s for s in both if c_by_seed[s].get("config") == g_by_seed[s].get("config")]
            if config_mismatches is not None:
                config_mismatches += [f"{bed} {arm} seed {s}" for s in both if s not in common_seeds]
            if not common_seeds:
                continue

            c_common = [c_by_seed[s] for s in common_seeds]
            g_common = [g_by_seed[s] for s in common_seeds]

            # 1. Update Wall Clock
            c_up_walls = [float(r["update_cost"]["wall_clock_total"]) for r in c_common if "wall_clock_total" in r.get("update_cost", {})]
            g_up_walls = [float(r["update_cost"]["wall_clock_total"]) for r in g_common if "wall_clock_total" in r.get("update_cost", {})]
            c_mean_wall = float(np.mean(c_up_walls)) if c_up_walls else None
            g_mean_wall = float(np.mean(g_up_walls)) if g_up_walls else None
            wall_speedup = (c_mean_wall / g_mean_wall) if (c_mean_wall is not None and g_mean_wall is not None and g_mean_wall > 0) else None

            # 2. Answer Latency (b=1, b=32)
            c_ans_1 = [float(r["answer_cost"]["1"]["latency_median_s"]) for r in c_common if "1" in r.get("answer_cost", {}) and "latency_median_s" in r["answer_cost"]["1"]]
            g_ans_1 = [float(r["answer_cost"]["1"]["latency_median_s"]) for r in g_common if "1" in r.get("answer_cost", {}) and "latency_median_s" in r["answer_cost"]["1"]]
            c_mean_ans_1 = float(np.mean(c_ans_1)) if c_ans_1 else None
            g_mean_ans_1 = float(np.mean(g_ans_1)) if g_ans_1 else None
            ans_1_speedup = (c_mean_ans_1 / g_mean_ans_1) if (c_mean_ans_1 is not None and g_mean_ans_1 is not None and g_mean_ans_1 > 0) else None

            c_ans_32 = [float(r["answer_cost"]["32"]["latency_median_s"]) for r in c_common if "32" in r.get("answer_cost", {}) and "latency_median_s" in r["answer_cost"]["32"]]
            g_ans_32 = [float(r["answer_cost"]["32"]["latency_median_s"]) for r in g_common if "32" in r.get("answer_cost", {}) and "latency_median_s" in r["answer_cost"]["32"]]
            c_mean_ans_32 = float(np.mean(c_ans_32)) if c_ans_32 else None
            g_mean_ans_32 = float(np.mean(g_ans_32)) if g_ans_32 else None
            ans_32_speedup = (c_mean_ans_32 / g_mean_ans_32) if (c_mean_ans_32 is not None and g_mean_ans_32 is not None and g_mean_ans_32 > 0) else None

            # 3. Peak VRAM
            vram_peaks = [float(r.get("vram_peak_bytes", 0) or 0) for r in g_common]
            mean_vram_mb = (float(np.mean(vram_peaks)) / 1e6) if vram_peaks else 0.0

            # 4. Accuracy & Paired Diff
            c_accs = [float(r["accuracy"]) for r in c_common if "accuracy" in r]
            g_accs = [float(r["accuracy"]) for r in g_common if "accuracy" in r]
            c_mean_acc = float(np.mean(c_accs)) if c_accs else 0.0
            g_mean_acc = float(np.mean(g_accs)) if g_accs else 0.0
            diff = paired_diff(g_accs, c_accs) if (len(g_accs) == len(c_accs) and len(g_accs) > 0) else None

            # 5. Counted Multiply-Adds Equality
            up_macs_equal = True
            ans_1_macs_equal = True
            ans_32_macs_equal = True
            for c_r, g_r in zip(c_common, g_common):
                c_u = c_r.get("update_cost", {}).get("counted_macs")
                g_u = g_r.get("update_cost", {}).get("counted_macs")
                if c_u is not None and g_u is not None and c_u != g_u:
                    up_macs_equal = False
                c_a1 = c_r.get("answer_cost", {}).get("1", {}).get("counted_macs")
                g_a1 = g_r.get("answer_cost", {}).get("1", {}).get("counted_macs")
                if c_a1 is not None and g_a1 is not None and c_a1 != g_a1:
                    ans_1_macs_equal = False
                c_a32 = c_r.get("answer_cost", {}).get("32", {}).get("counted_macs")
                g_a32 = g_r.get("answer_cost", {}).get("32", {}).get("counted_macs")
                if c_a32 is not None and g_a32 is not None and c_a32 != g_a32:
                    ans_32_macs_equal = False

            macs_all_equal = up_macs_equal and ans_1_macs_equal and ans_32_macs_equal

            # 6. GPU energy (net of the idle power, only where the idle baseline was stable)
            net_ans = [
                float(r["energy"]["net_joules_per_answer_batch_32"])
                for r in g_common
                if isinstance(r.get("energy"), dict) and r["energy"].get("net_joules_per_answer_batch_32") is not None
            ]
            net_upd = [
                float(r["energy"]["net_joules_per_100_updates"])
                for r in g_common
                if isinstance(r.get("energy"), dict) and r["energy"].get("net_joules_per_100_updates") is not None
            ]

            bed_rows.append({
                "arm": arm,
                "n_seeds": len(common_seeds),
                "seeds": common_seeds,
                "cpu_update_wall_s": c_mean_wall,
                "gpu_update_wall_s": g_mean_wall,
                "update_speedup": wall_speedup,
                "cpu_answer_lat_1_ms": (c_mean_ans_1 * 1000.0) if c_mean_ans_1 is not None else None,
                "gpu_answer_lat_1_ms": (g_mean_ans_1 * 1000.0) if g_mean_ans_1 is not None else None,
                "answer_1_speedup": ans_1_speedup,
                "cpu_answer_lat_32_ms": (c_mean_ans_32 * 1000.0) if c_mean_ans_32 is not None else None,
                "gpu_answer_lat_32_ms": (g_mean_ans_32 * 1000.0) if g_mean_ans_32 is not None else None,
                "answer_32_speedup": ans_32_speedup,
                "gpu_vram_peak_mb": mean_vram_mb,
                "cpu_accuracy": c_mean_acc,
                "gpu_accuracy": g_mean_acc,
                "paired_diff": diff,
                "macs_equal": macs_all_equal,
                "up_macs_equal": up_macs_equal,
                "ans_macs_equal": ans_1_macs_equal and ans_32_macs_equal,
                "gpu_net_j_per_answer_b32": float(np.mean(net_ans)) if net_ans else None,
                "gpu_net_j_per_100_updates": float(np.mean(net_upd)) if net_upd else None,
                "gpu_energy_cells_with_net": min(len(net_ans), len(net_upd)),
            })
        if bed_rows:
            comparison[bed] = bed_rows

    return comparison


def build_hybrid_records(
    cpu_records: Sequence[Dict[str, Any]],
    gpu_records: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build hybrid records where repeated arms take GPU cost metrics, pool arms stay on CPU."""
    gpu_lookup = {}
    for g in gpu_records:
        gpu_lookup[(record_bed_key(g), g.get("arm"), g.get("seed"))] = g

    hybrid = []
    for c in cpu_records:
        key = (record_bed_key(c), c.get("arm"), c.get("seed"))
        arm = c.get("arm")
        if arm in REPEATED_ARMS and key in gpu_lookup and gpu_lookup[key].get("config") == c.get("config"):
            g = gpu_lookup[key]
            h = dict(c)
            h["update_cost"] = g.get("update_cost")
            h["answer_cost"] = g.get("answer_cost")
            h["device"] = g.get("device", "cuda")
            h["gpu"] = g.get("gpu")
            h["vram_peak_bytes"] = g.get("vram_peak_bytes")
            hybrid.append(h)
        else:
            hybrid.append(c)
    return hybrid


def count_gpu_substitutions(
    cpu_records: Sequence[Dict[str, Any]],
    gpu_records: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, int]]:
    """Per bed key: how many repeated-arm CPU cells have a GPU cell that takes their place in the hybrid reading, and how many exist."""
    gpu_configs = {(record_bed_key(g), g.get("arm"), g.get("seed")): g.get("config") for g in gpu_records}
    counts: Dict[str, Dict[str, int]] = {}
    for c in cpu_records:
        if c.get("arm") not in REPEATED_ARMS:
            continue
        key = (record_bed_key(c), c.get("arm"), c.get("seed"))
        entry = counts.setdefault(key[0], {"replaced": 0, "repeated_cpu_cells": 0})
        entry["repeated_cpu_cells"] += 1
        if key in gpu_configs and gpu_configs[key] == c.get("config"):      # same configuration only: see build_hybrid_records
            entry["replaced"] += 1
    return counts


def format_verdict_sentence(
    bed: str,
    cpu_v: Dict[str, Any],
    gpu_v: Dict[str, Any],
    reading: str = "R1_primary",
    n_gpu_cells: Optional[int] = None,
) -> str:
    """Generate one precise English sentence stating whether Rule 6.1 verdict changed.

    With ``n_gpu_cells`` (the number of GPU cells that replaced CPU cells on this bed) a bed with no GPU cell is reported as
    NOT EVALUATED: an unchanged verdict is only a statement when something was changed.
    """
    if n_gpu_cells is not None and n_gpu_cells == 0:
        return f"For bed {bed}, the Rule 6.1 ({reading}) hardware comparison is NOT EVALUATED: no GPU cell replaces a CPU cell on this bed."
    c_res = cpu_v.get(reading, {})
    g_res = gpu_v.get(reading, {})
    c_fired = c_res.get("fired", False)
    g_fired = g_res.get("fired", False)
    c_trig = c_res.get("triggering_baselines", [])
    g_trig = g_res.get("triggering_baselines", [])

    if c_fired == g_fired and set(c_trig) == set(g_trig):
        if not c_fired:
            return f"For bed {bed}, the Rule 6.1 ({reading}) efficiency verdict remains UNCHANGED: no baselines trigger under either CPU or GPU hardware readings."
        else:
            trig_str = ", ".join(sorted(c_trig))
            return f"For bed {bed}, the Rule 6.1 ({reading}) efficiency verdict remains UNCHANGED: baselines {trig_str} trigger under both CPU and GPU hardware readings."
    else:
        if not c_fired and g_fired:
            trig_str = ", ".join(sorted(g_trig))
            return f"For bed {bed}, the Rule 6.1 ({reading}) efficiency verdict CHANGED from not firing on CPU to firing on GPU (triggered by {trig_str})."
        elif c_fired and not g_fired:
            trig_str = ", ".join(sorted(c_trig))
            return f"For bed {bed}, the Rule 6.1 ({reading}) efficiency verdict CHANGED from firing on CPU (triggered by {trig_str}) to not firing on GPU."
        else:
            return f"For bed {bed}, the Rule 6.1 ({reading}) efficiency verdict CHANGED triggering baselines from {sorted(c_trig)} on CPU to {sorted(g_trig)} on GPU."


def generate_summary_figure(comparison: Dict[str, List[Dict[str, Any]]], out_path: Path) -> Optional[Path]:
    """Generate summary bar chart of GPU speedup per arm across beds."""
    flat_items = []
    for bed, rows in comparison.items():
        for r in rows:
            if r.get("update_speedup") is not None:
                flat_items.append({
                    "label": f"{r['arm']}\n({bed})",
                    "arm": r["arm"],
                    "speedup": r["update_speedup"],
                })

    if not flat_items:
        return None

    labels = [item["label"] for item in flat_items]
    speedups = [item["speedup"] for item in flat_items]
    colors = [ARM_COLORS.get(item["arm"], "#4c72b0") for item in flat_items]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 5), dpi=150)
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, speedups, width=0.45, color=colors, edgecolor="none", alpha=0.9)
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, label="1.0x (Parity)")

    min_sp = min(speedups)
    max_sp = max(speedups)
    ax.set_title(
        f"M1 Wide Baselines GPU Speedup (Update Wall-Clock: min {min_sp:.2f}x, max {max_sp:.2f}x)",
        fontsize=11,
        pad=10,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Speedup (CPU Wall-Clock / GPU Wall-Clock)", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_gpu_report(
    cpu_dir: str | Path,
    gpu_dir: str | Path,
    out_dir: str | Path,
) -> None:
    """Generate comprehensive GPU repeat report comparing CPU and GPU cells."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cpu_records, cpu_split = load_records(cpu_dir)
    gpu_records, gpu_split = load_records(gpu_dir)

    if not cpu_records:
        raise ValueError(f"No valid records found in CPU directory: {cpu_dir}")
    if not gpu_records:
        raise ValueError(f"No valid records found in GPU directory: {gpu_dir}")

    if cpu_split != gpu_split:
        raise ValueError(f"Split mismatch: CPU records are {cpu_split} but GPU records are {gpu_split}. Refusing to mix!")

    split_label = cpu_split

    config_mismatches: List[str] = []
    comparison = compute_comparison_tables(cpu_records, gpu_records, config_mismatches)
    hybrid_records = build_hybrid_records(cpu_records, gpu_records)

    cpu_rule_6_1 = evaluate_rule_6_1_all(cpu_records)
    gpu_rule_6_1 = evaluate_rule_6_1_all(hybrid_records)

    # Collect verdict changes
    all_beds = sorted(set(cpu_rule_6_1.keys()) | set(gpu_rule_6_1.keys()))
    substitutions = count_gpu_substitutions(cpu_records, gpu_records)
    sentences: Dict[str, Dict[str, str]] = {}
    for bed in all_beds:
        n_gpu = substitutions.get(bed, {}).get("replaced", 0)
        sentences[bed] = {
            r_name: format_verdict_sentence(bed, cpu_rule_6_1.get(bed, {}), gpu_rule_6_1.get(bed, {}), r_name, n_gpu_cells=n_gpu)
            for r_name in ("R1_primary", "R1_secondary", "R2_primary", "R2_secondary")
        }

    # Summary figure
    fig_path = out_path / "gpu_speedup.png"
    generate_summary_figure(comparison, fig_path)

    # Save json verdicts
    verdict_payload = {
        "split": split_label,
        "cpu_rule_6_1": cpu_rule_6_1,
        "gpu_rule_6_1": gpu_rule_6_1,
        "sentences": sentences,
        "gpu_substitutions": substitutions,
        "config_mismatches": config_mismatches,
        "comparison": comparison,
    }
    with open(out_path / "gpu_verdicts.json", "w", encoding="utf-8") as f:
        json.dump(verdict_payload, f, indent=2, cls=_CostlabJSONEncoder)

    # Build Markdown report
    md_lines: List[str] = [
        f"# M1 Wide Baselines GPU Repeat Report ({split_label})",
        "",
        "## Executive Summary",
        "",
        "This report measures the repeated performance of the wide baselines (`A3`, `A3-2`, `A4`, `A4-2`, `A6`, `A8`) on NVIDIA GPU "
        "relative to host CPU execution, while continual pools (`A1`, `A2`) remain on CPU.",
        "",
        "In the restated verdict the repeated arms take the update and answer COST fields (wall-clock, latency, counted multiply-adds) "
        "of their GPU cells; accuracy stays the CPU accuracy of the same arm and seeds, and the paired GPU minus CPU accuracy "
        "difference is in the tables below. Seeds without a GPU cell (for example the deterministic seeds) keep their CPU cells.",
        "",
        "GPU cells that replaced CPU cells, per bed: "
        + "; ".join(f"{b}: {v['replaced']} of {v['repeated_cpu_cells']}" for b, v in sorted(substitutions.items()))
        + ".",
        "",
        (
            "CPU and GPU cells with different stored configurations, left out of every table and of the restated verdict: "
            + "; ".join(config_mismatches) + "."
            if config_mismatches
            else "CPU and GPU cells with different stored configurations: none."
        ),
        "",
        "### Key Findings:",
    ]

    for bed in all_beds:
        s = sentences.get(bed, {}).get("R1_primary")
        if s:
            md_lines.append(f"- **{bed}**: {s}")

    md_lines.extend([
        "",
        "## Performance Comparison Tables",
        "",
    ])

    for bed, rows in comparison.items():
        md_lines.extend([
            f"### Bed {bed}",
            "",
            "| Arm | Seeds | CPU Up (s) | GPU Up (s) | Up Speedup | CPU Ans1 (ms) | GPU Ans1 (ms) | Ans1 Speedup | CPU Ans32 (ms) | GPU Ans32 (ms) | Ans32 Speedup | Peak VRAM (MB) | CPU Acc | GPU Acc | Paired Diff (95% CI) | MACs Equal | GPU net J per answer (b32) | GPU net J per 100 updates |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ])
        for r in rows:
            p_diff_str = "n/a"
            if r["paired_diff"]:
                pd = r["paired_diff"]
                p_diff_str = format_val_ci(pd["mean_diff"], pd["ci_lo"], pd["ci_hi"], is_pct=True, signed=True)

            mac_str = "EQUAL" if r["macs_equal"] else "MISMATCH"
            c_up_str = f"{r['cpu_update_wall_s']:.2f}" if r['cpu_update_wall_s'] is not None else "n/a (det)"
            g_up_str = f"{r['gpu_update_wall_s']:.2f}" if r['gpu_update_wall_s'] is not None else "n/a"
            up_sp_str = f"**{r['update_speedup']:.2f}x**" if r['update_speedup'] is not None else "n/a (det)"

            c_ans1_str = f"{r['cpu_answer_lat_1_ms']:.2f}" if r['cpu_answer_lat_1_ms'] is not None else "n/a (det)"
            g_ans1_str = f"{r['gpu_answer_lat_1_ms']:.2f}" if r['gpu_answer_lat_1_ms'] is not None else "n/a"
            ans1_sp_str = f"{r['answer_1_speedup']:.2f}x" if r['answer_1_speedup'] is not None else "n/a (det)"

            c_ans32_str = f"{r['cpu_answer_lat_32_ms']:.2f}" if r['cpu_answer_lat_32_ms'] is not None else "n/a (det)"
            g_ans32_str = f"{r['gpu_answer_lat_32_ms']:.2f}" if r['gpu_answer_lat_32_ms'] is not None else "n/a"
            ans32_sp_str = f"{r['answer_32_speedup']:.2f}x" if r['answer_32_speedup'] is not None else "n/a (det)"

            md_lines.append(
                f"| {r['arm']} | {len(r['seeds'])} | {c_up_str} | {g_up_str} | {up_sp_str} | "
                f"{c_ans1_str} | {g_ans1_str} | {ans1_sp_str} | {c_ans32_str} | {g_ans32_str} | {ans32_sp_str} | "
                f"{r['gpu_vram_peak_mb']:.1f} | {r['cpu_accuracy']*100:.2f}% | "
                f"{r['gpu_accuracy']*100:.2f}% | {p_diff_str} | {mac_str} | "
                f"{_fmt_energy(r.get('gpu_net_j_per_answer_b32'), 6)} | {_fmt_energy(r.get('gpu_net_j_per_100_updates'), 1)} |"
            )
        md_lines.append("")

    md_lines.extend([
        "## Restated Rule 6.1 Efficiency Verdicts",
        "",
        "| Bed | Reading | CPU Verdict | GPU Verdict | Status | Verdict Impact |",
        "|---|---|---|---|---|---|",
    ])

    for bed in all_beds:
        for r_name, r_label in [
            ("R1_primary", "R1 Primary (Wall-Clock)"),
            ("R1_secondary", "R1 Secondary (MACs)"),
            ("R2_primary", "R2 Primary (Wall-Clock)"),
            ("R2_secondary", "R2 Secondary (MACs)"),
        ]:
            c_v = cpu_rule_6_1.get(bed, {}).get(r_name, {})
            g_v = gpu_rule_6_1.get(bed, {}).get(r_name, {})
            c_status = c_v.get("status", "N/A")
            g_status = g_v.get("status", "N/A")
            c_fired = c_v.get("fired", False)
            g_fired = g_v.get("fired", False)
            if substitutions.get(bed, {}).get("replaced", 0) == 0:
                impact = "NOT EVALUATED (no GPU cell on this bed)"
            else:
                impact = "UNCHANGED" if c_fired == g_fired and set(c_v.get("triggering_baselines", [])) == set(g_v.get("triggering_baselines", [])) else "**CHANGED**"
            c_trig = ", ".join(c_v.get("triggering_baselines", [])) or "none"
            g_trig = ", ".join(g_v.get("triggering_baselines", [])) or "none"
            md_lines.append(
                f"| {bed} | {r_label} | {c_status} (trig: {c_trig}) | {g_status} (trig: {g_trig}) | {g_status} | {impact} |"
            )

    md_lines.extend([
        "",
        "## Verdict Decisions and Statements",
        "",
    ])

    for bed in all_beds:
        md_lines.append(f"### {bed}")
        for r_name in ("R1_primary", "R1_secondary", "R2_primary", "R2_secondary"):
            sentence = sentences.get(bed, {}).get(r_name, "")
            if sentence:
                md_lines.append(f"- **{r_name}**: {sentence}")
        md_lines.append("")

    report_path = out_path / "REPORT_M1_GPU.md"
    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Generated report at {report_path}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate GPU repeat report comparing CPU and GPU cells.")
    parser.add_argument("--cpu", required=True, help="Directory containing CPU cell records.")
    parser.add_argument("--gpu", required=True, help="Directory containing GPU cell records.")
    parser.add_argument("--out", required=True, help="Output directory for reports and figures.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_gpu_report(cpu_dir=args.cpu, gpu_dir=args.gpu, out_dir=args.out)


if __name__ == "__main__":
    main()
