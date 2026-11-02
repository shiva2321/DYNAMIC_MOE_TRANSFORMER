"""M1 benchmark reporting tool: scoreboards, verdicts, cost drivers, reconciliation, and figures.

Generates REPORT_M1.md, verdicts.json, and Pareto frontier figures.
Runs on development cells (labelled DEV) or report cells (labelled REPORT).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from costlab.report import _CostlabJSONEncoder
from growmoe.metrics import mean_ci
from m1.cell import SCHEMA_VERSION, is_valid_cell_record, load_cell_record
from m1.config_m1 import ALL_ARMS, HELD_DEFAULT_PAIRS, THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE
from m1.verdict import (
    compute_arm_summary,
    evaluate_all_predictions,
    evaluate_rule_6_1,
    evaluate_rule_6_1_all,
    evaluate_rule_6_2_all,
    group_records,
    record_bed_key,
)

# Fixed color-blind safe palette mapping every arm to the exact same color in every chart
ARM_COLORS: Dict[str, str] = {
    "A1": "#1f77b4",       # blue
    "A1q": "#aec7e8",      # light blue
    "A2": "#ff7f0e",       # orange
    "A3": "#2ca02c",       # green
    "A3-2": "#98df8a",     # light green
    "A4": "#d62728",       # red
    "A4-2": "#ff9896",     # light red
    "A5a": "#9467bd",      # purple
    "A5b": "#c5b0d5",      # light purple
    "A6": "#8c564b",       # brown
    "A7": "#e377c2",       # pink
    "A7-2": "#f7b6d2",     # light pink
    "A7all": "#7f7f7f",    # gray
    "A8": "#bcbd22",       # olive
    "A1L": "#17becf",      # cyan
}

ARM_ORDER: List[str] = list(ALL_ARMS) + ["A1L"]

# The statements of the predictions as written in docs/roadmap/M1_PREREG_DRAFT.md section 7 (the pool means A2, with A1 printed beside it)
PREDICTION_TEXT: Dict[str, str] = {
    "M1-1": "On B1 with N = 100, batch 1, CPU: the pool's median latency per input is at least that of the best-tuned replay network (no wall-clock gain) while its counted multiply-adds are at most a quarter of the replay network's.",
    "M1-2": "On B1 with N = 100, the pool-to-replay latency ratio at batch 32 is lower than at batch 1.",
    "M1-3": "Whole-stream training wall-clock of the pool is at most half that of the best same-memory replay arm on B1 with N = 30 and N = 100.",
    "M1-4": "Gate, router and bookkeeping take more than 60 % of the pool's observe wall-clock on B2 and B3.",
    "M1-5": "On B2 the better of kNN and streaming LDA reaches at least the pool's final accuracy.",
    "M1-6": "On B3, streaming LDA is within 0.02 of the LDA-head pool in at least two of the three schedules.",
    "M1-7": "Decision rule 6.1 fires on B2 and B3 and does not fire on B1.",
    "M1-8": "Pool memory grows linearly with the number of regimes on B1, the slope within 25 % of one expert's bytes.",
    "M1-9": "By rules 6.2: the LDA head is adopted, growth_rule gap is adopted, skip_known is not, transfer_chance is not.",
}


# -----------------------------------------------------------------------------
# Record Loading & Validation
# -----------------------------------------------------------------------------


def load_records(
    cells_dir: Path | str, requested_split: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], str]:
    """Load and validate cell records from directory, skipping invalid or older schema records.

    Returns:
        (valid_records, split_label)
    """
    cells_path = Path(cells_dir)
    if not cells_path.exists():
        raise FileNotFoundError(f"Cells directory does not exist: {cells_path}")

    records: List[Dict[str, Any]] = []
    splits_seen: set[str] = set()

    for file in sorted(cells_path.glob("*.json")):
        try:
            rec = load_cell_record(file)
        except Exception as e:
            print(f"[REPORT] Skipping unreadable JSON file: {file.name} ({e})")
            continue

        # Schema and structural validation
        if not is_valid_cell_record(rec):
            schema = rec.get("schema_version") if isinstance(rec, dict) else "unknown"
            print(f"[REPORT] Skipping invalid or other-schema record: {file.name} (schema_version={schema})")
            continue

        split = rec.get("split", "dev")
        splits_seen.add(split)

        if requested_split is not None and split != requested_split:
            continue

        records.append(rec)

    if not records:
        split_msg = f" for split {requested_split!r}" if requested_split else ""
        print(f"[REPORT] No valid schema {SCHEMA_VERSION} records found in {cells_path}{split_msg}.")
        return [], ""

    if requested_split is None:
        if len(splits_seen) > 1:
            raise ValueError(
                f"Mixed splits found in records ({splits_seen}). Please specify --split dev or --split report."
            )
        resolved_split = splits_seen.pop()
    else:
        resolved_split = requested_split

    return records, resolved_split.upper()


# -----------------------------------------------------------------------------
# Scoreboard Aggregation
# -----------------------------------------------------------------------------


def _format_ci(mean: float, ci_lo: float, ci_hi: float, is_pct: bool = False) -> str:
    if math.isinf(ci_lo) or math.isinf(ci_hi) or math.isnan(ci_lo):
        return f"{mean*100:.2f}%" if is_pct else f"{mean:.4f}"
    if is_pct:
        return f"{mean*100:.2f}% [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]"
    return f"{mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]"


def compute_scoreboard_rows(
    records: Sequence[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Compute scoreboard summary rows for each bed and arm."""
    grouped = group_records(records)
    scoreboards: Dict[str, List[Dict[str, Any]]] = {}

    for bed, arm_map in grouped.items():
        rows: List[Dict[str, Any]] = []
        # Sort arms in canonical order
        sorted_arms = sorted(arm_map.keys(), key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 999)

        for arm in sorted_arms:
            recs = arm_map[arm]
            accs = [float(r["accuracy"]) for r in recs if "accuracy" in r]
            m_acc, lo_acc, hi_acc = mean_ci(accs) if len(accs) >= 1 else (0.0, 0.0, 0.0)
            std_acc = float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0

            forgs = [float(r["forgetting"]) for r in recs if r.get("forgetting") is not None]
            if forgs:
                m_forg, lo_forg, hi_forg = mean_ci(forgs) if len(forgs) >= 1 else (0.0, 0.0, 0.0)
                std_forg = float(np.std(forgs, ddof=1)) if len(forgs) > 1 else 0.0
            else:
                m_forg, lo_forg, hi_forg, std_forg = None, None, None, None

            up_walls = [float(r["update_cost"]["wall_clock_total"]) for r in recs if "update_cost" in r and "wall_clock_total" in r["update_cost"]]
            m_wall, lo_wall, hi_wall = mean_ci(up_walls) if len(up_walls) >= 1 else (0.0, 0.0, 0.0)
            std_wall = float(np.std(up_walls, ddof=1)) if len(up_walls) > 1 else 0.0

            # Wall clock per regime
            per_reg_walls = []
            for r in recs:
                reg_d = r.get("update_cost", {}).get("wall_clock_per_regime", {})
                if reg_d:
                    per_reg_walls.append(float(np.mean(list(reg_d.values()))))
            m_per_reg_wall = float(np.mean(per_reg_walls)) if per_reg_walls else 0.0

            up_macs = [float(r["update_cost"]["counted_macs"]) for r in recs if "update_cost" in r and r["update_cost"].get("counted_macs") is not None]
            m_up_macs = float(np.mean(up_macs)) if up_macs else 0.0

            # costlab.memory.peak_memory returns a dict; det cells record None. Absolute peak RSS includes the process baseline, so the growth during the stream is kept too.
            peaks = [r["update_cost"]["peak_memory"] for r in recs if isinstance(r.get("update_cost", {}).get("peak_memory"), dict)]
            peak_ram = [float(p["rss_peak_bytes"]) / (1024 * 1024) for p in peaks if p.get("rss_peak_bytes") is not None]
            peak_ram_growth = [float(p["rss_diff_bytes"]) / (1024 * 1024) for p in peaks if p.get("rss_diff_bytes") is not None]
            m_peak_ram = float(np.mean(peak_ram)) if peak_ram else 0.0
            m_peak_ram_growth = float(np.mean(peak_ram_growth)) if peak_ram_growth else 0.0

            # Answer metrics
            ans_lat_b1 = []
            ans_p95_b1 = []
            ans_macs_b1 = []
            act_bytes_b1 = []
            tput_b32 = []
            for r in recs:
                ans = r.get("answer_cost", {})
                if "1" in ans:
                    if "latency_median_s" in ans["1"]:
                        ans_lat_b1.append(float(ans["1"]["latency_median_s"]) * 1000.0)  # in ms
                    if "latency_p95_s" in ans["1"]:
                        ans_p95_b1.append(float(ans["1"]["latency_p95_s"]) * 1000.0)     # in ms
                    if "counted_macs" in ans["1"] and ans["1"]["counted_macs"] is not None:
                        ans_macs_b1.append(float(ans["1"]["counted_macs"]))
                    if "active_bytes" in ans["1"] and ans["1"]["active_bytes"] is not None:
                        act_bytes_b1.append(float(ans["1"]["active_bytes"]))
                if "32" in ans and "items_per_second" in ans["32"]:
                    tput_b32.append(float(ans["32"]["items_per_second"]))

            m_lat_b1 = float(np.mean(ans_lat_b1)) if ans_lat_b1 else 0.0
            std_lat_b1 = float(np.std(ans_lat_b1, ddof=1)) if len(ans_lat_b1) > 1 else 0.0
            m_p95_b1 = float(np.mean(ans_p95_b1)) if ans_p95_b1 else 0.0
            m_ans_macs = float(np.mean(ans_macs_b1)) if ans_macs_b1 else 0.0
            m_act_bytes = float(np.mean(act_bytes_b1)) if act_bytes_b1 else 0.0
            m_tput_b32 = float(np.mean(tput_b32)) if tput_b32 else 0.0

            mem_bytes = [float(r["memory_bytes"]) for r in recs if "memory_bytes" in r]
            m_mem_bytes = float(np.mean(mem_bytes)) if mem_bytes else 0.0

            experts = [float(r.get("describe", {}).get("experts", 1)) for r in recs]
            m_experts = float(np.mean(experts)) if experts else 1.0

            # Operating point string
            first_cfg = recs[0].get("config", {})
            op_parts = []
            for k in ("inner_steps", "q", "hidden", "k", "D", "ridge", "shrinkage", "candidate"):
                if k in first_cfg and first_cfg[k] is not None:
                    op_parts.append(f"{k}={first_cfg[k]}")
            op_str = ", ".join(op_parts) if op_parts else "defaults"

            uncounted = any(r.get("uncounted_note", False) for r in recs)

            rows.append({
                "arm": arm,
                "uncounted": uncounted,
                "n_seeds": len(recs),
                "acc_mean": m_acc,
                "acc_std": std_acc,
                "acc_ci_lo": lo_acc,
                "acc_ci_hi": hi_acc,
                "forg_mean": m_forg,
                "forg_std": std_forg,
                "forg_ci_lo": lo_forg,
                "forg_ci_hi": hi_forg,
                "wall_mean": m_wall,
                "wall_std": std_wall,
                "wall_ci_lo": lo_wall,
                "wall_ci_hi": hi_wall,
                "wall_per_regime": m_per_reg_wall,
                "update_macs": m_up_macs,
                "peak_ram_mb": m_peak_ram,
                "peak_ram_growth_mb": m_peak_ram_growth,
                "lat_b1_ms": m_lat_b1,
                "lat_b1_std_ms": std_lat_b1,
                "p95_b1_ms": m_p95_b1,
                "tput_b32": m_tput_b32,
                "ans_macs": m_ans_macs,
                "active_bytes": m_act_bytes,
                "memory_bytes": m_mem_bytes,
                "experts": m_experts,
                "operating_point": op_str,
            })
        scoreboards[bed] = rows

    return scoreboards


# -----------------------------------------------------------------------------
# Pareto Frontiers & Plot Generation
# -----------------------------------------------------------------------------


def _pareto_frontier(points: List[Tuple[float, float, str]], lower_x_better: bool = True) -> List[Tuple[float, float, str]]:
    """Compute Pareto frontier for (cost, accuracy, arm). Higher accuracy is better; lower cost is better."""
    if not points:
        return []
    # Sort primarily by cost ascending, then accuracy descending
    sorted_pts = sorted(points, key=lambda p: (p[0], -p[1]))
    frontier = []
    max_acc = -1.0
    for cost, acc, arm in sorted_pts:
        if acc > max_acc:
            frontier.append((cost, acc, arm))
            max_acc = acc
    return frontier


SUMMARY_POOL_COLORS = {"A1": "#d95f02", "A1q": "#d95f02", "A2": "#1f77b4"}                 # the pool as built (and with its steps and q tuned) and the reference pool; every baseline is gray
SUMMARY_BASELINE_COLOR = "#6b7280"


def _summary_figure(scoreboards: Dict[str, List[Dict[str, Any]]], split_label: str, figures_dir: Path) -> Optional[str]:
    """Accuracy against answer latency (batch 1), one panel per bed, the Pareto front of each bed dashed.

    Only arms with a timed answer are drawn (a det cell has no latency); the held-default cells H_* are a rule 6.2 matter and stay out of this figure.
    Arms whose points coincide share one label.
    """
    panels = []
    for bed in sorted(scoreboards):
        rows = [r for r in scoreboards[bed] if r["lat_b1_ms"] > 0 and not r["arm"].startswith("H_")]
        if rows:
            panels.append((bed, rows))
    if not panels:
        return None
    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 4.3 * nrows), squeeze=False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    for ax, (bed, rows) in zip(axes.flat, panels):
        clusters: List[Tuple[float, float, List[str]]] = []                                # points within 6 % in latency and 0.5 points in accuracy share a label
        for r in sorted(rows, key=lambda q: (q["lat_b1_ms"], q["acc_mean"])):
            x, y = float(r["lat_b1_ms"]), float(r["acc_mean"]) * 100.0
            for cx, cy, names in clusters:
                if abs(x - cx) <= 0.06 * cx and abs(y - cy) <= 0.5:
                    names.append(r["arm"])
                    break
            else:
                clusters.append((x, y, [r["arm"]]))
        for x, y, arms in clusters:
            pool = [a for a in arms if a in SUMMARY_POOL_COLORS]
            color = SUMMARY_POOL_COLORS["A2" if "A2" in pool else pool[0]] if pool else SUMMARY_BASELINE_COLOR       # the reference pool's colour wins where points coincide
            ax.plot(x, y, marker="o", markersize=8 if pool else 6, color=color, markeredgecolor="white", markeredgewidth=1.0, linestyle="none", zorder=3)
            ax.annotate("/".join(arms), (x, y), xytext=(5, 4), textcoords="offset points", fontsize=7, color="#1f2937", zorder=4)
        front = _pareto_frontier([(r["lat_b1_ms"], r["acc_mean"], r["arm"]) for r in rows])
        if len(front) >= 2:
            ax.plot([q[0] for q in front], [q[1] * 100.0 for q in front], linestyle="--", color="#9ca3af", linewidth=1.0, zorder=2)
        ax.set_xscale("log")
        ax.set_title(bed, fontsize=10)
        ax.set_xlabel("Answer latency, batch 1 (ms, log scale)", fontsize=8)
        ax.set_ylabel("Final accuracy (%)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.margins(x=0.18, y=0.12)
    fig.suptitle(f"M1 summary [{split_label}]: accuracy against answer latency, one panel per bed (dashed: Pareto front within the bed; orange = A1 and A1q, blue = A2, gray = baselines)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fn = "summary_figure.png"
    fig.savefig(figures_dir / fn, dpi=200)
    plt.close(fig)
    return fn


def generate_plots(
    scoreboards: Dict[str, List[Dict[str, Any]]],
    out_dir: Path,
    split_label: str,
) -> List[str]:
    """Generate accuracy vs cost figures for each bed and one summary figure."""
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_figures: List[str] = []

    for bed, rows in scoreboards.items():
        if not rows:
            continue

        # 1. Accuracy vs Update Wall-Clock
        fig, ax = plt.subplots(figsize=(8, 5))
        pts_update = [(r["wall_mean"], r["acc_mean"], r["arm"]) for r in rows]
        for r in rows:
            arm = r["arm"]
            color = ARM_COLORS.get(arm, "#333333")
            ax.plot(
                r["wall_mean"],
                r["acc_mean"] * 100.0,
                marker="o",
                markersize=6,
                markeredgewidth=1.0,
                color=color,
                label=arm,
            )

        frontier_update = _pareto_frontier(pts_update)
        if len(frontier_update) >= 2:
            fx = [p[0] for p in frontier_update]
            fy = [p[1] * 100.0 for p in frontier_update]
            ax.plot(fx, fy, linestyle="--", color="#888888", linewidth=1.0, label="Pareto Front")

        pareto_arms = ", ".join(p[2] for p in frontier_update)
        ax.set_xlabel("Update Wall-Clock Total (seconds)")
        ax.set_ylabel("Final Accuracy (%)")
        ax.set_title(f"{bed} [{split_label}]: Accuracy vs Update Wall-Clock (Pareto: {pareto_arms})")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

        fn_update = f"fig_acc_vs_update_{bed}.png"
        path_update = figures_dir / fn_update
        fig.tight_layout()
        fig.savefig(path_update, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated_figures.append(f"figures/{fn_update}")

        # 2. Accuracy vs Answer Latency (batch 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        pts_ans = [(r["lat_b1_ms"], r["acc_mean"], r["arm"]) for r in rows if r["lat_b1_ms"] > 0]
        for r in rows:
            if r["lat_b1_ms"] <= 0:
                continue
            arm = r["arm"]
            color = ARM_COLORS.get(arm, "#333333")
            ax.plot(
                r["lat_b1_ms"],
                r["acc_mean"] * 100.0,
                marker="s",
                markersize=6,
                markeredgewidth=1.0,
                color=color,
                label=arm,
            )

        frontier_ans = _pareto_frontier(pts_ans)
        if len(frontier_ans) >= 2:
            fx = [p[0] for p in frontier_ans]
            fy = [p[1] * 100.0 for p in frontier_ans]
            ax.plot(fx, fy, linestyle="--", color="#888888", linewidth=1.0, label="Pareto Front")

        pareto_ans_arms = ", ".join(p[2] for p in frontier_ans)
        ax.set_xlabel("Answer Latency Batch 1 (ms)")
        ax.set_ylabel("Final Accuracy (%)")
        ax.set_title(f"{bed} [{split_label}]: Accuracy vs Answer Latency (Pareto: {pareto_ans_arms})")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)

        fn_ans = f"fig_acc_vs_answer_{bed}.png"
        path_ans = figures_dir / fn_ans
        fig.tight_layout()
        fig.savefig(path_ans, dpi=200, bbox_inches="tight")
        plt.close(fig)
        generated_figures.append(f"figures/{fn_ans}")

    # 3. Summary figure: one panel per bed. Accuracy on different beds is not comparable, so no front is drawn across beds.
    fn_summary = _summary_figure(scoreboards, split_label, figures_dir)
    if fn_summary:
        generated_figures.append(f"figures/{fn_summary}")

    return generated_figures


# -----------------------------------------------------------------------------
# Top 3 Cost Drivers & Reconciliation
# -----------------------------------------------------------------------------


def compute_cost_drivers(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute top 3 cost drivers for update and answer from pool records."""
    pool_recs = [r for r in records if r.get("arm") in ("A2", "A1") and r.get("pool_time_split")]
    update_drivers: List[Tuple[str, float, float]] = []
    if pool_recs:
        obs_times = [float(r["pool_time_split"].get("observe", 0.0)) for r in pool_recs]
        train_times = [float(r["pool_time_split"].get("train_expert", 0.0)) for r in pool_recs]
        refit_times = [float(r["pool_time_split"].get("router_refit", 0.0)) for r in pool_recs]

        mean_obs = float(np.mean(obs_times))
        mean_train = float(np.mean(train_times))
        mean_refit = float(np.mean(refit_times))
        mean_gate_bookkeep = max(0.0, mean_obs - mean_train)
        tot_update = mean_obs + mean_refit

        components = [
            ("Router refit & prediction routing", mean_refit),
            ("Gate scan & candidate clustering", mean_gate_bookkeep),
            ("Expert weight updates (train_expert)", mean_train),
        ]
        components.sort(key=lambda c: c[1], reverse=True)
        for name, dur in components:
            share = (dur / tot_update * 100.0) if tot_update > 0 else 0.0
            update_drivers.append((name, dur, share))

    # Answer cost drivers: how one answer's multiply-adds split between the gate scan and the expert, from the accounting blocks of the reference pool records (A2)
    answer_drivers: List[Dict[str, Any]] = []
    grouped = group_records(records)
    for bed in sorted(grouped):
        recs = [r for r in grouped[bed].get("A2", []) if r.get("accounting")]
        if not recs:
            continue
        gate = float(np.mean([r["accounting"]["gate_macs"] for r in recs]))
        expert = float(np.mean([r["accounting"]["expert_macs"] for r in recs]))
        measured = [r["accounting"]["gate_macs_measured_predict"] for r in recs if "gate_macs_measured_predict" in r["accounting"]]
        counted = [r["answer_cost"]["1"]["counted_macs"] for r in recs if r.get("answer_cost", {}).get("1")]
        answer_drivers.append({
            "bed": bed,
            "n_records": len(recs),
            "gate_macs": gate,
            "expert_macs": expert,
            "gate_share": (gate / (gate + expert)) if (gate + expert) > 0 else 0.0,
            "gate_macs_measured": float(np.mean(measured)) if measured else None,
            "counted_macs": float(np.mean(counted)) if counted else None,
        })

    return {
        "update_drivers": update_drivers,
        "answer_drivers": answer_drivers,
    }


def _reconciliation_beds(grouped: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> List[str]:
    """One bed per family: the largest B1 (most regimes), B2, and B3seq, when present."""
    beds: List[str] = []
    b1 = [b for b in grouped if b.startswith("B1")]
    if b1:
        beds.append(max(b1, key=lambda b: (int(b.split("_n")[-1]) if "_n" in b and b.split("_n")[-1].isdigit() else 0, b)))
    for fam in ("B2", "B3seq"):
        if fam in grouped:
            beds.append(fam)
    return beds


def compute_reconciliation_table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Set the counts of growmoe.accounting (recorded in each cell as ``accounting``) against the counts of costlab, for one bed per family.

    Nothing is re-derived here: a record without an ``accounting`` block gets a row that says so. The accounting count is static (the whole gate scan plus one expert);
    costlab counts what one answer really multiplied, so a ratio below one is expected when the pool's gate shortlist is on.
    """
    grouped = group_records(records)
    rows: List[Dict[str, Any]] = []
    for bed in _reconciliation_beds(grouped):
        bed_arms = grouped[bed]
        for arm in sorted(bed_arms, key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 999):
            recs = bed_arms[arm]
            with_acc = [r for r in recs if r.get("accounting")]
            same_cells = with_acc or recs                                                    # the ratio must compare the same cells: costlab over the cells that carry accounting
            costlab = [r["answer_cost"]["1"]["counted_macs"] for r in same_cells if r.get("answer_cost", {}).get("1")]
            row: Dict[str, Any] = {"bed": bed, "arm": arm, "n_records": len(recs), "n_with_accounting": len(with_acc),
                                   "costlab_macs": float(np.mean(costlab)) if costlab else None, "accounting_macs": None, "macs_ratio": None,
                                   "accounting_bytes": None, "costlab_active_bytes": None}
            if with_acc:
                acc = float(np.mean([r["accounting"]["predict_macs"] for r in with_acc]))
                row["accounting_macs"] = acc
                row["macs_ratio"] = (row["costlab_macs"] / acc) if (acc > 0 and row["costlab_macs"] is not None) else None
                row["accounting_bytes"] = float(np.mean([r["accounting"]["memory_bytes"] for r in with_acc]))
            active = [r["answer_cost"]["1"].get("active_bytes") for r in same_cells if r.get("answer_cost", {}).get("1")]
            row["costlab_active_bytes"] = float(np.mean(active)) if active else None
            rows.append(row)
    return rows

    bed_arms = grouped[first_bed]
    sorted_arms = sorted(bed_arms.keys(), key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 999)

    for arm in sorted_arms:
        recs = bed_arms[arm]
        r0 = recs[0]
        costlab_macs = r0.get("answer_cost", {}).get("1", {}).get("counted_macs", 0)
        costlab_bytes = r0.get("memory_bytes", 0)

        # Expected predict MACs from configuration
        cfg = r0.get("config", {})
        bed_info = r0.get("bed_info", {})
        d = bed_info.get("d", 128)
        classes = bed_info.get("classes", 10)
        hidden = cfg.get("hidden", 2048)

        # Theoretical weight MACs for MLP
        mlp_macs = hidden * (d + classes)

        if arm in ("A1", "A1q", "A2", "A1L"):
            experts = r0.get("describe", {}).get("experts", 1)
            gate_macs = experts * d
            accounting_macs = gate_macs + mlp_macs
            note = "gate signature scan + active expert MLP forward"
        elif arm in ("A3", "A3-2", "A4", "A4-2", "A8"):
            accounting_macs = mlp_macs
            note = "single network MLP forward pass"
        elif arm in ("A7", "A7-2", "A7all"):
            k = cfg.get("k", 5)
            accounting_macs = k * d
            note = "kNN memory cosine distance scan"
        elif arm == "A5a":
            accounting_macs = classes * d
            note = "Streaming LDA centroid score scan"
        elif arm == "A5b":
            accounting_macs = classes * d * d
            note = "Mahalanobis quadratic form scoring"
        elif arm == "A6":
            D_feat = cfg.get("D", 1024)
            accounting_macs = d * D_feat + D_feat * classes
            note = "Random projection + linear ridge head"
        else:
            accounting_macs = costlab_macs
            note = "baseline architecture"

        macs_ratio = (costlab_macs / accounting_macs) if accounting_macs > 0 else 1.0
        bytes_ratio = 1.0  # Memory bytes come directly from pool_report

        reconcil_rows.append({
            "arm": arm,
            "costlab_macs": costlab_macs,
            "accounting_macs": accounting_macs,
            "macs_ratio": macs_ratio,
            "costlab_bytes": costlab_bytes,
            "accounting_bytes": costlab_bytes,
            "bytes_ratio": bytes_ratio,
            "note": note,
        })

    return reconcil_rows


# -----------------------------------------------------------------------------
# Markdown Report Generation
# -----------------------------------------------------------------------------


def build_markdown_report(
    scoreboards: Dict[str, List[Dict[str, Any]]],
    rule_6_1_results: Dict[str, Dict[str, Any]],
    rule_6_2_results: Dict[str, Any],
    predictions: Dict[str, Dict[str, Any]],
    cost_drivers: Dict[str, Any],
    reconciliation: List[Dict[str, Any]],
    figure_paths: List[str],
    split_label: str,
) -> str:
    """Generate the full REPORT_M1.md markdown string."""
    lines: List[str] = []

    lines.append(f"# M1 Efficiency Benchmark Report [{split_label}]\n")
    lines.append(f"**Generated on**: `{split_label}` split evaluation records.")
    lines.append(f"**Record schema**: `{SCHEMA_VERSION}`.\n")
    lines.append("---\n")

    # First Page: Plain-English Summary
    lines.append("## Executive Summary\n")
    lines.append(
        "M1 measures the accuracy and computational cost of **Baseline v1** and its baselines across "
        "multiple benchmark beds using a unified cost meter (`costlab`). Every method is tuned on development seeds "
        "only, and evaluated under pinned thread affinity (logical CPUs 0 and 2, 2 threads).\n"
    )
    lines.append("### Summary Figure\n")
    summary_fig = next((f for f in figure_paths if "summary" in f), None)
    if summary_fig:
        lines.append(f"![M1 Summary Pareto Front]({summary_fig})\n")

    # High-level Verdicts Table
    lines.append("### Key Milestone Decisions & Predictions\n")
    lines.append("| Item | Description | Status | Details / Numbers |")
    lines.append("|:---|:---|:---:|:---|")
    for pred_id in sorted(predictions.keys()):
        p = predictions[pred_id]
        st = p.get("status", "UNKNOWN")
        reason = p.get("reason", "")
        if st == "NOT APPLICABLE":
            det = f"N/A ({reason})"
        else:
            nums = p.get("numbers", {})
            num_strs = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(nums.items())[:3]]
            det = ", ".join(num_strs)
        lines.append(f"| **{pred_id}** | {PREDICTION_TEXT.get(pred_id, pred_id)} | `{st}` | {det} |")
    lines.append("\n---\n")

    # Section 1: Scoreboard per Bed
    lines.append(f"## 1. Scoreboards [{split_label}]\n")
    for bed, rows in scoreboards.items():
        lines.append(f"### Bed: {bed} [{split_label}]\n")
        lines.append(
            "| Arm | Accuracy (mean+/-std [95% CI]) | Forgetting | Update Wall (s) | Update MACs | "
            "Ans Lat B1 (ms) | Ans MACs | Memory (B) | Experts | Operating Point |"
        )
        lines.append(
            "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|"
        )
        for r in rows:
            arm_name = f"{r['arm']}*" if r["uncounted"] else r["arm"]
            acc_str = _format_ci(r["acc_mean"], r["acc_ci_lo"], r["acc_ci_hi"], is_pct=True)
            forg_str = _format_ci(r["forg_mean"], r["forg_ci_lo"], r["forg_ci_hi"], is_pct=True) if r["forg_mean"] is not None else "N/A"
            wall_str = f"{r['wall_mean']:.2f} +/- {r['wall_std']:.2f}"
            up_macs_str = f"{r['update_macs']:,.0f}"
            ans_lat_str = f"{r['lat_b1_ms']:.2f} +/- {r['lat_b1_std_ms']:.2f}"
            ans_macs_str = f"{r['ans_macs']:,.0f}"
            mem_str = f"{r['memory_bytes']:,.0f}"
            exp_str = f"{r['experts']:.1f}"

            lines.append(
                f"| **{arm_name}** | {acc_str} | {forg_str} | {wall_str} | {up_macs_str} | "
                f"{ans_lat_str} | {ans_macs_str} | {mem_str} | {exp_str} | `{r['operating_point']}` |"
            )
        lines.append("\n*Arms marked with `*` carry uncounted_note (non-matrix-multiply operations).* \n")

    lines.append("---\n")

    # Section 2: Rule 6.1 Decisions
    lines.append(f"## 2. Pre-registration Rule 6.1 Decisions [{split_label}]\n")
    lines.append(
        "Rule 6.1 triggers when an eligible baseline is within 1 point (0.01) of pool accuracy at "
        "lower update cost AND lower answer cost.\n"
    )
    lines.append("| Bed | Pool | Reading | Fired? | Triggering Baselines | Pool Accuracy | Baseline Accuracy & Latency |")
    lines.append("|:---|:---:|:---:|:---:|:---|:---:|:---|")
    for bed, bed_res in rule_6_1_results.items():
        for key in ("R1_primary", "R1_secondary", "R2_primary", "R2_secondary"):
            r = bed_res[key]
            fired_str = "**YES**" if r.get("fired") else "NO"
            trigs = ", ".join(r.get("triggering_baselines", [])) if r.get("triggering_baselines") else "None"
            pool_acc = r.get("pool_metrics", {}).get("mean_accuracy")
            pool_acc_str = f"{pool_acc*100:.2f}%" if pool_acc is not None else "N/A"
            b_strs = []
            for b_arm in r.get("triggering_baselines", []):
                bm = r.get("baseline_metrics", {}).get(b_arm, {})
                b_strs.append(f"{b_arm}: {bm.get('mean_accuracy', 0)*100:.2f}%")
            b_det = "; ".join(b_strs) if b_strs else "-"
            lines.append(
                f"| {bed} | {r.get('reference_pool')} | {r.get('reading')} | {fired_str} | {trigs} | {pool_acc_str} | {b_det} |"
            )
    lines.append("\n---\n")

    # Section 3: Rule 6.2 Held Defaults Adoption
    lines.append(f"## 3. Pre-registration Rule 6.2 Held Defaults Adoption [{split_label}]\n")
    lines.append("| Held Default | Candidate vs Ref | Adopted? | Status | Decision Conditions & Supporting Evidence |")
    lines.append("|:---|:---|:---:|:---:|:---|")
    for name, r in rule_6_2_results.items():
        ad_str = "**ADOPT**" if r.get("adopted") else "REJECT"
        st = r.get("status", "UNKNOWN")
        det_parts = []
        if name == "lda_head":
            det_parts.append(f"gain_all={r.get('cond_gain_all')}")
            det_parts.append(f"ci_two={r.get('cond_ci_two')}")
            det_parts.append(f"worst_schedule_lat_rise={r.get('max_lat_rise', 0)*100:+.2f}% (max {THRESHOLD_LDA_HEAD_WALL_CLOCK_MAX_RISE*100:+.0f}%)")
        elif name == "growth":
            det_parts.append(f"acc_ok={r.get('cond_acc_not_drop')}")
            det_parts.append(f"wall_rel_change={r.get('wall_relative_change', 0)*100:+.2f}% (<= -15%)")
        elif name == "skip":
            det_parts.append(f"acc_ok={r.get('cond_acc_ok')}")
            det_parts.append(f"obs_drop_b2_b3={r.get('obs_drop_b2_or_b3')}")
        elif name == "chance":
            det_parts.append(f"gain_one_bed={r.get('gain_one_bed')}")
            det_parts.append(f"cost_ok_b2_b3={r.get('cost_ok_b2_b3')}")
        lines.append(f"| **{name}** | {HELD_DEFAULT_PAIRS.get(name)} | `{ad_str}` | `{st}` | {', '.join(det_parts)} |")
    lines.append("\n---\n")

    # Section 4: Top Three Cost Drivers
    lines.append(f"## 4. Top Three Cost Drivers [{split_label}]\n")
    lines.append("### Update Time Cost Drivers (from `pool_time_split`)\n")
    lines.append("| Rank | Component | Duration (s) | Share of Update Time |")
    lines.append("|:---:|:---|:---:|:---:|")
    for i, (name, dur, share) in enumerate(cost_drivers.get("update_drivers", []), 1):
        lines.append(f"| {i} | {name} | {dur:.4f} s | {share:.1f}% |")

    lines.append("\n### Answer Multiply-Add Split of the Reference Pool A2 (from the `accounting` block of each cell)\n")
    lines.append("| Bed | Cells | Gate scan MACs (static) | Expert MACs | Gate share (static) | Gate MACs measured per input | Counted MACs per answer (costlab) |")
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    answer_rows = cost_drivers.get("answer_drivers", [])
    for r in answer_rows:
        meas = f"{r['gate_macs_measured']:,.0f}" if r["gate_macs_measured"] is not None else "n/a"
        cnt = f"{r['counted_macs']:,.0f}" if r["counted_macs"] is not None else "n/a"
        lines.append(f"| {r['bed']} | {r['n_records']} | {r['gate_macs']:,.0f} | {r['expert_macs']:,.0f} | {r['gate_share'] * 100:.1f}% | {meas} | {cnt} |")
    if not answer_rows:
        lines.append("| not recorded | 0 | - | - | - | - | - |")
    lines.append("\n---\n")

    # Section 5: Reconciliation Table
    lines.append(f"## 5. Costlab vs Accounting Reconciliation Table [{split_label}]\n")
    lines.append("Accounting = `growmoe.accounting.pool_report` of the trained arm (recorded in the cell); costlab = multiply-adds counted while one input is answered. "
                 "Active bytes are the parameter bytes one answer touches (costlab), a different quantity from the stored bytes of the accounting. "
                 "Costlab figures are taken over the same cells as the accounting figures; a ratio above one means costlab counted more multiplies than the static model "
                 "(for example a naive per-class covariance product), below one means a shortlist skipped part of the scan.\n")
    lines.append("| Bed | Arm | Cells (with accounting) | Costlab MACs | Accounting MACs | Ratio | Accounting stored bytes | Costlab active bytes |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|")

    def _fmt(v: Optional[float], spec: str = ",.0f") -> str:
        return format(v, spec) if v is not None else "not recorded"

    for r in reconciliation:
        ratio = _fmt(r["macs_ratio"], ".3f")
        if r["accounting_macs"] == 0 and r["costlab_macs"]:
            ratio = "accounting counts 0"                                                   # growmoe.accounting has no model for this arm; costlab counts the multiplies
        lines.append(
            f"| {r['bed']} | **{r['arm']}** | {r['n_records']} ({r['n_with_accounting']}) | {_fmt(r['costlab_macs'])} | {_fmt(r['accounting_macs'])} | "
            f"{ratio} | {_fmt(r['accounting_bytes'])} | {_fmt(r['costlab_active_bytes'])} |"
        )
    lines.append("\n---\n")

    # Section 6: List of Generated Figures
    lines.append(f"## 6. Generated Figures [{split_label}]\n")
    for f in figure_paths:
        lines.append(f"- [{Path(f).name}]({f})")
    lines.append("\n")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main Report Pipeline
# -----------------------------------------------------------------------------


def generate_report(
    cells_dir: Path | str, out_dir: Path | str, split: Optional[str] = None
) -> Dict[str, Any]:
    """Execute the full report generation pipeline."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    records, split_label = load_records(cells_dir, requested_split=split)
    if not records:
        print(f"[REPORT] Nothing to report for {cells_dir}.")
        return {}
    print(f"[REPORT] Loaded {len(records)} valid records for split {split_label}")

    scoreboards = compute_scoreboard_rows(records)
    rule_6_1_res = evaluate_rule_6_1_all(records)
    rule_6_2_res = evaluate_rule_6_2_all(records)
    predictions_res = evaluate_all_predictions(records)
    cost_drivers_res = compute_cost_drivers(records)
    reconcil_res = compute_reconciliation_table(records)

    figure_paths = generate_plots(scoreboards, out_path, split_label)

    md_content = build_markdown_report(
        scoreboards=scoreboards,
        rule_6_1_results=rule_6_1_res,
        rule_6_2_results=rule_6_2_res,
        predictions=predictions_res,
        cost_drivers=cost_drivers_res,
        reconciliation=reconcil_res,
        figure_paths=figure_paths,
        split_label=split_label,
    )

    report_md_path = out_path / "REPORT_M1.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[REPORT] Wrote report to {report_md_path}")

    verdicts_json_path = out_path / "verdicts.json"
    verdicts_data = {
        "split": split_label,
        "rule_6_1": rule_6_1_res,
        "rule_6_2": rule_6_2_res,
        "predictions": predictions_res,
        "cost_drivers": cost_drivers_res,
        "reconciliation": reconcil_res,
        "figures": figure_paths,
    }
    with open(verdicts_json_path, "w", encoding="utf-8") as f:
        json.dump(verdicts_data, f, indent=2, cls=_CostlabJSONEncoder)
    print(f"[REPORT] Wrote verdicts to {verdicts_json_path}")

    return verdicts_data


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 Efficiency Benchmark Report Generator")
    parser.add_argument("--cells", required=True, help="Directory containing cell record JSONs")
    parser.add_argument("--out", required=True, help="Output directory for reports and figures")
    parser.add_argument("--split", default=None, choices=["dev", "report", "synthetic"], help="Split filter")
    args = parser.parse_args()

    generate_report(args.cells, args.out, split=args.split)


if __name__ == "__main__":
    main()
