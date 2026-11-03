"""V0 Merged Report Generator.

Merges several demo outputs (schedules, seeds) into one summary table and figure.
Enforces development seed restrictions (refuses report seeds 810-829).
Supports SYNTHETIC fixtures with explicit labeling.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from costlab.env import fixed_threads
from m1.report import ARM_COLORS

FORBIDDEN_REPORT_SEEDS = set(range(810, 830))
COMPARATOR_ORDER = ["A3", "A7", "A5a", "A5b", "A6", "A7all", "A8"]


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


def write_summary_figure(path: Path, records: List[Dict[str, Any]]) -> None:
    """Render summary figure comparing final metrics across merged runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    labels = [f"{r.get('schedule', '?')}_s{r.get('seed', '?')}" for r in records]
    x = np.arange(len(records))

    present_comps = [a for a in COMPARATOR_ORDER if any(a in r.get("arms", {}) for r in records)]
    all_arms = ["V0"] + present_comps
    k_arms = len(all_arms)
    bar_width = 0.8 / max(1, k_arms)

    # 1. Final accuracy
    for i, arm_id in enumerate(all_arms):
        accs = []
        for r in records:
            arm_data = r.get("arms", {}).get(arm_id)
            accs.append(arm_data["final_accuracy"] if arm_data else 0.0)
        offset = x + (i - (k_arms - 1) / 2.0) * bar_width
        c = "#1f77b4" if arm_id == "V0" else ARM_COLORS.get(arm_id, "#555555")
        ax1.bar(offset, accs, bar_width, label=arm_id, color=c)

    ax1.set_title("Final Accuracy by Run", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Accuracy", fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(fontsize=7)

    # 2. Final memory (MB)
    for i, arm_id in enumerate(all_arms):
        mems = []
        for r in records:
            arm_data = r.get("arms", {}).get(arm_id)
            mems.append((arm_data["final_memory_bytes"] / (1024 * 1024)) if arm_data else 0.0)
        offset = x + (i - (k_arms - 1) / 2.0) * bar_width
        c = "#1f77b4" if arm_id == "V0" else ARM_COLORS.get(arm_id, "#555555")
        ax2.bar(offset, mems, bar_width, label=arm_id, color=c)

    ax2.set_title("Final Memory Footprint (MB)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Memory (MB)", fontsize=9)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(fontsize=7)

    # 3. Novelty AUROC
    aucs = [r.get("arms", {}).get("V0", {}).get("auroc", {}).get("mean", 0.5) for r in records]
    ax3.bar(x, aucs, bar_width * 2.0, color="#ff7f0e", label="V0 Mean AUROC")
    ax3.axhline(0.5, color="red", linestyle="--", linewidth=1.2, label="Chance (0.50)")
    ax3.set_title("V0 Novelty AUROC (vs OOD)", fontsize=11)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax3.set_ylabel("AUROC", fontsize=9)
    ax3.set_ylim(0.0, 1.05)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    tmp_path = path.with_suffix(path.suffix + ".tmp.png")
    fig.savefig(tmp_path, dpi=120)
    plt.close(fig)
    tmp_path.replace(path)


def run_report(
    run_dirs: Sequence[str | Path],
    out_dir: str | Path,
    cpus: Optional[List[int]] = None,
    threads: int = 2,
) -> Dict[str, Any]:
    """Merge demo run records, enforce validation, and generate summary table and figure.

    Args:
        run_dirs: List of run output directories containing record.json.
        out_dir: Output directory for merged table and figure.
        cpus: Logical processors to pin.
        threads: Intra-op thread count.

    Returns:
        Summary dict of the report.

    Raises:
        FileNotFoundError: If record.json is missing in any directory.
        ValueError: If a report seed (810-829) is encountered.
    """
    cpus_list = cpus if cpus is not None else [8, 10]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with fixed_threads(n=threads, cpus=cpus_list):
        records: List[Dict[str, Any]] = []
        is_synthetic = False

        for r_path in run_dirs:
            rp = Path(r_path)
            rf = rp / "record.json" if rp.is_dir() else rp
            if not rf.exists():
                raise FileNotFoundError(f"Run directory {rp} is missing record.json: {rf}")

            rec = json.loads(rf.read_text(encoding="utf-8"))

            seed_val = rec.get("seed")
            if isinstance(seed_val, int) and seed_val in FORBIDDEN_REPORT_SEEDS:
                raise ValueError(
                    f"Report refused: run in {rp} uses frozen report seed {seed_val}. "
                    "Seeds 810-829 are strictly frozen report seeds and cannot be processed in development."
                )

            if (
                rec.get("synthetic") is True
                or str(seed_val).upper() == "SYNTHETIC"
                or str(rec.get("schedule")).upper() == "SYNTHETIC"
            ):
                is_synthetic = True

            records.append(rec)

        header_label = "SYNTHETIC" if is_synthetic else "DEV (development seeds only)"

        present_comps = [a for a in COMPARATOR_ORDER if any(a in r.get("arms", {}) for r in records)]

        # Build table.md
        header_cols = ["Schedule", "Seed", "V0 Acc"] + [f"{a} Acc" for a in present_comps] + [
            "V0 Experts",
            "V0 Mem (MB)",
            "Mean AUROC",
            "V0 Latency b1 (ms)",
        ]
        lines = [
            f"# {header_label} V0 Walking Skeleton Merged Report",
            "",
            "| " + " | ".join(header_cols) + " |",
            "|---|---|---|" + "|".join("---" for _ in present_comps) + "|---|---|---|---|",
        ]

        for r in records:
            sched = r.get("schedule", "unknown")
            s = r.get("seed", "unknown")
            arms_dict = r.get("arms", {})
            v0_a = arms_dict.get("V0", {})
            v0_acc = f"{v0_a.get('final_accuracy', 0.0):.2%}"
            comp_cells = [
                f"{arms_dict[a]['final_accuracy']:.2%}" if a in arms_dict else "-"
                for a in present_comps
            ]
            v0_exp = v0_a.get("final_experts", len(v0_a.get("per_regime_accuracy", [])))
            v0_mem = f"{v0_a.get('final_memory_bytes', 0) / (1024*1024):.2f}"
            mean_auc = (
                f"{v0_a.get('auroc', {}).get('mean', 0.5):.4f}"
                if "auroc" in v0_a
                else "-"
            )
            lat = (
                v0_a.get("answer_latency", {})
                .get("1", {})
                .get("latency_median_s", 0.0)
                * 1000.0
            )
            lat_str = f"{lat:.2f}"

            row_cells = [sched, str(s), v0_acc] + comp_cells + [str(v0_exp), v0_mem, mean_auc, lat_str]
            lines.append("| " + " | ".join(row_cells) + " |")

        lines.extend([
            "",
            "### Component Register Reference",
            "",
            "V0 is assembled from:",
            "- **Encoder:** MiniLM through `unimoe/frontends.py` (cached offline, d=128)",
            "- **Pool:** `UniPool` with tuned operating point `A1q` (inner_steps=50, q=0.05)",
            "- **Head:** `ULDA` (running LDA head inside every expert)",
            "- **Unknown Flag:** Conformal p-value and density flag (evaluated via AUROC vs OOD text)",
            "- **Switched Off:** Asking reader, vertical growth, newborns, skip/chance flags",
            "",
            "> [!NOTE]",
            "> All runs above are demonstrations on development seeds. No claim of superiority is made.",
        ])

        tbl_file = out_path / "table.md"
        fig_file = out_path / "figure.png"

        atomic_write_text(tbl_file, "\n".join(lines))
        write_summary_figure(fig_file, records)

        return {
            "label": header_label,
            "n_runs": len(records),
            "table_path": str(tbl_file),
            "figure_path": str(fig_file),
        }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V0 Merged Report Generator.")
    parser.add_argument("--runs", nargs="+", required=True, help="List of run output directories.")
    parser.add_argument("--out", type=str, required=True, help="Output directory for merged report.")
    parser.add_argument("--cpus", type=str, default="8,10", help="CPUs to pin.")
    parser.add_argument("--threads", type=int, default=2, help="Thread count.")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    cpus = [int(c.strip()) for c in args.cpus.split(",")] if args.cpus else [8, 10]
    run_report(
        run_dirs=args.runs,
        out_dir=args.out,
        cpus=cpus,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
