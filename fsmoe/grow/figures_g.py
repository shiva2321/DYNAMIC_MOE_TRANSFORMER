"""The growth figure (``python -m fsmoe.grow.figures_g --seeds 710 ...``): accuracy against memory, where the governor puts capacity, the other scenarios, and what governing costs. Palette slots 1-4 and
neutral grey, values on the bars, bars from zero."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fsmoe.bed import ROOT
from fsmoe.figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURFACE, _bars, _style
from fsmoe.figures_b import YELLOW

from .report_g import load, rows_by_need, total

GREY, LIGHT = "#8a8985", "#c9c7c1"


def _ci(vals):
    v = np.asarray(vals, dtype=float)
    return 0.0 if len(v) < 2 else 2.776 * v.std(ddof=1) / np.sqrt(len(v))


def build(seeds: Sequence[int], out: Path) -> Path:
    L = lambda sc, a: load(sc, a, seeds)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    ax = axes[0, 0]
    fixed = [("F8", "8"), ("F16", "16"), ("F32", "32"), ("F64", "64")]
    xs = [np.mean([total(r, "total") for r in L("mixed", a)]) / 1e3 for a, _ in fixed]
    ys = [np.mean([r["mean_acc"] for r in L("mixed", a)]) for a, _ in fixed]
    ax.plot(xs, ys, color=GREY, linewidth=1.5, zorder=1)
    for (a, lab), x, y in zip(fixed, xs, ys):
        ax.scatter([x], [y], s=70, color=GREY, zorder=2, edgecolor=SURFACE, linewidth=1.2)
        ax.annotate(f"fixed {lab}", (x, y), xytext=(x * 0.86, y + 0.006), fontsize=8.2, color=INK2)
    pts = [("F8d3", "fixed 8x3", GREY, "^"), ("Gng", "notebook-first, no growth", AQUA, "o"), ("GnoNB", "network at birth + governor", ORANGE, "o"), ("G", "governed", BLUE, "o"), ("BF", "brute force", YELLOW, "o")]
    handles = [matplotlib.lines.Line2D([], [], marker="o", color=GREY, linestyle="-", label="fixed experts, one size for all (8 / 16 / 32 / 64 units)")]
    for a, lab, c, mk in pts:
        rs = L("mixed", a)
        x, y = np.mean([total(r, "total") for r in rs]) / 1e3, np.mean([r["mean_acc"] for r in rs])
        x *= {"Gng": 0.97, "G": 1.03}.get(a, 1.0)                              # governed and notebook-first sit on the same point: nudged apart
        ax.errorbar([x], [y], yerr=[_ci([r["mean_acc"] for r in rs])], color=c, capsize=3, linewidth=1.2, zorder=3)
        ax.scatter([x], [y], s=130, color=c, marker=mk, zorder=4, edgecolor=SURFACE, linewidth=1.5)
        handles.append(matplotlib.lines.Line2D([], [], marker=mk, linestyle="", color=c, markersize=8, label=f"{lab}: {y:.3f} at {x:.0f}k parameters"))
    ax.set_xscale("log")
    ax.set_xticks([30, 50, 100, 200, 500])
    ax.set_xticklabels(["30", "50", "100", "200", "500"])
    ax.minorticks_off()
    ax.set_xlabel("total parameters (thousands, log scale)", color=INK2, fontsize=9)
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    _style(ax, "A  Accuracy against memory, mixed needs\n    (bars: 95 % interval over the 5 seeds)", "mean accuracy over 16 regimes")
    ax = axes[0, 1]
    cats = ["notebook only", "8 units", "wider (16+ units)", "deeper (2+ blocks)"]
    colors = [LIGHT, GREY, BLUE, ORANGE]
    need = rows_by_need(L("mixed", "G"))
    bottoms = np.zeros(3)
    kinds = ["width", "depth", "none"]
    cls = lambda x: 0 if x["hidden"] == 0 else (3 if x["depth"] >= 2 else (2 if x["hidden"] >= 16 else 1))
    shares = np.array([[np.mean([cls(x) == k for x in need[n]]) if need[n] else 0.0 for n in kinds] for k in range(4)])
    for k in range(4):
        ax.bar(np.arange(3), shares[k], 0.6, bottom=bottoms, color=colors[k], label=cats[k])
        for i in range(3):
            if shares[k][i] >= 0.06:
                ax.text(i, bottoms[i] + shares[k][i] / 2, f"{shares[k][i]:.2f}", ha="center", va="center", fontsize=8.5, color=INK)
        bottoms += shares[k]
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"needs width\n({len(need['width'])} regimes)", f"needs depth\n({len(need['depth'])})", f"needs nothing\n({len(need['none'])})"])
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    _style(ax, "B  What the governor gave each regime\n    (by what the offline oracle says it needs)", "share of regimes")
    ax = axes[1, 0]
    groups = ["easy\n(nothing needs growth)", "mixed", "big\n(48 regimes)"]
    series = [("F8", "fixed 8 units", GREY), ("Gnt", "governed, without the tier (big only)", AQUA), ("G", "governed (in big: with the tier)", BLUE)]
    vals = [[np.mean([r["mean_acc"] for r in L(sc, a)]) if L(sc, a) else float("nan") for sc in ("easy", "mixed", "big")] for a, _, _ in series]
    _bars(ax, groups, vals, [s[1] for s in series], [s[2] for s in series], ylim=(0, 1.0), legend=False, fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for _, l, c in series], frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    _style(ax, "C  Three worlds: the tier costs accuracy in the big one", "mean accuracy")
    ax = axes[1, 1]
    arms = [("F8", "fixed\n8"), ("F64", "fixed\n64"), ("Gng", "notebook-\nfirst"), ("GnoNB", "net at\nbirth"), ("G", "governed"), ("BF", "brute\nforce")]
    tr = [np.mean([r["summary"]["macs_train"] / 1e9 for r in L("mixed", a)]) for a, _ in arms]
    gv = [np.mean([r["summary"]["macs_trial"] / 1e9 for r in L("mixed", a)]) for a, _ in arms]
    xs = np.arange(len(arms))
    ax.bar(xs, tr, 0.6, color=BLUE, label="training")
    ax.bar(xs, gv, 0.6, bottom=tr, color=ORANGE, label="growth trials and promotion tests")
    for x, t, g in zip(xs, tr, gv):
        ax.text(x, t + g + 0.02 * max(np.array(tr) + np.array(gv)), f"{t + g:.0f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([n for _, n in arms], fontsize=8.5)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _style(ax, "D  What it cost to train and to govern, mixed\n    (billions of multiply-adds per run)", "G-MACs")
    fig.suptitle("FS-MoE growth: experts and a pool that grow as needed, one governor", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=7.5, w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    print(build(a.seeds, ROOT / "results" / "fsmoe" / "FIGURE_GROWTH.png"))
