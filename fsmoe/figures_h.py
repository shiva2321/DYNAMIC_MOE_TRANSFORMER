"""The hybrid figure (``python -m fsmoe.figures_h --seeds 710 ...``): D1 against the hybrid grid, and where every arm sits between arithmetic and code. Palette slots 1-4, values on the bars, bars from zero."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .bed import ROOT
from .domains import NAMES
from .figures import AQUA, BLUE, GRID, INK, INK2, ORANGE, SURFACE, _bars, _style
from .figures_b import YELLOW
from .report_h import arms

COLORS = {"D1r": ORANGE, "H4": BLUE, "H8": AQUA, "H12": YELLOW}
SHOWN = ("D1r", "H4", "H8", "H12")
NAME = {"D1r": "D1 (equal shares, code throttled)", "H4": "H4 (4 new-class items per chunk)", "H8": "H8 (8 per chunk)", "H12": "H12 (12 per chunk)"}


def build(seeds: Sequence[int], out: Path) -> Path:
    A = arms(seeds)
    mean = lambda arm, f: float(np.mean([f(x) for x in A[arm]]))
    keys = ["arith", "code", "lang", "all"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    ax = axes[0, 0]
    _bars(ax, ["arithmetic\n(3 new)", "code\n(5 new)", "language\n(20 new)", "all 28"], [[mean(a, lambda x, k=k: x["after"]["learned"][k]) for k in keys] for a in SHOWN], [NAME[a] for a in SHOWN],
          [COLORS[a] for a in SHOWN], ylim=(0, 1.12), legend=False, fontsize=6.3)
    _style(ax, "A  Never-taught skills learned, by domain", "learned (accuracy >= 0.5)")
    ax = axes[0, 1]
    before = mean("A3b", lambda x: x["before"]["known_ho"]["all"])
    _bars(ax, ["arithmetic", "code", "language", "all"], [[mean(a, lambda x, k=k: x["after"]["known_ho"][k]) for k in keys] for a in SHOWN], [NAME[a] for a in SHOWN], [COLORS[a] for a in SHOWN],
          ylim=(0, 1.12), legend=False, fontsize=6.3)
    ax.axhline(before, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    _style(ax, f"B  Known skills in a new wording, after the stream\n    (dashed line: all, before the stream, {before:.3f})", "accuracy")
    fig.legend(handles=[mpatches.Patch(color=COLORS[a], label=NAME[a]) for a in SHOWN], loc="center", bbox_to_anchor=(0.5, 0.505), ncol=2, frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1, 0]
    ax.add_patch(mpatches.Rectangle((0.5, 0.9), 0.55, 0.15, facecolor=GRID, edgecolor="none", alpha=0.7, zorder=0))
    ax.text(0.52, 0.915, "target zone of H1:\narithmetic >= 0.50, code >= 0.90", fontsize=8, color=INK2, va="bottom")
    pts = [("A3b", "arrival order (E-058)", INK2, "s"), ("D2", "D2 (E-058)", INK2, "^")] + [(a, NAME[a].split(" (")[0].replace("D1r", "D1") if a == "D1r" else NAME[a], COLORS[a], "o") for a in SHOWN]
    short = {"A3b": "A3b", "D2": "D2", "D1r": "D1", "H4": "H4", "H8": "H8", "H12": "H12"}
    offs = {"A3b": (-0.075, 0.012), "D2": (0.015, -0.032), "D1r": (0.015, 0.012), "H4": (-0.02, 0.016), "H8": (0.0, 0.016), "H12": (-0.02, 0.016)}
    handles = []
    for a, lab, c, mk in pts:
        x, y = mean(a, lambda r: r["after"]["learned"]["arith"]), mean(a, lambda r: r["after"]["learned"]["code"])
        ax.scatter([x], [y], s=120, color=c, marker=mk, zorder=3, edgecolor=SURFACE, linewidth=1.5)
        ax.annotate(short[a], (x, y), xytext=(x + offs[a][0], y + offs[a][1]), fontsize=8.5, color=INK)
        handles.append(matplotlib.lines.Line2D([], [], marker=mk, linestyle="", color=c, markersize=8, label=f"{lab}: arithmetic {x:.2f}, code {y:.2f}"))
    ax.legend(handles=handles, frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.7, 1.06)
    ax.set_xlabel("arithmetic skills learned (of 3)", color=INK2, fontsize=9)
    _style(ax, "C  Where each arm sits: arithmetic against code\n    (grey: the E-058 arms of the same seeds)", "code skills learned (of 5)")
    ax = axes[1, 1]
    new = [mean(a, lambda x: x["new_trained_by_domain"].get("code", 0)) for a in SHOWN]
    tot = [mean(a, lambda x: x["trained_by_domain"].get("code", 0)) for a in SHOWN]
    known = [t - n for t, n in zip(tot, new)]
    xs = np.arange(len(SHOWN))
    ax.bar(xs, new, 0.6, color=ORANGE, label="items of never-taught code classes")
    ax.bar(xs, known, 0.6, bottom=new, color="#c9c7c1", label="items of known code skills")
    for x, n, k in zip(xs, new, known):
        ax.text(x, n / 2, f"{n:.0f}", ha="center", va="center", fontsize=9, color=INK)
        ax.text(x, n + k + 0.012 * max(tot) * 1.0, f"{n + k:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(["D1", "H4", "H8", "H12"])
    ax.set_ylim(0, max(tot) * 1.12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    _style(ax, "D  Code items trained per stream\n    (D1 uses 85 of the 358 new-class ones)", "fresh items trained")
    fig.suptitle("FS-MoE hybrid: never-taught classes exempt from the domain throttle", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=6.5, w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    print(build(a.seeds, ROOT / "results" / "fsmoe" / "FIGURE_HYBRID.png"))
