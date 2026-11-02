"""The notebook-first pool figure (``python -m fsmoe.figures_n --seeds 710 ...``): accuracy against memory, the parameters saved, regimes kept, and what judging costs. Palette slots 1-2 for the notebook-first
arms, greys for the shipped pool, values on the bars."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from .bed import ROOT
from .figures import AQUA, BLUE, INK, INK2, SURFACE, _bars, _style
from .report_n import ARMS, REGIMES, load

GREYS = {"U8": "#c9c7c1", "U32": "#a5a39d", "U128": "#6f6e6a"}
COLOR = {**GREYS, "NB": BLUE, "NBc": AQUA}
SHORT = {"U8": "shipped, 8 units", "U32": "shipped, 32 units", "U128": "shipped, 128 units (default)", "NB": "notebook-first", "NBc": "notebook-first, sizes 8 and 32"}


def _ci(vals):
    v = np.asarray(vals, dtype=float)
    return 0.0 if len(v) < 2 else 2.776 * v.std(ddof=1) / np.sqrt(len(v))


def build(seeds: Sequence[int], out: Path) -> Path:
    L = lambda bed, a: load(bed, a, seeds)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    ax = axes[0, 0]
    handles = []
    for a in ARMS:
        rs = L("f30", a)
        x, y = np.mean([r["bytes"] for r in rs]) / 1e6, np.mean([r["acc"] for r in rs])
        ax.errorbar([x], [y], yerr=[_ci([r["acc"] for r in rs])], color=COLOR[a], capsize=3, linewidth=1.2, zorder=2)
        ax.scatter([x], [y], s=130, color=COLOR[a], zorder=3, edgecolor=SURFACE, linewidth=1.5)
        handles.append(matplotlib.lines.Line2D([], [], marker="o", linestyle="", color=COLOR[a], markersize=8, label=f"{SHORT[a]}: {y:.3f} at {x:.1f} MB"))
    ax.set_xlabel("pool memory (MB)", color=INK2, fontsize=9)
    ax.legend(handles=handles, frameon=False, fontsize=8.2, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1)
    _style(ax, "A  Accuracy against memory, 30 function regimes\n    (bars: 95 % interval over the 5 seeds)", "mean accuracy")
    ax = axes[0, 1]
    beds = ["f10", "f30", "g", "t"]
    names = ["10 function\nregimes", "30 function\nregimes", "growth world\n(16)", "text bed\n(3 experts)"]
    series = [("U128", GREYS["U128"]), ("NB", BLUE)]
    vals = [[np.mean([r["params"] for r in L(b, a)]) / 1e3 for b in beds] for a, _ in series]
    _bars(ax, names, vals, [SHORT[a] for a, _ in series], [c for _, c in series], ylim=(0, max(max(v) for v in vals) * 1.15), legend=False, fmt="{:.0f}", fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=c, label=SHORT[a]) for a, c in series], frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    _style(ax, "B  Network parameters, thousands\n    (the notebook-first pool keeps almost none)", "network parameters (thousands)")
    ax = axes[1, 0]
    beds3 = ["f10", "f30", "g"]
    vals = [[np.mean([r["experts"] / REGIMES[b] for r in L(b, a)]) for b in beds3] for a in ARMS]
    _bars(ax, ["10 function regimes", "30 function regimes", "growth world (16)"], vals, [SHORT[a] for a in ARMS], [COLOR[a] for a in ARMS], ylim=(0, 1.2), legend=False, fmt="{:.2f}", fontsize=6.3)
    ax.legend(handles=[mpatches.Patch(color=COLOR[a], label=SHORT[a]) for a in ARMS], frameon=False, fontsize=8, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    _style(ax, "C  Experts per regime: small networks lose regimes\n    (1.0 = one expert for every regime)", "experts per regime")
    ax = axes[1, 1]
    arms = ["U128", "NB", "NBc"]
    secs = [np.mean([r["seconds"] for r in L("f30", a)]) / np.mean([r["seconds"] for r in L("f30", "U128")]) for a in arms]
    jud = [np.mean([r["judging_macs"] for r in L("f30", a)]) / 1e9 for a in arms]
    xs = np.arange(len(arms))
    ax.bar(xs, secs, 0.55, color=[COLOR[a] for a in arms])
    for x, s, j in zip(xs, secs, jud):
        ax.text(x, s + 0.02, f"{s:.2f} x", ha="center", va="bottom", fontsize=9, color=INK)
        ax.text(x, s / 2, f"judging\n{j:.0f} G-MACs" if j else "no judging", ha="center", va="center", fontsize=8.5, color=SURFACE if COLOR[arms[int(x)]] != GREYS["U8"] else INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT[a].replace(", sizes", ",\nsizes").replace(" (default)", "\n(default)") for a in arms], fontsize=8.5)
    ax.set_ylim(0, max(secs) * 1.2)
    _style(ax, "D  Wall time on the 30-regime stream, shipped pool = 1\n    (training candidate networks is the cost)", "relative wall time")
    fig.suptitle("FS-MoE notebook-first newborns in the real pool", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=7.5, w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    print(build(a.seeds, ROOT / "results" / "fsmoe" / "FIGURE_NBPOOL.png"))
