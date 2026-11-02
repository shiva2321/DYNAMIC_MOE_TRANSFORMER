"""The end-to-end figure (``python -m fsmoe.figures_e2e --seeds 710 ...``): palette slots 1-5 of the validated reference palette (adjacent pairs validated), values on the bars, bars from zero."""

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bed import ROOT
from .domains import NAMES
from .e2e import ARMS
from .figures import BLUE, INK, INK2, ORANGE, SURFACE, AQUA, _bars, _style
from .report_x import load

YELLOW, MAGENTA = "#eda100", "#e87ba4"
COLORS = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
LABELS = ["A0 pool alone, 10 % labels", "A1 + global head", "A2 + tool on 10 % of code", "A3 + tool on all code", "A4 as A3, 20 % labels"]


def build(seeds: Sequence[int], out: Path) -> Path:
    runs = load(seeds)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    groups = ["arithmetic\n(3 new)", "code\n(5 new)", "language\n(20 new)", "all 28"]
    keys = ["arith", "code", "lang", "all"]
    mean = lambda arm, f: float(np.mean([f(r["arms"][arm]) for r in runs]))
    show = ["A0", "A1", "A3", "A4"]                                                    # A2 sits between A1 and A3 and is in the report; four arms keep the bars readable
    colors = [COLORS[[n for n, _ in ARMS].index(a)] for a in show]
    labels = [LABELS[[n for n, _ in ARMS].index(a)] for a in show]
    ax = axes[0, 0]
    _bars(ax, groups, [[mean(n, lambda a, k=k: a["after"]["learned"][k]) for k in keys] for n in show], labels, colors, ylim=(0, 1.12), legend=False, fontsize=7)
    _style(ax, "A  Never-taught skills learned, by domain", "learned (accuracy >= 0.5)")
    ax = axes[0, 1]
    before = float(np.mean([r["arms"]["A0"]["before"]["known_ho"]["all"] for r in runs]))
    _bars(ax, ["arithmetic", "code", "language", "all"], [[mean(n, lambda a, k=k: a["after"]["known_ho"][k]) for k in keys] for n in show], labels, colors, ylim=(0, 1.12), legend=False, fontsize=7)
    ax.axhline(before, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    _style(ax, f"B  Known skills in a new wording, after the stream\n    (dashed line: all, before the stream, {before:.3f})", "accuracy")
    handles = [matplotlib.patches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.505), ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1, 0]
    names = [n for n, _ in ARMS]
    _bars(ax, names, [[mean(n, lambda a: a["human_labels"]) for n in names], [mean(n, lambda a: a["tool_calls"]) for n in names]], ["human labels used", "tool calls (sandbox runs)"], [BLUE, ORANGE],
          ylim=(0, max(mean(n, lambda a: a["tool_calls"]) for n in names) * 1.15), fmt="{:.0f}")
    _style(ax, "C  What the checks cost: human labels against sandbox runs", "checks per stream")
    ax = axes[1, 1]
    tool_arms = [n for n, c in ARMS if c["tool"] is not None]
    _bars(ax, tool_arms, [[mean(n, lambda a: a["clusters_opened"]) for n in tool_arms], [5.0] * len(tool_arms)], ["classes the tool opened", "never-taught code skills"], [BLUE, ORANGE],
          ylim=(0, 8), fmt="{:.1f}")
    _style(ax, "D  The tool opens one class per new behaviour", "classes")
    fig.suptitle("FS-MoE end to end: the tool, the human channel and the fast memory in one stream", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=6.5, w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    print(build(a.seeds, ROOT / "results" / "fsmoe" / "FIGURE_E2E.png"))
