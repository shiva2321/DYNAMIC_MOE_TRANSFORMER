"""The summary figure: four panels, one finding each, built from the generated JSON of the report seeds (``python -m fsmoe.figures --r-seeds ... --s-seeds ... --t-seeds ...``).

Colours are the first three slots of the validated reference palette (blue, orange, aqua: they validate all-pairs), text in ink colours, values written on the bars so colour never carries identity alone.
"""

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bed import ROOT
from .domains import NAMES
from .recog_lab import OUT as R_OUT
from .system_lab import EXTRA_POLICIES, OUT as S_OUT, POLICIES
from .traces import OUT as T_OUT

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def _load(pattern: str, base: Path):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(base.glob(pattern))]


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold", wrap=True)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _bars(ax, groups, series, labels, colors, ylim=(0, 1.05), width=0.8, legend=True, fmt="{:.2f}", fontsize=7.5):
    n = len(series)
    w = width / n
    top = ylim[1]
    for i, (vals, lab, c) in enumerate(zip(series, labels, colors)):
        x = np.arange(len(groups)) + (i - (n - 1) / 2) * w
        bars = ax.bar(x, vals, w * 0.92, color=c, label=lab)
        for xi, v in zip(x, vals):
            if v == v:
                ax.text(xi, v + 0.012 * top, fmt.format(v), ha="center", va="bottom", fontsize=fontsize, color=INK)
    ax.set_xticks(np.arange(len(groups)))
    ax.set_xticklabels(groups)
    ax.set_ylim(*ylim)
    if legend:
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1 if max(len(l) for l in labels) > 38 else n)


def build(r_seeds: Sequence[int], s_seeds: Sequence[int], t_seeds: Sequence[int], out: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=SURFACE)
    # A: can a look tell a known skill in a new wording from a new near-twin skill?
    ax = axes[0, 0]
    vals = {s: [] for s in ("maha", "disagree")}
    for d in NAMES:
        runs = [json.loads((R_OUT / f"R_s{s}_{d}_raw.json").read_text(encoding="utf-8")) for s in r_seeds if (R_OUT / f"R_s{s}_{d}_raw.json").exists()]
        for s in vals:
            vals[s].append(float(np.mean([r["signals"][s]["auroc"]["ho_vs_near"] for r in runs])) if runs else float("nan"))
    _bars(ax, ["arithmetic", "code", "language"], [vals["maha"], vals["disagree"]], ["distance to the taught skills (look)", "fast-slow disagreement"], [BLUE, ORANGE])
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.48, 0.512, "chance", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.axhline(0.95, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.48, 0.962, "bar for a usable detector (0.95)", ha="left", va="bottom", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.1)
    _style(ax, "A  A look cannot tell a new wording\n    from a new near-twin skill", "AUROC (1 = perfectly separable)")
    # B: does the next-state error separate them?
    ax = axes[0, 1]
    series = {s: [] for s in ("look", "result_nearest", "result_all")}
    noises = (0.05, 0.3, 1.0)
    for n in noises:
        runs = [json.loads(p.read_text(encoding="utf-8")) for s in t_seeds if (p := T_OUT / f"T_rec_s{s}_div6_n{n}.json").exists()]
        for s in series:
            series[s].append(float(np.mean([r["signals"][s]["auroc_known_ho_vs_withheld"] for r in runs])) if runs else float("nan"))
    _bars(ax, [f"noise {n}" for n in noises], [series["look"], series["result_nearest"], series["result_all"]],
          ["look", "look shortlists one expert, its next-state error judges", "every expert's next-state error"], [BLUE, ORANGE, AQUA])
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.48, 0.512, "chance", ha="left", va="bottom", fontsize=8, color=INK2)
    _style(ax, "B  Once the next state arrives, every expert's\n    prediction error can", "AUROC, held-out surface vs withheld regime")
    # C: learning new skills with a verification budget
    ax = axes[1, 0]
    runs = _load("S_s*.json", S_OUT)
    runs = [r for r in runs if r["seed"] in set(s_seeds)]
    extras = [json.loads((S_OUT / f"S_extra_s{r['seed']}.json").read_text(encoding="utf-8")) for r in runs if (S_OUT / f"S_extra_s{r['seed']}.json").exists()]
    names, vals = [], []
    for policy, b in POLICIES + EXTRA_POLICIES:
        post_hoc = (policy, b) in EXTRA_POLICIES
        v = [d["learned_fraction"] for r in (extras if post_hoc else runs) for d in r["deploy"] if d["policy"] == policy and abs(d["budget"] - b) < 1e-9]
        names.append("no\nverif." if policy == "none" else f"{'uncert.' if policy == 'uncertainty' else policy}\n{int(b * 100)} %" + ("*" if post_hoc else ""))
        vals.append(float(np.mean(v)) if v else float("nan"))
    _bars(ax, names, [vals], ["share of never-taught skills the pool ends up answering"], [BLUE], width=0.7)
    ax.legend().remove()
    ax.tick_params(axis="x", labelsize=7.5)
    _style(ax, "C  A small budget of checked answers: how many never-taught\n    skills the pool ends up answering (* added after the main readout)", "share of new skills learned (accuracy >= 0.5)")
    ax.tick_params(axis="x", labelsize=7.5)
    # D: accuracy on taught skills by system
    ax = axes[1, 1]
    systems = ("FS", "replay_eqmem", "knn_eqmem", "lda", "joint")
    tw, ho = [], []
    for s in systems:
        rs = [r["teaching"][s] for r in runs if s in r["teaching"]]
        tw.append(float(np.mean([r["mean_tw"] for r in rs])) if rs else float("nan"))
        ho.append(float(np.mean([r["mean_ho"] for r in rs])) if rs else float("nan"))
    _bars(ax, ["FS pool", "replay\nnet", "kNN", "LDA", "joint\n(ceiling)"], [tw, ho], ["taught wording", "held-out wording"], [BLUE, ORANGE], ylim=(0, 1.08))
    _style(ax, "D  Taught skills across the three domains,\n    in the taught and in a new wording", "accuracy (mean of 3 domains)")
    fig.suptitle("FS-MoE: what the recognizer, the result check and the system can and cannot do", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.5, w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-seeds", type=int, nargs="+", default=list(range(710, 720)))
    ap.add_argument("--s-seeds", type=int, nargs="+", default=list(range(710, 715)))
    ap.add_argument("--t-seeds", type=int, nargs="+", default=list(range(710, 715)))
    a = ap.parse_args()
    print(build(a.r_seeds, a.s_seeds, a.t_seeds, ROOT / "results" / "fsmoe" / "FIGURE_SUMMARY.png"))
