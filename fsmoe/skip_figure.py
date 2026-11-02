"""Summary figure of the skip flag checks (E-063): ``python -m fsmoe.skip_figure`` writes results/fsmoe/K/FIGURE_SKIP.png from the cells in results/fsmoe/K."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .skip_check import OUT

BEDS = [("f10", "function regimes (10)", (710, 711, 712, 713, 714)), ("f30", "function regimes (30)", (710, 711, 712, 713, 714)), ("bank", "banking77", (710, 711, 712)), ("t", "text teaching bed", (710, 711, 712, 713, 714)),
        ("umix", "UniMoE mixed", (500, 501, 502, 503, 504)), ("udrift", "UniMoE drifting", (500, 501, 502, 503, 504)), ("useq", "UniMoE sequential", (500, 501, 502, 503, 504))]
ARMS = (("k0.1", "#2a6f97", "0.1"), ("k0.25", "#d98324", "0.25"))


def cell(bed, s, arm):
    return json.loads((OUT / f"{bed}_s{s}_{arm}.json").read_text(encoding="utf-8"))


def rows_skipped(c):
    k = c["skip"]
    return k["skipped_rows"] / max(k["trained_rows"] + k["skipped_rows"], 1)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1]})
    y = np.arange(len(BEDS))[::-1]
    h = 0.34
    for ax in axes:
        ax.set_yticks(y)
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticklabels([b[1] for b in BEDS])
    for j, (arm, colour, label) in enumerate(ARMS):
        off = -h / 2 if j == 0 else h / 2
        for i, (bed, _, seeds) in enumerate(BEDS):
            ys = y[i] + off
            r = 100 * np.mean([rows_skipped(cell(bed, s, arm)) for s in seeds])
            axes[0].barh(ys, r, height=h * 0.9, color=colour, label=f"threshold {label}" if i == 0 else None)
            axes[0].text(r + 1, ys, f"{r:.0f}%", va="center", fontsize=8)
            gaps = [100 * (cell(bed, s, arm)["acc"] - cell(bed, s, "off")["acc"]) for s in seeds]
            axes[1].scatter(gaps, [ys] * len(gaps), s=14, color=colour, alpha=0.45, zorder=3)
            axes[1].plot([np.mean(gaps)] * 2, [ys - h * 0.45, ys + h * 0.45], color=colour, linewidth=2.5, zorder=4)
            ss = seeds[1:]                                                             # the first seed of each bed carries a cold start, see the report
            o_obs = np.mean([cell(bed, s, "off")["observe_seconds"] for s in ss])
            o_tr = np.mean([cell(bed, s, "off")["train_seconds"] for s in ss])
            tr = 100 * (1 - np.mean([cell(bed, s, arm)["train_seconds"] for s in ss]) / o_tr)
            ob = 100 * (1 - np.mean([cell(bed, s, arm)["observe_seconds"] for s in ss]) / o_obs)
            axes[2].barh(ys, tr, height=h * 0.9, color=colour, alpha=0.35)
            axes[2].barh(ys, ob, height=h * 0.9 * 0.45, color=colour)
    axes[0].set_title("Rows skipped (share of rows)", fontsize=10, loc="left")
    axes[1].set_title("Accuracy change against flag off (points)", fontsize=10, loc="left")
    axes[2].set_title("Wall time saved: training (light) and whole run (dark)", fontsize=10, loc="left")
    axes[1].axvline(0, color="#555555", linewidth=1)
    axes[0].set_xlim(0, 100)
    axes[1].set_xlim(-1.2, 1.2)
    axes[2].set_xlim(-5, 90)
    axes[2].axvline(0, color="#555555", linewidth=1)
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    axes[1].text(0.02, -0.14, "each dot is one seed, the bar is the mean", transform=axes[1].transAxes, fontsize=8, color="#555555")
    axes[2].text(0.02, -0.14, "exploratory: first seed of each bed left out (cold start)", transform=axes[2].transAxes, fontsize=8, color="#555555")
    fig.suptitle("The skip rule: it fires on easy streams, costs almost no accuracy, and saves little total time because training is a small part of the run", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "FIGURE_SKIP.png", dpi=140)
    print(OUT / "FIGURE_SKIP.png")


if __name__ == "__main__":
    main()
