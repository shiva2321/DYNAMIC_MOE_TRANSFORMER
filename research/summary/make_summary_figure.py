"""One summary figure for research/ARCHITECTURE_FINDINGS_ONE_PAGE_2026-11-02.md (``python research/summary/make_summary_figure.py``, run from the repo root).

Panel B is recomputed from results/fsmoe/G/mixed_s71*_*.json (five seeds). The other numbers are copied from the lab notebook (docs/labnotebook/CLAIMS.md and
entries) and each one is tagged below with the claim or entry it comes from; they were not recomputed here. Palette: slots 1-3 of the validated reference palette
(blue, orange, aqua), ink colours for text, values written on the marks."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "SUMMARY_FIGURE_2026-11-02.png"
G_DIR = ROOT / "results" / "fsmoe" / "G"
SEEDS = range(710, 715)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985"


def _style(ax, title, xlabel="", ylabel=""):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def load_g(arm):
    return [json.loads((G_DIR / f"mixed_s{s}_{arm}.json").read_text(encoding="utf-8")) for s in SEEDS if (G_DIR / f"mixed_s{s}_{arm}.json").exists()]


def stats(arm):
    rs = load_g(arm)
    acc = np.array([r["mean_acc"] for r in rs])
    par = np.array([float(r["summary"]["params"]["total"]) for r in rs])
    return acc, par.mean() / 1e3


def ci95(v):
    v = np.asarray(v, dtype=float)
    return 2.776 * v.std(ddof=1) / np.sqrt(len(v))


def panel_a(ax):
    # C-76 (E-038, dev seeds): margin of the pool over the best same-memory single network with replay, 95 % intervals
    n = ["10 regimes", "30 regimes", "100 regimes"]
    gain, lo, hi = np.array([6.2, 9.1, 13.0]), np.array([4.5, 8.0, 11.7]), np.array([7.9, 10.3, 14.2])
    ax.bar(np.arange(3), gain, 0.55, color=BLUE, yerr=[gain - lo, hi - gain], error_kw=dict(ecolor=INK2, capsize=4, linewidth=1.1))
    for i, g in enumerate(gain):
        ax.text(i + 0.30, g + 0.15, f"+{g:.1f}", ha="left", va="bottom", fontsize=9.5, color=INK)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"{n[0]}\npool 43.0 vs single 36.8", f"{n[1]}", f"{n[2]}\npool 41.6 vs single 28.6"])
    ax.set_ylim(0, 17)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _style(ax, "A  Adding experts as regimes arrive:\n    gain over one network of the same memory", ylabel="points of accuracy (bars: 95 % interval)")


def panel_b(ax):
    fixed = [("F8", "8"), ("F16", "16"), ("F32", "32"), ("F64", "64")]
    xs = [stats(a)[1] for a, _ in fixed]
    ys = [stats(a)[0].mean() for a, _ in fixed]
    ax.plot(xs, ys, color=GREY, linewidth=1.5, zorder=1)
    offsets = {"F8": (-0.06, 0.006, "right", "bottom"), "F16": (-0.04, 0.004, "right", "bottom"), "F32": (0.03, -0.006, "left", "top"), "F64": (0.0, -0.007, "center", "top")}
    for (a, lab), x, y in zip(fixed, xs, ys):
        ax.scatter([x], [y], s=70, color=GREY, zorder=2, edgecolor=SURFACE, linewidth=1.2)
        dx, dy, ha, va = offsets[a]
        ax.annotate(f"fixed, {lab} units\n{y:.3f}", (x, y), xytext=(x * np.exp(dx), y + dy), fontsize=8.2, color=INK2, ha=ha, va=va)
    g, gp = stats("G")
    ax.errorbar([gp], [g.mean()], yerr=[ci95(g)], color=BLUE, capsize=3, linewidth=1.2, zorder=3)
    ax.scatter([gp], [g.mean()], s=140, color=BLUE, zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate(f"size chosen per expert at birth\n{g.mean():.3f} at {gp:.0f}k parameters", (gp, g.mean()), xytext=(gp * 1.05, g.mean() + 0.010), fontsize=8.6, color=BLUE, ha="left", va="bottom", fontweight="bold")
    n, npar = stats("GnoNB")
    ax.errorbar([npar], [n.mean()], yerr=[ci95(n)], color=ORANGE, capsize=3, linewidth=1.2, zorder=3)
    ax.scatter([npar], [n.mean()], s=110, color=ORANGE, marker="s", zorder=4, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate(f"start at 8 units, grow later\n{n.mean():.3f} at {npar:.0f}k", (npar, n.mean()), xytext=(npar * 1.03, n.mean() - 0.020), fontsize=8.4, color=ORANGE, ha="left", va="top")
    f32, f64 = stats("F32")[0], stats("F64")[0]
    d32, d64 = g - f32, g - f64
    ax.text(0.02, 0.98, f"paired by seed, governed minus fixed 32: {d32.mean():+.3f} (±{ci95(d32):.3f}, {int((d32 > 0).sum())}/{len(d32)} seeds)\ngoverned minus fixed 64: {d64.mean():+.3f} (±{ci95(d64):.3f}) at {100 * gp / stats('F64')[1]:.0f}% of its parameters\nbrute force (every expert big): {stats('BF')[0].mean():.3f} at {stats('BF')[1]:.0f}k, off the chart",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([30, 40, 60, 90])
    ax.set_xticklabels(["30", "40", "60", "90"])
    ax.minorticks_off()
    ax.set_xlim(26, 115)
    ax.set_ylim(0.63, 0.78)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    _style(ax, "B  Expert size: accuracy against parameters\n    (16 regimes, 5 seeds)", xlabel="total parameters (thousands, log scale; axis starts at 0.63)", ylabel="mean accuracy")


def panel_c(ax):
    # E-026 / C-51 (2 seeds): plain network, network + 20,000 stored examples, experts told the operator.
    # E-041 / C-87 (pre-registered, fresh seeds, 6 seeds): shipped pool 15.2, transfer rule + router 7.1
    labels = ["plain network", "network + 20,000\nstored examples", "pool as shipped", "pool with transfer rule\n+ router (adopted)", "experts told the operator\n(not a learner)"]
    vals = [55.1, 28.9, 15.2, 7.1, 0.0]
    cols = [GREY, GREY, BLUE, BLUE, AQUA]
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, 0.6, color=cols)
    for yi, v in zip(y, vals):
        ax.text(v + 0.8, yi, f"{v:.1f}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 65)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _style(ax, "C  Forgetting on operator-by-operator\n    arithmetic (lower is better)", xlabel="points of accuracy lost on earlier operators")


def panel_d(ax):
    # C-42 (E-025, dev seeds): 0.08 at 100 regimes, 0.20 at 10. C-25 / RESULTS_RV2B (fresh seeds): ratios 0.090 / 0.147 / 0.157.
    # C-170 + E-061: memory 0.78 (f10, f30), 0.99 (g), 0.86 (text bed). C-163 / E-060: (6.5 + 29.0) / 20.8. E-061: wall time 147 / 82 s on f30.
    # E-063: whole-run time saving at most about 7.5 %. E-025 (shootout, dev seeds, before the router): whole-run seconds, pool / equal-memory replay net
    # = 7/20, 22/99, 133/526 at 10 / 30 / 100 regimes.
    rows = [
        ("answer-time multiply-adds:\npool vs equal-memory single net", 0.08, "0.08 at 100 regimes, 0.20 at 10 (dev)"),
        ("gate multiply-adds:\nfingerprint shortlist vs exact gate", 0.09, "0.09 to 0.16, three streams; from 64 experts"),
        ("whole-run wall time: pool vs\nequal-memory single net", 133 / 526, "0.25 at 100 regimes; 0.22 to 0.35 over 10-100 (dev)"),
        ("memory: notebook-first pool\nvs pool born with 128 units", 0.78, "0.78 to 0.99 over four beds"),
        ("whole-run time with the skip flag\nvs without", 0.925, "0.93 at best (easy streams)"),
        ("training + governing multiply-adds:\ngoverned vs fixed 64 units", 35.5 / 20.8, "1.71 (governing alone is 4.5x training)"),
        ("wall time: notebook-first pool\nvs pool born with 128 units", 147 / 82, "1.79 (1.17 with two candidate sizes)"),
    ]
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], 0.6, color=[AQUA if r[1] < 1 else ORANGE for r in rows])
    ax.axvline(1.0, color=INK2, linewidth=1.1, linestyle=(0, (4, 3)))
    for yi, r in zip(y, rows):
        ax.text(r[1] + 0.03, yi, r[2], va="center", fontsize=8.2, color=INK, bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), zorder=5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 4.3)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    _style(ax, "D  Cost as a ratio to a stated reference\n    (left of the dashed line = cheaper)", xlabel="ratio (1.0 = same as the reference)")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.6), facecolor=SURFACE)
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.subplots_adjust(left=0.13, right=0.98, top=0.95, bottom=0.13, hspace=0.42, wspace=0.42)
    fig.text(0.01, 0.012,
             "Sources and grades. A: C-76, E-038, dev seeds, synthetic function-regime streams (B). B: E-060 report seeds 710-714, a lab world (B), recomputed from results/fsmoe/G. "
             "C: plain, replay and told-operator bars E-026 (2 seeds); pool bars E-041 pre-registered, fresh seeds (A).\n"
             "D: answer-time compute C-42 and whole-run time E-025 (dev, B), shortlist C-25 (fresh, A), memory E-061 (B), training and wall time E-060, E-061, skip flag E-063 (B). "
             "Different beds, seeds and grades: compare within a panel only. All runs are small networks on synthetic or sentence-embedding streams.",
             fontsize=7.6, color=INK2, va="bottom", ha="left", wrap=True)
    fig.savefig(OUT, dpi=140, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
