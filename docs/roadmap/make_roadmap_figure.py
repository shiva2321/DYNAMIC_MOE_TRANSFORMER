"""The road plan figure (``python docs/roadmap/make_roadmap_figure.py``, run from the repo root): the road (milestones, probes, dependencies), how far the evidence has climbed
(the level table of STATE_BOARD.md) and where the effort goes (the ranges of MILESTONES.md). Every status and range is copied by hand from those two files: when they change, change
the lists below and rerun. Palette: slots 1-3 of the validated reference palette (blue, orange, aqua), neutral greys, text in ink colours; status is written in the cells, not colour alone."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent / "ROADMAP_FIGURE.png"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, GREY, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#8a8985", "#c9c7c1"
BLUE_T, AQUA_T = "#dce9f8", "#cdeee1"

W, H = 5.0, 3.6  # half width and half height of a milestone box
# (id, name, x, y, effort text); x and y on a 0-96 by 0-60 canvas
MILESTONES = {
    "M0": ("Foundations", 6, 30, "M: 3-4"),
    "M1": ("Efficiency\nscoreboard", 19, 30, "M: 3-5"),
    "M2": ("Standard text\nbenchmark", 35, 47, "M-L: 4-6"),
    "M3": ("Cheap novelty\nand nuance", 35, 30, "L: 6-10"),
    "M4": ("Router and pool\nat scale", 35, 13, "L-XL: 8-12"),
    "M5": ("Small language\nmodel", 54, 40, "XL: 12-25"),
    "M6": ("Other input\ntypes", 54, 20, "L: 6-10"),
    "M7": ("The complete\nmodel", 71, 30, "L-XL: 8-14"),
    "M8": ("Replication,\nrelease", 86, 30, "L: 5-9"),
}
PROBES = [("P1", "small-LM pool", "feeds M5", 6.5, 6), ("P2", "exact removal", "feeds M4", 16, 6), ("P3", "cascade cost", "feeds M3", 25.5, 6)]
PHASES = [("A  Safe and measured", 0, 26), ("B + C  Credible, cheap, bounded (side by side)", 26, 46), ("D + E  Own features,\nother inputs", 46, 62), ("F  Prove and share", 62, 96)]

ABILITIES = ["Routing without task labels", "Experts that learn and answer", "Novelty and nuance detection", "Continual learning without forgetting", "Knowledge lifecycle (dormant, remove, correct)",
             "Efficiency when answering", "Efficiency when updating", "Input types other than text"]
LEVELS = [("L1", "synthetic\nstreams"), ("L2", "real data,\nfair baselines"), ("L3", "standard\nbenchmark"), ("L4", "trained rep.\nand scale"), ("L5", "independent\nreplication")]
STATUS = [  # yes / part / no, in the order of ABILITIES x LEVELS (STATE_BOARD section 1)
    ["yes", "part", "no", "no", "part"], ["yes", "part", "no", "no", "no"], ["yes", "part", "no", "no", "no"], ["yes", "part", "no", "no", "part"],
    ["part", "no", "no", "no", "no"], ["part", "no", "no", "no", "no"], ["part", "no", "no", "no", "no"], ["part", "no", "no", "no", "no"]]
EFFORT = [("M0", 3, 4), ("M1", 3, 5), ("P1", 2, 3), ("P2", 1, 2), ("P3", 1, 1), ("M2", 4, 6), ("M3", 6, 10), ("M4", 8, 12), ("M5", 12, 25), ("M6", 6, 10), ("M7", 8, 14), ("M8", 5, 9)]


def arrow(ax, a, b, rad=0.0, color=GREY, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>", mutation_scale=11, linewidth=1.3, color=color, linestyle=ls, zorder=2))


def road(ax):
    ax.set_xlim(0, 96)
    ax.set_ylim(0, 60)
    ax.axis("off")
    for i, (name, x0, x1) in enumerate(PHASES):
        ax.add_patch(Rectangle((x0, 0), x1 - x0, 60, color="#f3f2ee" if i % 2 == 0 else "#ffffff", zorder=0))
        ax.text((x0 + x1) / 2, 58.8, name, ha="center", va="top", fontsize=8.8, color=INK2, fontweight="bold", linespacing=1.1)
    for mid, (name, x, y, eff) in MILESTONES.items():
        ax.add_patch(FancyBboxPatch((x - W, y - H), 2 * W, 2 * H, boxstyle="round,pad=0.25,rounding_size=1.2", facecolor=BLUE_T if mid != "M0" else "#fbd9ca",
                                    edgecolor=BLUE if mid != "M0" else ORANGE, linewidth=1.8, zorder=3))
        ax.text(x, y + 1.9, mid, ha="center", va="center", fontsize=11, color=INK, fontweight="bold", zorder=4)
        ax.text(x, y - 0.6, name, ha="center", va="center", fontsize=8, color=INK, zorder=4, linespacing=1.05)
        ax.text(x, y - H - 1.5, f"{eff} sessions", ha="center", va="top", fontsize=7.6, color=INK2, zorder=4)
    px = {k: (v[1], v[2]) for k, v in MILESTONES.items()}
    arrow(ax, (px["M0"][0] + W + 0.3, 30), (px["M1"][0] - W - 0.3, 30))
    for t in ("M2", "M3", "M4"):  # M1 fans out into three lanes that run side by side
        arrow(ax, (px["M1"][0] + W + 0.3, 30), (px[t][0] - W - 0.3, px[t][1]))
    jx = 44.2  # the join bar
    ax.plot([jx, jx], [13, 47], color=GREY, linewidth=1.8, zorder=2, solid_capstyle="round")
    for t in ("M2", "M3", "M4"):
        ax.plot([px[t][0] + W + 0.3, jx], [px[t][1]] * 2, color=GREY, linewidth=1.3, zorder=2)
    arrow(ax, (jx, 40), (px["M5"][0] - W - 0.3, 40))
    arrow(ax, (jx, 20), (px["M6"][0] - W - 0.3, 20))
    arrow(ax, (px["M5"][0] + W + 0.3, 40), (px["M7"][0] - W - 0.3, 32.6), rad=0.0)
    arrow(ax, (px["M6"][0] + W + 0.3, 20), (px["M7"][0] - W - 0.3, 27.4), rad=0.0)
    arrow(ax, (px["M7"][0] + W + 0.3, 30), (px["M8"][0] - W - 0.3, 30))
    ax.text(0.8, 14.4, "Early probes: run any time after the cost meter of M0", fontsize=8, color=INK2, va="bottom", style="italic")
    for pid, name, feeds, x, y in PROBES:
        ax.add_patch(FancyBboxPatch((x - 4.4, y - 3.3), 8.8, 6.6, boxstyle="round,pad=0.2,rounding_size=1.0", facecolor="#ffffff", edgecolor=GREY, linewidth=1.3, linestyle=(0, (3, 2)), zorder=3))
        ax.text(x, y + 1.9, pid, ha="center", va="center", fontsize=9.5, color=INK, fontweight="bold", zorder=4)
        ax.text(x, y + 0.0, name, ha="center", va="center", fontsize=7.4, color=INK2, zorder=4)
        ax.text(x, y - 1.7, feeds, ha="center", va="center", fontsize=7.4, color=BLUE, zorder=4)
    ax.add_patch(Polygon([[4.8, 38.9], [7.2, 38.9], [6, 36.2]], closed=True, color=ORANGE, zorder=5))
    ax.text(6, 39.7, "we are here", ha="center", va="bottom", fontsize=8.6, color=ORANGE, fontweight="bold", zorder=5)
    ax.text(79, 16, "Finish line: all of F1-F8 pass (M7),\nthen reproduced by someone else (M8)", ha="center", va="center", fontsize=8.4, color=INK2, style="italic")
    ax.text(0, -1.3, "Solid boxes: milestones with pass or fail tests. Dashed boxes: early probes (they measure, they have no pass bar). Arrows and the bar: what must come first. Effort in sessions (estimates).",
            fontsize=8, color=INK2, va="top")
    ax.set_title("A  The road: nine milestones and three early probes", loc="left", fontsize=11.5, color=INK, fontweight="bold")


def heat(ax):
    ax.set_xlim(-3.9, 5)
    ax.set_ylim(len(ABILITIES), -1.4)
    ax.axis("off")
    for j, (lev, desc) in enumerate(LEVELS):
        ax.text(j + 0.5, -1.15, lev, ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.text(j + 0.5, -0.12, desc, ha="center", va="bottom", fontsize=7.3, color=INK2, linespacing=1.05)
    for i, ab in enumerate(ABILITIES):
        ax.text(-0.1, i + 0.5, ab, ha="right", va="center", fontsize=8.6, color=INK)
        for j in range(5):
            s = STATUS[i][j]
            fc, ec, txt, hatch = {"yes": (AQUA, SURFACE, "yes", None), "part": (AQUA_T, AQUA, "part", "///"), "no": ("#f1f0ec", GRID, "no", None)}[s]
            ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, facecolor=fc, edgecolor=ec, hatch=hatch, linewidth=1.0))
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=8.6, color=INK if s != "no" else GREY, fontweight="bold" if s == "yes" else "normal")
    ax.set_title("B  What is there today: how far the evidence has climbed", loc="left", fontsize=11.5, color=INK, fontweight="bold")


def effort(ax):
    ys = list(range(len(EFFORT)))[::-1]
    for y, (mid, lo, hi) in zip(ys, EFFORT):
        probe = mid.startswith("P")
        ax.barh(y, hi - lo if hi > lo else 0.35, left=lo if hi > lo else lo - 0.17, height=0.56, color=LIGHT if probe else BLUE, zorder=3)
        ax.text(hi + 0.4, y, f"{lo}-{hi}" if hi > lo else f"{lo}", va="center", fontsize=8.2, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in EFFORT], fontsize=9)
    ax.set_xlim(0, 29)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.set_xlabel("sessions (low to high estimate)", color=INK2, fontsize=9)
    lo_tot, hi_tot = sum(e[1] for e in EFFORT), sum(e[2] for e in EFFORT)
    ax.text(28.6, ys[-1] - 0.1, f"all together: about {lo_tot}-{hi_tot} sessions", ha="right", va="bottom", fontsize=8.6, color=INK2)
    ax.set_title("C  Where the effort goes", loc="left", fontsize=11.5, color=INK, fontweight="bold")


def main():
    fig = plt.figure(figsize=(15, 11.4), facecolor=SURFACE)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.12, 1.0], width_ratios=[1.6, 1.0], left=0.035, right=0.985, top=0.955, bottom=0.115, hspace=0.30, wspace=0.14)
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(SURFACE)
    road(ax)
    axh = fig.add_subplot(gs[1, 0])
    axh.set_facecolor(SURFACE)
    heat(axh)
    axe = fig.add_subplot(gs[1, 1])
    axe.set_facecolor(SURFACE)
    effort(axe)
    fig.text(0.035, 0.014, "Road plan v0.1 (draft, 2026-11-02). Statuses from docs/roadmap/STATE_BOARD.md (evidence in the lab notebook through E-066); milestones and efforts from docs/roadmap/MILESTONES.md.\n"
             "yes = shown with grade A or B evidence and controls; part = shown in part or with a known limit; no = not tested. Efforts are estimates with wide error bars.", fontsize=7.8, color=INK2, va="bottom")
    fig.savefig(OUT, dpi=140, facecolor=SURFACE)
    print(OUT)


if __name__ == "__main__":
    main()
