"""The domain-balance figure (``python -m fsmoe.figures_b --seeds 710 ...``): arrival order against the two balanced chunkings and the uniform-prior head. Palette slots 1-4, values on the bars, bars from zero."""

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
from .e2e import B_OUT, OUT as X_OUT
from .figures import AQUA, BLUE, INK, ORANGE, SURFACE, _bars, _style

YELLOW = "#eda100"


def build(seeds: Sequence[int], out: Path) -> Path:
    runs = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := B_OUT / f"B_s{s}.json").exists()]
    pri = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := B_OUT / f"B_priors_s{s}.json").exists()]
    extra = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := X_OUT / f"X_extra_s{s}.json").exists()]
    arms = [("A3b arrival order (E-057 A3)", runs, "A3b", BLUE), ("D1 equal shares, surplus unused", runs, "D1", ORANGE), ("D2 equal shares, scarce replayed", runs, "D2", AQUA),
            ("A3u arrival order, uniform priors in the head", pri, "A3u", YELLOW)]
    mean = lambda src, arm, f: float(np.mean([f(r["arms"][arm]) for r in src]))
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5), facecolor=SURFACE)
    keys = ["arith", "code", "lang", "all"]
    ax = axes[0, 0]
    _bars(ax, ["arithmetic\n(3 new)", "code\n(5 new)", "language\n(20 new)", "all 28"], [[mean(s, a, lambda x, k=k: x["after"]["learned"][k]) for k in keys] for _, s, a, _c in arms], [l for l, *_ in arms],
          [c for *_, c in arms], ylim=(0, 1.12), legend=False, fontsize=7)
    _style(ax, "A  Never-taught skills learned, by domain", "learned (accuracy >= 0.5)")
    ax = axes[0, 1]
    before = float(np.mean([r["arms"]["A3b"]["before"]["known_ho"]["all"] for r in runs]))
    _bars(ax, ["arithmetic", "code", "language", "all"], [[mean(s, a, lambda x, k=k: x["after"]["known_ho"][k]) for k in keys] for _, s, a, _c in arms], [l for l, *_ in arms],
          [c for *_, c in arms], ylim=(0, 1.12), legend=False, fontsize=7)
    ax.axhline(before, color="#52514e", linewidth=1, linestyle=(0, (4, 3)))
    _style(ax, f"B  Known skills in a new wording, after the stream\n    (dashed line: all, before the stream, {before:.3f})", "accuracy")
    handles = [matplotlib.patches.Patch(color=c, label=l) for l, _s, _a, c in arms]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.505), ncol=2, frameon=False, fontsize=9, labelcolor="#52514e")
    ax = axes[1, 0]
    a3 = {"arith": float(np.mean([r["arms"]["A3r"]["human_by_domain"]["arith"] for r in extra])) if extra else float("nan"),
          "code": float(np.mean([r["arms"]["A3r"]["tool_known_trained"] + r["arms"]["A3r"]["tool_new"] for r in extra])) if extra else float("nan"),
          "lang": float(np.mean([r["arms"]["A3r"]["human_by_domain"]["lang"] for r in extra])) if extra else float("nan")}
    sets = [[a3[d] for d in NAMES]] + [[mean(runs, arm, lambda x, d=d: x["trained_by_domain"].get(d, 0)) for d in NAMES] for arm in ("D1", "D2")]
    _bars(ax, ["arithmetic", "code", "language"], sets, ["arrival order (A3b)", "D1", "D2"], [BLUE, ORANGE, AQUA], ylim=(0, 2600), fmt="{:.0f}", legend=False)
    _style(ax, "C  Items each domain trains the pool with (fresh items;\n    D2's scarce domains are also replayed)", "items per stream")
    ax = axes[1, 1]
    _bars(ax, [l.split(" ")[0] for l, *_ in arms], [[mean(s, a, lambda x: x["after"]["learned"]["all"]) for _, s, a, _c in arms], [mean(s, a, lambda x: x["after"]["known_ho"]["all"]) for _, s, a, _c in arms]],
          ["never-taught skills learned (all 28)", "known skills in a new wording (all)"], [BLUE, ORANGE], ylim=(0, 1.12), legend=False)
    _style(ax, "D  No arm wins on both: the headline stays about 0.35", "")
    handles = [matplotlib.patches.Patch(color=c, label=l) for l, c in (("arrival order (A3b)", BLUE), ("D1", ORANGE), ("D2", AQUA))]
    axes[1, 0].legend(handles=handles, frameon=False, fontsize=8.5, labelcolor="#52514e", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    handles = [matplotlib.patches.Patch(color=c, label=l) for l, c in (("never-taught skills learned (all 28)", BLUE), ("known skills in a new wording (all)", ORANGE))]
    axes[1, 1].legend(handles=handles, frameon=False, fontsize=8.5, labelcolor="#52514e", loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    fig.suptitle("FS-MoE domain-balanced training: equal domain shares in every chunk, and the head's class priors", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=6.5, w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    print(build(a.seeds, ROOT / "results" / "fsmoe" / "FIGURE_BALANCE.png"))
