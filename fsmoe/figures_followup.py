"""The follow-up figure (``python -m fsmoe.figures_followup --seeds 710 ...``): execution outcomes on real code (A, B) and the new-skill learning path (C, D).
Same palette and rules as fsmoe/figures.py: first three validated slots, values written on the bars, bars from zero."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bed import ROOT
from .exec_traces import OUT as E_OUT
from .figures import AQUA, BLUE, INK, INK2, ORANGE, SURFACE, _bars, _style
from .learn_lab import ARMS, MODES, OUT as L_OUT


def build(seeds: Sequence[int], out: Path) -> Path:
    E = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := E_OUT / f"E_s{s}.json").exists()]
    Lr = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := L_OUT / f"L_s{s}.json").exists()]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=SURFACE)
    sigs = ["look", "cascade", "result_all", "fingerprint"]
    names = ["look\n(text only)", "look names one\nexpert, it judges", "every expert\njudges", "behaviour\nfingerprint"]
    ax = axes[0, 0]
    M = "1"
    near = [float(np.mean([r["by_m"][M][s]["auroc_near"] for r in E])) if E else float("nan") for s in sigs]
    lib = [float(np.mean([r["by_m"][M][s]["auroc_far"] for r in E])) if E else float("nan") for s in sigs]
    _bars(ax, names, [near, lib], ["against withheld near-twins (product, count_odd)", "against withheld real library functions"], [BLUE, ORANGE], ylim=(0, 1.1))
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.48, 0.512, "chance", ha="left", va="bottom", fontsize=8, color=INK2)
    _style(ax, "A  Running the code settles it: known functions in new styles\n    vs new functions, one execution list (6 steps)", "AUROC")
    ax = axes[0, 1]
    tpl = [float(np.mean([r["by_m"]["10"][s]["flag"]["known_ho_template"] for r in E])) if E else float("nan") for s in sigs[:3]]
    libf = [float(np.mean([r["by_m"]["10"][s]["flag"]["known_ho_library"] for r in E])) if E else float("nan") for s in sigs[:3]]
    _bars(ax, names[:3], [tpl, libf], ["known function, template style", "known function, real library idiom"], [BLUE, ORANGE], ylim=(0, 0.35))
    _style(ax, "B  Share of known functions wrongly called new\n    (stream cut, 5 % aimed)", "false flag")
    # C and D
    groups = ["oracle 10 %", "diverse novelty 10 %"]
    fast = [json.loads((L_OUT / f"L_fast_s{s}.json").read_text(encoding="utf-8")) for s in seeds if (L_OUT / f"L_fast_s{s}.json").exists()]

    def val(key, mode, policy, runs):
        v = [d[key] for r in runs for d in r["deploy"] if d["policy"] == policy and d["mode"] == mode]
        return float(np.mean(v)) if v else float("nan")

    ax = axes[1, 0]
    series = [[val("learned_fraction", m, p, Lr) for p, _ in ARMS] for m in ("interleaved", "focused_replay")] + [[val("learned_fraction", "interleaved", p, fast) for p, _ in ARMS]]
    _bars(ax, groups, series, ["pool, interleaved (as before)", "pool, new skill taught as its own batch + replay", "pool + one global closed-form head"], [BLUE, ORANGE, AQUA], ylim=(0, 1.1))
    _style(ax, "C  Share of the 28 never-taught skills learned from\n    the same budget of checked answers", "learned (accuracy >= 0.5)")
    ax = axes[1, 1]
    series = [[val("known_ho_final", m, p, Lr) for p, _ in ARMS] for m in ("interleaved", "focused_replay")] + [[val("known_ho_final", "interleaved", p, fast) for p, _ in ARMS]]
    _bars(ax, groups, series, ["pool, interleaved (as before)", "pool, focused + replay", "pool + global closed-form head"], [BLUE, ORANGE, AQUA], ylim=(0, 1.1))
    before = float(np.mean([d["known_ho_before"] for r in Lr for d in r["deploy"]])) if Lr else float("nan")
    ax.axhline(before, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(-0.48, before + 0.012, f"before the stream {before:.3f}", ha="left", va="bottom", fontsize=8, color=INK2)
    _style(ax, "D  Known skills in a new wording, after the stream", "accuracy")
    fig.suptitle("FS-MoE follow-up: execution outcomes on real code, and the new-skill learning path", x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.5, w_pad=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    print(build(a.seeds, ROOT / "results" / "fsmoe" / "FIGURE_FOLLOWUP.png"))
