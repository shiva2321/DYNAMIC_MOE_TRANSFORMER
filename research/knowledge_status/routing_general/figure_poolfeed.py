"""Summary figure of lab G5: ``python figure_poolfeed.py --seeds 100-109`` writes results/routing_general/g5/FIGURE_G5.png."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import table_poolfeed as TP  # noqa: E402

ARMS = [("emb", "pool today (sentence vector)", "#6b6b6b"), ("emb:c", "pool today + chance flag", "#9a9a9a"), ("key:c", "reader only", "#d98324"), ("both:c", "reader + vector", "#2a6f97"),
        ("both:c,e0.5", "reader + vector (tuned)", "#1b4f72"), ("both:c,e0.5,k0.95", "tuned + verdict cache", "#3c9d6b"), ("oracle:c", "perfect key (ceiling)", "#b5b5b5")]
PANELS = [("nmi", "Experts line up with skills (NMI)", 1.0), ("acc_taught_new", "Accuracy: known skills, new wordings", 1.0), ("acc_novel", "Accuracy: the new skill", 1.0), ("llm_texts", "Language-model passes per seed", None)]


def main(seeds) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), sharey=True)
    y = np.arange(len(ARMS))[::-1]
    for ax, (key, title, hi) in zip(axes, PANELS):
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(axis="x", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for yy, (arm, label, colour) in zip(y, ARMS):
            rs = TP.load(seeds, arm)
            if not rs:
                continue
            v = np.array([TP.metric(r, key) for r in rs])
            if key == "llm_texts":
                ax.barh(yy, v.mean(), height=0.6, color=colour)
                ax.text(v.mean() + 40, yy, f"{v.mean():.0f}", va="center", fontsize=8)
            else:
                ax.scatter(v, [yy] * len(v), s=14, color=colour, alpha=0.4, zorder=3)
                ax.plot([v.mean()] * 2, [yy - 0.3, yy + 0.3], color=colour, linewidth=3, zorder=4)
                ax.text(min(v.max(), 1.0) + 0.02, yy, f"{v.mean():.2f}", va="center", fontsize=8)
        if hi:
            ax.set_xlim(0, hi + 0.12)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([a[1] for a in ARMS])
    axes[3].set_xlim(0, 2600)
    fig.text(0.01, 0.01, f"each dot is one seed ({seeds[0]}-{seeds[-1]}), the bar is the mean; the pool, stream and test messages are the same in every row, only the key the pool sees differs", fontsize=8, color="#555555")
    fig.suptitle("A reader next to the sentence vector lifts the pool on every group; a reader alone lines up the experts but does not lift accuracy", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = TP.OUT / "FIGURE_G5.png"
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109")
    main(TP.seed_range(ap.parse_args().seeds))
