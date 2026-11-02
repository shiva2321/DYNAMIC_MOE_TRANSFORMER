"""Summary figure of lab G6: ``python figure_topic.py --banking 100-109 --news 100-104`` writes results/routing_general/g6/FIGURE_G6.png."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import table_topic as TT  # noqa: E402

ARMS = [("emb", "pool today (sentence vector)", "#6b6b6b"), ("emb:p16", "pool today, smaller clusters (m_min 16)", "#9a9a9a"), ("both:c,e0.5", "reader + vector", "#2a6f97"),
        ("both:c,e0.5,p16", "reader + vector, m_min 16", "#1b4f72"), ("both:c,e0.5,s,p16", "shuffled verdicts, m_min 16 (control)", "#d98324"), ("key:c,p16", "reader only, m_min 16", "#b5651d")]
DOMAINS = [("banking", "banking77 (77 intents)"), ("agnews", "AG News (4 topics)"), ("newsgroups", "20 Newsgroups")]
ROWS = [("acc_all", "Accuracy, all four groups"), ("acc_novel", "Accuracy on the new class")]


def main(bank_seeds, news_seeds) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    y = np.arange(len(ARMS))[::-1]
    for ci, (domain, title) in enumerate(DOMAINS):
        seeds = bank_seeds if domain == "banking" else news_seeds
        for ri, (key, row_title) in enumerate(ROWS):
            ax = axes[ri][ci]
            ax.set_title(f"{title}: {row_title}" if ri == 0 else row_title, fontsize=9, loc="left")
            ax.grid(axis="x", color="#dddddd", linewidth=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for yy, (arm, label, colour) in zip(y, ARMS):
                rs = TT.load(domain, seeds, arm)
                if not rs:
                    continue
                v = np.array([TT.metric(r, key) for r in rs])
                ax.scatter(v, [yy] * len(v), s=14, color=colour, alpha=0.4, zorder=3)
                ax.plot([v.mean()] * 2, [yy - 0.3, yy + 0.3], color=colour, linewidth=3, zorder=4)
                ax.text(min(v.max(), 1.0) + 0.02, yy, f"{v.mean():.2f}", va="center", fontsize=8)
            ax.set_xlim(0, 1.12)
    for ri in range(2):
        axes[ri][0].set_yticks(y)
        axes[ri][0].set_yticklabels([a[1] for a in ARMS], fontsize=8)
    fig.text(0.01, 0.01, f"each dot is one seed (banking {bank_seeds[0]}-{bank_seeds[-1]}, news {news_seeds[0]}-{news_seeds[-1]}), the bar is the mean; same stream, test messages and pool in every row; only the key and the pool's smallest cluster differ", fontsize=8, color="#555555")
    fig.suptitle("The reader beside the sentence vector in topical domains: no reliable gain; the new class depends on the pool's smallest cluster, and a noisy key does best in the news domains", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = TT.OUT / "FIGURE_G6.png"
    fig.savefig(out, dpi=140)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--banking", default="100-109")
    ap.add_argument("--news", default="100-104")
    a = ap.parse_args()
    main(TT.seed_range(a.banking), TT.seed_range(a.news))
