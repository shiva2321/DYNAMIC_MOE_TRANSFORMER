"""Paired statistics of lab G5 on the report seeds: per-seed differences against the pool as it is today, with a standard error and a sign test; writes results/routing_general/g5/STATS_G5.md.
``python stats_poolfeed.py --seeds 100-109 [--arms ...]``"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import table_poolfeed as TP  # noqa: E402

KEYS = (("acc_all", "accuracy, all four groups"), ("acc_taught_new", "known skills, new wordings"), ("acc_taught", "known skills, both wordings"), ("acc_novel", "the new skill"), ("nmi", "NMI expert-skill"))
GROUPS = ("taught_teach", "taught_new", "novel_teach", "novel_new")


def sign_p(k: int, n: int) -> float:
    """Two-sided exact sign test against 'as often better as worse' (ties dropped by the caller)."""
    tail = sum(math.comb(n, i) for i in range(max(k, n - k), n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def build(seeds, arms, base="emb") -> str:
    ref = {r["seed"]: r for r in TP.load(seeds, base)}
    L = [f"# Lab G5: paired differences against `{base}` (seeds {seeds[0]}-{seeds[-1]})", "", "Mean difference per seed, its standard error, the range over seeds, and the exact sign test (two-sided) on the seeds where the arm is better or worse.", ""]
    for key, name in KEYS:
        L += [f"## {name}", "", "| arm | mean diff | s.e. | min / max | better in | sign test p |", "|---|---|---|---|---|---|"]
        for arm in arms:
            if arm == base:
                continue
            rs = [r for r in TP.load(seeds, arm) if r["seed"] in ref]
            if not rs:
                continue
            d = np.array([TP.metric(r, key) - TP.metric(ref[r["seed"]], key) for r in rs])
            nz = d[d != 0]
            p = sign_p(int((nz > 0).sum()), len(nz)) if len(nz) else 1.0
            L.append(f"| {arm} | {d.mean():+.3f} | {d.std(ddof=1) / math.sqrt(len(d)):.3f} | {d.min():+.3f} / {d.max():+.3f} | {int((d > 0).sum())}/{len(d)} | {p:.3f} |")
        L.append("")
    L += ["## Accuracy per test group (mean over seeds)", "", "| arm | known skills, teaching wordings | known skills, new wordings | new skill, teaching wordings | new skill, new wordings |", "|---|---|---|---|---|"]
    for arm in [base] + [a for a in arms if a != base]:
        rs = TP.load(seeds, arm)
        if rs:
            L.append(f"| {arm} | " + " | ".join(f"{np.mean([r['final'][g] for r in rs]):.3f}" for g in GROUPS) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--arms", nargs="+", default=["emb:c", "key:c", "both:c", "both:c,e0.5", "both:c,e0.5,k0.95", "oracle:c"])
    ap.add_argument("--out", default="STATS_G5")
    a = ap.parse_args()
    text = build(TP.seed_range(a.seeds), a.arms)
    (TP.OUT / f"{a.out}.md").write_text(text, encoding="utf-8")
    print(text)
