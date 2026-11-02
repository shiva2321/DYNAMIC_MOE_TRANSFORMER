"""Scores the predictions G1-G9 of lab G5 (docs/superpowers/specs/2026-11-01-poolfeed-design.md) on the report seeds and writes results/routing_general/g5/SCORECARD_G5.md:
``python score_poolfeed.py --seeds 100-109``."""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import table_poolfeed as TP  # noqa: E402

FUSED, CACHE = "both:c,e0.5", "both:c,e0.5,k0.95"


def paired(seeds, arm, base, key):
    A, B = {r["seed"]: r for r in TP.load(seeds, arm)}, {r["seed"]: r for r in TP.load(seeds, base)}
    common = sorted(set(A) & set(B))
    d = np.array([TP.metric(A[s], key) - TP.metric(B[s], key) for s in common])
    return d, len(common)


def mean(seeds, arm, key):
    return float(np.mean([TP.metric(r, key) for r in TP.load(seeds, arm)]))


def build(seeds) -> str:
    rows = []
    n = len(TP.load(seeds, "emb"))
    d, k = paired(seeds, FUSED, "emb", "acc_taught_new")
    rows.append(("G1", f"`{FUSED}` beats `emb` on known skills in new wordings by at least +0.05 on average and in at least 7 of {k} seeds", d.mean() >= 0.05 and (d > 0).sum() >= 0.7 * k, f"{d.mean():+.3f}, better in {(d > 0).sum()}/{k} seeds"))
    d, k = paired(seeds, FUSED, "emb", "nmi")
    rows.append(("G2", "its NMI expert-skill is at least 0.15 above `emb` on average and above it in at least 7 seeds", d.mean() >= 0.15 and (d > 0).sum() >= 0.7 * k, f"{d.mean():+.3f}, better in {(d > 0).sum()}/{k} seeds"))
    d, k = paired(seeds, FUSED, "emb", "acc_all")
    rows.append(("G3", "its all-groups accuracy is at least 0.02 above `emb` on average", d.mean() >= 0.02, f"{d.mean():+.3f}, better in {(d > 0).sum()}/{k} seeds"))
    d, k = paired(seeds, "key:c", "emb", "acc_novel")
    rows.append(("G4", "`key:c` is at least 0.10 below `emb` on the new skill on average", d.mean() <= -0.10, f"{d.mean():+.3f}, better in {(d > 0).sum()}/{k} seeds"))
    cs, ref = TP.load(seeds, CACHE), TP.load(seeds, FUSED)
    cut = 1.0 - mean(seeds, CACHE, "llm_texts") / mean(seeds, FUSED, "llm_texts")
    wrong = mean(seeds, CACHE, "wrong") / mean(seeds, CACHE, "hits")
    da, _ = paired(seeds, CACHE, FUSED, "acc_all")
    dn, _ = paired(seeds, CACHE, FUSED, "nmi")
    rows.append(("G5", "the cache arm cuts language-model passes by 25-50 % against `both:c,e0.5`, at most 10 % of its hits are wrong, all-groups accuracy within 0.03 and NMI within 0.05", 0.25 <= cut <= 0.50 and wrong <= 0.10 and abs(da.mean()) <= 0.03 and abs(dn.mean()) <= 0.05,
                 f"passes cut {cut:.0%}, wrong hits {wrong:.1%}, accuracy {da.mean():+.3f}, NMI {dn.mean():+.3f}"))
    d, k = paired(seeds, "oracle:c", FUSED, "nmi")
    rows.append(("G6", "`oracle:c` has an NMI at least 0.10 above the fused arm", d.mean() >= 0.10, f"{d.mean():+.3f}, above in {(d > 0).sum()}/{k} seeds"))
    fs = mean(seeds, "emb", "false_spawns")
    rows.append(("G7", "`emb` spawns at most 0.3 experts per seed in phase 2 on average", fs <= 0.3, f"{fs:.2f} per seed"))
    de, _ = paired(seeds, "emb:c", "emb", "experts")
    da, _ = paired(seeds, "emb:c", "emb", "acc_all")
    rows.append(("G8", "the E-062 flag gives `emb:c` at least 0.8 more experts after phase 3 than `emb` without raising all-groups accuracy by more than 0.02", de.mean() >= 0.8 and da.mean() <= 0.02, f"experts {de.mean():+.2f}, accuracy {da.mean():+.3f}"))
    rs = TP.load(seeds, "key:c")
    ms = np.array([1000 * r["reader_seconds"] / r["reader_texts"] for r in rs if r["reader_seconds"] > 5])
    rows.append(("G9", "the reader costs between 25 and 60 ms per message (the seeds whose language-model passes were not already cached)", len(ms) > 0 and 25 <= ms.mean() <= 60, f"{ms.mean():.0f} ms (range {ms.min():.0f}-{ms.max():.0f}, {len(ms)} seeds)" if len(ms) else "no uncached seed"))
    held = sum(1 for r in rows if r[2])
    L = ["# Lab G5: predictions G1-G9 on the report seeds", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed** (seeds {seeds[0]}-{seeds[-1]}, {n} seeds).", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    return "\n".join(L) + "\n"


HEAD, SAFE, KCACHE = "both:c,e0.5,t0.9", "both:c,e0.5,t0.9,m0.8", "both:c,e0.5,k0.9"


def build_head(seeds) -> str:
    """H1-H7 (design note, addendum): the trained head against the uncached fused arm, the cache at 0.9 and the pool today."""
    base = mean(seeds, FUSED, "llm_texts")
    rows = []
    h_asked = mean(seeds, HEAD, "llm_texts")
    rows.append(("H1", "the main head arm uses at least 30 % fewer language-model passes than the unique-text count of the uncached fused arm (at most 1,444 per seed)", h_asked <= 1444, f"{h_asked:.0f} per seed ({1 - h_asked / base:.0%} fewer than {base:.0f})"))
    wrong = mean(seeds, HEAD, "wrong") / mean(seeds, HEAD, "hits")
    rows.append(("H2", "at most 7 % of the head's answers disagree with the full reader", wrong <= 0.07, f"{wrong:.1%}"))
    da, k = paired(seeds, HEAD, FUSED, "acc_all")
    dn, _ = paired(seeds, HEAD, FUSED, "nmi")
    rows.append(("H3", "its all-groups accuracy is within 0.03 and its NMI within 0.05 of the uncached fused arm on average", abs(da.mean()) <= 0.03 and abs(dn.mean()) <= 0.05, f"accuracy {da.mean():+.3f}, NMI {dn.mean():+.3f}"))
    kw = mean(seeds, KCACHE, "wrong") / mean(seeds, KCACHE, "hits")
    share = mean(seeds, HEAD, "hits") / mean(seeds, HEAD, "served")
    rows.append(("H4", "the head disagrees less often than the cache at 0.9 did on the report seeds while answering at least 50 % of the messages", wrong < kw and share >= 0.50, f"head {wrong:.1%} at {share:.0%} answered; cache {kw:.1%}"))
    sw = mean(seeds, SAFE, "wrong") / mean(seeds, SAFE, "hits")
    scut = 1 - mean(seeds, SAFE, "llm_texts") / base
    rows.append(("H5", "the safer arm (m0.8) has at most 4 % disagreements and cuts at least 20 % of the passes", sw <= 0.04 and scut >= 0.20, f"{sw:.1%} disagreements, {scut:.0%} fewer passes"))
    worst = paired(seeds, HEAD, FUSED, "acc_all")[0].min()
    rows.append(("H6", "in no seed is the main head arm more than 0.10 below the uncached fused arm in all-groups accuracy", worst >= -0.10, f"worst seed {worst:+.3f}"))
    d, k = paired(seeds, HEAD, "emb", "acc_all")
    rows.append(("H7", "the main head arm beats the pool today on all-groups accuracy in at least 9 of 10 seeds", (d > 0).sum() >= 9, f"{d.mean():+.3f}, better in {(d > 0).sum()}/{k} seeds"))
    held = sum(1 for r in rows if r[2])
    L = ["# Lab G5, the trained head: predictions H1-H7 on the report seeds", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed** (seeds {seeds[0]}-{seeds[-1]}).", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    arms = ["emb", FUSED, KCACHE, HEAD, SAFE]
    return "\n".join(L) + "\n\n" + TP.table(seeds, arms).replace("# Lab G5, the pool fed by different keys", "## The table of the report seeds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="100-109")
    ap.add_argument("--head", action="store_true", help="score H1-H7 of the trained head instead of G1-G9")
    a = ap.parse_args()
    seeds = TP.seed_range(a.seeds)
    if a.head:
        text = build_head(seeds)
        (TP.OUT / "SCORECARD_G5_head.md").write_text(text, encoding="utf-8")
        print(text)
        sys.exit(0)
    text = build(seeds)
    arms = ["emb", "emb:c", "key:c", "both:c", FUSED, CACHE, "oracle:c"]
    text += "\n" + TP.table(seeds, arms).replace("# Lab G5, the pool fed by different keys", "## The table of the report seeds")
    (TP.OUT / "SCORECARD_G5.md").write_text(text, encoding="utf-8")
    print(text)
