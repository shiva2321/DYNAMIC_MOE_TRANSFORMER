"""Scores the predictions T1-T8 of lab G6 (docs/superpowers/specs/2026-11-01-poolfeed-topic-design.md) and writes results/routing_general/g6/SCORECARD_G6.md:
``python score_topic.py --banking 100-109 --news 100-104``."""

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import table_topic as TT  # noqa: E402

FUSED, FUSED16, HEAD = "both:c,e0.5", "both:c,e0.5,p16", "both:c,e0.5,t0.9,m0.8,p16"
ARMS = ["emb", FUSED, "emb:p16", "key:c,p16", FUSED16, "both:c,e0.5,s,p16", HEAD]


def mean(domain, seeds, arm, key):
    rs = TT.load(domain, seeds, arm)
    return float(np.mean([TT.metric(r, key) for r in rs])) if rs else float("nan")


def paired(domain, seeds, arm, base, key):
    A, B = {r["seed"]: r for r in TT.load(domain, seeds, arm)}, {r["seed"]: r for r in TT.load(domain, seeds, base)}
    common = sorted(set(A) & set(B))
    return np.array([TT.metric(A[s], key) - TT.metric(B[s], key) for s in common]), len(common)


def build(bank_seeds, news_seeds) -> str:
    news = ("agnews", "newsgroups")
    rows = []
    v = {d: (mean(d, news_seeds, "emb", "acc_novel"), mean(d, news_seeds, FUSED, "acc_novel")) for d in news}
    rows.append(("T1", "default recipe, news domains: the new-class accuracy of `emb` and of `both:c,e0.5` is at most 0.25 in both domains", all(max(x) <= 0.25 for x in v.values()), "; ".join(f"{d}: emb {x[0]:.3f}, fused {x[1]:.3f}" for d, x in v.items())))
    g = {d: paired(d, news_seeds, "emb:p16", "emb", "acc_novel")[0].mean() for d in news}
    rows.append(("T2", "`emb:p16` raises the new-class accuracy over default `emb` by at least 0.20 in AG News and 0.10 in 20 Newsgroups", g["agnews"] >= 0.20 and g["newsgroups"] >= 0.10, "; ".join(f"{d}: {x:+.3f}" for d, x in g.items())))
    g = {d: paired(d, news_seeds, FUSED16, "emb:p16", "acc_all")[0] for d in news}
    rows.append(("T3", "news domains: `both:c,e0.5,p16` within 0.08 of `emb:p16` on all four groups", all(abs(x.mean()) <= 0.08 for x in g.values()), "; ".join(f"{d}: {x.mean():+.3f} ({int((x > 0).sum())}/{len(x)} seeds better)" for d, x in g.items())))
    d4, k = paired("banking", bank_seeds, FUSED, "emb", "acc_all")
    rows.append(("T4", "banking: `both:c,e0.5` beats `emb` on all four groups by at least 0.02 on average and in at least 7 of 10 seeds", d4.mean() >= 0.02 and (d4 > 0).sum() >= 0.7 * k, f"{d4.mean():+.3f}, better in {(d4 > 0).sum()}/{k} seeds"))
    g = {d: paired(d, bank_seeds if d == "banking" else news_seeds, "key:c,p16", "emb:p16", "acc_all")[0].mean() for d in ("banking",) + news}
    rows.append(("T5", "`key:c,p16` is at least 0.25 below `emb:p16` on all four groups in all three domains", all(x <= -0.25 for x in g.values()), "; ".join(f"{d}: {x:+.3f}" for d, x in g.items())))
    s_b = paired("banking", bank_seeds, "both:c,e0.5,s,p16", FUSED16, "acc_all")[0].mean()
    s_n = {d: paired(d, news_seeds, "both:c,e0.5,s,p16", "emb:p16", "acc_all")[0].mean() for d in news}
    rows.append(("T6", "banking: the shuffled control is at least 0.03 below `both:c,e0.5,p16`; news: it is not more than 0.10 above `emb:p16`", s_b <= -0.03 and all(x <= 0.10 for x in s_n.values()), f"banking {s_b:+.3f}; " + "; ".join(f"{d} {x:+.3f}" for d, x in s_n.items())))
    sh = mean("banking", bank_seeds, HEAD, "hits") / mean("banking", bank_seeds, HEAD, "served")
    sw = mean("banking", bank_seeds, HEAD, "wrong") / max(mean("banking", bank_seeds, HEAD, "hits"), 1e-9)
    cut = 1 - mean("banking", bank_seeds, HEAD, "llm_texts") / mean("banking", bank_seeds, FUSED16, "llm_texts")
    da, _ = paired("banking", bank_seeds, HEAD, FUSED16, "acc_all")
    rows.append(("T7", "banking head arm: answers at least 25 % of the stream, at most 5 % of its answers differ from the full reader, at least 20 % fewer distinct texts asked, four-group accuracy within 0.03 of `both:c,e0.5,p16`",
                 sh >= 0.25 and sw <= 0.05 and cut >= 0.20 and abs(da.mean()) <= 0.03, f"{sh:.0%} answered, {sw:.1%} differ, {cut:.0%} fewer texts, accuracy {da.mean():+.3f}"))
    nh = {d: mean(d, news_seeds, HEAD, "hits") / max(mean(d, news_seeds, HEAD, "served"), 1e-9) for d in news}
    rows.append(("T8", "news domains: the head answers at most 10 % of the stream's messages", all(x <= 0.10 for x in nh.values()), "; ".join(f"{d}: {x:.0%}" for d, x in nh.items())))
    held = sum(1 for r in rows if r[2])
    L = ["# Lab G6: predictions T1-T8 on the report seeds", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed** (banking seeds {bank_seeds[0]}-{bank_seeds[-1]}, news seeds {news_seeds[0]}-{news_seeds[-1]}).", "",
         "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v_} |" for i, t, ok, v_ in rows]
    for d, seeds in (("banking", bank_seeds), ("agnews", news_seeds), ("newsgroups", news_seeds)):
        L += ["", TT.table(d, seeds, ARMS).replace("# Lab G6, ", "## ")]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--banking", default="100-109")
    ap.add_argument("--news", default="100-104")
    a = ap.parse_args()
    text = build(TT.seed_range(a.banking), TT.seed_range(a.news))
    (TT.OUT / "SCORECARD_G6.md").write_text(text, encoding="utf-8")
    print(text)
