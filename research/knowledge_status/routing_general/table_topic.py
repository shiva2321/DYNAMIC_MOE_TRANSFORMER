"""Tables of lab G6 (lab_poolfeed_topic.py): ``python table_topic.py --domain banking --seeds 100-109 --arms emb both:c,e0.5 --out report``; writes results/routing_general/g6/TABLE_<domain>_<out>.md and prints it."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
OUT = _HERE.parent / "results" / "routing_general" / "g6"
GROUPS = ("taught_plain", "taught_shift", "novel_plain", "novel_shift")


def seed_range(text: str):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def load(domain, seeds, arm):
    name = arm.replace(":", "_").replace(",", "-")
    return [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := OUT / f"{domain}_{name}_s{s}.json").exists()]


def metric(r, k):
    f = r["final"]
    return {"experts": r["experts"], "p1": r["per_phase"][0]["experts"], "false_spawns": r["false_spawns"], "spawns_p3": r["spawns_p3"], "nmi": r["nmi_expert_skill"],
            "taught_plain": f["taught_plain"], "taught_shift": f["taught_shift"], "novel_plain": f["novel_plain"], "novel_shift": f["novel_shift"],
            "acc_taught": 0.5 * (f["taught_plain"] + f["taught_shift"]), "acc_novel": 0.5 * (f["novel_plain"] + f["novel_shift"]), "acc_shift": 0.5 * (f["taught_shift"] + f["novel_shift"]),
            "acc_all": 0.25 * sum(f.values()), "llm_texts": r.get("asked_unique", r["reader_texts"]), "served": r.get("stream_served", r.get("cache_served", float("nan"))), "hits": r.get("stream_hits", r.get("cache_hits", float("nan"))),
            "wrong": r.get("stream_wrong", r.get("cache_wrong", float("nan")))}[k]


def table(domain, seeds, arms) -> str:
    L = [f"# Lab G6, {domain}: the pool fed by different keys (seeds {seeds[0]}-{seeds[-1]}; mean over seeds)", "",
         "| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-class | taught classes: plain / shifted | new class: plain / shifted | all four groups | LM passes per seed | cheap-reader answers (disagree) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        rs = load(domain, seeds, arm)
        if not rs:
            continue
        m = lambda k: float(np.mean([metric(r, k) for r in rs]))
        cheap = f"{m('hits') / m('served'):.0%} ({m('wrong') / m('hits'):.1%})" if not math.isnan(m("hits")) else "-"
        L.append(f"| {arm} | {len(rs)} | {m('p1'):.1f} / {m('experts'):.1f} | {m('false_spawns'):.2f} | {m('spawns_p3'):.2f} | {m('nmi'):.2f} | {m('taught_plain'):.3f} / {m('taught_shift'):.3f} | {m('novel_plain'):.3f} / {m('novel_shift'):.3f} | {m('acc_all'):.3f} | {m('llm_texts'):.0f} | {cheap} |")
    base = {r["seed"]: r for r in load(domain, seeds, "emb")}
    if base:
        L += ["", "## Paired against `emb` (same seeds; mean difference with its standard error, and the number of seeds in which the arm is better)", "", "| arm | all four groups | taught classes, shifted | new class (both wordings) | NMI expert-class |", "|---|---|---|---|---|"]
        for arm in arms:
            if arm == "emb":
                continue
            rs = [r for r in load(domain, seeds, arm) if r["seed"] in base]
            if not rs:
                continue
            cells = []
            for k in ("acc_all", "taught_shift", "acc_novel", "nmi"):
                d = np.array([metric(r, k) - metric(base[r["seed"]], k) for r in rs])
                se = d.std(ddof=1) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
                cells.append(f"{d.mean():+.3f} ({se:.3f}), {int((d > 0).sum())}/{len(d)}")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="banking")
    ap.add_argument("--seeds", default="150-154")
    ap.add_argument("--arms", nargs="+", default=["emb", "emb:c", "key:c", "both:c", "both:c,e0.5"])
    ap.add_argument("--out", default="dev")
    a = ap.parse_args()
    text = table(a.domain, seed_range(a.seeds), a.arms)
    (OUT / f"TABLE_{a.domain}_{a.out}.md").write_text(text, encoding="utf-8")
    print(text)
