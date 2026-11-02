"""Tables of lab G5 (lab_poolfeed.py): ``python table_poolfeed.py --seeds 150-154 [--arms emb key ...] [--out NAME]`` prints and writes results/routing_general/g5/TABLE_<NAME>.md."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
OUT = _HERE.parent / "results" / "routing_general" / "g5"


def seed_range(text: str):
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def load(seeds, arm):
    name = arm.replace(":", "_").replace(",", "-")
    rs = []
    for s in seeds:
        p = OUT / f"{name}_s{s}.json"
        if p.exists():
            rs.append(json.loads(p.read_text(encoding="utf-8")))
    return rs


def metric(r, k):
    f = r["final"]
    return {"experts": r["experts"], "p1": r["per_phase"][0]["experts"], "false_spawns": r["false_spawns"], "spawns_p3": r["spawns_p3"], "nmi": r["nmi_expert_skill"],
            "acc_taught": 0.5 * (f["taught_teach"] + f["taught_new"]), "acc_taught_new": f["taught_new"], "acc_novel": 0.5 * (f["novel_teach"] + f["novel_new"]), "acc_novel_new": f["novel_new"],
            "acc_all": 0.25 * sum(f.values()), "unassigned": r["unassigned_share"], "reader_s": r["reader_seconds"], "pool_s": r["pool_seconds"], "macs": r["predict_macs"],
            "llm_texts": r.get("asked_unique", r["reader_texts"]), "served": r.get("stream_served", r.get("cache_served", float("nan"))), "hits": r.get("stream_hits", r.get("cache_hits", float("nan"))),
            "wrong": r.get("stream_wrong", r.get("cache_wrong", float("nan")))}[k]


def table(seeds, arms) -> str:
    L = [f"# Lab G5, the pool fed by different keys (seeds {seeds[0]}-{seeds[-1]}; mean over seeds)", "",
         "| arm | seeds | experts after phases 1 / 3 | false spawns (phase 2) | spawns phase 3 | NMI expert-skill | acc taught skills (teach / new wording) | acc new skill (teach / new wording) | acc all four groups | unassigned share | language-model passes per seed | cache hits (wrong share of hits) | pool s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        rs = load(seeds, arm)
        if not rs:
            continue
        m = lambda k: float(np.mean([metric(r, k) for r in rs]))
        L.append(f"| {arm} | {len(rs)} | {m('p1'):.1f} / {m('experts'):.1f} | {m('false_spawns'):.2f} | {m('spawns_p3'):.2f} | {m('nmi'):.2f} | {m('acc_taught'):.3f} ({np.mean([r['final']['taught_teach'] for r in rs]):.3f} / {m('acc_taught_new'):.3f}) | "
                 f"{m('acc_novel'):.3f} ({np.mean([r['final']['novel_teach'] for r in rs]):.3f} / {m('acc_novel_new'):.3f}) | {m('acc_all'):.3f} | {m('unassigned'):.2f} | {m('llm_texts'):.0f} | {(f"{m('hits') / m('served'):.0%} ({m('wrong') / m('hits'):.1%})" if not np.isnan(m('hits')) else "-")} | {m('pool_s'):.0f} |")
    base = load(seeds, "emb")
    if base:
        L += ["", "## Paired against `emb` (same seeds; mean difference, and the number of seeds in which the arm is better)", "", "| arm | acc all | acc taught in new wordings | acc new skill | NMI expert-skill |", "|---|---|---|---|---|"]
        by = {r["seed"]: r for r in base}
        for arm in arms:
            if arm == "emb":
                continue
            rs = [r for r in load(seeds, arm) if r["seed"] in by]
            if not rs:
                continue
            cells = []
            for k in ("acc_all", "acc_taught_new", "acc_novel", "nmi"):
                d = np.array([metric(r, k) - metric(by[r["seed"]], k) for r in rs])
                cells.append(f"{d.mean():+.3f} ({int((d > 0).sum())}/{len(d)})")
            L.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="150-154")
    ap.add_argument("--arms", nargs="+", default=["emb", "key", "both", "oracle"])
    ap.add_argument("--out", default="dev")
    a = ap.parse_args()
    text = table(seed_range(a.seeds), a.arms)
    (OUT / f"TABLE_{a.out}.md").write_text(text, encoding="utf-8")
    print(text)
