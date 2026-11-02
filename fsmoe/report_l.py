"""Layer L report: ``python -m fsmoe.report_l --seeds 710 ...`` reads results/fsmoe/L/L_s*.json and writes results/fsmoe/L/REPORT_L.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .learn_lab import ARMS, MODES, OUT
from .report_s import fmt


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    runs = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"L_s{s}.json").exists()]
    L = [f"# Layer L report: the new-skill learning path (seeds {[r['seed'] for r in runs]})", "",
         f"The pool was re-taught with the same seed; the recorded teaching numbers reproduced exactly in {sum(bool(r.get('teaching_reproduced')) for r in runs)} of {len(runs)} seeds. "
         "Same stream and policies as E-055; the arms differ only in how verified items reach the pool (fsmoe/learn_lab.py). learned = answers a never-taught skill with accuracy of at least 0.5 on rows the stream never contained. "
         "`flipped` = known held-out-wording items that were right before the stream and wrong after; the three right-hand columns split them by the class they are now assigned to.", "",
         "| policy | budget | mode | verified | skills met / withheld | learned fraction | novel accuracy (end) | known ho before | known ho after | known tw after | flipped (share of known ho) | flipped to a new skill | to its twin | to another known skill | consolidations | replay memory MB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for policy, budget in ARMS:
        for mode in MODES:
            rs = [d for r in runs for d in r["deploy"] if d["policy"] == policy and abs(d["budget"] - budget) < 1e-9 and d["mode"] == mode]
            if not rs:
                continue
            share = lambda k: [d[k] / max(d["flipped"], 1) for d in rs]
            L.append(f"| {policy} | {budget:.2f} | {mode} | {fmt([d['verified'] for d in rs], 0)} | {fmt([d['novel_skills_verified'] for d in rs], 1)} / {rs[0]['n_withheld']} | {fmt([d['learned_fraction'] for d in rs], 2)} | "
                     f"{fmt([d['novel_acc_final'] for d in rs])} | {fmt([d['known_ho_before'] for d in rs])} | {fmt([d['known_ho_final'] for d in rs])} | {fmt([d['known_tw_final'] for d in rs])} | "
                     f"{fmt([d['flipped'] / d['n_known_ho'] for d in rs])} | {fmt(share('flipped_to_new_skill'), 2)} | {fmt(share('flipped_to_twin'), 2)} | {fmt(share('flipped_to_other_known'), 2)} | "
                     f"{fmt([d['consolidations'] for d in rs], 1)} | {fmt([d['replay_memory_bytes'] / 1e6 for d in rs], 2)} |")
    head = ("| arm | policy | budget | mode | verified | skills met / withheld | learned fraction | novel accuracy (end) | known ho before | known ho after | known tw after | flipped (share of known ho) | "
            "flipped to a new skill | to its twin | to another known skill |")
    for title, pat, label in (("Wording-balance arms (added after the first cells; the old skills see the stream's wordings through verified known items)", "L_extra_s{}.json", "pool"),
                              ("Global closed-form head next to the pool (addendum 2 of the design)", "L_fast_s{}.json", "pool + global head")):
        ex = [json.loads((out / pat.format(r["seed"])).read_text(encoding="utf-8")) for r in runs if (out / pat.format(r["seed"])).exists()]
        if not ex:
            continue
        L += ["", f"## {title}", "", f"({len(ex)} seeds" + (f"; after teaching only, the pool with the global head reads taught skills {fmt([r['teaching_mean_tw'] for r in ex])} and held-out wordings {fmt([r['teaching_mean_ho'] for r in ex])}, "
              f"the pool alone {fmt([r['pool_alone_mean_tw'] for r in ex])} and {fmt([r['pool_alone_mean_ho'] for r in ex])}" if "pool_alone_mean_tw" in ex[0] else "") + ")", "", head, "|" + "---|" * 15]
        arms = []
        for r in ex:
            for d in r["deploy"]:
                key = (d["policy"], d["budget"], d["mode"])
                if key not in arms:
                    arms.append(key)
        for policy, budget, mode in arms:
            rs = [d for r in ex for d in r["deploy"] if d["policy"] == policy and d["mode"] == mode and abs(d["budget"] - budget) < 1e-9]
            share = lambda k: [d[k] / max(d["flipped"], 1) for d in rs]
            L.append(f"| {label} | {policy} | {budget:.2f} | {mode} | {fmt([d['verified'] for d in rs], 0)} | {fmt([d['novel_skills_verified'] for d in rs], 1)} / {rs[0]['n_withheld']} | {fmt([d['learned_fraction'] for d in rs], 2)} | "
                     f"{fmt([d['novel_acc_final'] for d in rs])} | {fmt([d['known_ho_before'] for d in rs])} | {fmt([d['known_ho_final'] for d in rs])} | {fmt([d['known_tw_final'] for d in rs])} | "
                     f"{fmt([d['flipped'] / d['n_known_ho'] for d in rs])} | {fmt(share('flipped_to_new_skill'), 2)} | {fmt(share('flipped_to_twin'), 2)} | {fmt(share('flipped_to_other_known'), 2)} |")
    conf = out / "CONFOUND.md"
    if conf.exists():
        L += ["", "## The wording-confound test (closed-form classifier, no pool)", "", conf.read_text(encoding="utf-8").split("\n", 2)[2]]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_L.md").write_text(text, encoding="utf-8")
    print(text)
