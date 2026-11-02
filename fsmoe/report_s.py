"""Layer S report: ``python -m fsmoe.report_s --seeds 710 711 ...`` reads results/fsmoe/S/S_s*.json and writes results/fsmoe/S/REPORT_S.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np
from scipy import stats

from .domains import NAMES
from .system_lab import EXTRA_POLICIES, OUT, POLICIES

SYSTEMS = ("FS", "replay_eqmem", "knn_eqmem", "lda", "joint")


def fmt(vals: Sequence[float], p: int = 3) -> str:
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.{p}f}"
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.{p}f} [{v.mean() - h:.{p}f}, {v.mean() + h:.{p}f}]"


def load(seeds: Sequence[int], out: Path = OUT) -> List[dict]:
    return [json.loads((out / f"S_s{s}.json").read_text(encoding="utf-8")) for s in seeds if (out / f"S_s{s}.json").exists()]


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    runs = load(seeds, out)
    L = [f"# Layer S report: the system on the three-domain stream (seeds {[r['seed'] for r in runs]})", "",
         "Taught in the order arithmetic, code, language; read on test rows the pool never trained on. tw = taught skills in a teaching wording, ho = taught skills in a HELD-OUT wording. "
         "Mean [95 % interval over seeds]. FS = the validated pool with the closed-form head in every expert (`ULDA`). Baselines get the pool's memory (kNN, replay) and their own tuned recipe (dev seed).", "",
         "## S1 teaching", "", "| system | tw (mean of 3 domains) | ho (mean of 3 domains) | tw arith | tw code | tw lang | ho arith | ho code | ho lang | forgetting arith | forgetting code | experts | memory MB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SYSTEMS:
        rs = [r["teaching"][s] for r in runs if s in r["teaching"]]
        if not rs:
            continue
        L.append(f"| {s} | {fmt([r['mean_tw'] for r in rs])} | {fmt([r['mean_ho'] for r in rs])} | " + " | ".join(fmt([r["final"][d]["tw"] for r in rs]) for d in NAMES) + " | "
                 + " | ".join(fmt([r["final"][d]["ho"] for r in rs]) for d in NAMES) + f" | {fmt([r['forgetting']['arith'] for r in rs])} | {fmt([r['forgetting']['code'] for r in rs])} | "
                 f"{fmt([r['n_experts'] for r in rs], 1)} | {fmt([r['memory_bytes'] / 1e6 for r in rs], 1)} |")
    fs = [r["teaching"]["FS"] for r in runs]
    L += ["", f"FS events over the whole stream (summed over seeds): {_events(fs)}.", "",
          "## S3 teams (held-out wordings, taught skills)", "", "| answer | accuracy | mean team size |", "|---|---|---|",
          f"| top-1 (the pool's own routing) | {fmt([r['team']['top1_ho'] for r in runs])} | 1 |",
          f"| adaptive team (mass 0.99, cap 3, tau {runs[0]['team']['tau'] if runs else '?'}) | {fmt([r['team']['team_ho'] for r in runs])} | {fmt([r['team']['mean_size'] for r in runs], 2)} |", "",
          "## S2 deployment: a budget of verified outcomes, spent by a policy", "",
          "A withheld skill counts as learned when the pool answers it with accuracy >= 0.5 on rows the stream never contained. `precision` = share of verified items that were novel (the stream is about 14 % novel). "
          "ho before / after = accuracy on held-out wordings of taught skills before and after the stream.", "",
          "| policy | budget | verified | precision | skills met / withheld | learned fraction | novel accuracy (end) | novel accuracy (online, 2nd half) | known ho before | known ho after | known tw after | experts |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for policy, budget in POLICIES:
        rs = [d for r in runs for d in r["deploy"] if d["policy"] == policy and abs(d["budget"] - budget) < 1e-9]
        if not rs:
            continue
        L.append(f"| {policy} | {budget:.2f} | {fmt([d['verified'] for d in rs], 0)} | {fmt([d['precision_novel'] for d in rs], 2)} | "
                 f"{fmt([d['novel_skills_verified'] for d in rs], 1)} / {rs[0]['n_withheld']} | {fmt([d['learned_fraction'] for d in rs], 2)} | {fmt([d['novel_acc_final'] for d in rs])} | "
                 f"{fmt([d['novel_acc_online'] for d in rs])} | {fmt([d['known_ho_before'] for d in rs])} | {fmt([d['known_ho_final'] for d in rs])} | {fmt([d['known_tw_final'] for d in rs])} | "
                 f"{fmt([d['n_experts'] for d in rs], 1)} |")
    extra = [json.loads((out / f"S_extra_s{r['seed']}.json").read_text(encoding="utf-8")) for r in runs if (out / f"S_extra_s{r['seed']}.json").exists()]
    if extra:
        L += ["", "## S2 extra (added after the main readout): the oracle's budget curve and a diversity-aware novelty policy", "",
              f"The pool was re-taught with the same seed; the recorded teaching numbers reproduced exactly in {sum(e['teaching_reproduced'] for e in extra)} of {len(extra)} seeds. "
              "`diverse` = novelty ranking restricted to inputs far from each other (fsmoe.system_lab.diverse_pick).", "",
              "| policy | budget | verified | precision | skills met / withheld | learned fraction | novel accuracy (end) | known ho before | known ho after | experts |", "|---|---|---|---|---|---|---|---|---|---|"]
        for policy, budget in EXTRA_POLICIES:
            rs = [d for e in extra for d in e["deploy"] if d["policy"] == policy and abs(d["budget"] - budget) < 1e-9]
            if rs:
                L.append(f"| {policy} | {budget:.2f} | {fmt([d['verified'] for d in rs], 0)} | {fmt([d['precision_novel'] for d in rs], 2)} | {fmt([d['novel_skills_verified'] for d in rs], 1)} / {rs[0]['n_withheld']} | "
                         f"{fmt([d['learned_fraction'] for d in rs], 2)} | {fmt([d['novel_acc_final'] for d in rs])} | {fmt([d['known_ho_before'] for d in rs])} | {fmt([d['known_ho_final'] for d in rs])} | {fmt([d['n_experts'] for d in rs], 1)} |")
    return "\n".join(L) + "\n"


def _events(tr: List[dict]) -> str:
    tot: dict = {}
    for r in tr:
        for k, v in r["events"].items():
            tot[k] = tot.get(k, 0) + v
    return ", ".join(f"{k} {v}" for k, v in sorted(tot.items())) or "none"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_S.md").write_text(text, encoding="utf-8")
    print(text)
