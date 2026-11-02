"""Growth report: ``python -m fsmoe.grow.report_g --seeds 710 ...`` reads results/fsmoe/G/<scenario>_s<seed>_<arm>.json and writes results/fsmoe/G/REPORT_G.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from fsmoe.bed import ROOT
from fsmoe.report_s import fmt

from .lab import OUT

LABEL = {"F8": "fixed, 8 units", "F16": "fixed, 16 units", "F32": "fixed, 32 units", "F64": "fixed, 64 units", "F8d3": "fixed, 8 units x 3 blocks", "BF": "brute force (every expert grows at every look)",
         "Gng": "notebook-first, no width/depth growth", "GnoNB": "network at birth + governor", "Gnt": "governor without the tier", "G": "governed (all four directions)"}
FIXED = ("F8", "F16", "F32", "F64", "F8d3")
MIXED_ARMS = ("F8", "F16", "F32", "F64", "F8d3", "BF", "Gng", "GnoNB", "G")
EASY_ARMS = ("F8", "F32", "BF", "G")
BIG_ARMS = ("F8", "Gnt", "G")


def load(scenario: str, arm: str, seeds: Sequence[int], out: Path = OUT) -> List[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"{scenario}_s{s}_{arm}.json").exists()]


def total(r: dict, key: str) -> float:
    return float(r["summary"]["params"][key])


def rows_by_need(runs: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {"width": [], "depth": [], "none": []}
    for r in runs:
        for row in r["structure"].get("rows", []):
            out[row["need"]].append(row)
    return out


def share(rows: List[dict], pred) -> float:
    return float(np.mean([pred(x) for x in rows])) if rows else float("nan")


GREW_W = lambda x: x["hidden"] >= 16
GREW_D = lambda x: x["depth"] >= 2
NB_ONLY = lambda x: x["hidden"] == 0


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    L = [f"# Growth report (seeds {list(seeds)})", "", "Mean [95 % interval over seeds]. Parameters = network + notebook + centroid scalars. Multiply-adds per answer = gate + expert + notebook, in thousands; training and governing multiply-adds in billions.", ""]
    mixed = {a: load("mixed", a, seeds, out) for a in MIXED_ARMS}
    L += ["## Mixed needs (16 regimes: 4 linear, 8 wide, 4 deep)", "", "| arm | mean accuracy | worst regime | total parameters | of which networks | answer k-MACs | training G-MACs | governing G-MACs | experts |", "|---|---|---|---|---|---|---|---|---|"]
    for a in MIXED_ARMS:
        rs = mixed[a]
        if not rs:
            continue
        L.append(f"| {LABEL[a]} | {fmt([r['mean_acc'] for r in rs])} | {fmt([r['min_acc'] for r in rs])} | {fmt([total(r, 'total') for r in rs], 0)} | {fmt([total(r, 'net') for r in rs], 0)} | "
                 f"{fmt([r['macs_answer'] / 1e3 for r in rs], 2)} | {fmt([r['summary']['macs_train'] / 1e9 for r in rs], 1)} | {fmt([r['summary']['macs_trial'] / 1e9 for r in rs], 1)} | {fmt([r['summary']['experts'] for r in rs], 1)} |")
    L += ["", "### Growth against the oracle (every regime of every seed counted once; `width` = a network of at least 16 units, `depth` = at least 2 blocks, `notebook` = no network)", "",
          "| arm | need (regimes) | with width | with depth | notebook only | mean accuracy |", "|---|---|---|---|---|---|"]
    for a in ("G", "Gng", "GnoNB", "BF", "F8"):
        for need, rows in rows_by_need(mixed[a]).items():
            if rows:
                L.append(f"| {LABEL[a]} | {need} ({len(rows)}) | {share(rows, GREW_W):.2f} | {share(rows, GREW_D):.2f} | {share(rows, NB_ONLY):.2f} | {np.mean([x['acc'] for x in rows]):.3f} |")
    L += ["", "### What the governor did (events per seed, mean)", "", "| arm | spawn | absorb | promote | keep notebook | growth trial rejected | grow | keep | revert |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ("G", "Gng", "GnoNB", "BF"):
        rs = mixed[a]
        if rs:
            ev = lambda k: float(np.mean([r["summary"]["events"].get(k, 0) for r in rs]))
            L.append(f"| {LABEL[a]} | {ev('spawn'):.1f} | {ev('absorb'):.1f} | {ev('promote'):.1f} | {ev('keep_notebook'):.1f} | {ev('trial'):.1f} | {ev('grow'):.1f} | {ev('keep'):.1f} | {ev('revert'):.1f} |")
    easy = {a: load("easy", a, seeds, out) for a in EASY_ARMS}
    L += ["", "## Easy (8 linear regimes: nothing needs growth)", "", "| arm | mean accuracy | worst regime | total parameters | answer k-MACs | growth events per seed | experts |", "|---|---|---|---|---|---|---|"]
    for a in EASY_ARMS:
        rs = easy[a]
        if rs:
            L.append(f"| {LABEL[a]} | {fmt([r['mean_acc'] for r in rs])} | {fmt([r['min_acc'] for r in rs])} | {fmt([total(r, 'total') for r in rs], 0)} | {fmt([r['macs_answer'] / 1e3 for r in rs], 2)} | "
                     f"{np.mean([r['summary']['events'].get('grow', 0) for r in rs]):.1f} | {fmt([r['summary']['experts'] for r in rs], 1)} |")
    big = {a: load("big", a, seeds, out) for a in BIG_ARMS}
    L += ["", "## Big (48 regimes in 8 groups: the flat gate costs 1,536 multiply-adds per sample, the budget is 640)", "", "| arm | mean accuracy | worst regime | total parameters | answer k-MACs | experts | purity | tier built / refused (events per seed) |", "|---|---|---|---|---|---|---|---|"]
    for a in BIG_ARMS:
        rs = big[a]
        if rs:
            built = np.mean([r["summary"]["events"].get("tier", 0) for r in rs])
            refused = np.mean([r["summary"]["events"].get("tier_refused", 0) for r in rs])
            L.append(f"| {LABEL[a]} | {fmt([r['mean_acc'] for r in rs])} | {fmt([r['min_acc'] for r in rs])} | {fmt([total(r, 'total') for r in rs], 0)} | {fmt([r['macs_answer'] / 1e3 for r in rs], 2)} | "
                     f"{fmt([r['summary']['experts'] for r in rs], 1)} | {fmt([r['structure']['purity'] for r in rs], 3)} | {built:.1f} / {refused:.1f} |")
    L += ["", "## Pool against the regimes (G)", "", "| scenario | experts | regimes | purity | regimes with an expert of their own |", "|---|---|---|---|---|"]
    for sc, runs in (("mixed", mixed["G"]), ("easy", easy["G"]), ("big", big["G"])):
        if runs:
            L.append(f"| {sc} | {fmt([r['structure']['experts'] for r in runs], 1)} | {runs[0]['structure']['regimes']} | {fmt([r['structure']['purity'] for r in runs], 3)} | {fmt([r['structure']['regimes_with_own_expert'] for r in runs], 3)} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_G.md").write_text(text, encoding="utf-8")
    print(text)
