"""Notebook-first pool report: ``python -m fsmoe.report_n --seeds 710 ...`` reads results/fsmoe/N/<bed>_s<seed>_<arm>.json and writes results/fsmoe/N/REPORT_N.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .report_s import fmt
from .nbpool_lab import OUT

LABEL = {"U8": "shipped pool, 8-unit networks", "U32": "shipped pool, 32-unit networks", "U128": "shipped pool (128-unit networks, the default)", "NB": "notebook-first", "NBc": "notebook-first, candidate sizes 8 and 32", "U8x": "ablation: shipped pool, 8-unit networks + the worse-than-chance rule", "U128x": "ablation: shipped pool, 128-unit networks + the worse-than-chance rule", "NBn": "ablation: notebook experts only (never judged)"}
ARMS = ("U8", "U32", "U128", "NB", "NBc")
ABL = ("U8x", "U128x", "NBn")
REGIMES = {"f10": 10, "f30": 30, "g": 16}
BEDNAME = {"f10": "Function regimes, 10 regimes (growmoe viability stream)", "f30": "Function regimes, 30 regimes", "g": "Growth-lab world through the real pool (16 regimes, 8 need a wide network)", "t": "FS-MoE teaching bed (3 domains, real text embeddings)"}


def load(bed: str, arm: str, seeds: Sequence[int], out: Path = OUT) -> List[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"{bed}_s{s}_{arm}.json").exists()]


def n_net(r: dict) -> float:
    nb = r.get("nb")
    return float(nb["net_experts"]) if nb else float(r["experts"])


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    L = [f"# Notebook-first pool report (seeds {list(seeds)})", "", "Mean [95 % interval over seeds]. Network parameters = the pool's parameter count (notebook experts carry one inert scalar each). Memory = the pool's bytes (networks, closed-form heads, replay, signatures, held-out samples). "
         "Judging = multiply-adds spent training candidate networks, in billions.", ""]
    for bed in ("f10", "f30", "g"):
        L += [f"## {BEDNAME[bed]}", "", "| arm | accuracy | worst regime | experts | experts per regime | network parameters | memory MB | answer k-MACs | judging G-MACs | experts with a network | wall seconds |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for a in ARMS:
            rs = load(bed, a, seeds, out)
            if rs:
                L.append(f"| {LABEL[a]} | {fmt([r['acc'] for r in rs])} | {fmt([r['min_regime_acc'] for r in rs])} | {fmt([r['experts'] for r in rs], 1)} | {fmt([r['experts'] / REGIMES[bed] for r in rs], 2)} | {fmt([r['params'] for r in rs], 0)} | "
                         f"{fmt([r['bytes'] / 1e6 for r in rs], 2)} | {fmt([r['predict_macs'] / 1e3 for r in rs], 1)} | {fmt([r['judging_macs'] / 1e9 for r in rs], 1)} | {fmt([n_net(r) for r in rs], 1)} | {fmt([r['seconds'] for r in rs], 0)} |")
        L.append("")
    L += [f"## {BEDNAME['t']}", "", "| arm | known skills, teaching wording | known skills, new wording | experts | network parameters | memory MB | judging G-MACs | experts with a network | wall seconds |", "|---|---|---|---|---|---|---|---|---|"]
    for a in ARMS:
        rs = load("t", a, seeds, out)
        if rs:
            L.append(f"| {LABEL[a]} | {fmt([r['mean_tw'] for r in rs])} | {fmt([r['mean_ho'] for r in rs])} | {fmt([r['experts'] for r in rs], 1)} | {fmt([r['params'] for r in rs], 0)} | {fmt([r['bytes'] / 1e6 for r in rs], 2)} | "
                     f"{fmt([r['judging_macs'] / 1e9 for r in rs], 1)} | {fmt([n_net(r) for r in rs], 1)} | {fmt([r['seconds'] for r in rs], 0)} |")
    L += ["", "## Exploratory ablation (added after the first results, not pre-declared): which part of the port does the work?", "", "The worse-than-chance rule alone on the shipped pool (`x` arms), and notebook experts only with no size test (`NBn`). Accuracy, experts per regime, network parameters and memory; the growth-world bed has the 8-unit arm only.", "",
          "| bed | arm | accuracy | experts per regime | network parameters | memory MB |", "|---|---|---|---|---|---|"]
    for bed in ("f10", "f30", "g", "t"):
        for a in ("U8", "U8x", "U128", "U128x", "NB", "NBn"):
            rs = load(bed, a, seeds, out)
            if rs:
                accv = [r["acc"] if bed != "t" else 0.5 * (r["mean_tw"] + r["mean_ho"]) for r in rs]
                reg = REGIMES.get(bed, 3)
                L.append(f"| {bed} | {LABEL[a]} | {fmt(accv)} | {fmt([r['experts'] / reg for r in rs], 2)} | {fmt([r['params'] for r in rs], 0)} | {fmt([r['bytes'] / 1e6 for r in rs], 2)} |")
    sizes = {}
    for bed in ("f10", "f30", "g", "t"):
        for r in load(bed, "NB", seeds, out):
            for _, v in (r["nb"] or {}).get("net_sizes", {}).items():
                sizes[tuple(v)] = sizes.get(tuple(v), 0) + 1
    L += ["", "## Networks the size test gave (notebook-first, all beds and seeds)", "", "| units x blocks | experts |", "|---|---|"] + [f"| {h} x {d} | {n} |" for (h, d), n in sorted(sizes.items())]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_N.md").write_text(text, encoding="utf-8")
    print(text)
