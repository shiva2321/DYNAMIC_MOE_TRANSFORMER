"""Layer E report: ``python -m fsmoe.report_e --seeds 710 ...`` reads results/fsmoe/E/E_s*.json and writes results/fsmoe/E/REPORT_E.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .exec_traces import MS, OUT
from .report_s import fmt

SIGNALS = ("look", "cascade", "result_all", "fingerprint")


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    runs = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"E_s{s}.json").exists()]
    L = [f"# Layer E report: execution outcomes on real code (seeds {[r['seed'] for r in runs]})", "",
         "Known = five taught list functions in held-out styles (template styles and real library idioms); near = withheld product and count_odd; far = withheld real library functions (length, median, range, second largest, "
         "mean, last). AUROC = known in held-out styles vs withheld. Flags at the conformal cut (q = 0.05, n = 40 known snippets). M = lists each snippet is executed on. "
         "`fingerprint` = outputs on fixed probes against the taught families' reference outputs (nothing learned: a ceiling).", "",
         f"Snippets per seed (mean): known in teaching styles {fmt([r['n']['known_tw'] for r in runs], 0)}, known in held-out styles {fmt([r['n']['known_ho'] for r in runs], 0)} "
         f"({fmt([r['n_library']['known_ho'] for r in runs], 0)} library idioms), near {fmt([r['n']['near'] for r in runs], 0)}, far {fmt([r['n']['far'] for r in runs], 0)}. "
         f"Transitions per snippet at M = {', '.join(str(m) for m in MS)}: {', '.join(str(runs[0]['steps_per_snippet'][str(m)]) for m in MS)} (first seed).", ""]
    for M in MS:
        L += [f"## M = {M} executions per snippet", "",
              "| signal | AUROC vs near-twins | AUROC vs real library functions | flag known, template style | flag known, library idiom | flag known, teaching style | near found (template) | near found (library) | far found |",
              "|---|---|---|---|---|---|---|---|---|"]
        for s in SIGNALS:
            g = lambda k: [r["by_m"][str(M)][s]["flag"][k] for r in runs]
            L.append(f"| {s} | {fmt([r['by_m'][str(M)][s]['auroc_near'] for r in runs])} | {fmt([r['by_m'][str(M)][s]['auroc_far'] for r in runs])} | {fmt(g('known_ho_template'))} | "
                     f"{fmt(g('known_ho_library'))} | {fmt(g('known_tw'))} | {fmt(g('near_template'))} | {fmt(g('near_library'))} | {fmt(g('far'))} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_E.md").write_text(text, encoding="utf-8")
    print(text)
