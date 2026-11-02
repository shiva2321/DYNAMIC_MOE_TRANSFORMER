"""Layer T report: ``python -m fsmoe.report_t --seeds 710 ...`` reads results/fsmoe/T/*.json and writes results/fsmoe/T/REPORT_T.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .report_s import fmt
from .traces import OUT

NOISES = (0.05, 0.3, 1.0)
CASES = ("simple_shift", "simple_mix", "composed_3", "composed_5", "interact_1", "interact_3")


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    L = [f"# Layer T report: traces (seeds {list(seeds)})", "",
         "Recognition: known = the six taught regimes; withheld = four regimes never taught (same state distribution, other functions). A held-out surface moves the nuisance features only. "
         "flag = share called new at the conformal cut (q = 0.05, n = 40 known inputs from the stream's own surfaces). div2 / div6 = how many surfaces the experts were taught on.", ""]
    for div in (2, 6):
        for noise in NOISES:
            runs = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"T_rec_s{s}_div{div}_n{noise}.json").exists()]
            if not runs:
                continue
            L += [f"## recognition, {div} teaching surfaces, next-state noise {noise} ({len(runs)} seeds)", "",
                  "| signal | flag known (teaching surface) | flag known (held-out surface) | flag withheld | flag far | AUROC held-out known vs withheld | AUROC held-out known vs far | MACs/input |", "|---|---|---|---|---|---|---|---|"]
            for s in ("look", "result_nearest", "result_all"):
                g = lambda k: [r["signals"][s]["flag"][k] for r in runs]
                wh = [(a + b) / 2 for a, b in zip(g("withheld_tw"), g("withheld_ho"))]
                L.append(f"| {s} | {fmt(g('known_tw'))} | {fmt(g('known_ho'))} | {fmt(wh)} | {fmt(g('far'))} | {fmt([r['signals'][s]['auroc_known_ho_vs_withheld'] for r in runs])} | "
                         f"{fmt([r['signals'][s]['auroc_known_ho_vs_far'] for r in runs])} | {runs[0]['cost_macs'][s]} |")
            L.append("")
    runs = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"T_depth_s{s}.json").exists()]
    if runs:
        L += [f"## depth growth: start at 8 hidden units, grow only when stuck ({len(runs)} seeds)", "",
              "Final mean squared error of the next-state prediction (target variance in the last column), the grown net's final size and the fixed nets of other shapes trained for the same number of epochs.", "",
              "poor = the fit is still worse than this share of the target variance. `grown` uses 0.2 (declared before the run), `grown 0.05` a stricter 0.05 (added after the first run showed the first four cases never trigger growth).", "",
              "| case | grown: error | hidden x depth | params | widen / deepen kept | grown 0.05: error | hidden x depth | params | widen / deepen kept | fixed 8x1 | fixed 64x1 | fixed 128x1 | fixed 32x2 | fixed 32x3 | target variance |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in CASES:
            rs = [r["cases"][c] for r in runs if c in r["cases"]]
            if not rs:
                continue
            fx = lambda i: fmt([x["fixed"][i]["final_mse"] for x in rs], 4)
            kept = lambda key, kind: np.mean([sum(1 for e in x[key]["events"] if e["kind"] == kind and e["gain_after"] >= 0.03) for x in rs])
            g5 = [x["grown_05"] for x in rs if "grown_05" in x]
            L.append(f"| {c} | {fmt([x['grown']['final_mse'] for x in rs], 4)} | {fmt([x['grown']['hidden'] for x in rs], 0)} x {fmt([x['grown']['depth'] for x in rs], 1)} | "
                     f"{fmt([x['grown']['params'] for x in rs], 0)} | {kept('grown', 'widen'):.1f} / {kept('grown', 'deepen'):.1f} | "
                     + (f"{fmt([x['final_mse'] for x in g5], 4)} | {fmt([x['hidden'] for x in g5], 0)} x {fmt([x['depth'] for x in g5], 1)} | {fmt([x['params'] for x in g5], 0)} | "
                        f"{np.mean([sum(1 for e in x['events'] if e['kind'] == 'widen' and e['gain_after'] >= 0.03) for x in g5]):.1f} / {np.mean([sum(1 for e in x['events'] if e['kind'] == 'deepen' and e['gain_after'] >= 0.03) for x in g5]):.1f} | " if g5 else "n/a | n/a | n/a | n/a | ")
                     + f"{fx(0)} | {fx(1)} | {fx(2)} | {fx(3)} | {fx(4)} | {fmt([x['grown']['target_var'] for x in rs], 3)} |")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_T.md").write_text(text, encoding="utf-8")
    print(text)
