"""Scores the growth predictions P1-P14 (docs/superpowers/specs/2026-11-01-fsmoe-growth-design.md). ``python -m fsmoe.grow.scorecard_g --seeds 710 ...`` writes results/fsmoe/SCORECARD_G.md."""

import argparse
from typing import Sequence

import numpy as np

from fsmoe.bed import ROOT

from .report_g import FIXED, GREW_D, GREW_W, load, rows_by_need, share, total


def build(seeds: Sequence[int]) -> str:
    M = lambda sc, a: load(sc, a, seeds)
    mean = lambda rs, f: float(np.mean([f(r) for r in rs]))
    acc = lambda rs: mean(rs, lambda r: r["mean_acc"])
    par = lambda rs: mean(rs, lambda r: total(r, "total"))
    mixed = {a: M("mixed", a) for a in ("F8", "F16", "F32", "F64", "F8d3", "BF", "Gng", "GnoNB", "G")}
    easy = {a: M("easy", a) for a in ("F8", "G")}
    big = {a: M("big", a) for a in ("F8", "G")}
    G = mixed["G"]
    rows = []
    best_fixed = max(acc(mixed[a]) for a in FIXED)
    rows.append(("P1", "G beats F8 and F16 by at least 0.02 each and is at most 0.01 below the best fixed arm", acc(G) - acc(mixed["F8"]) >= 0.02 and acc(G) - acc(mixed["F16"]) >= 0.02 and acc(G) >= best_fixed - 0.01,
                 f"G {acc(G):.3f}, F8 {acc(mixed['F8']):.3f}, F16 {acc(mixed['F16']):.3f}, best fixed {best_fixed:.3f}"))
    eq = [a for a in FIXED if par(mixed[a]) <= 1.2 * par(G)]
    best_eq = max(acc(mixed[a]) for a in eq) if eq else float("nan")
    rows.append(("P2", "G is at least as accurate as the best fixed arm of equal memory (at most 1.2 x G's parameters)", bool(eq) and acc(G) >= best_eq, f"G {acc(G):.3f} with {par(G):.0f} parameters; equal-memory arms {eq} best {best_eq:.3f}"))
    rows.append(("P3", "BF has at least 5 times G's parameters and an accuracy at most 0.01 above G's", par(mixed["BF"]) >= 5 * par(G) and acc(mixed["BF"]) <= acc(G) + 0.01, f"BF {par(mixed['BF']):.0f} parameters ({par(mixed['BF']) / par(G):.1f} x), accuracy {acc(mixed['BF']):.3f} against G {acc(G):.3f}"))
    need = rows_by_need(G)
    rows.append(("P4", "at least 80 % of width-need regimes are served by an expert with at least 16 units in G", share(need["width"], GREW_W) >= 0.8, f"{share(need['width'], GREW_W):.2f} of {len(need['width'])}"))
    rows.append(("P5", "at least 40 % of depth-need regimes are served by an expert with at least 2 blocks in G", share(need["depth"], GREW_D) >= 0.4, f"{share(need['depth'], GREW_D):.2f} of {len(need['depth'])}"))
    fg = share(need["none"], lambda x: GREW_W(x) or GREW_D(x))
    rows.append(("P6", "at most 20 % of none-need regimes are served by an expert with at least 16 units or 2 blocks in G", fg <= 0.2, f"{fg:.2f} of {len(need['none'])}"))
    ge = mean(easy["G"], lambda r: r["summary"]["events"].get("grow", 0))
    rows.append(("P7", "easy: G has at most 1 growth event per seed on average and an accuracy at least 0.01 above F8's with fewer parameters", ge <= 1.0 and acc(easy["G"]) >= acc(easy["F8"]) + 0.01 and par(easy["G"]) < par(easy["F8"]),
                 f"{ge:.1f} growth events per seed; G {acc(easy['G']):.3f} ({par(easy['G']):.0f} parameters), F8 {acc(easy['F8']):.3f} ({par(easy['F8']):.0f})"))
    rows.append(("P8", "growth adds at most 0.02 to the notebook-first arm without it (G - Gng <= 0.02)", acc(G) - acc(mixed["Gng"]) <= 0.02, f"G {acc(G):.3f}, Gng {acc(mixed['Gng']):.3f}, difference {acc(G) - acc(mixed['Gng']):+.3f}"))
    rows.append(("P9", "G is at least as accurate as GnoNB minus 0.005 with no more parameters", acc(G) >= acc(mixed["GnoNB"]) - 0.005 and par(G) <= par(mixed["GnoNB"]),
                 f"G {acc(G):.3f} ({par(G):.0f}), GnoNB {acc(mixed['GnoNB']):.3f} ({par(mixed['GnoNB']):.0f})"))
    ratio = mean(G, lambda r: r["summary"]["macs_trial"] / max(r["summary"]["macs_train"], 1.0))
    rows.append(("P10", "growth trials and promotion tests in G cost at most 12 times the training multiply-adds", ratio <= 12.0, f"{ratio:.1f} times"))
    rows.append(("P11", "big: G is at least 0.02 above F8 with no more multiply-adds per answer", acc(big["G"]) >= acc(big["F8"]) + 0.02 and mean(big["G"], lambda r: r["macs_answer"]) <= mean(big["F8"], lambda r: r["macs_answer"]),
                 f"G {acc(big['G']):.3f} ({mean(big['G'], lambda r: r['macs_answer']):.0f} MACs), F8 {acc(big['F8']):.3f} ({mean(big['F8'], lambda r: r['macs_answer']):.0f})"))
    refused = sum(1 for r in big["G"] if r["summary"]["events"].get("tier", 0) == 0)
    rows.append(("P12", "big: the tier is refused (never built) in at least 4 of 5 seeds", refused >= 4, f"never built in {refused} of {len(big['G'])} seeds; built events per seed {[r['summary']['events'].get('tier', 0) for r in big['G']]}"))
    ok = True
    notes = []
    for sc, runs in (("mixed", G), ("big", big["G"])):
        if runs:
            ratio_e = mean(runs, lambda r: r["structure"]["experts"] / r["structure"]["regimes"])
            pur = mean(runs, lambda r: r["structure"]["purity"])
            ok = ok and 0.95 <= ratio_e <= 1.10 and pur >= 0.95
            notes.append(f"{sc}: {ratio_e:.2f} experts per regime, purity {pur:.3f}")
    rows.append(("P13", "G ends with 0.95-1.10 experts per regime and purity at least 0.95 (mixed and big)", ok, "; ".join(notes)))
    rows.append(("P14", "G's worst-regime accuracy is at least F8's (mixed)", mean(G, lambda r: r["min_acc"]) >= mean(mixed["F8"], lambda r: r["min_acc"]), f"G {mean(G, lambda r: r['min_acc']):.3f}, F8 {mean(mixed['F8'], lambda r: r['min_acc']):.3f}"))
    held = sum(1 for r in rows if r[2])
    L = [f"# Growth scorecard (seeds {list(seeds)})", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (ROOT / "results" / "fsmoe" / "SCORECARD_G.md").write_text(text, encoding="utf-8")
    print(text)
