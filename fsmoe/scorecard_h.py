"""Scores the hybrid predictions H1-H8 (docs/superpowers/specs/2026-10-31-fsmoe-hybrid-design.md). ``python -m fsmoe.scorecard_h --seeds 710 ...`` writes results/fsmoe/SCORECARD_H.md."""

import argparse
from typing import Sequence

import numpy as np

from .bed import ROOT
from .report_h import arms, load

HYB = ("H4", "H8", "H12")


def build(seeds: Sequence[int]) -> str:
    h, _ = load(seeds)
    A = arms(seeds)
    m = lambda arm, f: float(np.mean([f(x) for x in A[arm]]))
    L_ = lambda arm, d: m(arm, lambda x: x["after"]["learned"][d])
    K_ = lambda arm, d: m(arm, lambda x: x["after"]["known_ho"][d])
    rows = []
    ok1 = [a for a in HYB if L_(a, "arith") >= 0.5 and L_(a, "code") >= 0.9]
    rows.append(("H1", "at least one hybrid arm has arithmetic learned >= 0.50 AND code learned >= 0.90", bool(ok1), "; ".join(f"{a}: arithmetic {L_(a, 'arith'):.3f}, code {L_(a, 'code'):.3f}" for a in HYB)))
    rows.append(("H2", "the grid is monotone: code learned H12 >= H8 >= H4 and arithmetic learned H12 <= H4", L_("H12", "code") >= L_("H8", "code") >= L_("H4", "code") and L_("H12", "arith") <= L_("H4", "arith"),
                 f"code {L_('H4', 'code'):.3f} / {L_('H8', 'code'):.3f} / {L_('H12', 'code'):.3f}; arithmetic {L_('H4', 'arith'):.3f} / {L_('H8', 'arith'):.3f} / {L_('H12', 'arith'):.3f}"))
    rep = sum(bool(r["arms"]["D1r"].get("reproduced")) for r in h)
    rows.append(("H3", "D1r reproduces D1's learned fraction and known held-out accuracy to 1e-9 in 5 of 5 seeds", rep == len(h) == 5, f"{rep} of {len(h)} seeds"))
    rows.append(("H4", "learned fraction of all 28 between 0.33 and 0.38 in every hybrid arm", all(0.33 <= L_(a, "all") <= 0.38 for a in HYB), ", ".join(f"{a} {L_(a, 'all'):.3f}" for a in HYB)))
    rows.append(("H5", "language learned <= 0.30 in every hybrid arm", all(L_(a, "lang") <= 0.30 for a in HYB), ", ".join(f"{a} {L_(a, 'lang'):.3f}" for a in HYB)))
    rows.append(("H6", "code known skills in held-out wordings between 0.90 and 0.94 in every hybrid arm", all(0.90 <= K_(a, "code") <= 0.94 for a in HYB), ", ".join(f"{a} {K_(a, 'code'):.3f}" for a in HYB)))
    rows.append(("H7", "known skills in held-out wordings, all domains, >= 0.86 in every hybrid arm", all(K_(a, "all") >= 0.86 for a in HYB), ", ".join(f"{a} {K_(a, 'all'):.3f}" for a in HYB)))
    rows.append(("H8", "language known skills in held-out wordings <= 0.79 in every hybrid arm", all(K_(a, "lang") <= 0.79 for a in HYB), ", ".join(f"{a} {K_(a, 'lang'):.3f}" for a in HYB)))
    held = sum(1 for r in rows if r[2])
    L = [f"# Hybrid scorecard (seeds {[r['seed'] for r in h]})", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (ROOT / "results" / "fsmoe" / "SCORECARD_H.md").write_text(text, encoding="utf-8")
    print(text)
