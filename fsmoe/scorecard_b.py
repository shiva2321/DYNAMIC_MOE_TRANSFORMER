"""Scores the balance predictions M1-M8 (docs/superpowers/specs/2026-10-31-fsmoe-balance-design.md). ``python -m fsmoe.scorecard_b --seeds 710 ...`` writes results/fsmoe/SCORECARD_B.md."""

import argparse
from typing import Sequence

import numpy as np

from .bed import ROOT
from .report_b import load


def build(seeds: Sequence[int]) -> str:
    runs = load(seeds)
    m = lambda arm, f: float(np.mean([f(r["arms"][arm]) for r in runs]))
    L_ = lambda arm, d: m(arm, lambda a: a["after"]["learned"][d])
    K_ = lambda arm, d: m(arm, lambda a: a["after"]["known_ho"][d])
    rows = []
    rows.append(("M1", "D1 raises arithmetic learned to >= 0.50 and keeps code learned >= 0.80", L_("D1", "arith") >= 0.5 and L_("D1", "code") >= 0.8, f"arithmetic {L_('D1', 'arith'):.3f} (A3b {L_('A3b', 'arith'):.3f}); code {L_('D1', 'code'):.3f}"))
    rows.append(("M2", "D1's code known skills in held-out wordings end between 0.93 and 0.95", 0.93 <= K_("D1", "code") <= 0.95, f"{K_('D1', 'code'):.3f} (A3b {K_('A3b', 'code'):.3f})"))
    rows.append(("M3", "D2 raises arithmetic learned to >= 0.50 AND keeps code learned >= 0.95 AND code known skills in held-out wordings >= 0.95",
                 L_("D2", "arith") >= 0.5 and L_("D2", "code") >= 0.95 and K_("D2", "code") >= 0.95, f"arithmetic {L_('D2', 'arith'):.3f}; code {L_('D2', 'code'):.3f}; code known ho {K_('D2', 'code'):.3f}"))
    rows.append(("M4", "language learned stays <= 0.30 under D1 and D2", max(L_("D1", "lang"), L_("D2", "lang")) <= 0.30, f"D1 {L_('D1', 'lang'):.3f}, D2 {L_('D2', 'lang'):.3f} (A3b {L_('A3b', 'lang'):.3f})"))
    rows.append(("M5", "D2's learned fraction of all 28 is >= 0.34", L_("D2", "all") >= 0.34, f"{L_('D2', 'all'):.3f} (A3b {L_('A3b', 'all'):.3f})"))
    rows.append(("M6", "D2's known skills in held-out wordings, all domains, end at >= 0.88", K_("D2", "all") >= 0.88, f"{K_('D2', 'all'):.3f} (A3b {K_('A3b', 'all'):.3f})"))
    rows.append(("M7", "D2's language known skills in held-out wordings stay >= 0.78", K_("D2", "lang") >= 0.78, f"{K_('D2', 'lang'):.3f} (A3b {K_('A3b', 'lang'):.3f})"))
    acc = m("D1", lambda a: a["domain_head_accuracy"])
    rows.append(("M8", "the domain head puts >= 0.98 of the human-labelled items in the right domain", acc >= 0.98, f"{acc:.3f}"))
    import json
    from .e2e import B_OUT
    pr = [json.loads((B_OUT / f"B_priors_s{r['seed']}.json").read_text(encoding="utf-8")) for r in runs if (B_OUT / f"B_priors_s{r['seed']}.json").exists()]
    if pr:
        mp = lambda arm, f: float(np.mean([f(r["arms"][arm]) for r in pr]))
        a_a, a_c, a_k = mp("A3u", lambda a: a["after"]["learned"]["arith"]), mp("A3u", lambda a: a["after"]["learned"]["code"]), mp("A3u", lambda a: a["after"]["known_ho"]["code"])
        rows.append(("M9", "A3u learns >= 0.50 of the arithmetic skills and >= 0.95 of the code skills with code known skills in held-out wordings >= 0.95", a_a >= 0.5 and a_c >= 0.95 and a_k >= 0.95,
                     f"arithmetic {a_a:.3f}; code {a_c:.3f}; code known ho {a_k:.3f}"))
        a_all = mp("A3u", lambda a: a["after"]["learned"]["all"])
        rows.append(("M10", "A3u learns >= 0.36 of all 28 never-taught skills", a_all >= 0.36, f"{a_all:.3f}"))
        a_ho = mp("A3u", lambda a: a["after"]["known_ho"]["all"])
        rows.append(("M11", "A3u's known skills in held-out wordings, all domains, end at >= 0.88", a_ho >= 0.88, f"{a_ho:.3f}"))
        d_u, d_o = mp("D2u", lambda a: a["after"]["learned"]["arith"]), L_("D2", "arith")
        rows.append(("M12", "D2u's arithmetic learned is >= D2's + 0.10", d_u >= d_o + 0.10, f"D2u {d_u:.3f}, D2 {d_o:.3f}"))
    held = sum(1 for r in rows if r[2])
    L = [f"# Domain-balanced training scorecard (seeds {[r['seed'] for r in runs]})", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (ROOT / "results" / "fsmoe" / "SCORECARD_B.md").write_text(text, encoding="utf-8")
    print(text)
