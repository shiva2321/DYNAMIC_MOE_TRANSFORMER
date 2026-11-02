"""Scores the end-to-end predictions P1-P9 (docs/superpowers/specs/2026-10-31-fsmoe-e2e-design.md). ``python -m fsmoe.scorecard_e2e --seeds 710 ...`` writes results/fsmoe/SCORECARD_E2E.md."""

import argparse
from typing import Sequence

import numpy as np

from .bed import ROOT
from .report_x import load


def build(seeds: Sequence[int]) -> str:
    runs = load(seeds)
    m = lambda arm, f: float(np.mean([f(r["arms"][arm]) for r in runs]))
    rows = []
    c3 = m("A3", lambda a: a["after"]["learned"]["code"])
    rows.append(("P1", "A3 learns at least 0.80 of the 5 never-taught code skills with no human label on code", c3 >= 0.8, f"{c3:.3f}"))
    o3 = m("A3", lambda a: a["clusters_opened"])
    rows.append(("P2", "A3 opens at most 8 classes for them", o3 <= 8, f"{o3:.2f} classes"))
    c2 = m("A2", lambda a: a["after"]["learned"]["code"])
    rows.append(("P3", "A2 (tool on 10 % of code inputs) learns at least 0.40 of the code skills", c2 >= 0.4, f"{c2:.3f}"))
    l3, l1 = m("A3", lambda a: a["after"]["learned"]["all"]), m("A1", lambda a: a["after"]["learned"]["all"])
    rows.append(("P4", "A3 learns at least 0.10 more of all 28 never-taught skills than A1", l3 - l1 >= 0.10, f"A3 {l3:.3f}, A1 {l1:.3f}"))
    h3, h1 = m("A3", lambda a: a["human_labels"]), m("A1", lambda a: a["human_labels"])
    rows.append(("P5", "A3 uses at most 0.65 of A1's human labels", h3 <= 0.65 * h1, f"{h3:.0f} against {h1:.0f} ({h3 / h1:.2f})"))
    k3 = m("A3", lambda a: a["after"]["known_ho"]["code"])
    rows.append(("P6", "A3's known code skills in held-out wordings end at 0.95 or more", k3 >= 0.95, f"{k3:.3f} (before {m('A3', lambda a: a['before']['known_ho']['code']):.3f})"))
    ll, lk = m("A3", lambda a: a["after"]["learned"]["lang"]), m("A3", lambda a: a["after"]["known_ho"]["lang"])
    rows.append(("P7", "under A3 language learned <= 0.35 and language known skills in held-out wordings <= 0.85", ll <= 0.35 and lk <= 0.85, f"learned {ll:.3f}; known ho {lk:.3f}"))
    l4 = m("A4", lambda a: a["after"]["learned"]["lang"])
    rows.append(("P8", "A4 (human labels doubled) learns at least 0.05 more language skills than A3", l4 - ll >= 0.05, f"A4 {l4:.3f}, A3 {ll:.3f}"))
    noncode = sum(r["arms"][a]["tool_ran_far_noncode"] for r in runs for a in ("A2", "A3", "A4"))
    frac = float(np.mean([r["arms"]["A3"]["tool_ran_far_code"] / max(r["arms"]["A3"]["far_code_in_stream"], 1) for r in runs]))
    rows.append(("P9", "no text from outside code is ever executed, and at most 5 % of the far code snippets are executed", noncode == 0 and frac <= 0.05,
                 f"non-code texts run: {noncode}; far code snippets run in the sandbox: {frac:.3f} (running is harmless there, the sandbox has no capabilities; the behaviours it produced opened {m('A3', lambda a: a['clusters_opened']) - 5:.1f} extra classes)"))
    import json
    from .e2e import OUT
    ex = [json.loads((OUT / f"X_extra_s{r['seed']}.json").read_text(encoding="utf-8")) for r in runs if (OUT / f"X_extra_s{r['seed']}.json").exists()]
    if ex:
        me = lambda arm, f: float(np.mean([f(r["arms"][arm]) for r in ex]))
        q_a, q_c, q_k = me("A5", lambda a: a["after"]["learned"]["arith"]), me("A5", lambda a: a["after"]["learned"]["code"]), me("A5", lambda a: a["after"]["known_ho"]["code"])
        rows.append(("Q1", "A5 learns at least 0.50 of the arithmetic skills and 0.80 of the code skills, with code known skills in held-out wordings at 0.93 or more", q_a >= 0.5 and q_c >= 0.8 and q_k >= 0.93,
                     f"arithmetic {q_a:.3f} (A3r {me('A3r', lambda a: a['after']['learned']['arith']):.3f}); code {q_c:.3f}; code known ho {q_k:.3f}"))
        share = lambda arm: me(arm, lambda a: a["human_by_domain"]["arith"] / max(a["human_by_domain"]["arith"] + a["human_by_domain"]["lang"], 1))
        rows.append(("Q2", "the arithmetic share of the non-code human labels in A3 is within 0.05 of A1's (the dip is not a label-allocation effect)", abs(share("A3r") - share("A1r")) <= 0.05,
                     f"A3 {share('A3r'):.3f}, A1 {share('A1r'):.3f}"))
    held = sum(1 for r in rows if r[2])
    L = [f"# End-to-end scorecard (seeds {[r['seed'] for r in runs]})", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (ROOT / "results" / "fsmoe" / "SCORECARD_E2E.md").write_text(text, encoding="utf-8")
    print(text)
