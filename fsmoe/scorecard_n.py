"""Scores the notebook-first pool predictions P1-P12 (docs/superpowers/specs/2026-11-01-fsmoe-nbpool-design.md). ``python -m fsmoe.scorecard_n --seeds 710 ...`` writes results/fsmoe/SCORECARD_N.md."""

import argparse
from typing import Sequence

import numpy as np

from .bed import ROOT
from .report_n import REGIMES, load, n_net


def build(seeds: Sequence[int]) -> str:
    M = lambda bed, a: load(bed, a, seeds)
    mean = lambda rs, f: float(np.mean([f(r) for r in rs]))
    acc = lambda bed, a: mean(M(bed, a), lambda r: r["acc"])
    mem = lambda bed, a: mean(M(bed, a), lambda r: r["bytes"]) / 1e6
    prm = lambda bed, a: mean(M(bed, a), lambda r: r["params"])
    rows = []
    f = ("f10", "f30")
    d1 = {b: acc(b, "U128") - acc(b, "NB") for b in f}
    rows.append(("P1", "NB's accuracy is at most 0.015 below U128's on f10 and f30", all(v <= 0.015 for v in d1.values()), ", ".join(f"{b}: NB {acc(b, 'NB'):.3f} vs U128 {acc(b, 'U128'):.3f} (gap {d1[b]:+.3f})" for b in f)))
    d2 = {b: acc(b, "U128") - acc(b, "NBc") for b in f}
    rows.append(("P2", "NBc's accuracy is at most 0.01 below U128's on f10 and f30", all(v <= 0.01 for v in d2.values()), ", ".join(f"{b}: NBc {acc(b, 'NBc'):.3f} vs U128 {acc(b, 'U128'):.3f} (gap {d2[b]:+.3f})" for b in f)))
    r3 = prm("f30", "NB") / prm("f30", "U128")
    rows.append(("P3", "f30: NB's network parameters are at most 5 % of U128's", r3 <= 0.05, f"{prm('f30', 'NB'):.0f} against {prm('f30', 'U128'):.0f} ({100 * r3:.1f} %)"))
    r4 = mem("f30", "NB") / mem("f30", "U128")
    rows.append(("P4", "f30: NB's memory is at most 0.85 times U128's and NBc's at most NB's plus 0.1 MB", r4 <= 0.85 and mem("f30", "NBc") <= mem("f30", "NB") + 0.1, f"NB {mem('f30', 'NB'):.2f} MB ({r4:.2f} x U128's {mem('f30', 'U128'):.2f}), NBc {mem('f30', 'NBc'):.2f} MB"))
    rows.append(("P5", "f30: NB's accuracy is at least U32's minus 0.01 with no more memory than U32", acc("f30", "NB") >= acc("f30", "U32") - 0.01 and mem("f30", "NB") <= mem("f30", "U32"),
                 f"NB {acc('f30', 'NB'):.3f} ({mem('f30', 'NB'):.2f} MB), U32 {acc('f30', 'U32'):.3f} ({mem('f30', 'U32'):.2f} MB)"))
    er = lambda bed, a: mean(M(bed, a), lambda r: r["experts"]) / REGIMES[bed]
    rows.append(("P6", "NB keeps at least 0.97 experts per regime on f10 and f30 while U8 keeps at most 0.8 on f30", all(er(b, "NB") >= 0.97 for b in f) and er("f30", "U8") <= 0.8,
                 f"NB {er('f10', 'NB'):.2f} / {er('f30', 'NB'):.2f}, U8 on f30 {er('f30', 'U8'):.2f}"))
    tw = {k: mean(M("t", a), lambda r, k=k: r[k]) for a in ("U128", "NB") for k in ("mean_tw", "mean_ho")}
    d7 = (mean(M("t", "U128"), lambda r: r["mean_tw"]) - mean(M("t", "NB"), lambda r: r["mean_tw"]), mean(M("t", "U128"), lambda r: r["mean_ho"]) - mean(M("t", "NB"), lambda r: r["mean_ho"]))
    rows.append(("P7", "t: NB's taught-wording and new-wording accuracy are each at most 0.015 below U128's, with at most 0.95 times its memory", max(d7) <= 0.015 and mem("t", "NB") <= 0.95 * mem("t", "U128"),
                 f"gaps {d7[0]:+.3f} / {d7[1]:+.3f}; memory {mem('t', 'NB'):.2f} against {mem('t', 'U128'):.2f} MB"))
    r8 = prm("g", "NB") / prm("g", "U128")
    rows.append(("P8", "g: NB's accuracy is at least U128's minus 0.01 with at most 10 % of its network parameters", acc("g", "NB") >= acc("g", "U128") - 0.01 and r8 <= 0.10,
                 f"NB {acc('g', 'NB'):.3f} vs U128 {acc('g', 'U128'):.3f}; parameters {100 * r8:.1f} %"))
    share = {b: mean(M(b, "NB"), lambda r: n_net(r) / r["experts"]) for b in ("f10", "f30", "g", "t")}
    rows.append(("P9", "on every bed at most 30 % of NB's experts end with a network", all(v <= 0.30 for v in share.values()), ", ".join(f"{b} {100 * v:.0f} %" for b, v in share.items())))
    r10 = mean(M("f30", "NB"), lambda r: r["seconds"]) / mean(M("f30", "U128"), lambda r: r["seconds"])
    rows.append(("P10", "f30: NB's wall time is at most 1.6 times U128's", r10 <= 1.6, f"{r10:.2f} times"))
    g10, g30 = acc("f10", "NB") - acc("f10", "U8"), acc("f30", "NB") - acc("f30", "U8")
    rows.append(("P11", "NB beats U8's accuracy by at least 0.03 on f10 and 0.1 on f30", g10 >= 0.03 and g30 >= 0.1, f"f10 {g10:+.3f}, f30 {g30:+.3f}"))
    gap = acc("g", "U128") - acc("g", "U8")
    rows.append(("P12", "g: U128 is not better than U8 by more than 0.02", gap <= 0.02, f"U128 {acc('g', 'U128'):.3f}, U8 {acc('g', 'U8'):.3f} ({gap:+.3f})"))
    held = sum(1 for r in rows if r[2])
    L = [f"# Notebook-first pool scorecard (seeds {list(seeds)})", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (ROOT / "results" / "fsmoe" / "SCORECARD_N.md").write_text(text, encoding="utf-8")
    print(text)
