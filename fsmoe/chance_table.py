"""Scores the predictions W1-W6 of the worse-than-chance flag (docs/superpowers/specs/2026-11-01-fsmoe-chance-flag-design.md) and writes results/fsmoe/N/CHANCE_FLAG.md: ``python -m fsmoe.chance_table``."""

import json
from pathlib import Path

import numpy as np

from unimoe import experiment as E

from .bed import ROOT
from .chance_check import ARMS as U_ARMS, OUT as U_OUT, SCHEDULES
from .chance_check_bank import ARMS as B_ARMS, OUT as B_OUT

N = ROOT / "results" / "fsmoe" / "N"
U_SEEDS, F_SEEDS, B_SEEDS = (500, 501, 502, 503, 504), (710, 711, 712, 713, 714), (710, 711, 712)


def ucell(sch: str, arm: str, seed: int):
    base = U_OUT / f"unimoe-{sch}" / f"{arm}_s{seed}"
    if not base.with_suffix(".npz").exists():
        return None
    d = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    return float(np.load(base.with_suffix(".npz"))["R"][-1].mean()), d["per_phase"][-1]["n_experts"], d["per_phase"][-1]["events"]


def fcell(bed: str, arm: str, seed: int):
    p = N / f"{bed}_s{seed}_{arm}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build() -> str:
    rows, L = [], []
    mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
    # UniMoE suite
    u = {(sch, arm): [ucell(sch, arm, s) for s in U_SEEDS] for sch in SCHEDULES for arm in U_ARMS}
    acc = lambda sch, arm: mean([c[0] for c in u[(sch, arm)] if c])
    cells = [abs(a[0] - b[0]) for sch in ("seq", "drift") for a, b in zip(u[(sch, "ULDA-onw")], u[(sch, "ULDA-off")]) if a and b]
    near = sum(1 for g in cells if g <= 0.002)
    ok1 = all(abs(acc(s, "ULDA-onw") - acc(s, "ULDA-off")) <= 0.003 for s in ("seq", "drift")) and near >= 8
    rows.append(("W1", "UniMoE suite, ULDA: weak-only mean on seq and drift within 0.003 of flag off, and at least 8 of 10 cells within 0.002", ok1,
                 f"seq {acc('seq', 'ULDA-onw'):.4f} vs off {acc('seq', 'ULDA-off'):.4f}; drift {acc('drift', 'ULDA-onw'):.4f} vs {acc('drift', 'ULDA-off'):.4f}; {near} of {len(cells)} cells within 0.002 (unconditional rule: seq {acc('seq', 'ULDA-on'):.4f}, drift {acc('drift', 'ULDA-on'):.4f})"))
    ok2 = all(abs(acc(s, "h16-onw") - acc(s, "h16-off")) <= 0.004 for s in ("seq", "drift"))
    rows.append(("W2", "UniMoE suite, 16-unit experts: weak-only mean on seq and drift within 0.004 of flag off", ok2,
                 f"seq {acc('seq', 'h16-onw'):.4f} vs off {acc('seq', 'h16-off'):.4f}; drift {acc('drift', 'h16-onw'):.4f} vs {acc('drift', 'h16-off'):.4f}"))
    # function regimes
    reg = {"f10": 10, "f30": 30}
    f = {(bed, arm): [fcell(bed, arm, s) for s in F_SEEDS] for bed in reg for arm in ("U8", "U8x", "U8w", "U128", "U128x", "U128w")}
    facc = lambda bed, arm: mean([r["acc"] for r in f[(bed, arm)] if r])
    fer = lambda bed, arm: mean([r["experts"] / reg[bed] for r in f[(bed, arm)] if r])
    for pid, bed in (("W3", "f30"), ("W4", "f10")):
        rows.append((pid, f"{bed}: the shipped pool with 8-unit networks and the weak-only rule keeps at least 0.9 experts per regime and scores at least 0.44", fer(bed, "U8w") >= 0.9 and facc(bed, "U8w") >= 0.44,
                     f"{fer(bed, 'U8w'):.2f} experts per regime, accuracy {facc(bed, 'U8w'):.3f} (off {facc(bed, 'U8'):.3f}, unconditional {facc(bed, 'U8x'):.3f})"))
    rows.append(("W5", "f10 and f30: the shipped pool with 128-unit networks and the weak-only rule is within 0.005 of flag off", all(abs(facc(b, "U128w") - facc(b, "U128")) <= 0.005 for b in reg),
                 ", ".join(f"{b}: {facc(b, 'U128w'):.4f} vs {facc(b, 'U128'):.4f} (unconditional {facc(b, 'U128x'):.4f})" for b in reg)))
    # banking
    bk = {arm: [json.loads(p.read_text(encoding="utf-8")) for s in B_SEEDS if (p := B_OUT / f"bank_s{s}_{arm}.json").exists()] for arm in B_ARMS}
    bacc = lambda arm: mean([r["acc"] for r in bk[arm]])
    if all(bk[a] for a in ("off128", "onw128", "off16", "onw16")):
        rows.append(("W6", "banking77: weak-only with 128-unit experts within 0.01 of flag off, with 16-unit experts at least off minus 0.01", abs(bacc("onw128") - bacc("off128")) <= 0.01 and bacc("onw16") >= bacc("off16") - 0.01,
                     f"128 units {bacc('onw128'):.4f} vs {bacc('off128'):.4f} (unconditional {bacc('on128'):.4f}); 16 units {bacc('onw16'):.4f} vs {bacc('off16'):.4f} (unconditional {bacc('on16'):.4f})"))
    held = sum(1 for r in rows if r[2])
    L += [f"# The worse-than-chance flag: standard checks and predictions W1-W6", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    L += ["", "## UniMoE real suite (seven text domains; seeds 500-504; mean final accuracy, experts, spawns, extends)", "", "| schedule | arm | accuracy | experts | spawns | extends | max gap to the stored E-054 cell (flag off) |", "|---|---|---|---|---|---|---|"]
    for sch in SCHEDULES:
        for arm in U_ARMS:
            cs = [c for c in u[(sch, arm)] if c]
            if cs:
                gap = "-"
                if arm == "ULDA-off":
                    gs = [abs(c[0] - float(np.load(E.OUT / "p4_main" / f"unimoe-{sch}" / f"ULDA_s{s}.npz")["R"][-1].mean())) for s, c in zip(U_SEEDS, u[(sch, arm)]) if c]
                    gap = f"{max(gs):.2e}"
                L.append(f"| {sch} | {arm} | {mean([c[0] for c in cs]):.4f} | {mean([c[1] for c in cs]):.2f} | {mean([c[2].get('spawn', 0) for c in cs]):.2f} | {mean([c[2].get('extend', 0) for c in cs]):.2f} | {gap} |")
    L += ["", "## Function regimes (seeds 710-714; accuracy, experts per regime)", "", "| bed | arm | accuracy | experts per regime |", "|---|---|---|---|"]
    for bed in reg:
        for arm in ("U8", "U8x", "U8w", "U128", "U128x", "U128w"):
            if any(f[(bed, arm)]):
                L.append(f"| {bed} | {arm} | {facc(bed, arm):.4f} | {fer(bed, arm):.2f} |")
    L += ["", "## banking77 class-incremental (seeds 710-712)", "", "| arm | accuracy | experts | spawns | extends |", "|---|---|---|---|---|"]
    for arm in B_ARMS:
        if bk[arm]:
            L.append(f"| {arm} | {bacc(arm):.4f} | {mean([r['experts'] for r in bk[arm]]):.1f} | {mean([r['events'].get('spawn', 0) for r in bk[arm]]):.1f} | {mean([r['events'].get('extend', 0) for r in bk[arm]]):.1f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    text = build()
    (N / "CHANCE_FLAG.md").write_text(text, encoding="utf-8")
    print(text)
