"""Scores the predictions S1-S7 of the skip flag (docs/superpowers/specs/2026-11-01-fsmoe-skip-design.md) and writes results/fsmoe/K/SKIP_CHECK.md: ``python -m fsmoe.skip_table``."""

import json
import re
from pathlib import Path

import numpy as np

from .bed import ROOT
from .skip_check import OUT, U_OUT

U_SEEDS, F_SEEDS, B_SEEDS = (500, 501, 502, 503, 504), (710, 711, 712, 713, 714), (710, 711, 712)
U_SCH = ("seq", "mix", "drift")
STORED = ROOT / "results" / "unimoe" / "p4_chance"
PYTEST_SUMMARY = OUT / "pytest_summary.txt"
S5_SPEC = {("seq", "k0.1"): (0.05, 0.15, 0.004), ("seq", "k0.25"): (0.12, 0.35, 0.006), ("mix", "k0.1"): (0.0, 0.02, 0.002), ("mix", "k0.25"): (0.0, 0.10, 0.004),
           ("drift", "k0.1"): (0.0, 0.02, 0.002), ("drift", "k0.25"): (0.0, 0.10, 0.004)}      # (rows skipped from, to, accuracy tolerance): fixed in the design document from the dev cells BEFORE the report-seed UniMoE cells were run


def cell(bed: str, seed: int, arm: str):
    p = OUT / f"{bed}_s{seed}_{arm}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cells(bed: str, seeds, arm: str):
    return [c for c in (cell(bed, s, arm) for s in seeds) if c]


def rows_skipped(c: dict) -> float:
    s = c["skip"]
    n = s["trained_rows"] + s["skipped_rows"]
    return s["skipped_rows"] / n if n else 0.0


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float("nan")


def acc_gap(bed: str, seeds, arm: str) -> float:
    off, on = cells(bed, seeds, "off"), cells(bed, seeds, arm)
    return mean([c["acc"] for c in on]) - mean([c["acc"] for c in off])


def final_mean(path: Path) -> float:
    return float(np.load(path)["R"][-1].mean())


def build() -> str:
    rows, L = [], []
    # S1: the off cells equal the stored flag-off cells
    gaps = []
    for sch in U_SCH:
        for s in U_SEEDS:
            new, old = U_OUT / f"unimoe-{sch}" / f"off_s{s}.npz", STORED / f"unimoe-{sch}" / f"ULDA-off_s{s}.npz"
            if new.exists() and old.exists():
                gaps.append(abs(final_mean(new) - final_mean(old)))
    suite = PYTEST_SUMMARY.read_text(encoding="utf-8").strip() if PYTEST_SUMMARY.exists() else "(not recorded)"
    m = re.search(r"(\d+) passed", suite)
    ok1 = len(gaps) == 15 and max(gaps) <= 1e-9 and m is not None and int(m.group(1)) >= 1101
    rows.append(("S1", "the full suite passes (at least 1,101 tests) and the 15 flag-off UniMoE cells equal the stored E-062 cells to 1e-9", ok1, f"suite: {suite}; {len(gaps)} cells compared, largest gap {max(gaps) if gaps else float('nan'):.2e}"))
    # S2: hard streams
    ok2, parts = True, []
    for bed in ("f10", "f30"):
        for arm in ("k0.1", "k0.25"):
            cs = cells(bed, F_SEEDS, arm)
            frac = mean([c["skip"]["skipped"] / max(c["skip"]["trained"] + c["skip"]["skipped"], 1) for c in cs])
            gap = acc_gap(bed, F_SEEDS, arm)
            ok2 &= bool(cs) and frac < 0.02 and abs(gap) <= 0.002
            parts.append(f"{bed} {arm}: {frac:.1%} of batches, accuracy {gap:+.4f}")
    rows.append(("S2", "f10 and f30: at both thresholds under 2 % of batches skipped and accuracy within 0.002 of off", ok2, "; ".join(parts)))
    # S3: text bed
    t01, t25 = cells("t", F_SEEDS, "k0.1"), cells("t", F_SEEDS, "k0.25")
    ok3 = bool(t01 and t25) and mean([rows_skipped(c) for c in t01]) >= 0.40 and abs(acc_gap("t", F_SEEDS, "k0.1")) <= 0.003 and mean([rows_skipped(c) for c in t25]) >= 0.70 and abs(acc_gap("t", F_SEEDS, "k0.25")) <= 0.006
    rows.append(("S3", "text bed: 0.1 skips at least 40 % of rows within 0.003 of off; 0.25 skips at least 70 % within 0.006", ok3,
                 f"0.1: {mean([rows_skipped(c) for c in t01]):.1%} of rows, accuracy {acc_gap('t', F_SEEDS, 'k0.1'):+.4f}; 0.25: {mean([rows_skipped(c) for c in t25]):.1%}, {acc_gap('t', F_SEEDS, 'k0.25'):+.4f}"))
    # S4: banking
    b01, b25 = cells("bank", B_SEEDS, "k0.1"), cells("bank", B_SEEDS, "k0.25")
    r01, r25 = mean([rows_skipped(c) for c in b01]), mean([rows_skipped(c) for c in b25])
    ok4 = bool(b01 and b25) and r01 < 0.10 and abs(acc_gap("bank", B_SEEDS, "k0.1")) <= 0.005 and 0.10 <= r25 <= 0.30 and abs(acc_gap("bank", B_SEEDS, "k0.25")) <= 0.01
    rows.append(("S4", "banking77: 0.1 skips under 10 % of rows within 0.005 of off; 0.25 skips 10-30 % within 0.01", ok4, f"0.1: {r01:.1%}, {acc_gap('bank', B_SEEDS, 'k0.1'):+.4f}; 0.25: {r25:.1%}, {acc_gap('bank', B_SEEDS, 'k0.25'):+.4f}"))
    # S5: UniMoE real suite
    ok5, parts = True, []
    for (sch, arm), (lo, hi, tol) in S5_SPEC.items():
        cs = cells(f"u{sch}", U_SEEDS, arm)
        r, gap = mean([rows_skipped(c) for c in cs]), acc_gap(f"u{sch}", U_SEEDS, arm)
        ok5 &= bool(cs) and lo <= r <= hi and abs(gap) <= tol
        parts.append(f"{sch} {arm}: {r:.1%} of rows (declared {lo:.0%}-{hi:.0%}), accuracy {gap:+.4f} (within {tol})")
    rows.append(("S5", "UniMoE suite: share of rows skipped inside the declared range and accuracy within the declared tolerance, six cells", ok5, "; ".join(parts)))
    # S6: the saving is small
    beds = [("f10", F_SEEDS), ("f30", F_SEEDS), ("t", F_SEEDS), ("bank", B_SEEDS)] + [(f"u{s}", U_SEEDS) for s in U_SCH]
    ok6, parts = True, []
    for bed, seeds in beds:
        off = cells(bed, seeds, "off")
        share = mean([c["train_seconds"] / c["observe_seconds"] for c in off])
        saves = {arm: 1 - mean([c["observe_seconds"] for c in cells(bed, seeds, arm)]) / mean([c["observe_seconds"] for c in off]) for arm in ("k0.1", "k0.25") if cells(bed, seeds, arm) and off}
        ok6 &= bool(off) and share <= 0.35 and all(v <= 0.25 for v in saves.values())
        parts.append(f"{bed}: train share {share:.1%}, saved " + ", ".join(f"{a} {v:+.1%}" for a, v in saves.items()))
    rows.append(("S6", "train_expert takes at most 35 % of observe time with the flag off, and no arm saves more than 25 % of observe time", ok6, "; ".join(parts)))
    # S7: experts within 1 (0.1) and 2 (0.25)
    worst = {"k0.1": 0, "k0.25": 0}
    for bed, seeds in beds:
        for s in seeds:
            o = cell(bed, s, "off")
            for arm in worst:
                k = cell(bed, s, arm)
                if o and k:
                    worst[arm] = max(worst[arm], abs(o["experts"] - k["experts"]))
    ok7 = worst["k0.1"] <= 1 and worst["k0.25"] <= 2
    rows.append(("S7", "the number of experts is within 1 of off at 0.1 and within 2 at 0.25 in every cell", ok7, f"largest difference: 0.1 {worst['k0.1']}, 0.25 {worst['k0.25']}"))
    held = sum(1 for r in rows if r[2])
    L += ["# The skip flag: standard checks and predictions S1-S7", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    L += ["", "## Per bed (mean over seeds; accuracy, experts, batches and rows skipped, wall seconds inside train_expert / observe)", "", "| bed | arm | seeds | accuracy | experts | batches skipped | rows skipped | train s | observe s |", "|---|---|---|---|---|---|---|---|---|"]
    for bed, seeds in beds:
        for arm in ("off", "k0.1", "k0.25"):
            cs = cells(bed, seeds, arm)
            if cs:
                bs = mean([c["skip"]["skipped"] / max(c["skip"]["trained"] + c["skip"]["skipped"], 1) for c in cs])
                L.append(f"| {bed} | {arm} | {len(cs)} | {mean([c['acc'] for c in cs]):.4f} | {mean([c['experts'] for c in cs]):.1f} | {bs:.1%} | {mean([rows_skipped(c) for c in cs]):.1%} | {mean([c['train_seconds'] for c in cs]):.1f} | {mean([c['observe_seconds'] for c in cs]):.1f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    text = build()
    (OUT / "SKIP_CHECK.md").write_text(text, encoding="utf-8")
    print(text)
