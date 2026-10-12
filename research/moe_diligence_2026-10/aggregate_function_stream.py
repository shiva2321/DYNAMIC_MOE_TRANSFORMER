"""Combine the per-seed raw files of the function-regime pilot into one table (mean [95% CI] over all seeds found).
The oracle row's memory is recomputed here (one 41,792-parameter expert per regime, float32 = 167.2 KB each); the per-run
printout of spike_function_stream.py mis-scaled that one column."""
import glob
import json
import sys

from dil_common import *  # noqa

raw = {}
files = sorted(glob.glob("function_stream_raw_seeds_*.json"))
for f in files:
    for k, v in json.load(open(f)).items():
        raw.setdefault(k, []).extend(v)
print("files combined:", files)
sizes = sorted({int(k.split("|")[0]) for k in raw})
order = ["kNN stores everything", "kNN same bytes as PhyloMoE", "Replay net 1 step(s)", "Replay net 20 step(s)",
         "PhyloMoE 1 step(s)", "PhyloMoE 20 step(s)", "Oracle: one expert per regime"]
n_seeds = len(raw[f"{sizes[0]}|{order[0]}"])
print(f"\n=== Function regimes (8 classes, chance 12.5%): final average accuracy %, {n_seeds} seeds, mean [95% CI] ===")
print(f"{'regimes':>7s} {'method':32s} {'accuracy':>22s} {'experts':>8s} {'memory KB':>10s}")
for n in sizes:
    for key in order:
        rows = raw[f"{n}|{key}"]
        mem = n * 167.2 if key.startswith("Oracle") else float(np.mean([r[2] for r in rows]))
        print(f"{n:7d} {key:32s} {fmt([r[0] for r in rows]):>22s} {np.mean([r[1] for r in rows]):8.1f} {mem:10.0f}")
    print()
print("paired difference PhyloMoE(20 steps) minus Replay net(20 steps), percentage points:")
for n in sizes:
    a = [r[0] for r in raw[f"{n}|PhyloMoE 20 step(s)"]]
    b = [r[0] for r in raw[f"{n}|Replay net 20 step(s)"]]
    d = [x - y for x, y in zip(a, b)]
    m, lo, hi = mean_ci(d)
    print(f"  {n:3d} regimes: {m*100:+.1f} [{lo*100:+.1f}, {hi*100:+.1f}]  per-seed: {[round(x*100,1) for x in d]}")
