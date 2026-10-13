"""Compact vote-vs-set-gate table for the notebook and the report (reads the integration JSON files; numbers are not typed by hand).   (entry E-023)

    python research/growmoe_results/rv3/final_table.py [variant]      default variant: md_windows_loss
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
variant = sys.argv[1] if len(sys.argv) > 1 else "md_windows_loss"
rows = [r for f in sorted(HERE.glob("spike_set_gate_g*.json")) for r in json.loads(f.read_text(encoding="utf-8"))]
streams = ["OV-0.85", "OV-0.6", "OV-0.3", "OV-0.3-MIXED", "OV-0.0", "F-15", "S1", "S2-5", "S2-30"]
print(f"vote (current default) vs {variant}; dev seeds 100-102; accuracy %, mean [min, max] of the per-seed difference")
print("stream          vote   gate   diff [min, max]       experts vote -> gate / regimes   known batches called novel   set-novel calls")
for s in streams:
    v = {r["seed"]: r for r in rows if r["stream"] == s and r["variant"] == "vote"}
    g = {r["seed"]: r for r in rows if r["stream"] == s and r["variant"] == variant}
    seeds = sorted(set(v) & set(g))
    if not seeds:
        continue
    d = [100 * (g[i]["acc"] - v[i]["acc"]) for i in seeds]
    kn = sum(g[i]["known_novel"] + g[i]["known_known"] + g[i]["known_skip"] + g[i]["known_none"] for i in seeds)
    print(f"{s:14s} {100 * np.mean([v[i]['acc'] for i in seeds]):6.1f} {100 * np.mean([g[i]['acc'] for i in seeds]):6.1f}  {np.mean(d):+6.1f} [{min(d):+5.1f}, {max(d):+5.1f}]   "
          f"{np.mean([v[i]['experts'] for i in seeds]):5.1f} -> {np.mean([g[i]['experts'] for i in seeds]):5.1f} / {g[seeds[0]]['regimes']:2d}"
          f"{sum(g[i]['known_novel'] for i in seeds):22d} / {kn:4d}{np.mean([g[i]['set_novel'] for i in seeds]):18.1f}   (n={len(seeds)})")
single = [r for r in rows if r["variant"] == variant and r["stream"] not in ("OV-0.3-MIXED",)]
kn = sum(r["known_novel"] + r["known_known"] + r["known_skip"] + r["known_none"] for r in single)
fa = sum(r["known_novel"] for r in single)
print(f"\npooled over the single-source streams (all but OV-0.3-MIXED): known batches called novel {fa} / {kn} = {100 * fa / max(kn, 1):.2f} %")
