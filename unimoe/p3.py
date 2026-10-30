"""P3 experiments: what explains the evaluation. Every cell is cached like the main grid (same stamp, same runner); nothing here changes what a cell contains.

    python -m unimoe.p3 expert-quality      # E1: oracle experts told the regime, linear vs MLP vs wider vs longer-trained
    python -m unimoe.p3 ablate              # E3: U-real minus one component (sequential)
    python -m unimoe.p3 scale               # E4: experts that start too small (16 units): does growth rescue them?
    python -m unimoe.p3 table NAME          # print the final means of the cells of one experiment
"""

import argparse
import json
import sys
from typing import Dict, List

import numpy as np

from . import experiment as E
from . import systems as SY

SEEDS = (500, 501, 502)

# tag -> (system, recipe override on top of the tuned recipe of the system it extends)
EXPERT_QUALITY: Dict[str, tuple] = {
    "oracle_mlp128_s20": ("oracle_experts", {}),                                           # the main grid's oracle experts (cached cell, reproduced here for the table)
    "oracle_linear_s20": ("oracle_experts", {"depth": 0}),
    "oracle_linear_s50": ("oracle_experts", {"depth": 0, "inner_steps": 50}),
    "oracle_mlp128_s50": ("oracle_experts", {"inner_steps": 50}),
    "oracle_mlp512_s20": ("oracle_experts", {"hidden": 512}),
    "oracle_mlp512_s50": ("oracle_experts", {"hidden": 512, "inner_steps": 50}),
}
FAST = {"grow_min_age": 30, "grow_window": 20, "grow_cooldown": 20}          # the maturity condition scaled to streams of 150-250 batches (the defaults need 300 batches per expert)
SCALE: Dict[str, tuple] = {
    "shipped-h16": ("shipped", {"hidden": 16}),
    "U-real-h16": ("U-real", {"hidden": 16}),
    "U-deep-h16": ("U-deep", {"hidden": 16}),
    "U-real-no-growth-h16": ("U-real-no-growth", {"hidden": 16}),
    "U-real-h512": ("U-real", {"hidden": 512}),
    "U-real-h16-fast": ("U-real", {"hidden": 16, **FAST}),
    "U-real-gap0.35-fast": ("U-real", {"grow_gap": 0.35, **FAST}),
    "U-real-gap0.50-fast": ("U-real", {"grow_gap": 0.50, **FAST}),
}


GROW_GAP: Dict[str, tuple] = {f"U-real-gap{g}": ("U-real", {"grow_gap": g}) for g in (0.25, 0.35, 0.50)}
GROW_GAP_FAST: Dict[str, tuple] = {f"U-real-fast-gap{g}": ("U-real", {"grow_gap": g, **FAST}) for g in (0.17, 0.35, 0.50)}


def run_variants(group: str, variants: Dict[str, tuple], schedules=("seq",), seeds=SEEDS) -> None:
    tuned = E.load_tuning()
    runs = E.OUT / group
    for seed in seeds:
        for sch in schedules:
            st = E.streams(seed)[sch]
            for tag, (system, extra) in variants.items():
                base = dict(tuned.get(system) or tuned.get("U-real" if system.startswith("U-real") else system))
                rec = E.cell(system, st, seed, {**base, **extra}, runs, tag=tag)
                print(f"seed {seed} {sch:<5s} {tag:<24s} final mean {E.final_mean(rec):.3f} experts {rec.per_phase[-1]['n_experts']:>3d} grow events {rec.per_phase[-1]['events'].get('grow', 0)} {rec.seconds:5.0f}s", flush=True)


def table(group: str) -> List[str]:
    from . import report as RP
    recs = RP.collect(E.OUT / group)
    tags = sorted({k[1] for k in recs})
    rows = []
    for tag in tags:
        for sch in sorted({k[0] for k in recs if k[1] == tag}):
            ks = [k for k in sorted(recs) if k[1] == tag and k[0] == sch]
            v = [E.final_mean(recs[k]) for k in ks]
            ex = [recs[k].per_phase[-1]["n_experts"] for k in ks]
            gr = [recs[k].per_phase[-1]["events"].get("grow", 0) for k in ks]
            pr = [recs[k].report["params"] for k in ks]
            rows.append(f"{tag:<26s} {sch:<5s} final mean {np.mean(v):.3f} (seeds {', '.join(f'{x:.3f}' for x in v)}) experts {np.mean(ex):.1f} grow events {np.mean(gr):.1f} params {np.mean(pr):.0f}")
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("expert-quality", "ablate", "scale", "grow-gap", "grow-gap-fast", "table"))
    ap.add_argument("name", nargs="?", default="")
    a = ap.parse_args(argv)
    if a.cmd == "expert-quality":
        run_variants("p3_expert_quality", EXPERT_QUALITY)
    elif a.cmd == "ablate":
        E.run(SEEDS, ("seq",), list(SY.ABLATION_SYSTEMS))
    elif a.cmd == "scale":
        run_variants("p3_scale", SCALE)
    elif a.cmd == "grow-gap":
        run_variants("p3_grow_gap", GROW_GAP, schedules=("mix",))
    elif a.cmd == "grow-gap-fast":
        run_variants("p3_grow_gap_fast", GROW_GAP_FAST, schedules=("mix",))
    else:
        print("\n".join(table(a.name)))


if __name__ == "__main__":
    main()
