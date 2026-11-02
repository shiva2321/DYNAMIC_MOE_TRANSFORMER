"""The worse-than-chance flag on the banking77 class-incremental stream (the real-text stream of the M5 checks; the current default pool, U-real: transfer spawn rule, router, gap-growth switches of
`unimoe.systems.phylo_cfg`): ``python -m fsmoe.chance_check_bank run --seeds 710 --arms off128 on128 off16 on16``, ``python -m fsmoe.chance_check_bank table``. One JSON per (seed, arm) in
results/unimoe/p4_chance_bank. Arms: off / on = flag None / 1.0, 128 or 16 units per expert (growth off for the 16-unit arms, as the small-start control of E-054)."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from growmoe.fairfight import run_fast
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_real import banking_incremental
from unimoe.systems import phylo_cfg

from .bed import ROOT

OUT = ROOT / "results" / "unimoe" / "p4_chance_bank"
STEPS = 20
ARMS = {"off128": {}, "on128": {"transfer_chance": 1.0}, "off16": {"hidden": 16, "grow": False}, "on16": {"hidden": 16, "grow": False, "transfer_chance": 1.0}, "onw128": {"transfer_chance": 1.0, "transfer_chance_weak_only": True}, "onw16": {"hidden": 16, "grow": False, "transfer_chance": 1.0, "transfer_chance_weak_only": True}}


def run(seed: int, arm: str) -> dict:
    path = OUT / f"bank_s{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT))
    from research.growmoe_results.m0.run_m0 import HybridFrontend
    fe = HybridFrontend(d=128, seed=0)
    st = banking_incremental(seed=seed, frontend=fe)
    t0 = time.time()
    res = run_fast(lambda: PhyloMoE(phylo_cfg("U-real", st.n_classes, seed, {"inner_steps": STEPS, "q": 0.02, "alpha_spawn": 0.01, "m_min": 32, **ARMS[arm]}, st.d)), st, seed, eval_every="final")
    kinds = {}
    for e in res.events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    out = {"seed": seed, "arm": arm, "acc": res.acc, "experts": res.report.n_experts, "params": res.report.params, "bytes": res.report.memory_bytes, "events": kinds, "seconds": time.time() - t0}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    tmp.replace(path)
    print(f"bank s{seed} {arm}: acc {out['acc']:.4f} experts {out['experts']} params {out['params']} bytes {out['bytes'] / 1e6:.2f} MB spawn {kinds.get('spawn', 0)} extend {kinds.get('extend', 0)} {out['seconds']:.0f}s", flush=True)
    return out


def table(seeds) -> str:
    L = ["| arm | accuracy | experts | spawns | extends | memory MB |", "|---|---|---|---|---|---|"]
    for arm in ARMS:
        rs = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := OUT / f"bank_s{s}_{arm}.json").exists()]
        if rs:
            L.append(f"| {arm} | {np.mean([r['acc'] for r in rs]):.4f} | {np.mean([r['experts'] for r in rs]):.1f} | {np.mean([r['events'].get('spawn', 0) for r in rs]):.1f} | {np.mean([r['events'].get('extend', 0) for r in rs]):.1f} | {np.mean([r['bytes'] for r in rs]) / 1e6:.2f} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[710])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    if a.cmd == "run":
        for s in a.seeds:
            for arm in a.arms:
                run(s, arm)
    else:
        text = table(a.seeds)
        (OUT / "TABLE_BANK.md").write_text(text, encoding="utf-8")
        print(text)
