"""Runs the growth arms on a scenario (``python -m fsmoe.grow.lab run --scenario mixed --seeds 700 --arms G F8 ...``) and writes one JSON per (scenario, seed, arm) to results/fsmoe/G, atomically,
so a power cut loses one arm at most; finished files are skipped on a re-run. ``oracle`` writes the answer key of a scenario and seed first (``python -m fsmoe.grow.lab oracle --scenario mixed --seeds 700``)."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from fsmoe.bed import ROOT

from .governor import BruteGovernor, GovCfg, Governor
from .system import GrowSystem, SysCfg, evaluate
from .world import SCENARIOS, GrowthWorld, oracle_table

OUT = ROOT / "results" / "fsmoe" / "G"
GROUPS = {"big": 8}                                                    # the big scenario clusters its 48 regimes into 8 groups of 6
GATE_BUDGET = 640.0                                                     # 20 experts x 32 inputs: flat gate multiply-adds per sample above which the governor may build a tier

ARMS: Dict[str, dict] = {
    "F8": dict(sys=dict(h0=8, depth0=1)),
    "F16": dict(sys=dict(h0=16, depth0=1)),
    "F32": dict(sys=dict(h0=32, depth0=1)),
    "F64": dict(sys=dict(h0=64, depth0=1)),
    "F8d3": dict(sys=dict(h0=8, depth0=3)),
    "G": dict(sys=dict(net_at_birth=False), gov=GovCfg),
    "Gng": dict(sys=dict(net_at_birth=False), gov=lambda: GovCfg(grow=False)),
    "GnoNB": dict(sys=dict(net_at_birth=True), gov=GovCfg),
    "Gnt": dict(sys=dict(net_at_birth=False), gov=lambda: GovCfg(tier=False)),
    "BF": dict(sys=dict(net_at_birth=True), brute=True),
    # tuning variants of G, run on dev seeds 700-702 only (the default G is the cheapest one that matched the best accuracy, 0.733 on average)
    "Ga": dict(sys=dict(net_at_birth=False), gov=lambda: GovCfg(promote_steps=300, trial_ratio=2.0)),
    "Gb": dict(sys=dict(net_at_birth=False), gov=lambda: GovCfg(promote_steps=150, trial_ratio=2.0)),
    "Gc": dict(sys=dict(net_at_birth=False), gov=lambda: GovCfg(promote_steps=200, trial_ratio=2.0, promote_sizes=((8, 1), (8, 3), (32, 1)))),
}


def build(arm: str, seed: int, scenario: str) -> GrowSystem:
    spec = ARMS[arm]
    cfg = SysCfg(seed=seed, gate_budget=GATE_BUDGET, **spec["sys"])
    gov = BruteGovernor() if spec.get("brute") else (Governor(spec["gov"]()) if "gov" in spec else None)
    return GrowSystem(cfg, gov)


def score_structure(system: GrowSystem, world: GrowthWorld, ev: dict, oracle: Optional[dict]) -> dict:
    """What the pool looks like against the answer key: experts against regimes, and which regimes got the growth their oracle need asked for."""
    reg_count = Counter()
    pure_num, pure_den = 0.0, 0.0
    for e in system.experts.values():
        tot = sum(e.regimes.values())
        if tot:
            pure_num += max(e.regimes.values())
            pure_den += tot
    served = {}
    for r in world.regimes:
        best = max(system.experts.values(), key=lambda e: e.regimes.get(r.rid, 0), default=None)
        served[r.rid] = None if best is None else best.id
        reg_count[r.rid] = 0 if best is None else best.regimes.get(r.rid, 0)
    out = {"experts": len(system.experts), "regimes": len(world.regimes), "purity": pure_num / max(pure_den, 1.0), "regimes_with_own_expert": float(np.mean([
        served[r.rid] is not None and max(system.experts[served[r.rid]].regimes, key=system.experts[served[r.rid]].regimes.get) == r.rid for r in world.regimes]))}
    if oracle is not None:
        rows = []
        for r in world.regimes:
            need = oracle[r.rid]["need"]
            e = None if served[r.rid] is None else system.experts[served[r.rid]]
            h, depth = (0, 0) if e is None else e.size()
            grew = {"width": h > system.cfg.h0, "depth": depth > system.cfg.depth0}
            rows.append({"regime": r.rid, "kind": r.kind, "need": need, "hidden": h, "depth": depth, "grew_width": bool(grew["width"]), "grew_depth": bool(grew["depth"]), "acc": ev["acc"][r.rid]})
        out["rows"] = rows
    return out


def run_arm(scenario: str, seed: int, arm: str, out_dir: Path = OUT) -> dict:
    path = out_dir / f"{scenario}_s{seed}_{arm}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    world = GrowthWorld(seed, SCENARIOS[scenario], n_groups=GROUPS.get(scenario, 0))
    opath = out_dir / f"oracle_{scenario}_s{seed}.json"
    oracle = {int(k): v for k, v in json.loads(opath.read_text(encoding="utf-8")).items()} if opath.exists() else None
    system = build(arm, seed, scenario)
    t0 = time.time()
    for b in world.stream():
        system.observe(b.x, b.y, torch.full((b.x.shape[0],), b.regime))
    ev = evaluate(system, world)
    res = {"scenario": scenario, "seed": seed, "arm": arm, "seconds": time.time() - t0, "mean_acc": ev["mean"], "min_acc": ev["min"], "macs_answer": ev["macs_answer"], "acc": ev["acc"],
           "summary": system.summary(), "structure": score_structure(system, world, ev, oracle), "events": system.events}
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    p = res["summary"]["params"]
    print(f"{scenario} s{seed} {arm}: mean acc {res['mean_acc']:.3f} min {res['min_acc']:.3f} experts {res['summary']['experts']}/{len(world.regimes)} params {p['total']} (net {p['net']}) "
          f"macs/answer {res['macs_answer']:.0f} events {res['summary']['events']} {res['seconds']:.0f}s", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "oracle"])
    ap.add_argument("--scenario", default="mixed", choices=list(SCENARIOS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    a = ap.parse_args()
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    for s in a.seeds:
        if a.cmd == "oracle":
            oracle_table(GrowthWorld(s, SCENARIOS[a.scenario], n_groups=GROUPS.get(a.scenario, 0)), a.scenario, OUT, only=("wide", "deep") if a.scenario == "big" else ("lin", "wide", "deep"))
        else:
            for arm in a.arms:
                run_arm(a.scenario, s, arm)
