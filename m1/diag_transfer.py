"""Why does Baseline v1 make 61-67 experts for 10 regimes on B1 (E-068, C-195)? Instrument the transfer verdict of a Baseline v1 pool and test the founding-cluster repair.

Development seeds 800-804 only; N = 10. Each run logs every call of ``PhyloMoE._transfer_verdict`` (instance-level wrapper, no file of growmoe/ is edited): did the test run, did it find a conflict,
how many experts had enough reference losses, and on which step it happened; a spawn on the step right after another spawn is a "consecutive" spawn (a chain reaction on one regime).
Configurations: as built at q 0.05; with only the twin fix (sig_dedupe, infant_relax 1: the repair of F-31 / E-037, off by default); with U-real; q 0.02 as built.
Run: python -m m1.diag_transfer --out results/m1/dev/diag_transfer_B1_n10.json      (development diagnosis, not an M1 result)
"""

from __future__ import annotations

import argparse
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from costlab.env import environment, fixed_threads
from growmoe.config import PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams_m1 import function_regimes
from m1.cell import save_cell_record

U_REAL = dict(set_gate=True, set_gate_confirm=2, sig_dedupe=True, infant_relax=1.0, growth_rule="gap")     # unimoe.systems.VARIANT_SWITCHES["U-real"]
TWIN_FIX = dict(sig_dedupe=True, infant_relax=1.0)
CONFIGS: List[Tuple[str, float, Dict[str, Any]]] = [
    ("q 0.05 as built", 0.05, {}),
    ("q 0.05 with the twin fix only", 0.05, TWIN_FIX),
    ("q 0.05 with U-real", 0.05, U_REAL),
    ("q 0.02 as built", 0.02, {}),
]


def run_one(seed: int, q: float, overrides: Dict[str, Any], n_regimes: int = 10) -> Dict[str, Any]:
    """Train a Baseline v1 pool (5 steps per batch, pool seed = seed + 7) on B1 and log what its transfer verdict did."""
    st = function_regimes(seed=seed, n_regimes=n_regimes)
    pool = PhyloMoE(PhyloCfg(d=st.d, n_classes=st.n_classes, seed=seed + 7, inner_steps=5, route_rule="accept_then_logf", q=q, **overrides))
    calls: List[Dict[str, Any]] = []
    original = pool._transfer_verdict

    def verdict(x_taken, y_taken, force_parent):
        ready = sum(1 for r in pool.experts.values() if len(r.meta.get("ref", ())) >= pool.cfg.transfer_min_ref)
        v = original(x_taken, y_taken, force_parent)
        calls.append({"step": int(pool.step), "n_experts": len(pool.experts), "n_ready": ready, "n_cluster": int(len(x_taken)), "ran": v is not None,
                      "conflict": None if v is None else bool(v.conflict), "loss_cluster": None if v is None else float(v.loss_cluster), "loss_ref": None if v is None else float(v.loss_ref)})
        return v

    pool._transfer_verdict = verdict
    t0 = time.time()
    for _, seg, idx in st._walk(64, seed):
        pool.observe(seg.x[idx], y=seg.y[idx])
    ran = [c for c in calls if c["ran"]]
    return {"seed": seed, "experts": len(pool.experts), "events": dict(Counter(e.kind for e in pool.events)), "verdicts": len(calls), "verdicts_not_run": len(calls) - len(ran),
            "conflicts": sum(1 for c in ran if c["conflict"]), "no_conflict": sum(1 for c in ran if not c["conflict"]),
            "consecutive_calls": sum(1 for a, b in zip(calls, calls[1:]) if b["step"] == a["step"] + 1),
            "mean_loss_cluster": (sum(c["loss_cluster"] for c in ran) / len(ran)) if ran else None, "mean_loss_ref": (sum(c["loss_ref"] for c in ran) / len(ran)) if ran else None,
            "mean_ready_experts": (sum(c["n_ready"] for c in calls) / len(calls)) if calls else None, "mean_experts_at_call": (sum(c["n_experts"] for c in calls) / len(calls)) if calls else None,
            "seconds": time.time() - t0}


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Diagnose the fragmentation of Baseline v1 on B1 (development seeds only).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[800, 801, 802, 803, 804])
    ap.add_argument("--out", default="results/m1/dev/diag_transfer_B1_n10.json")
    ap.add_argument("--cpus", default="0,2")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args(argv)
    if any(s < 800 or s > 804 for s in args.seeds):
        raise ValueError("this diagnostic uses the M1 development seeds 800-804 only")
    rows: List[Dict[str, Any]] = []
    with fixed_threads(n=args.threads, cpus=[int(c) for c in args.cpus.split(",")]) as pinning:
        for name, q, overrides in CONFIGS:
            for seed in args.seeds:
                r = {"config": name, "q": q, "overrides": overrides, **run_one(seed, q, overrides)}
                rows.append(r)
                print(f"{name:34s} seed {seed}: experts {r['experts']:3d} verdicts {r['verdicts']:3d} (not run {r['verdicts_not_run']}, conflict {r['conflicts']}, no conflict {r['no_conflict']}) "
                      f"on consecutive batches {r['consecutive_calls']:3d}  {r['seconds']:.0f}s", flush=True)
    save_cell_record(args.out, {"what": "mechanism of the fragmentation of Baseline v1 on B1; development seeds only; not an M1 result", "n_regimes": 10, "pinning": pinning,
                                "environment": environment(), "rows": rows})
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
