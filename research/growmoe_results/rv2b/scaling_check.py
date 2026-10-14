"""RV2-B statement C4: gate cost at many experts, exact scan vs the shortlist cascade, on synthetic pools (no training).

    python research/growmoe_results/rv2b/scaling_check.py [--k 4] [--shortlist 2]

Each pool holds E experts with 300-point regimes (256 prototypes each, d=128). Inputs are drawn from random experts' regimes. Wall-clock is the
median of 5 repeats of one ``_gate_matrices`` call after a warm-up call (which builds the fingerprints). Writes scaling_check.json.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

from growmoe.accounting import measured_gate_macs_per_input                              # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.experts import MLPExpert                                                   # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.signature import VmfSignature                                              # noqa: E402

D, GRID, BATCHES = 128, (8, 16, 32, 64, 128, 500), (64, 1024)


def _unit(a):
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def build(n_experts: int, shortlist: int, k: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    centres = _unit(rng.standard_normal((n_experts, D)))
    pool = PhyloMoE(PhyloCfg(d=D, hidden=16, n_classes=4, seed=seed, q=0.02, max_experts=4096, route_rule="accept_then_logf",
                             gate_shortlist=shortlist, gate_fingerprint_k=k, gate_shortlist_min_experts=2))
    for e in range(n_experts):
        sig = VmfSignature(D, 256, seed=e)
        sig.update(torch.from_numpy(_unit(4.0 * centres[e] + rng.standard_normal((300, D))).astype(np.float32)))
        sig.fit()
        pool.add_expert(MLPExpert(D, 16, 1, 4, seed=e), parent=None, state="active", meta={"sig": sig})
    return pool, centres


def inputs(centres, n: int, seed: int = 1):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(centres), n)
    return torch.from_numpy(_unit(4.0 * centres[idx] + rng.standard_normal((n, D))).astype(np.float32))


def time_gate(pool, x, repeats: int = 5):
    eids = list(pool.experts)
    pool._gate_matrices(x, eids)                                                        # warm-up: builds the fingerprints
    pool.reset_gate_stats()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        pool._gate_matrices(x, eids)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), measured_gate_macs_per_input(pool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--shortlist", type=int, default=2)
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = {"d": D, "k": args.k, "shortlist": args.shortlist, "rows": {}}
    for n_exp in GRID:
        exact, centres = build(n_exp, 0, args.k)
        casc, _ = build(n_exp, args.shortlist, args.k)
        row = {}
        for b in BATCHES:
            x = inputs(centres, b)
            te, me = time_gate(exact, x)
            tc, mc = time_gate(casc, x)
            row[str(b)] = {"exact_s": te, "cascade_s": tc, "ratio_time": tc / te, "exact_macs": me, "cascade_macs": mc, "ratio_macs": mc / me}
        out["rows"][str(n_exp)] = row
        print(n_exp, {b: (round(v["ratio_macs"], 3), round(v["ratio_time"], 2)) for b, v in row.items()}, flush=True)
    (HERE / "scaling_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
