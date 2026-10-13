"""THROWAWAY spike on DEV seeds (100-102): how small can a batch be before the set-level test loses its power?   (gatekeeper research, entry E-023)

Clean setting like A1 (a signature per regime built from its first-visit data, in-regime batches against foreign batches), but the batch size b is varied. Statistic: the
permutation test on the distance of means against the expert's stored prototypes (999 permutations). Reported: false alarm and detection at alpha 0.01 and 0.001.

    python research/growmoe_results/rv3/spike_power_vs_batch.py
"""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import spike_set_tests as T                                                             # noqa: E402
from growmoe.signature import VmfSignature                                              # noqa: E402
from growmoe.streams_m1 import function_regimes                                         # noqa: E402

torch.set_num_threads(2)
SIZES, RHOS, SEEDS, ALPHAS = (8, 16, 32, 64, 128), (0.3, 0.6, 0.85), (100, 101, 102), (0.01, 0.001)


def run(rho: float, seed: int) -> dict:
    st = function_regimes(seed, n_regimes=10, overlap=rho)
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    sigs = {}
    for r in range(10):
        seg = next(s for s in st.segments if s.regime == r and s.first_visit)
        sig = VmfSignature(st.d, 256, seed=r)
        sig.update(seg.x)
        sig.fit()
        sigs[r] = sig
    out = {}
    for b in SIZES:
        own, foreign = [], []
        for r, sig in sigs.items():
            xs = st.regimes[r].x_te
            for lo in range(0, len(xs) - b + 1, b):
                own.append(T.meandist_p(sig.protos, xs[lo:lo + b], g))
            for r2 in rng.choice([q for q in range(10) if q != r], size=3, replace=False):
                xs2 = st.regimes[int(r2)].x_te
                for lo in range(0, len(xs2) - b + 1, b):
                    foreign.append(T.meandist_p(sig.protos, xs2[lo:lo + b], g))
        out[b] = {"n_own": len(own), "n_foreign": len(foreign), **{f"fa_{a}": float(np.mean(np.asarray(own) <= a)) for a in ALPHAS},
                  **{f"det_{a}": float(np.mean(np.asarray(foreign) <= a)) for a in ALPHAS}}
    return {"rho": rho, "seed": seed, "by_b": out}


if __name__ == "__main__":
    rows = []
    for rho in RHOS:
        for seed in SEEDS:
            rows.append(run(rho, seed))
            print("done", rho, seed, flush=True)
    (HERE / "spike_power_vs_batch.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nrho   b     false alarm @0.01 / @0.001      detection @0.01 / @0.001")
    for rho in RHOS:
        for b in SIZES:
            rr = [r["by_b"][b] for r in rows if r["rho"] == rho]
            m = lambda k: float(np.mean([x[k] for x in rr]))                          # noqa: E731
            print(f"{rho:4.2f} {b:4d}   {m('fa_0.01'):8.3f} {m('fa_0.001'):8.3f}              {m('det_0.01'):8.3f} {m('det_0.001'):8.3f}", flush=True)
