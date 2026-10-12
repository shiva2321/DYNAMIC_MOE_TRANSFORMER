"""THROWAWAY spike on DEV seeds (100-102): can a strided SUBSET of each expert's stored prototypes ("landmarks") be the stage-1 fingerprint?

No extra structure to maintain: the reservoir is an unordered uniform sample of the expert's stream, so every n-th prototype is a valid, always-current
miniature of the expert (a Nystrom-style landmark KDE with the expert's own kappa). Compared with the centroid and 8 k-means sub-centroids of spike_fingerprint.py.

    python research/growmoe_results/rv2/spike_landmarks.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spike_fingerprint as SF                                                          # noqa: E402

_orig = SF.build_scorers
LM = (8, 16, 32)


def wrapped(pool, eids, xn, d, seed):
    S, macs, mp = _orig(pool, eids, xn, d, seed)
    for n_lm in LM:
        cols, total = [], 0
        for e in eids:
            sig = pool.experts[e].meta["sig"]
            p = sig.protos
            sub = p[:: max(1, len(p) // n_lm)][:n_lm]
            cols.append(torch.logsumexp(sig.kappa * (xn @ sub.T), dim=1) - math.log(len(sub)) + sig._log_c[sig._kappa])
            total += len(sub)
        S[f"lm{n_lm}"] = torch.stack(cols, 1)
        macs[f"lm{n_lm}"] = int(total * d)
    return S, macs, mp


SF.build_scorers = wrapped

if __name__ == "__main__":
    rows = [SF.run_one(s, seed) for s in SF.STREAMS for seed in SF.SEEDS]
    (HERE / "spike_landmarks.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    keep = ["centroid", "subproto-8", "lm8", "lm16", "lm32"]
    for stream in SF.STREAMS:
        rr = [r for r in rows if r["stream"] == stream]
        ex = np.mean([r["exact_macs"] for r in rr]) / 1000
        print(f"\n=== {stream}: {np.mean([r['E'] for r in rr]):.1f} experts, exact {100 * np.mean([r['acc_exact'] for r in rr]):.1f} %, {ex:.0f} kMACs ===")
        print(f"{'stage 1':11s} {'alone %':>8s} {'agree %':>8s} {'s1 kMACs':>9s} | top-2 recall %  acc %  kMACs (x cheaper)")
        for sc in keep:
            m = lambda k: float(np.mean([r[sc][k] for r in rr]))                           # noqa: E731
            k2 = m("macs_cascade@2") / 1000
            print(f"{sc:11s} {100 * m('acc_direct'):8.1f} {100 * m('agree_exact'):8.1f} {m('macs_stage1') / 1000:9.1f} | {100 * m('recall@2'):13.1f} {100 * m('acc_cascade@2'):6.1f} {k2:7.0f} ({ex / k2:.1f}x)")
