"""THROWAWAY spike on DEV seeds (100-102): what happens to the set-level gate when its exchangeability assumption is violated?   (gatekeeper research, entry E-023)

The set test assumes the batches of one regime are random draws of that regime. Real streams are often ordered (by class, by time, by document). Here every segment's samples are
sorted by label before they are cut into batches of 64, so each batch holds one or two classes only. The vote pool and the set-gate pool (md_recent, the corrected candidate rule)
are run on the same sorted streams; the expert count against the number of true regimes shows whether the set gate over-fires.

    python research/growmoe_results/rv3/spike_sorted_batches.py [--streams F-15,OV-0.0 --order label]
"""

import argparse
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import spike_set_gate as G                                                              # noqa: E402


def sorted_walk(self, batch_size: int, seed: int):
    for k, seg in enumerate(self.segments):
        order = torch.argsort(seg.y, stable=True)
        for lo in range(0, len(order), batch_size):
            yield k, seg, order[lo:lo + batch_size]


def run(name: str, seed: int, variant: str) -> dict:
    st = G.A.get_stream(name, seed, False)
    st._walk = types.MethodType(sorted_walk, st)                                        # the cached stream object is patched in place; restored below
    try:
        row = G.run(name, seed, variant)
    finally:
        del st._walk
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", default="F-15,OV-0.0")
    ap.add_argument("--variants", default="vote,md_recent")
    ap.add_argument("--tag", default="md_recent")
    args = ap.parse_args()
    rows = []
    for name in args.streams.split(","):
        for variant in args.variants.split(","):
            for seed in G.SEEDS:
                rows.append(run(name, seed, variant))
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    (HERE / f"spike_sorted_batches_{args.tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nlabel-sorted batches: stream, variant, accuracy %, experts / true regimes, set-novel calls, known batches called novel / known batches")
    for name in args.streams.split(","):
        for variant in args.variants.split(","):
            rr = [r for r in rows if r["stream"] == name and r["variant"] == variant]
            kn = sum(r["known_novel"] + r["known_known"] + r["known_skip"] + r["known_none"] for r in rr)
            print(f"{name:8s} {variant:10s} acc {100 * np.mean([r['acc'] for r in rr]):5.1f}  experts {np.mean([r['experts'] for r in rr]):5.1f} / {rr[0]['regimes']}  "
                  f"set-novel {np.mean([r['set_novel'] for r in rr]):5.1f}  known batches called novel {sum(r['known_novel'] for r in rr)} / {kn}", flush=True)
