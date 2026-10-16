"""Seed aliasing on the project's own stage-3 streams (throwaway; no product file touched).

The teacher of regime r of ``function_regimes`` is ``MLPExpert(seed = stream_seed * 1000 + r)``; the pool founds expert number k with a fresh ``MLPExpert(seed = cfg.seed * 1000 + k)``. When the pool's seed equals the
stream's seed, the expert founded for regime r (id r, when the pool founds exactly one expert per regime in order) starts with the first hidden units of the TEACHER of regime r. This runs the arms of the dev
runner on a stream with cfg.seed = stream seed (as the dev runner does) and with cfg.seed = stream seed + 7 (no aliasing), and compares the final accuracy.

    python alias_check_s3.py F-15 --seeds 100 101 --arms S3,S3+T
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "growmoe_results" / "rv2d"))
sys.path.insert(0, str(ROOT / "research" / "growmoe_results" / "rv2c"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import rv2d_lib as L  # noqa: E402  (puts the stage-3 runner on the path)
import run_rv2c as R3  # noqa: E402
from growmoe.fairfight import _Scorer, run_fast  # noqa: E402

TMARGIN = 0.25


def run(stream: str, seed: int, pool_seed: int, arm: str) -> dict:
    t0 = time.perf_counter()
    st = R3.build_stream(stream, seed, False)
    steps = R3.steps_for(stream)
    kw = dict(L.ARMS[arm])
    if "spawn_rule" in kw:
        kw["transfer_margin"] = TMARGIN
    cfg = dataclasses.replace(R3.arm_cfg(L.S3_ARM, st, seed, steps), seed=pool_seed, **kw)
    holder = []

    def make():
        holder.append(R3.LoggingPool(cfg))
        return holder[0]

    run_fast(make, st, seed, eval_every="final")
    pool = holder[0]
    return {"stream": stream, "arm": arm, "stream_seed": seed, "pool_seed": pool_seed, "acc": float(_Scorer(pool, st).row().mean()), "experts": len(pool.experts), "regimes": len(st.regimes),
            "seconds": time.perf_counter() - t0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stream")
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--arms", default="S3")
    args = ap.parse_args()
    torch.set_num_threads(2)
    out = HERE / f"alias_check_s3_{args.stream}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    for r in res:
        r.setdefault("arm", "S3")
    done = {(r["arm"], r["stream_seed"], r["pool_seed"]) for r in res}
    for seed in args.seeds:
        for arm in args.arms.split(","):
            for pool_seed in (seed, seed + 7):
                if (arm, seed, pool_seed) in done:
                    continue
                res.append(run(args.stream, seed, pool_seed, arm))
                tmp = out.with_name(out.name + ".tmp")
                with open(tmp, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(res, indent=1))
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, out)
                print(res[-1], flush=True)
    for seed in args.seeds:
        for arm in args.arms.split(","):
            a = [r for r in res if r["arm"] == arm and r["stream_seed"] == seed and r["pool_seed"] == seed]
            b = [r for r in res if r["arm"] == arm and r["stream_seed"] == seed and r["pool_seed"] == seed + 7]
            if a and b:
                print(f"{args.stream} {arm} seed {seed}: pool seed = stream seed {100 * a[0]['acc']:.1f} ({a[0]['experts']} experts) | pool seed + 7 {100 * b[0]['acc']:.1f} ({b[0]['experts']} experts) | alias bonus {100 * (a[0]['acc'] - b[0]['acc']):+.1f}")


if __name__ == "__main__":
    main()
