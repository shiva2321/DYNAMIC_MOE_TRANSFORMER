"""Study 5c: is the growth policy's minimum-gain threshold (2 % relative held-out loss reduction) what stops an under-sized expert from growing?

Same start as Study 5 (expert of 16 hidden units, reservoir 20,000, mixed-operator arithmetic) with the threshold lowered; nothing else changes. Config field only (``GrowthCfg.min_gain``).

    python probe_grow_threshold.py --seed 0 --min-gain 0.005
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import diag_lib as D  # noqa: E402
from diag_lib import A, L  # noqa: E402
import growmoe.growth as growth  # noqa: E402
from growmoe.config import GrowthCfg  # noqa: E402

OUT = HERE / "grow_threshold"
CALLS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gain", type=float, default=0.005)
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--per-cell", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"h{a.hidden}_g{str(a.min_gain).replace('.', 'p')}_s{a.seed}"          # no dot in the stem: Path.with_suffix would swallow it
    if stem.with_suffix(".json").exists():
        return
    orig = growth.GrowthPolicy.step

    def step_logged(self, expert, x_tr, y_tr, x_va, y_va, rng):
        d = orig(self, expert, x_tr, y_tr, x_va, y_va, rng)
        CALLS.append({"hidden": int(expert.hidden), "depth": len(expert.blocks), "reason": d.reason, "kind": d.kind, "increment": int(d.increment), "best_gain": float(d.gain), "best_p": float(d.p_value)})
        return d

    growth.GrowthPolicy.step = step_logged
    train = L.build_train("FULL", a.seed, 8000)
    test = L.build_test(a.per_cell)
    seen = L.build_seen(train, 500, a.seed)
    learner = L.Pool(5, False, name=f"GrowMoE-h{a.hidden}-s5-r20000-g{a.min_gain}", hidden=a.hidden, replay_size=20000)
    t0 = time.time()
    learner._new(a.seed)
    learner.pool.grow_cfg = GrowthCfg(min_gain=a.min_gain)                      # read when the founding expert's policy is created
    learner._feed(train, np.random.default_rng([a.seed, 0x5F1]).permutation(train.n_samples))
    info = learner._info(t0)
    res, _ = L.evaluate(learner, test, a.per_cell, seen)
    s = D.summarise(res)
    rec = next(iter(learner.pool.experts.values()))
    out = {"seed": a.seed, "min_gain": a.min_gain, "start_hidden": a.hidden, "unseen_exact": s["mean_exact"], "seen_exact": s["seen_exact"], "params": info["params"], "final_hidden": int(rec.module.hidden),
           "final_depth": len(rec.module.blocks), "growth_events": [{"step": e.step, **e.fields} for e in learner.pool.events.of_kind("grow")], "policy_calls": list(CALLS), "seconds": time.time() - t0}
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))
    print(f"[start {a.hidden}, min_gain {a.min_gain}, s{a.seed}] unseen {100 * s['mean_exact']:.1f}  seen {100 * s['seen_exact']:.1f}  params {info['params']:,}  final hidden {rec.module.hidden} depth {len(rec.module.blocks)}  "
          f"growth events {len(out['growth_events'])}  {time.time() - t0:.0f}s", flush=True)
    for e in out["growth_events"]:
        print(f"   grow at step {e['step']}: {e['kind']} +{e.get('increment')}  gain {100 * e['gain']:+.1f}%  p {e['p']:.3f}", flush=True)


if __name__ == "__main__":
    main()
