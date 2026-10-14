"""Study 5: does the pool really expand an expert vertically and horizontally AS NEEDED?  A controlled test with a known answer.

The expert is deliberately started far too small (16 hidden units, one block) on the mixed-operator arithmetic stream, where fixed experts are known (Study 1) to gain from more width.
The reservoir is 20,000 examples, so the growth policy has plenty of held-out data to decide on (it decided on 51 held-out examples in the shipped configuration).
We log every growth decision and compare the final accuracy with FIXED experts of widths 16, 32, 64, 128 and 512 trained with the identical recipe.

    python probe_grow_from_small.py --seed 0
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

OUT = HERE / "grow_from_small"
D.ARCHS.update({"E16": ("expert", 16, 1, True), "E32": ("expert", 32, 1, True), "E64": ("expert", 64, 1, True)})
CALLS = []


def install():
    orig = growth.GrowthPolicy.step

    def step_logged(self, expert, x_tr, y_tr, x_va, y_va, rng):
        d = orig(self, expert, x_tr, y_tr, x_va, y_va, rng)
        CALLS.append({"n_train": int(len(x_tr)), "n_valid": int(len(x_va)), "hidden": int(expert.hidden), "depth": len(expert.blocks), "params": int(expert.num_params()),
                      "reason": d.reason, "kind": d.kind, "increment": int(d.increment), "best_gain": float(d.gain), "best_p": float(d.p_value), "tried": d.tried})
        return d

    growth.GrowthPolicy.step = step_logged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-cell", type=int, default=200)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    stem = OUT / f"s{a.seed}"
    if stem.with_suffix(".json").exists():
        return
    install()
    train = L.build_train("FULL", a.seed, 8000)
    test = L.build_test(a.per_cell)
    seen = L.build_seen(train, 500, a.seed)
    out = {"seed": a.seed, "fixed": {}}
    for arch in ("E16", "E32", "E64", "E128", "E512"):
        t0 = time.time()
        learner = D.Online(arch, 5, 20000)
        info = learner.fit(train, a.seed)
        res, _ = L.evaluate(learner, test, a.per_cell, seen)
        s = D.summarise(res)
        out["fixed"][arch] = {"unseen_exact": s["mean_exact"], "seen_exact": s["seen_exact"], "params": info["params"], "seconds": time.time() - t0}
        print(f"[fixed {arch} s{a.seed}] unseen {100 * s['mean_exact']:.1f}  seen {100 * s['seen_exact']:.1f}  params {info['params']:,}  {time.time() - t0:.0f}s", flush=True)
    CALLS.clear()
    t0 = time.time()
    learner = L.Pool(5, False, name="GrowMoE-h16-s5-r20000", hidden=16, replay_size=20000)
    info = learner.fit(train, a.seed)
    res, _ = L.evaluate(learner, test, a.per_cell, seen)
    s = D.summarise(res)
    rec = next(iter(learner.pool.experts.values()))
    growth_events = [{"step": e.step, **e.fields} for e in learner.pool.events.of_kind("grow")]
    out["growing_pool"] = {"unseen_exact": s["mean_exact"], "seen_exact": s["seen_exact"], "params": info["params"], "final_hidden": int(rec.module.hidden), "final_depth": len(rec.module.blocks),
                           "events": info["events"], "growth_events": growth_events, "policy_calls": list(CALLS), "seconds": time.time() - t0}
    D.atomic_write_text(stem.with_suffix(".json"), json.dumps(out))
    print(f"[growing pool from 16 s{a.seed}] unseen {100 * s['mean_exact']:.1f}  seen {100 * s['seen_exact']:.1f}  params {info['params']:,}  final hidden {rec.module.hidden} depth {len(rec.module.blocks)}  "
          f"growth events {len(growth_events)}  policy calls {len(CALLS)}  {time.time() - t0:.0f}s", flush=True)
    for e in growth_events:
        print(f"   grow at step {e['step']}: {e['kind']} +{e.get('increment')}  gain {100 * e['gain']:+.1f}%  p {e['p']:.3f}  params added {e['params_added']}", flush=True)
    for c in CALLS:
        print(f"   call: {c['reason']:10s} hidden {c['hidden']:4d} depth {c['depth']} n_train {c['n_train']} n_valid {c['n_valid']} best gain {100 * c['best_gain']:+.1f}% p {c['best_p']:.3f}", flush=True)


if __name__ == "__main__":
    main()
