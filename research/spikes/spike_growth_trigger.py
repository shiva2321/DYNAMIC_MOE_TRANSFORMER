"""SPIKE 3 (throwaway, labelled) -- does the growth policy grow when the teacher needs it and not when data are scarce?

Protocol (fixed before looking at any result). For a teacher in S5 (width w, depth dep) and a training-set size n,
a student starts at width 4, depth 1. Each round: train 600 steps (batch 64, Adam 3e-3, fresh optimiser) on 80 % of
the n samples, record held-out test accuracy, then call GrowthPolicy.step with the remaining 20 % as validation data.
Six rounds, rng seeded per run. References trained for the same total number of steps: a static width-4 student and
a static width-128 student. n = 20000 (learnable) and n = 2000 (data-limited, the spec's original size).
Expected: more growth for larger teachers when data suffice, little or none when data-limited.

Run from the repository root:  python research/spikes/spike_growth_trigger.py --seeds 3
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

from growmoe import streams as S                      # noqa: E402
from growmoe.config import GrowthCfg                  # noqa: E402
from growmoe.experts import MLPExpert                 # noqa: E402
from growmoe.growth import GrowthPolicy               # noqa: E402
from growmoe.metrics import mean_ci                   # noqa: E402

ROUNDS, STEPS, BATCH = 6, 600, 64
TEACHERS = [(8, 1), (32, 2), (128, 3)]
SIZES = [20000, 2000]


def fit(model, x, y, steps, rng):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        i = torch.from_numpy(rng.integers(0, len(x), BATCH))
        model.train_step(x[i], y[i], opt)


def acc(model, x, y):
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def one_run(seed, width, depth, n):
    st = S.s5_ladder(seed, width=width, depth=depth, n_train=n, n_test=1000)
    x, y = st.segments[0].x, st.segments[0].y
    xt, yt = st.test_sets()[0]
    cut = int(0.8 * len(y))
    xtr, ytr, xva, yva = x[:cut], y[:cut], x[cut:], y[cut:]
    rng = np.random.default_rng(seed + 31)
    student = MLPExpert(128, 4, 1, 8, seed=seed)
    student.mark_seen(torch.arange(8))
    policy = GrowthPolicy(GrowthCfg())
    hist = []
    for r in range(ROUNDS):
        fit(student, xtr, ytr, STEPS, rng)
        test = acc(student, xt, yt)
        before = (student.hidden, len(student.blocks))
        d = policy.step(student, xtr, ytr, xva, yva, np.random.default_rng(seed * 100 + r))
        hist.append({"round": r, "test_acc": test, "before": before, "kind": d.kind, "reason": d.reason,
                     "gain": d.gain, "after": (student.hidden, len(student.blocks)), "params": student.num_params()})
    fit(student, xtr, ytr, STEPS, rng)
    final = acc(student, xt, yt)
    refs = {}
    for name, hidden in (("static w4", 4), ("static w128", 128)):
        m = MLPExpert(128, hidden, 1, 8, seed=seed)
        m.mark_seen(torch.arange(8))
        fit(m, xtr, ytr, (ROUNDS + 1) * STEPS, np.random.default_rng(seed + 31))
        refs[name] = {"acc": acc(m, xt, yt), "params": m.num_params()}
    return {"seed": seed, "width": width, "depth": depth, "n": n, "history": hist, "final_acc": final,
            "final_hidden": student.hidden, "final_depth": len(student.blocks), "final_params": student.num_params(),
            "commits": sum(h["kind"] != "none" for h in hist), "refs": refs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="research/growmoe_results/phase0/spike3_growth.json")
    args = ap.parse_args()
    runs, t0 = [], time.time()
    for n in SIZES:
        for width, depth in TEACHERS:
            for seed in range(args.seeds):
                runs.append(one_run(seed, width, depth, n))
                r = runs[-1]
                print(f"n={n} teacher w{width} d{depth} seed {seed}: grew {r['commits']}x to hidden {r['final_hidden']} "
                      f"depth {r['final_depth']}, acc {r['final_acc']:.3f} [{time.time() - t0:.0f}s]", flush=True)
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out).write_text(json.dumps(runs))

    print("\nsummary (mean over seeds): commits, final width / depth, params, test accuracy vs static references")
    print(f"{'n_train':>8s} {'teacher':>10s} {'commits':>8s} {'hidden':>7s} {'depth':>6s} {'params':>8s} {'acc grown':>10s} {'acc w4':>8s} {'acc w128':>9s}")
    for n in SIZES:
        for width, depth in TEACHERS:
            rs = [r for r in runs if r["n"] == n and r["width"] == width and r["depth"] == depth]
            print(f"{n:8d} {'w%d d%d' % (width, depth):>10s} {np.mean([r['commits'] for r in rs]):8.1f} "
                  f"{np.mean([r['final_hidden'] for r in rs]):7.1f} {np.mean([r['final_depth'] for r in rs]):6.1f} "
                  f"{np.mean([r['final_params'] for r in rs]):8.0f} {np.mean([r['final_acc'] for r in rs]):10.3f} "
                  f"{np.mean([r['refs']['static w4']['acc'] for r in rs]):8.3f} {np.mean([r['refs']['static w128']['acc'] for r in rs]):9.3f}")


if __name__ == "__main__":
    main()
