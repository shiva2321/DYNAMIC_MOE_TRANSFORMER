"""Exploratory extras after the confirmatory run (NOT pre-registered; labelled as such in RESULTS_M0.md). Fresh seeds, no verdict reads these.

    python research/growmoe_results/m0/diagnose_extras.py speed          # original code path vs the new path, dev seed 100
    python research/growmoe_results/m0/diagnose_extras.py routing        # D1: where does A lose to oracle routing? seeds 200-201
    python research/growmoe_results/m0/diagnose_extras.py ablate         # D2: A's expert-training recipe on DEV seeds 100-102 (F-15)

speed: the ORIGINAL path = bench.run_stream (per-batch predict, one predict call per regime, per-segment evaluation) with the original
per-sample ``PhyloMoE._predict_reference``; the NEW path = run_fast(eval_every="final") with the vectorised predict. Same final accuracy
is asserted.
routing: A (tuned budget from tuning.json) on F-15 and F-30, original runner so that it reports routing accuracy (share of held-out
samples sent to the expert that owns their regime), spawn precision / recall and time-to-spawn; B4 on the same seeds for reference.
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_m0 as M                                                                      # noqa: E402
from growmoe import bench                                                               # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.fairfight import run_fast                                                  # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402

torch.set_num_threads(3)
TUNING = json.loads((HERE / "tuning.json").read_text(encoding="utf-8"))


def speed() -> None:
    rows, seed = [], 100
    for name in ("S2-5", "F-15", "F-30"):
        st = M.get_stream(name, seed, False)
        fam = M.FAMILY[name]
        make = M.factory("A", st, seed, M.params_for("A", fam, TUNING))
        keep = PhyloMoE.predict
        PhyloMoE.predict = PhyloMoE._predict_reference
        try:
            t0 = time.perf_counter()
            old = bench.run_stream(make, st, seed=seed)
            t_old = time.perf_counter() - t0
        finally:
            PhyloMoE.predict = keep
        new = run_fast(make, st, seed=seed, eval_every="final")
        rows.append({"stream": name, "method": "A", "steps": M.params_for("A", fam, TUNING)["steps"], "seconds_original": t_old,
                     "seconds_new": new.wall_clock, "speedup": t_old / new.wall_clock,
                     "final_accuracy_equal": bool(np.array_equal(old.R[-1], new.R[-1]))})
        print(rows[-1], flush=True)
    (HERE / "speed.json").write_text(json.dumps({"seed": seed, "rows": rows}, indent=1), encoding="utf-8")


def routing() -> None:
    rows = []
    for name in ("F-15", "F-30"):
        fam = M.FAMILY[name]
        for seed in (200, 201):
            st = M.get_stream(name, seed, False)
            for m in ("A", "B4"):
                res = bench.run_stream(M.factory(m, st, seed, M.params_for(m, fam, TUNING)), st, seed=seed)
                ss = res.spawn_stats
                rows.append({"stream": name, "seed": seed, "method": m, "accuracy": res.accuracy_stats["avg"], "routing_acc": res.routing_acc,
                             "n_experts": res.pool_summary.get("n_experts"), "n_regimes": len(st.regimes),
                             "spawn_precision": ss.get("precision"), "spawn_recall": ss.get("recall"), "time_to_spawn": ss.get("time_to_spawn"),
                             "false_novelty_rate": res.false_novelty_rate, "novel_detection_rate": res.novel_detection_rate,
                             "final_regime_acc": {int(k): v for k, v in res.final_regime_acc.items()}})
                print({k: v for k, v in rows[-1].items() if k != "final_regime_acc"}, flush=True)
    (HERE / "diagnostics_D1.json").write_text(json.dumps({"rows": rows}, indent=1, default=float), encoding="utf-8")


VARIANTS = {                                                                            # all on top of A at its tuned 50 steps
    "A as run in M0": {},
    "fresh init (init_mode=scratch)": {"init_mode": "scratch"},
    "no per-expert replay (replay_size=0)": {"replay_size": 0},
    "never extend, always spawn (delta_forget=-1)": {"delta_forget": -1.0},
    "fresh init + no replay + never extend": {"init_mode": "scratch", "replay_size": 0, "delta_forget": -1.0},
    "  same recipe, 5 steps per batch": {"init_mode": "scratch", "replay_size": 0, "delta_forget": -1.0, "inner_steps": 5},
    "  same recipe, 20 steps per batch": {"init_mode": "scratch", "replay_size": 0, "delta_forget": -1.0, "inner_steps": 20},
}


def ablate_task(spec: dict) -> dict:
    name, seed, label = "F-15", spec["seed"], spec["variant"]
    assert seed >= 100, "ablations use dev seeds only"
    st = M.get_stream(name, seed, False)
    kw = dict(M.A_KW, inner_steps=M.params_for("A", "func", TUNING)["steps"])
    kw.update(VARIANTS[label])
    res = run_fast(lambda: PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, **kw)), st, seed, eval_every="final")
    return {"variant": label, "seed": seed, "acc": res.acc, "n_experts": res.n_experts, "seconds": res.wall_clock}


def ablate() -> None:
    specs = [{"variant": v, "seed": sd} for v in VARIANTS for sd in M.DEV_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=2, initializer=M._init_worker, initargs=(3,)) as pool:
        for r in pool.map(ablate_task, specs):
            rows.append(r)
            print(r, flush=True)
    (HERE / "ablation_dev.json").write_text(json.dumps({"stream": "F-15", "dev_seeds": list(M.DEV_SEEDS), "rows": rows}, indent=1), encoding="utf-8")
    print("\nvariant                                          dev accuracy %   experts")
    for v in VARIANTS:
        rr = [r for r in rows if r["variant"] == v]
        print(f"{v:48s} {100 * np.mean([r['acc'] for r in rr]):6.1f}        {np.mean([r['n_experts'] for r in rr]):5.1f}")


if __name__ == "__main__":
    {"speed": speed, "routing": routing, "ablate": ablate}[sys.argv[1]]()
