"""Exploratory, THROWAWAY diagnostic on DEV seeds only (100-102); nothing here is pre-registered.

D3b showed that oracle-gated experts trained with A's own recipe reach 52.7 % (5 steps) / 47.2 % (50 steps) on F-15 dev, while A itself
sits at 39-40 %. So the 7-13 points are lost in A's machinery (gate, probation, alarms, extend/spawn), not in its expert recipe.
D4 switches single mechanisms off through existing config flags and splits the loss into PLASTICITY (accuracy on a regime right
after its own segment) and FORGETTING (that minus the final accuracy).

    python research/growmoe_results/m0/diagnose_machinery.py
"""

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_m0 as M                                                                      # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.fairfight import run_fast                                                  # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402

torch.set_num_threads(3)

VARIANTS = {
    "A as run in M0 (50 steps)": dict(inner_steps=50),
    "competence monitor off (no alarms)": dict(inner_steps=50, competence=False),
    "no probation (spawn at once)": dict(inner_steps=50, no_probation=True),
    "no alarms + no probation": dict(inner_steps=50, competence=False, no_probation=True),
    "no alarms + no probation + no growth": dict(inner_steps=50, competence=False, no_probation=True, grow=False),
    "A, 5 steps": dict(inner_steps=5),
    "no alarms + no probation, 5 steps": dict(inner_steps=5, competence=False, no_probation=True),
}


def task(spec: dict) -> dict:
    label, seed = spec["variant"], spec["seed"]
    assert seed >= 100, "dev seeds only"
    st = M.get_stream("F-15", seed, False)
    kw = dict(M.A_KW)
    kw.update(VARIANTS[label])
    res = run_fast(lambda: PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, **kw)), st, seed, eval_every="segments")
    first, immediate = set(), []
    for k, seg in enumerate(st.segments):
        if seg.first_visit and seg.regime not in first:
            first.add(seg.regime)
            immediate.append(res.R[k, seg.regime])                                       # accuracy on the regime right after its own segment
    final = float(res.R[-1].mean())
    return {"variant": label, "seed": seed, "final": final, "immediate": float(np.mean(immediate)), "n_experts": res.n_experts,
            "events": {k: sum(1 for e in res.events if e.kind == k) for k in sorted({e.kind for e in res.events})}}


if __name__ == "__main__":
    specs = [{"variant": v, "seed": sd} for v in VARIANTS for sd in M.DEV_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=2, initializer=M._init_worker, initargs=(3,)) as pool:
        for r in pool.map(task, specs):
            rows.append(r)
            print(r, flush=True)
    (HERE / "diagnostics_D4.json").write_text(json.dumps({"stream": "F-15", "rows": rows}, indent=1), encoding="utf-8")
    print("\nvariant                                   immediate %  final %  forgot  experts  events (mean per run)")
    for v in VARIANTS:
        rr = [r for r in rows if r["variant"] == v]
        imm, fin = 100 * np.mean([r["immediate"] for r in rr]), 100 * np.mean([r["final"] for r in rr])
        ev = {}
        for r in rr:
            for k, n in r["events"].items():
                ev[k] = ev.get(k, 0) + n / len(rr)
        print(f"{v:42s} {imm:8.1f} {fin:9.1f} {imm - fin:7.1f} {np.mean([r['n_experts'] for r in rr]):8.1f}  "
              + ", ".join(f"{k} {n:.1f}" for k, n in sorted(ev.items())))
