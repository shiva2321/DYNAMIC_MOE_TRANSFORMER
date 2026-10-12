"""Exploratory, THROWAWAY: do the proposed guard streams (S3 drift, S4 outliers, S7 label remap) react to the alarm path at all?
A guard that cannot fail is useless, so before pre-registering them we measure, on DEV seeds 100-102, how much A's accuracy moves when
the competence monitor is switched off or armed only after infancy. Scratch subclass only; no product code is touched.

    python research/growmoe_results/m0/diagnose_guards.py
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
from diagnose_arming import ArmAfterInfancy                                             # noqa: E402
from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.fairfight import run_fast                                                  # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.streams import s3_drift, s4_outliers, s7_mapping                           # noqa: E402

torch.set_num_threads(3)

STREAMS = {
    "S3": lambda sd: s3_drift(seed=sd, n_samples=2000, n_swaps=24, n_segments=16, n_test=150),
    "S4": lambda sd: s4_outliers(seed=sd, outlier_rate=0.01, label_noise=0.1, n_regimes=5, n_train=400, n_test=100),
    "S7": lambda sd: s7_mapping(seed=sd, with_ctx=False),
}
VARIANTS = {
    "A old (20 steps)": (PhyloMoE, dict(inner_steps=20)),
    "alarms off (20 steps)": (PhyloMoE, dict(inner_steps=20, competence=False)),
    "arm after infancy (20 steps)": (ArmAfterInfancy, dict(inner_steps=20)),
    "arm after infancy (5 steps)": (ArmAfterInfancy, dict(inner_steps=5)),
}


def task(spec: dict) -> dict:
    name, label, seed = spec["stream"], spec["variant"], spec["seed"]
    assert seed >= 100, "dev seeds only"
    cls, kw_extra = VARIANTS[label]
    st = STREAMS[name](seed)
    kw = dict(M.A_KW)
    kw.update(kw_extra)
    res = run_fast(lambda: cls(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, **kw)), st, seed, eval_every="final")
    return {"stream": name, "variant": label, "seed": seed, "final": res.acc, "n_experts": res.n_experts, "n_regimes": len(st.regimes),
            "events": {k: sum(1 for e in res.events if e.kind == k) for k in sorted({e.kind for e in res.events})}}


if __name__ == "__main__":
    specs = [{"stream": s, "variant": v, "seed": sd} for s in STREAMS for v in VARIANTS for sd in M.DEV_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=2, initializer=M._init_worker, initargs=(3,)) as pool:
        for r in pool.map(task, specs):
            rows.append(r)
    (HERE / "diagnostics_guards.json").write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    print("stream variant                       final %   per seed                 experts / regimes  events (mean)")
    for key in dict.fromkeys((r["stream"], r["variant"]) for r in rows):
        rr = [r for r in rows if (r["stream"], r["variant"]) == key]
        ev = {}
        for r in rr:
            for k, n in r["events"].items():
                ev[k] = ev.get(k, 0) + n / len(rr)
        print(f"{key[0]:3s} {key[1]:30s} {100 * np.mean([r['final'] for r in rr]):6.1f}   {[round(100 * r['final'], 1) for r in rr]}   "
              f"{np.mean([r['n_experts'] for r in rr]):5.1f} / {rr[0]['n_regimes']}   " + ", ".join(f"{k} {n:.1f}" for k, n in sorted(ev.items())))
