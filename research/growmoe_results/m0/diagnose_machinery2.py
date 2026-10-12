"""Exploratory, THROWAWAY diagnostic on DEV seeds only (100-102); second wave after diagnose_machinery.py (D4).

D4 found: with the competence monitor ("alarms") and the probation buffer both switched off, A at 5 steps per batch rises from
39.2 % to 46.3 % on F-15 dev (oracle gating with A's own expert recipe: 52.7 %). Alarms fire about once per regime and 11 of A's 13
"extend" events come from the alarm path. D5 separates the pieces (alarms, probation, extend-vs-spawn probe) and checks what the
same switches do to the similarity-type dev streams S2-5 and S1.

    python research/growmoe_results/m0/diagnose_machinery2.py
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

NEVER_EXTEND = dict(delta_forget=-1.0)
F15 = {
    "alarms off + no probation, 5 steps (D4)": dict(inner_steps=5, competence=False, no_probation=True),
    "  + never extend": dict(inner_steps=5, competence=False, no_probation=True, **NEVER_EXTEND),
    "alarms off, probation ON, 5 steps": dict(inner_steps=5, competence=False),
    "alarms off, probation ON, never extend, 5 steps": dict(inner_steps=5, competence=False, **NEVER_EXTEND),
    "alarms ON, no probation, never extend, 5 steps": dict(inner_steps=5, no_probation=True, **NEVER_EXTEND),
    "alarms off + no probation + never extend, 20 steps": dict(inner_steps=20, competence=False, no_probation=True, **NEVER_EXTEND),
    "alarms off + no probation + never extend, 50 steps": dict(inner_steps=50, competence=False, no_probation=True, **NEVER_EXTEND),
}
SIDE = {                                                                                 # what the same switches cost on the other families
    "S2-5": {"A tuned (20 steps)": dict(inner_steps=20),
             "alarms off + no probation + never extend, 5 steps": dict(inner_steps=5, competence=False, no_probation=True, **NEVER_EXTEND)},
    "S1": {"A tuned (5 steps)": dict(inner_steps=5),
           "alarms off + no probation + never extend, 5 steps": dict(inner_steps=5, competence=False, no_probation=True, **NEVER_EXTEND)},
}


def task(spec: dict) -> dict:
    name, label, seed, kw_extra = spec["stream"], spec["variant"], spec["seed"], spec["kw"]
    assert seed >= 100, "dev seeds only"
    st = M.get_stream(name, seed, False)
    kw = dict(M.A_KW)
    kw.update(kw_extra)
    res = run_fast(lambda: PhyloMoE(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, **kw)), st, seed, eval_every="segments")
    first, immediate = set(), []
    for k, seg in enumerate(st.segments):
        if seg.first_visit and seg.regime not in first:
            first.add(seg.regime)
            immediate.append(res.R[k, seg.regime])
    return {"stream": name, "variant": label, "seed": seed, "final": float(res.R[-1].mean()), "immediate": float(np.mean(immediate)),
            "n_experts": res.n_experts, "n_regimes": len(st.regimes),
            "events": {k: sum(1 for e in res.events if e.kind == k) for k in sorted({e.kind for e in res.events})}}


if __name__ == "__main__":
    specs = [{"stream": "F-15", "variant": v, "seed": sd, "kw": kw} for v, kw in F15.items() for sd in M.DEV_SEEDS]
    specs += [{"stream": s, "variant": v, "seed": sd, "kw": kw} for s, d in SIDE.items() for v, kw in d.items() for sd in M.DEV_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=2, initializer=M._init_worker, initargs=(3,)) as pool:
        for r in pool.map(task, specs):
            rows.append(r)
            print({k: v for k, v in r.items() if k != "events"}, r["events"], flush=True)
    (HERE / "diagnostics_D5.json").write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    print("\nstream variant                                              immediate %  final %  experts / regimes  events (mean)")
    for key in dict.fromkeys((r["stream"], r["variant"]) for r in rows):
        rr = [r for r in rows if (r["stream"], r["variant"]) == key]
        ev = {}
        for r in rr:
            for k, n in r["events"].items():
                ev[k] = ev.get(k, 0) + n / len(rr)
        print(f"{key[0]:5s} {key[1]:56s} {100 * np.mean([r['immediate'] for r in rr]):7.1f} {100 * np.mean([r['final'] for r in rr]):8.1f} "
              f"{np.mean([r['n_experts'] for r in rr]):8.1f} / {rr[0]['n_regimes']}   " + ", ".join(f"{k} {n:.1f}" for k, n in sorted(ev.items())))
