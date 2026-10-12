"""Exploratory, THROWAWAY diagnostic on DEV seeds only (100-102); a scratch subclass, no product code is touched.

Root cause found in D3-D5 (see probe in the session log): the competence monitor of a newborn expert is armed from the loss of
the UNTRAINED network on its first batch (baseline mean 2.47). After the first batch of training (50 inner steps on ~60 samples)
its held-out loss rises to 4.3 (early over-fitting), the one-sided CUSUM reads that as concept drift, raises an alarm, and the
alarm path swaps the newborn's weights for a 60-step fine-tune on 32 samples. growmoe/competence.py itself says: "arm the monitor
only after the expert's loss has plateaued".

D6 tests the minimal fix: keep every infant's monitor reset until infancy is over (so its baseline is measured on the settled
expert), leave everything else as in M0. It must (a) recover the F-15 loss and (b) keep what the alarms do on S1.

    python research/growmoe_results/m0/diagnose_arming.py
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


class ArmAfterInfancy(PhyloMoE):
    """PhyloMoE whose newborn experts do not accumulate a monitor baseline until their infancy (``infant_calls`` batches) is over."""

    def observe(self, x, y=None, ctx=None):
        stash = {}
        for eid, rec in self.experts.items():                                           # silence infants that existed before this call
            if rec.state == "infant" and rec.meta.get("monitor") is not None:
                stash[eid] = rec.meta["monitor"]
                rec.meta["monitor"] = None
        out = super().observe(x, y, ctx)
        for eid, mon in stash.items():
            rec = self.experts.get(eid)
            if rec is not None and rec.meta.get("monitor") is None:
                rec.meta["monitor"] = mon
        for rec in self.experts.values():                                               # anything still infant (incl. born in this call) starts clean
            if rec.state == "infant" and rec.meta.get("monitor") is not None:
                rec.meta["monitor"].reset()
                rec.meta["recent"].clear()
        return out


CASES = [
    ("F-15", "A as run in M0 (50 steps)", PhyloMoE, dict(inner_steps=50)),
    ("F-15", "arm after infancy, 50 steps", ArmAfterInfancy, dict(inner_steps=50)),
    ("F-15", "A, 5 steps", PhyloMoE, dict(inner_steps=5)),
    ("F-15", "arm after infancy, 5 steps", ArmAfterInfancy, dict(inner_steps=5)),
    ("F-15", "arm after infancy, 20 steps", ArmAfterInfancy, dict(inner_steps=20)),
    ("S1", "A tuned (5 steps)", PhyloMoE, dict(inner_steps=5)),
    ("S1", "arm after infancy, 5 steps", ArmAfterInfancy, dict(inner_steps=5)),
    ("S2-5", "A tuned (20 steps)", PhyloMoE, dict(inner_steps=20)),
    ("S2-5", "arm after infancy, 20 steps", ArmAfterInfancy, dict(inner_steps=20)),
]


def task(spec: dict) -> dict:
    name, label, seed = spec["stream"], spec["variant"], spec["seed"]
    assert seed >= 100, "dev seeds only"
    cls = {"PhyloMoE": PhyloMoE, "ArmAfterInfancy": ArmAfterInfancy}[spec["cls"]]
    st = M.get_stream(name, seed, False)
    kw = dict(M.A_KW)
    kw.update(spec["kw"])
    res = run_fast(lambda: cls(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, **kw)), st, seed, eval_every="segments")
    first, immediate = set(), []
    for k, seg in enumerate(st.segments):
        if seg.first_visit and seg.regime not in first:
            first.add(seg.regime)
            immediate.append(res.R[k, seg.regime])
    return {"stream": name, "variant": label, "seed": seed, "final": float(res.R[-1].mean()), "immediate": float(np.mean(immediate)),
            "n_experts": res.n_experts, "n_regimes": len(st.regimes),
            "events": {k: sum(1 for e in res.events if e.kind == k) for k in sorted({e.kind for e in res.events})}}


if __name__ == "__main__":
    specs = [{"stream": s, "variant": v, "cls": c.__name__, "kw": kw, "seed": sd} for s, v, c, kw in CASES for sd in M.DEV_SEEDS]
    rows = []
    with ProcessPoolExecutor(max_workers=2, initializer=M._init_worker, initargs=(3,)) as pool:
        for r in pool.map(task, specs):
            rows.append(r)
            print({k: v for k, v in r.items() if k != "events"}, r["events"], flush=True)
    (HERE / "diagnostics_D6.json").write_text(json.dumps({"rows": rows}, indent=1), encoding="utf-8")
    print("\nstream variant                           immediate %  final %  forgot  experts / regimes  events (mean)")
    for key in dict.fromkeys((r["stream"], r["variant"]) for r in rows):
        rr = [r for r in rows if (r["stream"], r["variant"]) == key]
        ev = {}
        for r in rr:
            for k, n in r["events"].items():
                ev[k] = ev.get(k, 0) + n / len(rr)
        imm, fin = 100 * np.mean([r["immediate"] for r in rr]), 100 * np.mean([r["final"] for r in rr])
        print(f"{key[0]:5s} {key[1]:32s} {imm:8.1f} {fin:8.1f} {imm - fin:6.1f} {np.mean([r['n_experts'] for r in rr]):8.1f} / {rr[0]['n_regimes']}   "
              + ", ".join(f"{k} {n:.1f}" for k, n in sorted(ev.items())))
