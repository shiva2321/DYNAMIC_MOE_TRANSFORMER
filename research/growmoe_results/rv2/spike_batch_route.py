"""THROWAWAY spike on DEV seeds (100-102): does deciding at the level of the BATCH (a set of inputs that "resonate" together) cure the merging / contamination under overlap?

The per-sample gate is nearly blind to a new regime once centre cosine >= 0.6 (spike_onset.py: 2 % per-sample rejection at 0.85), but a 64-sample batch is
detected 100 % of the time with 0 % false alarms. Patched copy of PhyloMoE.observe (no product file touched), variants:

  V0  baseline (the repaired pool).
  V1  batch-vote assignment: all non-novel samples of a batch train the expert most of them were routed to (only when that expert holds >= 50 % of them).
  V2  V1 + set-level novelty: Fisher's combination of the per-sample novelty p-values (max over experts) of the batch; if p_set < set_alpha the whole batch is novel
      (goes to the probation buffer as one candidate cluster) and nothing in it trains an existing expert.

Also reports the PURITY of training: for each expert, the share of the samples it trained on that came from its dominant regime.

    python research/growmoe_results/rv2/spike_batch_route.py
"""

import inspect
import json
import os
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "m2a"))
sys.path.insert(0, str(HERE.parent / "m0"))
sys.path.insert(0, str(HERE.parents[2]))
os.environ.setdefault("OMP_NUM_THREADS", "3")

import numpy as np                                                                      # noqa: E402
import torch                                                                            # noqa: E402

import run_m2a as R                                                                     # noqa: E402
from growmoe.pools import phylo as phylo_mod                                            # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from spike_overlap import overlap_stream                                                # noqa: E402

torch.set_num_threads(3)

_PATCH = """\
if getattr(self, "batch_mode", 0) >= 2 and not self.cfg.no_conformal:
    _stat = -2.0 * torch.log(p_nov.double().clamp_min(1e-12)).sum()
    _p_set = float(torch.special.gammaincc(torch.tensor(float(b), dtype=torch.float64), _stat / 2.0))
    self._last_p_set = _p_set
    if _p_set < self.set_alpha:
        novel = torch.ones_like(novel)
        assigned = torch.full_like(assigned, -1)
if getattr(self, "batch_mode", 0) >= 1:
    _ok = assigned >= 0
    if int(_ok.sum()) >= 8:
        _vals, _cnts = torch.unique(assigned[_ok], return_counts=True)
        if int(_cnts.max()) * 2 >= int(_ok.sum()):
            assigned[_ok] = _vals[int(_cnts.argmax())]
"""


def _patched_observe():
    src = textwrap.dedent(inspect.getsource(PhyloMoE.observe))
    lines = src.split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip() == "spawned = []")
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i] = [indent + l if l else l for l in _PATCH.rstrip("\n").split("\n")]
    ns: dict = {}
    exec("\n".join(lines), vars(phylo_mod), ns)
    return ns["observe"]


class BatchRoutePool(PhyloMoE):
    batch_mode = 0
    set_alpha = 1e-4

    def __init__(self, *a, batch_mode=0, set_alpha=1e-4, **k):
        super().__init__(*a, **k)
        self.batch_mode, self.set_alpha = batch_mode, set_alpha


BatchRoutePool.observe = _patched_observe()


def run(rho, seed, mode):
    st, _ = overlap_stream(seed, rho)
    pool = R.factory("A_new", st, seed, {"steps": 5})()
    pool = BatchRoutePool(pool.cfg, batch_mode=mode)
    ks = [k for k, _, _ in st._walk(64, seed)]
    counts = {}
    n_novel_batches = 0
    for bi, (x, y, ctx) in enumerate(st.algo_batches(64, seed)):
        regime = st.segments[ks[bi]].regime
        rec = pool.observe(x, y, ctx)
        n_novel_batches += int(bool(rec.novel.all()))
        for e in rec.assigned.tolist():
            if e >= 0:
                counts[(regime, e)] = counts.get((regime, e), 0) + 1
    tot = sum(counts.values())
    by_e = {}
    for (r, e), c in counts.items():
        by_e.setdefault(e, []).append(c)
    purity = sum(max(v) for v in by_e.values()) / max(1, tot)
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    acc = float((pool.predict(x).probs.argmax(1) == y).float().mean())
    ev = {k: sum(1 for e in pool.events if e.kind == k) for k in ("spawn", "extend", "alarm")}
    return {"rho": rho, "seed": seed, "mode": mode, "acc": acc, "purity": purity, "experts": len(pool.experts), "novel_batches": n_novel_batches, **ev}


if __name__ == "__main__":
    rows = []
    for rho in (0.3, 0.6, 0.85):
        for mode in (0, 1, 2):
            for seed in (100, 101, 102):
                rows.append(run(rho, seed, mode))
                print(rows[-1], flush=True)
    (HERE / "spike_batch_route.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\ncentre cos   variant   accuracy %   training purity %   experts (10 regimes)   spawn extend alarm   batches wholly novel")
    for rho in (0.3, 0.6, 0.85):
        for mode in (0, 1, 2):
            rr = [r for r in rows if r["rho"] == rho and r["mode"] == mode]
            m = lambda k: np.mean([r[k] for r in rr])                                      # noqa: E731
            print(f"{rho:9.2f} {'V' + str(mode):>8s} {100 * m('acc'):12.1f} {100 * m('purity'):19.1f} {m('experts'):14.1f}   {m('spawn'):9.1f} {m('extend'):6.1f} {m('alarm'):5.1f} {m('novel_batches'):12.1f}")
