"""THROWAWAY spike on DEV seeds (100-102): at training time the label is known, so let it help decide WHICH expert learns a sample.

spike_overlap.py showed that when regimes overlap in input space the repaired pool collapses (cosine 0.3 between centres: 50.5 % -> 33.7 %)
even though even the best expert per regime in hindsight only reaches 35 %: the experts are contaminated, because the density gate accepts
a foreign sample and the wrong expert trains on it.

Patch tested (applied to a COPY of PhyloMoE.observe at run time; no product file is touched): after the gate has computed the per-expert
log-densities, subtract lambda * cross-entropy of the sample's label under each expert (capped), so that among the experts that accept
the sample the one that already explains its label best is the one that trains on it (a MOSAIC-style responsibility).

    python research/growmoe_results/rv2/spike_label_assign.py
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
if y_t is not None and getattr(self, "label_lambda", 0.0) > 0.0:
    with torch.no_grad():
        _yv = y_t.clamp_min(0)
        _ce = torch.stack([F.cross_entropy(self.experts[_e].module(x), _yv, reduction="none") for _e in eids], dim=1).clamp(max=10.0)
        _ce = torch.where((y_t >= 0).unsqueeze(1), _ce, torch.zeros_like(_ce))
    LogF = LogF - self.label_lambda * _ce
"""


def _patched_observe():
    src = textwrap.dedent(inspect.getsource(PhyloMoE.observe))
    lines = src.split("\n")
    i = next(k for k, l in enumerate(lines) if "LogF = torch.stack(logf_list, dim=1)" in l)
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i + 1:i + 1] = [indent + l if l else l for l in _PATCH.rstrip("\n").split("\n")]
    ns: dict = {}
    exec("\n".join(lines), vars(phylo_mod), ns)
    return ns["observe"]


class LabelAwareA(PhyloMoE):
    label_lambda = 0.0

    def __init__(self, *a, label_lambda=0.0, **k):
        super().__init__(*a, **k)
        self.label_lambda = label_lambda


LabelAwareA.observe = _patched_observe()


def run(rho, seed, lam):
    st, _ = overlap_stream(seed, rho)
    pool = R.factory("A_new", st, seed, {"steps": 5})()
    pool = LabelAwareA(pool.cfg, label_lambda=lam)
    for x, y, ctx in st.algo_batches(64, seed):
        pool.observe(x, y, ctx)
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    acc = float((pool.predict(x).probs.argmax(1) == y).float().mean())
    ev = {k: sum(1 for e in pool.events if e.kind == k) for k in ("spawn", "extend", "alarm")}
    return {"rho": rho, "seed": seed, "lam": lam, "acc": acc, "experts": len(pool.experts), **ev}


if __name__ == "__main__":
    rows = []
    for rho in (0.0, 0.3, 0.6):
        for lam in (0.0, 1.0, 3.0):
            for seed in (100, 101, 102):
                rows.append(run(rho, seed, lam))
                print(rows[-1], flush=True)
    (HERE / "spike_label_assign.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\ncentre cos   lambda   accuracy %   experts (10 regimes)   spawn extend alarm")
    for rho in (0.0, 0.3, 0.6):
        for lam in (0.0, 1.0, 3.0):
            rr = [r for r in rows if r["rho"] == rho and r["lam"] == lam]
            m = lambda k: np.mean([r[k] for r in rr])                                      # noqa: E731
            print(f"{rho:9.1f} {lam:8.1f} {100 * m('acc'):12.1f} {m('experts'):14.1f}   {m('spawn'):9.1f} {m('extend'):6.1f} {m('alarm'):5.1f}")
