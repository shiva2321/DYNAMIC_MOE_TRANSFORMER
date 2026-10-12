"""THROWAWAY spike on DEV seeds (100-102): is batch-consistent assignment safe where batches are NOT single-source, and on the standard streams?

Variants (patched copy of PhyloMoE.observe, no product file touched), all act only on the samples the gate already accepted:
  V0        baseline (the repaired pool)
  vote50    all accepted samples of a batch train the expert most of them were routed to, if that expert holds >= 50 % of them
  vote80    the same with a share of >= 80 %
  llr50     the expert is chosen by the SUM of the log-densities over the batch (evidence adds up) instead of by counting votes; share >= 50 %

Streams: ov0.3 / ov0.6 (10 regimes, centre cosine 0.3 / 0.6, single-source batches), ov0.3-mixed (the same stream with every two consecutive segments interleaved, so every batch is a
50/50 mixture of two regimes), and the standard streams F-15, S2-5, S1 at the M2-A tuned steps.

    python research/growmoe_results/rv2/spike_batch_guard.py
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
from growmoe.streams import Segment, Stream                                             # noqa: E402
from spike_overlap import overlap_stream                                                # noqa: E402

torch.set_num_threads(3)
TUNING = json.loads((HERE.parent / "m2a" / "tuning_m2a.json").read_text(encoding="utf-8"))

_PATCH = """\
_vm = getattr(self, "vote_mode", "")
if _vm:
    _ok = assigned >= 0
    if int(_ok.sum()) >= 8:
        if _vm == "vote":
            _vals, _cnts = torch.unique(assigned[_ok], return_counts=True)
            _owner, _share = int(_vals[int(_cnts.argmax())]), float(_cnts.max()) / float(_ok.sum())
        else:
            _owner = eids[int(LogF[_ok].sum(0).argmax())]
            _share = float((assigned[_ok] == _owner).float().mean())
        if _share >= self.vote_share:
            assigned[_ok] = _owner
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


class VotePool(PhyloMoE):
    vote_mode, vote_share = "", 0.5

    def __init__(self, *a, vote_mode="", vote_share=0.5, **k):
        super().__init__(*a, **k)
        self.vote_mode, self.vote_share = vote_mode, vote_share


VotePool.observe = _patched_observe()
VARIANTS = {"V0": ("", 0.5), "vote50": ("vote", 0.5), "vote80": ("vote", 0.8), "llr50": ("llr", 0.5)}


def mix_pairs(st):
    segs, out, i = st.segments, [], 0
    while i < len(segs):
        if i + 1 < len(segs):
            a, b = segs[i], segs[i + 1]
            out.append(Segment(a.regime, a.first_visit and b.first_visit, torch.cat([a.x, b.x]), torch.cat([a.y, b.y]), torch.cat([a.src_ids, b.src_ids]),
                               torch.cat([a.y_clean, b.y_clean]), torch.cat([a.is_outlier, b.is_outlier]), None))
            i += 2
        else:
            out.append(segs[i])
            i += 1
    return Stream(st.name + "_mixed", st.d, st.n_classes, st.regimes, out, dict(st.meta))


def get(stream, seed):
    if stream.startswith("ov"):
        rho = float(stream[2:5])
        st, _ = overlap_stream(seed, rho)
        return (mix_pairs(st) if stream.endswith("mixed") else st), 5
    st = R.get_stream(stream, seed, False)
    return st, TUNING["A_new"][R.FAMILY[stream]]["steps"]


def run(stream, seed, variant):
    st, steps = get(stream, seed)
    pool = R.factory("A_new", st, seed, {"steps": steps})()
    mode, share = VARIANTS[variant]
    pool = VotePool(pool.cfg, vote_mode=mode, vote_share=share)
    for x, y, ctx in st.algo_batches(64, seed):
        pool.observe(x, y, ctx)
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    acc = float((pool.predict(x).probs.argmax(1) == y).float().mean())
    ev = {k: sum(1 for e in pool.events if e.kind == k) for k in ("spawn", "extend", "alarm")}
    return {"stream": stream, "seed": seed, "variant": variant, "acc": acc, "experts": len(pool.experts), **ev}


if __name__ == "__main__":
    rows = []
    STREAMS = ["ov0.3", "ov0.3-mixed", "ov0.6", "F-15", "S2-5", "S1"]
    for stream in STREAMS:
        for variant in VARIANTS:
            for seed in (100, 101, 102):
                rows.append(run(stream, seed, variant))
                print(rows[-1], flush=True)
    (HERE / "spike_batch_guard.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print("\nstream         variant   accuracy %   experts   spawn extend alarm")
    for stream in STREAMS:
        for variant in VARIANTS:
            rr = [r for r in rows if r["stream"] == stream and r["variant"] == variant]
            m = lambda k: np.mean([r[k] for r in rr])                                      # noqa: E731
            print(f"{stream:13s} {variant:8s} {100 * m('acc'):10.1f} {m('experts'):9.1f}   {m('spawn'):5.1f} {m('extend'):6.1f} {m('alarm'):5.1f}")
