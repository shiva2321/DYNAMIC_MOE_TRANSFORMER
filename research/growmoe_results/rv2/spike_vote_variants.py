"""THROWAWAY spike on DEV seeds (100-102): should a sample move to the batch's modal expert ONLY if that expert's own gate accepts it?

  vote50    (spike_batch_guard.py) every accepted sample of the batch trains the modal expert, if it holds >= 50 % of them
  voter50   the same, but a sample moves only if the modal expert's gate also accepts it (P >= min(q, p_nov), the per-sample rule's own eligibility test)

The restricted rule should protect a minority regime sitting inside a batch (70/30 mixture) and cost nothing on single-source batches.

    python research/growmoe_results/rv2/spike_vote_variants.py
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
import spike_batch_guard as G                                                           # noqa: E402
from growmoe.pools import phylo as phylo_mod                                            # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.streams import Segment, Stream                                             # noqa: E402
from spike_overlap import overlap_stream                                                # noqa: E402

torch.set_num_threads(3)

_PATCH = """\
_vm = getattr(self, "vote_mode", "")
if _vm:
    _ok = assigned >= 0
    if int(_ok.sum()) >= 8:
        _vals, _cnts = torch.unique(assigned[_ok], return_counts=True)
        _owner, _share = int(_vals[int(_cnts.argmax())]), float(_cnts.max()) / float(_ok.sum())
        if _share >= 0.5:
            if _vm == "voter":
                _acc = P[:, eids.index(_owner)].double() >= thresholds.double()
                assigned[_ok & _acc] = _owner
            else:
                assigned[_ok] = _owner
"""


def _patched_observe():
    lines = textwrap.dedent(inspect.getsource(PhyloMoE.observe)).split("\n")
    i = next(k for k, l in enumerate(lines) if l.strip() == "spawned = []")
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i:i] = [indent + l if l else l for l in _PATCH.rstrip("\n").split("\n")]
    ns: dict = {}
    exec("\n".join(lines), vars(phylo_mod), ns)
    return ns["observe"]


class VotePool(PhyloMoE):
    vote_mode = ""

    def __init__(self, *a, vote_mode="", **k):
        super().__init__(*a, **k)
        self.vote_mode = vote_mode


VotePool.observe = _patched_observe()
MODES = {"V0": "", "vote50": "vote", "voter50": "voter"}


def mix_pairs_frac(st, frac_b):
    segs, out, i = st.segments, [], 0
    while i < len(segs):
        if i + 1 < len(segs):
            a, b = segs[i], segs[i + 1]
            sl = slice(0, min(int(round(len(a.y) * frac_b / (1 - frac_b))), len(b.y)))
            out.append(Segment(a.regime, a.first_visit and b.first_visit, torch.cat([a.x, b.x[sl]]), torch.cat([a.y, b.y[sl]]), torch.cat([a.src_ids, b.src_ids[sl]]),
                               torch.cat([a.y_clean, b.y_clean[sl]]), torch.cat([a.is_outlier, b.is_outlier[sl]]), None))
            i += 2
        else:
            out.append(segs[i])
            i += 1
    return Stream(st.name + "_mixed73", st.d, st.n_classes, st.regimes, out, dict(st.meta))


def get(stream, seed):
    if stream == "ov0.3-mixed73":
        st, _ = overlap_stream(seed, 0.3)
        return mix_pairs_frac(st, 0.3), 5
    return G.get(stream, seed)


def run(stream, seed, variant):
    st, steps = get(stream, seed)
    pool = R.factory("A_new", st, seed, {"steps": steps})()
    pool = VotePool(pool.cfg, vote_mode=MODES[variant])
    for x, y, ctx in st.algo_batches(64, seed):
        pool.observe(x, y, ctx)
    x = torch.cat([r.x_te for r in st.regimes.values()])
    y = torch.cat([r.y_te for r in st.regimes.values()])
    return {"stream": stream, "seed": seed, "variant": variant, "acc": float((pool.predict(x).probs.argmax(1) == y).float().mean()), "experts": len(pool.experts)}


if __name__ == "__main__":
    rows = []
    for stream in ["ov0.3", "ov0.3-mixed", "ov0.3-mixed73", "ov0.6", "F-15", "S1"]:
        for variant in (["V0", "vote50", "voter50"] if stream == "ov0.3-mixed73" else ["voter50"]):
            for seed in (100, 101, 102):
                rows.append(run(stream, seed, variant))
                print(rows[-1], flush=True)
    (HERE / "spike_vote_variants.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    old = [r for r in json.loads((HERE / "spike_batch_guard.json").read_text(encoding="utf-8")) if r["variant"] in ("V0", "vote50")]
    old = [{**r, "variant": r["variant"]} for r in old]
    allr = rows + old
    print("\nstream          V0     vote50   voter50   (accuracy %, mean of seeds 100-102)")
    for stream in ["ov0.3", "ov0.3-mixed", "ov0.3-mixed73", "ov0.6", "F-15", "S1"]:
        cells = []
        for v in ("V0", "vote50", "voter50"):
            rr = [r for r in allr if r["stream"] == stream and r["variant"] == v]
            cells.append(f"{100 * np.mean([r['acc'] for r in rr]):6.1f}" if rr else "     -")
        print(f"{stream:14s} " + "   ".join(cells))
