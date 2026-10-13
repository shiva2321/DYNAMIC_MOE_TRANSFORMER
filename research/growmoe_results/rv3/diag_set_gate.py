"""THROWAWAY diagnostic: log every set-test decision of the live set-gate on one stream/seed (why does it over-fire?).   (entry E-023)

    python research/growmoe_results/rv3/diag_set_gate.py OV-0.6 100 md_recent [trace]
"""
import collections
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spike_set_gate as G
import run_rv2a as A
from growmoe.config import PhyloCfg

name, seed, variant = sys.argv[1], int(sys.argv[2]), sys.argv[3]
trace = len(sys.argv) > 4 and sys.argv[4] == "trace"
st = A.get_stream(name, seed, False)
pool = G.SetGatePool(PhyloCfg(seed=seed, d=st.d, n_classes=st.n_classes, inner_steps=G.STEPS[A.family(name)], q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf"),
                     set_cfg=G.VARIANTS[variant])
orig = pool._set_decide
log = []


def spy(eids, xb, vo=None):
    verdict, owner = orig(eids, xb, vo)
    log.append((verdict, owner, len(xb), list(pool.last_detail), vo))
    return verdict, owner


pool._set_decide = spy
for i, ((k, seg, _), (x, y, ctx)) in enumerate(zip(st._walk(64, seed), st.algo_batches(64, seed))):
    n0, ne0 = len(log), len(pool.experts)
    out = pool.observe(x, y, ctx)
    if trace:
        print(f"{i:3d} reg {seg.regime} {'1st' if seg.first_visit else 'rev'} E={ne0:3d}->{len(pool.experts):3d} flagged={int(out.novel.sum()):2d} " + " ".join(f"[{v} vote=e{vo} owner=e{o} n={n} " + " ".join(f"(e{e} d={d} p={pp:.3f} ref={r})" for e, d, pp, r in det) + "]" for v, o, n, det, vo in log[n0:]), flush=True)
    if trace and i >= int(__import__("os").environ.get("NB", "70")):
        break
print("events:", dict(collections.Counter(e.kind for e in pool.events)), "set decisions:", dict(collections.Counter(pool.set_events)), "experts", len(pool.experts))
