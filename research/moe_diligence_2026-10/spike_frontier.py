"""Spike L: memory-vs-accuracy frontier at 30 permuted-digit regimes (the spec's S2 size), 3 seeds.
Methods: kNN with capacity sweep, replay net with buffer sweep (20 steps), PhyloMoE default, and two right-sized PhyloMoE variants
(smaller hidden width, label head sized to the task, smaller signature and replay reservoirs). Memory = parameters + stored vectors (float32).
Throwaway diligence script; repo code untouched."""
import sys
import time

from dil_common import *  # noqa

from growmoe.baselines import ReplayNet
from growmoe.bench import run_stream
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from growmoe.streams import S2_SCHEDULE, s2_permuted

SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
D, BF = 128, 4


def sig_factory(protos):
    def make(d, max_protos, seed):
        return VmfSignature(d=d, max_protos=protos, seed=seed)
    return make


def a_mem(pool):
    vecs = sum(r.meta["sig"].n + (r.replay.n if r.replay is not None else 0) for r in pool.experts.values())
    params = sum(p.numel() for r in pool.experts.values() for p in r.module.parameters())
    return (params * BF + vecs * D * BF) / 1024.0, params, vecs


A_VARIANTS = {
    "A_default(h128,64cls,256 protos)": dict(),
    "A_small(h32,10cls,64 protos,64 replay)": dict(hidden=32, n_classes=10, replay_size=64, sig=64),
    "A_tiny(h16,10cls,32 protos,32 replay)": dict(hidden=16, n_classes=10, replay_size=32, sig=32),
}
rows = {}
t0 = time.time()
for sd in SEEDS:
    st = s2_permuted(seed=sd, n_regimes=30, schedule=S2_SCHEDULE, n_train=400, n_test=100)
    for name, kw in A_VARIANTS.items():
        kw = dict(kw)
        sigp = kw.pop("sig", None)
        hold = []

        def mk():
            cfg = PhyloCfg(seed=sd, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", router=False, spawn_rule="probe", growth_rule="probe", **kw)   # pre-stage-4 pool
            if sigp is not None:
                cfg.sig_cls = sig_factory(sigp)
            p = PhyloMoE(cfg)
            hold.append(p)
            return p

        res = run_stream(mk, st, seed=sd)
        kb, params, vecs = a_mem(hold[-1])
        rows.setdefault(name, []).append((res.accuracy_stats["avg"], kb, len(hold[-1].experts)))
        print(f"seed {sd} {name}: acc={res.accuracy_stats['avg']*100:.1f} mem={kb:.0f}KB experts={len(hold[-1].experts)} ({time.time()-t0:.0f}s)", flush=True)
    for cap in (500, 1000, 2000, 4000, 8000, 12000):
        res = run_stream(lambda: KnnPool(10, D, k=5, capacity=cap, seed=sd), st, seed=sd)
        rows.setdefault(f"kNN cap={cap}", []).append((res.accuracy_stats["avg"], cap * D * BF / 1024.0, 1))
    for buf in (512, 2048, 8192):
        hold = []

        def mkb():
            p = ReplayNet(CommonCfg(seed=sd, n_classes=10, inner_steps=20), replay_size=buf)
            hold.append(p)
            return p

        res = run_stream(mkb, st, seed=sd)
        pb = hold[-1]
        kb = (sum(p.numel() for p in pb.net.parameters()) * BF + pb.buffer.n * D * BF) / 1024.0
        rows.setdefault(f"B2 replay buf={buf} (20 steps)", []).append((res.accuracy_stats["avg"], kb, 1))
    print(f"seed {sd} done ({time.time()-t0:.0f}s)", flush=True)

print(f"\n=== 30 permuted-digit regimes (400 train each = 12,000 samples = 6,000 KB raw): accuracy vs memory, {len(SEEDS)} seeds ===")
print(f"{'method':42s} {'memory KB':>10s} {'accuracy %':>24s} {'experts':>8s}")
for name, r in sorted(rows.items(), key=lambda kv: np.mean([x[1] for x in kv[1]])):
    print(f"{name:42s} {np.mean([x[1] for x in r]):10.0f} {fmt([x[0] for x in r]):>24s} {np.mean([x[2] for x in r]):8.1f}")
