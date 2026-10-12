"""Spike A: strong simple baselines + memory accounting + event counts, on the BENCHMARK'S OWN stream config.
Reproduces A and B2 first (must match RESULTS.md), then adds kNN / NCM / SLDA. Throwaway diligence script."""
import sys
import time
from collections import Counter

from dil_common import *  # noqa

from growmoe.baselines import ReplayNet
from growmoe.bench import run_stream
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s1_types, s2_permuted

SEEDS = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 10))
BYTES_F = 4
D = 128


def make_streams(sd):
    s1 = s1_types(seed=sd, n_train=400, n_test=150)
    s2 = s2_permuted(seed=sd, n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100)
    return {"s1": s1, "s2": s2}


def a_memory(pool):
    vecs, nexp = 0, len(pool.experts)
    for rec in pool.experts.values():
        vecs += rec.meta["sig"].n
        if rec.replay is not None:
            vecs += rec.replay.n
    params = sum(p.numel() for r in pool.experts.values() for p in r.module.parameters())
    return vecs, params, nexp


rows = {}  # (pool, stream) -> dict of lists
events_a = {}


def add(key, **kw):
    r = rows.setdefault(key, {})
    for k, v in kw.items():
        r.setdefault(k, []).append(v)


t0 = time.time()
for sd in SEEDS:
    streams = make_streams(sd)
    for sname, st in streams.items():
        C = st.n_classes
        # ---- A (PhyloMoE), the benchmark's frozen config
        holder = []

        def mk_a():
            p = PhyloMoE(PhyloCfg(seed=sd, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf"))
            holder.append(p)
            return p

        res = run_stream(mk_a, st, seed=sd)
        pa = holder[-1]
        vecs, params, nexp = a_memory(pa)
        kb = (params * BYTES_F + vecs * D * BYTES_F) / 1024.0
        add(("A_phylomoe", sname), acc=res.accuracy_stats["avg"], forg=res.accuracy_stats["forgetting"], mem_kb=kb,
            vecs=vecs, params=params, nexp=nexp)
        cnt = Counter(e.kind for e in pa.events)
        events_a.setdefault(sname, []).append(cnt)
        a_vecs = vecs
        a_kb = kb

        # ---- B2 replay
        holder.clear()

        def mk_b2():
            p = ReplayNet(CommonCfg(seed=sd, n_classes=C))
            holder.append(p)
            return p

        res = run_stream(mk_b2, st, seed=sd)
        pb = holder[-1]
        kb = (sum(p.numel() for p in pb.net.parameters()) * BYTES_F + pb.buffer.n * D * BYTES_F) / 1024.0
        add(("B2_replay", sname), acc=res.accuracy_stats["avg"], forg=res.accuracy_stats["forgetting"], mem_kb=kb,
            vecs=pb.buffer.n, params=sum(p.numel() for p in pb.net.parameters()), nexp=1)

        # ---- simple baselines
        for label, mk in [
            ("kNN5_all_samples", lambda: KnnPool(C, D, k=5, capacity=None, seed=sd)),
            ("kNN5_same_#vectors_as_A", lambda: KnnPool(C, D, k=5, capacity=max(a_vecs, 16), seed=sd)),
            ("kNN5_same_BYTES_as_A", lambda: KnnPool(C, D, k=5, capacity=max(int(a_kb * 1024 / (D * BYTES_F)), 16), seed=sd)),
            ("NCM", lambda: NcmPool(C, D, seed=sd)),
            ("SLDA", lambda: SldaPool(C, D, seed=sd)),
        ]:
            holder.clear()

            def mk_x():
                p = mk()
                holder.append(p)
                return p

            res = run_stream(mk_x, st, seed=sd)
            p = holder[-1]
            kb = p.stored_vectors() * D * BYTES_F / 1024.0
            add((label, sname), acc=res.accuracy_stats["avg"], forg=res.accuracy_stats["forgetting"], mem_kb=kb,
                vecs=p.stored_vectors(), params=0, nexp=1)
    print(f"seed {sd} done  ({time.time()-t0:.0f}s)", flush=True)

print()
print(f"=== Final average accuracy over regimes (%), mean [95% CI] over {len(SEEDS)} seeds; memory = params + stored vectors (float32) ===")
for sname in ("s2", "s1"):
    print(f"\n-- stream {sname} (benchmark config: " + ("5 permuted regimes, 400 train each" if sname == "s2" else "7 data types + revisits") + ")")
    print(f"{'method':26s} {'accuracy':>22s} {'forgetting':>20s} {'memory KB':>10s} {'stored vecs':>11s} {'params':>9s} {'experts':>8s}")
    for (label, s), r in rows.items():
        if s != sname:
            continue
        print(f"{label:26s} {fmt(r['acc']):>22s} {fmt(r['forg']):>20s} {np.mean(r['mem_kb']):10.0f} "
              f"{np.mean(r['vecs']):11.0f} {np.mean(r['params']):9.0f} {np.mean(r['nexp']):8.1f}")

print("\n=== Event counts per run for A (mean over seeds) ===")
for sname, lst in events_a.items():
    kinds = sorted({k for c in lst for k in c})
    print(sname, {k: round(float(np.mean([c.get(k, 0) for c in lst])), 2) for k in kinds})
