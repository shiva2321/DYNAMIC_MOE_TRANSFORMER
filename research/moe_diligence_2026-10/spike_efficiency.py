"""Spike M: what does one PREDICTION cost? PhyloMoE after a 30-regime run vs kNN vs a single replay-trained net.
Wall-clock on CPU (4 threads) plus an analytic multiply-add count. Throwaway diligence script (repo code untouched)."""
import time

from dil_common import *  # noqa

from growmoe.baselines import ReplayNet
from growmoe.bench import run_stream
from growmoe.config import CommonCfg, PhyloCfg
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import S2_SCHEDULE, s2_permuted

D = 128
sd = 0
st = s2_permuted(seed=sd, n_regimes=30, schedule=S2_SCHEDULE, n_train=400, n_test=100)
hold = {}


def mk_a():
    p = PhyloMoE(PhyloCfg(seed=sd, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf"))
    hold["A"] = p
    return p


def mk_k_all():
    p = KnnPool(10, D, k=5, capacity=None, seed=sd)
    hold["knn_all"] = p
    return p


def mk_k_2k():
    p = KnnPool(10, D, k=5, capacity=4000, seed=sd)      # 4000 stored vectors = 2000 KB
    hold["knn_4000"] = p
    return p


def mk_b2():
    p = ReplayNet(CommonCfg(seed=sd, n_classes=10, inner_steps=20))
    hold["B2"] = p
    return p


for mk in (mk_a, mk_k_all, mk_k_2k, mk_b2):
    run_stream(mk, st, seed=sd)

# a fixed query batch drawn from all 30 regimes' test sets
xs = torch.cat([reg.x_te[:3] for reg in st.regimes.values()])[:64]


def timeit(fn, reps):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


pa = hold["A"]
E = len(pa.experts)
n_protos = sum(r.meta["sig"].n for r in pa.experts.values())
expert_macs = sum(p.numel() for p in next(iter(pa.experts.values())).module.parameters() if p.dim() == 2)   # weight matrices only
gate_macs_asimpl = 2 * n_protos * D          # pvalue() and logpdf() each recompute the score against every stored prototype
gate_macs_once = n_protos * D
knn_all_macs = hold["knn_all"].stored_vectors() * D
knn_4k_macs = hold["knn_4000"].stored_vectors() * D
b2_macs = sum(p.numel() for p in hold["B2"].net.parameters() if p.dim() == 2)

print(f"PhyloMoE after 30 regimes: {E} live experts, {n_protos} stored signature prototypes in total")
print(f"{'method':34s} {'ms/sample B=64':>15s} {'ms/sample B=1':>14s} {'multiply-adds/sample':>22s}")
rows = [
    ("PhyloMoE (as implemented)", lambda x: pa.predict(x), gate_macs_asimpl + expert_macs),
    ("kNN, 12,000 stored (everything)", lambda x: hold["knn_all"].predict(x), knn_all_macs),
    ("kNN, 4,000 stored (2 MB)", lambda x: hold["knn_4000"].predict(x), knn_4k_macs),
    ("Replay network (one MLP)", lambda x: hold["B2"].predict(x), b2_macs),
]
for name, fn, macs in rows:
    t64 = timeit(lambda: fn(xs), 5) / xs.shape[0] * 1000
    x1 = xs[:1]
    t1 = timeit(lambda: fn(x1), 20) * 1000
    print(f"{name:34s} {t64:15.3f} {t1:14.3f} {macs:22,d}")

print()
print(f"PhyloMoE multiply-adds per sample, split: gate (scan every expert's stored vectors) = {gate_macs_asimpl:,} as implemented "
      f"({gate_macs_once:,} if the score were computed once); the ONE expert that answers = {expert_macs:,}.")
print(f"=> the gate costs {gate_macs_asimpl/expert_macs:.0f}x the expert it selects; total vs kNN-everything: "
      f"{(gate_macs_asimpl+expert_macs)/knn_all_macs:.2f}x")
