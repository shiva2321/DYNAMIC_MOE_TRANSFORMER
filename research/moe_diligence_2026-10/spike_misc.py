"""Spikes F, H, E, D, G (throwaway diligence): norm-gate degeneracy, Pool C parameter accounting,
TwoNN at probation-cluster sizes, capacity-vs-labels demo, routing cost vs number of experts."""
import math
import time

from dil_common import *  # noqa
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------- F
print("=== F. NormGatedLinear on unit-sphere inputs ===")
from geometry.neuron import NormGatedLinear  # noqa: E402

torch.manual_seed(0)
layer = NormGatedLinear(128, 16, curvature=1.0)
with torch.no_grad():
    layer.time_weight.copy_(torch.randn(16) * 0.7)
    layer.bias.copy_(torch.randn(16) * 0.1)
x = torch.randn(512, 128)
x = x / x.norm(dim=1, keepdim=True)
rho_const = math.sqrt(1.0 / layer.curvature + 1.0)
plain = F.linear(x, layer.weight, layer.bias - layer.time_weight * rho_const)
print("max |NormGated(x) - plain linear with shifted bias| on unit vectors:", float((layer(x) - plain).abs().max()))
xs = torch.randn(512, 128) * 3.0                       # NOT unit norm: the norm term now varies
plain_s = F.linear(xs, layer.weight, layer.bias - layer.time_weight * rho_const)
print("same comparison on inputs with varying norm (so the gate can act):", float((layer(xs) - plain_s).abs().max()))

# ----------------------------------------------------------------------------------------------- H
print("\n=== H. Pool C parameter accounting on S2 (benchmark config) ===")
from growmoe.bench import run_stream  # noqa: E402
from growmoe.config import KeyedCfg, PhyloCfg  # noqa: E402
from growmoe.pools.keyed import KeyedDeltaMoE  # noqa: E402
from growmoe.pools.phylo import PhyloMoE  # noqa: E402
from growmoe.streams import s2_permuted  # noqa: E402

for sd in range(3):
    st = s2_permuted(seed=sd, n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100)
    hold = []

    def mk_c():
        p = KeyedDeltaMoE(KeyedCfg(seed=sd, c=2.0, rank=16, n_stm=32))
        hold.append(p)
        return p

    res = run_stream(mk_c, st, seed=sd)
    pc = hold[-1]
    summary_n = pc.summary()["n_params"]
    uniq = {}
    for r in pc.experts.values():
        for p in r.module.parameters():
            uniq[id(p)] = p.numel()
    uniq_n = sum(uniq.values())
    own = sum(r.module.num_params() for r in pc.experts.values())
    base_n = sum(p.numel() for p in pc.base_mlp.parameters())
    print(f"seed {sd}: experts={len(pc.experts)}  summary n_params={summary_n}  unique params={uniq_n}  "
          f"(own deltas {own} + shared base {base_n} = {own + base_n})  acc={res.accuracy_stats['avg']*100:.1f}")
    hold.clear()
    holda = []

    def mk_a():
        p = PhyloMoE(PhyloCfg(seed=sd, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf"))
        holda.append(p)
        return p

    resa = run_stream(mk_a, st, seed=sd)
    pa = holda[-1]
    print(f"          Pool A same seed: experts={len(pa.experts)} summary n_params={pa.summary()['n_params']} acc={resa.accuracy_stats['avg']*100:.1f}")

# ----------------------------------------------------------------------------------------------- E
print("\n=== E. Intrinsic-dimension estimates at probation-cluster sizes (true dimension known; embedded in R^128, unit-normalised) ===")
from geomlab.diagnostics import intrinsic_dim_twonn  # noqa: E402


def agent_formula(X):
    """Re-implementation of the second agent's AutoSizer estimate (median of log mu, then 1/x)."""
    with torch.no_grad():
        dists = torch.cdist(X.double(), X.double())
        dists.fill_diagonal_(float("inf"))
        r1, _ = dists.min(dim=-1)
        dists.scatter_(-1, dists.argmin(dim=-1, keepdim=True), float("inf"))
        r2, _ = dists.min(dim=-1)
        mu = (r2 / (r1 + 1e-9)).clamp(min=1.0001)
        d_int = float(torch.median(torch.log(mu)))
        return min(max(1.0, 1.0 / (d_int + 1e-6)), 128.0)


rng = np.random.default_rng(0)
print(f"{'true d':>6s} {'N':>5s} | {'repo TwoNN mean+-sd':>22s} | {'agent formula mean+-sd':>24s}")
for d_true in (2, 5, 10, 20):
    for N in (32, 64, 128, 256, 512):
        est_repo, est_agent = [], []
        for rep in range(30):
            basis = np.linalg.qr(rng.standard_normal((128, d_true)))[0]          # d_true orthonormal directions
            z = rng.standard_normal((N, d_true)) @ basis.T + 0.01 * rng.standard_normal((N, 128))
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
            X = torch.from_numpy(z.astype(np.float32))
            est_repo.append(intrinsic_dim_twonn(X))
            est_agent.append(agent_formula(X))
        print(f"{d_true:6d} {N:5d} | {np.mean(est_repo):8.2f} +- {np.std(est_repo):5.2f}      | {np.mean(est_agent):9.2f} +- {np.std(est_agent):5.2f}")

# ----------------------------------------------------------------------------------------------- D
print("\n=== D. Same inputs, different labelling function: needed expert width (accuracy %, 3 seeds) ===")
from growmoe.experts import MLPExpert  # noqa: E402
from growmoe.streams import _balanced_teacher  # noqa: E402

D_IN, C = 128, 8
widths = (2, 4, 8, 16, 32, 64, 128)
results = {"linear labels": {w: [] for w in widths}, "deep teacher (w32,d2)": {w: [] for w in widths}}
for sd in range(3):
    g = np.random.default_rng([sd, 0xD1])
    xs = g.standard_normal((9000, D_IN))
    xs = xs / np.linalg.norm(xs, axis=1, keepdims=True)
    X = torch.from_numpy(xs.astype(np.float32))
    Wl = torch.from_numpy(g.standard_normal((D_IN, C)).astype(np.float32))
    teacher = _balanced_teacher(32, 2, D_IN, C, seed=sd)
    with torch.no_grad():
        labels = {"linear labels": (X @ Wl).argmax(1), "deep teacher (w32,d2)": teacher(X).argmax(1)}
    Xtr, Xte = X[:6000], X[6000:]
    for name, y in labels.items():
        ytr, yte = y[:6000], y[6000:]
        for w in widths:
            m = MLPExpert(D_IN, w, 1, C, seed=sd)
            m.mark_seen(torch.arange(C))
            opt = torch.optim.Adam(m.parameters(), lr=3e-3)
            gen = np.random.default_rng([sd, w])
            for _ in range(1500):
                idx = torch.from_numpy(gen.integers(0, 6000, 64))
                m.train_step(Xtr[idx], ytr[idx], opt)
            m.eval()
            with torch.no_grad():
                results[name][w].append(float((m(Xte).argmax(1) == yte).float().mean()))
print(f"{'labels':24s} " + " ".join(f"w={w:<4d}" for w in widths))
for name, r in results.items():
    print(f"{name:24s} " + " ".join(f"{100*np.mean(r[w]):6.1f}" for w in widths))
print("(inputs are IDENTICAL random unit vectors in both rows: same N, same concentration, same intrinsic dimension)")

# ----------------------------------------------------------------------------------------------- G
print("\n=== G. Routing (predict) cost vs number of live experts, batch of 64 ===")
from growmoe.experts import MLPExpert as _E  # noqa: E402
from growmoe.signature import VmfSignature  # noqa: E402

for E in (8, 32, 128, 512):
    pool = PhyloMoE(PhyloCfg(seed=0, route_rule="accept_then_logf", max_experts=2048))
    g = np.random.default_rng(E)
    for e in range(E):
        sig = VmfSignature(128, 256, seed=e)
        z = g.standard_normal((256, 128))
        sig.update(torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32)))
        sig.fit()
        pool.add_expert(_E(128, 128, 1, 64, seed=e), parent=None, state="active",
                        meta={"sig": sig, "monitor": None, "recent": None, "infant_calls_seen": 5, "loss_hist": [], "policy": None})
    xq = torch.from_numpy((lambda z: (z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))(g.standard_normal((64, 128))))
    pool.predict(xq)                                                   # warm-up
    t0 = time.perf_counter()
    for _ in range(3):
        pool.predict(xq)
    dt = (time.perf_counter() - t0) / 3
    print(f"E={E:4d}: {dt*1000:8.1f} ms per batch of 64  ({dt/64*1000:6.2f} ms per sample)")
