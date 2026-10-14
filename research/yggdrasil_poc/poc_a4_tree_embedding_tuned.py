"""PoC-A4: tree embedding, Euclidean vs hyperbolic vs product geometry, with a FAIR (per-geometry tuned) protocol.

Why this replaces PoC-A: PoC-A used one learning rate and no decay for every geometry; PoC-A3 showed that at d=32 the
retrieval score swings between 0.58 and 1.00 for the SAME geometry just by changing the learning rate, so PoC-A's
high-dimension numbers were tuning noise. Here every (geometry, dimension) gets its own learning rate from a small grid,
chosen on a dev seed (100) and then evaluated on test seeds 0, 1, 2. Cosine learning-rate decay everywhere.

Geometries (all with the same number of coordinates d):
  euclid   R^d
  poincare Poincare ball H^d (exp map at the origin of a tangent parameter; geodesic distance)
  lorentz  hyperboloid H^d (same geometry, different numerics: acosh(-<x,y>_L))
  product  H^(d/2) x T^(d/2) (Poincare ball x flat torus; the Yggdrasil base + phase fibre without intensity)
Data: ternary tree of depth 6 (1093 nodes). Loss: relative squared error to the tree distance, learned global scale.
Metrics: mean relative distortion of all pairs; MAP of retrieving each node's tree neighbours (parent + children).
The two metrics are tuned SEPARATELY (best learning rate per metric), the same way for every geometry.

Run: python research/yggdrasil_poc/poc_a4_tree_embedding_tuned.py        (GPU, tiny tensors, < 300 MB, ~10 minutes)
"""

import json
import math
import os
import sys
import time
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIMS = tuple(int(x) for x in os.environ.get("POC_DIMS", "2,4,8,16,32").split(","))
KINDS = tuple(os.environ.get("POC_KINDS", "euclid,poincare,lorentz,product").split(","))
LRS = tuple(float(x) for x in os.environ.get("POC_LRS", "0.1,0.03,0.01,0.003").split(","))
STEPS, PAIRS = int(os.environ.get("POC_STEPS", "600")), 30000
DEV_SEED, TEST_SEEDS = 100, (0, 1, 2)


def build_tree(depth=6, branching=3):
    parents, frontier = [-1], [0]
    for _ in range(depth):
        new = []
        for u in frontier:
            for _ in range(branching):
                parents.append(u)
                new.append(len(parents) - 1)
        frontier = new
    n = len(parents)
    adj = [[] for _ in range(n)]
    for v, u in enumerate(parents):
        if u >= 0:
            adj[u].append(v)
            adj[v].append(u)
    dist = np.zeros((n, n), dtype=np.float32)
    for s in range(n):
        seen, q = {s: 0}, deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen[w] = seen[u] + 1
                    q.append(w)
        dist[s, list(seen)] = list(seen.values())
    width = max(len(a) for a in adj)
    nbr = torch.full((n, width), -1, dtype=torch.long)
    for u, a in enumerate(adj):
        nbr[u, : len(a)] = torch.tensor(a)
    return n, torch.from_numpy(dist).to(DEV), nbr.to(DEV)


def poincare_dist(u, v):
    uu = (u * u).sum(-1).clamp(max=1 - 1e-6)
    vv = (v * v).sum(-1).clamp(max=1 - 1e-6)
    xi = ((u - v) ** 2).sum(-1) / ((1 - uu) * (1 - vv))
    return 2.0 * torch.asinh(torch.sqrt(xi + 1e-12))


def to_ball(x):
    nrm = x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.tanh(nrm) * x / nrm


def to_hyperboloid(x):
    nrm = x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.cat([torch.cosh(nrm), torch.sinh(nrm) * x / nrm], -1)


def lorentz_dist(a, b):
    inner = -a[..., 0] * b[..., 0] + (a[..., 1:] * b[..., 1:]).sum(-1)
    return torch.acosh((-inner).clamp(min=1.0 + 1e-6))


class Embedding(torch.nn.Module):
    def __init__(self, kind, n, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind = kind
        dh = d // 2 if kind == "product" else d
        self.h = torch.nn.Parameter((torch.randn(n, dh, generator=g) * 0.1).to(DEV))
        self.t = torch.nn.Parameter((torch.rand(n, d - dh, generator=g) * 2 * math.pi).to(DEV)) if kind == "product" else None
        self.log_alpha = torch.nn.Parameter(torch.zeros((), device=DEV))

    def dist(self, i, j):
        if self.kind == "euclid":
            return ((self.h[i] - self.h[j]) ** 2).sum(-1).add(1e-12).sqrt()
        if self.kind == "lorentz":
            return lorentz_dist(to_hyperboloid(self.h[i]), to_hyperboloid(self.h[j]))
        dh = poincare_dist(to_ball(self.h[i]), to_ball(self.h[j]))
        if self.kind == "poincare":
            return dh
        dt = torch.remainder(self.t[i] - self.t[j] + math.pi, 2 * math.pi) - math.pi
        return torch.sqrt(dh ** 2 + (dt ** 2).sum(-1) + 1e-12)


@torch.no_grad()
def evaluate(model, tree_dist, nbr):
    n = tree_dist.shape[0]
    idx = torch.arange(n, device=DEV)
    D = torch.cat([model.dist(idx[:, None].expand(n, n).reshape(-1)[k:k + 200000], idx[None, :].expand(n, n).reshape(-1)[k:k + 200000])
                   for k in range(0, n * n, 200000)]).reshape(n, n)
    off = tree_dist > 0
    alpha = (D[off] / tree_dist[off]).median()
    distortion = ((D[off] / alpha) / tree_dist[off] - 1.0).abs().mean().item()
    D2 = D.clone()
    D2.fill_diagonal_(float("inf"))
    order = D2.argsort(dim=1)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(n, device=DEV).expand(n, n))
    valid = nbr >= 0
    r = (rank.gather(1, nbr.clamp(min=0)).float() + 1).masked_fill(~valid, float("inf"))
    r, _ = r.sort(dim=1)
    m = valid.sum(1).float()
    prec = torch.arange(1, nbr.shape[1] + 1, device=DEV).float()[None] / r
    ap = (prec.masked_fill(~torch.isfinite(r), 0.0)).sum(1) / m
    return distortion, ap.mean().item()


def train_eval(kind, d, seed, lr, tree_dist, nbr):
    n = tree_dist.shape[0]
    model = Embedding(kind, n, d, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    g = torch.Generator(device=DEV).manual_seed(1000 + seed)
    for _ in range(STEPS):
        i = torch.randint(0, n, (PAIRS,), generator=g, device=DEV)
        j = torch.randint(0, n, (PAIRS,), generator=g, device=DEV)
        keep = i != j
        i, j = i[keep], j[keep]
        target = tree_dist[i, j]
        a = model.log_alpha.exp()
        loss = (((model.dist(i, j) - a * target) / (a * target)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return evaluate(model, tree_dist, nbr)


def main():
    n, tree_dist, nbr = build_tree()
    print(f"tree: {n} nodes (ternary, depth 6); device {DEV}; {STEPS} Adam steps with cosine decay; lr grid {LRS}; "
          f"selection seed {DEV_SEED}, test seeds {TEST_SEEDS}", flush=True)
    res, t0 = {}, time.time()
    for d in DIMS:
        for kind in KINDS:
            if kind == "product" and d < 4:
                continue
            sel = {lr: train_eval(kind, d, DEV_SEED, lr, tree_dist, nbr) for lr in LRS}            # lr -> (distortion, MAP)
            lr_map = max(LRS, key=lambda lr: sel[lr][1])
            lr_dis = min(LRS, key=lambda lr: sel[lr][0])
            runs_map = [train_eval(kind, d, s, lr_map, tree_dist, nbr) for s in TEST_SEEDS]
            runs_dis = runs_map if lr_dis == lr_map else [train_eval(kind, d, s, lr_dis, tree_dist, nbr) for s in TEST_SEEDS]
            res[f"{kind}_d{d}"] = {"lr_map": lr_map, "lr_dis": lr_dis,
                                   "map": [r[1] for r in runs_map], "distortion": [r[0] for r in runs_dis],
                                   "sel": {str(k): v for k, v in sel.items()}}
            print(f"d={d:2d} {kind:8s} best-MAP {np.mean([r[1] for r in runs_map]):.3f} (lr {lr_map:g})   "
                  f"best-distortion {np.mean([r[0] for r in runs_dis]):.3f} (lr {lr_dis:g})   [{time.time() - t0:.0f}s]", flush=True)
    out = Path("research/yggdrasil_results")
    out.mkdir(parents=True, exist_ok=True)
    (out / os.environ.get("POC_OUT", "poc_a4_tree_embedding_tuned.json")).write_text(json.dumps(res))
    for title, key in (("MAP of retrieving each node's tree neighbours (higher is better; best lr per cell; mean of 3 seeds)", "map"),
                       ("mean relative distortion of all pairs (lower is better; best lr per cell; mean of 3 seeds)", "distortion")):
        print("\n" + title)
        print(f"{'d':>4s} " + " ".join(f"{k:>9s}" for k in KINDS))
        for d in DIMS:
            print(f"{d:4d} " + " ".join(f"{np.mean(res[f'{k}_d{d}'][key]):9.3f}" if f"{k}_d{d}" in res else f"{'-':>9s}" for k in KINDS))


if __name__ == "__main__":
    main()
