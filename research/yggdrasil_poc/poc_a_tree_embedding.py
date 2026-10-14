"""SUPERSEDED - kept as the record of an UNTUNED run. Use poc_a4_tree_embedding_tuned.py.
This script trained every geometry with one constant learning rate; poc_a3 showed the d=32 retrieval score swings between 0.58 and
1.00 with the learning rate alone, so the table this script printed (including "hyperbolic stops winning at high d") was tuning noise.

PoC-A: where does hyperbolic geometry beat Euclidean for a hierarchy, as a function of dimension?

Gemini's tree benchmark used a 31-node tree at a single dimension (16). Theory (Sala et al. 2018) says the advantage
shows up at LOW dimension and for LARGE trees. This measures it: a ternary tree of depth 6 (1093 nodes), embedding
dimension d in {2, 4, 8, 16, 32}, three geometries at equal parameter count:
  Euclidean R^d | Poincare ball H^d | product H^(d/2) x T^(d/2) (the Yggdrasil base+fiber without intensity)
Loss: relative squared error to tree distance with a learned global scale. Metrics (all pairs, held-out is not needed:
this is an embedding-capacity test): mean relative distortion, and mean average precision (MAP) of retrieving each
node's tree neighbours (parent and children) by embedding distance.

Run: python research/yggdrasil_poc/poc_a_tree_embedding.py   (CPU, ~2 minutes, < 200 MB)
"""

import json
import math
import sys
import time
import warnings
from collections import deque
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

torch.set_num_threads(4)
SEEDS = (0, 1, 2)
DIMS = (2, 4, 8, 16, 32)
STEPS, PAIRS = 600, 60000


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
        seen = {s: 0}
        q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen[w] = seen[u] + 1
                    q.append(w)
        dist[s, list(seen)] = list(seen.values())
    return n, adj, torch.from_numpy(dist)


def poincare_points(x):
    nrm = x.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return torch.tanh(nrm) * x / nrm                      # exp map at the origin, radius < 1 automatically


def poincare_dist(u, v):
    uu = (u * u).sum(-1).clamp(max=1 - 1e-6)
    vv = (v * v).sum(-1).clamp(max=1 - 1e-6)
    xi = ((u - v) ** 2).sum(-1) / ((1 - uu) * (1 - vv))
    return 2.0 * torch.asinh(torch.sqrt(xi + 1e-12))


def torus_dist(a, b):
    d = torch.remainder(a - b + math.pi, 2 * math.pi) - math.pi
    return torch.sqrt((d ** 2).sum(-1) + 1e-12)


class Embedding(torch.nn.Module):
    def __init__(self, kind, n, d, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.kind, self.d = kind, d
        dh = d if kind != "product" else d // 2
        self.h = torch.nn.Parameter(torch.randn(n, dh, generator=g) * 0.1)
        self.t = torch.nn.Parameter(torch.rand(n, d - dh, generator=g) * 2 * math.pi) if kind == "product" else None
        self.log_alpha = torch.nn.Parameter(torch.tensor(0.0))

    def dist(self, i, j):
        if self.kind == "euclid":
            return ((self.h[i] - self.h[j]) ** 2).sum(-1).add(1e-12).sqrt()
        dh = poincare_dist(poincare_points(self.h[i]), poincare_points(self.h[j]))
        if self.kind == "poincare":
            return dh
        return torch.sqrt(dh ** 2 + torch.remainder(self.t[i] - self.t[j] + math.pi, 2 * math.pi).sub(math.pi).pow(2).sum(-1) + 1e-12)


def evaluate(model, tree_dist, adj):
    n = tree_dist.shape[0]
    with torch.no_grad():
        i = torch.arange(n)[:, None].expand(n, n).reshape(-1)
        j = torch.arange(n)[None, :].expand(n, n).reshape(-1)
        D = torch.cat([model.dist(i[k:k + 200000], j[k:k + 200000]) for k in range(0, n * n, 200000)]).reshape(n, n)
    mask = tree_dist > 0
    ratio = D[mask] / tree_dist[mask]
    alpha = ratio.median()
    distortion = ((D[mask] / alpha) / tree_dist[mask] - 1.0).abs().mean().item()
    D2 = D.clone()
    D2.fill_diagonal_(float("inf"))
    order = D2.argsort(dim=1)
    aps = []
    for u in range(n):
        nb = set(adj[u])
        rank = order[u].tolist()
        hits, ap = 0, 0.0
        for r, v in enumerate(rank, 1):
            if v in nb:
                hits += 1
                ap += hits / r
                if hits == len(nb):
                    break
        aps.append(ap / len(nb))
    return distortion, float(np.mean(aps))


def run(kind, d, seed, tree_dist, adj):
    n = tree_dist.shape[0]
    model = Embedding(kind, n, d, seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    g = torch.Generator().manual_seed(1000 + seed)
    for step in range(STEPS):
        i = torch.randint(0, n, (PAIRS,), generator=g)
        j = torch.randint(0, n, (PAIRS,), generator=g)
        keep = i != j
        i, j = i[keep], j[keep]
        target = tree_dist[i, j]
        pred = model.dist(i, j)
        a = model.log_alpha.exp()
        loss = (((pred - a * target) / (a * target)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return evaluate(model, tree_dist, adj)


def main():
    n, adj, tree_dist = build_tree()
    print(f"tree: {n} nodes, depth 6, branching 3; seeds {SEEDS}; {STEPS} Adam steps")
    results, t0 = {}, time.time()
    kinds = ["euclid", "poincare", "product"]
    for d in DIMS:
        for kind in kinds:
            if kind == "product" and d < 4:
                continue
            vals = [run(kind, d, s, tree_dist, adj) for s in SEEDS]
            results[f"{kind}_d{d}"] = {"distortion": [v[0] for v in vals], "map": [v[1] for v in vals]}
            dist_m = np.mean([v[0] for v in vals]); map_m = np.mean([v[1] for v in vals])
            print(f"d={d:2d} {kind:8s} distortion {dist_m:.3f}  MAP {map_m:.3f}   [{time.time() - t0:.0f}s]", flush=True)
    out = Path("research/yggdrasil_results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "poc_a_tree_embedding.json").write_text(json.dumps(results))
    print("\nMAP of retrieving each node's tree neighbours (mean over 3 seeds); higher is better")
    print(f"{'d':>4s} {'Euclid':>9s} {'Poincare':>9s} {'H x T':>9s}")
    for d in DIMS:
        row = [np.mean(results[f'{k}_d{d}']['map']) if f'{k}_d{d}' in results else float('nan') for k in kinds]
        print(f"{d:4d} " + " ".join(f"{v:9.3f}" for v in row))
    print("\nmean relative distortion; lower is better")
    for d in DIMS:
        row = [np.mean(results[f'{k}_d{d}']['distortion']) if f'{k}_d{d}' in results else float('nan') for k in kinds]
        print(f"{d:4d} " + " ".join(f"{v:9.3f}" for v in row))


if __name__ == "__main__":
    main()
