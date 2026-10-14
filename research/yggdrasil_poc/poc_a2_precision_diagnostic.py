"""PoC-A2: why does the Poincare retrieval score (MAP) FALL as the dimension grows in PoC-A?

Two candidate causes:
  (1) float32 precision near the ball boundary (points crowd where 1 - |x|^2 is below float32 resolution);
  (2) the optimisation / the loss (not enough steps, loss dominated by far pairs).
Test: train in float32 exactly as PoC-A, then evaluate the SAME parameters in float64. If MAP jumps, it is precision;
if it stays, it is the optimisation. Also print how close to the boundary the points ended up.

Run: python research/yggdrasil_poc/poc_a2_precision_diagnostic.py   (CPU, ~2 minutes, < 200 MB)
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import poc_a_tree_embedding as A

torch.set_num_threads(4)


def train(kind, d, seed, tree_dist):
    n = tree_dist.shape[0]
    model = A.Embedding(kind, n, d, seed)
    opt = torch.optim.Adam(model.parameters(), lr=0.03)
    g = torch.Generator().manual_seed(1000 + seed)
    for _ in range(A.STEPS):
        i = torch.randint(0, n, (A.PAIRS,), generator=g)
        j = torch.randint(0, n, (A.PAIRS,), generator=g)
        keep = i != j
        i, j = i[keep], j[keep]
        target = tree_dist[i, j]
        a = model.log_alpha.exp()
        loss = (((model.dist(i, j) - a * target) / (a * target)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def poincare_dist_f64(u, v):
    eps = torch.finfo(u.dtype).eps
    uu = (u * u).sum(-1).clamp(max=1 - 4 * eps)
    vv = (v * v).sum(-1).clamp(max=1 - 4 * eps)
    xi = ((u - v) ** 2).sum(-1) / ((1 - uu) * (1 - vv))
    return 2.0 * torch.asinh(torch.sqrt(xi + 1e-24))


def main():
    n, adj, tree_dist = A.build_tree()
    f32_dist = A.poincare_dist
    print(f"{'geometry':9s} {'d':>3s} {'seed':>4s} | {'MAP f32':>8s} {'MAP f64':>8s} | {'dist f32':>8s} {'dist f64':>8s} | "
          f"{'r>0.999':>8s} {'r>0.9999':>8s} {'max |tangent|':>13s}")
    rows = []
    for kind, d in [("poincare", 8), ("poincare", 32), ("euclid", 32)]:
        for seed in (0, 1):
            model = train(kind, d, seed, tree_dist)
            A.poincare_dist = f32_dist
            dist32, map32 = A.evaluate(model, tree_dist, adj)
            model64 = model.double()
            A.poincare_dist = poincare_dist_f64
            dist64, map64 = A.evaluate(model64, tree_dist.double(), adj)
            A.poincare_dist = f32_dist
            if kind == "poincare":
                tn = model64.h.detach().norm(dim=-1)
                radius = torch.tanh(tn)
                f3, f4, mx = (radius > 0.999).float().mean().item(), (radius > 0.9999).float().mean().item(), tn.max().item()
            else:
                f3 = f4 = mx = float("nan")
            print(f"{kind:9s} {d:3d} {seed:4d} | {map32:8.3f} {map64:8.3f} | {dist32:8.3f} {dist64:8.3f} | "
                  f"{f3:8.3f} {f4:8.3f} {mx:13.2f}", flush=True)
            rows.append((kind, d, seed, map32, map64))
    print("\nread-out: if 'MAP f64' is clearly above 'MAP f32' for poincare d=32, float32 precision at the boundary is hurting;")
    print("if they are equal, the loss of retrieval quality comes from the optimisation, not from the arithmetic.")


if __name__ == "__main__":
    main()
