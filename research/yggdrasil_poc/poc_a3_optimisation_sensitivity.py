"""PoC-A3: is the 'hyperbolic stops winning at d=32' result a training-budget artefact?

PoC-A trained every geometry with one setting (600 Adam steps, lr 0.03). Here we give d=32 more steps and a gentler
learning rate and look at whether the Poincare retrieval score (MAP) recovers. If it does, PoC-A's high-dimension
numbers must not be read as 'hyperbolic stops helping' - only as 'not tuned'.

Run: python research/yggdrasil_poc/poc_a3_optimisation_sensitivity.py   (CPU, ~6 minutes, < 200 MB)
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import poc_a_tree_embedding as A

torch.set_num_threads(2)


def train(kind, d, seed, tree_dist, steps, lr):
    n = tree_dist.shape[0]
    model = A.Embedding(kind, n, d, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    g = torch.Generator().manual_seed(1000 + seed)
    for _ in range(steps):
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
        sched.step()
    return model


def main():
    n, adj, tree_dist = A.build_tree()
    t0 = time.time()
    print(f"{'geometry':9s} {'d':>3s} {'steps':>6s} {'lr':>6s} | {'MAP':>6s} {'distortion':>10s}   (seed 0, cosine-decayed lr)")
    for kind, d in [("poincare", 32), ("euclid", 32), ("product", 32)]:
        for steps, lr in [(600, 0.03), (600, 0.01), (2000, 0.01), (2000, 0.003)]:
            m = train(kind, d, 0, tree_dist, steps, lr)
            dist, mp = A.evaluate(m, tree_dist, adj)
            print(f"{kind:9s} {d:3d} {steps:6d} {lr:6.3f} | {mp:6.3f} {dist:10.3f}   [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
