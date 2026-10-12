import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from growmoe.config import GrowthCfg
from growmoe.experts import MLPExpert
from growmoe.growth import GrowthPolicy

def units(n, d, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, d))
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True)).float()

def make_compositional_task(sd, d=16, c=4, n_tr=1000, n_va=300):
    rng = np.random.default_rng(sd)
    x_tr = units(n_tr, d, sd * 10 + 1)
    x_va = units(n_va, d, sd * 10 + 2)
    
    # Hierarchical feature composition:
    # Level 1: quadratic interactions
    # Level 2: combination of quadratic interactions
    def ground_truth(x):
        # x is (B, d)
        z1 = torch.relu(x[:, :4] @ x[:, :4].T) # or pairwise
        # Simple composition:
        f1 = (x[:, 0] * x[:, 1] - x[:, 2] * x[:, 3])
        f2 = (x[:, 4] * x[:, 5] - x[:, 6] * x[:, 7])
        f3 = (x[:, 8] * x[:, 9] - x[:, 10] * x[:, 11])
        # Second layer composition:
        c1 = torch.sin(4.0 * f1) * torch.cos(4.0 * f2)
        c2 = torch.sin(4.0 * f2) * torch.cos(4.0 * f3)
        logits = torch.stack([c1 + c2, c1 - c2, -c1 + c2, -c1 - c2], dim=1)
        return logits.argmax(dim=1)
        
    y_tr = ground_truth(x_tr)
    y_va = ground_truth(x_va)
    return x_tr, y_tr, x_va, y_va

def test_depth_on_compositional(sd=100):
    x_tr, y_tr, x_va, y_va = make_compositional_task(sd)
    d, c = 16, 4
    rng = np.random.default_rng(sd)
    print(f"\n--- Compositional Task: Seed {sd} ---")
    for depth in [1, 2, 3]:
        s = MLPExpert(d_in=d, hidden=32, depth=depth, n_classes=c, seed=sd * 10 + 6)
        s.mark_seen(torch.arange(c))
        opt = torch.optim.Adam(s.parameters(), lr=3e-3)
        for _ in range(500):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
            s.train_step(x_tr[idx], y_tr[idx], opt)
        with torch.no_grad():
            loss = F.cross_entropy(s(x_va), y_va).item()
            acc = (s(x_va).argmax(1) == y_va).float().mean().item()
            print(f"Student depth {depth} (500 steps): val loss = {loss:.4f}, acc = {acc:.3f}")

if __name__ == "__main__":
    for s in [100, 101, 102]:
        test_depth_on_compositional(s)
