import sys
sys.path.insert(0, ".")
import numpy as np
import torch
import torch.nn.functional as F
from growmoe.experts import MLPExpert

def units(n, d, seed):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, d))
    return torch.from_numpy(x / np.linalg.norm(x, axis=1, keepdims=True)).float()

def test_scaled_teacher(scale=3.0, sd=100):
    d, c = 16, 4
    rng = np.random.default_rng(sd)
    x_tr = units(3000, d, sd * 10 + 1)
    x_va = units(500, d, sd * 10 + 2)

    # Teacher with stronger non-linearity (blocks have larger output weights)
    t = MLPExpert(d_in=d, hidden=16, depth=3, n_classes=c, seed=sd * 10 + 3)
    with torch.no_grad():
        for b in t.blocks:
            b.w2.mul_(scale)
    t.mark_seen(torch.arange(c))
    
    with torch.no_grad():
        y_tr = t(x_tr).argmax(dim=1)
        y_va = t(x_va).argmax(dim=1)
        
    print(f"\n--- Teacher scale={scale}, seed={sd} ---")
    for s_d in [1, 2, 3]:
        s = MLPExpert(d_in=d, hidden=16, depth=s_d, n_classes=c, seed=sd * 10 + 4)
        s.mark_seen(torch.arange(c))
        opt = torch.optim.Adam(s.parameters(), lr=3e-3)
        for _ in range(800):
            idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
            s.train_step(x_tr[idx], y_tr[idx], opt)
        with torch.no_grad():
            loss = F.cross_entropy(s(x_va), y_va).item()
            acc = (s(x_va).argmax(1) == y_va).float().mean().item()
            print(f"Student d={s_d} (800 steps): val loss = {loss:.4f}, acc = {acc:.3f}")

if __name__ == "__main__":
    for sc in [1.0, 2.0, 4.0]:
        test_scaled_teacher(sc, 100)
