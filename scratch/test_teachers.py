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

def test_teachers(sd=100):
    d, c = 16, 4
    rng = np.random.default_rng(sd)
    x_tr = units(2000, d, sd * 10 + 1)
    x_va = units(500, d, sd * 10 + 2)

    # Let's test different teacher types
    # Case A: Standard MLPExpert teacher with depth 3, hidden 16
    t_mlp = MLPExpert(d_in=d, hidden=16, depth=3, n_classes=c, seed=sd * 10 + 3)
    t_mlp.mark_seen(torch.arange(c))
    with torch.no_grad():
        y_tr_a = t_mlp(x_tr).argmax(dim=1)
        y_va_a = t_mlp(x_va).argmax(dim=1)
        
    print(f"--- Seed {sd}: MLPExpert depth-3 teacher ---")
    for s_w in [8, 16]:
        for s_d in [1, 2, 3]:
            s = MLPExpert(d_in=d, hidden=s_w, depth=s_d, n_classes=c, seed=sd * 10 + 4)
            s.mark_seen(torch.arange(c))
            opt = torch.optim.Adam(s.parameters(), lr=3e-3)
            for _ in range(800):
                idx = torch.from_numpy(rng.integers(0, len(x_tr), 64))
                s.train_step(x_tr[idx], y_tr_a[idx], opt)
            with torch.no_grad():
                loss = F.cross_entropy(s(x_va), y_va_a).item()
                acc = (s(x_va).argmax(1) == y_va_a).float().mean().item()
                print(f"Student w={s_w}, d={s_d} (800 steps): val loss = {loss:.4f}, acc = {acc:.3f}")

if __name__ == "__main__":
    for s in [100, 101]:
        test_teachers(s)
