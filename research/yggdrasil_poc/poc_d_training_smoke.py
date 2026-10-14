"""PoC-D: does a transformer with Yggdrasil heads TRAIN? (smoke test only - not a claim that the geometry helps)

Task: induction copy. A sequence is  [a1..a8, SEP, a1..a8]; the model must predict the second copy (needs attention to find the matching position).
Arms: euclid heads (plain SDPA) vs Ygg heads (hyperbolic + torus phase + salience bias, learned per-head curvature / temperature / phase strength),
same sizes. Checks: finite loss/gradients, curvature and temperature receive gradient, final accuracy, and the geometry operating point u.
Run: python research/yggdrasil_poc/poc_d_training_smoke.py     (GPU or CPU, < 300 MB, ~1 minute)
"""

import math
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.attention import clip_radius, safe_radius, yggdrasil_attention

DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, SEP, L = 33, 32, 8                          # 32 symbols + SEP
D, H, DH, M, LAYERS = 64, 4, 16, 4, 2


def inv_softplus(x):
    return math.log(math.expm1(x))


class Attn(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.qkv = nn.Linear(D, 3 * H * DH, bias=False)
        self.out = nn.Linear(H * DH, D, bias=False)
        if geometry == "ygg":
            self.phase = nn.Linear(D, 2 * H * M, bias=False)
            self.bias = nn.Linear(D, H, bias=False)
            self.g_c = nn.Parameter(torch.full((H,), inv_softplus(1.0)))
            self.g_tau = nn.Parameter(torch.full((H,), inv_softplus(1.0)))
            self.g_kappa = nn.Parameter(torch.full((H,), inv_softplus(1.0)))

    def forward(self, x, stats=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, H, DH).permute(2, 0, 3, 1, 4)           # (B,H,T,DH)
        if self.geometry == "euclid":
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            c = (F.softplus(self.g_c) + 1e-3).view(1, H, 1, 1)
            tau = (F.softplus(self.g_tau) + 1e-3).view(1, H, 1, 1)
            kappa = F.softplus(self.g_kappa).view(1, H, 1, 1)
            ph = self.phase(x).view(B, T, 2, H, M).permute(2, 0, 3, 1, 4)           # (2,B,H,T,M)
            beta = self.bias(x).permute(0, 2, 1)                                      # (B,H,T)
            q, k = clip_radius(q, safe_radius(torch.float32)), clip_radius(k, safe_radius(torch.float32))
            if stats is not None:
                sqc = torch.sqrt(c).view(1, H, 1)
                stats.append({"c": c.flatten().tolist(), "u_q": (sqc * torch.asinh(sqc * q.norm(dim=-1))).flatten(), "u_k": (sqc * torch.asinh(sqc * k.norm(dim=-1))).flatten()})
            o = yggdrasil_attention(q, k, v, c=c, tau=tau, theta_q=ph[0], theta_k=ph[1], kappa=kappa, key_bias=beta, is_causal=True)
        return self.out(o.permute(0, 2, 1, 3).reshape(B, T, H * DH))


class Block(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.n1, self.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.attn = Attn(geometry)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, stats=None):
        x = x + self.attn(self.n1(x), stats)
        return x + self.mlp(self.n2(x))


class Net(nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(2 * L + 1, D)
        self.blocks = nn.ModuleList(Block(geometry) for _ in range(LAYERS))
        self.norm = nn.RMSNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def forward(self, tok, stats=None):
        x = self.emb(tok) + self.pos(torch.arange(tok.shape[1], device=tok.device))
        for b in self.blocks:
            x = b(x, stats)
        return self.head(self.norm(x))


def batch(n, g):
    a = torch.randint(0, 32, (n, L), generator=g, device=DEV)
    return torch.cat([a, torch.full((n, 1), SEP, device=DEV), a], 1)


def run(geometry, steps=400, seed=0):
    torch.manual_seed(seed)
    net = Net(geometry).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=0.01)
    g = torch.Generator(device=DEV).manual_seed(seed)
    t0, nan_seen = time.time(), False
    for step in range(steps):
        seq = batch(128, g)
        with torch.autocast(DEV, dtype=torch.bfloat16, enabled=DEV == "cuda"):
            logits = net(seq[:, :-1])
        loss = F.cross_entropy(logits[:, L:].float().reshape(-1, V), seq[:, L + 1:].reshape(-1))        # only the copied half is scored
        opt.zero_grad()
        loss.backward()
        if not all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None):
            nan_seen = True
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
    net.eval()
    with torch.no_grad():
        seq = batch(1000, torch.Generator(device=DEV).manual_seed(999))
        stats = []
        acc = (net(seq[:, :-1], stats)[:, L:].argmax(-1) == seq[:, L + 1:]).float().mean().item()
    return net, loss.item(), acc, nan_seen, stats, time.time() - t0


if __name__ == "__main__":
    for geometry in ("euclid", "ygg"):
        net, loss, acc, nan_seen, stats, secs = run(geometry)
        line = f"{geometry:7s} final loss {loss:.3f}  copy accuracy {100 * acc:5.1f} %  non-finite grads: {nan_seen}  ({secs:.0f}s)"
        print(line)
        if geometry == "ygg":
            for i, s in enumerate(stats):
                uq, uk = s["u_q"], s["u_k"]
                print(f"  layer {i}: curvature c per head {[round(x, 3) for x in s['c']]};  u(q) median {uq.median():.2f} p95 {uq.quantile(0.95):.2f} max {uq.max():.2f};"
                      f"  u(k) median {uk.median():.2f} p95 {uk.quantile(0.95):.2f}")
            a = net.blocks[0].attn
            print(f"  gradient reached curvature/temperature/phase-strength params: "
                  f"{all(p.grad is not None and p.grad.abs().sum() > 0 for p in (a.g_c, a.g_tau, a.g_kappa))}")
