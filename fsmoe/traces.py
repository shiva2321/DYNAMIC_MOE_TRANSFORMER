"""Layer T: traces. Sequences with REAL next states, to test the result-predicting head and depth growth where the question is meaningful.

A world has a state s (16 numbers) and instruction regimes: each regime is a function s -> s' (shift, rotation, scaling, permutation, a nonlinear mix). Every regime sees the SAME state distribution,
so what an instruction is can only be read from what it does. A "surface" is a style: nuisance features z (16 numbers) whose mean differs per surface and that carry no information about the
function. An input is x = [s, z], its next state x' = [f(s) + noise, z'] with z' redrawn in the same surface. Regimes taught: 6 (surfaces 0 and 1 only). Withheld: 4 near-twins (the same kind of map
with other constants). Held-out surfaces: 2 and 3. So "known regime on a new surface" looks different (z moved) but acts the same, and "withheld regime" looks the same (s identical) but acts differently.
Recognizers compared at one calibration (conformal, n = 40 known inputs, q = 0.05): look (Mahalanobis distance of x to the nearest taught regime), result (prediction error of the nearest taught regime's
expert on the next state, in units of its own teaching error), and the cascade (the look shortlists one expert, the result judges it).
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from growmoe.experts import MLPExpert

from .bed import ROOT
from .detectors import Maha, auroc, conformal_cut, effective_rate

M = 16
OUT = ROOT / "results" / "fsmoe" / "T"
TAUGHT = 6
WITHHELD = 4
N_HELD = 2                                      # held-out surfaces; the teaching surfaces are 0 .. n_teach - 1 (2 = few, 6 = diverse) and the held-out ones come after them


def _orth(rng: np.random.Generator) -> torch.Tensor:
    q, _ = np.linalg.qr(rng.standard_normal((M, M)))
    return torch.from_numpy(q.astype(np.float32))


class World:
    def __init__(self, seed: int, n_teach_surf: int = 6, z_scale: float = 1.0) -> None:
        rng = np.random.default_rng([seed, 0x574F524C])
        self.n_teach_surf = n_teach_surf
        kinds = ["shift", "rot", "scale", "perm", "mix", "shift2"]
        self.fns, self.names = [], []
        for i in range(TAUGHT + WITHHELD):
            kind = kinds[i % 6] if i < TAUGHT else ["shift", "rot", "scale", "mix"][i - TAUGHT]
            if kind in ("shift", "shift2"):
                c = torch.from_numpy(rng.standard_normal(M).astype(np.float32)) * 1.2
                self.fns.append(lambda s, c=c: s + c)
            elif kind == "rot":
                Q = _orth(rng)
                self.fns.append(lambda s, Q=Q: s @ Q.T)
            elif kind == "scale":
                g = torch.from_numpy(np.exp(rng.standard_normal(M) * 0.4).astype(np.float32))
                self.fns.append(lambda s, g=g: s * g)
            elif kind == "perm":
                p = torch.from_numpy(rng.permutation(M))
                self.fns.append(lambda s, p=p: s[:, p])
            else:
                W = torch.from_numpy((rng.standard_normal((M, M)) / np.sqrt(M)).astype(np.float32))
                self.fns.append(lambda s, W=W: torch.tanh(s @ W.T) * 1.5)
            self.names.append(f"{kind}{i}")
        self.mean_z = torch.from_numpy(rng.standard_normal((n_teach_surf + N_HELD, M)).astype(np.float32)) * z_scale
        self.rng = rng

    def sample(self, regime: int, surface: int, n: int, noise: float, far: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        g = torch.Generator().manual_seed(int(self.rng.integers(0, 2 ** 31 - 1)))
        s = torch.randn(n, M, generator=g) * (3.0 if far else 1.0)
        z = self.mean_z[surface] + 0.3 * torch.randn(n, M, generator=g)
        z2 = self.mean_z[surface] + 0.3 * torch.randn(n, M, generator=g)
        s2 = self.fns[regime](s) + noise * torch.randn(n, M, generator=g)
        return torch.cat([s, z], 1), torch.cat([s2, z2], 1)


def train_expert(x: torch.Tensor, x2: torch.Tensor, seed: int, hidden: int = 64, depth: int = 1, epochs: int = 60) -> MLPExpert:
    e = MLPExpert(2 * M, hidden, depth, M, seed=seed)
    e.mark_seen(torch.arange(M))
    opt = torch.optim.Adam(e.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=g)
        for lo in range(0, len(x), 128):
            idx = order[lo:lo + 128]
            loss = F.mse_loss(e(x[idx]), x2[idx][:, :M])
            opt.zero_grad()
            loss.backward()
            opt.step()
    return e.eval()


@torch.no_grad()
def err(e: MLPExpert, x: torch.Tensor, x2: torch.Tensor) -> np.ndarray:
    return ((e(x) - x2[:, :M]) ** 2).mean(1).numpy()


def recognise(seed: int, noise: float, n_teach_surf: int = 6) -> dict:
    w = World(seed, n_teach_surf)
    teach_s = tuple(range(n_teach_surf))
    held_s = tuple(range(n_teach_surf, n_teach_surf + N_HELD))
    experts, floor = [], []
    X_t, y_t = [], []
    for r in range(TAUGHT):
        xs, x2s = zip(*[w.sample(r, s, 6000 // n_teach_surf if n_teach_surf > 2 else 3000, noise) for s in teach_s])
        x, x2 = torch.cat(xs), torch.cat(x2s)
        e = train_expert(x, x2, seed * 10 + r)
        experts.append(e)
        floor.append(float(np.median(err(e, x, x2))) + 1e-6)
        X_t.append(x)
        y_t.append(torch.full((len(x),), r))
    X_t, y_t = torch.cat(X_t), torch.cat(y_t)
    look = Maha().fit(X_t, y_t)

    def scores(x: torch.Tensor, x2: torch.Tensor) -> Dict[str, np.ndarray]:
        d = look.score(x)
        nearest = look.head.proba(x).argmax(1).numpy()                                  # the closest taught regime by look (class-conditional Gaussian)
        res_near = np.asarray([err(experts[int(nearest[i])], x[i:i + 1], x2[i:i + 1])[0] / floor[int(nearest[i])] for i in range(len(x))])
        res_min = np.min(np.stack([err(experts[r], x, x2) / floor[r] for r in range(TAUGHT)]), axis=0)
        return {"look": d, "result_nearest": res_near, "result_all": res_min}

    sets = {}
    for tag, regimes, surfaces, far in (("known_tw", range(TAUGHT), teach_s, False), ("known_ho", range(TAUGHT), held_s, False),
                                        ("withheld_tw", range(TAUGHT, TAUGHT + WITHHELD), teach_s, False), ("withheld_ho", range(TAUGHT, TAUGHT + WITHHELD), held_s, False),
                                        ("far", range(TAUGHT), teach_s, True)):
        parts = [w.sample(r, s, 100, noise, far) for r in regimes for s in surfaces]
        sets[tag] = scores(torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts]))
    cal = {k: np.concatenate([sets["known_tw"][k], sets["known_ho"][k]]) for k in ("look", "result_nearest", "result_all")}
    rng = np.random.default_rng([seed, 7])
    out: dict = {"noise": noise, "seed": seed, "n_teach_surf": n_teach_surf, "effective_rate": effective_rate(40, 0.05), "signals": {}}
    for s in ("look", "result_nearest", "result_all"):
        draws = [rng.choice(len(cal[s]), 40, replace=False) for _ in range(30)]
        rates = {k: float(np.mean([(v[s] > conformal_cut(cal[s][d], 0.05)).mean() for d in draws])) for k, v in sets.items()}
        out["signals"][s] = {"flag": rates, "auroc_known_ho_vs_withheld": auroc(sets["known_ho"][s], np.concatenate([sets["withheld_tw"][s], sets["withheld_ho"][s]])),
                             "auroc_known_ho_vs_far": auroc(sets["known_ho"][s], sets["far"][s])}
    hidden = 64
    out["cost_macs"] = {"look": int(look.macs), "result_nearest": int(look.macs + 2 * 2 * M * hidden + M * M), "result_all": int(TAUGHT * (2 * 2 * M * hidden + M * M))}
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- depth
def compose_fn(w: World, a: int, b: int, c: int):
    return lambda s: w.fns[c](w.fns[b](w.fns[a](s)))


def grow_when_stuck(x: torch.Tensor, x2: torch.Tensor, seed: int, hidden0: int = 8, max_hidden: int = 128, rounds: int = 14, steps: int = 150, eps: float = 0.03, poor_frac: float = 0.2) -> dict:
    """Start small. After every round of training: if the loss is still improving, keep training (not stuck); if it has plateaued (relative improvement < eps) AND the fit is still poor (loss above
    a fifth of the target variance), grow function-preservingly: widen x2 until ``max_hidden`` or until a widening has gained less than eps, then deepen. A growth is kept only if the next round
    improves the loss by more than eps (the growth is function-preserving, so the loss before it is the control; the plateau makes extra steps alone worthless)."""
    e = MLPExpert(2 * M, hidden0, 1, M, seed=seed)
    e.mark_seen(torch.arange(M))
    target_var = float(x2[:, :M].var())
    g = torch.Generator().manual_seed(seed)

    def fit_round(opt) -> float:
        loss = 0.0
        for _ in range(steps):
            idx = torch.randint(0, len(x), (128,), generator=g)
            l = F.mse_loss(e(x[idx]), x2[idx][:, :M])
            opt.zero_grad()
            l.backward()
            opt.step()
            loss = float(l)
        with torch.no_grad():
            return float(F.mse_loss(e(x), x2[:, :M]))

    opt = torch.optim.Adam(e.parameters(), lr=3e-3)
    hist, events, prev, widen_ok, last_kind, last_before = [], [], None, True, None, None
    for r in range(rounds):
        cur = fit_round(opt)
        hist.append(cur)
        if last_kind is not None and last_before is not None:
            gain = (last_before - cur) / max(last_before, 1e-9)
            events.append({"round": r, "kind": last_kind, "gain_after": gain})
            if last_kind == "widen" and gain < eps:
                widen_ok = False
            last_kind = None
        stuck = prev is not None and (prev - cur) / max(prev, 1e-9) < eps
        poor = cur > poor_frac * target_var
        if stuck and poor and r < rounds - 1:
            if widen_ok and e.hidden < max_hidden:
                e.widen(e.hidden)
                last_kind = "widen"
            elif len(e.blocks) < 4:
                e.deepen()
                last_kind = "deepen"
            else:
                prev = cur
                continue
            opt = torch.optim.Adam(e.parameters(), lr=3e-3)
            last_before = cur
            prev = None
            continue
        prev = cur
    return {"final_mse": hist[-1], "target_var": target_var, "hidden": e.hidden, "depth": len(e.blocks), "params": e.num_params(), "hist": hist, "events": events}


def fixed_net(x: torch.Tensor, x2: torch.Tensor, seed: int, hidden: int, depth: int, epochs: int = 40) -> dict:
    e = train_expert(x, x2, seed, hidden, depth, epochs)
    with torch.no_grad():
        return {"final_mse": float(F.mse_loss(e(x), x2[:, :M])), "hidden": hidden, "depth": depth, "params": e.num_params()}


def depth_lab(seed: int, noise: float = 0.05) -> dict:
    w = World(seed)
    out = {"seed": seed, "cases": {}}
    inter = lambda s: torch.tanh(3.0 * s * torch.roll(s, 1, dims=1))                   # pairwise interactions: needs hidden units and, composed, depth
    cases = {"simple_shift": lambda s: w.fns[0](s), "simple_mix": lambda s: w.fns[4](s),
             "composed_3": compose_fn(w, 4, 1, 4), "composed_5": lambda s: w.fns[4](w.fns[1](w.fns[4](w.fns[1](w.fns[4](s))))),
             "interact_1": inter, "interact_3": lambda s: inter(inter(inter(s)))}      # the last two were added after the first run showed the first four never need growth
    for name, f in cases.items():
        g = torch.Generator().manual_seed(seed + 5)
        s = torch.randn(3000, M, generator=g)
        z = torch.randn(3000, M, generator=g) * 0.3
        x = torch.cat([s, z], 1)
        x2 = torch.cat([f(s) + noise * torch.randn(3000, M, generator=g), z], 1)
        grown = grow_when_stuck(x, x2, seed)
        grown05 = grow_when_stuck(x, x2, seed, poor_frac=0.05)
        out["cases"][name] = {"grown": {k: v for k, v in grown.items() if k != "hist"}, "grown_hist": grown["hist"], "grown_05": {k: v for k, v in grown05.items() if k != "hist"},
                              "fixed": [fixed_net(x, x2, seed, h, d) for h, d in ((8, 1), (64, 1), (128, 1), (32, 2), (32, 3))]}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["recognise", "depth"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        if a.cmd == "recognise":
            for div in (2, 6):
                for noise in (0.05, 0.3, 1.0):
                    path = OUT / f"T_rec_s{seed}_div{div}_n{noise}.json"
                    if not path.exists():
                        path.write_text(json.dumps(recognise(seed, noise, div)), encoding="utf-8")
                        print("done", path.name, flush=True)
        else:
            path = OUT / f"T_depth_s{seed}.json"
            if not path.exists():
                path.write_text(json.dumps(depth_lab(seed)), encoding="utf-8")
                print("done", path.name, flush=True)
