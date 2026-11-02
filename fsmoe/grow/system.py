"""The growth lab's system: a pool of experts that routes by direction on the sphere, learns by test-then-train, and can be grown by a governor (``governor.py``).

An expert is a closed-form notebook (``unimoe.heads.LdaHead``: class sums and one pooled scatter, exact, never forgets) plus, once it has one, a small residual network (``growmoe.experts.MLPExpert``);
its answer is the 50/50 mixture of the two. Newborns can be notebook-only (``net_at_birth=False``): they answer from the first sample and get a network only when a held-out test says it helps.
Routing: every expert keeps the direction of its samples (a centroid) and the cosine below which a sample is "not mine" (a low quantile of its own cosines); a sample goes to the accepting expert with the
highest cosine; a sample nobody accepts waits in a room and becomes a new expert when the room holds a tight group that no existing expert covers. An optional tier groups the experts (k-means over
their centroids) and routes in two steps (best groups, then the experts in them; a sample nobody in them accepts gets a full scan), which is cheaper when there are many experts.
Cost is counted in multiply-adds: the gate, the expert and its notebook per answered sample; training steps and growth trials in total. Nothing here sees a regime id except the bookkeeping counters
(``Expert.regimes``) that only scoring reads.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from growmoe.experts import MLPExpert
from unimoe.heads import LdaHead


@dataclass
class SysCfg:
    d: int = 32
    k: int = 6
    h0: int = 8                       # units of a new network
    depth0: int = 1
    net_at_birth: bool = True         # False: a newborn is a notebook until a held-out test says a network helps
    steps: int = 4                    # optimiser steps per routed batch
    lr: float = 3e-3
    batch_mix: int = 64               # samples drawn from the expert's own buffer and mixed into every step
    train_frac: float = 0.85          # the rest of an expert's samples is held out and never trained on
    buf_cap: int = 6000
    hold_cap: int = 800
    birth_steps: int = 60             # steps on the founding samples when a network is born with the expert
    room_min: int = 64
    room_cap: int = 600
    room_pair: float = 0.2            # two samples of one regime have a cosine around 0.3, two of different regimes around 0: this separates them
    room_member: float = 0.4          # cosine to the group's centroid that makes a sample a member
    absorb_cos: float = 0.25          # a group whose median cosine to an existing expert's centroid reaches this is that expert's own outliers (a new regime sits near 0.1): absorbed, not founded
    accept_q: float = 0.005
    accept_floor: float = 0.38
    accept_default: float = 0.3
    gate_budget: float = float("inf")  # flat gate multiply-adds per sample above which a tier may be built (read by the governor)
    seed: int = 0


class Reservoir:
    """Uniform sample of everything added, up to ``cap`` rows (preallocated)."""

    def __init__(self, d: int, cap: int) -> None:
        self._x, self._y, self.cap, self.seen, self.n = torch.zeros(cap, d), torch.zeros(cap, dtype=torch.long), cap, 0, 0

    @property
    def x(self) -> torch.Tensor:
        return self._x[: self.n]

    @property
    def y(self) -> torch.Tensor:
        return self._y[: self.n]

    def add(self, x: torch.Tensor, y: torch.Tensor, rng: np.random.Generator) -> None:
        for i in range(x.shape[0]):
            self.seen += 1
            if self.n < self.cap:
                self._x[self.n], self._y[self.n] = x[i], y[i]
                self.n += 1
            else:
                j = int(rng.integers(0, self.seen))
                if j < self.cap:
                    self._x[j], self._y[j] = x[i], y[i]

    def __len__(self) -> int:
        return self.n


class Expert:
    def __init__(self, eid: int, cfg: SysCfg) -> None:
        self.id, self.cfg = eid, cfg
        self.csum, self.n = torch.zeros(cfg.d), 0
        self.nb = LdaHead(cfg.d, cfg.k, 0.1)
        self.net: Optional[MLPExpert] = None
        self.opt = None
        self.train = Reservoir(cfg.d, cfg.buf_cap)
        self.hold = Reservoir(cfg.d, cfg.hold_cap)
        self.cos: Deque[float] = deque(maxlen=400)
        self.age = 0                                   # batches trained
        self.state = "active" if cfg.net_at_birth else "newborn"     # newborn | notebook | active
        self.regimes: Counter = Counter()              # scoring only
        self.policy = None                             # the growth policy (governor)
        self.guard = None                              # what is kept to undo a growth (governor)
        self.next_decision = 0                         # train-buffer size at which a notebook-only expert is judged again

    # ---- geometry
    @property
    def centroid(self) -> torch.Tensor:
        return F.normalize(self.csum, dim=0)

    def threshold(self, cfg: SysCfg) -> float:
        if len(self.cos) < 40:
            return cfg.accept_default
        return max(float(np.quantile(np.asarray(self.cos), cfg.accept_q)), cfg.accept_floor)

    # ---- answering
    def proba(self, x: torch.Tensor) -> torch.Tensor:
        p_nb = self.nb.proba(x) if self.nb.total > 0 else torch.full((x.shape[0], self.cfg.k), 1.0 / self.cfg.k)
        if self.net is None:
            return p_nb
        self.net.eval()
        with torch.no_grad():
            p_net = F.softmax(self.net(x), dim=-1)
        return 0.5 * p_net + 0.5 * p_nb

    def macs(self) -> float:
        """Multiply-adds to answer one sample: the network (its parameters, a proxy) and the notebook (one class score per class)."""
        return float((self.net.num_params() if self.net is not None else 0) + self.cfg.k * self.cfg.d)

    def params(self) -> Tuple[int, int]:
        """(network parameters, notebook and centroid scalars)."""
        return (self.net.num_params() if self.net is not None else 0), self.nb.n_scalars() + self.cfg.d

    def size(self) -> Tuple[int, int]:
        return (0, 0) if self.net is None else (self.net.hidden, len(self.net.blocks))

    def new_net(self, hidden: int, depth: int, seed: int) -> None:
        self.net = MLPExpert(self.cfg.d, hidden, depth, self.cfg.k, seed=seed)
        self.rebuild_opt()

    def rebuild_opt(self) -> None:
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)


@dataclass
class Tier:
    groups: List[List[int]]            # expert ids per group
    centres: torch.Tensor              # (G, d) unit group directions
    top_g: int
    built_for: int                     # number of experts when it was built


class GrowSystem:
    def __init__(self, cfg: SysCfg, governor=None) -> None:
        self.cfg, self.governor = cfg, governor
        self.rng = np.random.default_rng([cfg.seed, 0x53595354])
        self.experts: Dict[int, Expert] = {}
        self.next_id = 0
        self.room_x: List[torch.Tensor] = []
        self.room_y: List[torch.Tensor] = []
        self.room_reg: List[int] = []
        self.tier: Optional[Tier] = None
        self.batches = 0
        self.macs_train = 0.0              # optimiser steps on routed batches, births and promotions
        self.macs_trial = 0.0              # growth trials and promotion tests (the cost of governing)
        self.macs_trial_grow = 0.0         # the growth trials alone (the governor's budget is on these)
        self.events: List[dict] = []
        self.dirty = True
        self._cache: Optional[Tuple[List[int], torch.Tensor, torch.Tensor]] = None

    # ---- bookkeeping
    def log(self, kind: str, **kw) -> None:
        self.events.append({"batch": self.batches, "kind": kind, **kw})

    def _arrays(self) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        if self._cache is None or self.dirty:
            ids = sorted(self.experts)
            C = torch.stack([self.experts[i].centroid for i in ids]) if ids else torch.zeros(0, self.cfg.d)
            T = torch.tensor([self.experts[i].threshold(self.cfg) for i in ids]) if ids else torch.zeros(0)
            self._cache, self.dirty = (ids, C, T), False
        return self._cache

    # ---- routing
    def route(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """(expert id per sample or -1 when nobody accepts, nearest expert id per sample, gate multiply-adds per sample)."""
        ids, C, T = self._arrays()
        n, d = x.shape[0], self.cfg.d
        if not ids:
            return torch.full((n,), -1, dtype=torch.long), torch.full((n,), -1, dtype=torch.long), 0.0
        idt = torch.tensor(ids)
        cos = cos_flat = x @ C.T
        gate = float(len(ids) * d)
        if self.tier is not None:
            member = torch.zeros(len(self.tier.groups), len(ids), dtype=torch.bool)
            for g, grp in enumerate(self.tier.groups):
                for e in grp:
                    if e in self.experts:
                        member[g, ids.index(e)] = True
            top = (x @ self.tier.centres.T).topk(min(self.tier.top_g, len(self.tier.groups)), dim=1).indices
            cos = cos_flat.masked_fill(~member[top].any(1), -2.0)
            fall = ~(cos >= T).any(1)                                   # nobody in the best groups accepts: a full scan (the "second look"), paid for
            cos = torch.where(fall[:, None], cos_flat, cos)
            gate = float(len(self.tier.groups) * d + self.tier.top_g * float(member.sum(1).float().mean()) * d + float(fall.float().mean()) * len(ids) * d)
        accept = cos >= T
        best = cos.masked_fill(~accept, -2.0).argmax(1)
        routed = torch.where(accept.any(1), idt[best], torch.full((n,), -1, dtype=torch.long))
        return routed, idt[cos.argmax(1)], gate

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """(class probabilities, the expert that answered, multiply-adds per sample averaged over the batch). A sample nobody accepts is answered by the nearest expert."""
        routed, nearest, gate = self.route(x)
        n = x.shape[0]
        probs = torch.full((n, self.cfg.k), 1.0 / self.cfg.k)
        who = torch.where(routed >= 0, routed, nearest)
        macs = torch.zeros(n)
        for e in who.unique().tolist():
            if e < 0:
                continue
            m = who == e
            probs[m] = self.experts[e].proba(x[m])
            macs[m] = self.experts[e].macs()
        return probs, who, gate + float(macs.mean())

    # ---- learning
    def observe(self, x: torch.Tensor, y: torch.Tensor, regimes: Optional[torch.Tensor] = None) -> None:
        """One labelled batch: route it, train the experts that own it, wait-room what nobody owns, found experts from the room, let the governor look."""
        x = F.normalize(x.float(), dim=1)
        routed, _, _ = self.route(x)
        for e in routed.unique().tolist():
            if e < 0:
                continue
            m = routed == e
            self._learn(self.experts[e], x[m], y[m], None if regimes is None else regimes[m])
        nov = routed < 0
        if bool(nov.any()):
            self.room_x.append(x[nov])
            self.room_y.append(y[nov])
            self.room_reg += (regimes[nov].tolist() if regimes is not None else [-1] * int(nov.sum()))
            self._room_step()
        self.batches += 1
        if self.governor is not None:
            self.governor.step(self)

    def _learn(self, e: Expert, x: torch.Tensor, y: torch.Tensor, reg: Optional[torch.Tensor]) -> None:
        cfg = self.cfg
        e.csum += x.sum(0)
        e.n += x.shape[0]
        e.cos.extend((x @ e.centroid).tolist())
        if reg is not None:
            e.regimes.update(reg.tolist())
        keep = torch.from_numpy(self.rng.random(x.shape[0]) < cfg.train_frac)
        xt, yt = x[keep], y[keep]
        e.hold.add(x[~keep], y[~keep], self.rng)
        if xt.shape[0]:
            e.nb.update(xt, yt)
            e.train.add(xt, yt, self.rng)
            if e.net is not None:
                self._steps(e, xt, yt, cfg.steps)
        e.age += 1
        self.dirty = True

    def _steps(self, e: Expert, xt: torch.Tensor, yt: torch.Tensor, steps: int) -> None:
        cfg = self.cfg
        for _ in range(steps):
            if len(e.train) > 0:
                idx = torch.from_numpy(self.rng.integers(0, len(e.train), size=min(cfg.batch_mix, len(e.train))))
                xb, yb = torch.cat([xt, e.train.x[idx]]), torch.cat([yt, e.train.y[idx]])
            else:
                xb, yb = xt, yt
            e.net.train_step(xb, yb, e.opt)
            self.macs_train += 3.0 * e.net.num_params() * xb.shape[0]

    # ---- the room and new experts
    def _room_step(self) -> None:
        """Take dense groups out of the room: the sample with the most close neighbours seeds a group; its centroid decides who belongs; a group of at least ``room_min`` either founds an expert or, when
        it sits close to an existing expert, is absorbed by it (share, never found a duplicate). What is left waits."""
        cfg = self.cfg
        X, Y, R = torch.cat(self.room_x), torch.cat(self.room_y), list(self.room_reg)
        if X.shape[0] > cfg.room_cap:                                  # the oldest half goes: a room that never tightens must not grow forever
            cut = X.shape[0] - cfg.room_cap // 2
            X, Y, R = X[cut:], Y[cut:], R[cut:]
        while X.shape[0] >= cfg.room_min:
            S = X @ X.T
            dens = (S >= cfg.room_pair).sum(1)
            i = int(dens.argmax())
            if int(dens[i]) < cfg.room_min:
                break
            c = F.normalize(X[S[i] >= cfg.room_pair].mean(0), dim=0)
            member = (X @ c) >= cfg.room_member
            if int(member.sum()) < cfg.room_min:
                break
            Xm, Ym, Rm = X[member], Y[member], [r for r, m in zip(R, member.tolist()) if m]
            ids, C, _ = self._arrays()
            med = (Xm @ C.T).median(0).values if ids else torch.zeros(0)
            j = int(med.argmax()) if ids else -1
            if j >= 0 and float(med[j]) >= cfg.absorb_cos:
                self._learn(self.experts[ids[j]], Xm, Ym, torch.tensor(Rm))
                self.log("absorb", expert=ids[j], n=int(Xm.shape[0]))
            else:
                self._found(Xm, Ym, Rm)
            X, Y, R = X[~member], Y[~member], [r for r, m in zip(R, member.tolist()) if not m]
        self.room_x, self.room_y, self.room_reg = ([X], [Y], R) if X.shape[0] else ([], [], [])

    def _found(self, X: torch.Tensor, Y: torch.Tensor, reg: List[int]) -> Expert:
        cfg = self.cfg
        e = Expert(self.next_id, cfg)
        self.next_id += 1
        self.experts[e.id] = e
        e.csum, e.n = X.sum(0), int(X.shape[0])
        e.cos.extend((X @ e.centroid).tolist())
        e.regimes.update(reg)
        keep = torch.from_numpy(self.rng.random(X.shape[0]) < cfg.train_frac)
        e.hold.add(X[~keep], Y[~keep], self.rng)
        e.train.add(X[keep], Y[keep], self.rng)
        e.nb.update(X[keep], Y[keep])
        if cfg.net_at_birth:
            e.new_net(cfg.h0, cfg.depth0, seed=cfg.seed * 1000 + e.id)
            self._steps(e, X[keep], Y[keep], cfg.birth_steps)
        e.age = 1
        self.dirty = True
        self.log("spawn", expert=e.id, n=int(X.shape[0]), notebook_only=not cfg.net_at_birth)
        return e

    # ---- accounting
    def params(self) -> Dict[str, int]:
        net = sum(e.params()[0] for e in self.experts.values())
        nb = sum(e.params()[1] for e in self.experts.values())
        return {"net": net, "notebook": nb, "total": net + nb}

    def summary(self) -> dict:
        return {"experts": len(self.experts), "states": dict(Counter(e.state for e in self.experts.values())), "sizes": {i: e.size() for i, e in sorted(self.experts.items())},
                "params": self.params(), "macs_train": self.macs_train, "macs_trial": self.macs_trial, "macs_trial_grow": self.macs_trial_grow, "tier": None if self.tier is None else {"groups": len(self.tier.groups), "top_g": self.tier.top_g},
                "events": dict(Counter(ev["kind"] for ev in self.events))}


def evaluate(system: GrowSystem, world) -> dict:
    """Accuracy of the routed answer on every regime's held test set, the expert that serves each regime and the cost of answering."""
    acc, serves, macs = {}, {}, []
    for rid, (x, y) in world.test.items():
        probs, who, m = system.predict(F.normalize(x.float(), dim=1))
        acc[rid] = float((probs.argmax(1) == y).float().mean())
        serves[rid] = int(who.mode().values)
        macs.append(m)
    return {"acc": acc, "mean": float(np.mean(list(acc.values()))), "min": float(np.min(list(acc.values()))), "serves": serves, "macs_answer": float(np.mean(macs))}
