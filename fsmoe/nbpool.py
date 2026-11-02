"""The notebook-first newborn, ported into the real pool (design: docs/superpowers/specs/2026-11-01-fsmoe-nbpool-design.md).

``NbPool`` is ``unimoe.pools.UniPool`` (the pool of growmoe's ``PhyloMoE`` with a closed-form head in every expert) with one switch, ``nb_first``: a newborn expert has NO network. Its answer is its closed-form head
(a ``NotebookExpert``: the pool's own interface, ``proba`` / ``clone`` / ``state_dict`` / ``seen``, over an ``LdaHead``, so the gate, the transfer verdict, the spawn path and the accounting work unchanged).
Every ``nb_every`` observe calls the pool judges each notebook expert that has ``nb_min`` training samples and ``nb_hold_min`` held-out ones (15 % of an expert's samples are held out and never trained on):
candidate networks (``nb_sizes``) are trained on the expert's own replay samples and each is compared, as it would answer (log-linear blend with the notebook, ``lda_blend``), with the notebook alone on the
held-out samples; the notebook stays unless a candidate wins by ``nb_margin``, otherwise the cheapest candidate within ``nb_knee`` of the best becomes the expert's network (``rec.module`` is replaced and its
optimiser rebuilt, exactly as a growth does) and the notebook becomes the expert's ``meta["lda"]`` head. A kept notebook is judged again when its data has doubled.
Kept simple on purpose: it requires ``spawn_rule="transfer"`` and ``growth_rule="gap"`` (the shipped defaults) and turns on growmoe's ``transfer_chance`` rule (a parent that does worse than chance on a cluster cannot vouch for it); notebook experts are left out of merges.
"""

import copy
import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from growmoe.experts import MLPExpert
from growmoe.replay import Reservoir
from unimoe.heads import LdaHead
from unimoe.pools import UniCfg, UniPool, blend
from unimoe.systems2 import uni_cfg

from .grow.governor import pick_size


class NotebookExpert(nn.Module):
    """An expert with no network: forward = the closed-form head's logits. Parameter-free apart from one inert scalar that lets the pool build an optimiser for it."""

    hidden = 0
    blocks: tuple = ()

    def __init__(self, d: int, n_classes: int, shrink: float = 0.1, seed: int = 0) -> None:
        super().__init__()
        self.d_in, self.n_classes, self.shrink, self.seed = d, n_classes, shrink, seed
        self.head = LdaHead(d, n_classes, shrink)
        self._inert = nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def seen(self) -> Tensor:
        return self.head.seen()

    def mark_seen(self, y: Tensor) -> None:                           # the head knows the classes it has met
        return None

    def forward(self, u: Tensor) -> Tensor:
        if self.head.total == 0:
            return torch.zeros(u.shape[0], self.n_classes)
        return self.head.logits(F.normalize(u.float(), dim=-1))

    def proba(self, u: Tensor) -> Tensor:
        return F.softmax(self(u), dim=-1)

    def loss(self, u: Tensor, y: Tensor) -> Tensor:
        return F.cross_entropy(self(u), y.long())

    def train_step(self, u: Tensor, y: Tensor, opt=None, replay=None, project: bool = False) -> float:
        """Absorb the batch (and the replay rows) into the head; returns the loss on the batch before the update. Used by the pool's probes and pretraining; the live expert learns through ``NbPool.train_expert``."""
        y = y.long()
        with torch.no_grad():
            before = float(F.cross_entropy(self(u), y)) if self.head.total else float("nan")
        x_all, y_all = (u, y) if replay is None else (torch.cat([u, replay[0]]), torch.cat([y, replay[1].long()]))
        self.head.update(F.normalize(x_all.float(), dim=-1), y_all)
        return before

    def num_params(self) -> int:
        return 0

    def clone(self) -> "NotebookExpert":
        return copy.deepcopy(self)

    def inherit(self, rho: float = 0.0, sigma: float = 0.0, rng=None) -> "NotebookExpert":
        return NotebookExpert(self.d_in, self.n_classes, self.shrink, self.seed)

    def shrink_perturb(self, lam: float, seed: int = 0) -> "NotebookExpert":
        return NotebookExpert(self.d_in, self.n_classes, self.shrink, seed)

    def state_dict(self, *args, **kwargs) -> Dict[str, Tensor]:
        return {"n": self.head.n.clone(), "s": self.head.s.clone(), "S": self.head.S.clone(), "total": torch.tensor(self.head.total)}

    def load_state_dict(self, sd, strict: bool = True):
        self.head.n, self.head.s, self.head.S, self.head.total = sd["n"].clone(), sd["s"].clone(), sd["S"].clone(), int(sd["total"])
        self.head._fit_cache = None


@dataclass
class NbCfg(UniCfg):
    nb_first: bool = True
    nb_sizes: tuple = ((8, 1), (8, 3), (32, 1), (64, 1))     # (units, blocks) of the candidate networks
    nb_min: int = 200                # training samples (the expert's replay) before the first judgement
    nb_hold_min: int = 60            # held-out samples needed
    nb_margin: float = 0.02          # a candidate must beat the notebook alone by this much accuracy
    nb_knee: float = 0.015           # take the cheapest candidate within this accuracy of the best
    nb_steps: int = 200              # optimiser steps per candidate
    nb_every: int = 10               # observe calls between judgements
    nb_train_frac: float = 0.85      # the rest of a notebook expert's samples is held out
    nb_hold_cap: int = 150

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.nb_first and (self.spawn_rule != "transfer" or self.growth_rule != "gap"):
            raise ValueError("nb_first needs spawn_rule='transfer' and growth_rule='gap' (the shipped defaults)")
        if not 0.0 < self.nb_train_frac < 1.0:
            raise ValueError("nb_train_frac must lie in (0, 1)")


def nb_cfg(n_classes: int, seed: int, recipe: Optional[dict] = None, d: int = 128, **nb) -> NbCfg:
    """The ULDA configuration of ``unimoe.systems2`` (the validated pool with a closed-form head in every expert) with the notebook-first fields on top."""
    base = uni_cfg("ULDA", n_classes, seed, recipe, (), d)
    kw = {f.name: getattr(base, f.name) for f in dataclasses.fields(base)}
    kw["transfer_chance"] = nb.pop("transfer_chance", 1.0)           # the worse-than-chance rule of growmoe's spawn test (E-062): on for the notebook-first pool, as in E-061
    kw.update(nb)
    return NbCfg(**kw)


class NbPool(UniPool):
    name = "A_phylomoe_nbfirst"

    def __init__(self, cfg: NbCfg) -> None:
        super().__init__(cfg)
        self._nb_rng = np.random.default_rng([cfg.seed, 0x4E42])
        self._nb_calls = 0
        self.nb_macs = 0.0                                              # multiply-adds spent on judging notebooks (candidate training)

    # ---- birth
    def add_expert(self, module, parent=None, state="active", meta=None):
        c = self.cfg
        if c.nb_first and not isinstance(module, NotebookExpert):
            module = NotebookExpert(c.d, c.n_classes, c.lda_shrink, seed=c.seed * 1000 + self._next_id)
        rec = super().add_expert(module, parent, state, meta)
        if isinstance(module, NotebookExpert):
            rec.meta["nb"] = {"hold": Reservoir(c.nb_hold_cap, c.d, np.random.default_rng([c.seed, 0x484F4C, rec.id])), "next_at": c.nb_min, "judged": 0}
        return rec

    def _handle_cluster(self, x_taken, y_taken, *args, **kwargs):
        """The shipped spawn path, then the exact bookkeeping of a notebook: a child notebook is primed once with its founding samples (the pretraining steps of the shipped path would count them many times);
        a cluster that joins a notebook parent unchanged enters its head once."""
        before_experts, n_events = set(self.experts), len(self.events)
        out = super()._handle_cluster(x_taken, y_taken, *args, **kwargs)
        valid = y_taken >= 0
        if bool(valid.any()):
            xv, yv = F.normalize(x_taken[valid].float(), dim=1), y_taken[valid].long()
            for eid in set(self.experts) - before_experts:
                mod = self.experts[eid].module
                if isinstance(mod, NotebookExpert):
                    mod.head = LdaHead(self.cfg.d, self.cfg.n_classes, self.cfg.lda_shrink)
                    mod.head.update(xv, yv)
            for ev in self.events.since(n_events):
                if ev.kind == "extend" and ev.fields.get("expert") in self.experts and isinstance(self.experts[ev.fields["expert"]].module, NotebookExpert):
                    self.experts[ev.fields["expert"]].module.head.update(xv, yv)
        return out

    # ---- learning
    def train_expert(self, eid: int, x: Tensor, y: Tensor, replay: bool = True, project: bool = False) -> float:
        rec = self.experts[eid]
        if not isinstance(rec.module, NotebookExpert):
            return super().train_expert(eid, x, y, replay, project)
        c = self.cfg
        keep = torch.from_numpy(self._nb_rng.random(x.shape[0]) < c.nb_train_frac)
        xn = F.normalize(x.float(), dim=1)
        hold = rec.meta["nb"]["hold"]
        if hold is not None:
            hold.add(xn[~keep], y[~keep].long())
        else:
            keep = torch.ones_like(keep)
        with torch.no_grad():
            loss = float(F.cross_entropy(rec.module(xn), y.long())) if rec.module.head.total else float("nan")
        if bool(keep.any()):
            rec.module.head.update(xn[keep], y[keep].long())
            if replay:
                rec.replay.add(xn[keep], y[keep].long())
        rec.n_assigned += x.shape[0]
        rec.last_active = self.step
        return loss

    def _known_batch(self, rec, preq_losses) -> bool:
        """A notebook's training is its closed-form absorb, so there is nothing to skip."""
        return False if isinstance(rec.module, NotebookExpert) else super()._known_batch(rec, preq_losses)

    def _gap_growth_step(self, rec, x, y, pre) -> None:
        if isinstance(rec.module, NotebookExpert):
            return
        return super()._gap_growth_step(rec, x, y, pre)

    def _attempt_merges(self) -> None:
        """Notebook experts sit out the shipped merge routine (it trains clones of networks)."""
        hidden = {e: self.experts.pop(e) for e in [e for e, r in self.experts.items() if isinstance(r.module, NotebookExpert)]}
        try:
            super()._attempt_merges()
        finally:
            self.experts.update(hidden)
            items = sorted(self.experts.items())
            self.experts.clear()
            self.experts.update(items)

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None):
        rec = super().observe(x, y, ctx)
        self._nb_calls += 1
        if self.cfg.nb_first and self._nb_calls % self.cfg.nb_every == 0:
            self._nb_judge_all()
        return rec

    # ---- notebook -> network
    def _nb_judge_all(self) -> None:
        for eid in [e for e, r in self.experts.items() if isinstance(r.module, NotebookExpert)]:
            self._nb_judge(eid)

    def _nb_judge(self, eid: int) -> None:
        c, rec = self.cfg, self.experts[eid]
        nb = rec.meta["nb"]
        hold = nb["hold"]
        if rec.state == "infant" or rec.replay.n < max(c.nb_min, nb["next_at"]) or hold is None or hold.n < c.nb_hold_min:
            return
        x, y = rec.replay.data
        hx, hy = hold.data
        note = rec.module
        p_nb = note.proba(hx)
        acc_nb = float((p_nb.argmax(1) == hy).float().mean())
        n = int(y.shape[0])
        tried = []
        for i, (hidden, depth) in enumerate(c.nb_sizes):
            net = MLPExpert(c.d, hidden, depth, c.n_classes, seed=c.seed * 1000 + eid * 17 + i)
            net.mark_seen(y)
            opt = torch.optim.Adam(net.parameters(), lr=c.lr)
            g = torch.Generator().manual_seed(c.seed * 1000 + eid * 17 + i)
            for _ in range(c.nb_steps):
                idx = torch.randint(0, n, (min(128, n),), generator=g)
                net.train_step(x[idx], y[idx], opt)
            self.nb_macs += 3.0 * net.num_params() * min(128, n) * c.nb_steps
            net.eval()
            with torch.no_grad():
                p_bl = blend(F.softmax(net(hx), dim=-1), p_nb, c.lda_blend)
            tried.append((net.num_params(), float((p_bl.argmax(1) == hy).float().mean()), net, hidden, depth))
        nb["judged"] += 1
        pick = pick_size([(t[0], t[1]) for t in tried], acc_nb, c.nb_margin, c.nb_knee)
        if pick is None:
            nb["next_at"] = 2 * n
            if nb["next_at"] > rec.replay.capacity:                              # its replay is full: it can never be judged again, so the held-out samples are released
                nb["hold"] = None
            self.events.add(kind="nb_keep", step=self.step, expert=eid, notebook_acc=acc_nb, best_blend_acc=max(t[1] for t in tried), n=n)
            return
        params, acc_bl, net, hidden, depth = tried[pick]
        rec.module, rec.meta["lda"] = net, note.head
        rec.opt = self.make_optimizer(net, rec.lr_mult)
        self._forget_competence_history(rec)
        nb["hold"] = None
        self.events.add(kind="nb_promote", step=self.step, expert=eid, notebook_acc=acc_nb, blend_acc=acc_bl, hidden=hidden, depth=depth, params=params, n=n)

    # ---- accounting
    def extra_cost(self, d: int):
        params, macs, scalars = super().extra_cost(d)
        for r in self.experts.values():
            if isinstance(r.module, NotebookExpert):
                scalars += r.module.head.n_scalars()
                macs += d * d // 4
            hold = r.meta.get("nb", {}).get("hold")
            if hold is not None:
                scalars += hold.n * (d + 1)
        return params, macs, scalars

    def summary(self) -> dict:
        out = super().summary()
        notes = [r for r in self.experts.values() if isinstance(r.module, NotebookExpert)]
        nets = [r for r in self.experts.values() if not isinstance(r.module, NotebookExpert)]
        out["nb"] = {"notebook_experts": len(notes), "net_experts": len(nets), "net_sizes": {r.id: (r.module.hidden, len(r.module.blocks)) for r in nets}, "judging_macs": self.nb_macs}
        return out
