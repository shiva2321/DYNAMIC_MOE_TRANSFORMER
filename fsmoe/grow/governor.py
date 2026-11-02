"""The governor: decides when the system grows, in which direction, and whether the growth stays.

One rule for every direction: grow only on evidence of need, check on data the expert never trained on that it helped, keep the cheapest option that works, undo it if it did not help, say why in the
ledger (``GrowSystem.events``). The four directions:
* a notebook-only newborn gets a network only if network + notebook beat the notebook alone on the expert's held-out samples by ``promote_margin`` (``notebook`` experts are judged again when their data has doubled);
* an expert is asked about growth (wider or deeper) only when it has trained long enough, holds enough data, its held-out error is above ``err_trigger`` and the trials so far have not cost more than
  ``trial_ratio`` times the training; the question is answered by growmoe's own
  trial (``GrowthPolicy``: shadows grown at several scales and an equally trained control, a paired bootstrap on held-out losses, the cheapest candidate that passes every gate, exponential back-off after
  a rejection); a committed growth is kept for ``guard_checks`` looks and undone if the held-out loss is worse by more than ``guard_eps``; nothing that would take the pool above ``max_params`` stays;
* more experts: the system's own room-and-spawn rule (``GrowSystem``), which absorbs a group that is close to an existing expert instead of founding a duplicate;
* a taller system: when the flat gate would cost more than ``gate_budget`` multiply-adds per sample, the experts are grouped by k-means over their centroids and routed in two steps; the tier stays only
  if it agrees with the flat routing on at least ``tier_agree`` of the experts' held-out samples and is cheaper by at least 20 %.
``BruteGovernor`` is the simple way to grow, for comparison: every expert gets wider or deeper at every look, alternately, with no test.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from growmoe.config import GrowthCfg
from growmoe.experts import MLPExpert
from growmoe.gapgrow import guard_reverts
from growmoe.growth import GrowthPolicy

from .system import Expert, GrowSystem, Tier


def _trial_cfg() -> GrowthCfg:
    return GrowthCfg(widen_factors=(2.0, 4.0), min_widen=4, train_steps=240, batch_size=64, lr=3e-3, min_gain=0.02, alpha=0.05, n_boot=500, max_hidden=128, max_depth=4)


@dataclass
class GovCfg:
    check_every: int = 10
    age_min: int = 6                 # batches an expert must have trained before it is judged
    hold_min: int = 100
    train_min: int = 600
    err_trigger: float = 0.04        # held-out error above which growth is asked about
    promote_min: int = 300           # training samples a notebook needs before a network is tried
    promote_margin: float = 0.02
    promote_steps: int = 200
    promote_sizes: tuple = ((8, 1), (8, 3), (32, 1), (64, 1))     # (units, blocks) of the candidate networks
    promote_knee: float = 0.015      # take the cheapest candidate within this accuracy of the best one
    grow: bool = True
    tier: bool = True
    guard_checks: int = 2
    guard_eps: float = 0.05
    max_params: Optional[int] = None
    trial_ratio: Optional[float] = 1.0   # growth trials may cost at most this many times what training has cost so far (multiply-adds); None = no limit
    tier_agree: float = 0.98
    tier_regrow: float = 1.25        # rebuild the tier when the pool is this much larger than when it was built
    trial: GrowthCfg = field(default_factory=_trial_cfg)


def pick_size(tried, notebook_acc: float, margin: float, knee: float) -> Optional[int]:
    """Which candidate network to keep. ``tried`` lists (parameters, held-out accuracy of network + notebook) per candidate. None when no candidate beats the notebook alone by ``margin``; otherwise the
    index of the cheapest candidate (fewest parameters) whose accuracy is within ``knee`` of the best."""
    best = max(a for _, a in tried)
    if best < notebook_acc + margin:
        return None
    return min((i for i, (_, a) in enumerate(tried) if a >= best - knee), key=lambda i: tried[i][0])


def _hold_error(e: Expert) -> float:
    p = e.proba(e.hold.x)
    return float(1.0 - (p.argmax(1) == e.hold.y).float().mean())


def _net_loss(net: MLPExpert, x: torch.Tensor, y: torch.Tensor) -> float:
    net.eval()
    with torch.no_grad():
        return float(F.cross_entropy(net(x), y))


class Governor:
    def __init__(self, cfg: Optional[GovCfg] = None) -> None:
        self.cfg = cfg if cfg is not None else GovCfg()

    def step(self, s: GrowSystem) -> None:
        if s.batches % self.cfg.check_every:
            return
        for e in list(s.experts.values()):
            self._guard(s, e)
            if e.state in ("newborn", "notebook"):
                self._promote(s, e)
            elif e.net is not None and self.cfg.grow:
                self._grow(s, e)
        if self.cfg.tier:
            self._tier(s)

    # ---- notebook -> network
    def _promote(self, s: GrowSystem, e: Expert) -> None:
        """Right-size a network at birth with the notebook as the control: every candidate size is trained on the expert's own samples and judged on its held-out ones, as network + notebook against the
        notebook alone. The notebook stays if no candidate wins by ``promote_margin``; otherwise the cheapest candidate within ``promote_knee`` of the best one is kept."""
        c = self.cfg
        n = len(e.train)
        if n < max(c.promote_min, e.next_decision) or len(e.hold) < c.hold_min or e.age < c.age_min:
            return
        p_nb = e.nb.proba(e.hold.x)
        acc_nb = float((p_nb.argmax(1) == e.hold.y).float().mean())
        tried = []
        for hidden, depth in c.promote_sizes:
            net = MLPExpert(s.cfg.d, hidden, depth, s.cfg.k, seed=s.cfg.seed * 1000 + e.id)
            opt = torch.optim.Adam(net.parameters(), lr=s.cfg.lr)
            g = torch.Generator().manual_seed(s.cfg.seed * 1000 + e.id)
            for _ in range(c.promote_steps):
                idx = torch.randint(0, n, (128,), generator=g)
                net.train_step(e.train.x[idx], e.train.y[idx], opt)
            s.macs_trial += 3.0 * net.num_params() * 128 * c.promote_steps
            net.eval()
            with torch.no_grad():
                p_bl = 0.5 * F.softmax(net(e.hold.x), dim=-1) + 0.5 * p_nb
            tried.append((net.num_params(), float((p_bl.argmax(1) == e.hold.y).float().mean()), net, opt, hidden, depth))
        pick = pick_size([(t[0], t[1]) for t in tried], acc_nb, c.promote_margin, c.promote_knee)
        best = max(a for _, a, *_ in tried)
        if pick is not None:
            params, acc_bl, net, opt, hidden, depth = tried[pick]
            e.net, e.opt, e.state = net, opt, "active"
            s.log("promote", expert=e.id, notebook_acc=acc_nb, blend_acc=acc_bl, params=params, hidden=hidden, depth=depth, candidates={f"{h}x{d}": round(a, 4) for _, a, _, _, h, d in tried})
        else:
            e.state, e.next_decision = "notebook", 2 * n
            s.log("keep_notebook", expert=e.id, notebook_acc=acc_nb, best_blend_acc=best)

    # ---- wider or deeper
    def _grow(self, s: GrowSystem, e: Expert) -> None:
        c = self.cfg
        if e.guard is not None or e.age < c.age_min or len(e.hold) < c.hold_min or len(e.train) < c.train_min:
            return
        err = _hold_error(e)
        if err <= c.err_trigger:
            return
        if c.trial_ratio is not None and s.macs_trial_grow > c.trial_ratio * s.macs_train:
            return
        if e.policy is None:
            e.policy = GrowthPolicy(c.trial)
        net = e.net
        seen_va = net.seen[e.hold.y]
        x_va, y_va = e.hold.x[seen_va], e.hold.y[seen_va]
        if x_va.shape[0] < c.hold_min // 2:
            return
        snap, before = net.clone(), _net_loss(net, e.hold.x[seen_va], e.hold.y[seen_va])
        rng = np.random.default_rng([s.cfg.seed, e.id, s.batches])
        params0 = net.num_params()
        dec = e.policy.step(net, e.train.x, e.train.y, x_va, y_va, rng)
        if dec.reason != "backoff" and dec.tried:
            cost = 3.0 * params0 * c.trial.batch_size * c.trial.train_steps * (1 + len(dec.tried))
            s.macs_trial += cost
            s.macs_trial_grow += cost
        if dec.kind == "none":
            if dec.reason != "backoff":
                s.log("trial", expert=e.id, outcome=dec.reason, held_error=err, best_gain=dec.gain)
            return
        e.rebuild_opt()
        if c.max_params is not None and s.params()["total"] > c.max_params:
            e.net = snap
            e.rebuild_opt()
            s.log("refused", expert=e.id, why="budget", direction=dec.kind)
            return
        e.guard = {"snap": snap, "before": before, "looks": 0}
        s.log("grow", expert=e.id, direction=dec.kind, params_added=dec.params_added, gain=dec.gain, p=dec.p_value, held_error=err, hidden=net.hidden, depth=len(net.blocks))

    def _guard(self, s: GrowSystem, e: Expert) -> None:
        if e.guard is None:
            return
        e.guard["looks"] += 1
        if e.guard["looks"] < self.cfg.guard_checks:
            return
        seen_va = e.net.seen[e.hold.y]
        after = _net_loss(e.net, e.hold.x[seen_va], e.hold.y[seen_va])
        if guard_reverts(e.guard["before"], after, self.cfg.guard_eps):
            e.net = e.guard["snap"]
            e.rebuild_opt()
            s.log("revert", expert=e.id, before=e.guard["before"], after=after)
        else:
            s.log("keep", expert=e.id, before=e.guard["before"], after=after)
        e.guard = None

    # ---- the tier
    def _tier(self, s: GrowSystem) -> None:
        c = self.cfg
        ids = sorted(s.experts)
        E, d = len(ids), s.cfg.d
        flat = float(E * d)
        if flat <= s.cfg.gate_budget:
            return
        if s.tier is not None and E < c.tier_regrow * s.tier.built_for:
            return
        if E < c.tier_regrow * getattr(s, "tier_tried", 0):                      # a refusal is not repeated until the pool has grown
            return
        s.tier_tried = E
        C = torch.stack([s.experts[i].centroid for i in ids])
        G = int(np.ceil(np.sqrt(E)))
        groups, centres = _kmeans_groups(C, ids, G, seed=s.cfg.seed * 7 + E)
        val = torch.cat([s.experts[i].hold.x[:60] for i in ids if len(s.experts[i].hold)])
        if val.shape[0] == 0:
            return
        s.tier = None
        flat_route, _, _ = s.route(val)
        for top_g in range(1, min(3, len(groups)) + 1):
            s.tier = Tier(groups, centres, top_g, E)
            tier_route, _, gate = s.route(val)
            agree = float((tier_route == flat_route).float().mean())
            if agree >= c.tier_agree and gate <= 0.8 * flat:
                s.log("tier", groups=len(groups), top_g=top_g, agreement=agree, gate=gate, flat=flat)
                return
        s.tier = None
        s.log("tier_refused", experts=E, flat=flat)


def _kmeans_groups(C: torch.Tensor, ids: List[int], G: int, seed: int, iters: int = 15):
    """Spherical k-means over the expert centroids: (list of lists of expert ids, unit group directions)."""
    rng = np.random.default_rng(seed)
    centres = C[torch.from_numpy(rng.choice(C.shape[0], size=G, replace=False))].clone()
    assign = torch.zeros(C.shape[0], dtype=torch.long)
    for _ in range(iters):
        assign = (C @ centres.T).argmax(1)
        for g in range(G):
            m = assign == g
            if bool(m.any()):
                centres[g] = F.normalize(C[m].mean(0), dim=0)
    groups = [[ids[i] for i in range(len(ids)) if int(assign[i]) == g] for g in range(G)]
    keep = [g for g in range(G) if groups[g]]
    return [groups[g] for g in keep], centres[keep]


class BruteGovernor:
    """The simple way to grow, for comparison: at every look every expert with a network gets wider (doubling, up to 128 units) and deeper (up to 4 blocks) in turn, with no test and no undo."""

    def __init__(self, check_every: int = 10, age_min: int = 6, max_hidden: int = 128, max_depth: int = 4) -> None:
        self.check_every, self.age_min, self.max_hidden, self.max_depth = check_every, age_min, max_hidden, max_depth

    def step(self, s: GrowSystem) -> None:
        if s.batches % self.check_every:
            return
        for e in s.experts.values():
            if e.net is None or e.age < self.age_min:
                continue
            turn = getattr(e, "brute_turn", 0)
            if turn % 2 == 0 and e.net.hidden * 2 <= self.max_hidden:
                e.net.widen(e.net.hidden)
                kind = "widen"
            elif len(e.net.blocks) < self.max_depth:
                e.net.deepen()
                kind = "deepen"
            else:
                continue
            e.brute_turn = turn + 1
            e.rebuild_opt()
            s.log("grow", expert=e.id, direction=kind, hidden=e.net.hidden, depth=len(e.net.blocks))
