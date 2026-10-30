"""The stream suite: real domains on one global label space, in three schedules, with a test set per regime and out-of-distribution sets that are never trained on.

Every domain gets a block of global labels (``label_blocks``); the digit-label domain keeps its twelve shared symbols inside its block. A regime is a group of classes of one text domain, or one
arithmetic operator. Schedules:
* ``seq``   the continual setting: domains in a seed-dependent order, the regimes of a domain consecutively (one phase each), then revisits of a few regimes drawn from anywhere;
* ``mix``   the stationary setting: the same first-visit data of every regime, shuffled i.i.d. and cut into phases (the setting where the pool never grew, C-48);
* ``drift`` the weight of each regime moves through time as a bump, so regimes fade in and out.
The algorithm sees ``(x, y)`` only; regime ids, first-visit flags and phase numbers are for the benchmark (``batches`` returns them, ``algo_batches`` does not).
"""

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from .domains import ALL_DOMAINS, Domain

SCHEDULES = ("seq", "mix", "drift")


@dataclass
class Regime:
    id: int
    name: str
    domain: str
    kind: str
    local: int                                       # index of the regime inside its domain
    x_te: torch.Tensor
    y_te: torch.Tensor                               # global labels
    meta: dict = field(default_factory=dict)


@dataclass
class Phase:
    name: str
    x: torch.Tensor
    y: torch.Tensor                                  # global labels
    regime: torch.Tensor                             # benchmark only
    first: torch.Tensor                              # benchmark only: the sample's regime had not appeared in an earlier phase


@dataclass
class UStream:
    name: str
    d: int
    n_classes: int
    regimes: Dict[int, Regime]
    phases: List[Phase]
    ood: Dict[str, torch.Tensor]
    label_blocks: Dict[str, Tuple[int, int]]         # domain -> (first global label, number of labels)
    meta: dict = field(default_factory=dict)

    def _order(self, k: int, seed: int) -> torch.Tensor:
        return torch.from_numpy(np.random.default_rng([seed, 0x42415443, k]).permutation(len(self.phases[k].y)))

    def batches(self, k: int, batch_size: int, seed: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor, dict]]:
        ph, order = self.phases[k], self._order(k, seed)
        for lo in range(0, len(order), batch_size):
            idx = order[lo:lo + batch_size]
            yield ph.x[idx], ph.y[idx], {"regime": ph.regime[idx], "first": ph.first[idx], "phase": k}

    def algo_batches(self, k: int, batch_size: int, seed: int = 0) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        for x, y, _ in self.batches(k, batch_size, seed):
            yield x, y

    def test_sets(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        return {r: (reg.x_te, reg.y_te) for r, reg in self.regimes.items()}


def build_suite(schedule: str, domains: Dict[str, Domain], seed: int, ood: Optional[Dict[str, torch.Tensor]] = None, n_visit: int = 400, n_visit_arith: int = 3000,
                n_test: int = 300, revisit_k: int = 6, revisit_frac: float = 0.5, n_phases_mix: int = 5, n_phases_drift: int = 10) -> UStream:
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
    names = [n for n in ALL_DOMAINS if n in domains]
    d_dim = domains[names[0]].x_train.shape[1]
    blocks, offset = {}, 0
    for n in names:
        blocks[n] = (offset, domains[n].n_classes)
        offset += domains[n].n_classes
    rng = np.random.default_rng([seed, 0x53554954])
    regimes: Dict[int, Regime] = {}
    pools: Dict[int, np.ndarray] = {}
    for n in names:
        d, (off, _) = domains[n], blocks[n]
        for local, rname in enumerate(d.regime_names):
            rid = len(regimes)
            te = np.flatnonzero((d.r_test == local).numpy())
            te = np.sort(rng.permutation(te)[:n_test])
            meta = {"test_index": te}                                                                  # which examples of the domain's test set this regime holds (to find their texts later)
            if "problem_id_test" in d.meta:
                meta["problem_id"] = d.meta["problem_id_test"][torch.from_numpy(te)]
            regimes[rid] = Regime(rid, rname, n, d.kind, local, d.x_test[te], d.y_test[te] + off, meta)
            pools[rid] = np.random.default_rng([seed, 0x504F4F4C, rid]).permutation(np.flatnonzero((d.r_train == local).numpy()))
    dom_of = {rid: regimes[rid].domain for rid in regimes}
    used = {rid: 0 for rid in regimes}

    def take(rid: int, n: int, first: bool) -> np.ndarray:
        pool = pools[rid]
        n = min(n, len(pool)) if first else n
        idx = pool[(used[rid] + np.arange(n)) % len(pool)]
        used[rid] += n
        return idx

    def visit_size(rid: int) -> int:
        return n_visit_arith if regimes[rid].kind == "digits" else n_visit

    def make(rid: int, idx: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        d, (off, _) = domains[dom_of[rid]], blocks[dom_of[rid]]
        ii = torch.from_numpy(idx)
        return d.x_train[ii], d.y_train[ii] + off

    def phase(name: str, parts: List[Tuple[int, np.ndarray]], seen: set) -> Phase:
        xs, ys, rs, fs = [], [], [], []
        for rid, idx in parts:
            x, y = make(rid, idx)
            xs.append(x)
            ys.append(y)
            rs.append(torch.full((len(idx),), rid, dtype=torch.long))
            fs.append(torch.full((len(idx),), rid not in seen, dtype=torch.bool))
        for rid, _ in parts:
            seen.add(rid)
        return Phase(name, torch.cat(xs), torch.cat(ys), torch.cat(rs), torch.cat(fs))

    phases: List[Phase] = []
    seen: set = set()
    if schedule == "seq":
        order = [names[i] for i in rng.permutation(len(names))]
        seq = [rid for n in order for rid in regimes if dom_of[rid] == n]
        for rid in seq:
            phases.append(phase(f"first_{regimes[rid].name}", [(rid, take(rid, visit_size(rid), True))], seen))
        for rid in rng.permutation(list(regimes))[:revisit_k]:
            rid = int(rid)
            phases.append(phase(f"revisit_{regimes[rid].name}", [(rid, take(rid, max(1, int(visit_size(rid) * revisit_frac)), False))], seen))
    elif schedule == "mix":
        parts = [(rid, take(rid, visit_size(rid), True)) for rid in regimes]
        rid_all = np.concatenate([np.full(len(i), r) for r, i in parts])
        idx_all = np.concatenate([i for _, i in parts])
        perm = rng.permutation(len(idx_all))
        for k, chunk in enumerate(np.array_split(perm, n_phases_mix)):
            sel = [(int(r), idx_all[chunk][rid_all[chunk] == r]) for r in np.unique(rid_all[chunk])]
            phases.append(phase(f"mix_{k}", sel, seen))
    else:
        R, P = len(regimes), n_phases_drift
        total = sum(min(visit_size(r), len(pools[r])) for r in regimes)
        per_phase = max(1, total // P)
        centers = np.linspace(0, P - 1, R)[rng.permutation(R)]
        sigma = max(1.0, 1.5 * P / R)
        for k in range(P):
            w = np.exp(-((k - centers) ** 2) / (2 * sigma ** 2)) + 0.02
            counts = rng.multinomial(per_phase, w / w.sum())
            sel = [(rid, take(rid, int(c), False)) for rid, c in zip(regimes, counts) if c > 0]
            phases.append(phase(f"drift_{k}", sel, seen))
    return UStream(f"unimoe-{schedule}", d_dim, offset, regimes, phases, dict(ood or {}), blocks, {"schedule": schedule, "seed": seed, "names": names})
