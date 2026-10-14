"""Stage 3a of routing v2: the batch-level gate (docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md, section 4).

The per-sample gate cannot see a foreign batch whose inputs each look plausible (real text categories give AUROC about 0.65). This gate asks a different
question of the samples the per-sample gate ACCEPTED: "do these samples, as a set, look like a random draw from what some expert was trained on?" The answer
is an exact permutation test (``growmoe.settest``) against the expert's reference samples (one window per recent source, ``growmoe.sources``).

This module is the decision logic only. The pool supplies the references and a loss function, so the logic is testable without a pool:

    gate = SetGate(cfg)
    decision = gate.decide(xb, yb, eids, refs_of, loss_of, step)       # known (with an owner) or novel as a set
    gate.note(decision.verdict, accepted_share, step)                  # feeds the circuit breaker

Every random draw comes from a generator keyed by (cfg.seed, step, expert, reference), so a decision is a pure function of the pool state and the batch.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .config import PhyloCfg
from .settest import cap_rows, generator, perm_reject

_CAP, _TEST = 0x434150, 0x534554                                                        # key parts: "CAP", "SET"


@dataclass
class SetDecision:
    verdict: str                                                                        # "known" | "novel"
    owner: int = -1                                                                     # the expert that trains the batch (-1: novel)
    admitted: List[int] = field(default_factory=list)                                  # admitted experts, in the order they were admitted
    tried: List[int] = field(default_factory=list)                                     # experts that were tested at all


class SetGate:
    STAT_KEYS = ("tests", "early_stops", "perms", "novel", "known", "override", "paused", "breaker_opens", "window_splits", "held", "released", "dropped")

    def __init__(self, cfg: PhyloCfg) -> None:
        self.cfg = cfg
        self.stats: Dict[str, int] = {k: 0 for k in self.STAT_KEYS}                     # counts only; no wall-clock
        self.hist: deque = deque(maxlen=cfg.set_gate_breaker_len)                       # True = the batch was called new as a set
        self.pause_until = -1
        self.streak = 0                                                                 # doubtful decided batches in a row (persistence, amendment A-15)

    # ------------------------------------------------------------------------------------------------------- persistence
    def confirm(self, verdict: str) -> str:
        """Persistence (amendment A-15). ``verdict`` is the raw verdict of ``decide`` ("known" or "novel"). With ``set_gate_confirm`` = c a "novel" verdict is
        acted upon only when it is the c-th doubtful decided batch in a row; the earlier ones return "held". A "known" verdict resets the streak. Batches
        that were not decided (fewer than ``batch_vote_min`` accepted samples, or a paused gate) leave the streak as it is. With c = 1 nothing is held."""
        if verdict == "known":
            self.streak = 0
            return "known"
        self.streak += 1
        return "novel" if self.streak >= self.cfg.set_gate_confirm else "held"

    # ------------------------------------------------------------------------------------------------------- breaker
    def paused(self, step: int) -> bool:
        return step < self.pause_until

    def note(self, verdict: str, accepted_share: float, step: int) -> bool:
        """Feed the circuit breaker with the verdict of one batch; only batches with at least 90 % of their samples accepted count (a batch the
        per-sample gate already doubted says nothing about the set test). Returns True iff the breaker opened now."""
        if accepted_share < 0.9:
            return False
        cfg = self.cfg
        self.hist.append(verdict == "novel")
        if len(self.hist) == self.hist.maxlen and sum(self.hist) / len(self.hist) > cfg.set_gate_breaker_rate:
            self.pause_until = step + cfg.set_gate_breaker_pause
            self.hist.clear()
            self.stats["breaker_opens"] += 1
            return True
        return False

    # ------------------------------------------------------------------------------------------------------- decision
    def decide(self, xb: Tensor, yb: Optional[Tensor], eids: Sequence[int], refs_of: Callable[[int], List[Tuple[Tensor, Tensor]]],
               loss_of: Callable[[int, Tensor, Tensor], float], step: int) -> SetDecision:
        """``xb`` (n, d): the samples the per-sample gate accepted; ``yb`` their labels (-1 = unlabelled) or None; ``refs_of(e)`` the references of
        expert ``e`` as (rows, mean row) pairs (empty: the expert cannot be tested and is never admitted); ``loss_of(e, x, y)`` the cross-entropy of
        expert ``e`` on labelled samples.

        All (expert, reference) pairs are sorted by the distance between the reference mean and the batch mean; at most ``set_gate_candidates`` distinct
        experts are tried, an expert is admitted at its first reference (nearest first) that does not reject. Nobody admitted: novel as a set."""
        cfg = self.cfg
        xt = cap_rows(xb, cfg.set_gate_max_rows, generator(cfg.seed, _CAP, step))
        owners, slots, rows, means = [], [], [], []
        for e in eids:
            for j, (r, m) in enumerate(refs_of(e)):
                owners.append(e)
                slots.append(j)
                rows.append(r)
                means.append(m)
        if not owners:
            self.stats["novel"] += 1
            return SetDecision("novel", -1, [], [])
        dist = (torch.stack(means) - xb.mean(0)).norm(dim=1).tolist()
        order = sorted(range(len(owners)), key=lambda i: (dist[i], owners[i], slots[i]))     # ties: lower expert id, then lower reference index
        pairs = [(dist[i], owners[i], slots[i], rows[i]) for i in order]
        tried: List[int] = []
        admitted: List[int] = []
        for _, e, j, r in pairs:
            if e in admitted:
                continue
            if e not in tried:
                if len(tried) == cfg.set_gate_candidates:
                    continue
                tried.append(e)
            ref = cap_rows(r, cfg.set_gate_max_rows, generator(cfg.seed, _CAP, step, e, j))
            rejected, _, used = perm_reject(ref, xt, cfg.set_gate_alpha, cfg.set_gate_perms, generator(cfg.seed, _TEST, step, e, j),
                                            cfg.set_gate_stat, cfg.set_gate_chunk)
            self.stats["tests"] += 1
            self.stats["perms"] += used
            self.stats["early_stops"] += int(used < cfg.set_gate_perms)
            if not rejected:
                admitted.append(e)
                if cfg.set_gate_ownership == "nearest":
                    break
        if not admitted:
            self.stats["novel"] += 1
            return SetDecision("novel", -1, [], tried)
        owner = admitted[0]
        if cfg.set_gate_ownership == "loss" and len(admitted) > 1 and yb is not None:
            ok = yb >= 0
            if int(ok.sum()) >= cfg.set_gate_min_labels:
                losses = {e: float(loss_of(e, xb[ok], yb[ok])) for e in admitted}
                owner = min(admitted, key=losses.get)                                   # ties: the one admitted first (nearest)
        self.stats["known"] += 1
        return SetDecision("known", owner, admitted, tried)
