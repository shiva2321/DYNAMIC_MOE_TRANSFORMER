"""Domain-balanced training of verified items (designs: docs/superpowers/specs/2026-10-31-fsmoe-balance-design.md, 2026-10-31-fsmoe-hybrid-design.md).

In the end-to-end stream the tool labels thousands of known code items for free while the human channel gives a few hundred, so arrival-order chunks are mostly code and crowd the other domains out. ``BalancedLearner``
builds every chunk of 16 from the domains seen so far in equal shares (inside a domain, items of classes the pool was never taught come first, balanced over their classes, then known items, oldest first):
* ``sub``  a chunk is formed only when EVERY domain seen holds its quota (the pace is the scarcest domain's pace; the abundant domain's surplus is never used);
* ``over`` a chunk is formed when ANY domain holds its quota of fresh items; a short domain fills its quota by replay from its own reservoir of verified items (every abundant item is used once, the scarce ones repeat);
* ``hybrid`` items of never-taught classes (any domain) are exempt from the throttle: they wait in one queue, in order of arrival, and a chunk is formed the moment it holds ``new_cap`` of them (so none ever waits for
  another domain); the rest of the chunk is the known items of the domains seen in equal shares (as ``sub``, on 16 - new_cap slots); a chunk is also formed when every domain holds its known share. No replay.
``DomainHead`` is the system's own guess of a human-labelled item's domain: a 3-class closed-form LDA fitted on the teaching rows, whose domain is known.
"""

from collections import Counter
from typing import Callable, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from unimoe.heads import LdaHead


class DomainHead:
    def __init__(self, d: int, n_domains: int = 3) -> None:
        self.head = LdaHead(d, n_domains, 0.1)

    def fit(self, X: torch.Tensor, domain: torch.Tensor) -> "DomainHead":
        self.head.update(X, domain.long())
        return self

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.head.proba(x.float().reshape(-1, x.shape[-1])).argmax(1)


class BalancedLearner:
    def __init__(self, observe: Callable[[torch.Tensor, torch.Tensor], object], mode: str, known_ids: Set[int], rng: np.random.Generator, chunk: int = 16, replay_cap: int = 500, new_cap: int = 8) -> None:
        if mode not in ("sub", "over", "hybrid"):
            raise ValueError(f"mode must be 'sub', 'over' or 'hybrid', got {mode!r}")
        if mode == "hybrid" and not 0 < new_cap < chunk:
            raise ValueError(f"new_cap must be between 1 and chunk - 1 ({chunk - 1}), got {new_cap}")
        self.observe, self.mode, self.known, self.rng, self.chunk, self.replay_cap, self.new_cap = observe, mode, set(known_ids), rng, chunk, replay_cap, new_cap
        self.fresh: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
        self.reservoir: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
        self.n_reservoir: Dict[int, int] = {}
        self.new_q: List[Tuple[torch.Tensor, int, int]] = []                       # hybrid: (x, label, domain) of never-taught classes
        self.seen: List[int] = []
        self.used: Counter = Counter()
        self.used_new: Counter = Counter()
        self.chunks = 0

    # ---- bookkeeping
    def add(self, x: torch.Tensor, label: int, domain: int) -> None:
        label = int(label)
        if self.mode == "hybrid" and label not in self.known:
            self.new_q.append((x, label, domain))
        else:
            if domain not in self.fresh:
                self.fresh[domain], self.reservoir[domain], self.n_reservoir[domain] = [], [], 0
                self.seen = sorted(self.seen + [domain])
            self.fresh[domain].append((x, label))
        self._flush(strict=True)

    def _quotas(self, domains: Sequence[int], size: int = 0) -> Dict[int, int]:
        base, rem = divmod(size or self.chunk, len(domains))
        return {d: base + (1 if i < rem else 0) for i, d in enumerate(domains)}

    @staticmethod
    def _round_robin(labels: Sequence[int]) -> List[int]:
        """Positions of ``labels`` taking one item of each class in turn (classes in label order, oldest first inside a class)."""
        by_class: Dict[int, List[int]] = {}
        for i, l in enumerate(labels):
            by_class.setdefault(l, []).append(i)
        out: List[int] = []
        while any(by_class.values()):
            for l in sorted(by_class):
                if by_class[l]:
                    out.append(by_class[l].pop(0))
        return out

    def _order(self, d: int) -> List[int]:
        """Indices of the fresh items of a domain: never-taught classes first (round-robin over those classes, oldest first), then known items oldest first."""
        items = self.fresh[d]
        new = [i for i, (_, l) in enumerate(items) if l not in self.known]
        rr = [new[j] for j in self._round_robin([items[i][1] for i in new])]
        return rr + [i for i, (_, l) in enumerate(items) if l in self.known]

    def _take(self, d: int, q: int) -> List[Tuple[torch.Tensor, int]]:
        pick = self._order(d)[:q]
        taken = [self.fresh[d][i] for i in pick]
        keep = set(pick)
        self.fresh[d] = [it for i, it in enumerate(self.fresh[d]) if i not in keep]
        return taken

    def _remember(self, d: int, items: List[Tuple[torch.Tensor, int]]) -> None:
        for it in items:
            self.n_reservoir[d] += 1
            if len(self.reservoir[d]) < self.replay_cap:
                self.reservoir[d].append(it)
            else:
                j = int(self.rng.integers(0, self.n_reservoir[d]))
                if j < self.replay_cap:
                    self.reservoir[d][j] = it

    def _emit(self, parts: List[Tuple[torch.Tensor, int]]) -> None:
        self.observe(torch.stack([x for x, _ in parts]), torch.tensor([l for _, l in parts]))
        self.chunks += 1

    # ---- chunk formation
    def _flush(self, strict: bool) -> None:
        if self.mode == "hybrid":
            while True:
                ready, q = self._hybrid_ready()
                if not ready or not self._build_hybrid(q):
                    return
        while self.seen:
            q = self._quotas(self.seen)
            ready = all(len(self.fresh[d]) >= q[d] for d in self.seen) if self.mode == "sub" else any(len(self.fresh[d]) >= q[d] for d in self.seen)
            if not ready:
                return
            self._build(q)

    def _build(self, q: Dict[int, int]) -> None:
        parts: List[Tuple[torch.Tensor, int]] = []
        for d in self.seen:
            taken = self._take(d, min(q[d], len(self.fresh[d])))
            fill: List[Tuple[torch.Tensor, int]] = []
            if self.mode == "over" and len(taken) < q[d] and self.reservoir[d]:
                idx = self.rng.integers(0, len(self.reservoir[d]), q[d] - len(taken))
                fill = [self.reservoir[d][int(i)] for i in idx]
            self._remember(d, taken)
            self.used[d] += len(taken)
            self.used_new[d] += sum(1 for _, l in taken if l not in self.known)
            parts += taken + fill
        if parts:
            self._emit(parts)

    def _hybrid_ready(self) -> Tuple[bool, Dict[int, int]]:
        q = self._quotas(self.seen, self.chunk - self.new_cap) if self.seen else {}
        known_ready = bool(self.seen) and all(len(self.fresh[d]) >= q[d] for d in self.seen)
        return known_ready or len(self.new_q) >= self.new_cap, q

    def _build_hybrid(self, q: Dict[int, int]) -> bool:
        parts: List[Tuple[torch.Tensor, int]] = []
        for x, l, d in self.new_q:                                                  # never more than new_cap: a chunk is formed the moment the queue holds new_cap items
            parts.append((x, l))
            self.used[d] += 1
            self.used_new[d] += 1
        self.new_q = []
        for d in self.seen:
            taken = self._take(d, min(q[d], len(self.fresh[d])))
            self.used[d] += len(taken)
            parts += taken
        if parts:
            self._emit(parts)
        return bool(parts)

    def finish(self) -> None:
        """Flush what is left at the same quotas: ``sub`` until the scarcest domain is empty, ``over`` until nothing fresh is left, ``hybrid`` until the new queue is empty and the scarcest domain's known items are used."""
        if self.mode == "hybrid":
            while True:
                q = self._quotas(self.seen, self.chunk - self.new_cap) if self.seen else {}
                known_left = bool(self.seen) and all(self.fresh[d] for d in self.seen)
                if not (self.new_q or known_left) or not self._build_hybrid(q):
                    return
        while self.seen:
            q = self._quotas(self.seen)
            if self.mode == "sub":
                if not all(self.fresh[d] for d in self.seen):
                    return
            elif not any(self.fresh[d] for d in self.seen):
                return
            self._build(q)
