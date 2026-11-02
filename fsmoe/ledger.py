"""A trust ledger that keeps score the way the winners of Axelrod's tournaments did, adapted to noise.

Each expert has a trust in [floor, 1]: how much to believe what it says about its own work. Verified outcomes (a label that arrives, an answer the outside anchor checks) move it:
* nice: it starts high (``start``), a new expert is trusted until it is caught wrong;
* retaliate: a verified failure cuts trust at once (rate ``down``), and a run of failures cuts it faster (the cut grows with the number in a row);
* forgive: every step the trust drifts back toward ``start`` (rate ``forgive``), and the first failure after successes is discounted by ``generosity``, so one bad batch in noisy data does not wreck an
  expert (strict tit-for-tat echoes a single error back and forth under noise; the generous and win-stay-lose-shift variants do not).
No outcome is ever taken from the expert itself: the caller passes outcomes the outside anchor produced.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrustLedger:
    start: float = 0.9
    up: float = 0.15
    down: float = 0.5
    generosity: float = 0.5
    forgive: float = 0.02
    floor: float = 0.1
    trust: Dict[int, float] = field(default_factory=dict)
    streak: Dict[int, int] = field(default_factory=dict)

    def add(self, eid: int) -> None:
        self.trust.setdefault(eid, self.start)
        self.streak.setdefault(eid, 0)

    def verified(self, eid: int, correct: bool) -> float:
        """Record one verified outcome (or a batch's verdict) for expert ``eid``; returns the new trust."""
        self.add(eid)
        t = self.trust[eid]
        if correct:
            t += self.up * (1.0 - t)
            self.streak[eid] = 0
        else:
            n = self.streak[eid]
            cut = self.down * (1.0 - (self.generosity if n == 0 else 0.0)) * min(1.0 + 0.5 * n, 2.0)
            t -= cut * t
            self.streak[eid] = n + 1
        self.trust[eid] = max(self.floor, min(1.0, t))
        return self.trust[eid]

    def tick(self) -> None:
        """One step of forgiveness for every expert: trust drifts back toward its starting value."""
        for e, t in self.trust.items():
            self.trust[e] = t + self.forgive * (self.start - t)

    def get(self, eid: int) -> float:
        return self.trust.get(eid, self.start)

    def thin(self, eid: int, threshold: float = 0.5) -> bool:
        return self.get(eid) < threshold
