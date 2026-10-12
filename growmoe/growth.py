"""Growth policy: propose -> validate -> commit, with exponential back-off after rejections.

``plateau`` says when an expert's training loss has stopped improving. ``GrowthPolicy.step`` then asks
whether more capacity would help on HELD-OUT data:

* propose   shadow copies grown at several scales (``widen`` to ``widen_factors`` x the current width, and
            ``deepen``), plus an unmodified control
* validate  control and shadows train for the same ``train_steps`` steps on the same minibatches; each
            shadow is compared with the control on held-out per-sample losses (one-sided paired bootstrap,
            Bonferroni over the candidates) and must beat it by ``min_gain`` (relative)
* commit    among the candidates that pass every gate, the one adding the FEWEST parameters is applied to the
            live expert as a function-preserving growth

The gates: relative gain over the control >= ``min_gain``, paired-bootstrap p < alpha / n_candidates, and strictly
better held-out loss than the expert had BEFORE the probe. The last gate matters for memorised (data-limited)
experts: with Adam their logits keep inflating while the probe trains, so the control's held-out loss rises
(measured 2.81 -> 4.6) and a wide shadow that merely degrades less (4.2) looks "11 % better than the control".

Why several scales: the held-out gain of capacity is not visible in small increments (Spike 3: for a width-32
teacher widening a width-4 student by +4 units gains +0.0 %, by +60 gains +5.3 %), so a policy that only tries
+50 % stays small forever. Why "fewest parameters": the gain saturates, so the largest candidate always has the
largest gain by a hair; sufficiency is defined by the two gates and the cheapest sufficient growth is taken.

The control keeps the policy honest: without it an under-trained expert would "benefit from growth" merely
because the probe trains it for more steps. A data-limited expert has a large train/held-out gap, so extra
capacity cannot lower its held-out loss and nothing is committed. A rejected proposal doubles the number of
calls skipped before the next one (capped); an acceptance resets it. Growth replaces parameters, so the
caller must rebuild its optimiser after a commit.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .config import GrowthCfg


def plateau(history, window: int = 40, tol: float = 0.01) -> bool:
    """True when the mean of the last ``window`` losses improved by less than ``tol`` (relative) on the window before."""
    h = np.asarray(history, dtype=np.float64)
    if h.size < 2 * window:
        return False
    recent, previous = h[-window:].mean(), h[-2 * window:-window].mean()
    return (previous - recent) < tol * max(abs(previous), 1e-12)


@dataclass
class GrowthDecision:
    kind: str = "none"                 # "none" | "widen" | "deepen"
    gain: float = 0.0                  # relative held-out loss reduction vs the equally trained control
    params_added: int = 0
    p_value: float = 1.0
    reason: str = ""
    increment: int = 0                 # hidden units added by a "widen" commit
    tried: dict = field(default_factory=dict)      # candidate -> {"gain", "p", "params_added"} for every one evaluated


class GrowthPolicy:
    def __init__(self, cfg: GrowthCfg = None) -> None:
        self.cfg = cfg if cfg is not None else GrowthCfg()
        self.backoff = self.cfg.backoff0           # calls to skip after the NEXT rejection
        self.cooldown = 0                          # calls still to skip

    def widen_increments(self, hidden: int) -> list:
        """Distinct, sorted numbers of hidden units to add: ``(factor - 1) * hidden`` floored at ``min_widen`` and capped."""
        room = self.cfg.max_hidden - hidden
        if room <= 0:
            return []
        return sorted({min(max(self.cfg.min_widen, int(round((f - 1.0) * hidden))), room) for f in self.cfg.widen_factors})

    @staticmethod
    def select(scored: dict, min_gain: float, level: float):
        """Cheapest candidate (fewest parameters added, then name) that passes every gate, else None. ``scored`` maps a
        candidate to (gain vs the equally trained control, its paired-bootstrap p, parameters added, gain vs the status quo).
        Gates: gain >= ``min_gain``, p < ``level`` AND strictly better than the expert as it is now."""
        passing = [(added, name) for name, (gain, p, added, vs_status) in scored.items()
                   if gain >= min_gain and p < level and vs_status > 0.0]
        return min(passing)[1] if passing else None

    def _paired_bootstrap_p(self, delta: np.ndarray, rng: np.random.Generator) -> float:
        idx = rng.integers(0, delta.shape[0], size=(self.cfg.n_boot, delta.shape[0]))
        return float((1 + (delta[idx].mean(axis=1) <= 0.0).sum()) / (self.cfg.n_boot + 1))

    def step(self, expert, x_tr, y_tr, x_va, y_va, rng: np.random.Generator) -> GrowthDecision:
        cfg = self.cfg
        if self.cooldown > 0:
            self.cooldown -= 1
            return GrowthDecision(reason="backoff")
        y_tr, y_va = y_tr.long(), y_va.long()
        seen = expert.seen.clone()
        seen[y_tr] = True
        if not bool(seen[y_va].all()):
            raise ValueError("validation labels include classes the expert has not seen")

        seeds = rng.integers(0, 2 ** 31 - 1, size=4)
        proposals = {}                                              # name -> (function that grows a model, units added)
        for k in self.widen_increments(expert.hidden):
            proposals[f"widen+{k}"] = (lambda m, k=k, s=int(seeds[1]) + k: m.widen(k, np.random.default_rng(s)), k)
        if len(expert.blocks) < cfg.max_depth:
            proposals["deepen"] = (lambda m, s=int(seeds[2]): m.deepen(np.random.default_rng(s)), 0)
        if not proposals:
            return GrowthDecision(reason="at capacity cap")

        data_rng = np.random.default_rng(int(seeds[0]))
        batches = [torch.from_numpy(data_rng.integers(0, x_tr.shape[0], size=cfg.batch_size))
                   for _ in range(cfg.train_steps)]

        def probe(model) -> np.ndarray:
            opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            for idx in batches:
                model.train_step(x_tr[idx], y_tr[idx], opt)
            model.eval()
            with torch.no_grad():
                return F.cross_entropy(model(x_va), y_va, reduction="none").numpy().astype(np.float64)

        expert.eval()
        with torch.no_grad():
            status = F.cross_entropy(expert(x_va), y_va, reduction="none").numpy().astype(np.float64)
        control = probe(expert.clone())
        base_params = expert.num_params()
        scored = {}
        for name, (grow, _) in proposals.items():
            shadow = expert.clone()
            grow(shadow)
            after = probe(shadow)
            delta = control - after
            gain = float(delta.mean() / max(float(control.mean()), 1e-12))
            vs_status = float((status.mean() - after.mean()) / max(float(status.mean()), 1e-12))
            scored[name] = (gain, self._paired_bootstrap_p(delta, np.random.default_rng(int(seeds[3]))),
                            shadow.num_params() - base_params, vs_status)
        tried = {n: {"gain": g, "p": p, "params_added": a, "vs_status": v} for n, (g, p, a, v) in scored.items()}

        pick = self.select(scored, cfg.min_gain, cfg.alpha / len(scored))
        if pick is None:
            self.cooldown = self.backoff
            self.backoff = min(2 * self.backoff, cfg.max_backoff)
            return GrowthDecision(gain=max(v[0] for v in scored.values()), p_value=min(v[1] for v in scored.values()),
                                  reason="rejected", tried=tried)
        grow, units = proposals[pick]
        before = expert.num_params()
        grow(expert)
        self.backoff, self.cooldown = cfg.backoff0, 0
        return GrowthDecision(kind="deepen" if pick == "deepen" else "widen", gain=scored[pick][0],
                              params_added=expert.num_params() - before, p_value=scored[pick][1], increment=units,
                              reason="committed", tried=tried)
