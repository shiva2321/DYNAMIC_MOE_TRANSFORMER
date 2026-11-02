"""UniPool: the shipped ``PhyloMoE`` plus the four changes the Stage 1 analysis asked for, each behind a switch that is OFF by default (with all of them off it is the shipped pool; tested).

* ``gap_mode="one_step"``  the growth rule's "fit gap" is read from a one-step probe on a copy of the expert (the share of a fresh batch's loss that ONE optimiser step removes), not after all the
  inner steps. The shipped statistic is 0.09 / 0.59 / 0.98 at 1 / 5 / 20 inner steps for the same expert, so the recipe, not the capacity, decided whether growth could fire.
* ``domain_head`` / ``domain_blocks``  the labels of a stream fall in per-domain blocks, so every labelled message also says which domain it came from. A running LDA over the class labels, summed
  within each domain's block, is trained from that for free and fused into the answer: "hard" restricts the answer to the predicted domain's classes, "soft" multiplies each class by its domain's probability (power ``domain_beta``).
* ``lda_blend``  every expert also keeps a running LDA head over the samples it trained on; its answer is blended with the expert's in log-probability. Its storage is counted in ``extra_cost``.
growmoe/ is not edited: everything is an override.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from growmoe.config import PhyloCfg
from growmoe.gapgrow import GapMonitor, params_of_widen, widen_increment
from growmoe.pools.phylo import PhyloMoE
from growmoe.types import Prediction

from . import heads as HD


@dataclass
class UniCfg(PhyloCfg):
    gap_mode: str = "recipe"                  # "recipe" (shipped) | "one_step"
    domain_blocks: tuple = ()                 # ((first label, number of labels), ...) per domain
    domain_head: str = "off"                  # "off" | "hard" | "soft"
    domain_beta: float = 1.0
    lda_blend: float = 0.0                    # weight of the per-expert LDA head in the answer, 0 = off
    lda_shrink: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.gap_mode not in ("recipe", "one_step"):
            raise ValueError(f"gap_mode must be 'recipe' or 'one_step', got {self.gap_mode!r}")
        if self.domain_head not in ("off", "hard", "soft"):
            raise ValueError(f"domain_head must be 'off', 'hard' or 'soft', got {self.domain_head!r}")
        if self.domain_head != "off" and not self.domain_blocks:
            raise ValueError("a domain head needs the label blocks (domain_blocks)")
        if not 0.0 <= self.lda_blend <= 1.0:
            raise ValueError(f"lda_blend must lie in [0, 1], got {self.lda_blend}")


# ------------------------------------------------------------------------------------------------------------------------------ pure fusion functions
def blend(p_mlp: Tensor, p_lda: Tensor, lam: float) -> Tensor:
    """Log-linear blend of two class distributions; lam = 0 returns the first unchanged."""
    if lam == 0.0:
        return p_mlp
    logp = (1.0 - lam) * torch.log(p_mlp.clamp_min(1e-12)) + lam * torch.log(p_lda.clamp_min(1e-12))
    return torch.softmax(logp, dim=-1)


def fuse_domain(probs: Tensor, dom: Tensor, blocks: Tuple[Tuple[int, int], ...], mode: str, beta: float) -> Tensor:
    """Fuse class probabilities (B, C) with domain probabilities (B, D). 'hard' keeps only the classes of the most probable domain (the answer is kept if that leaves nothing), 'soft' multiplies each
    class by its domain's probability to the power ``beta``; labels outside every block keep a tiny factor so they are not erased."""
    if mode == "off":
        return probs
    C, D = probs.shape[1], len(blocks)
    M = torch.zeros(C, D)
    for d, (off, n) in enumerate(blocks):
        M[off:min(off + n, C), d] = 1.0
    if mode == "hard":
        out = probs * M[:, dom.argmax(1)].T
    else:
        out = probs * (dom @ M.T).clamp_min(1e-6) ** beta
    s = out.sum(1, keepdim=True)
    return torch.where(s > 0, out / s.clamp_min(1e-30), probs)


# ------------------------------------------------------------------------------------------------------------------------------ the pool
class UniPool(PhyloMoE):
    def __init__(self, cfg: UniCfg) -> None:
        super().__init__(cfg)
        # the domain head is an LDA over the CLASS labels whose class probabilities are summed within each domain's block: a domain is a union of clusters, which one Gaussian per domain models badly
        self.domain_lda: Optional[HD.LdaHead] = HD.LdaHead(cfg.d, cfg.n_classes, cfg.lda_shrink) if cfg.domain_head != "off" else None
        self._dom_of_label = torch.full((cfg.n_classes,), -1, dtype=torch.long)
        self._dom_M = torch.zeros(cfg.n_classes, max(len(cfg.domain_blocks), 1))
        for d, (off, n) in enumerate(cfg.domain_blocks):
            self._dom_of_label[off:min(off + n, cfg.n_classes)] = d
            self._dom_M[off:min(off + n, cfg.n_classes), d] = 1.0

    # ---- learning
    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None):
        rec = super().observe(x, y, ctx)
        if self.domain_lda is not None and y is not None:
            m = self._dom_of_label[y.long()] >= 0
            if bool(m.any()):
                self.domain_lda.update(self._normalize(x.float())[m], y.long()[m])
        return rec

    def _one_step_probe(self, rec, x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        """Honest per-sample losses on (up to) 64 rows of the batch before and after ONE optimiser step taken on a copy of the expert (fresh optimiser state, the pool's learning rate)."""
        m = min(64, int(y.numel()))
        y = y.long()
        probe = rec.module.clone()
        probe.mark_seen(y)
        with torch.no_grad():
            pre = F.cross_entropy(probe(x[:m]), y[:m], reduction="none")
        probe.train_step(x, y, torch.optim.Adam(probe.parameters(), lr=self.cfg.lr))
        with torch.no_grad():
            post = F.cross_entropy(probe(x[:m]), y[:m], reduction="none")
        return pre, post

    def train_expert(self, eid: int, x: Tensor, y: Tensor, replay: bool = True, project: bool = False) -> float:
        rec, c = self.experts[eid], self.cfg
        if c.growth_rule == "gap" and c.gap_mode == "one_step" and c.grow and rec.state == "active":
            rec.meta["probe"] = self._one_step_probe(rec, x, y)                       # taken from the weights BEFORE the inner steps
        loss = super().train_expert(eid, x, y, replay, project)
        self._lda_absorb(rec, x, y)
        return loss

    def _lda_absorb(self, rec, x: Tensor, y: Tensor) -> None:
        c = self.cfg
        if c.lda_blend > 0.0:
            head = rec.meta.get("lda")
            if head is None:
                head = rec.meta["lda"] = HD.LdaHead(c.d, c.n_classes, c.lda_shrink)
            head.update(F.normalize(x.float(), dim=1), y)

    def _absorb_only(self, eid: int, x: Tensor, y: Tensor) -> None:
        """A skipped batch (E-063) still enters the expert's closed-form head: it is one cheap pass and the head never forgets."""
        super()._absorb_only(eid, x, y)
        self._lda_absorb(self.experts[eid], x, y)

    def _gap_growth_step(self, rec, x: Tensor, y: Tensor, pre: Tensor) -> None:
        """The shipped decision (mature, gap below the threshold, cooled down, room under the cap and the budget: widen) with the gap read from the one-step probe."""
        if self.cfg.gap_mode != "one_step":
            return super()._gap_growth_step(rec, x, y, pre)
        probe = rec.meta.pop("probe", None)
        cfg, stats = self.cfg, self._grow_stats()
        if probe is None or not cfg.grow or rec.state != "active":
            return
        pre1, post1 = probe
        stats["post_rows"] += int(post1.numel())
        stats["train_rows"] += int(y.numel())
        mon = rec.meta.get("gap")
        if mon is None:
            mon = rec.meta["gap"] = GapMonitor(cfg.grow_window)
        mon.add(pre1, post1)
        guard = rec.meta.get("guard")
        if guard is not None:
            guard.losses.append(float(pre1.mean()))
            if len(guard.losses) >= cfg.grow_guard_window:
                self._close_guard(rec, guard, mon)
            return
        gap = mon.gap()
        if gap is None or gap >= cfg.grow_gap or not mon.mature(cfg.grow_min_age, cfg.grow_plateau) or not mon.cooled(rec.meta.get("grow_cooldown", cfg.grow_cooldown)):
            return
        k = widen_increment(rec.module.hidden, cfg.grow_factor, self.grow_cfg.max_hidden, self.grow_cfg.min_widen)
        if k == 0:
            stats["blocked_cap"] += 1
            return
        added = params_of_widen(rec.module.d_in, len(rec.module.blocks), k)
        if cfg.grow_budget_params is not None and self._live_params() + added > cfg.grow_budget_params:
            stats["blocked_budget"] += 1
            return
        self._widen_expert(rec, mon, k, added, gap)

    # ---- answering
    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        pred = super().predict(x, ctx)
        c = self.cfg
        if c.lda_blend == 0.0 and self.domain_lda is None:
            return pred
        xn = self._normalize(x.float())
        probs = pred.probs.clone()
        if c.lda_blend > 0.0:
            for e in pred.expert.unique().tolist():
                head = self.experts[e].meta.get("lda") if e in self.experts else None
                if head is not None and head.total > 0:
                    rows = (pred.expert == e).nonzero().flatten()
                    probs[rows] = blend(probs[rows], head.proba(xn[rows]), c.lda_blend)
        if self.domain_lda is not None and self.domain_lda.total > 0:
            probs = fuse_domain(probs, self.domain_lda.proba(xn) @ self._dom_M, c.domain_blocks, c.domain_head, c.domain_beta)
        return Prediction(probs=probs, expert=pred.expert, p_nov=pred.p_nov, unknown=pred.unknown, meta=pred.meta)

    # ---- accounting
    def extra_cost(self, d: int):
        params, macs, scalars = super().extra_cost(d)
        heads: List[HD.LdaHead] = [r.meta["lda"] for r in self.experts.values() if "lda" in r.meta]
        if self.domain_lda is not None:
            heads.append(self.domain_lda)
        return params, macs + len(heads) * d * d // 4, scalars + sum(h.n_scalars() for h in heads)
