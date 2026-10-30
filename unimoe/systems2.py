"""The pool with the Stage 1 follow-up changes, by name (``pools.UniPool`` underneath; every variant is U-real plus switches):

* ``UG0``   U-real exactly (the control: UniPool with all switches off);
* ``UG1``   growth read from the one-step probe, with the maturity condition scaled to the stream (30 batches of age, windows of 20, cooldown of 20);
* ``UDOM`` / ``UDOMS``  a domain head from the label blocks, fused hard / soft;
* ``ULDA``  a running LDA head inside every expert, blended into its answer;
* ``UPLUS`` all of them;
* ``oracle_lda``  experts told the regime, each with an LDA head (the ceiling test of the LDA blend, free of routing).
The stream's label blocks are passed in (``blocks``) because the pool never sees domain ids, only labels."""

from typing import Optional, Sequence

import torch

from growmoe.config import PhyloCfg

from . import systems as SY
from .heads import LdaHead
from .pools import UniCfg, UniPool, blend

NEW = ("UG0", "UG1", "UDOM", "UDOMS", "ULDA", "UPLUS")
FAST = {"grow_min_age": 30, "grow_window": 20, "grow_cooldown": 20}
GROWTH = {"gap_mode": "one_step", "growth_rule": "gap", **FAST}
DOMAIN = {"domain_head": "hard"}
LDA = {"lda_blend": 0.5}
FLAGS = {"UG0": {}, "UG1": GROWTH, "UDOM": DOMAIN, "UDOMS": {"domain_head": "soft"}, "ULDA": LDA, "UPLUS": {**GROWTH, **DOMAIN, **LDA}}


def uni_cfg(name: str, n_classes: int, seed: int, recipe: Optional[dict] = None, blocks: Sequence = (), d: int = 128) -> UniCfg:
    if name not in FLAGS:
        raise ValueError(f"{name!r} is not one of {list(FLAGS)}")
    kw = dict(d=d, n_classes=n_classes, route_rule="accept_then_logf", seed=seed + 7, **{**SY.VARIANT_SWITCHES["U-real"], **FLAGS[name]})
    if FLAGS[name].get("domain_head", "off") != "off":
        kw["domain_blocks"] = tuple(tuple(b) for b in blocks)
    kw.update({k: v for k, v in (recipe or {}).items() if k in UniCfg.__dataclass_fields__})
    return UniCfg(**kw)


class OracleLdaExperts(SY.OracleExperts):
    """OracleExperts plus a running LDA head per regime, blended into each expert's answer with weight ``lda_blend``."""
    name = "oracle_lda(upper-bound)"

    def __init__(self, cfg, lda_blend: float, shrink: float = 0.1) -> None:
        super().__init__(cfg)
        self.lda_blend, self.shrink = lda_blend, shrink
        self.heads = {}

    def observe(self, x, y=None, ctx=None):
        rec = super().observe(x, y, ctx)
        if y is not None:
            for r in ctx.unique().tolist():
                m = ctx == r
                self.heads.setdefault(r, LdaHead(self.cfg.d, self.cfg.n_classes, self.shrink)).update(x[m], y[m])
        return rec

    def predict(self, x, ctx=None):
        pred = super().predict(x, ctx)
        if self.lda_blend > 0.0:
            probs = pred.probs.clone()
            for r in ctx.unique().tolist():
                if r in self.heads and self.heads[r].total > 0:
                    m = ctx == r
                    probs[m] = blend(probs[m], self.heads[r].proba(x[m]), self.lda_blend)
            pred.probs = probs
        return pred


def make_system2(name: str, n_classes: int, seed: int, recipe: Optional[dict] = None, blocks: Sequence = (), d: int = 128):
    recipe = dict(recipe or {})
    if name == "oracle_lda":
        cfg = SY._common(n_classes, seed, d, recipe)
        return OracleLdaExperts(cfg, recipe.get("lda_blend", 0.5), recipe.get("lda_shrink", 0.1))
    return UniPool(uni_cfg(name, n_classes, seed, recipe, blocks, d))
