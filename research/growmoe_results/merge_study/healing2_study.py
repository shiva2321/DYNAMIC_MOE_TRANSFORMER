"""Merge study v2, end to end (throwaway; no product file touched).

Why a v2: healing_study.py (v1) injected its false spawns at batches 6 and 9 of every 10-batch regime. healing_diag.py showed what that did: the batch-9 newborn was still an
INFANT when the next regime began, an infant accepts every input (conformal p-value floor 1/65 x infant_relax 4 = 0.06 > q = 0.02), so the pool saw no novelty, never founded the next
regime's expert by its own route (natural spawns 1 of 12) and the injected experts became the regime experts. v1's "-20 points" therefore measures a blind pool, not duplicates.

v2 injects early (batch 3, and optionally 4): the regime's own expert already exists (founded at batch 1 by the pool's own route, as in the clean run) and every newborn is a mature
expert (infancy = 5 observe calls) well before the regime ends. Arms (all with the same stream, seeds and injections):

    clean      no injection, no merge                              what the pool does when nothing goes wrong
    none       injections, no merge                                today's reality (the shipped merge repairs nothing)
    shipped    injections, the shipped _attempt_merges             (retrains the survivor, validates on memorised halves)
    preq       injections, redirect merge, prequential reference   retire j when i predicts j's stored samples as well as j honestly did (v1's 'redirect')
    hold       injections, redirect merge, paired non-inferiority on HELD-OUT samples: 1 in `hold_every` samples is withheld from training of every expert and filed under the
               expert the pool routes it to; retire i in favour of j when j's loss on i's held-out samples is not worse than i's own by more than a margin (one-sided, paired)
    agree      injections, label-free: retire the younger of two experts that agree on >= 80 % of the union of their stored prototypes
    none_h     injections, no merge, same withholding as `hold` (control: what withholding alone costs)
    clean_h    no injection, no merge, same withholding (control)

With --cap N the pool may hold at most N experts: the arms then show the silent stop.

    python healing2_study.py --inject 3 --seeds 0,1,2 --tag a
"""

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from growmoe.config import PhyloCfg  # noqa: E402
from growmoe.fairfight import _Scorer  # noqa: E402
from growmoe.pools.phylo import PhyloMoE  # noqa: E402
from growmoe.replay import Reservoir  # noqa: E402
from growmoe.settest import generator  # noqa: E402
from growmoe.streams_m1 import function_regimes  # noqa: E402
from growmoe.transfer import cluster_conflict  # noqa: E402

REGIMES, D, C = 12, 32, 4
HOLD_ARMS = ("clean_h", "none_h", "hold", "hold_g", "hold_soup", "hold_distill", "hold_evict")
RIVAL_ARMS = ("rival", "rival_ft", "rival_g")
NO_INJECT = ("clean", "clean_h", "clean_ens")
CAP_POLICY = {"cap_drop": "drop", "cap_extend": "extend", "cap_evict": "evict", "cap_evict_lru": "evict_lru", "cap_smart": "smart", "hold_evict": "evict"}       # what a full pool does with a cluster that would found an expert
NO_MERGE = ("clean", "none", "clean_h", "none_h", "ens", "clean_ens", "idshift") + tuple(a for a in CAP_POLICY if a != "hold_evict")
MODE = {"shipped": "shipped", "preq": "preq", "hold": "hold", "hold_g": "hold", "hold_soup": "hold", "hold_distill": "hold", "hold_evict": "hold", "agree": "agree", "agree50": "agree", "rival": "rival", "rival_ft": "rival", "rival_g": "rival"}
HOLD_CAP = 256
RIVAL_WIN, RIVAL_MIN = 512, 96


class RepairPhylo(PhyloMoE):
    """PhyloMoE whose merge can be replaced: shipped (unchanged), preq / hold (redirect: retire an expert another expert already serves as well as it does), agree (label-free)."""

    def __init__(self, cfg, mode="shipped", margin_k=0.10, z=2.33, min_hold=32, agree_min=0.80, ft_steps=0, abs_margin=None, consolidate=None, cap=None, cap_policy=None):
        super().__init__(cfg)
        self.mode, self.margin_k, self.z, self.min_hold, self.agree_min, self.ft_steps = mode, margin_k, z, min_hold, agree_min, ft_steps
        self.cap, self.cap_policy, self.cap_events, self.evicted = cap, cap_policy, 0, []          # the study's own cap (cfg.max_experts stays at 256 so the product cap never fires) and what it does
        self.consolidate = consolidate                                                # None | "soup" | "distill": what the survivor does with the retiree's knowledge (validated on held-out samples)
        self.abs_margin = abs_margin                                                  # None, or: the survivor must also beat the chance-level loss ln K on the retiree's samples by this much
        self.hold, self.healed, self.regime_of, self.rival_stat, self.rival_abs, self.consolidated = {}, [], {}, {}, {}, []

    # ------------------------------------------------------------------ a full pool: drop the cluster (today), force an extend into the nearest expert, or evict one expert and found the new one
    def _handle_cluster(self, x_taken, y_taken, evidence=0.0, force_parent=None, novel_indices=None, x_all=None, assigned=None, spawned_list=None):
        if self.cap_policy and self.cap and len(self.experts) >= self.cap:
            self.cap_events += 1
            if self.cap_policy == "drop":
                return None                                                           # what the shipped cap does, silently
            if self.cap_policy == "extend":
                parent = force_parent if force_parent in self.experts else max(self.experts, key=lambda e: self.experts[e].meta["sig"].affinity(x_taken, seed=self.step))
                self._extend_unchanged(parent, 0.0, x_taken, y_taken, _NoFields(), novel_indices, x_all, assigned)
                return None
            victim = self._victim()
            if victim is None:                                                         # smart: nothing is redundant, so share an expert instead of forgetting one
                parent = force_parent if force_parent in self.experts else max(self.experts, key=lambda e: self.experts[e].meta["sig"].affinity(x_taken, seed=self.step))
                self._extend_unchanged(parent, 0.0, x_taken, y_taken, _NoFields(), novel_indices, x_all, assigned)
                return None
            self._evict(victim)
            force_parent = None if force_parent == victim else force_parent
        return super()._handle_cluster(x_taken, y_taken, evidence=evidence, force_parent=force_parent, novel_indices=novel_indices, x_all=x_all, assigned=assigned, spawned_list=spawned_list)

    def _victim(self) -> int:
        """evict: the most redundant expert (the largest mutual signature coverage with another expert; the less used of the two), the least recently used one if none overlaps; evict_lru: the least recently used."""
        eids = list(self.experts)
        if self.cap_policy in ("evict", "smart"):
            best = None
            for a in eids:
                for b in eids:
                    if a == b:
                        continue
                    ov = min(float(self.experts[a].meta["sig"].pvalue(self.experts[b].meta["sig"].protos).mean()), float(self.experts[b].meta["sig"].pvalue(self.experts[a].meta["sig"].protos).mean()))
                    if ov >= self.cfg.merge_overlap:
                        loser = a if (self.experts[a].n_assigned, a) <= (self.experts[b].n_assigned, b) else b
                        if best is None or ov > best[0]:
                            best = (ov, loser)
            if best is not None:
                return best[1]
            if self.cap_policy == "smart":
                return None
        return min(eids, key=lambda e: (self.experts[e].last_active, self.experts[e].n_assigned, e))

    def _evict(self, victim: int) -> None:
        rv = self.experts[victim]
        for c_id, c_rec in list(self.experts.items()):
            if c_rec.parent == victim:
                self.set_parent(c_id, rv.parent)
        for k_id, k_rec in list(self.experts.items()):
            if k_rec.meta.get("supersedes") == victim:
                k_rec.meta.pop("supersedes", None)
        self.hold.pop(victim, None)
        for key in [k for k in self.rival_stat if victim in k]:
            del self.rival_stat[key]
            self.rival_abs.pop(key, None)
        self.remove_expert(victim)
        self.evicted.append((self.step, victim))
        self.events.add(kind="evict", step=self.step, expert=victim)

    # ------------------------------------------------------------------ rival tracking: the nearest rivals of an expert are scored on the batches routed to it, BEFORE anybody trains on them
    def rival_observe(self, x, y) -> None:
        """Call before ``observe``: for every expert i that receives samples of this batch, each rival j (j's signature covers i's region) is scored on the same samples. The samples are
        new to both, so loss_j - loss_i is an honest, paired, current difference; it costs one forward pass per rival and withholds nothing from training."""
        eids = [e for e, r in self.experts.items() if r.state != "infant"]
        if len(eids) < 2:
            return
        with torch.no_grad():
            pr = self.predict(x)
            owner, known = pr.expert, ~pr.unknown                                  # samples the gate flags as novel are not evidence about any rival
            xn = self._normalize(x.float())
            for i in owner.unique().tolist():
                if i not in eids:
                    continue
                rows = ((owner == i) & known).nonzero().squeeze(1)
                if rows.numel() == 0:
                    continue
                xi, yi = xn[rows], y[rows]
                ri = self.experts[i]
                li = F.cross_entropy(ri.module(xi), yi, reduction="none")
                for j in eids:
                    if j == i or float(self.experts[j].meta["sig"].pvalue(ri.meta["sig"].protos).mean()) < self.cfg.merge_overlap:
                        continue
                    lj = F.cross_entropy(self.experts[j].module(xi), yi, reduction="none")
                    self.rival_stat.setdefault((i, j), deque(maxlen=RIVAL_WIN)).extend((lj - li).tolist())
                    self.rival_abs.setdefault((i, j), deque(maxlen=RIVAL_WIN)).extend(lj.tolist())

    def _redundant_rival(self):
        best = None
        for (i, j), dq in self.rival_stat.items():
            if i not in self.experts or j not in self.experts or len(dq) < RIVAL_MIN:
                continue
            if self.experts[i].state == "infant" or self.experts[j].state == "infant":
                continue
            d = np.asarray(dq, dtype=np.float64)
            m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
            margin = self.margin_k * math.log(max(int(self.experts[i].module.seen.sum()), 2))
            if self.abs_margin is not None and float(np.mean(self.rival_abs[(i, j)])) > math.log(max(int(self.experts[i].module.seen.sum()), 2)) - self.abs_margin:
                continue                                                                    # 'not worse than i' is vacuous when i itself knows nothing: the survivor must be competent
            if m + self.z * se < margin and (best is None or m < best[0]):
                best = (m, j, i)
        return None if best is None else (best[1], best[2])

    # ------------------------------------------------------------------ held-out samples (never trained on by anyone)
    def add_holdout(self, x, y) -> None:
        if not self.experts:
            return
        with torch.no_grad():
            owner = self.predict(x).expert
        xn = self._normalize(x.float())
        for e in owner.unique().tolist():
            m = owner == e
            res = self.hold.setdefault(int(e), Reservoir(HOLD_CAP, self.cfg.d, np.random.default_rng([self.cfg.seed, 0x484C44, int(e)])))
            res.add(xn[m], y[m])

    # ------------------------------------------------------------------ the three candidate tests; each returns (keeper, retired) or None
    def _redundant_preq(self):
        cfg, best = self.cfg, None
        ok = [e for e, r in self.experts.items() if r.state != "infant" and r.replay is not None and r.replay.n >= 64 and len(r.meta.get("ref", ())) >= cfg.transfer_min_ref]
        for i in ok:
            for j in ok:
                if i == j:
                    continue
                ri, rj = self.experts[i], self.experts[j]
                if float(ri.meta["sig"].pvalue(rj.meta["sig"].protos).mean()) < cfg.merge_overlap:
                    continue
                rx, ry = rj.replay.data
                with torch.no_grad():
                    losses = F.cross_entropy(ri.module(rx), ry, reduction="none")
                ref = torch.tensor(list(rj.meta["ref"]))
                margin = self.margin_k * math.log(max(int(rj.module.seen.sum()), 2))
                conflict, p, diff = cluster_conflict(losses, ref, cfg.transfer_alpha, cfg.transfer_perms, margin, generator(cfg.seed, 0x4D47, self.step, i, j))
                if conflict:
                    continue
                score = (float(np.mean(ri.meta["ref"])), i)
                if best is None or (diff, score) < best[0]:
                    best = ((diff, score), i, j)
        return None if best is None else (best[1], best[2])

    def _redundant_hold(self):
        cfg, best = self.cfg, None
        ok = [e for e, r in self.experts.items() if r.state != "infant"]
        for i in ok:                                                                        # i = the candidate to retire
            r = self.hold.get(i)
            if r is None or r.n < self.min_hold:
                continue
            hx, hy = r.data
            ri = self.experts[i]
            with torch.no_grad():
                li = F.cross_entropy(ri.module(hx), hy, reduction="none")
            margin = self.margin_k * math.log(max(int(ri.module.seen.sum()), 2))
            for j in ok:                                                                    # j = the expert that would serve i's inputs from now on
                if j == i:
                    continue
                rj = self.experts[j]
                if float(rj.meta["sig"].pvalue(ri.meta["sig"].protos).mean()) < cfg.merge_overlap:
                    continue
                with torch.no_grad():
                    lj = F.cross_entropy(rj.module(hx), hy, reduction="none")
                if self.abs_margin is not None and float(lj.mean()) > math.log(max(int(ri.module.seen.sum()), 2)) - self.abs_margin:
                    continue                                                                # 'not worse than i' is vacuous when i itself knows nothing: the survivor must be competent
                d = lj - li                                                                 # paired: both models, same never-trained samples, same moment
                m, se = float(d.mean()), float(d.std() / math.sqrt(len(d)))
                if m + self.z * se < margin and (best is None or m < best[0]):
                    best = (m, j, i)
        return None if best is None else (best[1], best[2])

    def _redundant_agree(self):
        cfg, best = self.cfg, None
        ok = [e for e, r in self.experts.items() if r.state != "infant"]
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                ri, rj = self.experts[ok[a]], self.experts[ok[b]]
                if min(float(ri.meta["sig"].pvalue(rj.meta["sig"].protos).mean()), float(rj.meta["sig"].pvalue(ri.meta["sig"].protos).mean())) < cfg.merge_overlap:
                    continue
                X = torch.cat([ri.meta["sig"].protos, rj.meta["sig"].protos])
                with torch.no_grad():
                    ag = float((ri.module(X).argmax(1) == rj.module(X).argmax(1)).float().mean())
                if ag >= self.agree_min and (best is None or ag > best[0]):
                    keeper, retired = (ok[a], ok[b]) if ri.n_assigned >= rj.n_assigned else (ok[b], ok[a])
                    best = (ag, keeper, retired)
        return None if best is None else (best[1], best[2])

    def _attempt_merges(self) -> None:
        if self.mode == "shipped":
            before = set(self.experts)
            super()._attempt_merges()
            for j in before - set(self.experts):
                self.healed.append((self.step, None, j))
            return
        test = {"preq": self._redundant_preq, "hold": self._redundant_hold, "agree": self._redundant_agree, "rival": self._redundant_rival}[self.mode]
        for _ in range(4):
            pair = test()
            if pair is None:
                return
            self._absorb(*pair)

    def _absorb(self, i: int, j: int) -> None:
        """Retire j in favour of i; no weight is touched: j's signature, replay, reference and held-out samples move into i."""
        ri, rj = self.experts[i], self.experts[j]
        if self.consolidate:
            self._consolidate(i, j)
        ri.meta["sig"].merge_from(rj.meta["sig"])
        ri.meta["sig"].fit()
        if rj.replay is not None and rj.replay.n:
            ri.replay.add(*rj.replay.data)
        if rj.meta.get("windows") is not None:
            self._windows_of(ri).merge_from(rj.meta["windows"])
        self._merge_refs(ri, rj)
        if j in self.hold:
            hj = self.hold.pop(j)
            if hj.n:
                self.hold.setdefault(i, Reservoir(HOLD_CAP, self.cfg.d, np.random.default_rng([self.cfg.seed, 0x484C44, i]))).add(*hj.data)
        for c_id, c_rec in list(self.experts.items()):
            if c_rec.parent == j:
                self.set_parent(c_id, rj.parent)                                             # the retired expert's children take its place in the lineage (the survivor may be one of their descendants)
        for k_id, k_rec in list(self.experts.items()):
            if k_rec.meta.get("supersedes") == j:
                k_rec.meta.pop("supersedes", None)
        self.remove_expert(j)
        for key in [k for k in self.rival_stat if j in k or i in k]:                  # the statistics of both describe experts that no longer exist in this form
            del self.rival_stat[key]
            self.rival_abs.pop(key, None)
        if self.ft_steps and ri.replay is not None and ri.replay.n >= 32:             # consolidation: the survivor now also trains on the retiree's stored samples (a few steps)
            for _ in range(self.ft_steps):
                bx, by = ri.replay.sample(min(64, ri.replay.n))
                ri.module.train_step(bx, by, ri.opt, replay=None)
        self.healed.append((self.step, i, j))
        self.events.add(kind="merge", step=self.step, kept=i, removed=j, change_kept=0.0, change_removed=0.0)


def _consolidate(self, i: int, j: int) -> None:
    """The survivor i takes over j's knowledge. soup: the midpoint of the two weight vectors; distill: i is trained to reproduce the AVERAGE answer of both on the union of the two experts'
    stored prototypes (inputs only). Either candidate replaces i only if it is not worse on the held-out samples of both experts (never trained on by anyone)."""
    ri, rj = self.experts[i], self.experts[j]
    held = [self.hold[e].data for e in (i, j) if e in self.hold and self.hold[e].n >= 8]
    if not held:
        return
    hx, hy = torch.cat([h[0] for h in held]), torch.cat([h[1] for h in held])
    cand = ri.module.clone()
    if self.consolidate == "soup":
        with torch.no_grad():
            for pm, pa, pb in zip(cand.parameters(), ri.module.parameters(), rj.module.parameters()):
                pm.copy_(0.5 * (pa + pb))
    else:
        xs = torch.cat([ri.meta["sig"].protos, rj.meta["sig"].protos])
        with torch.no_grad():
            target = 0.5 * (ri.module.proba(xs) + rj.module.proba(xs))
        opt = torch.optim.Adam(cand.parameters(), lr=3e-3)
        for s_idx in range(100):
            sel = torch.randint(0, len(xs), (min(64, len(xs)),), generator=torch.Generator().manual_seed(self.cfg.seed + 7919 * self.step + s_idx))
            loss = F.kl_div(F.log_softmax(cand(xs[sel]), dim=-1), target[sel], reduction="batchmean")
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        better = float(F.cross_entropy(cand(hx), hy)) <= float(F.cross_entropy(ri.module(hx), hy))
    self.consolidated.append((self.step, i, j, bool(better)))
    if better:
        ri.module.load_state_dict(cand.state_dict())
        ri.opt = self.make_optimizer(ri.module, ri.lr_mult)


RepairPhylo._consolidate = _consolidate


def overlap_matrix(pool, eids):
    K = len(eids)
    O = torch.ones(K, K)
    for a, i in enumerate(eids):
        for b, j in enumerate(eids):
            if a != b:
                O[a, b] = float(pool.experts[i].meta["sig"].pvalue(pool.experts[j].meta["sig"].protos).mean())
    return torch.minimum(O, O.T)


def predict_ens(pool, x):
    """Answer-time only (no training effect): the sample goes to the expert the pool would choose, and its prediction is averaged with every OTHER eligible expert whose signature overlaps
    that expert's (twins)."""
    xn = pool._normalize(x.float())
    eids = list(pool.experts)
    P, LogF = pool._gate_matrices(xn, eids)
    elig = P.double() >= float(pool.cfg.q)
    pick = torch.where(elig.any(1), LogF.masked_fill(~elig, float("-inf")).argmax(1), LogF.argmax(1))
    twin = overlap_matrix(pool, eids) >= pool.cfg.merge_overlap
    member = twin[pick] & (elig | F.one_hot(pick, len(eids)).bool())
    probs = torch.zeros(len(xn), pool.cfg.n_classes)
    with torch.no_grad():
        for c, e in enumerate(eids):
            rows = member[:, c].nonzero().squeeze(1)
            if rows.numel():
                probs[rows] += pool.experts[e].module.proba(xn[rows])
    probs /= member.sum(1, keepdim=True).clamp_min(1)
    return probs, torch.tensor(eids)[pick]


class _NoFields:
    def fields(self) -> dict:
        return {}


POOL_SEED_OFFSET = 7        # the pool's seed is the stream's seed + 7. With offset 0 the expert founded for regime r starts with the teacher's own hidden units (MLPExpert(seed = s*1000 + id) against the
                            # teacher MLPExpert(seed = s*1000 + r)), a head start of 4-7 points that only pools whose ids line up with the regimes enjoy. All results with a tag WITHOUT the prefix "n_" were made with offset 0.


def make_pool(arm: str, seed: int, cap):
    policy = CAP_POLICY.get(arm)
    cfg = PhyloCfg(d=D, hidden=32, n_classes=C, seed=seed + POOL_SEED_OFFSET, q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=5, route_rule="accept_then_logf", set_gate=False, group_vote=False,
                   spawn_rule="transfer", transfer_min_ref=64, merge_every=(10 ** 9 if arm in NO_MERGE else 10), max_experts=256 if policy else (cap if cap else 256))
    return RepairPhylo(cfg, mode=MODE.get(arm, "shipped"), agree_min=0.5 if arm == "agree50" else 0.8, ft_steps=20 if arm == "rival_ft" else 0, abs_margin=0.2 if arm.endswith("_g") else None,
                       consolidate={"hold_soup": "soup", "hold_distill": "distill"}.get(arm), cap=cap if policy else None, cap_policy=policy)


def inject(pool, x, y) -> list:
    """Found a duplicate: hand the batch to _handle_cluster with the probe told it may never extend (the way the merge audit forced one). Returns the new expert ids."""
    saved = pool.cfg.spawn_rule, pool.cfg.delta_forget
    pool.cfg.spawn_rule, pool.cfg.delta_forget = "probe", -1.0
    before = set(pool.experts)
    try:
        pool._handle_cluster(x, y, evidence=50.0)
    finally:
        pool.cfg.spawn_rule, pool.cfg.delta_forget = saved
    return sorted(set(pool.experts) - before)


def run(arm: str, seed: int, inject_at=(3,), cap=None, hold_every: int = 8, n_train: int = 640) -> dict:
    t0 = time.process_time()
    st = function_regimes(seed, n_regimes=REGIMES, d=D, n_classes=C, n_train=n_train, n_test=200)
    pool = make_pool(arm, seed, cap)
    natural, injected, skipped = [], [], 0
    last_k, within, seg_natural = None, 0, 0
    counts, cap_segment = [], None
    first_visit_of, regime_of_seg = {}, {}
    for k, seg, idx in st._walk(64, seed):
        if k != last_k:
            if last_k is not None:
                counts.append(len(pool.experts))
            last_k, within, seg_natural = k, 0, 0
            if seg.first_visit and seg.regime not in first_visit_of:
                first_visit_of[seg.regime] = k
            regime_of_seg[k] = seg.regime
        within += 1
        x, y = seg.x[idx], seg.y[idx]
        x_ho = y_ho = None
        if arm in HOLD_ARMS and len(pool.experts) > 0:
            mask = torch.zeros(len(y), dtype=torch.bool)
            mask[::hold_every] = True
            x, y, x_ho, y_ho = x[~mask], y[~mask], x[mask], y[mask]
        before = set(pool.experts)
        if arm in RIVAL_ARMS:
            pool.rival_observe(x, y)
        pool.observe(x, y)
        new = sorted(set(pool.experts) - before)
        for e in new:
            natural.append(e)
            pool.regime_of[e] = seg.regime
        seg_natural += len(new)
        if x_ho is not None:
            pool.add_holdout(x_ho, y_ho)
        if arm not in NO_INJECT and seg.first_visit and within in inject_at and len(pool.experts) > 0 and not (cap and len(pool.experts) >= cap):
            if seg_natural == 0:                                                        # the regime has no expert of its own yet: a forced spawn would not be a duplicate
                skipped += 1
            elif arm == "idshift":                                                      # control: the expert ids move on exactly as they do with an injection, but nothing is founded
                pool._next_id += 1
            else:
                new = inject(pool, x, y)
                injected += new
                for e in new:
                    pool.regime_of[e] = seg.regime
        if cap and cap_segment is None and len(pool.experts) >= cap:
            cap_segment = k
    counts.append(len(pool.experts))
    # routing audit on the held-out test sets: who serves each regime?
    inj = set(injected)
    served, share_inj, acc_inj, acc_nat, n_inj, n_nat = {}, 0.0, 0, 0, 0, 0
    tot = 0
    row = np.zeros(REGIMES)
    for r, (xt, yt) in st.test_sets().items():
        with torch.no_grad():
            if arm in ("ens", "clean_ens"):
                probs, owner_t = predict_ens(pool, xt)
            else:
                pr = pool.predict(xt)
                probs, owner_t = pr.probs, pr.expert
        owner = owner_t.tolist()
        ok = (probs.argmax(1).cpu() == yt.cpu()).tolist()
        row[r] = float(np.mean(ok))
        served[r] = sorted({e for e in set(owner) if owner.count(e) >= 0.05 * len(owner)})
        for e, hit in zip(owner, ok):
            tot += 1
            if e in inj:
                n_inj += 1
                acc_inj += hit
            else:
                n_nat += 1
                acc_nat += hit
    first = sorted(first_visit_of.items(), key=lambda kv: kv[1])
    late = [r for r, k in first if cap_segment is not None and k > cap_segment]
    own_expert = {r: any(pool.regime_of.get(e) == r for e in natural) for r, _ in first}
    healed = [{"step": s, "keeper": i, "retired": j, "same_regime": (None if i is None else pool.regime_of.get(i) == pool.regime_of.get(j)), "retired_injected": j in inj,
               "keeper_injected": (i in inj) if i is not None else None} for s, i, j in pool.healed]
    return {"arm": arm, "seed": seed, "pool_seed_offset": POOL_SEED_OFFSET, "inject_at": list(inject_at), "cap": cap, "n_train": n_train, "hold_every": hold_every if arm in HOLD_ARMS else None,
            "final_experts": len(pool.experts), "experts_after_each_segment": counts, "natural_founded": len(natural), "injected": len(injected), "skipped_injections": skipped,
            "injected_alive": len([e for e in injected if e in pool.experts]), "natural_alive": len([e for e in natural if e in pool.experts]),
            "merges": len(healed), "merges_cross_regime": sum(1 for h in healed if h["same_regime"] is False), "healed": healed,
            "acc": float(row.mean()), "per_regime": row.tolist(), "regimes": REGIMES,
            "share_routed_to_injected": n_inj / max(tot, 1), "acc_when_injected": acc_inj / n_inj if n_inj else None, "acc_when_natural": acc_nat / n_nat if n_nat else None,
            "regimes_served_by_several": sum(1 for v in served.values() if len(v) > 1), "regimes_without_own_expert": sum(1 for v in own_expert.values() if not v),
            "consolidations": [list(c) for c in pool.consolidated], "cap_events": pool.cap_events, "evicted": len(pool.evicted), "evicted_were_injected": sum(1 for _, e in pool.evicted if e in inj),
            "cap_reached_in_segment": cap_segment, "late_regimes": late, "acc_late": float(np.mean([row[r] for r in late])) if late else None,
            "cpu_seconds": time.process_time() - t0}


def write(path: Path, res: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(res, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def table(res: list, arms: list) -> str:
    def ms(v):
        v = [x for x in v if x is not None]
        if not v:
            return "-"
        return f"{np.mean(v):.1f}" if len(v) == 1 else f"{np.mean(v):.1f} +-{np.std(v, ddof=1) / math.sqrt(len(v)):.1f}"

    lines = ["arm        seeds  accuracy        experts      injected alive  merges (cross-regime)  regimes served by >1  routed to injected  acc late  cpu s",
             "---------  -----  --------------  -----------  --------------  ---------------------  --------------------  ------------------  --------  -----"]
    for a in arms:
        rs = [r for r in res if r["arm"] == a]
        if not rs:
            continue
        lines.append(f"{a:9s}  {len(rs):5d}  {ms([100 * r['acc'] for r in rs]):14s}  {ms([r['final_experts'] for r in rs]):11s}  {ms([r['injected_alive'] for r in rs]):14s}  "
                     f"{ms([r['merges'] for r in rs]):>9s} ({sum(r['merges_cross_regime'] for r in rs)})         {ms([r['regimes_served_by_several'] for r in rs]):20s}  "
                     f"{ms([100 * r['share_routed_to_injected'] for r in rs]):18s}  {ms([100 * r['acc_late'] if r['acc_late'] is not None else None for r in rs]):8s}  {ms([r['cpu_seconds'] for r in rs])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inject", default="3")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--arms", default="clean,none,shipped,preq,hold,agree,none_h,clean_h")
    ap.add_argument("--cap", type=int, default=None)
    ap.add_argument("--hold-every", type=int, default=8)
    ap.add_argument("--n-train", type=int, default=640)
    ap.add_argument("--tag", default="a")
    args = ap.parse_args()
    torch.set_num_threads(2)
    inject_at = () if args.inject == "none" else tuple(int(v) for v in args.inject.split(","))
    seeds = [int(v) for v in args.seeds.split(",")]
    arms = args.arms.split(",")
    out = HERE / f"healing2_{args.tag}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    done = {(r["arm"], r["seed"]) for r in res}
    for seed in seeds:
        for arm in arms:
            if (arm, seed) in done:
                continue
            res.append(run(arm, seed, inject_at, args.cap, args.hold_every, args.n_train))
            write(out, res)
            r = res[-1]
            print(f"seed {seed} {arm:8s} experts {r['final_experts']:3d} (natural {r['natural_alive']:2d}, injected alive {r['injected_alive']:2d}/{r['injected']:2d}, merges {r['merges']:2d}, cross {r['merges_cross_regime']}) "
                  f"acc {100 * r['acc']:.1f}  [{r['cpu_seconds']:.0f}s]", flush=True)
    print("\n" + table(res, arms))


if __name__ == "__main__":
    main()
