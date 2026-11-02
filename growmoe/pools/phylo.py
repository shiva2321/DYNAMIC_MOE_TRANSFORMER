"""PhyloMoE: calibrated conformal gate, vMF signatures, and phylogenetic inheritance.

Architecture A implementation of ContinualPool.
Stages:
- 13a: Bootstrap founding expert 0 from buffered samples once >= m_min.
- 13b: Gate and known/novel split using conformal p-values and vMF logpdf routing.
- 13c: Probation buffer integration, sub-critical clustering, and evidence test.
- 13d: Spawn from parent's DNA (affinity-based shrink-perturb, clone, or inherit).
- 13e: Interference probe (extend parent vs spawn child based on held-out replay and cluster gap).
- 13f: Infant protection (gate relaxation, learning rate boost, and lifecycle transition).
"""

from collections import deque
import math
import warnings
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..config import PhyloCfg, GrowthCfg
from ..conformal import benjamini_hochberg, conformal_pvalues
from ..experts import MLPExpert
from ..competence import LossChangeMonitor
from ..gapgrow import GapMonitor, GrowthGuard, guard_reverts, params_of_widen, widen_increment
from ..growth import GrowthPolicy, plateau
from ..probation import Probation
from ..signature import VmfSignature
from ..fingerprint import Fingerprint
from ..groupvote import vote_groups
from ..setgate import SetGate
from ..settest import generator
from ..sources import SourceWindows
from ..types import StepRecord, Prediction
from ..replay import Reservoir
from ..router import FITS, RouteRule, cv_owner_accuracy, select_rule
from ..transfer import TransferVerdict, cluster_conflict
from ..vmf import log_vmf_norm
from .base import BasePool


class PhyloMoE(BasePool):
    name = "A_phylomoe"

    def __init__(self, cfg: Optional[PhyloCfg] = None, grow_cfg: Optional[GrowthCfg] = None) -> None:
        cfg = cfg if cfg is not None else PhyloCfg()
        super().__init__(cfg)
        self.cfg: PhyloCfg = cfg
        self.grow_cfg: GrowthCfg = grow_cfg if grow_cfg is not None else GrowthCfg()

        # Bootstrap buffer for initial regime discovery
        self._boot_x: List[Tensor] = []
        self._boot_y: List[Tensor] = []

        # 13c Probation buffer
        self.probation = Probation(cap=self.cfg.pool_cap, d=self.cfg.d)
        self.samples_seen = 0
        self.sig_cls = getattr(self.cfg, "sig_cls", None) or VmfSignature
        self.reset_gate_stats()
        self._set = SetGate(self.cfg)                                  # routing v2 stage 3a; idle unless cfg.set_gate
        self._held: list = []                                          # persistence (A-15): samples of a doubtful batch that wait for the next decision
        self._held_step = -10
        self.skip_stats = {"trained": 0, "skipped": 0, "trained_rows": 0, "skipped_rows": 0}   # E-063: batches (and rows) that took optimiser steps / were skipped as already known; counters only
        self._route_state()                                            # routing v2 stage 4a; idle unless cfg.router
        self._grow_stats()                                             # routing v2 stage 4c; idle unless cfg.growth_rule == "gap"
        self._cap_state()                                              # stage 4.1: what the cap did; counters only
        self._infant_state()                                           # stage 4.1: samples the infant relaxation let through; counters only

    @staticmethod
    def _forget_competence_history(rec) -> None:
        """Drop what the competence monitor and its recent-sample window collected so far: the baseline must come from the settled expert."""
        monitor = rec.meta.get("monitor")
        if monitor is not None:
            monitor.reset()
        recent = rec.meta.get("recent")
        if recent is not None:
            recent.clear()

    def _new_sig(self, seed: int):
        """A fresh signature of the configured class. ``dedupe`` is passed only when ``sig_dedupe`` is on, so a custom ``sig_cls`` without that argument keeps working."""
        return self.sig_cls(d=self.cfg.d, max_protos=256, seed=seed, **({"dedupe": True} if self.cfg.sig_dedupe else {}))

    def _normalize(self, x: Tensor) -> Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x / norm

    def _batch_consistent(self, assigned: Tensor) -> Tensor:
        """Batch vote (routing v2, stage 1). When one expert holds at least ``batch_vote_share`` of the accepted samples of a batch (and at
        least ``batch_vote_min`` were accepted), every accepted sample trains that expert. Novel samples (-1) are untouched, ties go to the
        smallest id, and the input is not modified."""
        out = assigned.clone()
        accepted = out >= 0
        n_acc = int(accepted.sum())
        if n_acc < self.cfg.batch_vote_min:
            return out
        ids, counts = torch.unique(out[accepted], return_counts=True)
        top = int(counts.max())
        if top < self.cfg.batch_vote_share * n_acc:
            return out
        out[accepted] = ids[counts == top].min()
        return out

    def _filter_superseded(self, candidate_eids: List[int]) -> List[int]:
        superseded = set()
        for e in candidate_eids:
            succ = self.experts[e].meta.get("supersedes")
            while succ is not None:
                if succ in superseded or succ == e:
                    break
                superseded.add(succ)
                if succ in self.experts:
                    succ = self.experts[succ].meta.get("supersedes")
                else:
                    break
        active_candidates = [e for e in candidate_eids if e not in superseded]
        return active_candidates if active_candidates else candidate_eids

    def _handle_cluster(
        self,
        x_taken: Tensor,
        y_taken: Tensor,
        evidence: float = 0.0,
        force_parent: Optional[int] = None,
        novel_indices: Optional[Tensor] = None,
        x_all: Optional[Tensor] = None,
        assigned: Optional[Tensor] = None,
        spawned_list: Optional[List[int]] = None,
    ) -> Optional[int]:
        verdict = self._transfer_verdict(x_taken, y_taken, force_parent) if self.cfg.spawn_rule == "transfer" else None      # routing v2 stage 4b
        if verdict is not None:
            parent_eid = verdict.parent                                  # the best predictor of the new labels, not the nearest signature
            aff = self.experts[parent_eid].meta["sig"].affinity(x_taken, seed=self.step)
        elif force_parent is not None:
            parent_eid = force_parent
            aff = self.experts[parent_eid].meta["sig"].affinity(x_taken, seed=self.step)
        else:
            affinities = {}
            for eid, rec in self.experts.items():
                affinities[eid] = rec.meta["sig"].affinity(x_taken, seed=self.step)
            parent_eid = max(affinities, key=affinities.get)
            aff = affinities[parent_eid]

        parent_rec = self.experts[parent_eid]
        next_id = self._next_id

        valid_mask = y_taken >= 0
        has_labels = valid_mask.any()
        has_replay = parent_rec.replay is not None and parent_rec.replay.n >= 32
        has_affinity = aff >= self.cfg.probe_min_affinity

        extended = False
        trained_child = None

        if verdict is not None:                                          # routing v2 stage 4b: the labels decide, not a probe
            if not verdict.conflict:
                self._extend_unchanged(parent_eid, aff, x_taken, y_taken, verdict, novel_indices, x_all, assigned)
                return None
            trained_child = self._pretrain_child(parent_rec, aff, next_id, x_taken, y_taken)
        elif has_labels and has_affinity and has_replay:
            # 13e Interference Probe
            rx, ry = parent_rec.replay.data
            n_rep = parent_rec.replay.n
            gen_split = torch.Generator().manual_seed(self.cfg.seed + self.step * 100)
            perm = torch.randperm(n_rep, generator=gen_split)
            half = n_rep // 2
            r_fit_x, r_fit_y = rx[perm[:half]], ry[perm[:half]]
            r_probe_x, r_probe_y = rx[perm[half:]], ry[perm[half:]]

            if self.cfg.init_mode == "scratch":
                child_mod = MLPExpert(
                    d_in=self.cfg.d,
                    hidden=self.cfg.hidden,
                    depth=self.cfg.depth,
                    n_classes=self.cfg.n_classes,
                    seed=self.cfg.seed * 1000 + next_id,
                )
            elif self.cfg.init_mode == "clone":
                child_mod = parent_rec.module.clone()
            elif self.cfg.init_mode == "inherit":
                rho = self.cfg.rho_max * (1.0 - aff)
                rng_inh = np.random.default_rng([self.cfg.seed, 0x494E48, next_id])
                child_mod = parent_rec.module.inherit(
                    rho=rho,
                    sigma=self.cfg.inherit_sigma,
                    rng=rng_inh,
                )
            else:  # shrink_perturb
                child_mod = parent_rec.module.shrink_perturb(
                    lam=aff,
                    seed=self.cfg.seed * 1000 + next_id,
                )

            shadow = parent_rec.module.clone()
            valid_labels = y_taken[valid_mask]
            child_mod.seen.zero_()
            child_mod.mark_seen(valid_labels)
            shadow.mark_seen(valid_labels)

            opt_child = torch.optim.Adam(child_mod.parameters(), lr=self.cfg.lr)
            opt_shadow = torch.optim.Adam(shadow.parameters(), lr=self.cfg.lr)

            n_c = valid_labels.numel()
            bs = min(self.cfg.batch_size, n_c)
            for s_idx in range(self.cfg.probe_steps):
                gen_step = torch.Generator().manual_seed(self.cfg.seed + 1000 * self.step + s_idx)
                c_sel = torch.randint(0, n_c, (bs,), generator=gen_step)
                bx_c, by_c = x_taken[valid_mask][c_sel], valid_labels[c_sel]

                n_fit = min(bs, r_fit_x.shape[0])
                gen_fit = torch.Generator().manual_seed(self.cfg.seed + 2000 * self.step + s_idx)
                f_sel = torch.randint(0, r_fit_x.shape[0], (n_fit,), generator=gen_fit)
                bx_fit, by_fit = r_fit_x[f_sel], r_fit_y[f_sel]

                child_mod.train_step(bx_c, by_c, opt_child, replay=None)
                shadow.train_step(bx_c, by_c, opt_shadow, replay=(bx_fit, by_fit))

            with torch.no_grad():
                l_r_shadow = F.cross_entropy(shadow(r_probe_x), r_probe_y).item()
                l_r_parent = F.cross_entropy(parent_rec.module(r_probe_x), r_probe_y).item()
                d_old = float(l_r_shadow - l_r_parent)

                l_c_shadow = F.cross_entropy(shadow(x_taken[valid_mask]), valid_labels).item()
                l_c_child = F.cross_entropy(child_mod(x_taken[valid_mask]), valid_labels).item()
                gap = float(l_c_shadow - l_c_child)

            if d_old <= self.cfg.delta_forget and gap <= self.cfg.delta_forget:
                parent_rec.module.load_state_dict(shadow.state_dict())
                parent_rec.opt = self.make_optimizer(parent_rec.module, parent_rec.lr_mult)
                parent_rec.meta["sig"].update(x_taken)
                parent_rec.meta["sig"].fit()
                parent_rec.replay.add(x_taken[valid_mask], valid_labels)

                self.events.add(
                    kind="extend",
                    step=self.step,
                    expert=parent_eid,
                    affinity=aff,
                    d_old=d_old,
                    gap=gap,
                    n_cluster=len(x_taken),
                )

                if novel_indices is not None and len(novel_indices) > 0 and x_all is not None and assigned is not None:
                    p_par = parent_rec.meta["sig"].pvalue(x_all[novel_indices])
                    near = p_par >= self.cfg.q
                    if near.any():
                        assigned[novel_indices[near]] = parent_eid

                extended = True
                return None
            else:
                trained_child = child_mod

        if not extended:
            if len(self.experts) >= self.cfg.max_experts:                 # a full pool: stage 4.1 reports it and, by default, shares the cluster with the nearest expert instead of losing it
                return self._cap_cluster(parent_eid, aff, x_taken, y_taken, novel_indices, x_all, assigned)

            if trained_child is not None:
                child_mod = trained_child
            else:
                if self.cfg.init_mode == "scratch":
                    child_mod = MLPExpert(
                        d_in=self.cfg.d,
                        hidden=self.cfg.hidden,
                        depth=self.cfg.depth,
                        n_classes=self.cfg.n_classes,
                        seed=self.cfg.seed * 1000 + next_id,
                    )
                elif self.cfg.init_mode == "clone":
                    child_mod = parent_rec.module.clone()
                elif self.cfg.init_mode == "inherit":
                    rho = self.cfg.rho_max * (1.0 - aff)
                    rng_inh = np.random.default_rng([self.cfg.seed, 0x494E48, next_id])
                    child_mod = parent_rec.module.inherit(
                        rho=rho,
                        sigma=self.cfg.inherit_sigma,
                        rng=rng_inh,
                    )
                else:  # shrink_perturb
                    child_mod = parent_rec.module.shrink_perturb(
                        lam=aff,
                        seed=self.cfg.seed * 1000 + next_id,
                    )

            lam = 1.0 if self.cfg.init_mode == "clone" else (0.0 if self.cfg.init_mode == "scratch" else aff)

            child_mod.seen.zero_()
            valid_labels = y_taken[valid_mask] if has_labels else torch.empty(0, dtype=torch.long)
            if valid_labels.numel() > 0:
                child_mod.mark_seen(valid_labels)

            child_sig = self._new_sig(self.cfg.seed * 1000 + next_id)
            child_sig.update(x_taken)
            child_sig.fit()

            child_mon = LossChangeMonitor(
                alpha=self.cfg.monitor_alpha,
                warmup=self.cfg.monitor_warmup,
                seed=self.cfg.seed * 1000 + next_id,
            )
            child_recent = deque(maxlen=self.cfg.m_min)

            child_rec = self.add_expert(
                child_mod,
                parent=parent_eid,
                state="infant",
                meta={
                    "sig": child_sig,
                    "monitor": child_mon,
                    "recent": child_recent,
                    "infant_calls_seen": 0,
                    "loss_hist": [],
                    "policy": GrowthPolicy(self.grow_cfg),
                },
            )
            self.set_lr_mult(child_rec.id, self.cfg.infant_lr_mult)

            if valid_labels.numel() > 0:
                child_rec.replay.add(x_taken[valid_mask], valid_labels)
            else:
                child_rec.replay.add(x_taken, y_taken)

            self.events.add(
                kind="spawn",
                step=self.step,
                expert=child_rec.id,
                parent=parent_eid,
                affinity=aff,
                evidence=evidence,
                n_cluster=len(x_taken),
                init=self.cfg.init_mode,
                lam=lam,
                **({} if verdict is None else verdict.fields()),
            )

            if novel_indices is not None and len(novel_indices) > 0 and x_all is not None and assigned is not None:
                p_child = child_sig.pvalue(x_all[novel_indices])
                near = p_child >= self.cfg.q
                if near.any():
                    assigned[novel_indices[near]] = child_rec.id

            if spawned_list is not None:
                spawned_list.append(child_rec.id)

            return child_rec.id

    def _would_create_supersedes_cycle(self, start_eid: int, target_eid: int) -> bool:
        """Check if setting start_eid's supersedes pointer to target_eid creates a cycle."""
        if start_eid == target_eid:
            return True
        curr = target_eid
        visited = {start_eid}
        while curr is not None:
            if curr in visited:
                return True
            visited.add(curr)
            if curr in self.experts:
                curr = self.experts[curr].meta.get("supersedes")
            else:
                break
        return False

    def _attempt_merges(self) -> None:
        live_eids = sorted(self.experts.keys())
        for idx_i, i in enumerate(live_eids):
            if i not in self.experts:
                continue
            rec_i = self.experts[i]
            if rec_i.state == "infant" or rec_i.replay is None or rec_i.replay.n < 16:
                continue
            sig_i: VmfSignature = rec_i.meta["sig"]
            if sig_i.protos is None or len(sig_i.protos) == 0:
                continue

            for j in live_eids[idx_i + 1 :]:
                if j not in self.experts:
                    continue
                rec_j = self.experts[j]
                if rec_j.state == "infant" or rec_j.replay is None or rec_j.replay.n < 16:
                    continue
                sig_j: VmfSignature = rec_j.meta["sig"]
                if sig_j.protos is None or len(sig_j.protos) == 0:
                    continue

                p_j_on_i = float(sig_j.pvalue(sig_i.protos).mean().item())
                p_i_on_j = float(sig_i.pvalue(sig_j.protos).mean().item())
                overlap = min(p_j_on_i, p_i_on_j)

                if overlap >= self.cfg.merge_overlap:
                    rx_i, ry_i = rec_i.replay.data
                    rx_j, ry_j = rec_j.replay.data

                    split_i = len(rx_i) // 2
                    split_j = len(rx_j) // 2
                    r_fit_xi, r_fit_yi = rx_i[:split_i], ry_i[:split_i]
                    r_val_xi, r_val_yi = rx_i[split_i:], ry_i[split_i:]
                    r_fit_xj, r_fit_yj = rx_j[:split_j], ry_j[:split_j]
                    r_val_xj, r_val_yj = rx_j[split_j:], ry_j[split_j:]

                    u_fit_x = torch.cat([r_fit_xi, r_fit_xj], dim=0)
                    u_fit_y = torch.cat([r_fit_yi, r_fit_yj], dim=0)

                    valid_u = u_fit_y >= 0
                    if not valid_u.any():
                        continue

                    clone_mod = rec_i.module.clone()
                    clone_mod.mark_seen(u_fit_y[valid_u])
                    opt_clone = torch.optim.Adam(clone_mod.parameters(), lr=self.cfg.lr)

                    valid_ux = u_fit_x[valid_u]
                    valid_uy = u_fit_y[valid_u]
                    n_u = valid_ux.shape[0]
                    bs = min(self.cfg.batch_size, n_u)

                    for s_idx in range(self.cfg.probe_steps):
                        gen_s = torch.Generator().manual_seed(self.cfg.seed + 3000 * self.step + s_idx)
                        sel = torch.randint(0, n_u, (bs,), generator=gen_s)
                        clone_mod.train_step(valid_ux[sel], valid_uy[sel], opt_clone, replay=None)

                    def _acc(mod, vx, vy):
                        valid_v = vy >= 0
                        if not valid_v.any():
                            return 1.0
                        with torch.no_grad():
                            preds = mod(vx[valid_v]).argmax(dim=-1)
                            return float((preds == vy[valid_v]).float().mean().item())

                    acc_i_pre = _acc(rec_i.module, r_val_xi, r_val_yi)
                    acc_i_post = _acc(clone_mod, r_val_xi, r_val_yi)
                    acc_j_pre = _acc(rec_j.module, r_val_xj, r_val_yj)
                    acc_j_post = _acc(clone_mod, r_val_xj, r_val_yj)

                    drop_i = acc_i_pre - acc_i_post
                    drop_j = acc_j_pre - acc_j_post

                    if drop_i <= self.cfg.delta_merge and drop_j <= self.cfg.delta_merge:
                        rec_i.module.load_state_dict(clone_mod.state_dict())
                        rec_i.opt = self.make_optimizer(rec_i.module, rec_i.lr_mult)
                        rec_i.meta["sig"].merge_from(sig_j)
                        rec_i.meta["sig"].fit()
                        rec_i.replay.add(rx_j, ry_j)
                        if rec_j.meta.get("windows") is not None:          # routing v2 stage 3a: the merged expert remembers the sources of both
                            self._windows_of(rec_i).merge_from(rec_j.meta["windows"])
                        if self.cfg.spawn_rule == "transfer":              # routing v2 stage 4b: the merged expert keeps the references of both
                            self._merge_refs(rec_i, rec_j)
                        if self.cfg.growth_rule == "gap":                  # routing v2 stage 4c: the merged weights are not the guarded ones, and the old windows describe the old module
                            self._forget_growth_state(rec_i)

                        for c_id, c_rec in list(self.experts.items()):
                            if c_rec.parent == j:
                                self.set_parent(c_id, i)

                        # Update supersedes pointers to strictly maintain an acyclic graph (DAG)
                        target_succ = rec_j.meta.get("supersedes")
                        if rec_i.meta.get("supersedes") == j:
                            if target_succ is not None and not self._would_create_supersedes_cycle(i, target_succ):
                                rec_i.meta["supersedes"] = target_succ
                            else:
                                rec_i.meta.pop("supersedes", None)
                        elif rec_i.meta.get("supersedes") is None:
                            if target_succ is not None and not self._would_create_supersedes_cycle(i, target_succ):
                                rec_i.meta["supersedes"] = target_succ

                        for k_id, k_rec in list(self.experts.items()):
                            if k_id != i and k_id != j and k_rec.meta.get("supersedes") == j:
                                if not self._would_create_supersedes_cycle(k_id, i):
                                    k_rec.meta["supersedes"] = i
                                else:
                                    k_rec.meta.pop("supersedes", None)

                        self.remove_expert(j)
                        self.events.add(
                            kind="merge",
                            step=self.step,
                            kept=i,
                            removed=j,
                            change_kept=float(acc_i_post - acc_i_pre),
                            change_removed=float(acc_j_post - acc_j_pre),
                        )
                        break

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
        """Observe a batch (x, y); routes known data, handles novelty, probes interference, and trains."""
        router_mark = len(self.events) if self.cfg.router else 0       # routing v2 stage 4a: where this call's events start
        x = self._normalize(x.float())
        b = x.shape[0]
        start_arrival = self.samples_seen
        self.samples_seen += b

        if y is not None:
            y_t = torch.as_tensor(y, dtype=torch.long)
        else:
            y_t = None

        # ----------------------------------------------------------- 13f Infant Lifecycle
        for eid, rec in list(self.experts.items()):
            if rec.state == "infant":
                calls = rec.meta.get("infant_calls_seen", 0) + 1
                rec.meta["infant_calls_seen"] = calls
                if calls >= self.cfg.infant_calls:
                    self.set_state(eid, "active")
                    self.set_lr_mult(eid, 1.0)
                    rec.last_active = self.step
                    if self.cfg.arm_monitor_after_infancy:
                        self._forget_competence_history(rec)

        # ----------------------------------------------------------- 13h Inactivity Lifecycle
        for eid, rec in list(self.experts.items()):
            if rec.state in ("active", "dormant"):
                inactive_steps = self.step - rec.last_active
                if rec.state == "active" and inactive_steps >= self.cfg.dormant_after:
                    self.set_state(eid, "dormant")
                elif rec.state == "dormant" and inactive_steps >= self.cfg.archive_after:
                    self.set_state(eid, "archived")
                    rec.replay = None
                    rec.opt = None
                    self.events.add(kind="archive", step=self.step, expert=eid)

        # ----------------------------------------------------------- 13a Bootstrap
        if len(self.experts) == 0:
            self._boot_x.append(x)
            if y_t is not None:
                self._boot_y.append(y_t)
            else:
                self._boot_y.append(torch.full((b,), -1, dtype=torch.long))

            n_boot = sum(bx.shape[0] for bx in self._boot_x)
            if n_boot < self.cfg.m_min:
                novel = torch.zeros(b, dtype=torch.bool)
                p_nov = torch.ones(b, dtype=torch.float32)
                assigned = torch.full((b,), -1, dtype=torch.long)
                self.tick()
                return StepRecord(
                    novel=novel,
                    p_nov=p_nov,
                    assigned=assigned,
                    loss=float("nan"),
                    spawned=[],
                    n_experts=0,
                    meta={"set_verdict": None} if self.cfg.set_gate else {},
                )

            # Once >= m_min samples are buffered, found expert 0 from scratch
            all_x = torch.cat(self._boot_x, dim=0)
            all_y = torch.cat(self._boot_y, dim=0)
            self._boot_x.clear()
            self._boot_y.clear()

            mod = MLPExpert(
                d_in=self.cfg.d,
                hidden=self.cfg.hidden,
                depth=self.cfg.depth,
                n_classes=self.cfg.n_classes,
                seed=self.cfg.seed * 1000 + 0,
            )
            sig = self._new_sig(self.cfg.seed * 1000 + 0)
            sig.update(all_x)
            sig.fit()
            monitor = LossChangeMonitor(
                alpha=self.cfg.monitor_alpha,
                warmup=self.cfg.monitor_warmup,
                seed=self.cfg.seed * 1000 + 0,
            )
            recent = deque(maxlen=self.cfg.m_min)

            rec = self.add_expert(
                mod,
                parent=None,
                state="active",
                meta={
                    "sig": sig,
                    "monitor": monitor,
                    "recent": recent,
                    "infant_calls_seen": self.cfg.infant_calls,
                    "loss_hist": [],
                    "policy": GrowthPolicy(self.grow_cfg),
                },
            )

            self.events.add(
                kind="spawn",
                step=self.step,
                expert=rec.id,
                parent=None,
                affinity=0.0,
                evidence=0.0,
                n_cluster=all_x.shape[0],
                init="scratch",
                lam=0.0,
            )

            loss = float("nan")
            valid = all_y >= 0
            if valid.any():
                loss = self.train_expert(
                    rec.id,
                    all_x[valid],
                    all_y[valid],
                    replay=True,
                    project=self.cfg.agem,
                )
                if not math.isnan(loss):
                    rec.meta.setdefault("loss_hist", []).append(loss)

            assigned = torch.full((b,), rec.id, dtype=torch.long)
            p_nov = sig.pvalue(x)
            novel = benjamini_hochberg(p_nov, self.cfg.q)
            self._note_cap_reached()
            self.tick()
            return StepRecord(
                novel=novel,
                p_nov=p_nov,
                assigned=assigned,
                loss=loss,
                spawned=[rec.id],
                n_experts=len(self.experts),
                meta={"set_verdict": None} if self.cfg.set_gate else {},
            )

        # ------------------------------------------------ 13b Gate & Routing
        eids = list(self.experts.keys())
        P, LogF = self._gate_matrices(x, eids, phase="observe")               # (B, K) each
        self._count_infant_suppression(x, P, eids)                            # stage 4.1: read-only counters

        if self.cfg.no_conformal:
            max_logf = torch.max(LogF, dim=1).values
            fixed_thresh = 0.0
            novel = max_logf < fixed_thresh
            p_nov = torch.sigmoid(max_logf)
        else:
            p_nov = torch.max(P, dim=1).values     # (B,)
            novel = benjamini_hochberg(p_nov, self.cfg.q)  # (B,) bool

        assigned = torch.full((b,), -1, dtype=torch.long)
        thresholds = torch.clamp_max(p_nov, self.cfg.q)  # min(q, p_nov)

        for i in range(b):
            if not novel[i]:
                if self.cfg.no_conformal:
                    best_eid = eids[int(torch.argmax(LogF[i]).item())]
                    assigned[i] = best_eid
                else:
                    eligible_mask = P[i] >= thresholds[i]
                    if not eligible_mask.any():
                        eligible_mask = (P[i] == p_nov[i])
                    eligible_idx = torch.where(eligible_mask)[0].tolist()
                    eligible_eids = [eids[idx] for idx in eligible_idx] if eligible_idx else [int(torch.argmax(P[i]).item())]
                    active_candidates = self._filter_superseded(eligible_eids)
                    if not active_candidates:
                        active_candidates = eligible_eids
                    best_eid = max(active_candidates, key=lambda e: LogF[i, eids.index(e)].item())
                    assigned[i] = best_eid
            else:
                assigned[i] = -1

        if self.cfg.batch_vote:                                    # routing v2 stage 1; reads only `assigned`, so it holds in every gate mode
            assigned = self._batch_consistent(assigned)

        set_verdict, probation_mask = None, novel                  # routing v2 stage 3a (OFF by default): the accepted samples are tested as a SET
        if self.cfg.set_gate:
            assigned, set_novel, set_verdict = self._set_gate_step(x, y_t, assigned, eids)
            if set_novel is not None:
                probation_mask = novel | set_novel                 # called new as a set: they join the individually flagged ones; `novel` itself stays the per-sample gate's

        spawned = []

        # ------------------------------------------------ 13c Probation Buffer
        novel_indices = torch.where(probation_mask)[0]
        full = len(self.experts) >= self.cfg.max_experts                       # stage 4.1: a full pool still looks at its novel samples when it shares (cap_policy), and says so when it drops
        if self.cfg.no_probation:
            if len(novel_indices) > 0 and full and self.cfg.cap_policy == "drop":
                self._note_cap_blocked(int(len(novel_indices)))
            if len(novel_indices) > 0 and (not full or self.cfg.cap_policy == "share"):
                x_nov = x[novel_indices]
                y_nov = y_t[novel_indices] if y_t is not None else torch.full((len(x_nov),), -1, dtype=torch.long)
                self._handle_cluster(
                    x_taken=x_nov,
                    y_taken=y_nov,
                    evidence=100.0,
                    novel_indices=novel_indices,
                    x_all=x,
                    assigned=assigned,
                    spawned_list=spawned,
                )
        else:
            if len(novel_indices) > 0:
                x_nov = x[novel_indices]
                y_nov = y_t[novel_indices] if y_t is not None else None
                arr_nov = start_arrival + novel_indices
                if set_verdict == "held":                              # persistence: the flagged samples of a doubtful batch wait with its accepted samples
                    self._held.append((x_nov, y_nov, arr_nov))
                else:
                    self.probation.add(x_nov, y_nov, arrival=arr_nov)

            ttl = int(math.ceil(self.cfg.m_min / self.cfg.min_share))
            self.probation.expire(now=self.samples_seen, ttl=ttl)

            if len(novel_indices) > 0 and len(self.probation) >= self.cfg.m_min and full and self.cfg.cap_policy == "drop":
                self._note_cap_blocked(int(len(novel_indices)))
            if len(self.probation) >= self.cfg.m_min and (not full or self.cfg.cap_policy == "share"):
                cand_clusters = self.probation.clusters(
                    min_neighbors=self.cfg.min_neighbors,
                    m_min=self.cfg.m_min,
                    edge_null=self.cfg.edge_null,
                )
                for c_idx in cand_clusters:
                    # Evidence test for cluster C:
                    cx = self.probation.x[c_idx]
                    if cx.shape[0] > 256:
                        cx_sub = cx[:256]
                    else:
                        cx_sub = cx

                    sig_c = self._new_sig(self.cfg.seed * 1000 + len(self.experts))
                    sig_c.update(cx_sub)
                    sig_c.fit()

                    if hasattr(sig_c, "kappa") and getattr(sig_c, "kappa") != 1.0:
                        logf_c_loo = sig_c.cal_scores + log_vmf_norm(self.cfg.d, sig_c.kappa)
                    else:
                        logf_c_loo = sig_c.cal_scores
                    protos_c = sig_c.protos
                    live_logfs = torch.stack([
                        rec.meta["sig"].logpdf(protos_c) for rec in self.experts.values()
                    ], dim=1)
                    max_live_logf = live_logfs.max(dim=1).values

                    evidence = float((logf_c_loo - max_live_logf).sum().item())
                    threshold = math.log(1.0 / self.cfg.alpha_spawn)

                    if evidence > threshold:
                        x_taken, y_taken = self.probation.take(c_idx)
                        self._handle_cluster(
                            x_taken=x_taken,
                            y_taken=y_taken,
                            evidence=evidence,
                            novel_indices=novel_indices,
                            x_all=x,
                            assigned=assigned,
                            spawned_list=spawned,
                        )
                        break
                    else:
                        self.events.add(
                            kind="abstain",
                            step=self.step,
                            reason="evidence",
                            n_cluster=len(c_idx),
                        )

        # Train live experts on their assigned known samples
        losses = []
        for eid in list(self.experts.keys()):
            mask_k = (assigned == eid)
            if not mask_k.any():
                continue
            xk = x[mask_k]
            rec = self.experts[eid]
            rec.last_active = self.step

            # 13h Revive if dormant or archived
            if rec.state in ("dormant", "archived"):
                if rec.state == "archived":
                    rec.opt = self.make_optimizer(rec.module, rec.lr_mult)
                    rng_rep = np.random.default_rng([self.cfg.seed, 0x524550, eid, self.step])
                    rec.replay = Reservoir(capacity=self.cfg.replay_size, d=self.cfg.d, rng=rng_rep)
                self.set_state(eid, "active")
                self.events.add(kind="revive", step=self.step, expert=eid)

            rec.meta["sig"].update(xk)
            if self.cfg.set_gate:                                  # routing v2 stage 3a: remember which source these samples came from
                self._update_windows(rec, xk)

            if y_t is not None:
                yk = y_t[mask_k]
                valid = yk >= 0
                if valid.any():
                    xk_v = xk[valid]
                    yk_v = yk[valid]

                    # 13g Competence prequential loss & CUSUM monitor
                    with torch.no_grad():
                        logits = rec.module(xk_v)
                        preq_losses = F.cross_entropy(logits, yk_v, reduction="none")

                    recent_deque = rec.meta.get("recent")
                    monitor: Optional[LossChangeMonitor] = rec.meta.get("monitor")
                    alarm_fired = False

                    # An infant's loss is still settling (it rises as the expert over-fits its first tiny batch), so with
                    # arm_monitor_after_infancy the monitor is neither fed nor tested until infancy is over.
                    settling = self.cfg.arm_monitor_after_infancy and rec.state == "infant"
                    for i in range(0 if settling else len(preq_losses)):
                        xi = xk_v[i].clone()
                        yi = yk_v[i].clone()
                        if recent_deque is not None:
                            recent_deque.append((xi, yi))
                        loss_val = float(preq_losses[i].item())
                        if self.cfg.competence and monitor is not None:
                            if monitor.update(loss_val):
                                alarm_fired = True

                    if self.cfg.spawn_rule == "transfer" and not settling:      # routing v2 stage 4b: the honest per-sample losses are this expert's reference for the transfer test
                        rec.meta.setdefault("ref", deque(maxlen=self.cfg.transfer_ref)).extend(preq_losses.tolist())

                    if alarm_fired and monitor is not None:
                        self.events.add(kind="alarm", step=self.step, expert=eid, statistic=monitor.statistic)
                        if recent_deque is not None and len(recent_deque) >= self.cfg.m_min:
                            recent_x = torch.stack([item[0] for item in recent_deque], dim=0)
                            recent_y = torch.stack([item[1] for item in recent_deque], dim=0)
                            child_id = self._handle_cluster(
                                x_taken=recent_x,
                                y_taken=recent_y,
                                evidence=0.0,
                                force_parent=eid,
                                spawned_list=spawned,
                            )
                            if child_id is not None:
                                self.experts[child_id].meta["supersedes"] = eid
                            monitor.reset()
                            recent_deque.clear()

                    if not alarm_fired and self._known_batch(rec, preq_losses):      # E-063 skip rule: the expert already predicts this batch, so no optimiser steps
                        self._absorb_only(eid, xk_v, yk_v)
                        self.skip_stats["skipped"] += 1
                        self.skip_stats["skipped_rows"] += int(xk_v.shape[0])
                    elif not alarm_fired:
                        self.skip_stats["trained"] += 1
                        self.skip_stats["trained_rows"] += int(xk_v.shape[0])
                        l_k = self.train_expert(
                            eid,
                            xk_v,
                            yk_v,
                            replay=True,
                            project=self.cfg.agem,
                        )
                        if not math.isnan(l_k):
                            losses.append(l_k)
                            rec.meta.setdefault("loss_hist", []).append(l_k)

                        if self.cfg.growth_rule == "gap":              # routing v2 stage 4c: growth from the within-batch fit gap replaces the shadow-trial policy below
                            self._gap_growth_step(rec, xk_v, yk_v, preq_losses)

                        # R6: Task 14 Growth inside Architecture A
                        elif (
                            self.cfg.grow
                            and rec.state == "active"
                            and rec.replay is not None
                            and rec.replay.n >= 128
                            and plateau(rec.meta.get("loss_hist", []))
                        ):
                            rx, ry = rec.replay.data
                            n_rep = rec.replay.n
                            n_tr = int(round(0.8 * n_rep))
                            rng_growth = np.random.default_rng([self.cfg.seed, 0x47524F57, eid, self.step])
                            perm = rng_growth.permutation(n_rep)
                            idx_tr = torch.from_numpy(perm[:n_tr])
                            idx_va = torch.from_numpy(perm[n_tr:])
                            x_tr, y_tr = rx[idx_tr], ry[idx_tr]
                            x_va, y_va = rx[idx_va], ry[idx_va]

                            if "policy" not in rec.meta:
                                rec.meta["policy"] = GrowthPolicy(self.grow_cfg)
                            policy: GrowthPolicy = rec.meta["policy"]
                            decision = policy.step(rec.module, x_tr, y_tr, x_va, y_va, rng=rng_growth)

                            if decision.reason == "committed":
                                rec.opt = self.make_optimizer(rec.module, rec.lr_mult)
                                key = "deepen" if decision.kind == "deepen" else f"widen+{decision.increment}"
                                vs_status = decision.tried.get(key, {}).get("vs_status", 0.0)
                                self.events.add(
                                    "grow",
                                    self.step,
                                    expert=eid,
                                    kind=decision.kind,
                                    increment=decision.increment,
                                    gain=decision.gain,
                                    p=decision.p_value,
                                    params_added=decision.params_added,
                                    vs_status=vs_status,
                                )

                            rec.meta["loss_hist"].clear()

        # 13h Distillation merge
        if (self.step + 1) % self.cfg.merge_every == 0:
            self._attempt_merges()

        if self.cfg.router:                                            # routing v2 stage 4a: count the call, mark the router stale on a structural change; nothing is fitted here
            self._router_after_observe(router_mark)

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self._note_cap_reached()
        self.tick()
        return StepRecord(
            novel=novel,
            p_nov=p_nov,
            assigned=assigned,
            loss=mean_loss,
            spawned=spawned,
            n_experts=len(self.experts),
            meta={"set_verdict": set_verdict} if self.cfg.set_gate else {},
        )

    # ------------------------------------------------------------------------------------------ routing v2 stage 4a: the router
    _ROUTER_EVENTS = ("spawn", "extend", "merge", "archive")

    def _route_state(self) -> RouteRule:
        """The router's state, created on first use (a pool pickled before stage 4 has none)."""
        if not hasattr(self, "_route"):
            self._route, self._route_age, self._route_dirty = RouteRule(), 0, False
        return self._route

    def _router_after_observe(self, mark: int) -> None:
        """Count the observe call; mark the router stale when an expert was spawned, extended, merged or archived. The router never influences training, so nothing is fitted
        here: the next predict that finds the router stale refits it (lazily)."""
        self._route_state()
        self._route_age += 1
        if any(e.kind in self._ROUTER_EVENTS for e in self.events.since(mark)):
            self._route_dirty = True

    def _route_is_stale(self) -> bool:
        rule = self._route_state()
        return rule.eids is None or rule.eids != sorted(self.experts) or self._route_dirty or self._route_age >= self.cfg.router_refit_every

    def _current_route_rule(self) -> RouteRule:
        if self._route_is_stale():
            self._refit_router()
        return self._route

    def _refit_router(self) -> None:
        """Cross-validate the candidates on the experts' stored prototypes (owner = the expert that holds the prototype), select a rule (density unless another is clearly better),
        fit the selected scorer on all of them. With fewer than two experts, or an expert holding fewer than ``router_min_protos`` prototypes, the density rule stays."""
        cfg, rule = self.cfg, self._route_state()
        eids = sorted(self.experts)
        counts = [self.experts[e].meta["sig"].n for e in eids]
        before = rule.name
        if len(eids) < 2 or min(counts) < cfg.router_min_protos:
            rule.set("density", None, eids, {}, sum(counts))
        else:
            X = torch.cat([self.experts[e].meta["sig"].protos for e in eids])
            y = torch.cat([torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)])
            names = tuple(n for n in cfg.router_candidates if not (n == "softmax" and len(eids) > cfg.router_softmax_max_experts))
            cv = cv_owner_accuracy(X, y, len(eids), cfg.router_folds, generator(cfg.seed, 0x52544F, self.step), names)
            chosen = select_rule(cv, before, cfg.router_margin)
            scorer = None if chosen == "density" else FITS[chosen](X, y, len(eids), generator(cfg.seed, 0x52544C, self.step))
            rule.set(chosen, scorer, eids, cv, int(X.shape[0]))
        self._route_age, self._route_dirty = 0, False
        self.events.add(kind="router_refit", step=self.step, selected=rule.name, cv=rule.cv, rows=rule.rows, experts=len(eids))
        if rule.name != before:
            self.events.add(kind="router_switch", step=self.step, **{"from": before, "to": rule.name})

    def _route_by_router(self, x: Tensor, eids: List[int], P: Tensor, LogF: Tensor, idx: Tensor) -> Tensor:
        """``idx`` is the density gate's choice (a column per input). With the rule 'density' selected it is returned untouched (today's path, bit for bit); with another rule the
        scores S = scorer(x) replace it: mode 'restrict' takes the best S among the experts the gate accepts (the best S overall when it accepts none), 'free' the best S overall."""
        if x.shape[0] == 0:
            return idx
        rule = self._current_route_rule()
        if rule.name == "density":
            return idx
        col = {e: j for j, e in enumerate(rule.eids)}
        S = rule.scorer(x)[:, [col[e] for e in eids]]                                   # columns in the order of ``eids``
        if self.cfg.router_mode == "free":
            return S.argmax(dim=1)
        accepted = (LogF >= 0.0) if self.cfg.no_conformal else (P.double() >= float(self.cfg.q))
        best_accepted = S.masked_fill(~accepted, float("-inf")).argmax(dim=1)
        return torch.where(accepted.any(dim=1), best_accepted, S.argmax(dim=1))

    # ------------------------------------------------------------------------------------------ routing v2 stage 4b: spawn or extend by a held-out transfer test
    _TRANSFER_MIN_CLUSTER = 8                                           # fewer labelled samples than this cannot carry the test

    def _transfer_reference(self, rec, exclude: int) -> list:
        """The expert's honest reference losses, newest last. With ``exclude`` > 0 the newest ``exclude`` entries are left out: on the alarm path the cluster IS those samples."""
        ref = list(rec.meta.get("ref", ()))
        return ref[:-exclude] if exclude > 0 else ref

    def _transfer_verdict(self, x_taken: Tensor, y_taken: Tensor, force_parent: Optional[int]) -> Optional[TransferVerdict]:
        """Score the cluster's labelled samples (no training) with every expert that has at least ``transfer_min_ref`` reference losses (only ``force_parent`` on the alarm path);
        the best one is the parent; ``cluster_conflict`` asks whether it fails on the cluster. None when the test cannot run (the caller then takes today's path)."""
        cfg = self.cfg
        valid = y_taken >= 0
        if int(valid.sum()) < self._TRANSFER_MIN_CLUSTER:
            return None
        if force_parent is not None:
            # Alarm path (refinement R-7): the cluster is the parent's own newest samples, which it may already have trained on, so re-scoring them with the current weights would
            # understate how badly it did. Their HONEST losses (taken before any training on them) are the newest entries of its reference: those are the cluster's losses, the
            # older entries are the reference.
            n = int(len(x_taken))
            stored = list(self.experts[force_parent].meta.get("ref", ()))
            if n > len(stored) or len(stored) - n < cfg.transfer_min_ref:
                return None
            eid, losses, ref = force_parent, torch.tensor(stored[-n:]), self._transfer_reference(self.experts[force_parent], n)
        else:
            xv, yv = x_taken[valid], y_taken[valid]
            best = None
            for e in list(self.experts):
                rec = self.experts[e]
                ref = self._transfer_reference(rec, 0)
                if len(ref) < cfg.transfer_min_ref:
                    continue
                with torch.no_grad():
                    scored = F.cross_entropy(rec.module(xv), yv, reduction="none")
                if best is None or float(scored.mean()) < best[0]:
                    best = (float(scored.mean()), e, scored, ref)
            if best is None:
                return None
            _, eid, losses, ref = best
        margin = cfg.transfer_margin * math.log(max(int(self.experts[eid].module.seen.sum()), 2))
        conflict, p, diff = cluster_conflict(losses, torch.tensor(ref), cfg.transfer_alpha, cfg.transfer_perms, margin, generator(cfg.seed, 0x545246, self.step, eid))
        veto = False
        if not conflict and cfg.transfer_chance is not None:                       # a parent that does worse than chance on the cluster cannot vouch for it
            bar = cfg.transfer_chance * math.log(max(int(self.experts[eid].module.seen.sum()), 2))
            veto = float(losses.mean()) >= bar and (not cfg.transfer_chance_weak_only or float(np.mean(ref)) >= bar)
            conflict = veto
        return TransferVerdict(parent=eid, conflict=conflict, p=p, diff=diff, loss_cluster=float(losses.mean()), loss_ref=float(np.mean(ref)), margin=margin,
                               n_cluster=int(losses.numel()), n_ref=len(ref), chance_veto=veto)

    def _extend_unchanged(self, parent_eid: int, aff: float, x_taken: Tensor, y_taken: Tensor, verdict: TransferVerdict, novel_indices, x_all, assigned) -> None:
        """No conflict: the parent already predicts these labels, so the cluster joins its signature and replay and NO weight changes (it keeps training on the stream as usual)."""
        parent_rec = self.experts[parent_eid]
        valid = y_taken >= 0
        parent_rec.meta["sig"].update(x_taken)
        parent_rec.meta["sig"].fit()
        parent_rec.replay.add(x_taken[valid], y_taken[valid])
        self.events.add(kind="extend", step=self.step, expert=parent_eid, affinity=aff, n_cluster=len(x_taken), **verdict.fields())
        if novel_indices is not None and len(novel_indices) > 0 and x_all is not None and assigned is not None:
            near = parent_rec.meta["sig"].pvalue(x_all[novel_indices]) >= self.cfg.q
            if near.any():
                assigned[novel_indices[near]] = parent_eid

    def _pretrain_child(self, parent_rec, aff: float, next_id: int, x_taken: Tensor, y_taken: Tensor):
        """The child of a transfer conflict: initialised exactly as the probe's child is, then ``spawn_pretrain_steps`` steps on the cluster alone (same seeds as the probe's)."""
        cfg = self.cfg
        if cfg.init_mode == "scratch":
            child = MLPExpert(d_in=cfg.d, hidden=cfg.hidden, depth=cfg.depth, n_classes=cfg.n_classes, seed=cfg.seed * 1000 + next_id)
        elif cfg.init_mode == "clone":
            child = parent_rec.module.clone()
        elif cfg.init_mode == "inherit":
            child = parent_rec.module.inherit(rho=cfg.rho_max * (1.0 - aff), sigma=cfg.inherit_sigma, rng=np.random.default_rng([cfg.seed, 0x494E48, next_id]))
        else:
            child = parent_rec.module.shrink_perturb(lam=aff, seed=cfg.seed * 1000 + next_id)
        valid = y_taken >= 0
        xv, yv = x_taken[valid], y_taken[valid]
        child.seen.zero_()
        child.mark_seen(yv)
        opt = torch.optim.Adam(child.parameters(), lr=cfg.lr)
        bs = min(cfg.batch_size, int(yv.numel()))
        for s_idx in range(cfg.spawn_pretrain_steps):
            gen = torch.Generator().manual_seed(cfg.seed + 1000 * self.step + s_idx)
            sel = torch.randint(0, int(yv.numel()), (bs,), generator=gen)
            child.train_step(xv[sel], yv[sel], opt, replay=None)
        return child

    def _merge_refs(self, rec_i, rec_j) -> None:
        """The survivor of a merge keeps the references of both (the newest ``transfer_ref`` of the union, the second expert's entries last)."""
        if "ref" in rec_i.meta or "ref" in rec_j.meta:
            rec_i.meta["ref"] = deque(list(rec_i.meta.get("ref", ())) + list(rec_j.meta.get("ref", ())), maxlen=self.cfg.transfer_ref)

    # ------------------------------------------------------------------------------------------ routing v2 stage 4c: growth from the within-batch fit gap
    def _grow_stats(self) -> dict:
        """Counters of the growth controller (the steps at which growth was wanted but blocked are counted per step). No wall-clock."""
        if not hasattr(self, "grow_stats"):
            self.grow_stats = {"post_rows": 0, "train_rows": 0, "blocked_cap": 0, "blocked_budget": 0, "reverts": 0}
        return self.grow_stats

    def _live_params(self) -> int:
        return sum(r.module.num_params() for r in self.experts.values())

    @staticmethod
    def _forget_growth_state(rec) -> None:
        rec.meta.pop("guard", None)
        rec.meta.pop("gap", None)

    def _gap_growth_step(self, rec, x: Tensor, y: Tensor, pre: Tensor) -> None:
        """Called after an active expert trained on this batch. ``pre`` = its honest per-sample losses on the batch BEFORE the steps (the competence monitor's input); the losses on
        the same rows AFTER the steps cost one no-grad forward of at most 64 rows. See ``growmoe.gapgrow`` for the signals; the decision: mature, gap below ``grow_gap``, cooled
        down, room under the width cap and the parameter budget -> widen by ``grow_factor`` (function-preserving), keep the old expert for ``grow_guard_window`` batches."""
        cfg, stats = self.cfg, self._grow_stats()
        if not cfg.grow or rec.state != "active":
            return
        m = min(64, int(y.numel()))
        with torch.no_grad():
            post = F.cross_entropy(rec.module(x[:m]), y[:m], reduction="none")
        stats["post_rows"] += m
        stats["train_rows"] += int(y.numel())
        mon = rec.meta.get("gap")
        if mon is None:
            mon = rec.meta["gap"] = GapMonitor(cfg.grow_window)
        mon.add(pre[:m], post)
        guard = rec.meta.get("guard")
        if guard is not None:                                          # a growth is on probation: no new decision until it is confirmed or undone
            guard.losses.append(float(pre.mean()))
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

    def _widen_expert(self, rec, mon: GapMonitor, k: int, added: int, gap: float) -> None:
        cfg = self.cfg
        old_module, old_opt = rec.module, rec.opt
        new_module = old_module.clone()
        new_module.widen(k, np.random.default_rng([cfg.seed, 0x47524F57, rec.id, self.step]))
        rec.module, rec.opt = new_module, self.make_optimizer(new_module, rec.lr_mult)       # growth replaces parameters: the optimiser is rebuilt, as after any committed growth
        rec.meta["guard"] = GrowthGuard(module=old_module, opt=old_opt, before=mon.recent_pre_mean(cfg.grow_guard_window), hidden_before=old_module.hidden)
        improvement = mon.improvement()
        mon.grew()
        self.events.add("grow", self.step, expert=rec.id, kind="widen", increment=k, gap=gap, improvement=improvement, hidden_before=old_module.hidden,
                        hidden_after=new_module.hidden, params_added=added, rule="gap")

    def _close_guard(self, rec, guard: GrowthGuard, mon: GapMonitor) -> None:
        """The guard window is over: keep the growth, or undo it (the expert and optimiser as they were) when the honest loss got worse by more than ``grow_guard_eps``."""
        cfg = self.cfg
        rec.meta.pop("guard", None)
        after = sum(guard.losses) / len(guard.losses)
        if guard_reverts(guard.before, after, cfg.grow_guard_eps):
            grown = rec.module.hidden
            rec.module, rec.opt = guard.module, guard.opt
            rec.meta["grow_cooldown"] = 2 * rec.meta.get("grow_cooldown", cfg.grow_cooldown)
            mon.grew()
            self._grow_stats()["reverts"] += 1
            self.events.add("grow_revert", self.step, expert=rec.id, hidden_from=grown, hidden_to=rec.module.hidden, before=guard.before, after=after)

    def extra_cost(self, d: int):
        """(parameters, extra multiply-adds per prediction, extra stored scalars) of the stage-4 machinery, read by ``accounting.pool_report``."""
        params = macs = scalars = 0
        if self.cfg.router and hasattr(self, "_route") and self._route.eids is not None:
            p, m, s = self._route.cost(d)
            params, macs, scalars = params + p, macs + m, scalars + s
        if self.cfg.spawn_rule == "transfer":
            scalars += sum(len(r.meta["ref"]) for r in self.experts.values() if "ref" in r.meta)
        if self.cfg.growth_rule == "gap":                                # a live growth guard keeps the pre-growth module: its parameters are real memory
            params += sum(r.meta["guard"].module.num_params() for r in self.experts.values() if "guard" in r.meta)
        return params, macs, scalars

    # ------------------------------------------------------------------------------------------ stage 4.1: a full pool is loud and shares; read-only telemetry
    def _cap_state(self) -> dict:
        """What the expert cap did (counters only). Created on first use, so a pool pickled before stage 4.1 has them too."""
        if not hasattr(self, "_cap"):
            self._cap = {"reached_step": None, "warned": False, "blocked": 0, "blocked_samples": 0, "shared": 0, "shared_samples": 0}
        return self._cap

    def _infant_state(self) -> dict:
        """What the infants' relaxed gate did (counters only); see ``_count_infant_suppression``."""
        if not hasattr(self, "_infant"):
            self._infant = {"batches_with_infant": 0, "novel_to_mature": 0, "suppressed": 0}
        return self._infant

    def _note_cap_reached(self) -> None:
        """The first time the pool holds ``max_experts`` experts: one ``cap_reached`` event. Reaching the cap changes nothing by itself, so it does not warn; the first time the cap acts it does."""
        st = self._cap_state()
        if st["reached_step"] is None and len(self.experts) >= self.cfg.max_experts:
            st["reached_step"] = self.step
            self.events.add(kind="cap_reached", step=self.step, experts=len(self.experts), cap=self.cfg.max_experts, policy=self.cfg.cap_policy)

    def _warn_cap_once(self) -> None:
        """The first time a full pool turns a cluster away or shares it: one ``RuntimeWarning`` per pool that says what the cap does from now on."""
        st = self._cap_state()
        if st["warned"]:
            return
        st["warned"] = True
        share = self.cfg.cap_policy == "share"
        what = ("is added to the nearest expert (cap_policy='share'): nothing is deleted, but that expert now serves more than one thing"
                if share else "is NOT learned (cap_policy='drop'): its samples are lost")
        warnings.warn(f"PhyloMoE: the pool is at max_experts={self.cfg.max_experts} (step {self.step}), so a region the pool has never seen {what}. "
                      "Raise max_experts if the stream can hold more distinct regions than that.", RuntimeWarning, stacklevel=3)

    def _note_cap_blocked(self, samples: int) -> None:
        """cap_policy 'drop': novel samples met a full pool and nothing was founded. Every time is counted; an event is written the 1st, 2nd, 4th, 8th ... time (a pool stuck at its cap would write one per batch)."""
        st = self._cap_state()
        st["blocked"] += 1
        st["blocked_samples"] += int(samples)
        self._warn_cap_once()
        if st["blocked"] & (st["blocked"] - 1) == 0:
            self.events.add(kind="cap_blocked", step=self.step, samples=int(samples), blocked=st["blocked"], experts=len(self.experts), cap=self.cfg.max_experts)

    def _cap_cluster(self, parent_eid: int, aff: float, x_taken: Tensor, y_taken: Tensor, novel_indices, x_all, assigned) -> None:
        """A cluster that would found an expert met a full pool. 'drop' = the shipped behaviour (nothing is founded, the cluster is lost), now counted and reported. 'share' (default) = the cluster
        joins the signature and the replay of the nearest expert and no weight changes; from then on the gate sends the region to that expert, which trains on it like on any of its own samples
        (when the labels conflict it becomes a compromise, which is the price of the cap; nothing is deleted and nothing is lost)."""
        n = int(x_taken.shape[0])
        if self.cfg.cap_policy == "drop":
            self._note_cap_blocked(n)
            return None
        st = self._cap_state()
        parent_rec = self.experts[parent_eid]
        valid = y_taken >= 0
        parent_rec.meta["sig"].update(x_taken)
        parent_rec.meta["sig"].fit()
        if parent_rec.replay is not None:                              # an archived expert has no replay yet; it gets a fresh one when it revives
            parent_rec.replay.add(x_taken[valid], y_taken[valid])
        st["shared"] += 1
        st["shared_samples"] += n
        self._warn_cap_once()
        self.events.add(kind="cap_share", step=self.step, expert=parent_eid, samples=n, experts=len(self.experts), cap=self.cfg.max_experts, affinity=aff)
        if novel_indices is not None and len(novel_indices) > 0 and x_all is not None and assigned is not None:
            near = parent_rec.meta["sig"].pvalue(x_all[novel_indices]) >= self.cfg.q
            if near.any():
                assigned[novel_indices[near]] = parent_eid
        return None

    def _count_infant_suppression(self, x: Tensor, P: Tensor, eids: List[int]) -> None:
        """Read-only counters of what the infants' relaxed gate hides (nothing here changes the pool). In every observe call that has an infant: ``novel_to_mature`` adds the samples the mature
        experts alone flag as novel (Benjamini-Hochberg on their best p-value) and ``suppressed`` adds the samples that the batch would have flagged had the infants' p-values not been multiplied
        by ``infant_relax`` but that the relaxed gate let through. Always suppressed <= novel_to_mature (BH is monotone in the p-values); ``infant_relax`` = 1 gives 0. Exact for the exact gate;
        with ``gate_shortlist`` > 0 the mature columns outside the shortlist are 0, so both counts are upper bounds there."""
        infants = [k for k, e in enumerate(eids) if self.experts[e].state == "infant"]
        if not infants:
            return
        st = self._infant_state()
        st["batches_with_infant"] += 1
        if len(infants) == len(eids) or self.cfg.no_conformal:
            return
        is_infant = torch.zeros(len(eids), dtype=torch.bool)
        is_infant[infants] = True
        q = self.cfg.q
        novel_mature = benjamini_hochberg(P[:, ~is_infant].max(dim=1).values, q)
        st["novel_to_mature"] += int(novel_mature.sum())
        if self.cfg.infant_relax <= 1.0 or not bool(novel_mature.any()):
            return
        plain = P.clone()
        for k in infants:
            plain[:, k] = self.experts[eids[k]].meta["sig"].pvalue(x)             # the infant's p-value without the relaxation
        flagged_plain = benjamini_hochberg(plain.max(dim=1).values, q)
        flagged = benjamini_hochberg(P.max(dim=1).values, q)
        st["suppressed"] += int((flagged_plain & ~flagged).sum())

    def redundancy_report(self, limit: int = 64) -> dict:
        """Which pairs of experts cover the same inputs. Read-only: nothing is merged, moved, trained or deleted and no event is written.

        ``overlap`` is the shipped merge rule's number: the mean conformal p-value of one expert's prototypes under the other's signature, the smaller of the two directions (about 0.5 for two samples
        of one region, near the floor 1/(n+1) for disjoint regions). Pairs at or above ``merge_overlap`` are listed with their ``agreement``: the share of both experts' prototypes on which the two
        predict the same class. Overlap says the inputs coincide, not that the functions do: a high overlap with a low agreement is two different functions on shared inputs, and neither may be evicted
        for it. Only the ``limit`` experts with the most prototypes are compared (the cost grows with limit squared); ``truncated`` says whether others were left out."""
        eids = sorted(e for e, r in self.experts.items() if r.meta["sig"].protos is not None and len(r.meta["sig"].protos) >= 2)
        truncated = len(eids) > limit
        if truncated:
            eids = sorted(sorted(eids, key=lambda e: -len(self.experts[e].meta["sig"].protos))[:limit])
        pairs = []
        for a, i in enumerate(eids):
            sig_i = self.experts[i].meta["sig"]
            for j in eids[a + 1:]:
                sig_j = self.experts[j].meta["sig"]
                overlap = min(float(sig_j.pvalue(sig_i.protos).mean()), float(sig_i.pvalue(sig_j.protos).mean()))
                if overlap < self.cfg.merge_overlap:
                    continue
                x = torch.cat([sig_i.protos, sig_j.protos])
                with torch.no_grad():
                    agreement = float((self.experts[i].module(x).argmax(-1) == self.experts[j].module(x).argmax(-1)).float().mean())
                pairs.append({"a": i, "b": j, "overlap": overlap, "agreement": agreement})
        return {"experts": len(self.experts), "compared": len(eids), "pairs": pairs, "truncated": truncated}

    # ------------------------------------------------------------------------------------------ routing v2 stage 3a: the batch-level gate
    @property
    def set_stats(self) -> dict:
        """Counters of the batch-level gate (tests, early stops, permutations, verdicts, overrides, pauses, breaker openings, window splits). No wall-clock."""
        return self._set.stats

    def _known_batch(self, rec, preq_losses: Tensor) -> bool:
        """E-063 skip rule: True when the expert is settled (active, trained on at least ``skip_min_assigned`` samples) and the mean loss it had on this batch BEFORE any step is at most ``skip_known`` x the chance
        loss ln(classes it has seen). Always False with ``skip_known`` None (the default)."""
        k = self.cfg.skip_known
        if k is None or rec.state != "active" or rec.n_assigned < self.cfg.skip_min_assigned:
            return False
        return float(preq_losses.mean()) <= k * math.log(max(int(rec.module.seen.sum()), 2))

    def _absorb_only(self, eid: int, x: Tensor, y: Tensor) -> None:
        """What ``train_expert`` does besides the optimiser steps: the batch enters the expert's reservoir and its counters."""
        rec = self.experts[eid]
        rec.replay.add(x, y)
        rec.n_assigned += x.shape[0]
        rec.last_active = self.step

    def summary(self) -> dict:
        out = super().summary()
        out["skip"] = dict(self.skip_stats)
        if self.cfg.set_gate:
            out["set_stats"] = dict(self._set.stats)
        cap = self._cap_state()                                        # stage 4.1: what a full pool did, and what the infants' relaxed gate hid
        out["cap"] = {"policy": self.cfg.cap_policy, "cap": self.cfg.max_experts, "reached_step": cap["reached_step"], "blocked": cap["blocked"],
                      "blocked_samples": cap["blocked_samples"], "shared": cap["shared"], "shared_samples": cap["shared_samples"]}
        out["infant"] = dict(self._infant_state())
        return out

    def _windows_of(self, rec) -> SourceWindows:
        """The expert's source windows, created on first use (so a pool that never enables the set gate never holds any)."""
        w = rec.meta.get("windows")
        if w is None:
            c = self.cfg
            w = rec.meta["windows"] = SourceWindows(
                max_windows=c.set_gate_windows, max_batches=c.set_gate_window_batches, min_ref=c.set_gate_min_ref, split_alpha=c.set_gate_split_alpha,
                n_perm=c.set_gate_perms, chunk=c.set_gate_chunk, stat=c.set_gate_stat, max_rows=c.set_gate_max_rows)
        return w

    def _update_windows(self, rec, xk: Tensor) -> None:
        """Add the samples a trained batch sent to this expert to its source windows."""
        if self._windows_of(rec).update(xk, generator(self.cfg.seed, 0x57494E, self.step, rec.id), self.step):    # 0x57494E = "WIN"
            self._set.stats["window_splits"] += 1

    def _set_refs(self, eid: int):
        """The expert's references for the set test as (rows, mean) pairs: one per source window with enough rows, else its stored prototypes (a young
        expert's prototypes are its training samples). An expert with neither has no reference and can never be admitted."""
        rec = self.experts[eid]
        w = rec.meta.get("windows")
        pairs = w.ref_pairs() if w is not None else []
        if pairs:
            return pairs
        protos = rec.meta["sig"].protos
        return [(protos, protos.mean(0))] if protos is not None and len(protos) > 0 else []

    def _set_loss(self, eid: int, x: Tensor, y: Tensor) -> float:
        with torch.no_grad():
            return float(F.cross_entropy(self.experts[eid].module(x), y))

    def _set_gate_step(self, x: Tensor, y_t: Optional[Tensor], assigned: Tensor, eids: List[int]):
        """Routing v2 stage 3a. The samples the per-sample gate accepted (and the batch vote assigned) are tested as a SET against the experts' source
        windows. Returns ``(assigned, set_novel_mask or None, verdict or None)``: novel as a set -> the accepted samples are unassigned and the mask tells the
        caller to send them to the probation buffer; known -> one expert trains all of them. Samples the per-sample gate flagged are never touched.
        Persistence (``set_gate_confirm`` = c > 1, amendment A-15): the first c - 1 doubtful decided batches in a row return "held": their accepted samples are
        unassigned and wait in ``self._held`` (``observe`` adds the flagged samples to the same list instead of the probation buffer); the c-th doubtful batch
        returns "novel" and the held samples join its cluster; a known batch drops them. The circuit breaker always sees the raw verdict."""
        gate, cfg = self._set, self.cfg
        accepted = assigned >= 0
        n_acc = int(accepted.sum())
        if n_acc < cfg.batch_vote_min:
            return assigned, None, None
        if gate.paused(self.step):
            gate.stats["paused"] += 1
            return assigned, None, "paused"
        ids, counts = torch.unique(assigned[accepted], return_counts=True)
        vote_owner = int(ids[counts == counts.max()].min())
        dec = gate.decide(x[accepted], y_t[accepted] if y_t is not None else None, eids, self._set_refs, self._set_loss, self.step)
        verdict = gate.confirm(dec.verdict)                            # persistence (A-15): "known", "held" or "novel" (acted on)
        out = assigned.clone()
        if verdict == "held":
            out[accepted] = -1
            gate.stats["held"] += 1
            self._hold(x, y_t, accepted)
            self.events.add(kind="set_held", step=self.step, n=n_acc, tried=dec.tried)
        elif verdict == "novel":
            self._release_held()
            out[accepted] = -1
            self.events.add(kind="set_novel", step=self.step, n=n_acc, tried=dec.tried)
        else:
            self._drop_held()
            out[accepted] = dec.owner
            if dec.owner != vote_owner:
                gate.stats["override"] += 1
                self.events.add(kind="set_override", step=self.step, n=n_acc, vote=vote_owner, owner=dec.owner, admitted=dec.admitted)
        if gate.note(dec.verdict, n_acc / assigned.shape[0], self.step):
            self.events.add(kind="breaker_open", step=self.step, until=gate.pause_until)
        return out, (accepted if verdict == "novel" else None), verdict

    _HOLD_TTL = 3                                                      # a held batch is only joined by the next decided batch if it comes within this many steps

    def _hold(self, x: Tensor, y_t: Optional[Tensor], accepted: Tensor) -> None:
        """Persistence (A-15): the accepted samples of a doubtful batch wait (the flagged ones follow in ``observe``)."""
        if self.step - self._held_step > self._HOLD_TTL:               # an older, never-resolved hold is stale
            self._held = []
        start = self.samples_seen - x.shape[0]
        self._held.append((x[accepted].clone(), y_t[accepted].clone() if y_t is not None else None, start + torch.where(accepted)[0]))
        self._held_step = self.step

    def _release_held(self) -> None:
        """The next decided batch is doubtful too: the held samples join the probation buffer (and so the cluster of this batch) if they are still fresh."""
        if self._held and self.step - self._held_step <= self._HOLD_TTL:
            for px, py, pa in self._held:
                self.probation.add(px, py, arrival=pa)
            self._set.stats["released"] += 1
        elif self._held:
            self._set.stats["dropped"] += 1
        self._held = []

    def _drop_held(self) -> None:
        """The next decided batch is known: the doubt was a false alarm and the held samples are dropped."""
        if self._held:
            self._set.stats["dropped"] += 1
        self._held = []

    # ------------------------------------------------------------------------------------------ routing v2 stage 3b: group answering
    def predict_grouped(self, x: Tensor, groups: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Answer ``x`` with the inputs of every group answered together. ``groups`` is a LongTensor of OPAQUE group ids chosen by the caller (one document, one
        session, one batch; negative = not grouped); they carry no regime or label information. With ``cfg.group_vote`` every group of at least ``cfg.group_min``
        inputs is answered by the expert most of its inputs chose on their own (routing v2 stage 3b); with the flag off, or an empty pool, the answer is
        ``predict(x, ctx)`` itself. ``predict`` is not changed: its signature is pinned by tests/growmoe/test_gm_phylo.py (no side channel into the pool)."""
        pred = self.predict(x, ctx)
        if not self.cfg.group_vote or len(self.experts) == 0:
            return pred
        return self._answer_groups(x, pred, groups)

    def _answer_groups(self, x: Tensor, pred: Prediction, groups: Tensor) -> Prediction:
        """Re-answer the inputs of every large enough group with the group's expert (see ``growmoe.groupvote``). The veto uses EXACT p-values from
        ``_gate_column``, never the cascade's restricted matrix (which holds 0 for experts outside an input's shortlist)."""
        cfg = self.cfg
        groups = torch.as_tensor(groups)
        xn = self._normalize(x.float())

        def accepts(eid: int, idx: Tensor) -> Tensor:
            p, logf = self._gate_column(eid, xn[idx])
            return (logf >= 0.0) if cfg.no_conformal else (p >= cfg.group_veto_q)

        new_expert, info = vote_groups(pred.expert, groups, cfg.group_min, accepts if cfg.group_veto else None)
        moved = new_expert != pred.expert
        probs = pred.probs
        if moved.any():
            probs = pred.probs.clone()
            with torch.no_grad():
                for eid in torch.unique(new_expert[moved]).tolist():
                    m = moved & (new_expert == eid)
                    probs[m] = self.experts[eid].module.proba(xn[m])
        return Prediction(probs=probs, expert=new_expert, p_nov=pred.p_nov, unknown=pred.unknown, meta={**pred.meta, "group_vote": info})

    _FAST_RULES = ("accept_then_logf", "argmax_logf", "argmax_p", "zscore_logf", "nearest_proto")

    def reset_gate_stats(self) -> None:
        """Multiply-adds spent by the gate, counted per call site ('observe' while learning, 'predict' while answering)."""
        self.gate_stats = {ph: {"samples": 0, "stage1": 0, "stage2": 0, "second_look": 0} for ph in ("observe", "predict")}

    def _gate_column(self, eid: int, x: Tensor):
        """Conformal p-values and log-densities of ``x`` under one expert's signature (infants get the relaxed gate). The score is computed ONCE
        (the reference loop computed it twice, once inside ``pvalue`` and once inside ``logpdf``); every value is bit-identical."""
        rec = self.experts[eid]
        sig = rec.meta["sig"]
        if isinstance(sig, VmfSignature):
            s = sig.score(x)
            p = conformal_pvalues(sig.cal_scores, s)
            logf = s + sig._log_c[sig._kappa]
        else:
            p, logf = sig.pvalue(x), sig.logpdf(x)
        if rec.state == "infant":
            p = torch.clamp(p * self.cfg.infant_relax, max=1.0)
        return p, logf

    def _gate_exact(self, x: Tensor, eids: List[int]):
        cols = [self._gate_column(eid, x) for eid in eids]
        return torch.stack([c[0] for c in cols], dim=1), torch.stack([c[1] for c in cols], dim=1)

    def _gate_matrices(self, x: Tensor, eids: List[int], phase: str = "predict"):
        """(B, K) conformal p-values and log-densities. With ``gate_shortlist > 0`` and enough experts the cascade of ``_gate_cascade`` is used; its
        novelty decisions equal the exact gate's (see that method). ``phase`` ('observe' or 'predict') only decides which counters are advanced."""
        stats = self.gate_stats[phase]
        stats["samples"] += x.shape[0]
        if self._cascade_ok(eids):
            return self._gate_cascade(x, eids, stats)
        stats["stage2"] += x.shape[0] * sum(self.experts[e].meta["sig"].n for e in eids) * self.cfg.d
        return self._gate_exact(x, eids)

    def _cascade_ok(self, eids: List[int]) -> bool:
        """The shortlist needs the conformal gate, vMF signatures with at least two prototypes, and enough experts to be worth it."""
        cfg = self.cfg
        if cfg.gate_shortlist <= 0 or cfg.no_conformal or len(eids) < max(cfg.gate_shortlist_min_experts, cfg.gate_shortlist + 1):
            return False
        return all(isinstance(self.experts[e].meta["sig"], VmfSignature) and self.experts[e].meta["sig"].n >= 2 for e in eids)

    def _fingerprints(self, eids: List[int]) -> Tensor:
        """(E, k, d) unit sub-centroids, one block per expert; an expert's block is rebuilt when its signature has seen ``gate_fingerprint_refresh`` more points."""
        k, refresh = self.cfg.gate_fingerprint_k, self.cfg.gate_fingerprint_refresh
        blocks = []
        for e in eids:
            rec = self.experts[e]
            fp = rec.meta.get("fp")
            if fp is None or fp.stale(rec.meta["sig"], refresh):
                fp = Fingerprint.build(rec.meta["sig"], k, seed=self.cfg.seed * 1000 + e)
                rec.meta["fp"] = fp
            blocks.append(fp.padded(k))
        return torch.stack(blocks)

    def _gate_cascade(self, x: Tensor, eids: List[int], stats: dict):
        """Two-stage gate. (1) Stage 1 scores x against every expert's sub-centroids (best cosine). (2) The top ``gate_shortlist`` experts per input
        (plus every infant) get the exact score. Pairs outside the shortlist stay P = 0, LogF = -inf. (3) Second look: every input whose restricted
        p_nov is at most q gets the exact scan over ALL experts.

        Guarantee: the Benjamini-Hochberg novelty flags equal the exact gate's. BH at level q rejects using only p-values <= q (a p above q can never
        satisfy p_(k) <= kq/B); restricted p <= exact p, and every input whose restricted p <= q is recomputed exactly, so the entries <= q coincide and all
        others exceed q in both versions. Only the choice among accepting experts is approximate."""
        cfg, d = self.cfg, self.cfg.d
        b, n_exp = x.shape[0], len(eids)
        cents = self._fingerprints(eids)                                                  # (E, k, d)
        k = cents.shape[1]
        stage1 = (x @ cents.reshape(n_exp * k, d).T).reshape(b, n_exp, k).amax(dim=2)     # (B, E)
        stats["stage1"] += b * n_exp * k * d
        need = torch.zeros(b, n_exp, dtype=torch.bool)
        need.scatter_(1, stage1.topk(cfg.gate_shortlist, dim=1).indices, True)
        for c, e in enumerate(eids):
            if self.experts[e].state == "infant":                                         # infants are few and small and must attract their regime
                need[:, c] = True
        P = torch.zeros(b, n_exp)
        LogF = torch.full((b, n_exp), -math.inf)
        for c, e in enumerate(eids):
            rows = need[:, c].nonzero().squeeze(1)
            if rows.numel():
                P[rows, c], LogF[rows, c] = self._gate_column(e, x[rows])
                stats["stage2"] += int(rows.numel()) * self.experts[e].meta["sig"].n * d
        look = (P.max(dim=1).values <= cfg.q * 1.001).nonzero().squeeze(1)              # the 1.001 only enlarges the set (safe side)
        if look.numel():
            P[look], LogF[look] = self._gate_exact(x[look], eids)
            stats["second_look"] += int(look.numel()) * sum(self.experts[e].meta["sig"].n for e in eids) * d
        return P, LogF

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Predict class probabilities and novelty without learning or signature updates.

        Vectorised over samples. Returns exactly what ``_predict_reference`` returns (asserted by
        tests/growmoe/test_gm_m0_fast_predict.py for every routing rule); the supersede-filtering ``default`` rule and an empty
        pool use the reference code.
        """
        route_rule = getattr(self, "route_rule", getattr(self.cfg, "route_rule", "default"))
        if route_rule not in self._FAST_RULES or len(self.experts) == 0:
            if self.cfg.router and route_rule not in self._FAST_RULES and len(self.experts) > 0:
                raise ValueError(f"cfg.router needs a fast route_rule, got {route_rule!r}: the reference path is router-blind")
            return self._predict_reference(x, ctx)

        x = self._normalize(x.float())
        b = x.shape[0]
        eids = list(self.experts.keys())
        P, LogF = self._gate_matrices(x, eids)

        if self.cfg.no_conformal:
            max_logf = torch.max(LogF, dim=1).values
            unknown = max_logf < 0.0
            p_nov = torch.sigmoid(max_logf)
        else:
            p_nov = torch.max(P, dim=1).values
            unknown = benjamini_hochberg(p_nov, self.cfg.q)

        if route_rule == "argmax_p":
            idx = P.argmax(dim=1)
        elif route_rule == "argmax_logf":
            idx = LogF.argmax(dim=1)
        elif route_rule == "zscore_logf":
            mu_k, std_k = [], []
            for eid in eids:
                cal = self.experts[eid].meta["sig"].cal_scores
                if cal is not None and len(cal) > 1:
                    mu_k.append(float(cal.float().mean().item()))
                    std_k.append(float(cal.float().std().item()) + 1e-6)
                else:
                    mu_k.append(0.0)
                    std_k.append(1.0)
            mu_tensor = torch.tensor(mu_k, dtype=torch.float32, device=LogF.device)
            std_tensor = torch.tensor(std_k, dtype=torch.float32, device=LogF.device)
            idx = ((LogF - mu_tensor.unsqueeze(0)) / std_tensor.unsqueeze(0)).argmax(dim=1)
        elif route_rule == "nearest_proto":
            sim_list = []
            for eid in eids:
                pr = self.experts[eid].meta["sig"].protos
                if pr is not None and len(pr) > 0:
                    pr_norm = pr / pr.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    sim_list.append((x @ pr_norm.T).max(dim=1).values)
                else:
                    sim_list.append(torch.full((b,), -1.0, device=x.device))
            idx = torch.stack(sim_list, dim=1).argmax(dim=1)
        else:                                                                           # accept_then_logf
            if self.cfg.no_conformal:
                idx = LogF.argmax(dim=1)
            else:
                eligible = P.double() >= float(self.cfg.q)                              # same float64 comparison as the loop's .item()
                best_eligible = LogF.masked_fill(~eligible, float("-inf")).argmax(dim=1)
                idx = torch.where(eligible.any(dim=1), best_eligible, LogF.argmax(dim=1))

        if self.cfg.router:                                                             # routing v2 stage 4a: the selected rule replaces the choice among experts, never the known-or-unknown decision
            idx = self._route_by_router(x, eids, P, LogF, idx)

        routed_expert = torch.tensor(eids, dtype=torch.long)[idx]
        probs = torch.zeros((b, self.cfg.n_classes), dtype=torch.float32)
        with torch.no_grad():
            for eid in eids:
                mask_k = (routed_expert == eid)
                if mask_k.any():
                    probs[mask_k] = self.experts[eid].module.proba(x[mask_k])
        return Prediction(probs=probs, expert=routed_expert, p_nov=p_nov, unknown=unknown)

    def _predict_reference(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """The original per-sample loop. Kept as the oracle that tests/growmoe/test_gm_m0_fast_predict.py compares ``predict`` to."""
        route_rule = getattr(self, "route_rule", getattr(self.cfg, "route_rule", "default"))
        x = self._normalize(x.float())
        b = x.shape[0]

        if len(self.experts) == 0:
            probs = torch.full(
                (b, self.cfg.n_classes),
                1.0 / self.cfg.n_classes,
                dtype=torch.float32,
            )
            expert = torch.full((b,), -1, dtype=torch.long)
            p_nov = torch.zeros(b, dtype=torch.float32)
            unknown = torch.ones(b, dtype=torch.bool)
            return Prediction(probs=probs, expert=expert, p_nov=p_nov, unknown=unknown)

        eids = list(self.experts.keys())
        p_list = []
        logf_list = []

        for eid in eids:
            rec = self.experts[eid]
            sig: VmfSignature = rec.meta["sig"]
            p_k = sig.pvalue(x)
            if rec.state == "infant":
                p_k = torch.clamp(p_k * self.cfg.infant_relax, max=1.0)
            logf_k = sig.logpdf(x)
            p_list.append(p_k)
            logf_list.append(logf_k)

        P = torch.stack(p_list, dim=1)
        LogF = torch.stack(logf_list, dim=1)

        if self.cfg.no_conformal:
            max_logf = torch.max(LogF, dim=1).values
            fixed_thresh = 0.0
            unknown = max_logf < fixed_thresh
            p_nov = torch.sigmoid(max_logf)
        else:
            p_nov = torch.max(P, dim=1).values
            unknown = benjamini_hochberg(p_nov, self.cfg.q)

        routed_expert = torch.zeros(b, dtype=torch.long)
        if route_rule == "argmax_p":
            for i in range(b):
                routed_expert[i] = eids[int(torch.argmax(P[i]).item())]
        elif route_rule == "argmax_logf":
            for i in range(b):
                routed_expert[i] = eids[int(torch.argmax(LogF[i]).item())]
        elif route_rule == "zscore_logf":
            # Per-expert z-scored log-density against own prototype calibration scores
            mu_k = []
            std_k = []
            for eid in eids:
                cal = self.experts[eid].meta["sig"].cal_scores
                if cal is not None and len(cal) > 1:
                    mu_k.append(float(cal.float().mean().item()))
                    std_k.append(float(cal.float().std().item()) + 1e-6)
                else:
                    mu_k.append(0.0)
                    std_k.append(1.0)
            mu_tensor = torch.tensor(mu_k, dtype=torch.float32, device=LogF.device)
            std_tensor = torch.tensor(std_k, dtype=torch.float32, device=LogF.device)
            Z = (LogF - mu_tensor.unsqueeze(0)) / std_tensor.unsqueeze(0)
            for i in range(b):
                routed_expert[i] = eids[int(torch.argmax(Z[i]).item())]
        elif route_rule == "nearest_proto":
            # Highest cosine similarity to any prototype of expert k
            sim_list = []
            for eid in eids:
                pr = self.experts[eid].meta["sig"].protos
                if pr is not None and len(pr) > 0:
                    pr_norm = pr / pr.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                    s_k = (x @ pr_norm.T).max(dim=1).values
                else:
                    s_k = torch.full((b,), -1.0, device=x.device)
                sim_list.append(s_k)
            Sim = torch.stack(sim_list, dim=1)
            for i in range(b):
                routed_expert[i] = eids[int(torch.argmax(Sim[i]).item())]
        elif route_rule == "accept_then_logf":
            for i in range(b):
                if self.cfg.no_conformal:
                    routed_expert[i] = eids[int(torch.argmax(LogF[i]).item())]
                else:
                    elig = [k for k in range(len(eids)) if P[i, k].item() >= self.cfg.q]
                    if elig:
                        el_eids = [eids[k] for k in elig]
                        routed_expert[i] = max(el_eids, key=lambda e: LogF[i, eids.index(e)].item())
                    else:
                        routed_expert[i] = eids[int(torch.argmax(LogF[i]).item())]
        else:
            # Default conformal-gated routing
            for i in range(b):
                if self.cfg.no_conformal:
                    best_eid = eids[int(torch.argmax(LogF[i]).item())]
                    routed_expert[i] = best_eid
                else:
                    threshold_i = min(self.cfg.q, p_nov[i].item())
                    eligible_idx = [k for k in range(len(eids)) if P[i, k].item() >= threshold_i]
                    if not eligible_idx:
                        eligible_idx = [int(torch.argmax(P[i]).item())]
                    eligible_eids = [eids[idx] for idx in eligible_idx]
                    active_candidates = self._filter_superseded(eligible_eids)
                    if not active_candidates:
                        active_candidates = eligible_eids
                    best_eid = max(active_candidates, key=lambda e: LogF[i, eids.index(e)].item())
                    routed_expert[i] = best_eid

        probs = torch.zeros((b, self.cfg.n_classes), dtype=torch.float32)
        for eid in eids:
            mask_k = (routed_expert == eid)
            if mask_k.any():
                probs[mask_k] = self.experts[eid].module.proba(x[mask_k]).detach()

        return Prediction(probs=probs, expert=routed_expert, p_nov=p_nov, unknown=unknown)
