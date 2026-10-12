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
from ..growth import GrowthPolicy, plateau
from ..probation import Probation
from ..signature import VmfSignature
from ..types import StepRecord, Prediction
from ..replay import Reservoir
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

    @staticmethod
    def _forget_competence_history(rec) -> None:
        """Drop what the competence monitor and its recent-sample window collected so far: the baseline must come from the settled expert."""
        monitor = rec.meta.get("monitor")
        if monitor is not None:
            monitor.reset()
        recent = rec.meta.get("recent")
        if recent is not None:
            recent.clear()

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
        if force_parent is not None:
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

        if has_labels and has_affinity and has_replay:
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
            if len(self.experts) >= self.cfg.max_experts:
                return None

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

            child_sig = self.sig_cls(
                d=self.cfg.d,
                max_protos=256,
                seed=self.cfg.seed * 1000 + next_id,
            )
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
            sig = self.sig_cls(d=self.cfg.d, max_protos=256, seed=self.cfg.seed * 1000 + 0)
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
            self.tick()
            return StepRecord(
                novel=novel,
                p_nov=p_nov,
                assigned=assigned,
                loss=loss,
                spawned=[rec.id],
                n_experts=len(self.experts),
            )

        # ------------------------------------------------ 13b Gate & Routing
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

        P = torch.stack(p_list, dim=1)        # (B, K)
        LogF = torch.stack(logf_list, dim=1)  # (B, K)

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

        spawned = []

        # ------------------------------------------------ 13c Probation Buffer
        novel_indices = torch.where(novel)[0]
        if self.cfg.no_probation:
            if len(novel_indices) > 0 and len(self.experts) < self.cfg.max_experts:
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
                self.probation.add(x_nov, y_nov, arrival=arr_nov)

            ttl = int(math.ceil(self.cfg.m_min / self.cfg.min_share))
            self.probation.expire(now=self.samples_seen, ttl=ttl)

            if len(self.probation) >= self.cfg.m_min and len(self.experts) < self.cfg.max_experts:
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

                    sig_c = self.sig_cls(
                        d=self.cfg.d,
                        max_protos=256,
                        seed=self.cfg.seed * 1000 + len(self.experts),
                    )
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

                    if not alarm_fired:
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

                        # R6: Task 14 Growth inside Architecture A
                        if (
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

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.tick()
        return StepRecord(
            novel=novel,
            p_nov=p_nov,
            assigned=assigned,
            loss=mean_loss,
            spawned=spawned,
            n_experts=len(self.experts),
        )

    _FAST_RULES = ("accept_then_logf", "argmax_logf", "argmax_p", "zscore_logf", "nearest_proto")

    def _gate_matrices(self, x: Tensor, eids: List[int]):
        """(B, K) conformal p-values and log-densities. For the standard signature the score is computed ONCE per expert (the
        reference loop computed it twice, once inside ``pvalue`` and once inside ``logpdf``); every value is bit-identical."""
        p_cols, f_cols = [], []
        for eid in eids:
            rec = self.experts[eid]
            sig = rec.meta["sig"]
            if isinstance(sig, VmfSignature):
                s = sig.score(x)
                p_k = conformal_pvalues(sig.cal_scores, s)
                logf_k = s + sig._log_c[sig._kappa]
            else:
                p_k, logf_k = sig.pvalue(x), sig.logpdf(x)
            if rec.state == "infant":
                p_k = torch.clamp(p_k * self.cfg.infant_relax, max=1.0)
            p_cols.append(p_k)
            f_cols.append(logf_k)
        return torch.stack(p_cols, dim=1), torch.stack(f_cols, dim=1)

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Predict class probabilities and novelty without learning or signature updates.

        Vectorised over samples. Returns exactly what ``_predict_reference`` returns (asserted by
        tests/growmoe/test_gm_m0_fast_predict.py for every routing rule); the supersede-filtering ``default`` rule and an empty
        pool use the reference code.
        """
        route_rule = getattr(self, "route_rule", getattr(self.cfg, "route_rule", "default"))
        if route_rule not in self._FAST_RULES or len(self.experts) == 0:
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
