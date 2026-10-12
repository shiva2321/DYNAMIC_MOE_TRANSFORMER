"""Task 16: Architecture C (Keyed Delta-MoE).

Features:
- Every expert is a DeltaExpert sharing a frozen base MLP.
- Per-expert key vector K_e (unit vector), updated online via EMA of accepted sample means.
- Gate and novelty testing via z-score of cos(K_e, u) on each expert's accepted samples:
    z_e(u) = (K_e . u - mu_e) / sigma_e
    novel if max_e z_e(u) < -c
- Novel samples accumulate in Short-Term Memory (STM).
- When |STM| >= n_stm:
  - If pool empty: train base MLP on STM, freeze it, spawn founding expert 0.
  - If pool non-empty: fork child DeltaExpert from nearest key, train on STM.
- Sibling delta merge: merge sibling experts with high key cosine via SVD averaging.
- Archive and revive: dormant after dormant_after, archived after archive_after;
  archived experts keep their keys in the gate and revive when assigned.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..accounting import unique_param_count
from ..config import KeyedCfg
from ..delta import DeltaExpert
from ..experts import MLPExpert
from ..replay import Reservoir
from ..types import Prediction, StepRecord
from .base import BasePool


class KeyedDeltaMoE(BasePool):
    """Architecture C: Keyed Delta-MoE with shared frozen base MLP."""

    name = "C_keyed_delta"

    def __init__(self, cfg: Optional[KeyedCfg] = None) -> None:
        cfg = cfg if cfg is not None else KeyedCfg()
        super().__init__(cfg)
        self.cfg: KeyedCfg = cfg

        # Shared frozen base MLP
        self.base_mlp: Optional[MLPExpert] = None

        # Short-Term Memory (STM)
        self.stm_x: List[Tensor] = []
        self.stm_y: List[Tensor] = []

        # Per-expert keys and Welford statistics of cos(K_e, u)
        self.keys: Dict[int, Tensor] = {}
        self._cos_mu: Dict[int, float] = {}
        self._cos_m2: Dict[int, float] = {}
        self._cos_n: Dict[int, int] = {}
        self._cos_sigma: Dict[int, float] = {}

    def _normalize(self, x: Tensor) -> Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return x / norm

    def _update_expert_welford(self, eid: int, cos_vals: np.ndarray) -> None:
        """Update online mean and variance of cosines for expert eid."""
        mu = self._cos_mu.get(eid, 0.0)
        m2 = self._cos_m2.get(eid, 0.0)
        n = self._cos_n.get(eid, 0)

        for val in cos_vals:
            n += 1
            delta = val - mu
            mu += delta / n
            delta2 = val - mu
            m2 += delta * delta2

        self._cos_mu[eid] = mu
        self._cos_m2[eid] = m2
        self._cos_n[eid] = n
        if n > 1:
            self._cos_sigma[eid] = max(math.sqrt(m2 / (n - 1)), 1e-4)
        else:
            self._cos_sigma[eid] = 1.0

    def _spawn_from_stm(self, x_stm: Tensor, y_stm: Tensor) -> int:
        next_id = self._next_id
        n_pts = len(x_stm)
        mean_u = self._normalize(x_stm.mean(dim=0, keepdim=True)).squeeze(0)

        valid_mask = y_stm >= 0
        rng_stm = np.random.default_rng([self.cfg.seed, 0x4B4559, next_id])

        # Empty pool case: create and train shared base MLP on bootstrap regime
        if self.base_mlp is None:
            self.base_mlp = MLPExpert(
                d_in=self.cfg.d,
                hidden=self.cfg.hidden,
                depth=self.cfg.depth,
                n_classes=self.cfg.n_classes,
                seed=self.cfg.seed,
            )
            # Train base MLP on STM if labels exist
            if valid_mask.any():
                valid_x = x_stm[valid_mask]
                valid_y = y_stm[valid_mask]
                self.base_mlp.mark_seen(valid_y)
                opt_base = torch.optim.Adam(self.base_mlp.parameters(), lr=self.cfg.lr)
                n_val = len(valid_x)
                for _ in range(self.cfg.base_steps):
                    idx = torch.from_numpy(rng_stm.integers(0, n_val, min(self.cfg.batch_size, n_val)))
                    self.base_mlp.train_step(valid_x[idx], valid_y[idx], opt_base)

            # Freeze base MLP
            for p in self.base_mlp.parameters():
                p.requires_grad = False

            # Founding DeltaExpert
            child_expert = DeltaExpert(
                base=self.base_mlp,
                rank=self.cfg.rank,
                lineage=[],
                seed=self.cfg.seed * 1000 + next_id,
            )
            parent_id = None
            init_mode = "bootstrap"
        else:
            # Fork from nearest live expert key
            live_eids = list(self.keys.keys())
            key_stack = torch.stack([self.keys[e] for e in live_eids], dim=0)  # (K, d)
            affinities = key_stack @ mean_u
            best_idx = int(affinities.argmax().item())
            parent_id = live_eids[best_idx]

            parent_expert = self.experts[parent_id].module
            child_expert = parent_expert.fork(seed=self.cfg.seed * 1000 + next_id)
            init_mode = "fork"

        # Fit child DeltaExpert on STM if labels exist
        if valid_mask.any():
            valid_x = x_stm[valid_mask]
            valid_y = y_stm[valid_mask]
            child_expert.mark_seen(valid_y)
            opt_child = self.make_optimizer(child_expert, 1.0)
            n_val = len(valid_x)
            for _ in range(self.cfg.inner_steps * 10):
                idx = torch.from_numpy(rng_stm.integers(0, n_val, min(self.cfg.batch_size, n_val)))
                child_expert.train_step(valid_x[idx], valid_y[idx], opt_child)

        # Register expert with BasePool
        rec = self.add_expert(
            child_expert,
            parent=parent_id,
            state="active",
            meta={"key": mean_u.clone()},
        )

        if valid_mask.any():
            rec.replay.add(x_stm[valid_mask], y_stm[valid_mask])
        else:
            rec.replay.add(x_stm, y_stm)

        # Initialize key and Welford statistics
        self.keys[rec.id] = mean_u.clone()
        cos_initial = (x_stm @ mean_u).cpu().numpy().astype(np.float64)
        self._update_expert_welford(rec.id, cos_initial)

        # Log spawn event
        self.events.add(
            "spawn",
            self.step,
            expert=rec.id,
            parent=parent_id,
            affinity=0.0 if parent_id is None else float(affinities[best_idx].item()),
            evidence=0.0,
            n_cluster=n_pts,
            init=init_mode,
            lam=0.0,
        )

        return rec.id

    def _check_sibling_merges(self) -> None:
        """Check and execute sibling delta merges for active experts with high key similarity."""
        active_eids = [eid for eid, rec in self.experts.items() if rec.state == "active" and rec.parent is not None]
        if len(active_eids) < 2:
            return

        merged_pairs = set()
        for i in range(len(active_eids)):
            eid_a = active_eids[i]
            if eid_a in merged_pairs or eid_a not in self.experts:
                continue
            for j in range(i + 1, len(active_eids)):
                eid_b = active_eids[j]
                if eid_b in merged_pairs or eid_b not in self.experts:
                    continue

                # Must share the same parent to be siblings
                if self.experts[eid_a].parent != self.experts[eid_b].parent:
                    continue

                # Check key cosine similarity
                cos_ab = float(torch.dot(self.keys[eid_a], self.keys[eid_b]).item())
                if cos_ab >= self.cfg.merge_sim:
                    older, younger = (eid_a, eid_b) if eid_a < eid_b else (eid_b, eid_a)
                    # Merge younger into older
                    rec_older = self.experts[older]
                    rec_younger = self.experts[younger]

                    rec_older.module.merge_siblings(rec_younger.module)
                    # Update key
                    new_key = self._normalize(0.5 * (self.keys[older] + self.keys[younger]))
                    self.keys[older] = new_key
                    rec_older.meta["key"] = new_key

                    # Combine replay
                    if rec_younger.replay.n > 0:
                        xy_y = rec_younger.replay.sample(rec_younger.replay.n)
                        rec_older.replay.add(xy_y[0], xy_y[1])

                    # Remove younger and reparent children
                    self.remove_expert(younger)
                    self.keys.pop(younger, None)
                    for child_id, parent_id in list(self._parents.items()):
                        if parent_id == younger:
                            self.set_parent(child_id, older)

                    self.events.add(
                        "merge",
                        self.step,
                        kept=older,
                        removed=younger,
                        change_kept=0.0,
                        change_removed=0.0,
                    )
                    merged_pairs.add(younger)
                    break

    def observe(self, x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None) -> StepRecord:
        """Observe a batch (x, y); evaluate key gate, update STM / keys / Welford stats."""
        x = self._normalize(x.float())
        b = x.shape[0]
        y_t = torch.as_tensor(y, dtype=torch.long) if y is not None else None

        spawned: List[int] = []

        # Empty pool case: buffer into STM until n_stm is reached
        if len(self.experts) == 0:
            for i in range(b):
                self.stm_x.append(x[i])
                if y_t is not None:
                    self.stm_y.append(y_t[i])
                else:
                    self.stm_y.append(torch.tensor(-1, dtype=torch.long))

            if len(self.stm_x) >= self.cfg.n_stm:
                all_stm_x = torch.stack(self.stm_x, dim=0)
                all_stm_y = torch.stack(self.stm_y, dim=0)
                self.stm_x.clear()
                self.stm_y.clear()
                new_eid = self._spawn_from_stm(all_stm_x, all_stm_y)
                spawned.append(new_eid)
                assigned = torch.full((b,), new_eid, dtype=torch.long)
                novel = torch.ones(b, dtype=torch.bool)
                p_nov = torch.zeros(b, dtype=torch.float32)
            else:
                assigned = torch.full((b,), -1, dtype=torch.long)
                novel = torch.ones(b, dtype=torch.bool)
                p_nov = torch.zeros(b, dtype=torch.float32)

            self.tick()
            return StepRecord(
                novel=novel,
                p_nov=p_nov,
                assigned=assigned,
                loss=float("nan"),
                spawned=spawned,
                n_experts=len(self.experts),
            )

        # Gate scoring: z_e = (cos(K_e, u) - mu_e) / sigma_e
        eids = sorted(self.keys.keys())
        key_mat = torch.stack([self.keys[eid] for eid in eids], dim=0)  # (K, d)
        cosines = x @ key_mat.t()  # (B, K)

        z_scores = torch.zeros_like(cosines)
        for j, eid in enumerate(eids):
            mu_j = self._cos_mu.get(eid, 0.0)
            sigma_j = max(self._cos_sigma.get(eid, 1.0), 1e-4)
            z_scores[:, j] = (cosines[:, j] - mu_j) / sigma_j

        best_z, best_idx = z_scores.max(dim=1)
        best_eids = [eids[idx.item()] for idx in best_idx]

        is_novel = (best_z < -self.cfg.c).cpu()
        margin = (best_z + self.cfg.c).cpu()
        p_nov = torch.sigmoid(margin)

        assigned = torch.full((b,), -1, dtype=torch.long)
        for i in range(b):
            if not is_novel[i]:
                assigned[i] = best_eids[i]

        # Handle novel samples -> STM
        if is_novel.any():
            nov_idx = torch.nonzero(is_novel).flatten()
            for idx in nov_idx:
                self.stm_x.append(x[idx])
                if y_t is not None:
                    self.stm_y.append(y_t[idx])
                else:
                    self.stm_y.append(torch.tensor(-1, dtype=torch.long))

            if len(self.stm_x) >= self.cfg.n_stm and len(self.experts) < self.cfg.max_experts:
                all_stm_x = torch.stack(self.stm_x, dim=0)
                all_stm_y = torch.stack(self.stm_y, dim=0)
                self.stm_x.clear()
                self.stm_y.clear()
                new_eid = self._spawn_from_stm(all_stm_x, all_stm_y)
                spawned.append(new_eid)
                assigned[nov_idx] = new_eid

        # Handle accepted samples -> train assigned experts, update EMA keys & Welford
        losses = []
        if (~is_novel).any():
            for eid in eids:
                mask_k = (assigned == eid)
                if not mask_k.any():
                    continue

                rec = self.experts[eid]

                # Revive if dormant or archived
                if rec.state in ("dormant", "archived"):
                    if rec.state == "archived":
                        rec.opt = self.make_optimizer(rec.module)
                        rng_rep = np.random.default_rng([self.cfg.seed, 0x524550, eid, self.step])
                        rec.replay = Reservoir(capacity=self.cfg.replay_size, d=self.cfg.d, rng=rng_rep)
                    self.set_state(eid, "active")
                    self.events.add("revive", self.step, expert=eid)

                rec.last_active = self.step
                xk = x[mask_k]

                # Update key via EMA
                mean_xk = self._normalize(xk.mean(dim=0, keepdim=True)).squeeze(0)
                eta = self.cfg.key_ema
                new_key = self._normalize((1.0 - eta) * self.keys[eid] + eta * mean_xk)
                self.keys[eid] = new_key
                rec.meta["key"] = new_key

                # Update Welford stats
                cos_k = (xk @ new_key).cpu().numpy().astype(np.float64)
                self._update_expert_welford(eid, cos_k)

                # Train classifier if labeled
                if y_t is not None:
                    yk = y_t[mask_k]
                    valid = yk >= 0
                    if valid.any():
                        l_k = self.train_expert(eid, xk[valid], yk[valid], replay=True)
                        if not math.isnan(l_k):
                            losses.append(l_k)

        # Sibling merge check
        if self.step > 0 and self.step % self.cfg.merge_every == 0:
            self._check_sibling_merges()

        # Inactivity checks: dormant / archive
        for eid, rec in list(self.experts.items()):
            idle = self.step - rec.last_active
            if idle >= self.cfg.archive_after and rec.state != "archived":
                self.set_state(eid, "archived")
                rec.replay = None
                rec.opt = None
                self.events.add("archive", self.step, expert=eid)
            elif idle >= self.cfg.dormant_after and rec.state == "active":
                self.set_state(eid, "dormant")

        mean_loss = float(np.mean(losses)) if losses else float("nan")
        self.tick()
        return StepRecord(
            novel=is_novel,
            p_nov=p_nov,
            assigned=assigned,
            loss=mean_loss,
            spawned=spawned,
            n_experts=len(self.experts),
        )

    def summary(self) -> dict:
        """Base summary with the shared frozen base counted once (the base class sums ``module.parameters()`` per expert,
        which counts the shared base once per expert and inflated Pool C's parameter total about 3x in Phase 1)."""
        out = super().summary()
        live = list(self.experts.values())
        if live:
            base = sum(p.numel() for p in self.base_mlp.parameters()) if self.base_mlp is not None else 0
            own = [r.module.num_params() for r in live]
            out["n_params"] = unique_param_count([r.module for r in live])
            out["active_params_per_sample"] = base + float(np.mean(own))
        return out

    def predict(self, x: Tensor, ctx: Optional[Tensor] = None) -> Prediction:
        """Predict class probabilities and novelty without learning or state mutation."""
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

        eids = sorted(self.keys.keys())
        key_mat = torch.stack([self.keys[eid] for eid in eids], dim=0)  # (K, d)
        cosines = x @ key_mat.t()  # (B, K)

        z_scores = torch.zeros_like(cosines)
        for j, eid in enumerate(eids):
            mu_j = self._cos_mu.get(eid, 0.0)
            sigma_j = max(self._cos_sigma.get(eid, 1.0), 1e-4)
            z_scores[:, j] = (cosines[:, j] - mu_j) / sigma_j

        best_z, best_idx = z_scores.max(dim=1)
        best_eids = torch.tensor([eids[idx.item()] for idx in best_idx], dtype=torch.long)

        margin = (best_z + self.cfg.c).cpu()
        p_nov = torch.sigmoid(margin)
        unknown = (p_nov < self.cfg.q) | (best_z < -self.cfg.c)

        probs = torch.zeros((b, self.cfg.n_classes), dtype=torch.float32)
        with torch.no_grad():
            for i in range(b):
                eid = best_eids[i].item()
                expert_mod = self.experts[eid].module
                probs[i] = expert_mod.proba(x[i:i+1]).squeeze(0)

        return Prediction(probs=probs, expert=best_eids, p_nov=p_nov, unknown=unknown)
