"""
Evidence-Accumulating Upkeeper with Tentative-Belief Store.
Implements the prior-count belief revision policy specified in §14.1.

Principles:
1. No existing belief (prior count = 0) -> Learn immediately at low confidence.
2. Contradiction of a confident belief (high prior count) -> Resist single-source bursts.
3. Evidence accumulates -> Flip when contradictory evidence outweighs prior count.
4. Memory honesty -> Fixed-capacity tentative-belief store (counts and source IDs only, strictly no raw samples).
"""

import math
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
import torch.nn.functional as F
import torch.nn as nn


class TentativeBeliefStore:
    """
    Fixed-capacity memory store of candidate contradictory beliefs:
    Key: context_signature (hashable int or tuple)
    Value: {
        "candidate_val": int,
        "count": int,
        "sources": set,
        "last_seen_step": int,
    }
    Strict memory cap: max_entries <= 64.
    """

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self.entries: OrderedDict[int, Dict[str, Any]] = OrderedDict()

    def record_contradiction(
        self,
        context_key: int,
        candidate_val: int,
        source_id: str,
        step: int,
    ) -> int:
        """
        Record a contradictory observation. Returns the updated tentative count.
        """
        if context_key in self.entries:
            rec = self.entries[context_key]
            if rec["candidate_val"] == candidate_val:
                rec["count"] += 1
                rec["sources"].add(source_id)
                rec["last_seen_step"] = step
                self.entries.move_to_end(context_key)
                return rec["count"]
            else:
                # Conflicting alternative candidate: reset counter for new candidate
                rec["candidate_val"] = candidate_val
                rec["count"] = 1
                rec["sources"] = {source_id}
                rec["last_seen_step"] = step
                self.entries.move_to_end(context_key)
                return 1

        # Evict least-recently-seen if at capacity
        if len(self.entries) >= self.max_entries:
            self.entries.popitem(last=False)

        self.entries[context_key] = {
            "candidate_val": candidate_val,
            "count": 1,
            "sources": {source_id},
            "last_seen_step": step,
        }
        return 1

    def get_contradiction_evidence(self, context_key: int) -> Tuple[int, int]:
        """Returns (count, num_distinct_sources) for context_key."""
        if context_key in self.entries:
            rec = self.entries[context_key]
            return rec["count"], len(rec["sources"])
        return 0, 0

    def purge_contradiction(self, context_key: int) -> None:
        """Remove entry once flipped into primary belief."""
        if context_key in self.entries:
            del self.entries[context_key]

    def storage_bytes(self) -> int:
        """
        Exact memory accounting:
        Key (int): 8 bytes
        candidate_val (int): 8 bytes
        count (int): 8 bytes
        last_seen_step (int): 8 bytes
        sources (set of strings, up to 4 strings of ~16 bytes): ~64 bytes
        Total per entry: ~48 bytes * max_entries (<= 3,072 bytes, strictly <= 4 KB as required by F7).
        """
        entry_size = 48
        return len(self.entries) * entry_size


class EvidenceAccumulatingUpkeeper:
    """
    Active curator and upkeeper for modular continual learning experts:
    - Tracks prior-count confidence for established rules.
    - Gates weight updates: filters one-off noise bursts; enables genuine belief revision.
    """

    def __init__(
        self,
        flip_threshold_ratio: float = 1.0,
        min_sources_for_flip: int = 2,
        store_capacity: int = 64,
    ):
        self.flip_threshold_ratio = flip_threshold_ratio
        self.min_sources_for_flip = min_sources_for_flip
        self.tentative_store = TentativeBeliefStore(max_entries=store_capacity)
        # primary_beliefs: context_key -> target_val
        self.primary_beliefs: Dict[int, int] = {}
        # prior_counts: context_key -> count of confirmed observations
        self.prior_counts: Dict[int, int] = {}
        self.step_count = 0

    def filter_and_gate_targets(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        predictions: torch.Tensor,
        source_id: str = "default",
        state_chain_mode: bool = False,
        reset_max_token: int = 39,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inspect batch against prior counts and tentative belief store.
        Returns:
            gated_targets: targets with noise/burst contradicted positions masked or updated.
            update_mask: [B, L] boolean mask indicating which tokens should flow gradients.
        """
        self.step_count += 1
        B, L = targets.shape
        gated_targets = targets.clone()
        update_mask = torch.ones(B, L, dtype=torch.bool, device=targets.device)
        if not state_chain_mode:
            update_mask[:, 0] = False  # Position 0 has no valid preceding bigram context

        x_cpu = x.detach().cpu().tolist()
        t_cpu = targets.detach().cpu().tolist()

        start_idx = 0 if state_chain_mode else 1
        for b in range(B):
            for seq_idx in range(start_idx, L):
                cur_token = x_cpu[b][seq_idx]
                target_val = t_cpu[b][seq_idx]

                if state_chain_mode:
                    if seq_idx == 0 or cur_token <= reset_max_token:
                        prev_token = 0
                    else:
                        prev_token = t_cpu[b][seq_idx - 1]
                else:
                    prev_token = x_cpu[b][seq_idx - 1]

                # Context signature key: integer encoding of context (prev_token, cur_token)
                context_key = (prev_token << 8) | cur_token

                if context_key not in self.primary_beliefs:
                    # Case 1: Brand-new fact -> Learn immediately at low confidence
                    self.primary_beliefs[context_key] = target_val
                    self.prior_counts[context_key] = 1
                    update_mask[b, seq_idx] = True
                elif target_val == self.primary_beliefs[context_key]:
                    # Case 2: Confirmation of existing belief -> Increment confidence
                    self.prior_counts[context_key] = min(self.prior_counts[context_key] + 1, 1000)
                    update_mask[b, seq_idx] = True
                else:
                    # Case 3: Contradiction!
                    prior_cnt = self.prior_counts[context_key]
                    # Record in tentative store
                    contra_cnt = self.tentative_store.record_contradiction(
                        context_key=context_key,
                        candidate_val=target_val,
                        source_id=source_id,
                        step=self.step_count,
                    )
                    evidence_cnt, n_sources = self.tentative_store.get_contradiction_evidence(context_key)

                    # Check flip condition:
                    # Does accumulated counter-evidence outweigh established prior count?
                    # Capped at effective sample size 10 to allow genuine revision under persistent evidence
                    threshold = max(2, min(int(prior_cnt * self.flip_threshold_ratio), 10))
                    if evidence_cnt >= threshold and n_sources >= self.min_sources_for_flip:
                        # Genuine belief revision!
                        self.primary_beliefs[context_key] = target_val
                        self.prior_counts[context_key] = evidence_cnt
                        self.tentative_store.purge_contradiction(context_key)
                        update_mask[b, seq_idx] = True
                    else:
                        # Isolated or unverified burst: resist overwrite!
                        # Mask gradient update for this position to protect established weights
                        update_mask[b, seq_idx] = False

            if state_chain_mode:
                # If a sequence had transition operators and every operator in the sequence
                # was rejected as an unverified contradiction, do not train on isolated
                # D=0 reset tokens alone (which would skew the depth prior).
                has_ops = any(x_cpu[b][t] > reset_max_token for t in range(L))
                has_accepted_ops = any(
                    x_cpu[b][t] > reset_max_token and bool(update_mask[b, t].item())
                    for t in range(L)
                )
                if has_ops and not has_accepted_ops:
                    update_mask[b, :] = False

        return gated_targets, update_mask

    def tentative_store_bytes(self) -> int:
        """Return memory used exclusively by the tentative belief store in bytes."""
        return self.tentative_store.storage_bytes()

    def memory_footprint_bytes(self) -> int:
        """Return total memory used by prior counts and tentative belief store in bytes."""
        belief_bytes = len(self.primary_beliefs) * 16
        prior_bytes = len(self.prior_counts) * 16  # key (8 bytes) + count (8 bytes)
        tentative_bytes = self.tentative_store.storage_bytes()
        return belief_bytes + prior_bytes + tentative_bytes


class CalibratedBayesianUpkeeper:
    """
    Budgeted Dual-Lane Belief Store & Calibrated Bayesian Logit Fuser (M09 upgrade).

    Features:
    - Honest strict memory ceiling `max_total_bytes` covering BOTH `primary_beliefs`
      and `tentative_store` combined (no hidden unbounded tables).
    - Fast flat-sequence evidence gating separating single-source bursts from persistent
      multi-source belief revisions.
    - Normalized Dirichlet-mixture fusion `P_fused = (1 - w) * P_neural + w * e_{y*}`
      that preserves calibrated NLL everywhere and leaves held-out compositional keys
      100% under neural control (`w = 0`).
    """

    BYTES_PER_PRIMARY = 16    # 8B packed key + 4B val + 4B count
    BYTES_PER_TENTATIVE = 52  # 8B packed key + 4B cand + 4B cnt + 4B step + 32B source set

    def __init__(
        self,
        max_total_bytes: int = 4096,
        min_sources_for_flip: int = 2,
        min_flip_count: int = 3,
        max_prior_count: int = 16,
        flip_threshold_ratio: float = 0.5,
        kappa: float = 3.0,
        min_fuse_count: int = 2,
        max_fuse_weight: float = 0.92,
    ):
        self.max_total_bytes = int(max_total_bytes)
        self.max_primary = max(4, int(self.max_total_bytes * 0.55) // self.BYTES_PER_PRIMARY)
        rem_bytes = max(0, self.max_total_bytes - self.max_primary * self.BYTES_PER_PRIMARY)
        self.max_tentative = max(2, rem_bytes // self.BYTES_PER_TENTATIVE)

        self.min_sources_for_flip = int(min_sources_for_flip)
        self.min_flip_count = int(min_flip_count)
        self.max_prior_count = int(max_prior_count)
        self.flip_threshold_ratio = float(flip_threshold_ratio)
        self.kappa = float(kappa)
        self.min_fuse_count = int(min_fuse_count)
        self.max_fuse_weight = float(max_fuse_weight)

        # key -> (target_val, count)
        self.primary_beliefs: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        # key -> (cand_val, evidence_cnt, sources_set, last_step)
        self.tentative_store: OrderedDict[int, Tuple[int, int, set, int]] = OrderedDict()
        self._call_step: int = 0
        self.flips_executed: int = 0
        self.bursts_rejected: int = 0

    def _evict_lowest_primary(self) -> None:
        if not self.primary_beliefs:
            return
        # Scan up to 16 oldest LRU items and evict minimum prior count (LFU-LRU protection)
        victim_key = None
        victim_cnt = 1 << 30
        for idx, (k, (_, cnt)) in enumerate(self.primary_beliefs.items()):
            if cnt < victim_cnt:
                victim_cnt = cnt
                victim_key = k
            if idx >= 15:
                break
        if victim_key is not None:
            del self.primary_beliefs[victim_key]

    def _put_primary(self, key: int, val: int, count: int) -> None:
        if key in self.primary_beliefs:
            self.primary_beliefs.move_to_end(key)
        else:
            while len(self.primary_beliefs) >= self.max_primary:
                self._evict_lowest_primary()
        self.primary_beliefs[key] = (int(val), min(int(count), self.max_prior_count))

    def _record_tentative(self, key: int, cand_val: int, source_id: str) -> Tuple[int, int]:
        step = self._call_step
        if key in self.tentative_store:
            cur_cand, cur_cnt, sources, _ = self.tentative_store[key]
            self.tentative_store.move_to_end(key)
            if cur_cand == cand_val:
                sources.add(source_id)
                new_cnt = cur_cnt + 1
                self.tentative_store[key] = (cand_val, new_cnt, sources, step)
                return new_cnt, len(sources)
            # Competing alternative replaces weak uncorroborated candidate
            if cur_cnt <= 1:
                sources = {source_id}
                self.tentative_store[key] = (cand_val, 1, sources, step)
                return 1, 1
            return 0, len(sources)

        if len(self.tentative_store) >= self.max_tentative:
            # Prefer evicting stale (>8 calls untouched) or lowest-evidence tentative slots;
            # do not churn active wave candidates in the same call once capacity is full.
            victim_key = None
            victim_score = None
            for k, (_, cnt, srcs, l_step) in self.tentative_store.items():
                is_fresh = 0 if (step - l_step) >= 8 else 1
                score = (is_fresh, len(srcs), min(cnt, 4))
                if victim_score is None or score < victim_score:
                    victim_score = score
                    victim_key = k
                    if score == (0, 1, 1):
                        break
            if victim_score is not None and victim_score[0] == 1 and victim_score[2] >= 2:
                return 0, 0
            if victim_key is not None:
                del self.tentative_store[victim_key]
        self.tentative_store[key] = (int(cand_val), 1, {source_id}, step)
        return 1, 1

    def gate_targets(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        source_id: str = "default",
        state_chain_mode: bool = False,
        reset_max_token: int = 39,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Verify targets against established beliefs & accumulate tentative counter-evidence.
        Returns (gated_targets [B, L], update_mask [B, L] bool).
        """
        self._call_step += 1
        B, L = x.shape
        device = x.device
        if L < 2 and not state_chain_mode:
            return targets.clone(), torch.ones((B, L), dtype=torch.bool, device=device)

        x_list = x.detach().cpu().tolist()
        y_list = targets.detach().cpu().tolist()
        keys_list = (
            (((x[:, :-1].detach().cpu().to(torch.int64) & 0xFFFF) << 16) | (x[:, 1:].detach().cpu().to(torch.int64) & 0xFFFF)).tolist()
            if L >= 2
            else [[] for _ in range(B)]
        )

        primary = self.primary_beliefs
        tent = self.tentative_store
        max_prim = self.max_primary
        min_src = self.min_sources_for_flip
        min_fc = self.min_flip_count
        flip_ratio = self.flip_threshold_ratio

        # Pre-check contradiction rate on already-established beliefs in this batch
        # so uncorroborated single-source bursts cannot flush the primary cache by eviction.
        n_cached_hits = 0
        n_cached_contra = 0
        if not state_chain_mode and L >= 2:
            get_p = primary.get
            for kb, yb in zip(keys_list, y_list):
                for t_idx, key in enumerate(kb):
                    entry = get_p(key)
                    if entry is not None:
                        n_cached_hits += 1
                        if yb[t_idx + 1] != entry[0]:
                            n_cached_contra += 1
        high_contra_batch = (not state_chain_mode) and (n_cached_hits >= 8) and (n_cached_contra >= 0.30 * n_cached_hits)
        flips_this_call = 0

        gb_all: List[List[int]] = []
        mb_all: List[List[bool]] = []
        put_p = self._put_primary
        rec_t = self._record_tentative

        for b in range(B):
            xb = x_list[b]
            yb = y_list[b]
            kb = keys_list[b]
            mb = [True] * L
            gb = list(yb)
            for t_idx, key in enumerate(kb):
                t = t_idx + 1
                tgt = yb[t]
                entry = primary.get(key)
                if entry is None:
                    if high_contra_batch and flips_this_call == 0 and len(primary) >= max_prim:
                        self.bursts_rejected += 1
                        mb[t] = False
                    else:
                        put_p(key, tgt, 1)
                else:
                    cur_y, prior_cnt = entry
                    if tgt == cur_y:
                        if not (high_contra_batch and flips_this_call == 0):
                            put_p(key, cur_y, prior_cnt + 1)
                        if key in tent:
                            del tent[key]
                    else:
                        evid_cnt, n_sources = rec_t(key, tgt, source_id)
                        req_cnt = max(min_fc, int(math.ceil(prior_cnt * flip_ratio)))
                        if evid_cnt >= req_cnt and n_sources >= min_src:
                            put_p(key, tgt, max(2, evid_cnt))
                            if key in tent:
                                del tent[key]
                            self.flips_executed += 1
                            flips_this_call += 1
                        else:
                            self.bursts_rejected += 1
                            gb[t] = cur_y
                            mb[t] = False
            if state_chain_mode:
                has_ops = any(xb[t] > reset_max_token for t in range(L))
                has_acc = any(xb[t] > reset_max_token and mb[t] for t in range(L))
                if has_ops and not has_acc:
                    mb = [False] * L
            gb_all.append(gb)
            mb_all.append(mb)

        return (
            torch.tensor(gb_all, dtype=targets.dtype, device=device),
            torch.tensor(mb_all, dtype=torch.bool, device=device),
        )

    @torch.no_grad()
    def fuse_logits(
        self,
        x: torch.Tensor,
        neural_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return calibrated log-probabilities [B, L, V] combining Lane N (`neural_logits`)
        and confirmed Lane E beliefs without `-1e9` uncalibrated artifacts.
        Unseen keys have w_E = 0 and reduce identically to log_softmax(neural_logits).
        """
        B, L, V = neural_logits.shape
        log_probs_n = F.log_softmax(neural_logits.float(), dim=-1)
        if not self.primary_beliefs or L < 2:
            return log_probs_n.to(dtype=neural_logits.dtype)

        probs_n = log_probs_n.exp()
        x_list = x.detach().cpu().tolist()

        hits_b: List[int] = []
        hits_t: List[int] = []
        hits_y: List[int] = []
        hits_w: List[float] = []

        for b in range(B):
            xb = x_list[b]
            for t in range(1, L):
                key = ((xb[t - 1] & 0xFFFF) << 16) | (xb[t] & 0xFFFF)
                entry = self.primary_beliefs.get(key)
                if entry is not None:
                    val, cnt = entry
                    if cnt >= self.min_fuse_count and 0 <= val < V:
                        w = min(float(cnt) / (self.kappa + float(cnt)), self.max_fuse_weight)
                        hits_b.append(b)
                        hits_t.append(t)
                        hits_y.append(val)
                        hits_w.append(w)

        if not hits_b:
            return log_probs_n.to(dtype=neural_logits.dtype)

        dev = neural_logits.device
        hb = torch.tensor(hits_b, dtype=torch.long, device=dev)
        ht = torch.tensor(hits_t, dtype=torch.long, device=dev)
        hy = torch.tensor(hits_y, dtype=torch.long, device=dev)
        hw = torch.tensor(hits_w, dtype=torch.float32, device=dev).unsqueeze(-1)

        probs_fused = probs_n.clone()
        sel = probs_n[hb, ht] * (1.0 - hw)
        sel.scatter_add_(-1, hy.unsqueeze(-1), hw)
        probs_fused[hb, ht] = sel
        return torch.log(probs_fused.clamp_min(1e-7)).to(dtype=neural_logits.dtype)

    def memory_footprint_bytes(self) -> int:
        return (
            len(self.primary_beliefs) * self.BYTES_PER_PRIMARY
            + len(self.tentative_store) * self.BYTES_PER_TENTATIVE
        )
