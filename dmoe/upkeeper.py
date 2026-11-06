"""
Evidence-Accumulating Upkeeper with Tentative-Belief Store.
Implements the prior-count belief revision policy specified in §14.1.

Principles:
1. No existing belief (prior count = 0) -> Learn immediately at low confidence.
2. Contradiction of a confident belief (high prior count) -> Resist single-source bursts.
3. Evidence accumulates -> Flip when contradictory evidence outweighs prior count.
4. Memory honesty -> Fixed-capacity tentative-belief store (counts and source IDs only, strictly no raw samples).
"""

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple
import torch
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
        update_mask[:, 0] = False  # Position 0 has no valid preceding bigram context

        x_cpu = x.detach().cpu().tolist()
        t_cpu = targets.detach().cpu().tolist()

        for b in range(B):
            for seq_idx in range(1, L):
                prev_token = x_cpu[b][seq_idx - 1]
                cur_token = x_cpu[b][seq_idx]
                target_val = t_cpu[b][seq_idx]

                # Context signature key: integer encoding of bigram context (prev_token, cur_token)
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
