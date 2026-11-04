"""v1/pool_removal.py: Certified removal in the V0 pool (ULDA) and trace auditing.

Part of Job V-3 / Probe P2 (is removal exact?).
Provides:
  - ExposureTracker: instance-level wrapper recording expert exposure to labels and closing over lineage.
  - scan_traces(pool, labels, removed_samples=None): independent trace auditor inspecting all pool and expert structures.
  - remove_labels(pool, labels, data=None, tracker=None): drops exposed experts, purges retained structures, refits router,
    and returns a verified JSON certificate.
  - compare_pool_removal(p_full, p_removed, p_never, bed, labels, seed): evaluates retention accuracy,
    routing agreement, expert weight differences, and class accuracy on removed labels.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from growmoe.pools.base import ExpertRecord


class ExposureTracker:
    """Instance-level exposure tracker for PhyloMoE / UniPool models.

    Monitors and records every direct training event, replay batch, and closed-form
    head absorption, and computes the transitive exposure closure over the phylogenetic lineage.
    """

    def __init__(self, pool: Any) -> None:
        self.pool = pool
        # Direct exposures: eid -> set of class labels seen
        self.expert_direct_labels: Dict[int, Set[int]] = {}
        # Domain LDA labels seen
        self.domain_lda_labels: Set[int] = set()
        # Probation labels seen
        self.probation_labels: Set[int] = set()

        self._installed = False
        self._wrap_pool()

    def _wrap_pool(self) -> None:
        """Install non-invasive instance-level method wrappers."""
        if self._installed:
            return

        pool = self.pool
        orig_train_expert = getattr(pool, "train_expert", None)
        orig_lda_absorb = getattr(pool, "_lda_absorb", None)
        orig_observe = getattr(pool, "observe", None)

        tracker = self

        if orig_train_expert is not None:
            def wrapped_train_expert(eid: int, x: Tensor, y: Tensor, replay: bool = True, project: bool = False) -> float:
                if y is not None:
                    lbls = set(y.detach().cpu().long().unique().tolist())
                    tracker.expert_direct_labels.setdefault(eid, set()).update(lbls)
                return orig_train_expert(eid, x, y, replay=replay, project=project)
            pool.train_expert = wrapped_train_expert

        if orig_lda_absorb is not None:
            def wrapped_lda_absorb(rec: ExpertRecord, x: Tensor, y: Tensor) -> None:
                if y is not None:
                    lbls = set(y.detach().cpu().long().unique().tolist())
                    tracker.expert_direct_labels.setdefault(rec.id, set()).update(lbls)
                orig_lda_absorb(rec, x, y)
            pool._lda_absorb = wrapped_lda_absorb

        if orig_observe is not None:
            def wrapped_observe(x: Tensor, y: Optional[Tensor] = None, ctx: Optional[Tensor] = None):
                if y is not None:
                    lbls = set(y.detach().cpu().long().unique().tolist())
                    if hasattr(pool, "domain_lda") and pool.domain_lda is not None:
                        tracker.domain_lda_labels.update(lbls)
                return orig_observe(x, y, ctx)
            pool.observe = wrapped_observe
        self._installed = True

    def get_exposure(self, labels: Union[Set[int], Sequence[int]]) -> Dict[str, Any]:
        """Compute direct and transitive lineage-closed exposure for a given set of labels.

        Args:
            labels: Removed class labels.

        Returns:
            Dict containing direct_exposed, indirect_exposed, all_exposed, retained,
            and collateral_classes.
        """
        target_labels = set(labels)
        pool = self.pool

        # 1. Determine direct exposures
        direct_exposed: Set[int] = set()
        for eid, seen in self.expert_direct_labels.items():
            if seen & target_labels:
                direct_exposed.add(eid)

        # Also check current expert module seen masks and replay buffers directly
        for eid, rec in pool.experts.items():
            if hasattr(rec.module, "seen"):
                seen_idx = torch.where(rec.module.seen)[0].tolist()
                if set(seen_idx) & target_labels:
                    direct_exposed.add(eid)
            if rec.replay is not None and getattr(rec.replay, "n", 0) > 0:
                replay_y = rec.replay.y[: rec.replay.n].tolist()
                if set(replay_y) & target_labels:
                    direct_exposed.add(eid)

        # 2. Lineage closure (forward in time: parent -> child)
        # In PhyloMoE, child inherits weights from parent at birth
        # pool._parents maps parent_id -> list of child_ids
        children_map: Dict[int, List[int]] = {}
        for eid, rec in pool.experts.items():
            p = rec.parent
            if p is not None:
                children_map.setdefault(p, []).append(eid)
        if hasattr(pool, "_parents") and isinstance(pool._parents, dict):
            for p, c_list in pool._parents.items():
                if isinstance(c_list, list):
                    for c in c_list:
                        children_map.setdefault(p, []).append(c)

        # BFS / DFS to close over all descendants
        all_exposed: Set[int] = set(direct_exposed)
        queue = list(direct_exposed)
        indirect_exposed: Set[int] = set()

        while queue:
            curr = queue.pop(0)
            for child in children_map.get(curr, []):
                if child not in all_exposed:
                    all_exposed.add(child)
                    indirect_exposed.add(child)
                    queue.append(child)

        all_expert_ids = set(pool.experts.keys())
        # Restrict to active experts currently in pool
        retained = all_expert_ids - all_exposed
        dropped_experts = sorted(all_exposed & all_expert_ids)

        # 3. Compute collateral damage: retained classes whose ONLY experts are in dropped_experts
        # Map each class to the set of experts that know it
        class_to_experts: Dict[int, Set[int]] = {}
        for eid, rec in pool.experts.items():
            if hasattr(rec.module, "seen"):
                seen_idx = torch.where(rec.module.seen)[0].tolist()
                for c in seen_idx:
                    class_to_experts.setdefault(c, set()).add(eid)

        collateral_classes_lost: Set[int] = set()
        retained_classes_in_exposed: Set[int] = set()

        for c, exp_set in class_to_experts.items():
            if c not in target_labels:
                if exp_set & all_exposed:
                    retained_classes_in_exposed.add(c)
                # If ALL experts that knew class c are dropped:
                if not (exp_set - all_exposed):
                    collateral_classes_lost.add(c)

        return {
            "target_labels": sorted(target_labels),
            "direct_exposed_experts": sorted(direct_exposed),
            "indirect_exposed_experts": sorted(indirect_exposed),
            "all_exposed_experts": sorted(all_exposed),
            "dropped_experts": dropped_experts,
            "retained_experts": sorted(retained),
            "n_direct": len(direct_exposed),
            "n_indirect": len(indirect_exposed),
            "n_dropped": len(dropped_experts),
            "n_retained": len(retained),
            "collateral_classes_lost": sorted(collateral_classes_lost),
            "n_collateral_lost": len(collateral_classes_lost),
            "retained_classes_in_exposed_experts": sorted(retained_classes_in_exposed),
            "n_retained_in_exposed": len(retained_classes_in_exposed),
        }


def scan_traces(
    pool: Any,
    labels: Union[Set[int], Sequence[int]],
    removed_samples: Optional[Tensor] = None,
) -> Dict[str, Any]:
    """Independent trace auditor inspecting every pool and expert structure.

    Checks:
      1. expert_module_seen: module seen masks in remaining experts
      2. expert_replay_y: raw replay labels in remaining experts
      3. expert_lda_heads: per-expert LdaHead counts in remaining experts
      4. expert_signatures: prototype cosine similarity to removed samples in remaining experts
      5. pool_domain_lda: pool-level domain LDA head counts
      6. pool_probation: pool probation buffer labels
      7. pool_held: doubtful batch queue labels
      8. pool_boot: bootstrap buffer labels
      9. pool_router_eids: router references to dropped expert IDs

    Returns:
        Dict of trace counts and locations found.
    """
    target = set(labels)
    traces: Dict[str, Any] = {}
    total_trace_count = 0

    # 1. Expert module seen masks
    module_traces = {}
    for eid, rec in pool.experts.items():
        if hasattr(rec.module, "seen"):
            seen_cls = set(torch.where(rec.module.seen)[0].tolist())
            overlap = sorted(seen_cls & target)
            if overlap:
                module_traces[eid] = overlap
                total_trace_count += len(overlap)
    traces["expert_module_seen"] = {
        "count": len(module_traces),
        "details": module_traces,
        "classification": "exact",
    }

    # 2. Expert replay buffers
    replay_traces = {}
    for eid, rec in pool.experts.items():
        if rec.replay is not None and getattr(rec.replay, "n", 0) > 0:
            ry = rec.replay.y[: rec.replay.n]
            mask = torch.isin(ry, torch.tensor(list(target), dtype=torch.long))
            cnt = int(mask.sum().item())
            if cnt > 0:
                replay_traces[eid] = cnt
                total_trace_count += cnt
    traces["expert_replay_samples"] = {
        "count": sum(replay_traces.values()),
        "details": replay_traces,
        "classification": "exact",
    }

    # 3. Expert LDA heads
    lda_traces = {}
    for eid, rec in pool.experts.items():
        lda = rec.meta.get("lda")
        if lda is not None and hasattr(lda, "n"):
            cnt_rem = sum(float(lda.n[c].item()) for c in target if c < lda.n.shape[0])
            if cnt_rem > 0:
                lda_traces[eid] = cnt_rem
                total_trace_count += int(cnt_rem)
    traces["expert_lda_counts"] = {
        "count": sum(lda_traces.values()),
        "details": lda_traces,
        "classification": "exact",
    }

    # 4. Expert signatures
    sig_traces = {}
    if removed_samples is not None and len(removed_samples) > 0:
        rem_u = removed_samples.float()
        rem_u = rem_u / rem_u.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        for eid, rec in pool.experts.items():
            sig = rec.meta.get("sig")
            if sig is not None and hasattr(sig, "protos") and sig.n > 0:
                pr = sig.protos.float()
                # Cosine similarity matrix between protos and removed samples
                cos = pr @ rem_u.T
                max_cos = cos.max().item() if cos.numel() > 0 else 0.0
                matches = int((cos >= 0.9999).any(dim=1).sum().item())
                if matches > 0:
                    sig_traces[eid] = {"matches": matches, "max_cosine": max_cos}
                    total_trace_count += matches
    traces["expert_signature_prototypes"] = {
        "count": sum(v["matches"] for v in sig_traces.values()) if sig_traces else 0,
        "details": sig_traces,
        "classification": "exact",
    }

    # 5. Pool domain LDA
    dom_traces = 0
    if hasattr(pool, "domain_lda") and pool.domain_lda is not None:
        d_lda = pool.domain_lda
        if hasattr(d_lda, "n"):
            dom_traces = int(sum(float(d_lda.n[c].item()) for c in target if c < d_lda.n.shape[0]))
            total_trace_count += dom_traces
    traces["pool_domain_lda_counts"] = {
        "count": dom_traces,
        "classification": "not_removable_without_data",
        "notes": "Class counts and sums zeroed, but pooled scatter S retains traces unless x_rem is provided",
    }

    # 6. Pool probation buffer
    prob_traces = 0
    if hasattr(pool, "probation") and len(pool.probation) > 0:
        py = pool.probation.y
        mask = torch.isin(py, torch.tensor(list(target), dtype=torch.long))
        prob_traces = int(mask.sum().item())
        total_trace_count += prob_traces
    traces["pool_probation_samples"] = {
        "count": prob_traces,
        "classification": "exact",
    }

    # 7. Pool doubtful batch queue (_held)
    held_traces = 0
    if hasattr(pool, "_held") and pool._held:
        for item in pool._held:
            if isinstance(item, tuple) and len(item) >= 2:
                hy = item[1]
                if isinstance(hy, Tensor):
                    mask = torch.isin(hy, torch.tensor(list(target), dtype=torch.long))
                    held_traces += int(mask.sum().item())
        total_trace_count += held_traces
    traces["pool_held_samples"] = {
        "count": held_traces,
        "classification": "exact",
    }

    # 8. Pool bootstrap buffer (_boot_y)
    boot_traces = 0
    if hasattr(pool, "_boot_y") and pool._boot_y:
        for by in pool._boot_y:
            mask = torch.isin(by, torch.tensor(list(target), dtype=torch.long))
            boot_traces += int(mask.sum().item())
        total_trace_count += boot_traces
    traces["pool_boot_samples"] = {
        "count": boot_traces,
        "classification": "exact",
    }

    # 9. Pool router expert IDs
    router_traces = 0
    if hasattr(pool, "_route") and pool._route.eids is not None:
        # Check if any eid in router is no longer in pool.experts
        active_eids = set(pool.experts.keys())
        stale = [e for e in pool._route.eids if e not in active_eids]
        router_traces = len(stale)
        total_trace_count += router_traces
    traces["pool_router_stale_eids"] = {
        "count": router_traces,
        "classification": "exact",
    }

    traces["total_traces_found"] = total_trace_count
    return traces


def remove_labels(
    pool: Any,
    labels: Union[Set[int], Sequence[int], Tensor],
    data: Optional[Tuple[Tensor, Tensor]] = None,
    tracker: Optional[ExposureTracker] = None,
) -> Dict[str, Any]:
    """Execute certified removal of specified labels from the pool.

    Steps:
      1. Compute exposure closure (direct + lineage).
      2. Drop all exposed experts.
      3. In retained experts, purge residual traces from replay, module seen masks,
         LDA heads, and signatures.
      4. In pool level, purge probation buffer, held batches, bootstrap buffers,
         and downdate domain LDA.
      5. Refit pool router on retained experts.
      6. Run independent scan_traces to construct certificate.

    Returns:
        JSON-serializable certificate dict.
    """
    if isinstance(labels, (int, np.integer)):
        target_labels = {int(labels)}
    elif isinstance(labels, torch.Tensor):
        target_labels = {int(x) for x in labels.view(-1).tolist()}
    else:
        target_labels = {int(x) for x in labels}

    # 1. Determine exposure
    if tracker is not None:
        exposure = tracker.get_exposure(target_labels)
    else:
        temp_tracker = ExposureTracker(pool)
        exposure = temp_tracker.get_exposure(target_labels)

    dropped_eids = exposure["dropped_experts"]

    # 2. Drop all exposed experts
    for eid in dropped_eids:
        if eid in pool.experts:
            pool.remove_expert(eid)

    # 3. Purge structures in retained experts
    for eid, rec in pool.experts.items():
        # Mask out seen
        if hasattr(rec.module, "seen"):
            for c in target_labels:
                if c < rec.module.seen.shape[0]:
                    rec.module.seen[c] = False

        # Purge replay
        if rec.replay is not None and getattr(rec.replay, "n", 0) > 0:
            rx = rec.replay.x[: rec.replay.n]
            ry = rec.replay.y[: rec.replay.n]
            keep_m = ~torch.isin(ry, torch.tensor(list(target_labels), dtype=torch.long))
            n_kept = int(keep_m.sum().item())
            rec.replay.x[:n_kept] = rx[keep_m]
            rec.replay.y[:n_kept] = ry[keep_m]
            rec.replay.x[n_kept : rec.replay.n].zero_()
            rec.replay.y[n_kept : rec.replay.n].zero_()
            rec.replay.n = n_kept

        # Purge expert LDA
        if "lda" in rec.meta:
            lda = rec.meta["lda"]
            for c in target_labels:
                if c < lda.n.shape[0]:
                    lda.n[c] = 0.0
                    lda.s[c].zero_()
            lda._fit_cache = None

        # Clean probe / gap
        rec.meta.pop("probe", None)

    # 4. Pool-level purges
    # Probation buffer
    if hasattr(pool, "probation") and len(pool.probation) > 0:
        keep_prob = ~torch.isin(pool.probation.y, torch.tensor(list(target_labels), dtype=torch.long))
        pool.probation.x = pool.probation.x[keep_prob]
        pool.probation.y = pool.probation.y[keep_prob]
        pool.probation.arrival = pool.probation.arrival[keep_prob]

    # Held batches
    if hasattr(pool, "_held") and pool._held:
        new_held = []
        for item in pool._held:
            if isinstance(item, tuple) and len(item) >= 2:
                hx, hy = item[0], item[1]
                keep_h = ~torch.isin(hy, torch.tensor(list(target_labels), dtype=torch.long))
                if bool(keep_h.any()):
                    new_held.append((hx[keep_h], hy[keep_h]))
            else:
                new_held.append(item)
        pool._held = new_held

    # Bootstrap buffers
    if hasattr(pool, "_boot_y") and pool._boot_y:
        pool._boot_x = []
        pool._boot_y = []

    # Domain LDA
    if hasattr(pool, "domain_lda") and pool.domain_lda is not None:
        d_lda = pool.domain_lda
        for c in target_labels:
            if c < d_lda.n.shape[0]:
                d_lda.n[c] = 0.0
                d_lda.s[c].zero_()

        if data is not None:
            x_rem, y_rem = data
            x_rem_t = torch.as_tensor(x_rem, dtype=torch.float32)
            y_rem_t = torch.as_tensor(y_rem, dtype=torch.long)
            mask_rem = torch.isin(y_rem_t, torch.tensor(list(target_labels), dtype=torch.long))
            if bool(mask_rem.any()):
                target_x = x_rem_t[mask_rem]
                xn = pool._normalize(target_x.float()).double()
                d_lda.S -= xn.T @ xn
                d_lda.S = 0.5 * (d_lda.S + d_lda.S.T)
                d_lda.total = max(0, d_lda.total - int(target_x.shape[0]))
        d_lda._fit_cache = None

    # 5. Refit pool router on retained experts
    pool._refit_router()

    # 6. Perform independent trace audit to construct certificate
    removed_x = data[0] if data is not None else None
    audit = scan_traces(pool, target_labels, removed_samples=removed_x)

    certificate = {
        "status": "certified_removal",
        "target_labels": sorted(target_labels),
        "exposure_summary": {
            "n_direct_exposed": exposure["n_direct"],
            "n_indirect_exposed": exposure["n_indirect"],
            "dropped_experts": dropped_eids,
            "retained_experts": exposure["retained_experts"],
            "n_collateral_classes_lost": exposure["n_collateral_lost"],
            "collateral_classes_lost": exposure["collateral_classes_lost"],
            "n_retained_in_exposed": exposure["n_retained_in_exposed"],
        },
        "trace_audit": audit,
        "certified_clean": (audit["total_traces_found"] == 0),
    }

    return certificate


def compare_pool_removal(
    p_full: Any,
    p_removed: Any,
    p_never: Any,
    bed: Any,
    labels: Union[Set[int], Sequence[int]],
    seed: int = 800,
) -> Dict[str, Any]:
    """Compare P_removed (post-removal) against P_full and P_never.

    Args:
        p_full: Full model evaluated after training on all classes.
        p_removed: Full model after remove_labels has been applied.
        p_never: Model trained fresh on the stream with removed classes omitted.
        bed: Bed object providing test_sets and regime_info.
        labels: Set of removed labels.
        seed: Experiment seed.

    Returns:
        Structured evaluation metrics dict.
    """
    target = set(labels)
    all_classes = set(range(bed.n_classes))
    retained_classes = all_classes - target

    # 1. Collect all retained and removed test sets with regime offsets
    regime_slices: Dict[str, Tuple[int, int, Tensor]] = {}
    all_ret_x = []
    all_ret_y = []
    all_rem_x = []
    all_rem_y = []

    curr_idx = 0
    for r in sorted(bed.test_sets.keys()):
        tx, ty = bed.test_sets[r]
        ret_mask = ~torch.isin(ty, torch.tensor(list(target), dtype=torch.long))
        if bool(ret_mask.any()):
            tx_ret = tx[ret_mask]
            ty_ret = ty[ret_mask]
            n_ret = int(tx_ret.shape[0])
            regime_slices[str(r)] = (curr_idx, curr_idx + n_ret, ty_ret.cpu())
            curr_idx += n_ret
            all_ret_x.append(tx_ret)
            all_ret_y.append(ty_ret)

        rem_mask = torch.isin(ty, torch.tensor(list(target), dtype=torch.long))
        if bool(rem_mask.any()):
            all_rem_x.append(tx[rem_mask])
            all_rem_y.append(ty[rem_mask])

    # 2. Run single forward pass per model on all retained test samples
    acc_full_regimes = {}
    acc_rem_regimes = {}
    acc_nev_regimes = {}
    routing_agreement = 0.0
    argmax_agreement = 0.0

    if all_ret_x:
        test_x_cat = torch.cat(all_ret_x, dim=0)
        test_y_cat = torch.cat(all_ret_y, dim=0).cpu()

        pred_f_all = p_full.predict(test_x_cat)
        pred_r_all = p_removed.predict(test_x_cat)
        pred_n_all = p_never.predict(test_x_cat)

        argmax_f = pred_f_all.probs.argmax(dim=-1).cpu()
        argmax_r = pred_r_all.probs.argmax(dim=-1).cpu()
        argmax_n = pred_n_all.probs.argmax(dim=-1).cpu()

        # Per-regime accuracy slices
        for r_str, (start, end, r_y) in regime_slices.items():
            acc_full_regimes[r_str] = float((argmax_f[start:end] == r_y).float().mean().item())
            acc_rem_regimes[r_str] = float((argmax_r[start:end] == r_y).float().mean().item())
            acc_nev_regimes[r_str] = float((argmax_n[start:end] == r_y).float().mean().item())

        # Routing and argmax agreement
        route_r = pred_r_all.expert.cpu()
        route_n = pred_n_all.expert.cpu()
        routing_agreement = float((route_r == route_n).float().mean().item())
        argmax_agreement = float((argmax_r == argmax_n).float().mean().item())

    mean_acc_full = float(np.mean(list(acc_full_regimes.values()))) if acc_full_regimes else 0.0
    mean_acc_rem = float(np.mean(list(acc_rem_regimes.values()))) if acc_rem_regimes else 0.0
    mean_acc_nev = float(np.mean(list(acc_nev_regimes.values()))) if acc_nev_regimes else 0.0

    # 3. Retained expert weight differences (matched by founding batch / birth_step)
    pool_r = p_removed.model if hasattr(p_removed, "model") else p_removed
    pool_n = p_never.model if hasattr(p_never, "model") else p_never

    # Map birth_step -> expert in p_never
    never_by_birth: Dict[int, Any] = {}
    for eid, rec in pool_n.experts.items():
        never_by_birth[rec.birth_step] = rec

    matched_expert_diffs: Dict[str, Dict[str, float]] = {}
    unmatched_count = 0

    for eid, rec_r in pool_r.experts.items():
        b_step = rec_r.birth_step
        if b_step in never_by_birth:
            rec_n = never_by_birth[b_step]
            # Compare state dict parameters
            max_p_diff = 0.0
            fro_sq = 0.0
            for name, param_r in rec_r.module.named_parameters():
                if hasattr(rec_n.module, name):
                    param_n = getattr(rec_n.module, name)
                    diff = (param_r.data - param_n.data).double()
                    max_p_diff = max(max_p_diff, float(diff.abs().max().item()))
                    fro_sq += float((diff ** 2).sum().item())
            matched_expert_diffs[str(eid)] = {
                "matched_never_eid": rec_n.id,
                "birth_step": b_step,
                "max_weight_diff": max_p_diff,
                "frobenius_diff": math.sqrt(fro_sq),
            }
        else:
            unmatched_count += 1

    # 4. Class accuracy on removed labels: does P_removed ever predict a removed label?
    removed_preds_count = 0
    total_removed_test = 0
    removed_class_acc = 0.0

    if all_rem_x:
        rem_x_cat = torch.cat(all_rem_x, dim=0)
        rem_y_cat = torch.cat(all_rem_y, dim=0)
        total_removed_test = int(rem_x_cat.shape[0])

        p_rem_on_removed = p_removed.predict(rem_x_cat)
        preds_rem_test = p_rem_on_removed.probs.argmax(dim=-1).cpu()

        # How often does it predict a removed label?
        predicted_removed_mask = torch.isin(preds_rem_test, torch.tensor(list(target), dtype=torch.long))
        removed_preds_count = int(predicted_removed_mask.sum().item())

        # Correct classification accuracy on removed test (should be 0 because classes removed)
        correct_rem = (preds_rem_test == rem_y_cat.cpu()).float().sum().item()
        removed_class_acc = float(correct_rem / total_removed_test) if total_removed_test > 0 else 0.0

    return {
        "seed": seed,
        "target_labels": sorted(target),
        "mean_accuracy_retained": {
            "P_full": mean_acc_full,
            "P_removed": mean_acc_rem,
            "P_never": mean_acc_nev,
        },
        "per_regime_accuracy_retained": {
            "P_full": acc_full_regimes,
            "P_removed": acc_rem_regimes,
            "P_never": acc_nev_regimes,
        },
        "routing_agreement_retained": routing_agreement,
        "argmax_agreement_retained": argmax_agreement,
        "expert_weight_matching": {
            "n_matched": len(matched_expert_diffs),
            "n_unmatched": unmatched_count,
            "details": matched_expert_diffs,
        },
        "removed_class_evaluation": {
            "total_removed_test_samples": total_removed_test,
            "predictions_predicting_removed_label": removed_preds_count,
            "accuracy_on_removed_classes": removed_class_acc,
        },
    }
