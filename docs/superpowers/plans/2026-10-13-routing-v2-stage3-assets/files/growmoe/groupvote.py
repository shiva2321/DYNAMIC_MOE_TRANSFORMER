"""Stage 3b of routing v2: answer a group of inputs together (docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md, section 5).

Inputs that belong together (one document, one session, one batch from one source) are answered by one expert: the one most of them would pick on their own.
Individually each input has only a weak signal about its source; a group has a strong one. A per-input veto keeps an input with its own expert when the
group's expert does not accept it, so a group that mixes two sources is not forced onto one expert.

This module is the voting rule only (pure, no pool); ``PhyloMoE.predict(x, groups=...)`` supplies the veto and recomputes the answers.
"""

from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor


def vote_groups(choice: Tensor, groups: Tensor, group_min: int,
                veto: Optional[Callable[[int, Tensor], Tensor]] = None) -> Tuple[Tensor, Dict[str, int]]:
    """``choice`` (B,) long: the expert id each input would be answered by on its own (negative: none). ``groups`` (B,) integer: a group id per input
    (negative: not grouped). For every group with at least ``group_min`` members the group's expert is the most frequent non-negative choice among its
    members (ties: the lowest id) and every member is routed to it, except that with ``veto`` a member keeps its own choice when
    ``veto(expert, member_indices)`` is False for it. Returns ``(new_choice, info)``; the inputs are not modified."""
    if choice.dim() != 1 or groups.shape != choice.shape:
        raise ValueError(f"choice and groups must be 1-D tensors of the same length, got {tuple(choice.shape)} and {tuple(groups.shape)}")
    if groups.is_floating_point() or groups.dtype == torch.bool:
        raise ValueError("groups must be an integer tensor of group ids")
    if group_min < 1:
        raise ValueError("group_min must be at least 1")
    out = choice.clone()
    info = {"groups": 0, "moved": 0, "vetoed": 0}
    for g in torch.unique(groups[groups >= 0]).tolist():
        idx = (groups == g).nonzero().squeeze(1)
        if idx.numel() < group_min:
            continue
        own = choice[idx]
        valid = own >= 0
        if not valid.any():
            continue
        ids, counts = torch.unique(own[valid], return_counts=True)
        star = int(ids[counts == counts.max()].min())
        take = torch.ones(idx.numel(), dtype=torch.bool)
        if veto is not None:
            take = veto(star, idx).bool().reshape(-1)
            info["vetoed"] += int((~take).sum())
        out[idx[take]] = star
        info["groups"] += 1
        info["moved"] += int((out[idx] != own).sum())
    return out, info
