"""Adaptive teams: use as many experts as an input needs and no more.

``choose_team`` turns routing weights (B, E) into a team per input: the smallest set of experts, in order of weight, whose weights add up to ``mass``, at most ``cap`` of them, weights renormalised.
An input one expert is sure about gets a team of one (the cost of a plain top-1 MoE); an input that two experts share gets both. ``mix`` blends the members' class distributions with those weights.
"""

from typing import Tuple

import torch
from torch import Tensor


def choose_team(weights: Tensor, mass: float = 0.99, cap: int = 3) -> Tuple[Tensor, Tensor]:
    """(members (B, cap) long with -1 padding, weights (B, cap) summing to 1 over the members)."""
    if weights.ndim != 2 or weights.shape[1] == 0:
        raise ValueError(f"weights must be (B, E) with E >= 1, got {tuple(weights.shape)}")
    w = weights.clamp_min(0.0)
    w = w / w.sum(1, keepdim=True).clamp_min(1e-30)
    k = min(cap, w.shape[1])
    top_w, top_i = w.topk(k, dim=1)
    before = torch.cumsum(top_w, dim=1) - top_w                      # mass taken by the members ranked above this one
    keep = before < mass - 1e-12
    keep[:, 0] = True
    kept_w = torch.where(keep, top_w, torch.zeros_like(top_w))
    kept_w = kept_w / kept_w.sum(1, keepdim=True).clamp_min(1e-30)
    members = torch.where(keep, top_i, torch.full_like(top_i, -1))
    if k < cap:
        pad = cap - k
        members = torch.cat([members, torch.full((members.shape[0], pad), -1, dtype=members.dtype)], 1)
        kept_w = torch.cat([kept_w, torch.zeros(kept_w.shape[0], pad)], 1)
    return members, kept_w


def mix(probs: Tensor, members: Tensor, weights: Tensor) -> Tensor:
    """``probs`` (E, B, C) class distributions of every expert on every input; returns (B, C) = the weighted blend of each input's team."""
    E, B, C = probs.shape
    out = torch.zeros(B, C)
    rows = torch.arange(B)
    for j in range(members.shape[1]):
        ok = members[:, j] >= 0
        if bool(ok.any()):
            out[ok] += weights[ok, j:j + 1] * probs[members[ok, j], rows[ok]]
    return out


def team_size(members: Tensor) -> Tensor:
    return (members >= 0).sum(1)
