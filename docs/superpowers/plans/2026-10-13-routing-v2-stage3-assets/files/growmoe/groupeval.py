"""Group evaluation for routing v2 stage 3b: build groups of held-out inputs and score a pool per input and per group on the SAME inputs.

``make_groups`` builds the evaluation sets the pre-registration names: pure groups (every member from one regime) and mixed groups (``size - minor`` members
from one regime and ``minor`` from the next one). ``evaluate_groups`` answers the rows twice with the same pool, once input by input and once group by group,
and returns both accuracies, so the difference is a paired comparison with no training noise in it.
"""

from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor


def make_groups(test_sets: Dict[int, Tuple[Tensor, Tensor]], size: int = 64, minor: int = 0, seed: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
    """Return ``(x, y, groups)``: rows of the held-out sets and a group id per row.

    ``minor == 0``: pure groups. Every regime contributes ``len(test set) // size`` groups of ``size`` rows, all from that regime (the remainder is left out).
    ``minor > 0``: mixed groups. A group holds ``size - minor`` rows of regime ``r`` and ``minor`` rows of the next regime in id order (cyclic); regime ``r``
    contributes as many groups as its set and its partner's set allow. Rows are drawn without replacement inside each role and a different permutation
    per regime and role, fixed by ``seed``. A regime without a full group contributes none; if no group at all can be built the call raises ValueError."""
    if size < 2 or not 0 <= minor < size or (minor and size - minor < minor):
        raise ValueError("size must be at least 2, 0 <= minor < size, and a mixed group needs size - minor >= minor")
    ids = sorted(test_sets)
    perm = {r: np.random.default_rng([seed, 0x47525550, r]).permutation(len(test_sets[r][1])) for r in ids}
    xs, ys, gs, gid = [], [], [], 0
    for pos, r in enumerate(ids):
        major = size - minor
        n_groups = len(perm[r]) // major
        if minor:
            partner = ids[(pos + 1) % len(ids)]
            if partner == r:
                continue
            n_groups = min(n_groups, len(perm[partner]) // minor)
        for g in range(n_groups):
            take = perm[r][g * major:(g + 1) * major]
            xs.append(test_sets[r][0][take])
            ys.append(test_sets[r][1][take])
            if minor:
                n_p = len(perm[partner])
                tail = perm[partner][n_p - (g + 1) * minor:n_p - g * minor]                # the partner's rows come from the other end of its permutation
                xs.append(test_sets[partner][0][tail])
                ys.append(test_sets[partner][1][tail])
            gs.append(torch.full((size,), gid, dtype=torch.long))
            gid += 1
    if not xs:
        raise ValueError(f"no group of {size} (minor {minor}) can be built from test sets of sizes {[len(test_sets[r][1]) for r in ids]}")
    return torch.cat(xs), torch.cat(ys), torch.cat(gs)


def evaluate_groups(pool, x: Tensor, y: Tensor, groups: Tensor) -> Dict[str, float]:
    """Accuracy of ``pool`` on the rows ``(x, y)`` answered input by input (``single``) and answered by group (``group``), plus how many answers changed.
    The pool must have group answering switched on (``pool.cfg.group_vote``): otherwise the two answers are identical by construction and the comparison
    would silently measure nothing."""
    if not getattr(pool.cfg, "group_vote", False):
        raise ValueError("pool.cfg.group_vote is False: the grouped answer would equal the single answer, so this evaluation would be meaningless")
    single = pool.predict(x)
    grouped = pool.predict_grouped(x, groups)
    hit_single = (single.probs.argmax(1) == y).float()
    hit_group = (grouped.probs.argmax(1) == y).float()
    return {"single": float(hit_single.mean()), "group": float(hit_group.mean()), "n": int(y.numel()), "n_groups": int(torch.unique(groups).numel()),
            "changed": int((single.probs.argmax(1) != grouped.probs.argmax(1)).sum())}
