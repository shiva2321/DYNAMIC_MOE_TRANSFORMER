"""How many experts, and which? (Lab D)

The router answers with a TEAM of experts for a part. The size is not fixed: an expert is in the team only if it is usable (competent enough to be trusted with the part) and adds something the others
do not; the cost of calling an expert is counted. Policies compared here: the single most probable expert, the conformal covering set (empty = nothing taught explains the part), a fixed top-2 or
top-3, and every taught expert. A team can only be told apart from a single expert when something checks its members: the stated answer of the part (a worked example, a teacher's answer, the
feedback of a stream) is reproduced, or not, by each member. Without that, extra experts add cost and no information, and the team answers with its most probable member.
"""

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

_KS = Path(__file__).resolve().parents[1]
for _p in (str(_KS.parents[1]), str(_KS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import ask_reader as AR  # noqa: E402
import skills  # noqa: E402

POLICIES = ("top1", "set", "top2", "top3", "all", "mass0.9", "mass0.99")


def build_team(policy: str, p_row: np.ndarray, covered_row: np.ndarray, known: Sequence[str], usable: Sequence[str], cap: int = 3) -> List[str]:
    """The team for one part. ``p_row`` are the asked probabilities over AR.OPS, ``covered_row`` the covering flags over ``known`` (taught skills + "none"); only skills in ``usable`` may be in
    the team. The team is ordered by probability, most probable first."""
    skills_known = [k for k in known if k != "none" and k in usable]
    by_p = sorted(skills_known, key=lambda k: -float(p_row[AR.OPS.index(k)]))
    if policy == "top1":
        return by_p[:1]
    if policy == "set":
        cov = {k for k, c in zip(known, covered_row) if c}
        return [k for k in by_p if k in cov][:cap]
    if policy in ("top2", "top3"):
        return by_p[:int(policy[3:])]
    if policy == "all":
        return by_p
    if policy.startswith("mass"):                                                                  # add experts until the team holds this share of the probability on the taught skills
        mass = float(policy[4:])
        total = sum(float(p_row[AR.OPS.index(k)]) for k in skills_known)
        team, held = [], 0.0
        for k in by_p[:cap]:
            team.append(k)
            held += float(p_row[AR.OPS.index(k)])
            if total <= 0 or held / total >= mass:
                break
        return team
    raise ValueError(f"unknown policy {policy!r}")


def choose(team: Sequence[str], a: int, b: int, answer: Optional[int], p_row: np.ndarray, verifiers: Sequence[str]) -> Optional[str]:
    """The skill the team settles on. With a stated answer, the members that can verify (``verifiers``) and reproduce it are kept; one hit decides, several hits go to the most probable of them,
    none leaves the most probable member. Without an answer, or with no verifier, the most probable member decides. An empty team settles on nothing."""
    if not team:
        return None
    if answer is not None:
        hits = []
        for s in team:
            if s not in verifiers:
                continue
            try:
                if skills.truth(s, a, b) == answer:
                    hits.append(s)
            except ValueError:
                continue
        if hits:
            return max(hits, key=lambda s: float(p_row[AR.OPS.index(s)]))
    return max(team, key=lambda s: float(p_row[AR.OPS.index(s)]))


def evaluate(policy: str, rows: Sequence[dict], known: Sequence[str], usable: Sequence[str], verifiers: Sequence[str], with_answer: bool, cap: int = 3) -> Dict[str, float]:
    """Over ``rows`` (each: p_row, covered_row, a, b, answer, skill): how often the true skill is in the team, the mean team size, and how often the team settles on the true skill."""
    inside, size, right = [], [], []
    for r in rows:
        team = build_team(policy, r["p_row"], r["covered_row"], known, usable, cap)
        inside.append(r["skill"] in team)
        size.append(len(team))
        right.append(choose(team, r["a"], r["b"], r["answer"] if with_answer else None, r["p_row"], verifiers) == r["skill"])
    return {"recall": float(np.mean(inside)), "size": float(np.mean(size)), "right": float(np.mean(right))}
