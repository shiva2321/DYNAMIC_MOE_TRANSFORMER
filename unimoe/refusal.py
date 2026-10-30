"""The refusal layer wired to the pool: say "not mine" for a message, with the false-refusal rate set on verified outcomes of the stream's own wording (the mechanism of notebook E-050).

Each reader gives every message a novelty score (larger = less like anything taught): the pool's own (1 minus its p-value) and the asking reader's (1 minus the probability of its best card). A
sample of VERIFIED known messages (a label confirms they belong) sets each reader's cut through conformal p-values; the readers are fused by "either" and the fused cut is set at level ``q``. With
fewer than 19 verified messages at q = 0.05 no cut exists and nothing is refused. The functions are pure; the experiment that feeds them real scores is ``p4_refusal.py``."""

import sys
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

_ROUTING_GENERAL = Path(__file__).resolve().parents[1] / "research" / "knowledge_status" / "routing_general"


def _measures():
    if str(_ROUTING_GENERAL) not in sys.path:
        sys.path.insert(0, str(_ROUTING_GENERAL))
    import measures as MS
    return MS


def refusal_decision(cal: Mapping[str, Sequence[float]], test: Mapping[str, Sequence[float]], sources: Sequence[str], q: float = 0.05, rule: str = "either") -> np.ndarray:
    """True where the message is refused. ``cal`` and ``test`` map a reader's name to its novelty scores of the verified known messages and of the messages to decide."""
    if not sources or any(s not in cal or len(cal[s]) == 0 for s in sources):
        raise ValueError("every reader needs calibration scores")
    MS = _measures()
    cal_map = {s: np.asarray(cal[s], dtype=np.float64) for s in sources}
    fused = MS.fused_scores(cal_map, {s: np.asarray(test[s], dtype=np.float64) for s in sources}, list(sources), rule)
    cut = MS.conformal_cut(MS.fused_scores(cal_map, cal_map, list(sources), rule, leave_one_out=True), q)
    return fused > cut


def selective_metrics(correct_known: np.ndarray, refuse_known: np.ndarray, refuse_ood: Mapping[str, np.ndarray]) -> Dict[str, object]:
    """What answering with a refusal option achieves: how much of the known traffic is answered, how accurate those answers are, and how much foreign text is still answered."""
    correct, refuse = np.asarray(correct_known, dtype=bool), np.asarray(refuse_known, dtype=bool)
    answered = ~refuse
    return {"false_refusal": float(refuse.mean()), "coverage": float(answered.mean()), "accuracy_all": float(correct.mean()),
            "accuracy_answered": float(correct[answered].mean()) if answered.any() else float("nan"),
            "ood_refused": {k: float(np.asarray(v, dtype=bool).mean()) for k, v in refuse_ood.items()},
            "ood_answered": {k: 1.0 - float(np.asarray(v, dtype=bool).mean()) for k, v in refuse_ood.items()}}
