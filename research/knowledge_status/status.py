"""The status machine (spec sections 2 and 6.2): for a part of a message, "solid", "thin", "unseen", "other" (known everyday content) or "conflict", with how sure it is.

What the machine is given, and nothing else: one ``SkillGroup`` per taught skill (its taught example vectors, how many examples it was taught, how well its solver does on problems it
has never seen) and one for Other content, plus held-out calibration vectors of the taught groups (surface families S1 to S3 only). It is never given a label for the part it
judges; ``assess`` takes a vector. A label-free mode (Phase 2, skills found from unlabelled streams) only has to build the same ``SkillGroup`` objects some other way.

How a part is judged
  1. NOVELTY. For every group a von Mises-Fisher kernel signature (``growmoe.signature.VmfSignature``, the same object the expert pool uses) scores the part, and the score is turned into
     a conformal p-value against the group's HELD-OUT calibration scores. A p-value at or below q means "unlike everything this group was taught". Under exchangeability a part of a
     taught group is flagged by its own group with probability at most q, whatever the distribution (finite-sample, distribution-free).
  2. COVERED OR UNSEEN. The part is "unseen" when EVERY group flags it. (Testing against all groups and not only the nearest one keeps the false-unseen rate at most q even when the
     reader sometimes puts a part nearer to the wrong group, for example division read as remainder; it is the usual conformal rule for open-set recognition.)
  3. ASSIGNMENT. Among the groups that do not flag it, the part goes to the one whose NEAREST taught prototype is closest in cosine (spec 6.2: "assigned by the nearest taught prototype").
  4. SOLID OR THIN. By the same rule that gives the status truth (``assignments.status_truth``): Solid needs at least ``solid_min_count`` examples AND held-out accuracy of at least
     ``solid_min_acc``; anything else that is covered is Thin. The grey zone of the truth rule is reported Thin here, because Solid means "demonstrated", and the grey cases are left out
     of the gated scores anyway. This is a declared simplification: competence is one number per group, not "competence near the part".
  Other content is reported "other" and never as a skill. It is still a taught group with a calibrated test of its own, so a content type nobody taught (a recipe sentence) is "unseen".

``evidence`` is the local count: how many taught examples of the assigned group lie as close to the part as the q-quantile of the calibration points' nearest-prototype cosines (the radius
inside which a typical held-out point has a taught neighbour), scaled up when the prototypes are a subsample of the taught examples. It is reported, and it is what Task 11 scores Thin
against Solid with; it does not override the group-level rule above, because held-out wordings sit a little farther from the prototypes than the taught ones and would otherwise all be Thin.

``assess_group`` judges a batch of parts that belong together with the permutation test of ``growmoe.settest`` against each group's taught vectors (nearest group first), which has
power a single vector does not have. ``related`` is never returned here (relation comes from worked traces, Task 12). ``conflict`` is returned only through ``flag_conflict``.
"""

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from assignments import THRESHOLDS, Solver
from growmoe import settest
from growmoe.signature import VmfSignature

STATUSES = ("solid", "thin", "unseen", "other", "related", "conflict")
SCORED = ("solid", "thin", "unseen")                    # the classes E2 scores with macro F1 (Other content and Conflict are reported apart)
_CHUNK = 2048
_ZERO = 1e-12


# ---------------------------------------------------------------------------------------------------------------------------------- data
@dataclass(frozen=True, eq=False)
class SkillGroup:
    """The only way the machine sees a skill. ``protos`` are unit-length (or any non-zero) vectors of taught examples in the reader's space, ``n_examples`` how many examples the skill
    was taught (the vectors may be a subsample), ``solver`` the small learner of the skill (used by traces, not by ``assess``), ``competence`` its accuracy on problems it never saw."""
    name: str
    protos: np.ndarray
    n_examples: int
    solver: Optional[Solver]
    competence: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a group needs a non-empty name")
        p = np.asarray(self.protos, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] < 2:
            raise ValueError(f"group {self.name!r}: needs at least two prototypes as a 2-D array, got shape {p.shape}")
        if not np.isfinite(p).all():
            raise ValueError(f"group {self.name!r}: prototypes must be finite")
        if (np.linalg.norm(p, axis=1) < _ZERO).any():
            raise ValueError(f"group {self.name!r}: a prototype is the zero vector")
        if not isinstance(self.n_examples, Integral) or self.n_examples < 1:
            raise ValueError(f"group {self.name!r}: n_examples must be a positive integer, got {self.n_examples!r}")
        if not (isinstance(self.competence, (int, float, np.floating)) and 0.0 <= float(self.competence) <= 1.0):
            raise ValueError(f"group {self.name!r}: competence must lie in [0, 1], got {self.competence!r}")
        object.__setattr__(self, "protos", p)


@dataclass(frozen=True, eq=False)
class Assessment:
    """``status`` is one of STATUSES. ``skill`` is the assigned skill (None for unseen and for other). ``p_value`` is the conformal (or permutation) p-value of the assigned group, or the
    largest over the groups when the part is unseen. ``competence`` is NaN when no group covers the part. ``nearest`` is the group with the nearest prototype whatever the status was."""
    status: str
    skill: Optional[str]
    p_value: float
    evidence: int
    competence: float
    group_size: int
    nearest: str = ""

    def _key(self):
        return (self.status, self.skill, self.p_value, self.evidence, None if math.isnan(self.competence) else self.competence, self.group_size, self.nearest)

    def __eq__(self, other) -> bool:                    # NaN competence compares equal to itself here, so two identical assessments are equal
        return isinstance(other, Assessment) and self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())


# ---------------------------------------------------------------------------------------------------------------------------------- rules
def group_state(count: int, accuracy: float, thresholds: Dict = THRESHOLDS) -> str:
    """"solid" or "thin" for a group taught ``count`` examples whose solver scored ``accuracy`` on unseen problems. The same rule as ``assignments.status_truth`` (count first, then
    accuracy), with the grey zone reported thin; the thresholds are the caller's, so a change of THRESHOLDS on dev takes effect without touching this function."""
    t = thresholds
    if count <= t["thin_max_count"]:
        return "thin"
    if count >= t["solid_min_count"] and accuracy >= t["solid_min_acc"] - 1e-9:
        return "solid"
    return "thin"


def conformal_p(cal_sorted: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Conformal p-values in float64: (1 + #{calibration scores <= score}) / (n + 1), small = novel. The same numbers as ``growmoe.conformal.conformal_pvalues``, but kept in float64 so that a
    p-value that is exactly q (19 calibration scores, none below the part, q = 0.05) compares equal to q instead of to a float32 neighbour of it."""
    n = len(cal_sorted)
    if n == 0:
        raise ValueError("conformal_p needs at least one calibration score")
    return (1.0 + np.searchsorted(cal_sorted, scores, side="right")) / (n + 1.0)


def macro_f1(truth: Sequence[str], pred: Sequence[str], classes: Sequence[str] = SCORED, outside: Sequence[str] = ()) -> dict:
    """Macro F1 over the classes that are PRESENT, i.e. that occur in the truth or in the predictions. A class that occurs in neither is left out and listed in ``absent`` (its F1 is
    undefined, and counting it as 0 or 1 would move the score for no reason); a class the machine predicts although the truth never has it is present and scores 0. A prediction
    listed in ``outside`` (for example "other" for an arithmetic part) is not a class of this score: it is a miss of the true class and a false positive of nothing."""
    if len(truth) != len(pred):
        raise ValueError("truth and pred must have the same length")
    if len(truth) == 0:
        raise ValueError("cannot score an empty list")
    unknown = (set(truth) - set(classes)) | (set(pred) - set(classes) - set(outside))
    if unknown:
        raise ValueError(f"unknown labels {sorted(unknown)}; the scored classes are {list(classes)} (filter grey and other cases out first)")
    t, p = np.asarray(truth), np.asarray(pred)
    per: Dict[str, float] = {}
    absent: List[str] = []
    for c in classes:
        tp, fp, fn = int(((t == c) & (p == c)).sum()), int(((t != c) & (p == c)).sum()), int(((t == c) & (p != c)).sum())
        if tp + fp + fn == 0:
            absent.append(c)
        else:
            per[c] = 2.0 * tp / (2.0 * tp + fp + fn)
    return {"macro_f1": float(np.mean(list(per.values()))), "per_class": per, "absent": absent, "present": list(per), "n": len(t)}


# ---------------------------------------------------------------------------------------------------------------------------------- the machine
def _as_rows(x, d: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != d:
        raise ValueError(f"part vectors must have dimension {d}, got shape {a.shape}")
    if a.shape[0] == 0:
        raise ValueError("no part vectors given")
    if not np.isfinite(a).all():
        raise ValueError("part vectors must be finite")
    norm = np.linalg.norm(a, axis=1)
    if (norm < _ZERO).any():
        raise ValueError("a zero vector has no direction; it cannot be assessed")
    return a / norm[:, None]


class StatusMachine:
    def __init__(self, groups: Dict[str, SkillGroup], other: SkillGroup, cal_vectors: np.ndarray, cal_groups: np.ndarray, q: float = 0.05,
                 thresholds: Dict = THRESHOLDS, seed: int = 0, max_protos: int = 256, n_perm: int = 999, ref_cap: int = 512) -> None:
        if not 0.0 < q < 1.0:
            raise ValueError(f"q must lie strictly between 0 and 1, got {q}")
        if not groups:
            raise ValueError("the machine needs at least one skill group")
        for key, g in groups.items():
            if key != g.name:
                raise ValueError(f"group key {key!r} does not match the group's name {g.name!r}")
        if other.name in groups:
            raise ValueError(f"a skill and the Other-content group cannot have the same name ({other.name!r})")
        self.names = [*groups, other.name]
        self.groups: Dict[str, SkillGroup] = {**groups, other.name: other}
        self.other_name = other.name
        self.q, self.thresholds, self.seed, self.n_perm, self.ref_cap = float(q), thresholds, int(seed), int(n_perm), int(ref_cap)
        self.d = other.protos.shape[1]
        for g in self.groups.values():
            if g.protos.shape[1] != self.d:
                raise ValueError(f"group {g.name!r} has dimension {g.protos.shape[1]}, the others have {self.d}")
        cal = np.asarray(cal_vectors, dtype=np.float64)
        cg = np.asarray(cal_groups)
        if cal.ndim != 2 or len(cal) != len(cg):
            raise ValueError(f"calibration vectors {cal.shape} and calibration groups {cg.shape} must be a matrix and one group name per row")
        if cal.shape[1] != self.d:
            raise ValueError(f"calibration vectors have dimension {cal.shape[1]}, the groups have {self.d}")
        if not np.isfinite(cal).all() or (np.linalg.norm(cal, axis=1) < _ZERO).any():
            raise ValueError("calibration vectors must be finite and non-zero")
        unknown = set(cg.tolist()) - set(self.names)
        if unknown:
            raise ValueError(f"calibration data name unknown group(s) {sorted(unknown)}")
        need = math.ceil(1.0 / self.q - 1e-9) - 1                    # fewer calibration scores than this can never produce a p-value <= q
        self._P: List[np.ndarray] = []
        self._sig: List[VmfSignature] = []
        self._cal: List[np.ndarray] = []
        self._radius: List[float] = []
        for k, name in enumerate(self.names):
            mine = cal[cg == name]
            if len(mine) < need:
                raise ValueError(f"group {name!r} has {len(mine)} calibration vectors; q={self.q} needs at least {need} for a test that can ever flag")
            P = (self.groups[name].protos / np.linalg.norm(self.groups[name].protos, axis=1, keepdims=True)).astype(np.float32)
            sig = VmfSignature(self.d, max_protos=max_protos, seed=self.seed + k, dedupe=True)
            sig.update(P)
            if sig.n < 2:
                raise ValueError(f"group {name!r} has fewer than two distinct prototypes")
            sig.fit()
            rows = (mine / np.linalg.norm(mine, axis=1, keepdims=True)).astype(np.float32)
            self._cal.append(np.sort(sig.score(rows).double().numpy()))
            self._radius.append(float(np.quantile((rows @ P.T).max(axis=1), self.q)))
            self._P.append(P)
            self._sig.append(sig)

    # -------------------------------------------------------------------------------------------------------------------------- helpers
    def _status_of(self, name: str) -> str:
        if name == self.other_name:
            return "other"
        g = self.groups[name]
        return group_state(g.n_examples, float(g.competence), self.thresholds)

    def _evidence(self, j: int, U32: np.ndarray) -> np.ndarray:
        """Taught examples of group j within its radius of each row, scaled from the prototypes to the examples."""
        m, scale = len(self._P[j]), self.groups[self.names[j]].n_examples / len(self._P[j])
        out = np.empty(len(U32), dtype=np.int64)
        for s in range(0, len(U32), _CHUNK):
            out[s:s + _CHUNK] = np.rint(((U32[s:s + _CHUNK] @ self._P[j].T) >= self._radius[j]).sum(axis=1) * scale)
        return out

    def _build(self, j: int, p: float, evidence: int, size: int, nearest: str) -> Assessment:
        name = self.names[j]
        status = self._status_of(name)
        return Assessment(status, None if status == "other" else name, float(p), int(evidence), float(self.groups[name].competence), size, nearest)

    # -------------------------------------------------------------------------------------------------------------------------- single parts
    def assess(self, part_vec: np.ndarray) -> Assessment:
        return self.assess_many(part_vec)[0]

    def assess_many(self, part_vecs: np.ndarray) -> List[Assessment]:
        """``assess`` for many parts at once (rows). The same decision for a row whatever else is in the call."""
        U = _as_rows(part_vecs, self.d)
        U32 = U.astype(np.float32)
        n, G = len(U), len(self.names)
        p, cosmax = np.empty((n, G)), np.empty((n, G), dtype=np.float32)
        for j in range(G):
            for s in range(0, n, _CHUNK):
                u = U32[s:s + _CHUNK]
                p[s:s + _CHUNK, j] = conformal_p(self._cal[j], self._sig[j].score(u).double().numpy())
                cosmax[s:s + _CHUNK, j] = (u @ self._P[j].T).max(axis=1)
        covered = p > self.q
        assigned = np.where(covered, cosmax, -np.inf).argmax(axis=1)
        nearest = cosmax.argmax(axis=1)
        evidence = np.zeros(n, dtype=np.int64)
        for j in range(G):
            rows = np.flatnonzero(covered.any(axis=1) & (assigned == j))
            if rows.size:
                evidence[rows] = self._evidence(j, U32[rows])
        out: List[Assessment] = []
        for i in range(n):
            if not covered[i].any():
                out.append(Assessment("unseen", None, float(p[i].max()), 0, math.nan, 1, self.names[nearest[i]]))
            else:
                j = int(assigned[i])
                out.append(self._build(j, p[i, j], evidence[i], 1, self.names[nearest[i]]))
        return out

    # -------------------------------------------------------------------------------------------------------------------------- batches
    def assess_group(self, part_vecs: np.ndarray) -> Assessment:
        """One assessment for a batch of parts that belong together: the groups are tried nearest first (mean over the batch of the nearest-prototype cosine) with the permutation test of
        ``growmoe.settest.perm_pvalue`` against the group's taught vectors (at most ``ref_cap`` of them); the first group that does not flag the batch takes it. If every group flags it,
        the batch is unseen. Each test draws from a generator keyed by (seed, group, batch size), so the answer does not depend on what was assessed before."""
        U = _as_rows(part_vecs, self.d)
        U32, k = U.astype(np.float32), len(U)
        mean_cos = np.array([(U32 @ P.T).max(axis=1).mean() for P in self._P])
        order = [int(j) for j in np.argsort(-mean_cos, kind="stable")]
        batch = torch.from_numpy(U32)
        ps: Dict[int, float] = {}
        for j in order:
            gen = settest.generator(self.seed, 0x6772, j, k)
            ref = settest.cap_rows(torch.from_numpy(self._P[j]), self.ref_cap, gen)
            ps[j] = settest.perm_pvalue(ref, batch, n_perm=self.n_perm, gen=gen)
            if ps[j] > self.q:
                return self._build(j, ps[j], int(round(float(self._evidence(j, U32).mean()))), k, self.names[order[0]])
        return Assessment("unseen", None, max(ps.values()), 0, math.nan, k, self.names[order[0]])

    # -------------------------------------------------------------------------------------------------------------------------- conflict
    def flag_conflict(self, part_vec: np.ndarray, known_answer_disagreement: bool) -> Assessment:
        """"conflict" for a part that a known skill covers (solid or thin) when the known answer disagrees with the context (a rule that changed). A part nothing covers has nothing to
        contradict and keeps its status. Not gated in Phase 1 (spec 5.6)."""
        a = self.assess(part_vec)
        if known_answer_disagreement and a.status in ("solid", "thin"):
            return replace(a, status="conflict")
        return a
