"""Scoring (spec section 7): status scores for E2, novelty-only scores for the baselines, the skill confusion, the tangled-message numbers B4a, B4b, B6, the wording claim and the
offline consequence estimate. Pure functions over lists, with no model and no I/O, so every number can be checked by hand.

A TRUTH is a ``Case``: the status the part should get ("solid", "thin", "unseen", "other", or "grey" for the zone in between that is reported but never scored), the surface
family, the true skill (None for non-arithmetic parts) and the kind of part: "arith" (a request in one of the families), "twin" (a near-twin of a request: only its operator
changed), "trap" (arithmetic words used for something else), "other" (an everyday sentence) or "unseen_content" (a content type that was never taught).
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

import assignments
import skills
import status as ST

CASE_STATUSES = ("solid", "thin", "unseen", "other", "grey")
PREDICTED = ("solid", "thin", "unseen", "other")
HELD_OUT = tuple(assignments.TEST_FAMILIES)
TAUGHT_WORDING = tuple(assignments.TEACH_FAMILIES)
DEFAULT_COSTS = {"false_unseen": 4.0, "missed_unseen": 7.0}      # pool points per error: a skill learned twice (E-034), a new skill absorbed by an old one (E-030)
NAN = float("nan")


@dataclass(frozen=True)
class Case:
    status: str
    family: str
    skill: Optional[str]
    kind: str

    def __post_init__(self) -> None:
        if self.status not in CASE_STATUSES:
            raise ValueError(f"unknown truth status {self.status!r}; the truth statuses are {CASE_STATUSES}")


def _rate(mask: np.ndarray, flags: np.ndarray) -> float:
    return float(flags[mask].mean()) if mask.any() else NAN


def _check(n_pred: int, truth: Sequence[Case]) -> None:
    if n_pred != len(truth):
        raise ValueError(f"predictions and truth must have the same length, got {n_pred} and {len(truth)}")


# ---------------------------------------------------------------------------------------------------------------------------------- novelty only
def flag_scores(flags: Sequence[bool], truth: Sequence[Case], scores: Optional[Sequence[float]] = None) -> dict:
    """Everything that needs only "was this part called unseen": the false-unseen rates on held-out wordings (S4 to S8) and on the taught wordings (S1 to S3), the recall of untaught
    skills, Other content and traps called unseen, never-taught content called unseen, and the same for near-twins. With raw novelty ``scores`` (higher = newer) also the AUROC of
    parts of untaught skills against parts of taught skills. Grey cases are left out of every number."""
    _check(len(flags), truth)
    f = np.asarray(flags, dtype=bool)
    kind = np.array([c.kind for c in truth])
    state = np.array([c.status for c in truth])
    family = np.array([c.family for c in truth])
    arith = (kind == "arith") & (state != "grey")
    taught = arith & np.isin(state, ("solid", "thin"))
    held = np.isin(family, HELD_OUT)
    out = {
        "false_unseen_heldout": _rate(taught & held, f),
        "false_unseen_taught_wording": _rate(taught & ~held, f),
        "false_unseen_by_family": {fam: _rate(taught & (family == fam), f) for fam in sorted(set(family[taught & held].tolist()))},
        "unseen_recall": _rate(arith & (state == "unseen"), f),
        "other_as_unseen": _rate(kind == "other", f),
        "trap_as_unseen": _rate(kind == "trap", f),
        "unseen_content_flagged": _rate(kind == "unseen_content", f),
        "twin_false_unseen": _rate((kind == "twin") & np.isin(state, ("solid", "thin")), f),
        "twin_unseen_recall": _rate((kind == "twin") & (state == "unseen"), f),
    }
    if scores is not None:
        s = np.asarray(scores, dtype=np.float64)
        _check(len(s), truth)
        use = arith & np.isin(state, ("solid", "thin", "unseen"))
        y = (state[use] == "unseen").astype(int)
        out["auroc_taught_vs_untaught"] = float(roc_auc_score(y, s[use])) if 0 < y.sum() < len(y) else NAN
    return out


# ---------------------------------------------------------------------------------------------------------------------------------- status
def status_scores(assessments: Sequence[ST.Assessment], truth: Sequence[Case]) -> dict:
    """E2 scores (B3a): macro F1 over Solid, Thin and Unseen for the arithmetic parts (an arithmetic part called Other content is a miss), the confusion counts, Thin against Solid by
    the AUROC of log(1 + evidence) x competence among parts a skill covers, and everything ``flag_scores`` gives. Grey cases are counted in ``n_grey`` and not scored."""
    _check(len(assessments), truth)
    bad = {a.status for a in assessments} - set(PREDICTED)
    if bad:
        raise ValueError(f"unexpected predicted status {sorted(bad)}; assess() returns only {PREDICTED}")
    out = flag_scores([a.status == "unseen" for a in assessments], truth)
    use = [k for k, c in enumerate(truth) if c.kind == "arith" and c.status != "grey"]
    out["n_arith"] = len(use)
    out["n_grey"] = sum(c.status == "grey" for c in truth)
    out["macro_f1"] = ST.macro_f1([truth[k].status for k in use], [assessments[k].status for k in use], outside=("other",)) if use else None
    conf = {t: {p: 0 for p in PREDICTED} for t in ST.SCORED}
    for k in use:
        conf[truth[k].status][assessments[k].status] += 1
    out["confusion"] = conf
    covered = [k for k in use if truth[k].status in ("solid", "thin") and assessments[k].status in ("solid", "thin")]
    y = np.array([truth[k].status == "solid" for k in covered], dtype=int)
    if 0 < y.sum() < len(y):
        sc = [math.log1p(assessments[k].evidence) * (0.0 if math.isnan(assessments[k].competence) else assessments[k].competence) for k in covered]
        out["thin_vs_solid_auroc"] = float(roc_auc_score(y, sc))
    else:
        out["thin_vs_solid_auroc"] = NAN
    kind = np.array([c.kind for c in truth])
    pred = np.array([a.status for a in assessments])
    out["other_labelled"] = float((pred[kind == "other"] == "other").mean()) if (kind == "other").any() else NAN
    out["trap_labelled_other"] = float((pred[kind == "trap"] == "other").mean()) if (kind == "trap").any() else NAN
    twins = [k for k, c in enumerate(truth) if c.kind == "twin" and c.status != "grey"]
    out["twin_accuracy"] = float(np.mean([assessments[k].status == truth[k].status for k in twins])) if twins else NAN
    return out


def skill_confusion(assessments: Sequence[ST.Assessment], truth: Sequence[Case]) -> Tuple[np.ndarray, List[str], List[str]]:
    """Misassignment between skills as its own failure mode (spec 6.2): rows are the TRUE skills, columns the group with the nearest taught prototype (the six skills and "other"),
    over every arithmetic part and near-twin whatever its status. An untaught skill has no column of its own, so its row shows which taught skill absorbs it."""
    _check(len(assessments), truth)
    rows, cols = list(skills.SKILLS), list(skills.SKILLS) + ["other"]
    m = np.zeros((len(rows), len(cols)), dtype=np.int64)
    for a, c in zip(assessments, truth):
        if c.kind in ("arith", "twin") and c.skill is not None:
            if a.nearest not in cols:
                raise ValueError(f"assessment has no nearest group ({a.nearest!r}); skill_confusion needs the nearest field")
            m[rows.index(c.skill), cols.index(a.nearest)] += 1
    return m, rows, cols


# ---------------------------------------------------------------------------------------------------------------------------------- tangled messages
Segment = Tuple[Tuple[int, int], ST.Assessment]


def _mask(n: int, spans) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in spans:
        m[s:e] = True
    return m


def _belongs(seg: Tuple[int, int], gold: Tuple[int, int]) -> bool:
    """A segment belongs to a gold part when MORE than half of its characters lie in it (a segment straddling two parts evenly belongs to neither)."""
    overlap = max(0, min(seg[1], gold[1]) - max(seg[0], gold[0]))
    return 2 * overlap > (seg[1] - seg[0])


def gold_status(part, states: Mapping[str, str]) -> str:
    if part.kind == "skill":
        return states[part.skill]
    return "unseen" if part.kind == "unseen_content" else "other"


def tangled_scores(msgs, assessments_by_part: Sequence[Sequence[Segment]], mode: str, states: Mapping[str, str]) -> dict:
    """B4a, B4b (and B4c) for two-part messages. ``assessments_by_part[i]`` lists ``((start, end), Assessment)`` for the parts that were assessed in message i: the gold parts
    ("gold") or the pieces of an automatic cutter ("auto_sentence", "auto_window"). ``states`` maps a skill to its taught state. A message is judged by what its gold parts are:
      B4a  a known (solid or thin) request and an unseen request. ``residual_hit``: the characters called unseen cover at least half of the unseen part and more than half of them lie
           in it (the leftover really is the new piece); ``known_not_flagged``: no piece that belongs to the known part is called unseen.
      B4b  a known request and Other content. ``nothing_flagged``: nothing in the message is called unseen; ``other_labelled``: every piece of the Other part is labelled other.
      B4c  an unseen request and Other content. ``unseen_found``: at least half of the unseen part is called unseen; ``other_labelled`` as above.
    Any other combination (two known requests, a grey state) is counted in ``n_ignored``."""
    if len(msgs) != len(assessments_by_part):
        raise ValueError(f"msgs and assessments_by_part must have the same length, got {len(msgs)} and {len(assessments_by_part)}")
    rows: Dict[str, List[dict]] = {"b4a": [], "b4b": [], "b4c": []}
    ignored = 0
    for msg, segs in zip(msgs, assessments_by_part):
        gold = [((p.start, p.end), gold_status(p, states)) for p in msg.parts]
        known = [g for g, s in gold if s in ("solid", "thin")]
        unseen = [g for g, s in gold if s == "unseen"]
        other = [g for g, s in gold if s == "other"]
        flagged = [span for span, a in segs if a.status == "unseen"]
        mask = _mask(len(msg.text), flagged)

        def cover(g) -> float:
            return float(mask[g[0]:g[1]].mean()) if g[1] > g[0] else 0.0

        def other_labelled(g) -> bool:
            mine = [a for span, a in segs if _belongs(span, g)]
            return bool(mine) and all(a.status == "other" for a in mine)

        if len(gold) == 2 and len(known) == 1 and len(unseen) == 1:
            u, k = unseen[0], known[0]
            precision = float(mask[u[0]:u[1]].sum() / mask.sum()) if mask.any() else 0.0
            rows["b4a"].append({"residual_hit": cover(u) >= 0.5 and precision > 0.5, "known_not_flagged": not any(_belongs(span, k) for span in flagged)})
        elif len(gold) == 2 and len(known) == 1 and len(other) == 1:
            rows["b4b"].append({"nothing_flagged": not flagged, "other_labelled": other_labelled(other[0])})
        elif len(gold) == 2 and len(unseen) == 1 and len(other) == 1:
            rows["b4c"].append({"unseen_found": cover(unseen[0]) >= 0.5, "other_labelled": other_labelled(other[0])})
        else:
            ignored += 1
    out: dict = {"mode": mode, "n_ignored": ignored}
    for key, group in rows.items():
        fields = {"b4a": ("residual_hit", "known_not_flagged"), "b4b": ("nothing_flagged", "other_labelled"), "b4c": ("unseen_found", "other_labelled")}[key]
        out[key] = {"n": len(group), **{f: (float(np.mean([r[f] for r in group])) if group else NAN) for f in fields}}
    return out


def b6_gap(gold: dict, auto: dict, tol: float = 0.10) -> dict:
    """B6: the two B4a numbers under automatic segmentation against the gold-span numbers; ``within`` is True when both differ by at most ``tol``."""
    gaps = {k: abs(gold["b4a"][k] - auto["b4a"][k]) for k in ("residual_hit", "known_not_flagged")}
    return {**gaps, "within": bool(all(g <= tol + 1e-9 for g in gaps.values()))}


# ---------------------------------------------------------------------------------------------------------------------------------- claims
def wording_claim(false_unseen_standard: float, false_unseen_adapted: float, bar: float = 0.05) -> str:
    """The calibration claim (spec 6.2): with calibration on S1 to S3 only the false-unseen rate on S4 to S8 must be at most ``bar``. If only the ADAPTED condition (a few examples of
    S4 to S8 in the calibration) reaches the bar, the claim is downgraded to "needs adaptation to new wording", not passed."""
    if math.isnan(false_unseen_standard):
        return "undetermined"
    if false_unseen_standard <= bar + 1e-12:
        return "holds"
    if not math.isnan(false_unseen_adapted) and false_unseen_adapted <= bar + 1e-12:
        return "needs adaptation to new wording"
    return "fails"


def gate(false_unseen_standard: float, false_unseen_adapted: float, bar: float = 0.05) -> dict:
    """The gate reads the standard condition only; the adapted number is reported beside it and can neither pass nor fail the gate."""
    return {"pass": bool(false_unseen_standard <= bar + 1e-12), "adapted_pass": bool(false_unseen_adapted <= bar + 1e-12), "gated_on": "standard",
            "claim": wording_claim(false_unseen_standard, false_unseen_adapted, bar)}


def fix_bar(dev_values: Sequence[float], draft: float, direction: str) -> float:
    """The rule written down before the dev run (PROGRESS.md, 2026-10-27): a bar to stay ABOVE (direction "min": macro F1, AUROC) is max(draft, mean - 2 sd) over the dev seeds, a bar
    to stay BELOW (direction "max": false-unseen rates) is min(draft, mean + 2 sd). A dev result can tighten a draft bar and can never make it easier; with fewer than two
    seeds there is no spread and the draft stands."""
    if direction not in ("min", "max"):
        raise ValueError(f"direction must be 'min' (stay above) or 'max' (stay below), got {direction!r}")
    v = np.array([x for x in dev_values if not math.isnan(x)], dtype=np.float64)
    if v.size == 0:
        raise ValueError("no values to fix a bar from")
    if v.size < 2:
        return float(draft)
    mean, sd = float(v.mean()), float(v.std(ddof=1))
    return float(max(draft, mean - 2 * sd)) if direction == "min" else float(min(draft, mean + 2 * sd))


def offline_consequence(errors: Mapping[str, float], costs: Mapping[str, float] = DEFAULT_COSTS) -> float:
    """The measured error counts weighted by the recorded costs (pool points per error): an ESTIMATE, not gated, see ``consequence_text``."""
    unknown = set(errors) - set(costs)
    if unknown:
        raise ValueError(f"unknown error kinds {sorted(unknown)}; the costs are known for {sorted(costs)}")
    if any(v < 0 for v in errors.values()):
        raise ValueError("error counts cannot be negative")
    return float(sum(errors[k] * costs[k] for k in errors))


def consequence_text(errors: Mapping[str, float], costs: Mapping[str, float] = DEFAULT_COSTS) -> str:
    value = offline_consequence(errors, costs)
    return (f"OFFLINE ESTIMATE of the cost of the status errors: {value:.0f} pool points = " + " + ".join(f"{errors[k]:g} {k} x {costs[k]:g}" for k in errors)
            + ". An estimate from recorded costs (E-034, E-030), not measured online and not gated; whether statuses drive learning well belongs to Phase 2.")
