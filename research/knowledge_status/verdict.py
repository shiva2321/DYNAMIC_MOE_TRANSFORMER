"""The verdict rules of the pre-registration: measured numbers against the FROZEN bars, statement by statement (spec section 7).

Pure and dependency-free on purpose: the freeze records its sha256 through the code stamp, and ``tests/knowledge_status/test_ks_freeze.py`` flips every comparison in this file in turn to
prove that the tests notice. The bars come in as an argument (the frozen ``bars.json``), never from constants here, so a bar cannot change without the hash changing.

``measured`` maps a measure name to a number (or None / NaN when that measure was not run); ``bars`` maps a measure name to ``{"bar": float, "direction": "min" | "max"}``.
"min" means the measure must stay AT OR ABOVE its bar, "max" AT OR BELOW. A measure that was not run is never a pass.
"""

import math
from typing import Dict, Mapping, Optional

TOL = 1e-12                                          # float slack only (a value exactly on its bar passes)

# measure -> (statement of spec section 7 it belongs to, direction)
MEASURES: Dict[str, tuple] = {
    "b1a": ("B1a", "min"), "b1b": ("B1b", "min"), "b1c": ("B1c", "min"),
    "macro_f1": ("B3a", "min"), "false_unseen_heldout": ("B3a", "max"), "thin_vs_solid_auroc": ("B3a", "min"), "other_as_unseen": ("B3a", "max"), "trap_as_unseen": ("B3a", "max"),
    "b4a_residual_hit": ("B4a", "min"), "b4a_known_not_flagged": ("B4a", "min"),
    "b4b_nothing_flagged": ("B4b", "min"), "b4b_other_labelled": ("B4b", "min"),
    "b5_mul_parent": ("B5", "min"), "b5_div_parent": ("B5", "min"), "b5_decoy_worst": ("B5", "max"),
    "b6_gap": ("B6", "max"), "b7_gap": ("B7", "max"),
}
STATEMENTS = ("B1a", "B1b", "B1c", "B2", "B3a", "B3b", "B4a", "B4b", "B5", "B6", "B7")
REPORTED_ONLY = ("B3b",)                             # the baselines are reported next to every claim; nothing is gated on them
B2_MEASURES = ("r0_b1a", "r0_b1b")                   # the word-matcher control R0 on B1a and B1b; B2 asks for a gap to the B1 bars of at least the "b2_gap" bar on one of them


def _number(x) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    x = float(x)
    return None if math.isnan(x) else x


def meets(value, bar: float, direction: str) -> str:
    """"pass", "fail" or "not_run" for one number against one bar."""
    if direction not in ("min", "max"):
        raise ValueError(f"direction must be 'min' (stay above) or 'max' (stay below), got {direction!r}")
    v = _number(value)
    if v is None:
        return "not_run"
    ok = v >= bar - TOL if direction == "min" else v <= bar + TOL
    return "pass" if ok else "fail"


def _statement(parts) -> str:
    """A statement passes only if every one of its measures passed; any failure fails it even if another measure was not run."""
    if any(p == "fail" for p in parts):
        return "fail"
    if parts and all(p == "pass" for p in parts):
        return "pass"
    return "not_run"


def verdict(measured: Mapping[str, Optional[float]], bars: Mapping[str, Mapping]) -> dict:
    """{"measures": {name: {value, bar, direction, verdict, statement}}, "statements": {id: "pass" | "fail" | "not_run" | "reported"}}."""
    for name in list(MEASURES) + ["b2_gap"]:
        if name not in bars:
            raise KeyError(f"no frozen bar for {name!r}")
        if bars[name]["direction"] not in ("min", "max"):
            raise ValueError(f"direction of {name!r} must be 'min' or 'max', got {bars[name]['direction']!r}")
    rows: Dict[str, dict] = {}
    for name, (statement, direction) in MEASURES.items():
        bar = bars[name]["bar"]
        value = _number(measured.get(name))
        rows[name] = {"value": value, "bar": bar, "direction": bars[name]["direction"], "verdict": meets(value, bar, bars[name]["direction"]), "statement": statement}
    states: Dict[str, str] = {}
    for sid in STATEMENTS:
        if sid in REPORTED_ONLY:
            states[sid] = "reported"
        elif sid != "B2":
            states[sid] = _statement([r["verdict"] for r in rows.values() if r["statement"] == sid])
    states["B2"] = _b2(measured, bars)
    return {"measures": rows, "statements": states}


def _b2(measured: Mapping, bars: Mapping) -> str:
    """Validity of the test bed: R0 falls short of the B1a bar OR the B1b bar (the spec's draft bar if the record carries one, so a bar tightened on dev cannot make this easier) by at least the B2 margin.
    Valid on either; not_run if neither could be measured."""
    margin = bars["b2_gap"]["bar"]
    gaps = []
    for r0, b1 in (("r0_b1a", "b1a"), ("r0_b1b", "b1b")):
        v = _number(measured.get(r0))
        gaps.append(None if v is None else min(bars[b1]["bar"], bars[b1].get("draft", bars[b1]["bar"])) - v)
    if any(g is not None and g >= margin - TOL for g in gaps):
        return "pass"
    if any(g is None for g in gaps):
        return "not_run"
    return "fail"


def consequences(result: Mapping, bars: Mapping[str, Mapping]) -> dict:
    """What the verdicts mean for the rest of the run (the kill rule of the pre-registration):
    - ``invalid``: B2 failed, the test bed is broken (anti-goal A1) and no result of the run counts;
    - ``status_void``: B1b is below the spec's draft bar, the reader cannot see an operator change, so no status claim built on it counts;
    - ``conditional``: B1a or B1c failed (the headline reader was carried on regardless, user decision D-40), so every status claim is reported as conditional on a reader that missed B1."""
    st = result["statements"]
    m = result["measures"]["b1b"]
    draft = bars["b1b"].get("draft", bars["b1b"]["bar"])
    b1b_below_draft = m["value"] is not None and m["value"] < min(draft, bars["b1b"]["bar"]) - TOL
    return {"invalid": st["B2"] == "fail", "status_void": bool(b1b_below_draft), "conditional": st["B1a"] == "fail" or st["B1c"] == "fail"}
