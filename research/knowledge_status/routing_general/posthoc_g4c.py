"""Post-hoc analysis of lab G4c (not a declared rule; the prediction is in PROGRESS.md, written before the numbers): do the calibrators and the contamination only move the cut along ONE curve of
false flags against found? Reads the cached cells of ``lab_robust.py``; nothing is recomputed.

    python research/knowledge_status/routing_general/posthoc_g4c.py
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import lab_robust as LR  # noqa: E402
import runlib  # noqa: E402

SYSTEMS = ("ask_exemplars", "vmf_conformal", "either_pair")
CONTAMINATED_MODES = {"arithmetic": ("real", "inj10", "inj20")}
NEAR = 0.05                                                                              # prediction (1): single-reader contaminated points within 0.05 of the clean curve


def corrected_flag_rate(n: int, k: int, q: float = 0.05) -> float:
    """The expected share of exchangeable known messages that one reader flags when its cut is the conformal cut of its n scores after the k largest were dropped: those k, plus the positions above
    the conformal rank of the m = n - k that remain, over n + 1. It equals (k+1)/(n+1) while m <= 38 at q = 0.05; above that the rank gives a lower cut. Zero when the sample is too small to flag."""
    m = n - k
    r = math.ceil((m + 1) * (1.0 - q) - 1e-9)
    if m < 1 or r > m:
        return 0.0
    return (k + m - r + 1) / (n + 1)


class Curve:
    """The clean curve: found as a function of the known-flagged rate, through the points given and the corners (0, 0) and (1, 1), averaged where false-flag rates coincide and never falling."""

    def __init__(self, points: Sequence[Tuple[float, float]]) -> None:
        d: Dict[float, List[float]] = {}
        for x, y in list(points) + [(0.0, 0.0), (1.0, 1.0)]:
            d.setdefault(round(float(x), 9), []).append(float(y))
        self.x = np.array(sorted(d))
        self.y = np.maximum.accumulate(np.array([float(np.mean(d[x])) for x in self.x]))

    def __call__(self, x: float) -> float:
        return float(np.interp(min(max(float(x), 0.0), 1.0), self.x, self.y))


def power_lost(curve: Curve, known_flagged: float, found: float) -> float:
    """What the clean curve finds at this false-flag rate minus what the point finds (positive = the point is below the curve)."""
    return curve(known_flagged) - found


def _load(domain: str, out_dir: Path) -> List[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob(f"{domain}_s*.json"))]


def _point(rows: Sequence[dict], mode: str, cal: str, nv, system: str, arm: str) -> Tuple[float, float]:
    return LR._g(rows, mode, cal, nv, system, arm, "known_flagged"), LR._g(rows, mode, cal, nv, system, arm, "unknown_found")


def clean_curve(rows: Sequence[dict], system: str, with_agree: bool) -> Curve:
    pts = []
    for mode in rows[0]["modes"]:
        for cal in LR.CALIBRATORS:
            if cal == "agree" and not with_agree:
                continue
            for nv in rows[0]["modes"][mode]["results"][cal]:
                kf, fd = _point(rows, mode, cal, nv, system, "clean")
                if not (np.isnan(kf) or np.isnan(fd)):
                    pts.append((kf, fd))
    return Curve(pts)


def analyse(per_domain: Dict[str, List[dict]]) -> str:
    L = ["# Lab G4c, post-hoc: one curve of false flags against found? (seeds 150-152; reads the cached cells; the prediction was written before this was computed)", "",
         "A calibrator only chooses a cut. For ONE reader every calibrator and both arms are judged on the same evaluation messages, so all their (known flagged, found) points belong to one curve. "
         "The clean curve is drawn through the clean points of every calibrator, mode and sample size (corners (0,0) and (1,1) added, linear in between). 'Lost at matched false flags' = what the "
         "clean curve finds at the point's known-flagged rate minus what the point finds. For the PAIR the curve is only approximate (the fused score depends on the sample), and agree is left out of it.", ""]
    worst: Dict[str, float] = {}
    for d, rows in per_domain.items():
        curves = {s: clean_curve(rows, s, with_agree=(s != "either_pair")) for s in SYSTEMS}
        L += [f"## {d}", ""]
        for mode in CONTAMINATED_MODES.get(d, ("inj10", "inj20")):
            L += [f"### {d}, mode {mode}, n = {LR.HEADLINE}: contaminated points against the clean curve", "",
                  "| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |", "|---|---|---|---|---|---|---|---|"]
            for cal in LR.CALIBRATORS:
                kf, fd = _point(rows, mode, cal, LR.HEADLINE, "either_pair", "contaminated")
                lost = {s: power_lost(curves[s], *_point(rows, mode, cal, LR.HEADLINE, s, "contaminated")) for s in SYSTEMS}
                for s in ("ask_exemplars", "vmf_conformal"):
                    worst[f"{d}/{mode}/{s}"] = max(worst.get(f"{d}/{mode}/{s}", 0.0), abs(lost[s]))
                matched = (fd / curves["either_pair"](kf)) if curves["either_pair"](kf) > 0 and cal != "agree" else float("nan")
                L.append(f"| {cal} | {LR._f(kf)} | {LR._f(fd)} | {LR._f(lost['either_pair'])} | {LR._f(lost['ask_exemplars'])} | {LR._f(lost['vmf_conformal'])} | "
                         f"{LR._f(LR._test(rows, mode, cal, LR.HEADLINE)['retention'])} | {LR._f(matched)} |")
            L.append("")
        ref = "real" if d == "arithmetic" else "inj10"
        ag = _point(rows, ref, "agree", LR.HEADLINE, "either_pair", "clean")
        tc = clean_curve(rows, "either_pair", with_agree=False)
        agree_share = float(np.nanmean([r["verification"]["reader_agrees"] for r in rows]))
        ask_ag = _point(rows, ref, "agree", LR.HEADLINE, "ask_exemplars", "clean")[0]
        L += [f"agree against the pair's trim curve ({d}, clean): known flagged {LR._f(ag[0])}, found {LR._f(ag[1])}; the curve finds {LR._f(tc(ag[0]))} at that rate, so agree is "
              f"{'ABOVE' if ag[1] > tc(ag[0]) else 'BELOW'} it by {abs(ag[1] - tc(ag[0])):.3f}. The asking reader agrees with the named experts on {agree_share:.3f} of the verified known messages "
              f"(1 minus that: {1 - agree_share:.3f}); under agree the pair flags {LR._f(ag[0])} and the asking reader alone {LR._f(ask_ag)}.", ""]
    text = {d: r for d, r in per_domain.items() if d in LR.TEXT_DOMAINS}
    if text:
        L += ["## The corrected flag-rate formula against the clean text-domain single readers", "", "| n | k | observed (mean of the two readers over the three text domains) | (k+1)/(n+1), declared | corrected formula |", "|---|---|---|---|---|"]
        diffs = []
        for n, ks in ((30, (0, 1, 2, 3)), (40, (0, 1, 2, 3, 6)), (60, (0, 1, 2, 3, 6))):
            for k in ks:
                vals = [LR._g(r, "real", LR.CAL_FOR_K[k], n, s, "clean", "known_flagged") for r in text.values() for s in ("ask_exemplars", "vmf_conformal")]
                vals = [v for v in vals if not np.isnan(v)]
                if vals:
                    o, c = float(np.mean(vals)), corrected_flag_rate(n, k)
                    diffs.append(abs(o - c))
                    L.append(f"| {n} | {k} | {o:.3f} | {(k + 1) / (n + 1):.3f} | {c:.3f} |")
        L += ["", f"Largest difference between observed and the corrected formula: {max(diffs):.3f}.", ""]
    L += ["## Largest distance of a contaminated single-reader point from the clean curve (prediction 1: at most 0.05)", ""]
    L += [f"- {k}: {v:.3f}" + ("  (over 0.05)" if v > NEAR else "") for k, v in worst.items()]
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    per_domain = {d: _load(d, LR.OUT) for d in ("arithmetic", "banking", "agnews", "newsgroups")}
    per_domain = {d: r for d, r in per_domain.items() if r}
    text = analyse(per_domain)
    runlib.atomic_write_text(_KS / "results" / "ROUTING_GENERAL_G4c_posthoc_150_152.md", text)
    print(text)
