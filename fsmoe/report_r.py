"""Layer R report: ``python -m fsmoe.report_r --seeds 700`` reads results/fsmoe/R/R_s*_*.json and writes results/fsmoe/R/REPORT_R.md (tables with mean [95 % interval over seeds]).

Declared bars (design section 5): false flag on known skills in held-out wordings <= 0.05 AND far withheld skills and far texts found >= 0.90 AND near-twin withheld skills found >= 0.70.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy import stats

from .recog_lab import FUSIONS, OUT, SIGNALS, SPACES
from .domains import NAMES

BARS = {"false_ho": 0.05, "far": 0.90, "near": 0.70}


def _fmt(vals: Sequence[float]) -> str:
    v = np.asarray([x for x in vals if x == x], dtype=float)
    if len(v) == 0:
        return "n/a"
    if len(v) == 1:
        return f"{v[0]:.3f}"
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return f"{v.mean():.3f} [{v.mean() - h:.3f}, {v.mean() + h:.3f}]"


def load(seeds: Sequence[int], out: Path = OUT) -> Dict[tuple, List[dict]]:
    res: Dict[tuple, List[dict]] = {}
    for dom in NAMES:
        for sp in SPACES:
            runs = [json.loads((out / f"R_s{s}_{dom}_{sp}.json").read_text(encoding="utf-8")) for s in seeds if (out / f"R_s{s}_{dom}_{sp}.json").exists()]
            if runs:
                res[(dom, sp)] = runs
    return res


def _col(runs: List[dict], path: Sequence[str]) -> List[float]:
    out = []
    for r in runs:
        v = r
        for k in path:
            v = v[k]
        out.append(v)
    return out


def _passes(f_ho: float, far: float, near: float) -> bool:
    return f_ho <= BARS["false_ho"] and far >= BARS["far"] and near >= BARS["near"]


def build(seeds: Sequence[int], out: Path = OUT) -> str:
    res = load(seeds, out)
    L = [f"# Layer R report: the recognizer lab (seeds {list(seeds)})", "",
         "False flag = a known skill (taught) in a HELD-OUT wording called new. Found = withheld skills (never taught) called new. Cut = conformal rule at q = 0.05, n = 40 known inputs for `stream`, "
         "the held-out part of the teaching rows for `teach`. Mean [95 % interval over seeds]. Near = withheld skills whose twin is taught; far = withheld skills with no taught near copy "
         "plus far texts (other registers; for code, real standard-library functions).", ""]
    for dom in NAMES:
        for sp in SPACES:
            runs = res.get((dom, sp))
            if not runs:
                continue
            n = runs[0]["n"]
            L += [f"## {dom} / {sp} space (test rows: {n}; far texts {runs[0]['n_far_text']}; seeds {len(runs)})", "",
                  "| signal | cut | false flag known (teaching wording) | false flag known (held-out wording) | near found | far found | far text found | AUROC held-out vs near | AUROC held-out vs far text | MACs/input | bars |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for s in SIGNALS:
                for cal in ("teach", "stream"):
                    g = lambda k: _col(runs, ["signals", s, cal, k])
                    far = [(x + y) / 2 if x == x else y for x, y in zip(g("novel_far"), g("far_text"))]       # far = mean of far withheld skills and far texts (just the texts when none is withheld)
                    ok = np.mean([_passes(a, f, b) for a, f, b in zip(g("known_ho"), far, g("novel_near"))]) if cal == "stream" else float("nan")
                    L.append(f"| {s} | {cal} | {_fmt(g('known_tw'))} | {_fmt(g('known_ho'))} | {_fmt(g('novel_near'))} | {_fmt(g('novel_far'))} | {_fmt(g('far_text'))} | "
                             f"{_fmt(_col(runs, ['signals', s, 'auroc', 'ho_vs_near']))} | {_fmt(_col(runs, ['signals', s, 'auroc', 'ho_vs_far_text']))} | {runs[0]['cost'][s]} | "
                             f"{'' if ok != ok else ('PASS' if ok >= 0.5 else 'fail')} |")
            L += ["", "Fusions (stream cut, each part at q = 0.05):", "", "| fusion | false flag known (teaching) | false flag known (held-out) | near found | far found | far text found | bars |", "|---|---|---|---|---|---|---|"]
            for name in FUSIONS:
                g = lambda k: _col(runs, ["fusion", name, k])
                far = [(x + y) / 2 if x == x else y for x, y in zip(g("novel_far"), g("far_text"))]
                ok = np.mean([_passes(a, f, b) for a, f, b in zip(g("known_ho"), far, g("novel_near"))])
                L.append(f"| {name} | {_fmt(g('known_tw'))} | {_fmt(g('known_ho'))} | {_fmt(g('novel_near'))} | {_fmt(g('novel_far'))} | {_fmt(g('far_text'))} | {'PASS' if ok >= 0.5 else 'fail'} |")
            L += ["", "Group test (mean novelty score of g inputs from one source, cut at q = 0.05 on known groups):", "",
                  "| g | signal | false alarm known (teaching) | false alarm known (held-out) | near found | far found |", "|---|---|---|---|---|---|"]
            for g_ in runs[0]["group"]:
                for s in ("knn", "maha", "disagree"):
                    gg = lambda k: _col(runs, ["group", g_, s, k])
                    L.append(f"| {g_} | {s} | {_fmt(gg('known_tw'))} | {_fmt(gg('known_ho'))} | {_fmt(gg('novel_near'))} | {_fmt(gg('novel_far'))} |")
            L += ["", f"Closed-form + network classifier (the label head) accuracy on known skills: teaching wording {_fmt(_col(runs, ['accuracy', 'known_tw']))}, held-out wording {_fmt(_col(runs, ['accuracy', 'known_ho']))}.", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    a = ap.parse_args()
    text = build(a.seeds)
    (OUT / "REPORT_R.md").write_text(text, encoding="utf-8")
    print(text)
