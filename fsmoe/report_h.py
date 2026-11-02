"""Hybrid report: ``python -m fsmoe.report_h --seeds 710 ...`` reads results/fsmoe/B/H_s*.json (the hybrid arms) and B_s*.json (the E-058 reference arms) and writes results/fsmoe/B/REPORT_H.md (mean [95 % interval over seeds])."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from .domains import NAMES
from .e2e import B_OUT, HYBRID_ARMS
from .report_s import fmt

LABEL = {"A3b": "A3b arrival order (E-058)", "D2": "D2 over: scarce domains replayed (E-058)", "D1r": "D1 sub, re-run with the new counters", "H4": "H4 hybrid, 4 new-class items per chunk",
         "H8": "H8 hybrid, 8 new-class items per chunk", "H12": "H12 hybrid, 12 new-class items per chunk"}
ORDER = ("A3b", "D2", "D1r", "H4", "H8", "H12")


def load(seeds: Sequence[int], out: Path = B_OUT):
    h = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"H_s{s}.json").exists()]
    b = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := out / f"B_s{s}.json").exists()]
    return h, b


def arms(seeds: Sequence[int], out: Path = B_OUT):
    """{arm name: [arm result per seed]} for the six arms of the report (the first two come from the E-058 files)."""
    h, b = load(seeds, out)
    res = {n: [r["arms"][n] for r in b] for n in ("A3b", "D2")}
    res.update({n: [r["arms"][n] for r in h] for n in ("D1r", "H4", "H8", "H12")})
    return res


def build(seeds: Sequence[int], out: Path = B_OUT) -> str:
    h, b = load(seeds, out)
    A = arms(seeds, out)
    L = [f"# Hybrid report (seeds {[r['seed'] for r in h]})", "",
         "The E-057 A3 setting (pool + global head, tool on all code inputs, 228 human labels); only how verified items enter the pool changes. Arms A3b and D2 are the E-058 results of the same seeds; D1r re-runs D1 with counters for new-class items "
         f"and reproduced it in {sum(bool(r['arms']['D1r'].get('reproduced')) for r in h)} of {len(h)} seeds. Never-taught skills: 3 arithmetic, 5 code, 20 language.", "",
         "## Never-taught skills learned and what was trained", "",
         "| arm | learned all 28 | arithmetic (3) | code (5) | language (20) | chunks | fresh items trained: arithmetic | code | language | of which new-class: arithmetic | code | language |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        a = A[name]
        tr = lambda key, d: fmt([x[key].get(d, 0) for x in a], 0) if key in a[0] else "n/a"
        L.append(f"| {LABEL[name]} | {fmt([x['after']['learned']['all'] for x in a], 2)} | " + " | ".join(fmt([x["after"]["learned"][d] for x in a], 2) for d in NAMES)
                 + f" | {fmt([x['chunks'] for x in a], 0) if 'chunks' in a[0] else 'n/a'} | " + " | ".join(tr("trained_by_domain", d) for d in NAMES) + " | " + " | ".join(tr("new_trained_by_domain", d) for d in NAMES) + " |")
    L += ["", "## Known skills after the stream (taught skills in held-out wordings; before the stream in the last row)", "", "| arm | all | arithmetic | code | language | taught wordings, all |", "|---|---|---|---|---|---|"]
    for name in ORDER:
        a = A[name]
        L.append(f"| {LABEL[name]} | " + " | ".join(fmt([x["after"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + f" | {fmt([x['after']['known_tw']['all'] for x in a])} |")
    a = A["A3b"]
    L.append("| before the stream | " + " | ".join(fmt([x["before"]["known_ho"][k] for x in a]) for k in ("all",) + NAMES) + f" | {fmt([x['before']['known_tw']['all'] for x in a])} |")
    L += ["", f"Human labels in every arm: {sorted({x['human_labels'] for n in ORDER for x in A[n]})} (one value per seed in the first row of each seed).", ""]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (B_OUT / "REPORT_H.md").write_text(text, encoding="utf-8")
    print(text)
