"""The realism set (spec 5.5, Task 13): one-step arithmetic word problems written by people (SVAMP, MIT licence; downloaded once with the user's permission into the local Hugging Face cache),
labelled by the operation of their reference equation. Used only to check that the findings are not artefacts of our templates (B7); never for teaching, calibration or tuning.

A problem is kept when its reference equation is exactly one operation on two numbers taken from the problem, both whole, the answer whole, a difference not negative, a quotient exact. SVAMP has
no remainder or maximum problems, so the set covers add, sub, mul and div. The text of a row is the problem body followed by its question.
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import corpora
import skills

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "realism_set.jsonl"
SPOT = HERE / "results" / "realism_spotcheck.md"
_EQ = re.compile(r"^\(\s*(\d+(?:\.0+)?)\s*([-+*/])\s*(\d+(?:\.0+)?)\s*\)$")
_OP = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
_SOURCE = "ChilleD/SVAMP"


def _snapshot_files() -> List[Path]:
    base = corpora.HUB / "datasets--ChilleD--SVAMP" / "snapshots"
    files = sorted(base.glob("*/train.json")) + sorted(base.glob("*/test.json")) if base.exists() else []
    if not files:
        raise FileNotFoundError(f"dataset {_SOURCE} is not in the local cache {corpora.HUB}; it is downloaded once with the user's permission (huggingface_hub.snapshot_download)")
    return files


def parse_equation(equation: str) -> Optional[Dict]:
    """{"skill", "operands"} if the reference equation is exactly one operation on two whole numbers, else None."""
    m = _EQ.match(str(equation).strip())
    if not m:
        return None
    a, op, b = int(float(m.group(1))), m.group(2), int(float(m.group(3)))
    skill = _OP[op]
    if skill in ("div",) and (b == 0 or a % b != 0):
        return None
    if skill == "sub" and a < b:
        return None
    return {"skill": skill, "operands": [a, b]}


def _rows_from(records: Iterable[dict], source: str) -> List[dict]:
    rows = []
    for index, r in enumerate(records):
        parsed = parse_equation(r["Equation"])
        if parsed is None:
            continue
        a, b = parsed["operands"]
        text = " ".join((str(r["Body"]).strip() + " " + str(r["Question"]).strip()).split())
        if skills.truth(parsed["skill"], a, b) != int(float(r["Answer"])):                      # the stated answer must be what the operation gives
            continue
        rows.append({"text": text, "skill": parsed["skill"], "operands": parsed["operands"], "source": source, "index": index, "id": str(r["ID"])})
    return rows


def all_candidates() -> List[dict]:
    rows: List[dict] = []
    for f in _snapshot_files():
        rows += _rows_from(json.loads(f.read_text(encoding="utf-8")), f"{_SOURCE}/{f.name}")
    seen, out = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


def build_realism_set(n: int = 300, seed: int = 100) -> List[dict]:
    """``n`` distinct one-step problems, the four operations in equal shares as far as the data allows, drawn with ``seed``."""
    pool = all_candidates()
    rng = random.Random(seed)
    by_skill = {s: [r for r in pool if r["skill"] == s] for s in ("add", "sub", "mul", "div")}
    for rows in by_skill.values():
        rng.shuffle(rows)
    out: List[dict] = []
    share = n // len(by_skill)
    for s in by_skill:
        out += by_skill[s][:share]
    leftovers = [r for s in by_skill for r in by_skill[s][share:]]
    rng.shuffle(leftovers)
    out += leftovers[:n - len(out)]
    if len(out) < n:
        raise ValueError(f"only {len(out)} one-step problems are available, {n} asked")
    rng.shuffle(out)
    return out


def spotcheck(rows: List[dict], k: int = 50, seed: int = 100) -> str:
    """Markdown a human can read: k rows with the problem, the label, the equation read back and the answer."""
    picked = random.Random(seed).sample(rows, min(k, len(rows)))
    L = ["# Realism set, spot check", "", f"{len(picked)} of {len(rows)} rows, drawn with seed {seed}. For each row say whether the label (the operation) is the one a reader would use to answer the question. "
         "Mark the row if not.", "", "| # | label | operands | problem | ok? |", "|---|---|---|---|---|"]
    for i, r in enumerate(picked, 1):
        L.append(f"| {i} | {r['skill']} | {r['operands'][0]}, {r['operands'][1]} | {r['text']} | |")
    return "\n".join(L) + "\n"


def assert_not_realism(texts: Iterable[str], rows: Optional[List[dict]] = None) -> None:
    """Teaching, calibration and tuning code calls this with the texts it is about to use: a realism problem among them is a bug."""
    rows = rows if rows is not None else build_realism_set()
    bad = {r["text"] for r in rows} & set(texts)
    if bad:
        raise AssertionError(f"{len(bad)} realism problems found among texts that must not contain any, for example {sorted(bad)[0][:80]!r}")


if __name__ == "__main__":
    rows = build_realism_set()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    SPOT.write_text(spotcheck(rows), encoding="utf-8")
    counts = {s: sum(r["skill"] == s for r in rows) for s in ("add", "sub", "mul", "div")}
    print(f"{len(rows)} rows from {len(all_candidates())} candidates, per operation {counts}; wrote {OUT.name} and {SPOT.name}")
