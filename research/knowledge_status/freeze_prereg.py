"""Freeze the pre-registration of the confirmatory run (Task 14). Run it ONCE, after the user's go:

    PYTHONPATH=<repo root> python freeze_prereg.py            # freeze (refuses to run twice)
    PYTHONPATH=<repo root> python freeze_prereg.py --check    # is what was frozen still what is on disk?

It reads the DEV result files only (seeds 100 to 129; E1 on 100 to 104, E4 on its qualifying seeds), applies the bar rule that was written down before the dev run
(``evaluate.fix_bar``: a bar to stay above is max(draft, mean - 2 sd), a bar to stay below is min(draft, mean + 2 sd), so a dev result can tighten a draft bar and never loosen it), and writes
``bars.json`` (everything that is fixed), ``bars.sha256`` (its hash) and ``PREREG.md`` (statements, bars, predictions with expected outcomes, decision rules, kill rule).
Nothing here runs a model, and nothing reads a confirmatory result.
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

import runlib
import verdict as VD

HERE = Path(__file__).resolve().parent
GROW = HERE.parents[1] / "growmoe"
FROZEN_FILES = ("bars.json", "bars.sha256", "PREREG.md")
HEADLINE = "R5_R4_qwen_L26"
REPRESENTATIONS = ["R5_R4_qwen_L26", "R4_qwen_L26", "R0_hash"]
Q, Q_CURVE, GROUP_SIZES = 0.05, [0.01, 0.02, 0.05, 0.10], [1, 4, 8, 16]
DEV_SEEDS, CONFIRM_SEEDS = list(range(100, 130)), list(range(400, 430))
E4_N_PER_KIND = 40                                          # traces of each kind per seed in E4 (the dev run used 40)
B6_PRIMARY = "auto_sentence"                                  # chosen on the dev results (the sentence / clause cutter keeps the B4a numbers closer to the gold-span ones than the window cutter)
CHANCE = {"b1a": 1 / 7, "b1b": 0.5, "b1c": 0.5}

_SLOT = re.compile(r"\{[a-z]+\}")
_TOKEN = re.compile(r"<s>|[a-z]+|[^\sa-z]")


# ------------------------------------------------------------------------------------------------------------------------------------ pool disjointness
def normalise(text: str) -> str:
    """Lower case, every {slot} replaced by <s>, spacing collapsed: two templates that differ only in their slot names or spacing are the same template."""
    return " ".join(_SLOT.sub("<s>", text.lower()).split())


def _templates(pool, family: Optional[str] = None) -> List[str]:
    out: List[str] = []
    for fam, frames in pool.frames.items():
        if family in (None, fam):
            out += [f.text for f in frames]
    for fam, cores in pool.cores.items():
        if family in (None, fam):
            out += [c.text for c in cores]
    if family is None:
        out += list(pool.traps)
    return [normalise(t) for t in out]


def pool_strings(pool) -> List[str]:
    """Every normalised template string of a pool: frames, cores and traps (the chatter and the object words are compared separately)."""
    return _templates(pool)


def _grams(text: str, n: int = 5) -> set:
    toks = ["^"] + _TOKEN.findall(text) + ["$"]                # boundary markers, so a short template can only match an identical short template
    return {tuple(toks[i:i + n]) for i in range(max(1, len(toks) - n + 1))}


def five_gram_overlap(pool, other, family: Optional[str] = None, n: int = 5) -> float:
    """The share of ``pool``'s word 5-grams (templates only, restricted to one family if given) that also occur anywhere in ``other``'s templates."""
    mine = set().union(*(_grams(t, n) for t in _templates(pool, family)))
    theirs = set().union(*(_grams(t, n) for t in _templates(other)))
    return len(mine & theirs) / len(mine) if mine else 0.0


# ------------------------------------------------------------------------------------------------------------------------------------ the hash and the code stamp
def file_sha256(path: Path) -> str:
    """sha256 of a file with CRLF folded to LF (as ``runlib.code_stamp`` does), so a Windows checkout that rewrites line endings does not look like a changed file."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def bars_hash(obj) -> str:
    """sha256 of the canonical JSON of the frozen record (sorted keys, no spaces), so the formatting of bars.json does not matter but every value does."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")).hexdigest()


def _imports_of(path: Path) -> List[str]:
    names: List[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "growmoe":
                names += [a.name for a in node.names]
            elif node.module.startswith("growmoe."):
                names.append(node.module.split(".")[1])
            elif node.level and path.parent.name == "growmoe":
                names.append(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            names += [a.name.split(".")[1] for a in node.names if a.name.startswith("growmoe.")]
    return names


def imported_growmoe_files(files: Sequence[Path], grow_dir: Path = GROW) -> List[Path]:
    """The growmoe source files that ``files`` import, directly or through each other (the closure), sorted."""
    seen: Dict[str, Path] = {}
    todo = [str(n) for f in files for n in _imports_of(Path(f))]
    while todo:
        name = todo.pop()
        path = grow_dir / f"{name}.py"
        if name in seen or not path.exists():
            continue
        seen[name] = path
        todo += _imports_of(path)
    return [seen[k] for k in sorted(seen)]


def default_code_files() -> List[Path]:
    own = sorted(HERE.glob("*.py"))
    return own + imported_growmoe_files(own)


def _label(f: Path) -> str:
    return f"growmoe/{f.name}" if f.parent.name == "growmoe" else f.name


# ------------------------------------------------------------------------------------------------------------------------------------ the dev values
def _stamp(expected, rep: Optional[str] = None) -> str:
    return expected(rep) if callable(expected) else expected


def _load(paths: Sequence[Path], expected: str, what: str) -> List[dict]:
    cells = []
    for p in sorted(paths, key=lambda q: int(re.search(r"s(\d+)\.json$", q.name).group(1))):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("code_stamp") != expected:
            raise ValueError(f"{what}: {p.name} was written by other code (stamp {rec.get('code_stamp')!r}, the code now gives {expected!r}); re-run it before the freeze")
        cells.append(rec)
    return cells


def _path(rec: dict, path: str):
    for key in path.split("/"):
        rec = rec[key]
    return rec


def collect_dev_values(results_dir: Path, expected_stamps: Mapping, headline: str = HEADLINE) -> Dict[str, List[float]]:
    """{measure: [dev value per seed]} from the E1, E2 and E4 dev result files, each checked against the stamp of the code that now exists."""
    results_dir = Path(results_dir)
    out: Dict[str, List[float]] = {}

    def add(name, v):
        if isinstance(v, (int, float)) and not math.isnan(v):
            out.setdefault(name, []).append(float(v))

    e1 = _load(list(results_dir.glob(f"e1/{headline}__s*.json")), _stamp(expected_stamps["e1"], headline), "E1")
    r0 = _load(list(results_dir.glob("e1/R0_hash__s*.json")), _stamp(expected_stamps["e1"], "R0_hash"), "E1 control")
    e2 = _load(list(results_dir.glob(f"e2/{headline}__s*.json")), _stamp(expected_stamps["e2"], headline), "E2")
    e4 = _load(list(results_dir.glob("e4/s*.json")), _stamp(expected_stamps["e4"]), "E4")
    if not (e1 and e2 and e4 and r0):
        raise ValueError(f"no dev results to fix the bars from (E1 {len(e1)}, E1 control {len(r0)}, E2 {len(e2)}, E4 {len(e4)} cells under {results_dir})")
    for c in e1:
        for k in ("b1a", "b1b", "b1c"):
            add(k, c[k])
    for c in r0:
        add("r0_b1a", c["b1a"])
        add("r0_b1b", c["b1b"])
    hq = f"q{Q:g}"
    for c in e2:
        s = _path(c, f"machine/{hq}/standard")
        add("macro_f1", s["macro_f1"]["macro_f1"])
        for k in ("false_unseen_heldout", "thin_vs_solid_auroc", "other_as_unseen", "unseen_recall"):          # unseen_recall has no bar; it is the number the unseen-skill claim is about
            add(k, s[k])
        t = c["tangled"]["machine"]
        for k in ("residual_hit", "known_not_flagged"):
            add(f"b4a_{k}", t["gold"]["b4a"][k])
        add("b4b_nothing_flagged", t["gold"]["b4b"]["nothing_flagged"])
        add("b4b_other_labelled", t["gold"]["b4b"]["other_labelled"])
        gaps = [abs(t["gold"]["b4a"][k] - t[B6_PRIMARY]["b4a"][k]) for k in ("residual_hit", "known_not_flagged")]
        add("b6_gap", max(gaps))
    for c in e4:
        r = c["rates"]
        add("b5_mul_parent", r["mul"]["parent_is_add"])
        add("b5_div_parent", r["div"]["parent_is_sub"])
        add("b5_decoy_worst", max(r[k]["any_parent"] for k in ("mul_wrong_steps", "div_wrong_steps", "max_irrelevant_valid_steps", "mod_unrelated_skill")))
    return out


def drafts() -> Dict[str, tuple]:
    """The draft bars of spec section 7 (name: (bar, direction)); the code constants are read here so that the freeze records what the runners use."""
    import ceiling
    import run_e2
    import run_e3
    d = {k: (v, "min") for k, v in ceiling.BARS.items()}
    d.update(run_e2.BARS)
    d.update({"b5_mul_parent": (run_e3.GENUINE_BAR, "min"), "b5_div_parent": (run_e3.GENUINE_BAR, "min"), "b5_decoy_worst": (run_e3.DECOY_BAR, "max"),
              "b6_gap": (0.10, "max"), "b7_gap": (0.10, "max"), "b2_gap": (0.15, "min")})
    return d


def fix_bars(dev: Mapping[str, Sequence[float]]) -> Dict[str, dict]:
    import evaluate as EV
    rows: Dict[str, dict] = {}
    for name, (draft, direction) in drafts().items():
        v = [x for x in dev.get(name, []) if not math.isnan(x)]
        row = {"draft": draft, "direction": direction, "n_dev_seeds": len(v)}
        if not v:
            row.update(bar=draft, dev_mean=None, dev_sd=None, dev_min=None, dev_max=None, rule="draft stands (no dev value exists for this measure)")
        else:
            row.update(bar=EV.fix_bar(v, draft, direction), dev_mean=float(np.mean(v)), dev_sd=float(np.std(v, ddof=1)) if len(v) > 1 else None, dev_min=float(min(v)), dev_max=float(max(v)),
                       rule=("draft stands (fewer than two dev seeds)" if len(v) < 2 else
                             ("max(draft, mean - 2 sd) over the dev seeds" if direction == "min" else "min(draft, mean + 2 sd) over the dev seeds")))
        rows[name] = row
    return rows


# ------------------------------------------------------------------------------------------------------------------------------------ the statements
STATEMENTS: Dict[str, dict] = {
    "B1a": {"text": "Wording invariance: a linear probe for skill identity (six skills plus none), trained on S1 to S3, is right on the held-out wordings S4 to S8 (chance 1/7).", "measures": ["b1a"]},
    "B1b": {"text": "Near-twin separation: by distance alone, parts of the same skill are closer than a part and its near-twin (same sentence, other operator); AUROC (chance 0.5).", "measures": ["b1b"]},
    "B1c": {"text": "Taught against untaught: the distance to the nearest taught prototype separates parts of taught skills from parts of untaught skills; AUROC (chance 0.5).", "measures": ["b1c"]},
    "B2": {"text": "Validity of the test bed: the word-matcher control R0 falls short of the B1a bar or the B1b bar by at least 0.15; otherwise the test bed is broken and no result counts (anti-goal A1).",
           "measures": ["r0_b1a", "r0_b1b", "b2_gap"]},
    "B3a": {"text": "Statuses of arithmetic parts: macro F1 over Solid / Thin / Unseen; false 'Unseen' on the held-out wordings S4 to S8 of Solid skills at q = 0.05 with calibration on S1 to S3 only "
                    "(the wording claim); Thin separated from Solid by AUROC; Other content called new arithmetic.",
            "measures": ["macro_f1", "false_unseen_heldout", "thin_vs_solid_auroc", "other_as_unseen"]},
    "B3b": {"text": "The baselines of spec 6.2 (vMF conformal, k-nearest neighbour, Mahalanobis, Relative Mahalanobis, classifier confidence, energy, autoencoder descriptor, group test) are reported on the "
                    "same cases with the same calibration, so every claim has its control. Reported, not gated.", "measures": []},
    "B4a": {"text": "A known part plus a new arithmetic part (gold spans): the left-over part is the truly unseen one, and the known part is not flagged.", "measures": ["b4a_residual_hit", "b4a_known_not_flagged"]},
    "B4b": {"text": "An arithmetic part plus Other content (gold spans): nothing is flagged as new arithmetic, and the Other part is labelled as Other.", "measures": ["b4b_nothing_flagged", "b4b_other_labelled"]},
    "B5": {"text": "Relation from worked traces: the parent set of a multiplication trace is exactly {add} and of a division trace exactly {sub}; decoy traces get a false relation rarely.",
           "measures": ["b5_mul_parent", "b5_div_parent", "b5_decoy_worst"]},
    "B6": {"text": "With automatic segmentation (the sentence / clause cutter, chosen on dev) the two B4a numbers stay within 0.10 of the gold-span numbers (the larger of the two gaps).", "measures": ["b6_gap"]},
    "B7": {"text": "Realism: the Solid-against-Unseen result on a real-text set agrees with the template result within 0.10.", "measures": ["b7_gap"]},
}
RULES = {
    "B1a": "pass when the mean over the 30 confirmatory seeds is at least the frozen bar; the headline reader is R5_R4_qwen_L26",
    "B1b": "pass when the mean over the 30 seeds is at least the frozen bar",
    "B1c": "pass when the mean over the 30 seeds is at least the frozen bar",
    "B2": "pass when R0 (mean over the 30 seeds) is below the frozen B1a bar OR the frozen B1b bar by at least the frozen margin; a fail makes the whole run invalid",
    "B3a": "pass only when ALL FOUR measures pass (macro F1 and AUROC at or above their bars, the two false-new rates at or below theirs); gated on the STANDARD calibration; the adapted condition is reported beside it and cannot pass the claim",
    "B3b": "reported only; no gate",
    "B4a": "pass only when both numbers are at or above their bars (gold spans)",
    "B4b": "pass only when both numbers are at or above their bars (gold spans)",
    "B5": "pass only when the multiplication rate and the division rate are at or above their bars AND the worst decoy rate is at or below its bar",
    "B6": "pass when the larger gap between the automatic and the gold-span B4a numbers is at or below the bar",
    "B7": "pass when the gap is at or below the bar; NOT RUN in Phase 1 if no realism set exists (see Not run)",
}
NOTES = {
    "B3a": "EXPECTED TO FAIL on the calibration claim: known skills in wordings that were not taught (above all short stories and noisy text) are called new far more often than the bar allows. "
           "Dev: the reader, not the detector, causes the shift (E-046; results/WORDING_LAB_STAGE1/2.md, FUNCTION_LAB.md, REWRITE_LAB.md); none of the repairs tried reached the bar. What is expected to hold "
           "is the part that finds NEW skills (see the unseen-found prediction below) and the Solid / Thin / Other reporting.",
    "B4a": "Expected to fail for the same reason: a known part written in a new wording is flagged, so the 'known part not flagged' number is low.",
    "B5": "Expected to FAIL on the genuine traces (the dev means are in the table above; my prediction of 0.85 to 0.97, written before the dev run, missed) and to PASS on the decoys. By wording on dev "
          "(results/E4_TABLE.md): verbal and number-word traces work, symbol traces ('4 + 4 = 8') and story traces almost never. Likely cause (not tested): the machine does not call those step texts Solid, "
          "a new-wording problem again. The confirmatory pool writes its S1 requests differently, so the symbol rows may move; this is stated, not predicted. The decoy bar is 0.000 because every dev decoy rate was "
          "exactly 0 (the rule min(draft, mean + 2 sd) leaves no room when there is no spread); a single false relation in the confirmatory run would fail that measure, and the genuine rows are expected to "
          "fail B5 regardless.",
    "B4b": "Expected to fail: the 'nothing flagged' number is low on dev (table above) for the same wording reason, while 'Other part labelled Other' is near its bar.",
    "B6": "Expected to fail: the dev mean of the larger gap (table above) is well over the bar.",
    "B7": "NOT RUN: the realism set of Task 13 could not be built from the local data (GSM8K gives 0 to 11 one-step problems; no other labelled set is cached; nothing is downloaded, NG5). Stated here so "
          "that its absence is part of the registration.",
}
SURPRISE = "A confirmatory mean outside [dev min, dev max] widened by one dev standard deviation is reported as a surprise next to its verdict."


def statement_rows(bars: Mapping[str, dict], dev_means: Mapping[str, Optional[float]]) -> Dict[str, dict]:
    v = VD.verdict(dev_means, {k: {"bar": r["bar"], "direction": r["direction"]} for k, r in bars.items()})
    rows = {}
    for sid, spec in STATEMENTS.items():
        expected = "reported" if sid == "B3b" else v["statements"][sid]
        if sid == "B7":
            expected = "not_run"
        rows[sid] = {"text": spec["text"], "measures": spec["measures"], "rule": RULES[sid], "expected": expected, "note": NOTES.get(sid, "")}
    return rows


# ------------------------------------------------------------------------------------------------------------------------------------ the freeze
def freeze(out_dir: Path = HERE, results_dir: Path = HERE / "results", pool_file: Path = HERE / "pool_confirm.py", code_files: Optional[Sequence[Path]] = None,
           expected_stamps: Optional[Mapping] = None) -> dict:
    """Write bars.json, bars.sha256 and PREREG.md into ``out_dir`` and return the record. Refuses to run when any of the three already exists."""
    out_dir = Path(out_dir)
    present = [n for n in FROZEN_FILES if (out_dir / n).exists()]
    if present:
        raise FileExistsError(f"already frozen ({', '.join(present)} exist in {out_dir}); a freeze is single use. If a power cut left a partial freeze (no bars.sha256), delete the partial files by hand and run again")
    if expected_stamps is None:
        expected_stamps = current_stamps()
    dev = collect_dev_values(results_dir, expected_stamps)
    bars = fix_bars(dev)
    means = {k: float(np.mean(v)) for k, v in dev.items()}
    import assignments
    import run_e2
    from dataclasses import asdict
    code_files = [Path(f) for f in (code_files if code_files is not None else default_code_files())]
    code_files = sorted(code_files, key=lambda f: (f.parent.name == "growmoe", f.name))
    pool_file = Path(pool_file)
    rec = {
        "frozen_for": "Phase 1 confirmatory run, single use", "bars": bars, "statements": statement_rows(bars, means), "thresholds": dict(assignments.THRESHOLDS),
        "e4_n_per_kind": E4_N_PER_KIND, "q": Q, "q_curve": Q_CURVE, "group_sizes": GROUP_SIZES, "sizes": asdict(run_e2.SIZES), "baseline_params": run_e2.BASELINE_PARAMS,
        "representations": REPRESENTATIONS, "headline_representation": HEADLINE, "b6_primary": B6_PRIMARY,
        "seeds": {"confirm": CONFIRM_SEEDS, "dev": DEV_SEEDS}, "pool_confirm_sha256": file_sha256(pool_file),
        "code_stamp": runlib.code_stamp(code_files), "code_files": [_label(f) for f in code_files],
        "dev_cells": {k: len(v) for k, v in sorted(dev.items())}, "dev_extras": {"unseen_recall": means["unseen_recall"]}, "chance": CHANCE, "surprise_rule": SURPRISE,
        "pool_overlap": _pool_overlap() if pool_file.resolve() == (HERE / "pool_confirm.py").resolve() else None,
    }
    rec = json.loads(json.dumps(rec))                                                    # exactly what the file will hold (tuples become lists), so the returned record can be compared with a re-read one
    text = render_prereg(rec)
    sha = bars_hash(rec)
    runlib.atomic_write_text(out_dir / "PREREG.md", text)
    runlib.atomic_write_text(out_dir / "bars.json", json.dumps(rec, sort_keys=True, indent=1, ensure_ascii=True) + "\n")
    runlib.atomic_write_text(out_dir / "bars.sha256", f"{sha}  bars.json\n")
    return rec


def current_stamps() -> dict:
    import run_e1
    import run_e2
    import run_e3
    return {"e1": run_e1.stamp_for, "e2": run_e2.stamp_for, "e4": runlib.code_stamp(run_e3.STAMP_FILES)}


def _pool_overlap() -> dict:
    import pools
    old = os.environ.get("KS_ALLOW_CONFIRM")
    os.environ["KS_ALLOW_CONFIRM"] = "1"
    try:
        pools.load_pool.cache_clear()
        dev, con = pools.load_pool("dev"), pools.load_pool("confirm")
        return {"five_gram_all": five_gram_overlap(con, dev), "five_gram_by_family": {f: five_gram_overlap(con, dev, family=f) for f in sorted(con.frames)},
                "identical_templates": len(set(pool_strings(dev)) & set(pool_strings(con)))}
    finally:
        if old is None:
            os.environ.pop("KS_ALLOW_CONFIRM", None)
        else:
            os.environ["KS_ALLOW_CONFIRM"] = old
        pools.load_pool.cache_clear()
        sys.modules.pop("pool_confirm", None)


def check_frozen(out_dir: Path = HERE, pool_file: Optional[Path] = None, code_files: Optional[Sequence[Path]] = None) -> dict:
    """{"ok": bool, "problem": str | None}: is bars.json still what bars.sha256 says, and (when given) are the pool file and the code still what was frozen?"""
    out_dir = Path(out_dir)
    try:
        rec = json.loads((out_dir / "bars.json").read_text(encoding="utf-8"))
        want = (out_dir / "bars.sha256").read_text(encoding="ascii").split()[0]
    except (OSError, ValueError, IndexError) as e:
        return {"ok": False, "problem": f"cannot read the frozen files: {e}"}
    if bars_hash(rec) != want:
        return {"ok": False, "problem": "bars.json no longer matches bars.sha256"}
    if pool_file is not None and file_sha256(pool_file) != rec["pool_confirm_sha256"]:
        return {"ok": False, "problem": "pool_confirm.py is not the file that was frozen"}
    if code_files is not None and runlib.code_stamp(sorted((Path(f) for f in code_files), key=lambda f: (f.parent.name == "growmoe", f.name))) != rec["code_stamp"]:
        return {"ok": False, "problem": "the code is not what was frozen (code stamp differs)"}
    return {"ok": True, "problem": None}


# ------------------------------------------------------------------------------------------------------------------------------------ PREREG.md
def _num(x, nd=3) -> str:
    return f"{x:.{nd}f}"


def render_prereg(rec: Mapping) -> str:
    L = ["# Pre-registration of the Phase 1 confirmatory run (frozen, single use)", "",
         "Everything in this file and in `bars.json` is fixed BEFORE any confirmatory seed runs. `bars.sha256` is the sha256 of the canonical JSON of `bars.json`; changing any bar, seed, pool file, code file "
         "or constant changes a hash. The confirmatory run is single use: it is run once, whatever it shows, and its result is reported as it comes.", "",
         "## What is frozen", "",
         f"- Seeds: confirmatory {rec['seeds']['confirm'][0]} to {rec['seeds']['confirm'][-1]} ({len(rec['seeds']['confirm'])} seeds, never used before). Dev seeds {rec['seeds']['dev'][0]} to "
         f"{rec['seeds']['dev'][-1]} set the bars.",
         f"- Readers (representations): {', '.join(rec['representations'])}; headline reader `{rec['headline_representation']}` (the frozen Qwen2.5-1.5B layer-26 state with a learned linear map trained on S1 to S3 "
         "teaching data only); the other two are controls.",
         f"- Calibration: held-out S1 to S3 parts only; headline level q = {rec['q']}; the curve over q in {rec['q_curve']} is reported; group-test sizes {rec['group_sizes']}.",
         f"- Taught-state rule (`THRESHOLDS`): {json.dumps(rec['thresholds'], sort_keys=True)}.",
         f"- Test messages: the confirmatory template pool `pool_confirm.py`, sha256 `{rec['pool_confirm_sha256']}`.",
         f"- Code: stamp `{rec['code_stamp']}` over {len(rec['code_files'])} files (all of `research/knowledge_status/*.py` and the `growmoe` files they import); the runners must refuse to run if it differs.",
         f"- Sizes and baseline settings: see `bars.json` (`sizes`, `baseline_params`).",
         "- How many dev seeds stand behind each bar is in the table (E1 ran on dev seeds 100 to 104 only, E4 on the seeds where addition and subtraction are really Solid, the AUROC for Thin against Solid "
         "only where a seed has both); a bar rests on fewer seeds where that column is small.", ""]
    ov = rec.get("pool_overlap")
    if ov:
        L += [f"The confirmatory pool shares no template with the dev pool (identical normalised templates: {ov['identical_templates']}); the share of its word 5-grams found in the dev pool is "
              f"{ov['five_gram_all']:.3f} overall and at most {max(ov['five_gram_by_family'].values()):.3f} in any one family (limit 0.2).", ""]
    L += ["## The statements, their bars and what we expect", "",
          "Bars come from the rule written before the dev run: a bar to stay above is max(draft, dev mean - 2 sd); a bar to stay below is min(draft, dev mean + 2 sd); a dev result can tighten a draft bar "
          "and never loosen it. The expected verdict is what the dev means give against the frozen bars; it is a prediction, not a result.", "",
          "| measure | statement | bar | draft | dev mean (sd) [min - max], seeds | rule that gave the bar |", "|---|---|---|---|---|---|"]
    owner = {m: sid for sid, s in rec["statements"].items() for m in s["measures"]}
    for name, b in rec["bars"].items():
        sd = f" ({_num(b['dev_sd'])})" if b["dev_sd"] is not None else ""
        dev = "no dev value" if b["dev_mean"] is None else f"{_num(b['dev_mean'])}{sd} [{_num(b['dev_min'])} - {_num(b['dev_max'])}], {b['n_dev_seeds']}"
        L.append(f"| {name} ({'at least' if b['direction'] == 'min' else 'at most'}) | {owner.get(name, '')} | {b['bar']:.3f} | {b['draft']:.3f} | {dev} | {b['rule']} |")
    L += ["", "### Statements (spec section 7), decision rule and expected verdict", ""]
    for sid, s in rec["statements"].items():
        L += [f"**{sid}.** {s['text']}", f"- Decision rule: {s['rule']}.", f"- Measures: {', '.join(s['measures']) if s['measures'] else 'none (reported)'}.",
              f"- Expected verdict: **{s['expected'].replace('_', ' ').upper()}**." + (f" {s['note']}" if s["note"] else ""), ""]
    L += ["## Predictions that are not bars", "",
          f"- Finding new skills stays high: the share of never-taught arithmetic parts called unseen is expected near its dev mean ({rec['dev_extras']['unseen_recall']:.3f}); no bar is registered for it.",
          "- The 'Other content called new arithmetic' number is expected to be LOWER than on dev: in the confirmatory pool the S6 chatter sentences come from the same corpora as the taught Other content "
          "(the dev pool used a short hand-written list, a defect of the dev test bed found after the dev numbers were seen). The traps stay hand-written. It is still expected to be above its bar.",
          "- The hashed word-matcher R0 fails B1a and B1b by a wide margin (dev: 0.54 and 0.01), which is what B2 needs.",
          f"- {SURPRISE}", "",
          "## Decision rules", "",
          "1. Each statement is judged from the mean over the confirmatory seeds that have the measure (all 30, except where a seed cannot define it: Thin against Solid needs both kinds of part; the E4 traces "
          "need a seed where addition and subtraction are really Solid) against its frozen bar, by `verdict.py` (flip-tested), nothing else.",
          "2. The wording claim is gated on the STANDARD calibration (S1 to S3 only). The adapted condition (a few S4 to S8 parts in the calibration) is reported, never gated: if only it reaches the bar, "
          "the claim reads 'needs adaptation to new wording'.",
          "3. B3b (baselines) is reported with every claim; nothing is gated on it. Conflict, the offline consequence estimate and the group-test numbers are reported and not gated.",
          "4. No bar, pool, seed, constant or code file is changed after the run; the run is not repeated; a failure is reported with its number.", "",
          "## Kill rule and paper decision", "",
          "- If B2 fails on the confirmatory seeds, the test bed is invalid for this run (anti-goal A1) and none of the other results counts.",
          "- If B1b fails for the headline reader, that reader cannot see an operator change; no status claim built on it counts, and Phase 1 ends with that negative result (spec section 7).",
          "- A statement whose bar is missed is a reported negative result, not a reason to re-run. Single use means exactly once.",
          "- Paper rule (spec section 10): write a paper if the test bed and the information-ceiling analysis (C-a and C-b) are done on fresh seeds and the reading finds nothing that already does them; add "
          "a method claim (C-c status machine, C-d relation from traces) only for statements whose bars passed.", "",
          "## Not run, stated in advance", "",
          "- B7 (realism check): not run; see the note under B7. It is a limit of Phase 1 and will be listed as one.",
          "- Conflict status: reported if it fires on its few cases, not gated (spec 5.6).", "",
          "## Found after the dev numbers were seen (disclosed)", "",
          "- The dev chatter defect described above; fixed in the confirmatory pool before the freeze, so dev and confirmatory numbers on chatter are not comparable.",
          "- The wording finding (none of the repairs tried reached the bar, E-046) and the E4 prediction miss; both are why B3a and B5 are expected to fail. The bars were NOT changed because of them.", ""]
    text = "\n".join(L)
    text.encode("ascii")                                             # the file is plain ASCII; a stray character is a bug here, not in the reader
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        r = check_frozen(HERE, HERE / "pool_confirm.py", default_code_files())
        print(r)
        sys.exit(0 if r["ok"] else 1)
    rec = freeze()
    print(f"frozen: bars.json sha256 {bars_hash(rec)}")
