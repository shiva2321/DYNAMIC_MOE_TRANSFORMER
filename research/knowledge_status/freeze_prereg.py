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
FS_ARMS = ("R5_FS_strict", "R5_FS_pre")                       # the from-scratch reader arms (Task 11b); reported, never gated
E1_ONLY_READERS = ["R1_minilm384", "R2_mpnet", "R3_bert", "R4_gpt2_L8", "R4_qwen_L22"]      # the rest of the ladder: the information ceiling (E1) only, reported, never gated (paper rule C-b)
BASE_REPRESENTATIONS = ["R5_R4_qwen_L26", "R4_qwen_L26", "R0_hash"]      # a from-scratch arm joins the confirmatory run only if the D-40 rule keeps it on dev
Q, Q_CURVE, GROUP_SIZES = 0.05, [0.01, 0.02, 0.05, 0.10], [1, 4, 8, 16]
DEV_SEEDS, CONFIRM_SEEDS = list(range(100, 130)), list(range(400, 430))
MIN_SEEDS = 10                                              # a measure needs at least this many confirmatory seeds to be judged; fewer is "not_run"
E4_N_PER_KIND = 40                                          # traces of each kind per seed in E4 (the dev run used 40)
B6_PRIMARY = "auto_sentence"                                  # chosen on the dev results (the sentence / clause cutter keeps the B4a numbers closer to the gold-span ones than the window cutter)
CHANCE = {"b1a": 1 / 7, "b1b": 0.5, "b1c": 0.5}

_SLOT = re.compile(r"\{[a-z]+\}")
_TOKEN = re.compile(r"<s>|[a-z]+|[^\sa-z]")


# ------------------------------------------------------------------------------------------------------------------------------------ pool disjointness
def normalise(text: str) -> str:
    """Lower case, every {slot} replaced by <s>, spacing collapsed: two templates that differ only in their slot names or spacing are the same template."""
    return " ".join(_SLOT.sub("<s>", text.lower()).replace(chr(10), " <nl> ").split())


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


def _cell_stamp(expected_stamps: Mapping, key: str, rep: str) -> str:
    """The stamp a cell of reader ``rep`` must carry: the from-scratch arms have their own (when the caller supplies one), the others the experiment's."""
    if rep in FS_ARMS and "fs" in expected_stamps:
        return _stamp(expected_stamps["fs"])
    return _stamp(expected_stamps[key], rep)


def _load(paths: Sequence[Path], expected: str, what: str, seeds: Optional[Sequence[int]] = None, pool: Optional[str] = None) -> List[dict]:
    cells = []
    for p in sorted(paths, key=lambda q: int(re.search(r"s(\d+)\.json$", q.name).group(1))):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("code_stamp") != expected:
            raise ValueError(f"{what}: {p.name} was written by other code (stamp {rec.get('code_stamp')!r}, the code now gives {expected!r}); re-run it before the freeze")
        seed = int(re.search(r"s(\d+)\.json$", p.name).group(1))
        if seeds is not None and seed not in seeds:
            raise ValueError(f"{what}: {p.name} is seed {seed}, which is outside the expected seeds {seeds[0]} to {seeds[-1]}")
        if pool is not None and rec.get("pool", pool) != pool:
            raise ValueError(f"{what}: {p.name} was made on pool {rec.get('pool')!r}, expected {pool!r}")
        cells.append(rec)
    return cells


def _path(rec: dict, path: str):
    for key in path.split("/"):
        rec = rec[key]
    return rec


def collect_dev_values(results_dir: Path, expected_stamps: Mapping, headline: str = HEADLINE, seeds: Optional[Sequence[int]] = None, pool: str = "dev",
                       require: Sequence[str] = ("e1", "e2", "e4", "b7")) -> Dict[str, List[float]]:
    """{measure: [value per seed]} from the E1, E2 and E4 result files of one pool (the dev cells at the freeze, the confirmatory cells in the report), each checked against the stamp of the code
    that now exists, against the expected seeds and against the pool it was made on. ``require`` names the experiments that must have at least one cell."""
    results_dir = Path(results_dir)
    seeds = list(seeds) if seeds is not None else (DEV_SEEDS if pool == "dev" else CONFIRM_SEEDS)
    out: Dict[str, List[float]] = {}

    def add(name, v):
        if isinstance(v, (int, float)) and not math.isnan(v):
            out.setdefault(name, []).append(float(v))

    e1 = _load(list(results_dir.glob(f"e1/{headline}__s*.json")), _cell_stamp(expected_stamps, "e1", headline), "E1", seeds, pool)
    r0 = _load(list(results_dir.glob("e1/R0_hash__s*.json")), _stamp(expected_stamps["e1"], "R0_hash"), "E1 control", seeds, pool)
    e2 = _load(list(results_dir.glob(f"e2/{headline}__s*.json")), _cell_stamp(expected_stamps, "e2", headline), "E2", seeds, pool)
    e4 = _load(list(results_dir.glob("e4/s*.json")), _stamp(expected_stamps["e4"]), "E4", seeds, pool)
    b7 = _load(list(results_dir.glob("realism/s*.json")), _stamp(expected_stamps["b7"]), "B7", seeds, pool) if "b7" in expected_stamps else []
    have = {"e1": bool(e1 and r0), "e2": bool(e2), "e4": bool(e4), "b7": bool(b7)}
    if not all(have[k] for k in require):
        raise ValueError(f"no {'dev' if pool == 'dev' else 'confirmatory'} results to read (E1 {len(e1)}, E1 control {len(r0)}, E2 {len(e2)}, E4 {len(e4)} cells under {results_dir})")
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
        for k in ("false_unseen_heldout", "thin_vs_solid_auroc", "other_as_unseen", "trap_as_unseen", "unseen_recall"):          # unseen_recall has no bar; it is the number the unseen-skill claim is about
            add(k, s[k])
        t = c["tangled"]["machine"]
        for k in ("residual_hit", "known_not_flagged"):
            add(f"b4a_{k}", t["gold"]["b4a"][k])
        add("b4b_nothing_flagged", t["gold"]["b4b"]["nothing_flagged"])
        add("b4b_other_labelled", t["gold"]["b4b"]["other_labelled"])
        gaps = [abs(t["gold"]["b4a"][k] - t[B6_PRIMARY]["b4a"][k]) for k in ("residual_hit", "known_not_flagged")]
        add("b6_gap", max(gaps))
    for c in b7:
        if c.get("diff") is not None:
            add("b7_diff", c["diff"])                                  # realism balanced accuracy minus template balanced accuracy, per seed
            if c.get("diff_small") is not None:
                add("b7_diff_small", c["diff_small"])                   # the same on the problems whose operands are both at most 99 (reported, not gated)
            add("b7_gap", abs(c["diff"]))                              # the per-seed size of the gap (only used to fix the bar; the verdict uses |mean diff|)
    for c in e4:
        r = c["rates"]
        add("b5_mul_parent", r["mul"]["parent_is_add"])
        add("b5_div_parent", r["div"]["parent_is_sub"])
        add("b5_decoy_worst", max(r[k]["any_parent"] for k in ("mul_wrong_steps", "div_wrong_steps", "max_irrelevant_valid_steps", "mod_unrelated_skill")))
    return out


def derive_means(values: Mapping[str, Sequence[float]], min_seeds: int = 1) -> Dict[str, float]:
    """The number each measure is judged on: the mean over the seeds that have it (and at least ``min_seeds`` of them). B7 is judged on |mean realism-minus-template difference|."""
    means = {k: float(np.mean(v)) for k, v in values.items() if len(v) >= min_seeds}
    means.pop("b7_gap", None)
    if "b7_diff" in means:
        means["b7_gap"] = abs(means.pop("b7_diff"))
    return means


def drafts() -> Dict[str, tuple]:
    """The draft bars of spec section 7 (name: (bar, direction)); the code constants are read here so that the freeze records what the runners use."""
    import ceiling
    import run_e2
    import run_e3
    d = {k: (v, "min") for k, v in ceiling.BARS.items()}
    d.update(run_e2.BARS)
    d["trap_as_unseen"] = (d["other_as_unseen"][0], "max")                 # Other content includes the look-alike traps (spec 5.3 and 5.4); the code scores them apart, so the bar is read twice
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
                    "(the wording claim; the number measured covers the TAUGHT skills, Solid and Thin, not only Solid as the spec bar reads); Thin separated from Solid by AUROC; Other content called new arithmetic, "
                    "everyday sentences and look-alike traps each against the bar.",
            "measures": ["macro_f1", "false_unseen_heldout", "thin_vs_solid_auroc", "other_as_unseen", "trap_as_unseen"]},
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
    "B2": "pass when R0 (mean over the 30 seeds) is below the spec's B1a bar OR the spec's B1b bar (the draft bar, or the frozen bar if that is lower) by at least the frozen margin; a fail makes the whole run invalid",
    "B3a": "pass only when ALL FIVE measures pass (macro F1 and AUROC at or above their bars, the three false-new rates at or below theirs); gated on the STANDARD calibration; the adapted condition is reported beside it and cannot pass the claim",
    "B3b": "reported only; no gate",
    "B4a": "pass only when both numbers are at or above their bars (gold spans)",
    "B4b": "pass only when both numbers are at or above their bars (gold spans)",
    "B5": "pass only when the multiplication rate and the division rate are at or above their bars AND the worst decoy rate is at or below its bar",
    "B6": "pass when the larger gap between the automatic and the gold-span B4a numbers is at or below the bar",
    "B7": "pass when the gap is at or below the bar; not_run if the realism set could not be scored (see the Realism section)",
}
NOTES = {
    "B3a": "EXPECTED TO FAIL on the calibration claim: known skills in wordings that were not taught (above all short stories and noisy text) are called new far more often than the bar allows. "
           "Dev: the reader, not the detector, causes the shift (E-046; results/WORDING_LAB_STAGE1/2.md, FUNCTION_LAB.md, REWRITE_LAB.md); none of the repairs tried reached the bar. Per measure on dev "
           "(table above): the false-new rate and the traps miss their bars widely, Thin against Solid and macro F1 sit on the bar with a wide spread (a coin flip), the everyday-sentences number is expected to "
           "move (see Predictions). What is expected to hold is the finding of NEW skills (see Predictions), not the reporting of Solid / Thin / Other.",
    "B4a": "Expected to fail for the same reason: a known part written in a new wording is flagged, so the 'known part not flagged' number is low.",
    "B5": "Expected to FAIL on the genuine traces (the dev means are in the table above; my prediction of 0.85 to 0.97, written before the dev run, missed) and to PASS on the decoys. By wording on dev "
          "(results/E4_TABLE.md): verbal and number-word traces work, symbol traces ('4 + 4 = 8') and story traces almost never. Likely cause (not tested): the machine does not call those step texts Solid, "
          "a new-wording problem again. The confirmatory pool writes its S1 requests differently, so the symbol rows may move; this is stated, not predicted. The decoy bar is 0.000 because every dev decoy rate was "
          "exactly 0 (the rule min(draft, mean + 2 sd) leaves no room when there is no spread); a single false relation in the confirmatory run would fail that measure, and the genuine rows are expected to "
          "fail B5 regardless.",
    "B4b": "Expected to fail: the 'nothing flagged' number is low on dev (table above) for the same wording reason, while 'Other part labelled Other' is near its bar.",
    "B6": "Expected to fail: the dev mean of the larger gap (table above) is well over the bar.",
    "B7": "Realism: scored on the one-step problems of the realism set (Task 13); see the Realism section of this file for the set and the rule.",
}
SURPRISE = "A confirmatory mean outside [dev min, dev max] widened by one dev standard deviation is reported as a surprise next to its verdict."


def statement_rows(bars: Mapping[str, dict], dev_means: Mapping[str, Optional[float]]) -> Dict[str, dict]:
    v = VD.verdict(dev_means, {k: {"bar": r["bar"], "direction": r["direction"], "draft": r["draft"]} for k, r in bars.items()})
    rows = {}
    for sid, spec in STATEMENTS.items():
        expected = "reported" if sid == "B3b" else v["statements"][sid]
        rows[sid] = {"text": spec["text"], "measures": spec["measures"], "rule": RULES[sid], "expected": expected, "note": NOTES.get(sid, "")}
    return rows


# ------------------------------------------------------------------------------------------------------------------------------------ the freeze
CONFIRM_ARTIFACTS = (HERE / "CONFIRM_STARTED.json", HERE / "CONFIRM_CLOSED.json", HERE / "results" / "confirm")


def _confirm_artifacts_present(paths: Sequence[Path], competence_dir: Path) -> List[str]:
    found = []
    for q in paths:
        q = Path(q)
        if q.is_file() or (q.is_dir() and any(q.iterdir())):
            found.append(q.name)
    if Path(competence_dir).is_dir() and any(Path(competence_dir).glob("confirm__*")):
        found.append("results/competence/confirm__*")
    return found


def fs_summary(results_dir: Path, expected_stamps: Mapping, arms: Sequence[str] = FS_ARMS, headline: str = HEADLINE) -> Dict[str, dict]:
    """The Task 11b rule (decision D-40) on dev: for each from-scratch arm the largest gap to the headline reader on B1b, B1c, macro F1 and false-new, and what the rule makes of it:
    within 0.05 = kept, within 0.15 = parked (not deleted), further = not worth it now. Arms without dev cells are listed as missing."""
    out: Dict[str, dict] = {}
    base = collect_dev_values(results_dir, expected_stamps, headline, require=("e1", "e2"))
    keys = ("b1b", "b1c", "macro_f1", "false_unseen_heldout")
    lower_is_better = {"false_unseen_heldout"}
    for arm in arms:
        try:
            vals = collect_dev_values(results_dir, expected_stamps, arm, require=("e1", "e2"))
        except ValueError as e:
            if "results to read" not in str(e):                                    # stale or misplaced cells are an error, only absent cells are a status
                raise
            out[arm] = {"status": "no dev cells"}
            continue
        gaps = {k: max(0.0, (float(np.mean(vals[k])) - float(np.mean(base[k]))) if k in lower_is_better else (float(np.mean(base[k])) - float(np.mean(vals[k])))) for k in keys}      # only being WORSE counts
        worst = max(gaps.values())
        out[arm] = {"means": {k: float(np.mean(vals[k])) for k in keys}, "headline_means": {k: float(np.mean(base[k])) for k in keys}, "gaps": gaps, "worst_gap": worst,
                    "found": float(np.mean(vals["unseen_recall"])), "headline_found": float(np.mean(base["unseen_recall"])),
                    "status": "kept" if worst <= 0.05 + 1e-12 else ("parked" if worst <= 0.15 + 1e-12 else "not worth it now")}
    return out


def _realism_counts(realism_file: Optional[Path]) -> Dict[str, int]:
    if realism_file is None or not Path(realism_file).exists():
        return {}
    counts: Dict[str, int] = {}
    for line in Path(realism_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            skill = json.loads(line)["skill"]
            counts[skill] = counts.get(skill, 0) + 1
    return counts


def freeze(out_dir: Path = HERE, results_dir: Path = HERE / "results", pool_file: Path = HERE / "pool_confirm.py", code_files: Optional[Sequence[Path]] = None,
           expected_stamps: Optional[Mapping] = None, confirm_artifacts: Optional[Sequence[Path]] = None, realism_file: Optional[Path] = HERE / "results" / "realism_set.jsonl") -> dict:
    """Write bars.json, bars.sha256 and PREREG.md into ``out_dir`` and return the record. Refuses to run when any of the three already exists, and when any trace of a confirmatory run exists."""
    out_dir = Path(out_dir)
    present = [n for n in FROZEN_FILES if (out_dir / n).exists()]
    if present:
        raise FileExistsError(f"already frozen ({', '.join(present)} exist in {out_dir}); a freeze is single use. If a power cut left a partial freeze (no bars.sha256), delete the partial files by hand and run again")
    traces = _confirm_artifacts_present(CONFIRM_ARTIFACTS if confirm_artifacts is None else confirm_artifacts, Path(results_dir) / "competence")
    if traces:
        raise FileExistsError(f"a confirmatory run has already started ({', '.join(traces)}); nothing may be frozen after that")
    if expected_stamps is None:
        expected_stamps = current_stamps()
    dev = collect_dev_values(results_dir, expected_stamps)
    bars = fix_bars(dev)
    means = derive_means(dev)
    simple = {k: {"bar": r["bar"], "direction": r["direction"], "draft": r["draft"]} for k, r in bars.items()}
    vd = VD.verdict(means, simple)
    for name, row in vd["measures"].items():
        bars[name]["expected"] = row["verdict"]
    bars["b2_gap"]["expected"] = vd["statements"]["B2"]
    fs = fs_summary(results_dir, expected_stamps) if "fs" in expected_stamps else {}
    representations = BASE_REPRESENTATIONS + [a for a in FS_ARMS if fs.get(a, {}).get("status") == "kept"]
    import assignments
    import run_e2
    from dataclasses import asdict
    code_files = [Path(f) for f in (code_files if code_files is not None else default_code_files())]
    code_files = sorted(code_files, key=lambda f: (f.parent.name == "growmoe", f.name))
    pool_file = Path(pool_file)
    real = pool_file.resolve() == (HERE / "pool_confirm.py").resolve()
    rec = {
        "frozen_for": "Phase 1 confirmatory run, single use", "bars": bars, "statements": statement_rows(bars, means), "thresholds": dict(assignments.THRESHOLDS),
        "e4_n_per_kind": E4_N_PER_KIND, "min_seeds": MIN_SEEDS, "q": Q, "q_curve": Q_CURVE, "group_sizes": GROUP_SIZES, "sizes": asdict(run_e2.SIZES), "baseline_params": run_e2.BASELINE_PARAMS,
        "representations": representations, "e1_only_readers": E1_ONLY_READERS, "headline_representation": HEADLINE, "b6_primary": B6_PRIMARY,
        "seeds": {"confirm": CONFIRM_SEEDS, "dev": DEV_SEEDS}, "pool_confirm_sha256": file_sha256(pool_file),
        "realism_set_sha256": file_sha256(realism_file) if realism_file is not None and Path(realism_file).exists() else None,
        "realism_counts": _realism_counts(realism_file),
        "code_stamp": runlib.code_stamp(code_files), "code_files": [_label(f) for f in code_files],
        "dev_cells": {k: len(v) for k, v in sorted(dev.items())},
        "dev_extras": {"unseen_recall": means["unseen_recall"], "r0_b1a": means["r0_b1a"], "r0_b1b": means["r0_b1b"], "b7_diff_small": means.get("b7_diff_small")}, "chance": CHANCE, "surprise_rule": SURPRISE,
        "fs_arms": fs,
        "pool_overlap": _pool_overlap() if real else None, "length_leak": _length_leak() if real else None,
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
    import run_realism
    import scratch_reader
    return {"e1": run_e1.stamp_for, "e2": run_e2.stamp_for, "e4": runlib.code_stamp(run_e3.STAMP_FILES), "b7": runlib.code_stamp(run_realism.STAMP_FILES), "fs": scratch_reader.stamp()}


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


def _length_leak() -> dict:
    import check_length_leak
    import pools
    old = os.environ.get("KS_ALLOW_CONFIRM")
    os.environ["KS_ALLOW_CONFIRM"] = "1"
    try:
        pools.load_pool.cache_clear()
        return {pool: {fam: list(row) for fam, row in check_length_leak.length_leak(pool).items()} for pool in ("dev", "confirm")}
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
    try:
        written = (out_dir / "PREREG.md").read_bytes().replace(b"\r\n", b"\n").decode("ascii")
    except (OSError, UnicodeDecodeError) as e:
        return {"ok": False, "problem": f"cannot read PREREG.md: {e}"}
    if written != render_prereg(rec):
        return {"ok": False, "problem": "PREREG.md is not what bars.json renders to (it was edited by hand)"}
    if pool_file is not None and file_sha256(pool_file) != rec["pool_confirm_sha256"]:
        return {"ok": False, "problem": "pool_confirm.py is not the file that was frozen"}
    if code_files is not None and runlib.code_stamp(sorted((Path(f) for f in code_files), key=lambda f: (f.parent.name == "growmoe", f.name))) != rec["code_stamp"]:
        return {"ok": False, "problem": "the code is not what was frozen (code stamp differs)"}
    return {"ok": True, "problem": None}


# ------------------------------------------------------------------------------------------------------------------------------------ PREREG.md
def _fmt(x, nd=3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_prereg(rec: Mapping) -> str:
    fs = rec.get("fs_arms") or {}
    L = ["# Pre-registration of the Phase 1 confirmatory run (frozen, single use)", "",
         "Everything in this file and in `bars.json` is fixed BEFORE any confirmatory seed runs. `bars.sha256` is the sha256 of the canonical JSON of `bars.json`, and `PREREG.md` must be exactly what "
         "`bars.json` renders to (the runners check both); changing any bar, seed, pool file, code file or constant changes a hash. The confirmatory run is single use: it is run once, through "
         "`run_confirm.py` and nothing else, whatever it shows, and its result is reported as it comes.", "",
         "## What is frozen", "",
         f"- Seeds: confirmatory {rec['seeds']['confirm'][0]} to {rec['seeds']['confirm'][-1]} ({len(rec['seeds']['confirm'])} seeds, never used before). Dev seeds {rec['seeds']['dev'][0]} to "
         f"{rec['seeds']['dev'][-1]} set the bars.",
         f"- Readers (representations): {', '.join(rec['representations'])}. Headline reader `{rec['headline_representation']}` (the frozen Qwen2.5-1.5B layer-26 state with a learned linear map trained on S1 to "
         "S3 teaching data only); `R4_qwen_L26` (the same state without the map) and `R0_hash` (hashed word pieces, the word-matcher control) are controls. The from-scratch arms of Task 11b take part only "
         "if the rule of D-40 keeps them on dev (see below). Verdicts are made on the headline reader; the other readers are reported.",
         f"- Calibration: held-out S1 to S3 parts only; headline level q = {rec['q']}; the curve over q in {rec['q_curve']} is reported; group-test sizes {rec['group_sizes']}.",
         f"- Taught-state rule (`THRESHOLDS`): {json.dumps(rec['thresholds'], sort_keys=True)}.",
         f"- Test messages: the confirmatory template pool `pool_confirm.py`, sha256 `{rec['pool_confirm_sha256']}`.",
         f"- Realism set (B7): `results/realism_set.jsonl` (SVAMP one-step problems), sha256 `{rec.get('realism_set_sha256')}`.",
         f"- Code: stamp `{rec['code_stamp']}` over {len(rec['code_files'])} files (all of `research/knowledge_status/*.py` and the `growmoe` files they import); the runners must refuse to run if it differs.",
         f"- Sizes and baseline settings: see `bars.json` (`sizes`, `baseline_params`). A measure needs at least {rec['min_seeds']} confirmatory seeds that define it, otherwise it is not judged.",
         "- How many dev seeds stand behind each bar is in the table (E1 ran on dev seeds 100 to 104 only; E4 on the seeds where addition and subtraction are really Solid; the AUROC for Thin against Solid "
         "only where a seed has both kinds of part; B7 only where a seed has a Solid and an Unseen skill among add, sub, mul, div); a bar rests on fewer seeds where that column is small.", ""]
    ov = rec.get("pool_overlap")
    if ov:
        L += [f"The confirmatory pool shares no template with the dev pool (identical normalised templates: {ov['identical_templates']}); the share of its word 5-grams found in the dev pool is "
              f"{ov['five_gram_all']:.3f} overall and at most {max(ov['five_gram_by_family'].values()):.3f} in any one family (limit 0.2).", ""]
    ll = rec.get("length_leak")
    if ll:
        L += ["Message length as a cue (anti-goal A1, no model; `check_length_leak.py`): accuracy of guessing the skill from the length of the request / of the whole message alone, inside one wording "
              "family, dev pool against confirmatory pool (chance 0.167, or 0.333 in S5). Some leak is natural; the confirmatory pool is somewhat higher in a few families (S8, S1, S4, S7 on the request length) and the check is within one family only. The real "
              "test of surface cues is B2.", "",
              "| family | dev: request, whole | confirmatory: request, whole |", "|---|---|---|"]
        for fam in sorted(ll["dev"]):
            d, c = ll["dev"][fam], ll["confirm"][fam]
            L.append(f"| {fam} | {d[0]:.3f}, {d[1]:.3f} | {c[0]:.3f}, {c[1]:.3f} |")
        L.append("")
    L += ["## The statements, their bars and what we expect", "",
          "Bars come from the rule written before the dev run: a bar to stay above is max(draft, dev mean - 2 sd); a bar to stay below is min(draft, dev mean + 2 sd); a dev result can tighten a draft bar "
          "and never loosen it. The rule was written for the status measures (E2 and E3); it was applied to the ceiling (E1), the traces (E4) and the realism check (B7) after their dev numbers were "
          "known, which is disclosed below. The expected verdict is what the dev means give against the frozen bars; it is a prediction, not a result.", "",
          "| measure | statement | bar | draft | dev mean (sd) [min - max], seeds | expected | rule that gave the bar |", "|---|---|---|---|---|---|---|"]
    owner = {m: sid for sid, s in rec["statements"].items() for m in s["measures"]}
    names = [m for sid in sorted(rec["statements"]) for m in rec["statements"][sid]["measures"] if m in rec["bars"]]          # a fixed order, whatever order the dict was read back in
    for name in names:
        b = rec["bars"][name]
        sd = f" ({_fmt(b['dev_sd'])})" if b["dev_sd"] is not None else ""
        dev = "no dev value" if b["dev_mean"] is None else f"{_fmt(b['dev_mean'])}{sd} [{_fmt(b['dev_min'])} - {_fmt(b['dev_max'])}], {b['n_dev_seeds']}"
        L.append(f"| {name} ({'at least' if b['direction'] == 'min' else 'at most'}) | {owner.get(name, '')} | {b['bar']:.3f} | {b['draft']:.3f} | {dev} | "
                 f"{b.get('expected', '').replace('_', ' ').upper()} | {b['rule']} |")
    L += ["", "### Statements (spec section 7), decision rule and expected verdict", ""]
    for sid in sorted(rec["statements"]):
        st = rec["statements"][sid]
        L += [f"**{sid}.** {st['text']}", f"- Decision rule: {st['rule']}.", f"- Measures: {', '.join(st['measures']) if st['measures'] else 'none (reported)'}.",
              f"- Expected verdict: **{st['expected'].replace('_', ' ').upper()}**." + (f" {st['note']}" if st["note"] else ""), ""]
    L += ["## Realism check (B7)", "",
          "One-step arithmetic word problems written by people (SVAMP, MIT licence), downloaded once with the user's permission, restricted to problems whose reference equation is one operation on two "
          "numbers of the problem; 300 rows (" + ", ".join(f"{k} {v}" for k, v in sorted((rec.get("realism_counts") or {}).items())) + "; SVAMP has no remainder or maximum problems and only a few "
          "dozen usable multiplication problems), drawn with seed 100 and fixed in `results/realism_set.jsonl`, whose hash the runner checks before it starts. No code other than the realism runner "
          "loads SVAMP, and the runner refuses any realism problem found among teaching, calibration or adaptation texts (`realism.assert_not_realism`). Per seed the same machine judges (a) the template parts of the held-out wordings S4 to S8 and (b) the realism problems; on each the balanced accuracy of the "
          "call 'Unseen or not' is taken over parts whose skill is Solid (should not be called Unseen) and parts whose skill is Unseen (should be); the template side holds the same four operations. B7 is the "
          "absolute value of the mean difference (realism minus templates) over the seeds that have both a Solid and an Unseen skill among add, sub, mul, div. A 50-row spot check of the labels "
          "(`results/realism_spotcheck.md`) was read by an independent reviewer: 48 of 50 clearly right, two doubtful, none wrong.",
          "Two confounds are named, not removed: a realism problem is a whole story with two to five numbers, a template part is one request, and about a quarter of the problems have an operand above 99 "
          "(the template bed uses one or two digits). A row restricted to problems whose operands are both at most 99 is reported next to B7 and not gated. Only about 4 in 10 dev seeds can be scored "
          "(" + str(rec["dev_cells"].get("b7_diff", 0)) + " of the 30 dev seeds), so there is a real chance (about 1 in 10 by a binomial estimate) that fewer than the minimum number of confirmatory seeds "
          "can be scored, in which case B7 is not judged.", ""]
    if fs:
        L += ["## From-scratch reader arms (Task 11b, user decision D-40)", "",
              "No pretrained model. `R5_FS_strict`: a hashed bag of word pieces, a small network trained contrastively on the teaching data only. `R5_FS_pre`: the same network first pre-trained without labels "
              "on everyday sentences from the local corpora, then trained the same way. Operating points were chosen on tuning seeds 150 to 154 (results/FS_TUNING.md). The rule of D-40, applied to the dev "
              "means: within 0.05 of the headline reader on B1b, B1c, macro F1 and false-new = kept; within 0.15 = parked (not deleted) until the user decides the whole system is from scratch; further = "
              "not worth it now. Only a kept arm is run in the confirmatory run (measured and reported, never gated); an arm that is not kept is not run there, and its dev result below is the finding.", "",
              "| arm | status on dev | worst gap | B1b | B1c | macro F1 | false-new | never-taught skills found |", "|---|---|---|---|---|---|---|---|"]
        for arm, f in sorted(fs.items()):
            if "means" in f:
                m = f["means"]
                L.append(f"| {arm} | {f['status']} | {f['worst_gap']:.3f} | {m['b1b']:.3f} | {m['b1c']:.3f} | {m['macro_f1']:.3f} | {m['false_unseen_heldout']:.3f} | {f['found']:.3f} |")
            else:
                L.append(f"| {arm} | {f['status']} | | | | | | |")
        L += ["", "The gap counts only being worse than the headline reader (a better value is no gap). A low false-new rate here is not a fix of the wording problem: these arms call almost nothing "
              "new, as the last column shows (the headline reader finds " + _fmt(next((f["headline_found"] for _, f in sorted(fs.items()) if "headline_found" in f), None)) + ").", ""]
    x = rec["dev_extras"]
    L += ["## Predictions that are not bars", "",
          f"- Finding new skills stays high: the share of never-taught arithmetic parts called unseen is expected near its dev mean ({x['unseen_recall']:.3f}); no bar is registered for it.",
          "- 'Other content called new arithmetic' (everyday sentences) is expected to move DOWN from its dev value: on dev the chatter parts were cut together with the frame words around them and the chatter "
          "list was hand-written; in the confirmatory pool the S6 frames put only whitespace around the chatter sentence, so the parts scored as Other are exactly sentences from the corpora that supply the "
          "taught Other content. It may or may not reach its bar; the traps number is a separate measure and is expected to stay far above its bar.",
          f"- The hashed word-matcher R0 is expected to fail B1a and B1b by a wide margin (dev means {x['r0_b1a']:.3f} and {x['r0_b1b']:.3f} against the draft bars 0.95 and 0.95), which is what B2 needs. "
          "B2 can show that the test bed does not let a word-matcher pass B1; it cannot show that no cue exists in S4 to S8.",
          f"- {SURPRISE}", "",
          "## Decision rules", "",
          "1. Each statement is judged from the mean over the confirmatory seeds that have the measure (all 30, except where a seed cannot define it: Thin against Solid needs both kinds of part; the E4 traces "
          "need a seed where addition and subtraction are really Solid; B7 needs a Solid and an Unseen skill among add, sub, mul, div; fewer than the minimum number of seeds means the measure is not judged) "
          "against its frozen bar, by `verdict.py` (flip-tested), nothing else.",
          "2. The wording claim is gated on the STANDARD calibration (S1 to S3 only). The adapted condition (a few S4 to S8 parts in the calibration) is reported, never gated: if only it reaches the bar, "
          "the claim reads 'needs adaptation to new wording'.",
          "3. B3b (baselines) is reported with every claim; nothing is gated on it. The offline consequence estimate, the group-test numbers, the per-wording B5 rows, the small-operand B7 row and the "
          "information-ceiling readers that are run for E1 only are reported and not gated.",
          "4. No bar, pool, seed, constant or code file is changed after the run; the run is not repeated; a failure is reported with its number. `run_confirm.py` is the only entry point; the development "
          "runners refuse the confirmatory pool.", "",
          "## Kill rules and paper decision (computed by `verdict.consequences`, printed at the top of the result)", "",
          "- B2 fails: the test bed is invalid for this run (anti-goal A1) and none of the other results counts.",
          "- B1b is below the spec's draft bar (or the frozen bar if that is lower) for the headline reader: that reader cannot see an operator change; no status claim built on it counts and Phase 1 ends "
          "with that negative result (spec section 7).",
          "- B1a or B1c fails for the headline reader: the spec says a representation that fails B1 is not used for status. The user decided (D-40) to carry the reader on regardless; therefore every "
          "status claim (B3a to B7) is reported as CONDITIONAL on a reader that missed B1, and a pass of such a claim is not a spec pass. B1a and B1c are expected to fail (dev), so the result is "
          "expected to carry the CONDITIONAL banner.",
          "- A statement whose bar is missed is a reported negative result, not a reason to re-run. Single use means exactly once.",
          "- Paper rule (spec section 10): write a paper if the test bed and the information-ceiling analysis (C-a and C-b) are done on fresh seeds and the reading finds nothing that already does them; add "
          "a method claim (C-c status machine, C-d relation from traces) only for statements whose bars passed.", "",
          "## Not run, stated in advance", "",
          "- The Conflict condition of spec 5.6 (a rule change on a taught skill) was not built; only `flag_conflict` is unit-tested. Nothing about Conflict is reported.",
          "- The information-ceiling ladder beyond the readers run here (the E1-only readers listed above) is dev-only evidence; C-b on fresh seeds covers exactly those readers.",
          "- The spec 5.6 count sweep (one skill taught 5, 20, 100, 500, 2000 examples) is not part of this run.",
          "- A5 (right on templates and wrong on real phrasing) is checked only by B7, on one real-text source and four operations.",
          "- The open THRESHOLDS question: on dev about a quarter of the skills taught as 'Solid' (2000 examples) are not solid by measured accuracy (multiplication, division and remainder are hard for the "
          "small solver); the thresholds stay at the draft values and the status truth is taken from the measured accuracy.", "",
          "## Found after the dev numbers were seen (disclosed)", "",
          "- The dev chatter defect: the dev S6 chatter was a short hand-written list, not drawn from the corpora of the taught Other content, and the dev scorer cut the Other parts together with the frame "
          "words. The confirmatory pool fixes both (corpus sentences, whitespace-only frames). Dev and confirmatory numbers on chatter and on Other content are not comparable.",
          "- The wording finding (none of the repairs tried reached the bar, E-046) and the E4 prediction miss; both are why B3a and B5 are expected to fail. The bars were NOT changed because of them.",
          "- The confirmatory S1 forms were written after the E4 diagnosis; the S1 form '{a} + {b} = ?' is close to the symbol trace step '4 + 4 = 8', and the confirmatory S2 has no 'equals' form that "
          "matches the number-word step. So the per-wording B5 rows may reshuffle (symbol rows up, number-word rows down) for template reasons, not for relation reasons. The trace wordings and the "
          "separators of tangled messages are the dev ones, shared by both pools.",
          "- The bar rule was extended to E1, E4 and B7 after their dev numbers were known; the decoy bar is 0.000 because every dev decoy rate was exactly 0.",
          "- Measure definitions differ from the spec's words in two places: the false-new rate on held-out wordings is measured over TAUGHT skills (Solid and Thin), and 'Other content' is two measures, "
          "everyday sentences and look-alike traps, each against the bar.",
          "- At the part level S6 and S8 parts are verbatim S1 to S3 request templates with held-out numbers (their dev false-new rates sit near the taught-wording rate), and some S5 test parts repeat a "
          "taught problem with its numbers swapped; the 'held-out wordings' S4 to S8 are therefore not all equally held out (see E2_DIAG.txt).", ""]
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
