"""E4 (relation from worked traces, spec 5.6 and 7): does the machine name the parents of a skill from a worked example, and not name any for a decoy? Dev seeds only; one result file per seed.

    PYTHONPATH=<repo root> python run_e3.py --candidates 100-159 --count 10 --per-kind 40

A seed qualifies only if addition and subtraction are REALLY Solid there (their solvers score at the Solid bar on problems they never saw): "relation needs the parent to be Solid". In a qualifying
seed addition and subtraction are taught about 2,000 examples, one more skill is taught 20, three are never taught. The traces are multiplication (repeated addition) and division (repeated
subtraction) in four wordings, and three kinds of decoy (``traces.py``). Every step is read by the learned-map reader, assessed by the status machine, and checked against the taught solver.
"""

import argparse
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from unittest import mock

import numpy as np

import assignments
import lab_function as LF
import reps
import run_e2 as R
import runlib
import status as ST
import traces as TR

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e4"
STAMP_FILES = R.STAMP_FILES + [HERE / "run_e3.py", HERE / "traces.py", HERE / "lab_function.py", HERE / "function_check.py", R.LEARNED_MAP]
KINDS = (("mul", None), ("div", None), ("mul", "wrong_steps"), ("div", "wrong_steps"), ("max", "irrelevant_valid_steps"), ("mod", "unrelated_skill"))
GENUINE_BAR, DECOY_BAR = 0.90, 0.10


def kind_name(skill: str, decoy: Optional[str]) -> str:
    return skill if decoy is None else f"{skill}_{decoy}"


def forced_assignment(rng: np.random.Generator, _orig=assignments.draw_assignment, **kw):
    """The ordinary draw with addition and subtraction forced Solid (the traces need them as parents)."""
    return _orig(rng, need_solid=frozenset({"add", "sub"}), **kw)


@dataclass
class TraceWorld(R.World):
    extra_texts: List[str] = field(default_factory=list)

    def all_texts(self) -> List[str]:
        return list(dict.fromkeys(super().all_texts() + self.extra_texts))


def qualifying_seeds(candidates: Sequence[int], count: int, pool: str = "dev", competence_fn: Optional[Callable] = None, sizes: R.Sizes = R.SIZES) -> List[int]:
    """The first ``count`` candidate seeds in which addition AND subtraction are Solid by their measured accuracy at the taught count."""
    if count < 1:
        raise ValueError("count must be at least 1")
    competence_fn = competence_fn or R.default_competence
    n, out = sizes.count_of("solid"), []
    for seed in candidates:
        ok = True
        for skill in ("add", "sub"):
            row = competence_fn(pool, skill, n, seed)
            ok = ok and assignments.status_truth(n, row["accuracy"], row["chance"]) == "solid"
        if ok:
            out.append(int(seed))
            if len(out) == count:
                break
    return out


def trace_set(seed: int, n_per_kind: int = 40) -> List[TR.Trace]:
    """``n_per_kind`` traces of each kind in ``KINDS``, the wordings taken in turn."""
    rng = R._stream(seed, 20)
    out: List[TR.Trace] = []
    for skill, decoy in KINDS:
        for i in range(n_per_kind):
            out.append(TR.make_trace(rng, skill, i % TR.N_VARIANTS, decoy))
    return out


def build_trace_world(seed: int, pool: str = "dev", sizes: R.Sizes = R.SIZES, competence_fn: Optional[Callable] = None, n_per_kind: int = 40) -> TraceWorld:
    with mock.patch.object(R.assignments, "draw_assignment", forced_assignment):
        world = R.build_world(seed, pool, sizes, competence_fn)
    extra = [st.text for t in trace_set(seed, n_per_kind) for st in t.steps]
    return TraceWorld(**{f.name: getattr(world, f.name) for f in fields(R.World)}, extra_texts=extra)


def run_seed(seed: int, out_dir, stamp: str, sizes: R.Sizes = R.SIZES, rep: Optional[reps.Representation] = None, competence_fn: Optional[Callable] = None, cache_dir=R.CACHE,
             n_per_kind: int = 40, solvers_fn: Optional[Callable] = None, epochs: int = 1000) -> dict:
    path = Path(out_dir) / f"s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world = build_trace_world(seed, "dev", sizes, competence_fn, n_per_kind)
    P, index = R.prepare(world, "R5_R4_qwen_L26", rep, cache_dir)

    def V(texts):
        return P[[index[t] for t in texts]]

    groups = {s: ST.SkillGroup(s, V([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in world.taught}
    other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
    cal_vec, cal_grp, _, _ = R.calibration_arrays(world, V)
    machine = ST.StatusMachine(groups, other, cal_vec, cal_grp, q=R.HEADLINE_Q, seed=seed)
    solvers = solvers_fn(world)[0] if solvers_fn else LF.train_solvers(world, epochs=epochs)[0]
    cache: Dict[str, ST.Assessment] = {}

    def assess(text: str) -> ST.Assessment:
        if text not in cache:
            cache[text] = machine.assess(P[index[text]])
        return cache[text]

    rows: Dict[str, List[dict]] = {kind_name(s, d): [] for s, d in KINDS}
    for t in trace_set(seed, n_per_kind):
        v = TR.verify_trace(t, assess, solvers)
        right = {"mul": "add", "div": "sub"}.get(t.header_skill)
        steps_ok = [assess(st.text) for st in t.steps]
        rows[kind_name(t.header_skill, t.decoy)].append({"parent_is_right": bool(v.parents == frozenset({right})) if right and t.decoy is None else None, "any_parent": bool(v.parents),
                                                         "chain_valid": bool(v.chain_valid), "verified_fraction": v.verified_steps / max(len(t.steps), 1),
                                                         "steps_assessed_solid": float(np.mean([a.status == "solid" for a in steps_ok])), "variant": TR_variant(t)})
    rates: Dict[str, dict] = {}
    for kind, rs in rows.items():
        r = {"n": len(rs), "any_parent": float(np.mean([x["any_parent"] for x in rs])), "chain_valid": float(np.mean([x["chain_valid"] for x in rs])),
             "verified_fraction": float(np.mean([x["verified_fraction"] for x in rs])), "steps_assessed_solid": float(np.mean([x["steps_assessed_solid"] for x in rs]))}
        if kind in ("mul", "div"):
            r["parent_is_add" if kind == "mul" else "parent_is_sub"] = float(np.mean([x["parent_is_right"] for x in rs]))
            r["by_variant"] = {str(v): float(np.mean([x["parent_is_right"] for x in rs if x["variant"] == v])) for v in range(TR.N_VARIANTS)}
        rates[kind] = r
    rec = {"seed": seed, "code_stamp": stamp, "assignment": world.assignment, "competence": world.comp, "rates": rates}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(rec))
    return rec


def TR_variant(trace: TR.Trace) -> int:
    """The wording of a trace, read back from the text (the four wordings differ in their opening)."""
    t = trace.text
    return 1 if t.startswith("Work out") else 3 if (". Counting:" in t) else 2 if not any(ch.isdigit() for ch in t) else 0


def report_e4(rows: List[dict], candidates: int, label: str = "learned-map reader") -> str:
    mean = lambda kind, key: float(np.nanmean([r["rates"][kind][key] for r in rows]))
    L = [f"# E4, relation from worked traces ({label}; {len(rows)} of {candidates} candidate seeds had addition and subtraction really Solid)", "",
         "B5: the parent set of a multiplication trace is exactly {add}, of a division trace exactly {sub}, in at least 0.90 of the traces; a decoy gets a false relation in at most 0.10.", "",
         "| trace | seeds | traces per seed | parent exactly right | any parent named | chain valid | steps verified | steps the machine calls Solid | bar |", "|---|---|---|---|---|---|---|---|---|"]
    for kind in ("mul", "div"):
        key = "parent_is_add" if kind == "mul" else "parent_is_sub"
        v = mean(kind, key)
        L.append(f"| {kind} (genuine) | {len(rows)} | {rows[0]['rates'][kind]['n']} | {v:.3f} | {mean(kind, 'any_parent'):.3f} | {mean(kind, 'chain_valid'):.3f} | {mean(kind, 'verified_fraction'):.3f} | "
                 f"{mean(kind, 'steps_assessed_solid'):.3f} | at least {GENUINE_BAR:.2f}: {'PASS' if v >= GENUINE_BAR else 'fail'} |")
    for skill, decoy in KINDS[2:]:
        kind = kind_name(skill, decoy)
        v = mean(kind, "any_parent")
        L.append(f"| {skill}, decoy {decoy} | {len(rows)} | {rows[0]['rates'][kind]['n']} | n/a | {v:.3f} | {mean(kind, 'chain_valid'):.3f} | {mean(kind, 'verified_fraction'):.3f} | "
                 f"{mean(kind, 'steps_assessed_solid'):.3f} | at most {DECOY_BAR:.2f}: {'PASS' if v <= DECOY_BAR else 'fail'} |")
    L += ["", "Genuine traces by wording (0 symbols, 1 verbal, 2 number words, 3 story), share with the parent exactly right:"]
    for kind in ("mul", "div"):
        L.append(f"- {kind}: " + ", ".join(f"{v}: {np.nanmean([r['rates'][kind]['by_variant'][str(v)] for r in rows]):.3f}" for v in range(TR.N_VARIANTS)))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="100-159")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--per-kind", type=int, default=40)
    args = ap.parse_args()
    runlib.set_threads(3)
    cands = R.seed_range(args.candidates)
    seeds = qualifying_seeds(cands, args.count)
    print(f"qualifying seeds ({len(seeds)} of the first candidates): {seeds}", flush=True)
    stamp = runlib.code_stamp(STAMP_FILES)
    rows = []
    for seed in seeds:
        r = run_seed(seed, OUT, stamp, n_per_kind=args.per_kind)
        rows.append(r)
        print(f"seed {seed}: mul parent right {r['rates']['mul']['parent_is_add']:.3f}  div parent right {r['rates']['div']['parent_is_sub']:.3f}  "
              f"decoys {[round(r['rates'][kind_name(s, d)]['any_parent'], 3) for s, d in KINDS[2:]]}", flush=True)
    used = max(seeds) - min(cands) + 1 if seeds else 0
    text = report_e4(rows, used)
    runlib.atomic_write_text(HERE / "results" / "E4_TABLE.md", text)
    print(text)
