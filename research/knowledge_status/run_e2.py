"""E2 (status of single parts) and E3 (tangled messages) on the dev pool: one small result file per (reader, seed), safe to run again after a power cut, then results/E2_TABLE.md.

    python run_e2.py --pool dev --seeds 100-129 --reps R5_R4_qwen_L26,R4_qwen_L26,R0_hash

A SEED is one teaching assignment (spec 5.6): two skills taught 2,000 examples (Solid), one taught 20 (Thin), three never taught (Unseen), drawn at random. The unit everywhere is a PART
(the request itself, the gold span; spec 5.4): teaching, calibration and tests all use the part text. For every reader (``R5_<id>`` is the learned map on top of ``<id>``, ``R4_qwen_L26`` and
``R0_hash`` are the plain reader and the word-matcher control) the cell builds one world (the texts do not depend on the reader), reads every text, centres the vectors on the taught
examples, and runs the status machine at q = 0.01, 0.02, 0.05, 0.10 in two conditions, STANDARD (calibration on held-out S1 to S3 data of the taught groups) and ADAPTED (the calibration also
holds a few S4 to S8 examples; reported apart, never gated), and the seven baselines of baselines.py at the same calibration. Group tests of 1, 4, 8, 16 parts, the tangled messages
(gold parts, sentence/clause pieces, sliding windows) and the near-twins are scored too.
"""

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

import assignments
import baselines as BL
import ceiling
import corpora
import evaluate as EV
import generator as G
import reps
import run_competence
import runlib
import segment as SG
import skills
import status as ST

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "e2"
CACHE = HERE / "_cache" / "reps"
GROW = HERE.parents[1] / "growmoe"
STAMP_FILES = [HERE / f for f in ("run_e2.py", "status.py", "baselines.py", "evaluate.py", "segment.py", "ceiling.py", "reps.py", "assignments.py", "generator.py", "corpora.py", "pools.py",
                                 "pool_dev.py", "skills.py", "runlib.py", "run_competence.py")] + [GROW / f for f in ("arith.py", "frontend.py", "signature.py", "settest.py", "conformal.py", "vmf.py")]
LEARNED_MAP = HERE / "learned_map.py"
HEADLINE_Q = 0.05
# draft bars of spec section 7 (name: (bar, direction)); dev may tighten them by evaluate.fix_bar, never loosen them; the freeze records the result
BARS: Dict[str, Tuple[float, str]] = {
    "macro_f1": (0.85, "min"), "false_unseen_heldout": (0.05, "max"), "thin_vs_solid_auroc": (0.85, "min"), "other_as_unseen": (0.05, "max"),
    "b4a_residual_hit": (0.85, "min"), "b4a_known_not_flagged": (0.95, "min"), "b4b_nothing_flagged": (0.95, "min"), "b4b_other_labelled": (0.95, "min"),
}
# per-baseline settings chosen on seeds 150 to 154 (results/BASELINE_TUNING.md, tune_baselines.py) before the report run; an empty dict keeps the default
BASELINE_PARAMS: Dict[str, dict] = {"knn_distance": {"k": 1}, "vmf_conformal": {"max_protos": 64}, "mahalanobis": {}, "relative_mahalanobis": {}, "classifier_confidence": {},
                                    "energy": {"C": 10.0}, "autoencoder_descriptor": {"hidden": 64, "epochs": 100}}


@dataclass(frozen=True)
class Sizes:
    counts: Tuple[Tuple[str, int], ...] = (("solid", 2000), ("thin", 20))       # examples taught to a Solid and to a Thin skill (spec 5.6)
    other_teach: int = 600                  # Other-content examples taught (two thirds everyday sentences, one third traps)
    cal: int = 120                          # held-out calibration parts per group (99 are needed for q = 0.01)
    adapt_per_family: int = 6               # S4 to S8 examples per taught skill and family in the adapted condition
    test_heldout: int = 40                  # held-out wording (S4 to S8) test parts per skill
    test_taught: int = 30                   # fresh taught-wording (S1 to S3) test parts per skill
    twins: int = 60
    traps: int = 60
    others: int = 100
    unseen_content: int = 60
    tangled: int = 40                       # messages of each of the three kinds
    batches: int = 20                       # group-test batches per size and kind
    batch_sizes: Tuple[int, ...] = (1, 4, 8, 16)
    background: int = 800                   # general sentences for Relative Mahalanobis
    map_cap: int = 300                      # teaching parts per taught skill the learned map is fitted on
    qs: Tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)

    def count_of(self, state: str) -> int:
        return dict(self.counts)[state]


SIZES = Sizes()


def seed_range(text: str) -> List[int]:
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def stamp_for(rep_id: str) -> str:
    return runlib.code_stamp(STAMP_FILES + ([LEARNED_MAP] if rep_id.startswith("R5_") else []))


def fingerprint(text: str) -> str:
    """The fingerprint ``assignments.text_keys`` uses (case and spacing ignored), to prove that no text is shared between teaching, calibration and test."""
    return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8", "surrogatepass")).hexdigest()


def part_status(part: G.Part, truth: Mapping[str, str]) -> str:
    return EV.gold_status(part, truth)


def segment_sets(msg: G.Message) -> Dict[str, List[Tuple[int, int]]]:
    return {"gold": SG.gold_parts(msg), "auto_sentence": SG.sentence_clause_parts(msg.text), "auto_window": SG.window_parts(msg.text)}


def default_competence(pool: str, skill: str, count: int, seed: int) -> dict:
    """The held-out accuracy of the skill's solver at its taught count: one cached file per (skill, count, seed), see run_competence.py."""
    return run_competence.run_cell(pool, skill, count, seed, runlib.code_stamp(run_competence.STAMP_FILES))


# ---------------------------------------------------------------------------------------------------------------------------------- the world
@dataclass
class World:
    seed: int
    pool: str
    sizes: Sizes
    assignment: Dict[str, str]
    comp: Dict[str, dict]
    truth: Dict[str, str]
    teach: Dict[str, List[assignments.Example]]
    teach_none: List[ceiling.Item]
    cal: Dict[str, List[str]]
    adapt: Dict[str, List[str]]
    adapt_families: List[str]
    test_texts: List[str]
    test_cases: List[EV.Case]
    tangled: List[G.Message]
    background: List[str]
    taught: List[str] = field(default_factory=list)

    def all_texts(self) -> List[str]:
        seq = [e.span for s in self.teach for e in self.teach[s]] + [i.text for i in self.teach_none]
        for group in list(self.cal.values()) + list(self.adapt.values()):
            seq += group
        seq += self.test_texts + self.background
        for m in self.tangled:
            for spans in segment_sets(m).values():
                seq += [m.text[a:b] for a, b in spans]
        return list(dict.fromkeys(seq))


def _stream(seed: int, *key: int) -> np.random.Generator:
    return np.random.default_rng([seed, 0xE2, *key])


def _fresh_taught_wording(rng, pool: str, skill: str, n: int, used: set) -> List[assignments.Example]:
    """``n`` new S1 to S3 examples whose texts (case and spacing ignored) are in nothing seen so far; ``used`` grows with them."""
    out: List[assignments.Example] = []
    for _ in range(60):
        for e in assignments.draw_teaching(rng, pool, skill, max(2 * (n - len(out)), 8)):
            k = fingerprint(e.text)
            if k not in used:
                used.add(k)
                out.append(e)
                if len(out) == n:
                    return out
    raise RuntimeError(f"could not draw {n} fresh taught-wording examples of {skill}")


def _chatter(e: assignments.Example) -> List[str]:
    """The sentences around the request in an S6 message (they are parts of their own: Other content)."""
    i = e.text.find(e.span)
    if e.family != "S6" or i < 0:
        return []
    return [p for p in (e.text[:i].strip(), e.text[i + len(e.span):].strip()) if p]


def build_world(seed: int, pool: str = "dev", sizes: Sizes = SIZES, competence_fn: Optional[Callable] = None) -> World:
    """Everything a cell needs that does not depend on the reader: the assignment and the measured competence, the teaching parts of the taught skills, the Other-content teaching,
    calibration and adaptation parts, the test parts with their truth, the tangled messages and the background sentences. Deterministic in (seed, pool, sizes)."""
    competence_fn = competence_fn or default_competence
    assignment = assignments.draw_assignment(_stream(seed, 1))
    taught = [s for s in skills.SKILLS if assignment[s] != "unseen"]
    comp: Dict[str, dict] = {}
    truth: Dict[str, str] = {}
    for s in skills.SKILLS:
        if assignment[s] == "unseen":
            truth[s] = "unseen"
            continue
        count = sizes.count_of(assignment[s])
        row = competence_fn(pool, s, count, seed)
        truth[s] = assignments.status_truth(count, row["accuracy"], row["chance"])
        comp[s] = {"state": assignment[s], "count": count, "accuracy": float(row["accuracy"]), "chance": float(row["chance"]), "status": truth[s]}
    teach = {s: assignments.draw_teaching(_stream(seed, 2, skills.SKILLS.index(s)), pool, s, comp[s]["count"]) for s in taught}
    used = {fingerprint(e.text) for s in teach for e in teach[s]}
    none_texts: set = set()
    rng = _stream(seed, 3)
    n_trap = sizes.other_teach // 3
    teach_none = ceiling._draw_none(rng, pool, sizes.other_teach - n_trap, "other", None, none_texts, "teach") + ceiling._draw_none(rng, pool, n_trap, "trap", 0, none_texts, "teach")
    # held-out calibration: fresh parts of the same kind as the teaching, S1 to S3 only (and Other content in the same mixture as its teaching)
    cal = {s: [e.span for e in _fresh_taught_wording(_stream(seed, 4, skills.SKILLS.index(s)), pool, s, sizes.cal, used)] for s in taught}
    cal_trap = max(1, sizes.cal // 3)
    cal["other"] = [i.text for i in ceiling._draw_none(rng, pool, sizes.cal - cal_trap, "other", None, none_texts, "cal") + ceiling._draw_none(rng, pool, cal_trap, "trap", 0, none_texts, "cal")]
    # adaptation: a few examples of the held-out wordings, for the separately reported condition only
    adapt: Dict[str, List[str]] = {}
    adapt_families: List[str] = []
    for s in taught:
        ex = assignments.draw_test(_stream(seed, 5, skills.SKILLS.index(s)), pool, s, sizes.adapt_per_family * len(assignments.TEST_FAMILIES), frozenset(used))
        adapt[s] = [e.span for e in ex]
        adapt_families += [e.family for e in ex]
        used |= {fingerprint(e.text) for e in ex}
    # tests
    texts: List[str] = []
    cases: List[EV.Case] = []
    for s in skills.SKILLS:
        for e in assignments.draw_test(_stream(seed, 6, skills.SKILLS.index(s)), pool, s, sizes.test_heldout, frozenset(used)):
            texts.append(e.span)
            cases.append(EV.Case(truth[s], e.family, s, "arith"))
            for piece in _chatter(e):
                texts.append(piece)
                cases.append(EV.Case("other", "S6", None, "other"))
        if sizes.test_taught:
            for e in _fresh_taught_wording(_stream(seed, 7, skills.SKILLS.index(s)), pool, s, sizes.test_taught, used):
                texts.append(e.span)
                cases.append(EV.Case(truth[s], e.family, s, "arith"))
    rng = _stream(seed, 8)
    made = 0
    while made < sizes.twins:
        skill, family = ceiling.INFIX[int(rng.integers(0, len(ceiling.INFIX)))], ("S1", "S2")[int(rng.integers(0, 2))]
        anchor = G.make_single(rng, pool, skill, family)
        to = [s for s in ceiling.INFIX if s != skill][int(rng.integers(0, len(ceiling.INFIX) - 1))]
        try:
            twin = G.make_near_twin(rng, anchor, to)
        except ValueError:
            continue
        p = next(x for x in twin.parts if x.kind == "skill")
        span = twin.text[p.start:p.end]
        if fingerprint(span) in used:
            continue
        texts.append(span)
        cases.append(EV.Case(truth[to], family, to, "twin"))
        made += 1
    for kind, n, parity, label, status in (("trap", sizes.traps, 1, "trap", "other"), ("other", sizes.others, None, "other", "other"), ("unseen", sizes.unseen_content, None, "unseen_content", "unseen")):
        for item in ceiling._draw_none(_stream(seed, 9, len(cases)), pool, n, kind, parity, none_texts, "test"):
            texts.append(item.text)
            cases.append(EV.Case(status, item.family if kind != "other" else "other", None, label))
    # tangled messages: a known skill with an unseen one (B4a), a known skill with Other content (B4b), an unseen skill with Other content (B4c)
    known = [s for s in taught if truth[s] in ("solid", "thin")]
    unseen = [s for s in skills.SKILLS if assignment[s] == "unseen"]
    if not known or not unseen:
        raise ValueError("tangled messages need a taught skill with a measurable status and an unseen skill")
    tangled: List[G.Message] = []
    rng = _stream(seed, 10)
    for kind in ("b4a", "b4b", "b4c"):
        made = attempts = 0
        while made < sizes.tangled:
            attempts += 1
            if attempts > 300 * sizes.tangled:
                raise RuntimeError(f"could not build {sizes.tangled} {kind} messages")
            k, u = known[int(rng.integers(0, len(known)))], unseen[int(rng.integers(0, len(unseen)))]
            flip = bool(rng.integers(0, 2))
            if kind == "b4a":
                kinds, labels = ("skill", "skill"), ((k, u) if not flip else (u, k))
            else:
                skill = k if kind == "b4b" else u
                kinds, labels = (("skill", "other"), (skill, None)) if not flip else (("other", "skill"), (None, skill))
            m = G.make_tangled(rng, pool, kinds, labels)
            if any(fingerprint(m.text[p.start:p.end]) in used for p in m.parts if p.kind == "skill"):
                continue
            tangled.append(m)
            made += 1
    seen_other = {t for i in teach_none for t in (i.text,)} | none_texts
    background = [t for t in corpora.load_other_sentences(2 * sizes.background + len(seen_other), seed) if t not in seen_other][:sizes.background]
    return World(seed, pool, sizes, assignment, comp, truth, teach, teach_none, cal, adapt, adapt_families, texts, cases, tangled, background, taught)


# ---------------------------------------------------------------------------------------------------------------------------------- one cell
def _native(o):
    if isinstance(o, dict):
        return {str(k): _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, np.ndarray):
        return _native(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def _reader(world: World, X: np.ndarray, index: Dict[str, int], seed: int, use_map: bool, mapper: Optional[Callable]) -> np.ndarray:
    """The vectors of every text after the reader: the learned map (R5 only; fitted on the teaching parts of the taught skills plus the none-messages, never on anything else), then
    centring on the mean of ALL teaching vectors (the primary form of the earlier measurements, Task 8) and unit length."""
    teach_texts = [e.span for s in world.taught for e in world.teach[s]] + [i.text for i in world.teach_none]
    if use_map:
        import learned_map
        cap = world.sizes.map_cap
        items: List[ceiling.Item] = []
        for s in world.taught:
            items += [ceiling.Item(e.span, e.span, s, e.family, e.template_ids, "teach", (e.a, e.b)) for e in world.teach[s][:cap]]
        items += world.teach_none[:cap]
        f = (mapper or learned_map.make_mapper(seed))(X[[index[i.text] for i in items]], items, list(world.taught))
    else:
        def f(Z):
            return Z
    prep = BL.centred_unit(f(X[[index[t] for t in teach_texts]]))
    return prep(f(X))


def _batches(world: World, rng: np.random.Generator) -> List[dict]:
    cases = world.test_cases
    idx = [i for i, c in enumerate(cases) if c.kind == "arith" and c.family in assignments.TEST_FAMILIES and c.status != "grey"]
    by_skill = {s: [i for i in idx if cases[i].skill == s] for s in skills.SKILLS}
    pools = {"taught": [s for s in world.taught if world.truth[s] in ("solid", "thin") and by_skill[s]], "untaught": [s for s in skills.SKILLS if world.truth[s] == "unseen" and by_skill[s]]}
    out = []
    for k in world.sizes.batch_sizes:
        for kind, ss in pools.items():
            for _ in range(world.sizes.batches if ss else 0):
                s = ss[int(rng.integers(0, len(ss)))]
                avail = by_skill[s]
                out.append({"size": k, "kind": kind, "skill": s, "rows": [int(i) for i in rng.choice(avail, size=min(k, len(avail)), replace=False)]})
    return out


def fit_arrays(world: World, V: Callable[[Sequence[str]], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """The teaching vectors with their class names (the taught skills and "other"): what every baseline is fitted on."""
    X = np.vstack([V([e.span for e in world.teach[s]]) for s in world.taught] + [V([i.text for i in world.teach_none])])
    y = np.array([s for s in world.taught for _ in world.teach[s]] + ["other"] * len(world.teach_none))
    return X, y


def calibration_arrays(world: World, V: Callable[[Sequence[str]], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(standard calibration vectors, their group names, adapted calibration vectors, their group names). The standard set is held-out S1 to S3 parts of the taught groups and of
    Other content; the adapted set adds a few S4 to S8 parts of the taught skills."""
    names = list(world.taught) + ["other"]
    cal_vec = np.vstack([V(world.cal[n]) for n in names])
    cal_grp = np.array([n for n in names for _ in world.cal[n]])
    ada_vec = np.vstack([cal_vec] + [V(world.adapt[s]) for s in world.taught])
    ada_grp = np.concatenate([cal_grp] + [np.array([s] * len(world.adapt[s])) for s in world.taught])
    return cal_vec, cal_grp, ada_vec, ada_grp


def _evaluate(world: World, P: np.ndarray, index: Dict[str, int], seed: int) -> dict:
    sizes = world.sizes

    def V(texts: Sequence[str]) -> np.ndarray:
        return P[[index[t] for t in texts]]

    taught = world.taught
    groups = {s: ST.SkillGroup(s, V([e.span for e in world.teach[s]]), len(world.teach[s]), None, world.comp[s]["accuracy"]) for s in taught}
    other = ST.SkillGroup("other", V([i.text for i in world.teach_none]), len(world.teach_none), None, 1.0)
    names = list(taught) + ["other"]
    cal_vec, cal_grp, ada_vec, ada_grp = calibration_arrays(world, V)
    cases, tvec = world.test_cases, V(world.test_texts)
    rec: dict = {"assignment": world.assignment, "competence": world.comp, "truth": world.truth, "n_teach": {s: len(world.teach[s]) for s in taught}, "n_cases": len(cases),
                 "n_calibration": {n: len(world.cal[n]) for n in names}}
    machine: dict = {}
    keep = {}
    for q in sizes.qs:
        key = f"q{q:g}"
        machine[key] = {}
        for cond, (cv, cg) in (("standard", (cal_vec, cal_grp)), ("adapted", (ada_vec, ada_grp))):
            m = ST.StatusMachine(groups, other, cv, cg, q=q, seed=seed)
            assessed = m.assess_many(tvec)
            machine[key][cond] = EV.status_scores(assessed, cases)
            keep[(q, cond)] = (m, assessed)
    rec["machine"] = machine
    m05, assessed05 = keep[(HEADLINE_Q, "standard")]
    mat, rows, cols = EV.skill_confusion(assessed05, cases)
    rec["skill_confusion"] = {"matrix": mat, "rows": rows, "cols": cols}
    errors = {"false_unseen": sum(a.status == "unseen" for a, c in zip(assessed05, cases) if c.kind == "arith" and c.status in ("solid", "thin") and c.family in assignments.TEST_FAMILIES),
              "missed_unseen": sum(a.status != "unseen" for a, c in zip(assessed05, cases) if c.kind == "arith" and c.status == "unseen")}
    rec["consequence"] = {"errors": errors, "value": EV.offline_consequence(errors), "text": EV.consequence_text(errors)}

    # the seven baselines, fitted on the same teaching vectors and held to the same calibration
    X_fit, y_fit = fit_arrays(world, V)
    background = V(world.background)
    seg_texts = {i: {mode: [m.text[a:b] for a, b in spans] for mode, spans in segment_sets(m).items()} for i, m in enumerate(world.tangled)}
    baselines: dict = {}
    base_flags: dict = {}
    for name in BL.BASELINE_NAMES:
        det = BL.make_baseline(name, background=background, **BASELINE_PARAMS.get(name, {})).fit(X_fit, y_fit)
        s_test, s_cal, s_ada = det.novelty_score(tvec), det.novelty_score(cal_vec), det.novelty_score(ada_vec)
        baselines[name] = {}
        for q in sizes.qs:
            key = f"q{q:g}"
            thr, thr_ada = BL.conformal_threshold(s_cal, q), BL.conformal_threshold(s_ada, q)
            baselines[name][key] = {"standard": EV.flag_scores(s_test > thr, cases, s_test), "adapted": EV.flag_scores(s_test > thr_ada, cases, s_test)}
            if q == HEADLINE_Q:
                base_flags[name] = (det, thr)
    rec["baselines"] = baselines

    # group tests: batches of parts of the same skill, judged together
    batches = _batches(world, _stream(seed, 11))
    mg = ST.StatusMachine(groups, other, cal_vec, cal_grp, q=HEADLINE_Q, seed=seed, n_perm=199)
    protos = {n: g.protos / np.linalg.norm(g.protos, axis=1, keepdims=True) for n, g in {**groups, "other": other}.items()}
    gm = {str(k): {"taught": [], "untaught": []} for k in sizes.batch_sizes}
    gb = {str(k): {"taught": [], "untaught": []} for k in sizes.batch_sizes}
    brng = _stream(seed, 12)
    for b in batches:
        batch = tvec[b["rows"]]
        gm[str(b["size"])][b["kind"]].append(mg.assess_group(batch).status == "unseen")
        near = max(protos, key=lambda n: float((batch @ protos[n].T).max(axis=1).mean()))
        ref = protos[near]
        ref = ref[np.sort(brng.choice(len(ref), size=min(len(ref), 512), replace=False))]
        gb[str(b["size"])][b["kind"]].append(BL.group_test(ref, batch, len(batch), n_perm=199, seed=seed) <= HEADLINE_Q)

    def rates(d):
        return {k: {"false_unseen": (float(np.mean(v["taught"])) if v["taught"] else float("nan")), "unseen_recall": (float(np.mean(v["untaught"])) if v["untaught"] else float("nan")),
                    "n_taught": len(v["taught"]), "n_untaught": len(v["untaught"])} for k, v in d.items()}

    rec["groups"] = {"machine": rates(gm), "baseline": rates(gb)}

    # tangled messages
    states = world.truth
    tang: dict = {"machine": {}, "baselines": {}}
    for mode in ("gold", "auto_sentence", "auto_window"):
        by_part = []
        for i, msg in enumerate(world.tangled):
            spans = segment_sets(msg)[mode]
            assessed = m05.assess_many(V(seg_texts[i][mode]))
            by_part.append(list(zip(spans, assessed)))
        tang["machine"][mode] = EV.tangled_scores(world.tangled, by_part, mode, states)
    tang["b6"] = {mode: EV.b6_gap(tang["machine"]["gold"], tang["machine"][mode]) for mode in ("auto_sentence", "auto_window")}
    for name, (det, thr) in base_flags.items():
        by_part = []
        for i, msg in enumerate(world.tangled):
            score = det.novelty_score(V(seg_texts[i]["gold"]))
            by_part.append([(span, ST.Assessment("unseen" if s > thr else "solid", None, 0.0, 0, math.nan, 1, "")) for span, s in zip(segment_sets(msg)["gold"], score)])
        tang["baselines"][name] = EV.tangled_scores(world.tangled, by_part, "gold", states)
    rec["tangled"] = tang
    return rec


def prepare(world: World, rep_id: str, rep: Optional[reps.Representation] = None, cache_dir=CACHE, mapper: Optional[Callable] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    """Read every text of the world with the reader ``rep_id`` (``R5_<id>`` is the learned map on top of ``<id>``): the centred unit vectors, and the row of each text."""
    use_map = rep_id.startswith("R5_")
    rep = rep or reps.get_rep(rep_id[3:] if use_map else rep_id)
    texts = world.all_texts()
    X = reps.cached_encode(rep, texts, cache_dir) if cache_dir is not None else rep.encode(texts)
    index = {t: k for k, t in enumerate(texts)}
    return _reader(world, X, index, world.seed, use_map, mapper), index


def run_cell(rep_id: str, seed: int, pool: str, out_dir, stamp: str, sizes: Sizes = SIZES, rep: Optional[reps.Representation] = None, competence_fn: Optional[Callable] = None,
             cache_dir=CACHE, mapper: Optional[Callable] = None) -> dict:
    path = Path(out_dir) / f"{rep_id}__s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.time()
    world = build_world(seed, pool, sizes, competence_fn)
    P, index = prepare(world, rep_id, rep, cache_dir, mapper)
    rec = _evaluate(world, P, index, seed)
    rec.update(rep=rep_id, seed=seed, pool=pool, n_texts=len(index), secs=round(time.time() - t0, 1), code_stamp=stamp)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(_native(rec)))
    return rec


# ---------------------------------------------------------------------------------------------------------------------------------- the table
def load_cells(out_dir, rep_id: str, seeds: Sequence[int], stamp: str) -> List[dict]:
    rows = []
    for s in seeds:
        path = Path(out_dir) / f"{rep_id}__s{s}.json"
        if runlib.cell_done(path, stamp):
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _get(rec: dict, path: str):
    for part in path.split("/"):
        rec = rec[part]
    return rec


def _vals(rows: List[dict], path: str) -> List[float]:
    out = []
    for r in rows:
        try:
            v = _get(r, path)
        except (KeyError, TypeError):
            continue
        if isinstance(v, (int, float)) and not math.isnan(v):
            out.append(float(v))
    return out


def _cell(values: List[float], bar: Optional[Tuple[float, str]] = None) -> str:
    if not values:
        return "n/a"
    m = float(np.mean(values))
    text = f"{m:.2f} [{min(values):.2f} - {max(values):.2f}]"
    if bar is not None:
        ok = m >= bar[0] if bar[1] == "min" else m <= bar[0]
        text += " PASS" if ok else " fail"
    return text


def report_e2(out_dir, rep_ids: Sequence[str], seeds: Sequence[int], stamp: Union[str, Callable[[str], str]], pool: str = "dev", sizes: Sizes = SIZES) -> str:
    stamp_of = stamp if callable(stamp) else (lambda _r: stamp)
    cells = {r: load_cells(out_dir, r, seeds, stamp_of(r)) for r in rep_ids}
    hq = f"q{HEADLINE_Q:g}"
    L = [f"# E2 and E3, status of parts and of tangled messages (pool `{pool}`, seeds {seeds[0]}-{seeds[-1]})", "",
         "Mean over seeds, with [min - max]; PASS / fail is the mean against the DRAFT bar of spec section 7 (see the last table for the dev-fixed bars). A seed is one teaching "
         "assignment: two skills Solid (about 2,000 examples), one Thin (about 20), three never taught. Calibration for the STANDARD condition uses held-out S1 to S3 parts only; the "
         "ADAPTED condition (a few S4 to S8 parts in the calibration) is reported next to it and is not gated: if only the adapted condition reaches the bar the claim is "
         "'needs adaptation to new wording'. `R5_<id>` is the learned map on top of `<id>`; `R0_hash` is the word-matcher control.", ""]
    L += [f"## B3a, the status machine at q = {HEADLINE_Q} (calibration on S1 to S3 only)", "",
          "| reader | seeds | macro F1 over solid/thin/unseen (bar 0.85) | false unseen on S4-S8 (bar 0.05) | thin vs solid AUROC (bar 0.85) | Other content called unseen (bar 0.05) | unseen skills found |",
          "|---|---|---|---|---|---|---|"]
    for r in rep_ids:
        rows = cells[r]
        if not rows:
            L.append(f"| {r} | 0 | missing | missing | missing | missing | missing |")
            continue
        p = f"machine/{hq}/standard/"
        L.append(f"| {r} | {len(rows)} | {_cell(_vals(rows, p + 'macro_f1/macro_f1'), BARS['macro_f1'])} | {_cell(_vals(rows, p + 'false_unseen_heldout'), BARS['false_unseen_heldout'])} | "
                 f"{_cell(_vals(rows, p + 'thin_vs_solid_auroc'), BARS['thin_vs_solid_auroc'])} | {_cell(_vals(rows, p + 'other_as_unseen'), BARS['other_as_unseen'])} | {_cell(_vals(rows, p + 'unseen_recall'))} |")
    L += ["", f"### The adapted condition at q = {HEADLINE_Q} (reported, not gated) and the wording claim", "",
          "| reader | false unseen on S4-S8, standard | false unseen on S4-S8, adapted | false unseen on taught wordings S1-S3 | claim from the means (bar 0.05) |", "|---|---|---|---|---|"]
    for r in rep_ids:
        rows = cells[r]
        if not rows:
            L.append(f"| {r} | missing | missing | missing | missing |")
            continue
        std, ada = _vals(rows, f"machine/{hq}/standard/false_unseen_heldout"), _vals(rows, f"machine/{hq}/adapted/false_unseen_heldout")
        claim = EV.wording_claim(float(np.mean(std)) if std else float("nan"), float(np.mean(ada)) if ada else float("nan"), BARS["false_unseen_heldout"][0])
        L.append(f"| {r} | {_cell(std)} | {_cell(ada)} | {_cell(_vals(rows, f'machine/{hq}/standard/false_unseen_taught_wording'))} | {claim} |")
    L += ["", "### The curve over q (standard condition)", "", "| reader | q | false unseen on S4-S8 | unseen skills found | Other content called unseen | near-twins of taught skills called unseen | near-twins of untaught skills found |", "|---|---|---|---|---|---|---|"]
    for r in rep_ids:
        for q in sizes.qs:
            rows, key = cells[r], f"q{q:g}"
            if rows:
                p = f"machine/{key}/standard/"
                L.append(f"| {r} | {q:g} | {_cell(_vals(rows, p + 'false_unseen_heldout'))} | {_cell(_vals(rows, p + 'unseen_recall'))} | {_cell(_vals(rows, p + 'other_as_unseen'))} | "
                         f"{_cell(_vals(rows, p + 'twin_false_unseen'))} | {_cell(_vals(rows, p + 'twin_unseen_recall'))} |")
    L += ["", f"## B3b, the baselines at the same calibration (q = {HEADLINE_Q}, standard condition)", "",
          "Each baseline flags a part as unseen when its novelty score is above the threshold taken from the same held-out calibration parts. AUROC = parts of untaught skills against parts of taught skills, no threshold.", "",
          "| reader | method | false unseen on S4-S8 | unseen skills found | AUROC taught vs untaught | Other content called unseen | false unseen, adapted |", "|---|---|---|---|---|---|---|"]
    for r in rep_ids:
        rows = cells[r]
        if not rows:
            continue
        p = f"machine/{hq}/standard/"
        L.append(f"| {r} | **status machine** | {_cell(_vals(rows, p + 'false_unseen_heldout'))} | {_cell(_vals(rows, p + 'unseen_recall'))} | n/a | {_cell(_vals(rows, p + 'other_as_unseen'))} | "
                 f"{_cell(_vals(rows, f'machine/{hq}/adapted/false_unseen_heldout'))} |")
        for name in BL.BASELINE_NAMES:
            b = f"baselines/{name}/{hq}/"
            L.append(f"| {r} | {name} | {_cell(_vals(rows, b + 'standard/false_unseen_heldout'))} | {_cell(_vals(rows, b + 'standard/unseen_recall'))} | "
                     f"{_cell(_vals(rows, b + 'standard/auroc_taught_vs_untaught'))} | {_cell(_vals(rows, b + 'standard/other_as_unseen'))} | {_cell(_vals(rows, b + 'adapted/false_unseen_heldout'))} |")
    L += ["", "## E3, tangled messages (status machine, q = 0.05)", "",
          "B4a: a known part with an unseen part; B4b: a known part with Other content; B4c: an unseen part with Other content. B6: B4a under automatic segmentation must stay within 0.10 of the gold-span number.", "",
          "| reader | segmentation | B4a leftover is the unseen part (bar 0.85) | B4a known part not flagged (bar 0.95) | B4b nothing flagged (bar 0.95) | B4b Other labelled (bar 0.95) | B4c unseen found | B6 within 0.10 |", "|---|---|---|---|---|---|---|---|"]
    for r in rep_ids:
        rows = cells[r]
        for mode in ("gold", "auto_sentence", "auto_window"):
            if not rows:
                continue
            t = f"tangled/machine/{mode}/"
            gold = mode == "gold"
            within = "n/a" if gold else f"{np.mean([bool(_get(x, f'tangled/b6/{mode}/within')) for x in rows]):.2f} of seeds"
            L.append(f"| {r} | {mode} | {_cell(_vals(rows, t + 'b4a/residual_hit'), BARS['b4a_residual_hit'] if gold else None)} | {_cell(_vals(rows, t + 'b4a/known_not_flagged'), BARS['b4a_known_not_flagged'] if gold else None)} | "
                     f"{_cell(_vals(rows, t + 'b4b/nothing_flagged'), BARS['b4b_nothing_flagged'] if gold else None)} | {_cell(_vals(rows, t + 'b4b/other_labelled'), BARS['b4b_other_labelled'] if gold else None)} | "
                     f"{_cell(_vals(rows, t + 'b4c/unseen_found'))} | {within} |")
    L += ["", "Baselines on the tangled messages (gold parts; they cannot label Other content, so only the flagging numbers are shown):", "",
          "| reader | method | B4a leftover is the unseen part | B4a known part not flagged | B4b nothing flagged |", "|---|---|---|---|---|"]
    for r in rep_ids:
        for name in BL.BASELINE_NAMES:
            rows = cells[r]
            if rows:
                t = f"tangled/baselines/{name}/"
                L.append(f"| {r} | {name} | {_cell(_vals(rows, t + 'b4a/residual_hit'))} | {_cell(_vals(rows, t + 'b4a/known_not_flagged'))} | {_cell(_vals(rows, t + 'b4b/nothing_flagged'))} |")
    L += ["", "## Group tests (batches of parts of one skill judged together, q = 0.05)", "", "| reader | method | batch size | false unseen (taught skills) | unseen skills found |", "|---|---|---|---|---|"]
    for r in rep_ids:
        for method in ("machine", "baseline"):
            for k in sizes.batch_sizes:
                rows = cells[r]
                if rows:
                    g = f"groups/{method}/{k}/"
                    L.append(f"| {r} | {'status machine (every group must flag)' if method == 'machine' else 'group test, nearest group only'} | {k} | {_cell(_vals(rows, g + 'false_unseen'))} | {_cell(_vals(rows, g + 'unseen_recall'))} |")
    L += ["", "## Misassignment between skills (which taught group a part's nearest prototype belongs to; rows = true skill, summed over seeds)", ""]
    for r in rep_ids:
        rows = cells[r]
        if rows:
            mat = np.sum([np.array(x["skill_confusion"]["matrix"]) for x in rows], axis=0)
            cols = rows[0]["skill_confusion"]["cols"]
            L += [f"`{r}`", "", "| true skill | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            for name, line in zip(rows[0]["skill_confusion"]["rows"], mat):
                L.append(f"| {name} | " + " | ".join(str(int(v)) for v in line) + " |")
            L.append("")
    L += ["## Offline consequence (an ESTIMATE, not gated)", ""]
    for r in rep_ids:
        rows = cells[r]
        if rows:
            fu, mu = np.mean([x["consequence"]["errors"]["false_unseen"] for x in rows]), np.mean([x["consequence"]["errors"]["missed_unseen"] for x in rows])
            L.append(f"- `{r}`: per seed {fu:.1f} false unseen and {mu:.1f} missed unseen parts; " + EV.consequence_text({"false_unseen": float(fu), "missed_unseen": float(mu)}))
    L += ["", "## Bars: draft and dev-fixed", "",
          "Rule written before the dev run: a bar to stay above becomes max(draft, mean - 2 sd over the dev seeds), a bar to stay below becomes min(draft, mean + 2 sd); a dev result can tighten a "
          "draft bar and never loosen it. The dev-fixed bars are the ones the freeze records. Source of the dev values: the first reader above, status machine, q = 0.05, standard condition.", "",
          "| measure | draft bar | dev mean | dev-fixed bar |", "|---|---|---|---|"]
    paths = {"macro_f1": "macro_f1/macro_f1", "false_unseen_heldout": "false_unseen_heldout", "thin_vs_solid_auroc": "thin_vs_solid_auroc", "other_as_unseen": "other_as_unseen"}
    tang_paths = {"b4a_residual_hit": "tangled/machine/gold/b4a/residual_hit", "b4a_known_not_flagged": "tangled/machine/gold/b4a/known_not_flagged",
                  "b4b_nothing_flagged": "tangled/machine/gold/b4b/nothing_flagged", "b4b_other_labelled": "tangled/machine/gold/b4b/other_labelled"}
    head = cells[rep_ids[0]] if rep_ids else []
    for name, (draft, direction) in BARS.items():
        path = f"machine/{hq}/standard/{paths[name]}" if name in paths else tang_paths[name]
        v = _vals(head, path)
        L.append(f"| {name} ({'at least' if direction == 'min' else 'at most'}) | {draft:.2f} | {(f'{np.mean(v):.3f}' if v else 'n/a')} | {(f'{EV.fix_bar(v, draft, direction):.3f}' if v else 'n/a')} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="dev")
    ap.add_argument("--seeds", default="100-129")
    ap.add_argument("--reps", default="R5_R4_qwen_L26,R4_qwen_L26,R0_hash")
    args = ap.parse_args()
    runlib.set_threads(3)
    seeds = seed_range(args.seeds)
    rep_ids = args.reps.split(",")
    for rep_id in rep_ids:
        for seed in seeds:
            r = run_cell(rep_id, seed, args.pool, OUT, stamp_for(rep_id))
            s = r["machine"][f"q{HEADLINE_Q:g}"]["standard"]
            print(f"{rep_id:16s} seed {seed}: macro F1 {s['macro_f1']['macro_f1']:.3f}  false-unseen(S4-S8) {s['false_unseen_heldout']:.3f}  unseen found {s['unseen_recall']:.3f}  {r['secs']}s", flush=True)
        reps.clear_models()
    text = report_e2(OUT, rep_ids, seeds, stamp_for, args.pool)
    runlib.atomic_write_text(HERE / "results" / "E2_TABLE.md", text)
    print(text)
