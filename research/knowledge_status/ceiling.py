"""E1, the information-ceiling test (spec 7): before any detector exists, can ANY signal in a representation tell the cases apart?

B1a  a trained linear probe for skill identity (six skills plus "none"), taught on the teaching wordings S1 to S3 and tested on the held-out wordings S4 to S8.
B1b  no training: is a request closer to the same request reworded (same numbers) than to its near-twin (same wrapper, same numbers, other operator)? AUROC of "closer".
B1c  no training on untaught skills: across teaching assignments, is a part of a taught skill closer to the nearest taught prototype than a part of an untaught skill?

B1b and B1c are computed twice: RAW (cosine on the vectors as they are) and CENTRED (the mean of the teaching embeddings is subtracted, then the vectors are made unit
length again), because raw language-model states all point nearly the same way (Task 6). The centred value is the primary one for every representation alike.
"""

import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import assignments
import generator as G
import reps
import skills

BARS = {"b1a": 0.95, "b1b": 0.95, "b1c": 0.90}              # draft bars (spec 7); dev may reset them, the freeze records them
INFIX = ("add", "sub", "mul", "div", "mod")                  # the skills that have a near-twin form
PROBE_CS = (0.01, 0.1, 1.0)


@dataclass(frozen=True)
class Sizes:
    teach: int = 300          # teaching messages per class (six skills plus "none")
    test: int = 150           # test messages per class
    pairs: int = 300          # (anchor, twin, partner) triples for B1b
    assignments: int = 20     # teaching assignments for B1c


SIZES = Sizes()

# a learned map (Task 9): given the embeddings of the training items, the items themselves and the skills taught, it returns the function that applies the map
Mapper = Callable[[np.ndarray, List["Item"], Sequence[str]], Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class Item:
    text: str                                  # the whole message, what the system sees
    span: str                                  # the request itself (gold span); the whole text for "none" items
    label: str                                 # a skill, or "none"
    family: str                                # S1..S8, or other / trap / unseen
    template_ids: Tuple[str, ...]
    role: str                                  # teach | test | anchor | twin | partner
    operands: Optional[Tuple[int, int]] = None


@dataclass
class Items:
    seed: int
    pool: str
    sizes: Sizes
    all: List[Item]

    def role(self, name: str) -> List[Item]:
        return [i for i in self.all if i.role == name]


# ------------------------------------------------------------------------------------------------------------------------------------ measures
def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def probe_accuracy(Xtr, ytr, Xte, yte, groups: Optional[Sequence] = None) -> float:
    """Accuracy of a standardised multinomial logistic regression. With ``groups`` (the teaching wordings) the regularisation is picked by leave-one-group-out
    cross-validation on the TRAINING data only, so the probe is tuned for generalising across wordings without ever seeing the test wordings."""
    Xtr, Xte, ytr, yte = np.asarray(Xtr), np.asarray(Xte), np.asarray(ytr), np.asarray(yte)

    def fit(X, y, c):
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=c, max_iter=300, tol=1e-3).fit(sc.transform(X), y)
        return lambda Z: clf.predict(sc.transform(Z))

    best_c = 0.1
    if groups is not None and len(set(groups)) > 1:
        groups = np.asarray(groups)
        scores = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for c in PROBE_CS:
                accs = []
                for g in np.unique(groups):
                    tr = groups != g
                    accs.append(float((fit(Xtr[tr], ytr[tr], c)(Xtr[~tr]) == ytr[~tr]).mean()))
                scores[c] = float(np.mean(accs))
        best_c = max(scores, key=scores.get)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pred = fit(Xtr, ytr, best_c)(Xte)
    return float((pred == yte).mean())


def pair_auroc(X: np.ndarray, same: Sequence[Tuple[int, int]], other: Sequence[Tuple[int, int]]) -> float:
    """AUROC that a "same" pair is closer (cosine distance) than an "other" pair: 1 means every same pair is closer, 0.5 is chance, 0 means every other pair is closer."""
    Xn = _unit(X)

    def dist(pairs):
        p = np.asarray(pairs, dtype=np.int64)
        return 1.0 - (Xn[p[:, 0]] * Xn[p[:, 1]]).sum(axis=1)

    ds, do = dist(same), dist(other)
    return float(roc_auc_score(np.r_[np.ones(len(ds)), np.zeros(len(do))], -np.r_[ds, do]))


def taught_auroc(protos: np.ndarray, X: np.ndarray, is_taught: np.ndarray) -> float:
    """AUROC that a part of a taught skill is closer to its nearest taught prototype than a part of an untaught skill (cosine distance)."""
    d = 1.0 - (_unit(X) @ _unit(protos).T).max(axis=1)
    return float(roc_auc_score(np.asarray(is_taught, dtype=int), -d))


def b1b_by_partner_family(X: np.ndarray, same: Sequence[Tuple[int, int]], other: Sequence[Tuple[int, int]], partner_family: Sequence[str]) -> Dict[str, float]:
    """B1b again, separately for triples whose reworded partner is in a taught wording (S1 to S3) and for those whose partner is a story (S4, never taught)."""
    fam = np.asarray(partner_family)
    out: Dict[str, float] = {}
    for key, chosen in (("b1b_taught", np.isin(fam, assignments.TEACH_FAMILIES)), ("b1b_story", fam == "S4")):
        ks = np.flatnonzero(chosen)
        out[key] = pair_auroc(X, [same[k] for k in ks], [other[k] for k in ks]) if len(ks) else float("nan")
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ the test bed
def _draw_none(rng, pool: str, n: int, kind: str, parity: Optional[int], exclude: set, role: str) -> List[Item]:
    """``n`` distinct "none" messages of one kind ("other" everyday sentence, "trap", or "unseen" science question). Traps are split by template parity so a test
    trap is never a teaching trap's template."""
    out: List[Item] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > 200 * n + 200:
            raise RuntimeError(f"could not draw {n} distinct {kind} messages")
        m = {"other": G.make_other, "trap": G.make_trap, "unseen": G.make_unseen_content}[kind](rng, pool)
        if m.text in exclude:
            continue
        if parity is not None and int(m.template_ids[0].split(".")[1]) % 2 != parity:
            continue
        exclude.add(m.text)
        out.append(Item(m.text, m.text, "none", m.family if kind != "other" else "other", m.template_ids, role))
    return out


def build_items(seed: int, pool: str = "dev", sizes: Sizes = SIZES) -> Items:
    rng = np.random.default_rng([seed, 0xE1])
    items: List[Item] = []
    taught_texts: set = set()
    for skill in skills.SKILLS:
        teach = assignments.draw_teaching(rng, pool, skill, sizes.teach)
        test = assignments.draw_test(rng, pool, skill, sizes.test, taught_keys=assignments.text_keys(teach))
        for role, group in (("teach", teach), ("test", test)):
            items += [Item(e.text, e.span, skill, e.family, e.template_ids, role, (e.a, e.b)) for e in group]
        taught_texts |= {e.text for e in teach}
    used = set(taught_texts)
    n_trap = sizes.teach // 3
    items += _draw_none(rng, pool, sizes.teach - n_trap, "other", None, used, "teach") + _draw_none(rng, pool, n_trap, "trap", 0, used, "teach")
    third = sizes.test // 3
    items += (_draw_none(rng, pool, sizes.test - 2 * third, "other", None, used, "test") + _draw_none(rng, pool, third, "trap", 1, used, "test")
              + _draw_none(rng, pool, third, "unseen", None, used, "test"))
    # B1b: the same request reworded (same skill and numbers, other wording) against its near-twin (same wrapper and numbers, other operator)
    triples = 0
    while triples < sizes.pairs:
        skill, family = INFIX[int(rng.integers(0, len(INFIX)))], ("S1", "S2")[int(rng.integers(0, 2))]
        anchor = G.make_single(rng, pool, skill, family)
        part = next(x for x in anchor.parts if x.kind == "skill")
        twin = None
        for to in [INFIX[int(k)] for k in rng.permutation(len(INFIX)) if INFIX[int(k)] != skill]:
            try:
                twin = G.make_near_twin(rng, anchor, to)
                break
            except ValueError:
                continue                                                    # for example a twin that would divide by zero; try the next operator
        if twin is None:
            continue
        fams = [f for f in ("S1", "S2", "S3", "S4") if f != family]
        partner = G.make_single(rng, pool, skill, fams[int(rng.integers(0, len(fams)))], operands=(part.a, part.b))
        for role, m, label in (("anchor", anchor, skill), ("twin", twin, next(x for x in twin.parts if x.kind == "skill").skill), ("partner", partner, skill)):
            p = next(x for x in m.parts if x.kind == "skill")
            items.append(Item(m.text, m.text[p.start:p.end], label, m.family, m.template_ids, role, (part.a, part.b)))
        triples += 1
    return Items(seed, pool, sizes, items)


# ------------------------------------------------------------------------------------------------------------------------------------ one cell
def run_ceiling(rep_id: str, seed: int, pool: str = "dev", sizes: Sizes = SIZES, rep: Optional[reps.Representation] = None, cache_dir: Optional[Path] = None,
                mapper: Optional[Mapper] = None) -> dict:
    """B1a, B1b, B1c (and their raw and span variants) for one representation and one seed. ``rep`` replaces the representation named by ``rep_id`` (used for fakes).
    ``mapper`` is a learned map put on top of the representation (R5): it is fitted once on the teaching messages of all six skills (for B1a and B1b) and once per
    teaching assignment on the taught skills only (for B1c), never on an evaluation text, a held-out wording or an untaught skill. The default is no map."""
    t0 = time.time()
    items = build_items(seed, pool, sizes)
    rep = rep or reps.get_rep(rep_id)
    texts = list(dict.fromkeys([i.text for i in items.all] + [i.span for i in items.all]))
    X = reps.cached_encode(rep, texts, cache_dir) if cache_dir is not None else rep.encode(texts)
    index = {t: k for k, t in enumerate(texts)}

    def emb(group: List[Item], attr: str = "text") -> np.ndarray:
        return X[[index[getattr(i, attr)] for i in group]]

    teach, test = items.role("teach"), items.role("test")
    ytr, yte = np.array([i.label for i in teach]), np.array([i.label for i in test])
    folds = np.array([i.family if i.family in assignments.TEACH_FAMILIES else assignments.TEACH_FAMILIES[k % 3] for k, i in enumerate(teach)])
    fit = mapper or (lambda Xtr, train, taught: (lambda Z: Z))
    held = {i.text for i in items.all if i.role in ("anchor", "twin", "partner")}           # a map is never fitted on a message it is later asked about
    fit_pool = [i for i in teach if i.text not in held]
    f = fit(emb(fit_pool), fit_pool, skills.SKILLS)
    out: Dict[str, object] = {"rep": rep_id, "seed": seed, "pool": pool}
    out["b1a"] = probe_accuracy(f(emb(teach)), ytr, f(emb(test)), yte, folds)
    out["b1a_span"] = probe_accuracy(f(emb(teach, "span")), ytr, f(emb(test, "span")), yte, folds)

    anchors, twins, partners = (f(emb(items.role(r))) for r in ("anchor", "twin", "partner"))
    n = len(anchors)
    Xb = np.vstack([anchors, partners, twins])
    same, other = [(k, n + k) for k in range(n)], [(k, 2 * n + k) for k in range(n)]
    centre = f(emb(teach)).mean(axis=0)
    Xc = _unit(Xb - centre)
    out["b1b_raw"] = pair_auroc(Xb, same, other)
    out["b1b"] = pair_auroc(Xc, same, other)
    out.update(b1b_by_partner_family(Xc, same, other, [p.family for p in items.role("partner")]))

    arng = np.random.default_rng([seed, 0xE1C])
    test_sk = [i for i in test if i.label != "none"]
    Xt0, lab = emb(test_sk), np.array([i.label for i in test_sk])
    nones = [i for i in fit_pool if i.label == "none"]
    raw, cen = [], []
    for _ in range(sizes.assignments):
        asg = assignments.draw_assignment(arng)
        rows, keys, trained_on = [], [], []
        for skill in skills.SKILLS:
            if asg[skill] == "unseen":
                continue
            mine = [i for i in teach if i.label == skill]
            mine = mine[:min(len(mine), assignments.COUNTS[asg[skill]])]      # a Thin skill was taught only a handful of messages
            rows.append(emb(mine))
            keys += [(skill, i.family) for i in mine]
            trained_on += [i for i in mine if i.text not in held]
        T0 = np.vstack(rows)
        g = fit(np.vstack([emb(trained_on), emb(nones)]), trained_on + nones, [s for s in skills.SKILLS if asg[s] != "unseen"])
        T, Xt = g(T0), g(Xt0)
        is_taught = np.array([asg[s] != "unseen" for s in lab])

        def protos(Z):
            return np.vstack([_unit(Z[[k for k, key in enumerate(keys) if key == g]].mean(axis=0, keepdims=True)) for g in sorted(set(keys))])

        raw.append(taught_auroc(protos(_unit(T)), Xt, is_taught))
        c = T.mean(axis=0)
        cen.append(taught_auroc(protos(_unit(T - c)), _unit(Xt - c), is_taught))
    out["b1c_raw"], out["b1c"], out["b1c_min"] = float(np.mean(raw)), float(np.mean(cen)), float(np.min(cen))
    out["chance"] = {"b1a": 1 / len(set(ytr)), "b1b": 0.5, "b1c": 0.5}
    out["sizes"] = asdict(sizes)
    out["n_texts"] = len(texts)
    out["secs"] = round(time.time() - t0, 1)
    return out
