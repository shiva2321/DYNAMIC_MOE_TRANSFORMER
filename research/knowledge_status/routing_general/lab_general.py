"""Lab G1: one reader and one rule across domains (spec docs/superpowers/specs/2026-10-30-general-routing-layer-design.md, section 4).

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/lab_general.py [first_seed last_seed]

For each domain (customer-service intents, news topics, newsgroups) and seed: a few classes are taught (60 messages each; the reader sees only the class name, and in the second arm two
of its messages), the rest are never taught, and a set of foreign messages (the other text domains and science questions) is added. The card reader and the domain-free rule are the same
for every domain. The look-based detectors of the arithmetic lab (Mahalanobis, kNN, vMF conformal) read the same messages as MiniLM vectors, fitted on all 60 teaching messages per class,
and are cut at the same 5 % false-flag rate on the same held-out calibration messages. Nothing in this file knows a domain except the loader names and the number of taught classes.
"""

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent
_KS = _HERE.parent
for _p in (str(_KS.parents[1]), str(_KS), str(_KS / "routing_lab"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

import baselines as BL  # noqa: E402
import corpora  # noqa: E402
import domains as DM  # noqa: E402
import measures as MS  # noqa: E402
import reader as RD  # noqa: E402
import reps  # noqa: E402
import route as RT  # noqa: E402
import runlib  # noqa: E402

OUT = _KS / "results" / "routing_general" / "g1"
ASK_DIR = _KS / "results" / "routing_general" / "ask"
VEC_DIR = _KS / "results" / "routing_general" / "vec"
STAMP_FILES = [_HERE / n for n in ("cards.py", "route.py", "reader.py", "measures.py", "domains.py", "arith_domain.py", "lab_general.py")] + [_KS / "baselines.py", _KS / "reps.py", _KS / "routing_lab" / "ask_reader.py"]
DOMAINS = {"banking": (DM.load_banking, 6, "agnews"), "agnews": (DM.load_agnews, 2, "banking77"), "newsgroups": (DM.load_newsgroups, 6, "banking77")}   # loader, taught classes, other text source
ARMS = ("names", "exemplars")
TAUS = (0.3, 0.5, 0.7, 0.9)
DETECTORS = ("mahalanobis", "knn_distance", "vmf_conformal")
Q = 0.05
FUSIONS = (("ask_exemplars", "vmf_conformal"), ("ask_exemplars", "mahalanobis"), ("ask_names", "vmf_conformal"))         # a reader that asks with a reader that looks
RULES = ("either", "both")                                                                                               # novel when any reader says so, or only when all do
ALL_DOMAINS = ("arithmetic", "banking", "agnews", "newsgroups")
N_TEACH, N_CAL, N_KNOWN, N_UNKNOWN, N_FOREIGN = 60, 30, 180, 300, 150
SEEDS = list(range(150, 153))


def foreign_messages(other_source: str, seed: int, n: int) -> List[str]:
    """Half from the other everyday-text source, half science questions: content no domain here teaches."""
    half = n // 2
    return corpora.load_other_sentences(half, seed, other_source) + corpora.load_unseen_content(n - half, seed)


def _native(x):
    if isinstance(x, dict):
        return {str(k): _native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_native(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def default_encode(texts: Sequence[str]) -> np.ndarray:
    return reps.cached_encode(reps.get_rep("R1_minilm384"), list(texts), VEC_DIR)


def _extras_found(S: Dict[str, np.ndarray], extras: Sequence[str], cut: float) -> Dict[str, float]:
    return {name: MS.flag_rate(S[name], cut) for name in extras}


def run_cell(world: DM.DomainWorld, foreign: Sequence[str], ask: Optional[Callable] = None, encode: Callable = default_encode, arms: Sequence[str] = ARMS,
             taus: Sequence[float] = TAUS, detectors: Sequence[str] = DETECTORS, q: float = Q, cache_dir=ASK_DIR, extras: Optional[Dict[str, Sequence[str]]] = None,
             fusions: Sequence[Sequence[str]] = FUSIONS) -> dict:
    """One domain world, one seed. ``extras`` are further sets of foreign-like messages (name -> texts), reported only by how many of them each reader or fusion flags at its own cut."""
    extras = {k: list(v) for k, v in (extras or {}).items()}
    cal_t = [t for t, _ in world.known_cal]
    known_t, truth = [t for t, _ in world.known_test], [i for _, i in world.known_test]
    sets = {"cal": cal_t, "known": known_t, "unknown": list(world.unknown_test), "foreign": list(foreign), **extras}
    rec: dict = {"domain": world.name, "taught": world.taught, "n": {k: len(v) for k, v in sets.items()}, "reader": {}, "detectors": {}, "fused": {}, "extras": {k: {} for k in extras}}
    S: Dict[str, Dict[str, np.ndarray]] = {}                                                 # source -> set -> novelty scores
    for arm in arms:
        rd = RD.CardReader(world.cards, with_exemplars=(arm == "exemplars"), cache_dir=cache_dir, ask=ask)
        P = {k: rd.probabilities(v) for k, v in sets.items()}
        dec = {k: RT.decide(P[k], world.states, 0.5) for k in extras}
        rec["reader"][arm] = {
            "tau": {str(t): MS.claim_rates(RT.decide(P["known"], world.states, t), truth, RT.decide(P["unknown"], world.states, t), RT.decide(P["foreign"], world.states, t)) for t in taus},
            "top1_known": float(np.mean(P["known"][:, :-1].argmax(axis=1) == np.asarray(truth))),
            "extras_unclaimed_tau0.5": {k: float(np.mean([d.status == "unclaimed" for d in dec[k]])) for k in extras}}
        S[f"ask_{arm}"] = {k: RT.unclaimed_score(P[k]) for k in sets}
    X, y = [], []
    for i, n in enumerate(world.taught):
        X += world.teach[n]
        y += [i] * len(world.teach[n])
    V = {k: encode(v) for k, v in sets.items()}
    Xv, y = encode(X), np.asarray(y)
    for name in detectors:
        det = BL.make_baseline(name).fit(Xv, y)
        S[name] = {k: det.novelty_score(V[k]) for k in sets}
    clf = LogisticRegression(max_iter=1000).fit(Xv, y)
    rec["classifier_top1_known"] = float((clf.predict(V["known"]) == np.asarray(truth)).mean())
    ops = {src: MS.operating_point(s["cal"], s["known"], s["unknown"], s["foreign"], q) for src, s in S.items()}
    for arm in arms:
        rec["reader"][arm]["op"] = ops[f"ask_{arm}"]
    rec["detectors"] = {name: ops[name] for name in detectors}
    for src, s in S.items():
        for k, v in _extras_found(s, extras, ops[src]["cut"]).items():
            rec["extras"][k][src] = v
    for pair in fusions:
        if not all(p in S for p in pair):
            continue
        for rule in RULES:
            cal_map = {p: S[p]["cal"] for p in pair}
            fused = {k: MS.fused_scores(cal_map, {p: S[p][k] for p in pair}, list(pair), rule) for k in sets if k != "cal"}
            fused["cal"] = MS.fused_scores(cal_map, cal_map, list(pair), rule, leave_one_out=True)
            op = MS.operating_point(fused["cal"], fused["known"], fused["unknown"], fused["foreign"], q)
            key = f"{'+'.join(pair)}|{rule}"
            rec["fused"][key] = op
            for k, v in _extras_found(fused, extras, op["cut"]).items():
                rec["extras"][k][key] = v
    return rec


def build_world(domain: str, seed: int):
    """(world, foreign messages, extra sets) of one domain and seed. The only branch on a domain name in the lab."""
    if domain == "arithmetic":
        import arith_domain as AD
        return AD.build(seed)
    loader, n_taught, other = DOMAINS[domain]
    teach, test, names = loader()
    return DM.build_labelled_world(domain, teach, test, names, n_taught, N_TEACH, N_CAL, N_KNOWN, N_UNKNOWN, seed), foreign_messages(other, seed, N_FOREIGN), {}


def run_seed(domain: str, seed: int, out_dir, stamp: str) -> dict:
    path = Path(out_dir) / f"{domain}_s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world, foreign, extras = build_world(domain, seed)
    rec = run_cell(world, foreign, extras=extras)
    rec.update({"seed": seed, "code_stamp": stamp})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(_native(rec)))
    return rec


def _mean(rows: List[dict], pick: Callable[[dict], float]) -> float:
    return float(np.nanmean([pick(r) for r in rows]))


def report(per_domain: Dict[str, List[dict]], seeds: Sequence[int], taus: Sequence[float] = TAUS) -> str:
    L = [f"# Lab G1, one reader and one rule across domains (seeds {seeds[0]}-{seeds[-1]})", "",
         "Known = test messages of taught classes; unknown = test messages of never-taught classes; foreign = other text domains and science questions. 'unclaimed' = the reader gives no card.",
         "The reader sees class names (arm names) or names plus two teaching messages (arm exemplars); the detectors are fitted on all 60 teaching messages per class.", ""]
    for domain, rows in per_domain.items():
        n = rows[0]["n"]
        L += [f"## {domain} ({len(rows)} seeds; {len(rows[0]['taught'])} taught classes; {n['known']} known, {n['unknown']} unknown, {n['foreign']} foreign messages per seed)", "",
              "Rule: claimed when the best card has probability of at least tau.", "",
              "| arm | tau | known claimed | known claimed RIGHT | right given claimed | unknown unclaimed | foreign unclaimed |", "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            for t in taus:
                g = lambda k: _mean(rows, lambda r: r["reader"][arm]["tau"][str(t)][k])
                L.append(f"| {arm} | {t} | {g('known_claimed'):.3f} | {g('known_right'):.3f} | {g('right_given_claimed'):.3f} | {g('unknown_unclaimed'):.3f} | {g('foreign_unclaimed'):.3f} |")
        L += ["", f"Same false-flag rate for everyone: the cut flags {Q} of held-out known messages.", "",
              "| reader or detector | known flagged | unknown found | foreign found | AUROC known vs unknown | AUROC known vs foreign |", "|---|---|---|---|---|---|"]
        for arm in ARMS:
            g = lambda k: _mean(rows, lambda r: r["reader"][arm]["op"][k])
            L.append(f"| ask, {arm} | {g('known_flagged'):.3f} | {g('unknown_found'):.3f} | {g('foreign_found'):.3f} | {g('auroc_unknown'):.3f} | {g('auroc_foreign'):.3f} |")
        for name in DETECTORS:
            g = lambda k: _mean(rows, lambda r: r["detectors"][name][k])
            L.append(f"| {name} (MiniLM) | {g('known_flagged'):.3f} | {g('unknown_found'):.3f} | {g('foreign_found'):.3f} | {g('auroc_unknown'):.3f} | {g('auroc_foreign'):.3f} |")
        for key in rows[0]["fused"]:
            g = lambda k: _mean(rows, lambda r: r["fused"][key][k])
            pair, rule = key.split("|")
            L.append(f"| FUSED {rule}: {pair.replace('+', ' + ')} | {g('known_flagged'):.3f} | {g('unknown_found'):.3f} | {g('foreign_found'):.3f} | {g('auroc_unknown'):.3f} | {g('auroc_foreign'):.3f} |")
        if rows[0]["extras"]:
            L += ["", "Extra sets (share flagged at each reader's own cut; for the claim rule, the share unclaimed at tau 0.5):", "", "| set | reader or detector | flagged |", "|---|---|---|"]
            for name in rows[0]["extras"]:
                for src in rows[0]["extras"][name]:
                    L.append(f"| {name} | {src.replace('+', ' + ')} | {_mean(rows, lambda r: r['extras'][name][src]):.3f} |")
                for arm in ARMS:
                    L.append(f"| {name} | ask {arm}, claim rule tau 0.5 | {_mean(rows, lambda r: r['reader'][arm]['extras_unclaimed_tau0.5'][name]):.3f} |")
        L += ["", "Who is it? (top-1 over the taught classes, known messages)", "",
              "| reader or classifier | top-1 |", "|---|---|"]
        for arm in ARMS:
            L.append(f"| ask, {arm} | {_mean(rows, lambda r: r['reader'][arm]['top1_known']):.3f} |")
        L += [f"| logistic regression on MiniLM, 60 messages per class | {_mean(rows, lambda r: r['classifier_top1_known']):.3f} |", ""]
    return "\n".join(L)


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    per_domain: Dict[str, List[dict]] = {d: [] for d in ALL_DOMAINS}
    for domain in ALL_DOMAINS:
        for seed in range(lo, hi + 1):
            r = run_seed(domain, seed, OUT, stamp)
            per_domain[domain].append(r)
            a = r["reader"]["exemplars"]["tau"]["0.5"]
            print(f"{domain} seed {seed}: exemplars tau0.5 known claimed {a['known_claimed']:.3f} right {a['known_right']:.3f} unknown unclaimed {a['unknown_unclaimed']:.3f} "
                  f"foreign unclaimed {a['foreign_unclaimed']:.3f}", flush=True)
    text = report(per_domain, list(range(lo, hi + 1)))
    runlib.atomic_write_text(_KS / "results" / f"ROUTING_GENERAL_G1_{lo}_{hi}.md", text)
    print(text)
