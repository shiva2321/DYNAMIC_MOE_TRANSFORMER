"""Lab G4: readers calibrated on verified outcomes from the stream's own wording (spec section 6; PROGRESS.md 2026-10-30).

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/lab_outcome.py [first_seed last_seed]

The stream's known messages are split per class into a pool and an evaluation half. A message of the pool counts as VERIFIED when its ownership is confirmed by an outcome: in the text domains
its label is one of the taught classes (exact); in arithmetic a taught expert reproduces the stated answer (a noisy check: only for messages the experts solve, and a never-taught message can be
vouched for by coincidence, which is measured). A random sample of n verified messages calibrates every reader's cut (arm "verified"), or is added to the held-out messages of the teaching
wording (arm "union"); the arm "fixed" is the calibration of E-048 (teaching wording only). Systems, judged on the evaluation half of the known messages and on the never-taught and foreign
messages: each reader alone, the fusions of the pair (asking, vMF) by "either" and "both", and "either" over all four readers. Five random draws per cell.
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

import baselines as BL  # noqa: E402
import lab_general as LG  # noqa: E402
import lab_pool as LP  # noqa: E402
import outcome as OC  # noqa: E402
import reader as RD  # noqa: E402
import route as RT  # noqa: E402
import runlib  # noqa: E402

OUT = _KS / "results" / "routing_general" / "g4"
STAMP_FILES = [_HERE / n for n in ("outcome.py", "lab_outcome.py", "lab_general.py", "lab_pool.py", "measures.py", "reader.py", "cards.py", "route.py", "domains.py", "arith_domain.py", "pool.py")] + [
    _KS / "baselines.py", _KS / "lab_function.py", _KS / "function_check.py", _KS / "routing_lab" / "ask_reader.py"]
READERS = ("ask_exemplars", "vmf_conformal", "mahalanobis", "knn_distance")
PAIR = ("ask_exemplars", "vmf_conformal")
SYSTEMS = {**{r: ([r], "either") for r in READERS}, "either_pair": (list(PAIR), "either"), "both_pair": (list(PAIR), "both"), "either_all": (list(READERS), "either")}
ARMS = ("fixed", "verified", "union")
NV = (10, 20, 30, 40)
HEADLINE = 30
DRAWS = 5
Q = LG.Q
N_CAL_TEXT = 60
KNOWN_LIMIT, POWER_MARGIN = 0.10, 0.05                                                   # rule C8
SEEDS = list(range(150, 153))


class TextVerifier:
    """In a labelled domain the label is the outcome: a known message is owned, a never-taught one is not."""

    def known(self, texts: Sequence[str]) -> np.ndarray:
        return np.ones(len(texts), dtype=bool)

    def unowned(self, texts: Sequence[str]) -> np.ndarray:
        return np.zeros(len(texts), dtype=bool)


class AnswerVerifier:
    """In arithmetic the stated answer is the outcome: a message is vouched for when the numbers can be read and some taught expert gives the stated answer."""

    def __init__(self, experts: Sequence[Callable], stated: Dict[str, int]) -> None:
        self.experts, self.stated = list(experts), stated

    def known(self, texts: Sequence[str]) -> np.ndarray:
        out = []
        for t in texts:
            f = LP.parse_pair(t)
            out.append(f is not None and t in self.stated and any(e(f) == self.stated[t] for e in self.experts))
        return np.array(out, dtype=bool)

    unowned = known                                                                      # the same rule, applied to messages nobody owns: how often a coincidence vouches for them


def score_sets(world, foreign: Sequence[str], ask: Optional[Callable], encode: Callable, cache_dir) -> Dict[str, Dict[str, np.ndarray]]:
    """Novelty scores of every reader on the calibration messages (teaching wording), all known test messages, never-taught and foreign ones."""
    sets = {"cal": [t for t, _ in world.known_cal], "known": [t for t, _ in world.known_test], "unknown": list(world.unknown_test), "foreign": list(foreign)}
    rd = RD.CardReader(world.cards, with_exemplars=True, cache_dir=cache_dir, ask=ask)
    S = {"ask_exemplars": {k: RT.unclaimed_score(rd.probabilities(v)) for k, v in sets.items()}}
    X, y = [], []
    for i, n in enumerate(world.taught):
        X += world.teach[n]
        y += [i] * len(world.teach[n])
    Xv, y = encode(X), np.asarray(y)
    V = {k: encode(v) for k, v in sets.items()}
    for name in (r for r in READERS if r != "ask_exemplars"):
        det = BL.make_baseline(name).fit(Xv, y)
        S[name] = {k: det.novelty_score(V[k]) for k in sets}
    return S


def run_cell(world, foreign: Sequence[str], verifier, ask: Optional[Callable] = None, encode: Callable = LG.default_encode, cache_dir=LG.ASK_DIR, seed: int = 0,
             draws: int = DRAWS, nvs: Sequence[int] = NV, q: float = Q, unowned_extra: Sequence[str] = ()) -> dict:
    S = score_sets(world, foreign, ask, encode, cache_dir)
    known_t, owner = [t for t, _ in world.known_test], [i for _, i in world.known_test]
    verified = verifier.known(known_t)
    solid = np.array([world.states[i] == "solid" for i in owner])
    rec: dict = {"domain": world.name, "n": {"cal": len(S["ask_exemplars"]["cal"]), "known": len(known_t), "unknown": len(world.unknown_test), "foreign": len(foreign)},
                 "verification": {"coverage": float(verified.mean()), "coverage_solid": float(verified[solid].mean()) if solid.any() else float("nan"),
                                  "coverage_thin": float(verified[~solid].mean()) if (~solid).any() else float("nan"),
                                  "false_verified_unknown": float(verifier.unowned(world.unknown_test).mean()),
                                  "false_verified_other": float(verifier.unowned(list(unowned_extra)).mean()) if len(unowned_extra) else float("nan")}}
    acc: Dict[str, Dict[str, Dict[str, List[dict]]]] = {a: {str(nv): {s: [] for s in SYSTEMS} for nv in nvs} for a in ARMS}
    n_act: Dict[str, List[int]] = {str(nv): [] for nv in nvs}
    for d in range(draws):
        pool, evl = OC.split_pool_eval(owner, np.random.default_rng([seed, 0x4734, d]))
        ev = {"known": {r: S[r]["known"][evl] for r in READERS}, "unknown": {r: S[r]["unknown"] for r in READERS}, "foreign": {r: S[r]["foreign"] for r in READERS}}
        fixed = {s: OC.system_metrics({r: S[r]["cal"] for r in src}, ev, src, rule, q) for s, (src, rule) in SYSTEMS.items()}
        for nv in nvs:
            samp = OC.verified_sample(pool, verified, nv, np.random.default_rng([seed, 0x4734, d, nv]))
            n_act[str(nv)].append(len(samp))
            for s in SYSTEMS:
                acc["fixed"][str(nv)][s].append(fixed[s])
            if len(samp) < 2:
                continue
            for arm in ("verified", "union"):
                for s, (src, rule) in SYSTEMS.items():
                    cal = {r: (S[r]["known"][samp] if arm == "verified" else np.concatenate([S[r]["cal"], S[r]["known"][samp]])) for r in src}
                    acc[arm][str(nv)][s].append(OC.system_metrics(cal, ev, src, rule, q))
    rec["results"] = {a: {nv: {s: OC.mean_over_draws([{k: v for k, v in m.items() if k != "cut"} for m in rows]) for s, rows in per.items() if rows} for nv, per in byn.items()} for a, byn in acc.items()}
    rec["n_actual"] = {nv: float(np.mean(v)) for nv, v in n_act.items()}
    return rec


def _mean(rows: List[dict], pick: Callable[[dict], float]) -> float:
    vals = [pick(r) for r in rows]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def verdicts(per_domain: Dict[str, List[dict]], nv: int = HEADLINE) -> dict:
    """Rules fixed before the run. C8: at n verified messages, in EVERY domain the either-fusion of the pair calibrated on the verified sample flags at most 0.10 of the held-out known
    messages and finds at least (the best reader alone, calibrated the same way) - 0.05 of the never-taught messages. C9 (sanity, true by construction): in arithmetic the fixed either-fusion
    flags at least 0.30 of the known messages and the verified one at most 0.10."""
    c8: Dict[str, dict] = {}
    for d, rows in per_domain.items():
        g = lambda arm, s, k: _mean(rows, lambda r: r["results"][arm][str(nv)][s][k] if s in r["results"][arm][str(nv)] else float("nan"))
        best = max(g("verified", r, "unknown_found") for r in READERS)
        kf, uf = g("verified", "either_pair", "known_flagged"), g("verified", "either_pair", "unknown_found")
        c8[d] = {"known_flagged": kf, "found": uf, "best_single_found": best, "fixed_known_flagged": g("fixed", "either_pair", "known_flagged"),
                 "pass": bool(kf <= KNOWN_LIMIT and uf >= best - POWER_MARGIN)}
    c9 = None
    if "arithmetic" in c8:
        c9 = {"fixed": c8["arithmetic"]["fixed_known_flagged"], "verified": c8["arithmetic"]["known_flagged"], "pass": bool(c8["arithmetic"]["fixed_known_flagged"] >= 0.30 and c8["arithmetic"]["known_flagged"] <= 0.10)}
    return {"C8": c8, "C8_pass": all(v["pass"] for v in c8.values()), "C9": c9}


def report(per_domain: Dict[str, List[dict]], seeds: Sequence[int], nv: int = HEADLINE) -> str:
    L = [f"# Lab G4, readers calibrated on verified outcomes of the stream's own wording (seeds {seeds[0]}-{seeds[-1]}, {DRAWS} random draws per cell)", "",
         f"Headline: {nv} verified messages calibrate each cut (arm verified); arm fixed = held-out messages of the teaching wording only (E-048); arm union = both. Judged on the held-out half of "
         "the known messages (nominal false-flag rate 0.05) and on never-taught and foreign messages.", ""]
    for d, rows in per_domain.items():
        v = lambda k: _mean(rows, lambda r: r["verification"][k])
        L += [f"## {d} ({len(rows)} seeds; {rows[0]['n']['known']} known, {rows[0]['n']['unknown']} never-taught, {rows[0]['n']['foreign']} foreign per seed)", "",
              f"Verification: {v('coverage'):.3f} of the known messages are verified (Solid-taught owners {v('coverage_solid'):.3f}, Thin {v('coverage_thin'):.3f}); never-taught messages wrongly vouched for "
              f"{v('false_verified_unknown'):.3f}" + (f", other unowned {v('false_verified_other'):.3f}" if not np.isnan(v('false_verified_other')) else "") + f". Verified messages available in the pool at n={nv}: "
              f"{_mean(rows, lambda r: r['n_actual'][str(nv)]):.1f} on average.", "",
              "| system | fixed: known flagged | found | AUROC | verified: known flagged | found | AUROC | union: known flagged | found |", "|---|---|---|---|---|---|---|---|---|"]
        for s in SYSTEMS:
            g = lambda arm, k: _mean(rows, lambda r: r["results"][arm][str(nv)][s][k] if s in r["results"][arm][str(nv)] else float("nan"))
            L.append(f"| {s} | {g('fixed', 'known_flagged'):.3f} | {g('fixed', 'unknown_found'):.3f} | {g('fixed', 'auroc_unknown'):.3f} | {g('verified', 'known_flagged'):.3f} | "
                     f"{g('verified', 'unknown_found'):.3f} | {g('verified', 'auroc_unknown'):.3f} | {g('union', 'known_flagged'):.3f} | {g('union', 'unknown_found'):.3f} |")
        L += ["", "How many verified messages are needed (either-fusion of the pair, arm verified):", "", "| n asked | n available | known flagged | never-taught found | foreign found |", "|---|---|---|---|---|"]
        for n_ in sorted({int(k) for k in rows[0]["results"]["verified"]}):
            g = lambda k: _mean(rows, lambda r: r["results"]["verified"][str(n_)]["either_pair"][k] if "either_pair" in r["results"]["verified"][str(n_)] else float("nan"))
            L.append(f"| {n_} | {_mean(rows, lambda r: r['n_actual'][str(n_)]):.1f} | {g('known_flagged'):.3f} | {g('unknown_found'):.3f} | {g('foreign_found'):.3f} |")
        L.append("")
    vd = verdicts(per_domain, nv)
    L += ["## Verdicts (rules fixed before the run)", ""]
    for d, x in vd["C8"].items():
        L.append(f"- C8, {d}: either-fusion calibrated on the verified sample flags {x['known_flagged']:.3f} of held-out known (<= {KNOWN_LIMIT}; the fixed calibration flags {x['fixed_known_flagged']:.3f}) and finds "
                 f"{x['found']:.3f} (best reader alone {x['best_single_found']:.3f}, need >= that - {POWER_MARGIN}). **{'PASS' if x['pass'] else 'FAIL'}**")
    L.append(f"- C8 overall (all four domains): **{'PASS' if vd['C8_pass'] else 'FAIL'}**")
    if vd["C9"]:
        L.append(f"- C9 (sanity, true by construction): arithmetic fixed {vd['C9']['fixed']:.3f} (>= 0.30), verified {vd['C9']['verified']:.3f} (<= 0.10). **{'PASS' if vd['C9']['pass'] else 'FAIL'}**")
    L.append("")
    return "\n".join(L)


def build(domain: str, seed: int):
    """(world, foreign, verifier, unowned_extra) of one domain and seed."""
    if domain == "arithmetic":
        import arith_domain as AD
        import lab_function as LF
        dw, foreign, extras, w = AD.build_full(seed)
        ex = LF.test_examples(w)
        solvers, _, _ = LF.train_solvers(w)
        stated = {t: e.answer for t, e in ex.items()}
        return dw, foreign, AnswerVerifier([LP.solver_expert(solvers[s]) for s in w.taught], stated), extras["unknown_teaching_wording"]
    world, foreign, _ = LG.build_world(domain, seed, N_CAL_TEXT)
    return world, foreign, TextVerifier(), []


def run_seed(domain: str, seed: int, out_dir, stamp: str) -> dict:
    path = Path(out_dir) / f"{domain}_s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world, foreign, verifier, extra = build(domain, seed)
    rec = run_cell(world, foreign, verifier, seed=seed, unowned_extra=extra)
    rec.update({"seed": seed, "code_stamp": stamp})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(LG._native(rec)))
    return rec


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    per_domain: Dict[str, List[dict]] = {d: [] for d in LG.ALL_DOMAINS}
    for domain in LG.ALL_DOMAINS:
        for seed in range(lo, hi + 1):
            r = run_seed(domain, seed, OUT, stamp)
            per_domain[domain].append(r)
            e = r["results"]["verified"][str(HEADLINE)].get("either_pair", {})
            f = r["results"]["fixed"][str(HEADLINE)]["either_pair"]
            print(f"{domain} seed {seed}: verified {r['n_actual'][str(HEADLINE)]:.0f}/{HEADLINE}; either_pair known flagged fixed {f['known_flagged']:.3f} -> verified {e.get('known_flagged', float('nan')):.3f}; "
                  f"found {f['unknown_found']:.3f} -> {e.get('unknown_found', float('nan')):.3f}", flush=True)
    text = report(per_domain, list(range(lo, hi + 1)))
    runlib.atomic_write_text(_KS / "results" / f"ROUTING_GENERAL_G4_{lo}_{hi}.md", text)
    print(text)
