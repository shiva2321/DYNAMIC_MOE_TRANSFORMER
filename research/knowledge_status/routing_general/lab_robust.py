"""Lab G4c: a robust calibration against wrongly verified messages (PROGRESS.md 2026-10-30, plan and rules C11-C13 written before the run).

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/lab_robust.py [first_seed last_seed]

The verified sample that calibrates every reader's cut can hold never-taught messages the outcome check wrongly vouches for; one of them above every known message sets a cut that is the largest score
(lab G4b: a third of the power lost at about 9 % wrong). Calibrators, all applied per reader to the same verified sample and then fused by "either": plain (G4), trim k (drop the k largest scores),
agree (the asking reader's top option must be an expert the outcome names). Contamination: real (arithmetic's answer check, as G4b), injected 10 % and 20 % (every domain). Paired samples: the clean
arm is the same random order without the wrong messages. The random streams of G4b are reused so that plain in the real mode must reproduce it.
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

import lab_general as LG  # noqa: E402
import lab_outcome as LO  # noqa: E402
import lab_pool as LP  # noqa: E402
import outcome as OC  # noqa: E402
import reader as RD  # noqa: E402
import robust as RB  # noqa: E402
import runlib  # noqa: E402

OUT = _KS / "results" / "routing_general" / "g4c"
STAMP_FILES = [_HERE / "robust.py", _HERE / "lab_robust.py"] + list(LO.STAMP_FILES)
SYSTEMS_C = {"ask_exemplars": (["ask_exemplars"], "either"), "vmf_conformal": (["vmf_conformal"], "either"), "either_pair": (list(LO.PAIR), "either")}
TRIMS = {"plain": 0, "trim1": 1, "trim2": 2, "trim3": 3, "trim6": 6, "agree": 0}
CALIBRATORS = tuple(TRIMS)
CAL_FOR_K = {0: "plain", 1: "trim1", 2: "trim2", 3: "trim3", 6: "trim6"}
MODES = ("real", "inj10", "inj20")
INJECT = {"inj10": 0.10, "inj20": 0.20}
TEXT_DOMAINS = ("banking", "agnews", "newsgroups")
NVC = (30, 40, 60)
HEADLINE = 30
DRAWS = 20
Q = LG.Q
KNOWN_LIMIT, RETENTION, THEORY_TOL = 0.10, 0.85, 0.03                                    # rules C11-C13
SEEDS = list(range(150, 153))


def reader_top(world, foreign: Sequence[str], ask: Optional[Callable], cache_dir) -> Dict[str, np.ndarray]:
    """The asking reader's top option for the known and the never-taught messages: a card index, or -1 for "none of these"."""
    rd = RD.CardReader(world.cards, with_exemplars=True, cache_dir=cache_dir, ask=ask)
    k = len(world.cards)
    out = {}
    for name, texts in (("known", [t for t, _ in world.known_test]), ("unknown", list(world.unknown_test))):
        top = rd.probabilities(texts).argmax(axis=1)
        out[name] = np.where(top >= k, -1, top)
    return out


def owner_sets(verifier, texts: Sequence[str], owner: Optional[Sequence[int]] = None) -> List[frozenset]:
    """The experts an outcome names for each message: those that reproduce the stated answer (arithmetic), or the true class (a labelled domain; empty for messages without one)."""
    if isinstance(verifier, LO.AnswerVerifier):
        out = []
        for t in texts:
            f = LP.parse_pair(t)
            out.append(frozenset(i for i, e in enumerate(verifier.experts) if f is not None and t in verifier.stated and e(f) == verifier.stated[t]))
        return out
    return [frozenset({int(o)}) for o in owner] if owner is not None else [frozenset() for _ in texts]


def robust_arms(S, owner: Sequence[int], verified: np.ndarray, v_unk: np.ndarray, top: Dict[str, np.ndarray], known_owners: Sequence[frozenset], unknown_owners: Sequence[frozenset],
                n_cards: int, u: float, seed: int, draws: int, nvs: Sequence[int], q: float, modes: Sequence[str] = MODES) -> dict:
    """mode -> {results: calibrator -> n -> system -> {contaminated, clean} metrics, share: calibrator -> n -> mean share of never-taught messages in the sample, n_actual: calibrator -> n ->
    mean sample size}. The draws are those of lab G4b (``contamination_arms``): the same split of the known messages and of the never-taught ones, the same random order, so plain in the real
    mode reproduces it. The injected modes add never-taught messages from the stream half as if verified (the number set so that they make the asked share of the verified pool); the experts the
    second opinion compares with are the real matches in the real mode and a random card in the injected ones."""
    owner, verified, v_unk = np.asarray(owner), np.asarray(verified, dtype=bool), np.asarray(v_unk, dtype=bool)
    n_known, n_unk = len(verified), len(v_unk)
    A = {r: np.concatenate([S[r]["known"], S[r]["unknown"]]) for r in LO.READERS}
    agree_known = RB.owner_agrees(top["known"], known_owners)
    agree_real = RB.owner_agrees(top["unknown"], unknown_owners)
    nvs = list(nvs)
    new = lambda: {m: {c: {nv: [] for nv in nvs} for c in CALIBRATORS} for m in modes}
    acc = {m: {c: {nv: {s: {"contaminated": [], "clean": []} for s in SYSTEMS_C} for nv in nvs} for c in CALIBRATORS} for m in modes}
    shares, sizes = new(), new()
    for d in range(draws):
        pool, evl = OC.split_pool_eval(owner, np.random.default_rng([seed, 0x4734, d]))
        perm = np.random.default_rng([seed, 0x4735, d]).permutation(n_unk)
        unk_pool, unk_eval = perm[:n_unk // 2], perm[n_unk // 2:]
        ev = {"known": {r: S[r]["known"][evl] for r in LO.READERS}, "unknown": {r: S[r]["unknown"][unk_eval] for r in LO.READERS}, "foreign": {r: S[r]["foreign"] for r in LO.READERS}}
        for mi, mode in enumerate(modes):
            flags = np.concatenate([verified, np.zeros(n_unk, dtype=bool)])
            if mode == "real":
                stream_unk = unk_pool[:int(round(len(pool) * u / (1.0 - u)))]
                flags[n_known + stream_unk] = v_unk[stream_unk]
                agree_unk, tag = agree_real, []
            else:
                stream_unk = unk_pool[:RB.injected_count(int(verified[pool].sum()), INJECT[mode])]
                flags[n_known + stream_unk] = True
                named = np.random.default_rng([seed, 0x4736, d, mi]).integers(0, n_cards, size=n_unk)
                agree_unk, tag = np.asarray(top["unknown"]) == named, [mi]
            agree_all = np.concatenate([agree_known, agree_unk])
            candidates = pool + [n_known + int(i) for i in stream_unk]
            for nv in nvs:
                order = OC.verified_order(candidates, flags, np.random.default_rng([seed, 0x4735, d, nv] + tag))
                for c in CALIBRATORS:
                    cand = [i for i in order if agree_all[i]] if c == "agree" else order
                    drawn = {"contaminated": cand[:nv], "clean": [i for i in cand if i < n_known][:nv]}
                    shares[mode][c][nv].append(float(np.mean([i >= n_known for i in drawn["contaminated"]])) if drawn["contaminated"] else 0.0)
                    sizes[mode][c][nv].append(float(len(drawn["contaminated"])))
                    k = TRIMS[c]
                    for arm, samp in drawn.items():
                        if len(samp) < 2 or len(samp) <= k:
                            continue
                        for s, (src, rule) in SYSTEMS_C.items():
                            cal = {r: A[r][samp] for r in src}
                            if k:
                                cal = RB.trim_top(cal, k)
                            acc[mode][c][nv][s][arm].append(OC.system_metrics(cal, ev, src, rule, q))
    mean = lambda rows: OC.mean_over_draws([{k: v for k, v in m.items() if k != "cut"} for m in rows])
    return {m: {"results": {c: {str(nv): {s: {arm: mean(rows) for arm, rows in arms.items() if rows} for s, arms in per.items() if any(arms.values())} for nv, per in byn.items()}
                            for c, byn in acc[m].items()},
                "share": {c: {str(nv): float(np.mean(v)) for nv, v in byn.items()} for c, byn in shares[m].items()},
                "n_actual": {c: {str(nv): float(np.mean(v)) for nv, v in byn.items()} for c, byn in sizes[m].items()}} for m in modes}


def run_cell(world, foreign: Sequence[str], verifier, ask: Optional[Callable] = None, encode: Callable = LG.default_encode, cache_dir=LG.ASK_DIR, seed: int = 0, draws: int = DRAWS,
             nvs: Sequence[int] = NVC, q: float = Q, contaminate: float = LO.CONTAMINATE) -> dict:
    S = LO.score_sets(world, foreign, ask, encode, cache_dir)
    known_t, owner = [t for t, _ in world.known_test], [i for _, i in world.known_test]
    unknown_t = list(world.unknown_test)
    verified, v_unk = verifier.known(known_t), verifier.unowned(unknown_t)
    known_owners = owner_sets(verifier, known_t, owner)
    top = reader_top(world, foreign, ask, cache_dir)
    rec = {"domain": world.name, "n": {"cal": len(S["ask_exemplars"]["cal"]), "known": len(known_t), "unknown": len(unknown_t), "foreign": len(foreign)},
           "verification": {"coverage": float(verified.mean()), "false_verified_unknown": float(v_unk.mean()),
                            "owner_in_set": float(np.mean([owner[i] in known_owners[i] for i in range(len(known_t)) if verified[i]])) if verified.any() else float("nan"),
                            "reader_agrees": float(np.mean(RB.owner_agrees(top["known"], known_owners)[verified])) if verified.any() else float("nan")}}
    rec["modes"] = robust_arms(S, owner, verified, v_unk, top, known_owners, owner_sets(verifier, unknown_t), len(world.cards), contaminate, seed, draws, nvs, q)
    return rec


def _g(rows: Sequence[dict], mode: str, cal: str, nv, system: str, arm: str, key: str) -> float:
    vals = []
    for r in rows:
        try:
            vals.append(r["modes"][mode]["results"][cal][str(nv)][system][arm][key])
        except KeyError:
            pass
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _test(rows: Sequence[dict], mode: str, cal: str, nv, system: str = "either_pair") -> dict:
    """The C11 test of one calibrator in one mode: both known-flagged limits and retention of its own clean power."""
    ckf, cf = _g(rows, mode, cal, nv, system, "clean", "known_flagged"), _g(rows, mode, cal, nv, system, "clean", "unknown_found")
    dkf, df = _g(rows, mode, cal, nv, system, "contaminated", "known_flagged"), _g(rows, mode, cal, nv, system, "contaminated", "unknown_found")
    return {"clean_known_flagged": ckf, "contaminated_known_flagged": dkf, "clean_found": cf, "contaminated_found": df, "retention": (df / cf) if cf > 0 else float("nan"),
            "pass": bool(ckf <= KNOWN_LIMIT and dkf <= KNOWN_LIMIT and df >= RETENTION * cf)}


def c11(arithmetic_rows: Sequence[dict], nv: int = HEADLINE) -> dict:
    """C11 (declared before the run): in arithmetic with the real answer check, some calibrator among trim1, trim2, trim3, agree has both known-flagged rates (clean and contaminated) at most
    0.10 AND keeps at least 0.85 of its own clean power under the contamination (plain keeps about 0.69)."""
    cells = {c: _test(arithmetic_rows, "real", c, nv) for c in ("trim1", "trim2", "trim3", "agree")}
    plain = _test(arithmetic_rows, "real", "plain", nv)
    return {"cells": cells, "plain_retention": plain["retention"], "plain": plain, "pass": any(x["pass"] for x in cells.values())}


def c12(per_domain: Dict[str, List[dict]]) -> dict:
    """C12 (sanity from theory): for the single readers in the text domains, clean, the mean known-flagged of trim k is within 0.03 of (k+1)/(n+1)."""
    text = [d for d in per_domain if d in TEXT_DOMAINS]
    cells = {}
    for n, ks in ((30, (0, 1, 2, 3)), (60, (0, 1, 2, 3, 6))):
        for k in ks:
            vals = [_g(per_domain[d], "real", CAL_FOR_K[k], n, s, "clean", "known_flagged") for d in text for s in ("ask_exemplars", "vmf_conformal")]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                m, th = float(np.mean(vals)), (k + 1) / (n + 1)
                cells[f"{n},{k}"] = {"mean": m, "theory": th, "ok": bool(abs(m - th) <= THEORY_TOL)}
    return {"cells": cells, "pass": bool(cells) and all(x["ok"] for x in cells.values())}


def c13(per_domain: Dict[str, List[dict]], nv: int = HEADLINE) -> dict:
    """C13 (generality): under 10 % injected contamination at n = 30, ONE calibrator passes the C11 test in every domain."""
    cells = {c: {d: _test(rows, "inj10", c, nv)["pass"] for d, rows in per_domain.items()} for c in CALIBRATORS}
    return {"cells": cells, "pass": any(v and all(v.values()) for v in cells.values())}


def _nanmean(vals: Sequence[float]) -> float:
    vals = [x for x in vals if not np.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")


def _f(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.3f}"


def report(per_domain: Dict[str, List[dict]], seeds: Sequence[int], nv: int = HEADLINE) -> str:
    L = [f"# Lab G4c, a robust calibration against wrongly verified messages (seeds {seeds[0]}-{seeds[-1]}, {DRAWS} paired draws per cell)", "",
         f"Headline: either-fusion of the pair (asking reader, vMF) at n = {nv} verified messages. Calibrators: plain (G4); trim k = the k largest scores of each reader's sample dropped; agree = the asking "
         "reader's top option must be an expert the outcome names. Modes: real = arithmetic's answer check (a quarter of the stream never-taught, as G4b); inj10 / inj20 = never-taught messages added as "
         "if verified so that they make 10 % / 20 % of the sample. Clean = the same random order without the wrong messages. Retention = contaminated found / clean found of the same calibrator.", ""]
    for d, rows in per_domain.items():
        v = lambda k: _nanmean([r["verification"].get(k, float("nan")) for r in rows])
        L += [f"## {d} ({len(rows)} seeds; {rows[0]['n']['known']} known, {rows[0]['n']['unknown']} never-taught, {rows[0]['n']['foreign']} foreign per seed)", "",
              f"Verification: {_f(v('coverage'))} of the known messages verified; never-taught wrongly vouched for {_f(v('false_verified_unknown'))}; the named experts include the true owner for "
              f"{_f(v('owner_in_set'))} of the verified known messages; the asking reader's top option is one of the named experts for {_f(v('reader_agrees'))} of them.", ""]
        for mode in MODES:
            if mode not in rows[0]["modes"] or (mode == "real" and d not in ("arithmetic",)):
                continue
            res = rows[0]["modes"][mode]["results"]
            L += [f"### {d}, mode {mode}, n = {nv}", "", "| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |", "|---|---|---|---|---|---|---|---|"]
            for c in (c_ for c_ in CALIBRATORS if c_ in res):
                t = _test(rows, mode, c, nv)
                na = _nanmean([r["modes"][mode]["n_actual"][c][str(nv)] for r in rows])
                sh = _nanmean([r["modes"][mode]["share"][c][str(nv)] for r in rows])
                L.append(f"| {c} | {na:.1f} | {_f(sh)} | {_f(t['clean_known_flagged'])} | {_f(t['clean_found'])} | {_f(t['contaminated_known_flagged'])} | {_f(t['contaminated_found'])} | {_f(t['retention'])} |")
            L.append("")
        if d != "arithmetic" and "real" in rows[0]["modes"]:
            L += [f"### {d}, the price of trimming on a clean sample (nothing wrongly verified)", "", "| n | calibrator | theory (k+1)/(n+1) | asking reader | vMF | pair: known flagged | pair: found |", "|---|---|---|---|---|---|---|"]
            for n_ in sorted({int(x) for x in rows[0]["modes"]["real"]["results"]["plain"]}):
                for k, c in CAL_FOR_K.items():
                    g = lambda s, key: _g(rows, "real", c, n_, s, "clean", key)
                    L.append(f"| {n_} | {c} | {(k + 1) / (n_ + 1):.3f} | {_f(g('ask_exemplars', 'known_flagged'))} | {_f(g('vmf_conformal', 'known_flagged'))} | "
                             f"{_f(g('either_pair', 'known_flagged'))} | {_f(g('either_pair', 'unknown_found'))} |")
            L.append("")
        if "inj10" in rows[0]["modes"]:
            L += [f"### {d}, retention under 10 % injected contamination as the sample grows", "", "| n | plain | trim1 | trim3 | trim6 | agree |", "|---|---|---|---|---|---|"]
            for n_ in sorted({int(x) for x in rows[0]["modes"]["inj10"]["results"]["plain"]}):
                L.append(f"| {n_} | " + " | ".join(_f(_test(rows, "inj10", c, n_)["retention"]) for c in ("plain", "trim1", "trim3", "trim6", "agree")) + " |")
            L.append("")
    L += ["## Verdicts (rules fixed before the run)", ""]
    if "arithmetic" in per_domain and "real" in per_domain["arithmetic"][0]["modes"]:
        v11 = c11(per_domain["arithmetic"], nv)
        for c, x in v11["cells"].items():
            L.append(f"- C11, {c}: clean known flagged {_f(x['clean_known_flagged'])}, contaminated {_f(x['contaminated_known_flagged'])} (<= {KNOWN_LIMIT}); found {_f(x['contaminated_found'])} against "
                     f"{_f(x['clean_found'])} clean, retention {_f(x['retention'])} (need >= {RETENTION}). **{'PASS' if x['pass'] else 'FAIL'}**")
        L.append(f"- C11 overall (plain keeps {_f(v11['plain_retention'])}): **{'PASS' if v11['pass'] else 'FAIL'}**")
    v12 = c12(per_domain)
    if v12["cells"]:
        L.append("- C12 (single readers, clean, text domains): " + "; ".join(f"n={k.split(',')[0]} k={k.split(',')[1]}: {x['mean']:.3f} vs {x['theory']:.3f}" for k, x in v12["cells"].items())
                 + f". **{'PASS' if v12['pass'] else 'FAIL'}** (tolerance {THEORY_TOL})")
    else:
        L.append("- C12: no text domain in this report.")
    if any("inj10" in rows[0]["modes"] for rows in per_domain.values()):
        v13 = c13(per_domain, nv)
        for c, per in v13["cells"].items():
            if any(per.values()):
                L.append(f"- C13, {c}: passes the C11 test in {', '.join(d for d, ok in per.items() if ok)}; fails in {', '.join(d for d, ok in per.items() if not ok) or 'none'}")
        L.append(f"- C13 overall (one calibrator in every domain under inj10): **{'PASS' if v13['pass'] else 'FAIL'}**")
    L.append("")
    return "\n".join(L)


def run_seed(domain: str, seed: int, out_dir, stamp: str) -> dict:
    path = Path(out_dir) / f"{domain}_s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world, foreign, verifier, _ = LO.build(domain, seed)
    rec = run_cell(world, foreign, verifier, seed=seed)
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
            p = r["modes"]["inj10"]["results"]["plain"][str(HEADLINE)]["either_pair"]
            print(f"{domain} seed {seed}: plain inj10 known flagged {p['clean']['known_flagged']:.3f} found clean {p['clean']['unknown_found']:.3f} contaminated {p['contaminated']['unknown_found']:.3f}", flush=True)
    text = report(per_domain, list(range(lo, hi + 1)))
    runlib.atomic_write_text(_KS / "results" / f"ROUTING_GENERAL_G4c_{lo}_{hi}.md", text)
    print(text)
