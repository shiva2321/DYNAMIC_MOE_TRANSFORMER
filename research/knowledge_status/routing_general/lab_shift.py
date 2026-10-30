"""Lab G3: is the flag rate of a reader on REWRITES of known messages a usable diagnostic of how far to trust it under a wording shift? (spec section 6; PROGRESS.md 2026-10-30)

    PYTHONPATH=<repo root> python research/knowledge_status/routing_general/lab_shift.py [first_seed last_seed]

Per domain and seed: the held-out known messages are split per class into a half that sets every reader's cut (A) and a half that the same language model rewrites with generic
instructions (B, at most 30 per class). Each reader's flag rate on the rewrites at its cut is its SHIFT FLAG; its flag rate on the real known test is its TRUE known-flagged rate (in
arithmetic that test is a real wording shift, in the text domains it is not). A reader is trusted when its shift flag is at most 2q. The system is the "either" fusion of the trusted
readers of the pair (ask exemplars, vMF conformal = the pool's own gate). Everything is computed in one run on one split, so the systems are compared fairly.
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
import lab_general as LG  # noqa: E402
import measures as MS  # noqa: E402
import reader as RD  # noqa: E402
import rewrites as RW  # noqa: E402
import route as RT  # noqa: E402
import runlib  # noqa: E402
import shift as SH  # noqa: E402

OUT = _KS / "results" / "routing_general" / "g3"
RW_DIR = _KS / "results" / "routing_general" / "rewrites"
STAMP_FILES = [_HERE / n for n in ("shift.py", "lab_shift.py", "lab_general.py", "measures.py", "reader.py", "cards.py", "route.py", "domains.py", "arith_domain.py")] + [
    _KS / "baselines.py", _KS / "rewrites.py", _KS / "routing_lab" / "ask_reader.py"]
READERS = ("ask_exemplars", "vmf_conformal", "mahalanobis", "knn_distance")
PAIR = ("ask_exemplars", "vmf_conformal")
KEEPS: Dict[str, Callable[[str, str], bool]] = {"arithmetic": RW.keep}               # the one domain hook: numbers (digits and number words) must survive a rewrite of arithmetic
N_CAL_TEXT = 60
CAP = 30
Q = LG.Q
LIMIT = 2 * Q                                                                           # a reader is trusted when it flags at most this share of rewrites of known messages
TRUE_BAD, TRUE_GOOD = 0.15, 0.10                                                        # rule C6a: a reader flagging more than TRUE_BAD of the real known test must be untrusted
SEEDS = list(range(150, 153))


def fused_op(S: Dict[str, Dict[str, np.ndarray]], sources: Sequence[str], rule: str, q: float, shift_in_cal: bool = False) -> dict:
    cal_map = {p: (np.concatenate([S[p]["cal"], S[p]["shift"]]) if shift_in_cal else S[p]["cal"]) for p in sources}
    f = {k: MS.fused_scores(cal_map, {p: S[p][k] for p in sources}, list(sources), rule) for k in ("known", "unknown", "foreign")}
    f["cal"] = MS.fused_scores(cal_map, cal_map, list(sources), rule, leave_one_out=True)
    return MS.operating_point(f["cal"], f["known"], f["unknown"], f["foreign"], q)


def run_cell(world, foreign: Sequence[str], generate: Callable, ask: Optional[Callable] = None, encode: Callable = LG.default_encode, keep: Callable = SH.keep_general,
             q: float = Q, limit: float = LIMIT, cap: int = CAP, cache_dir=LG.ASK_DIR, rw_dir=RW_DIR, seed: int = 0, readers: Sequence[str] = READERS) -> dict:
    cal_a, cal_b = SH.split_calibration(world.known_cal)
    V = SH.shift_set(SH.cap_per_class(cal_b, cap), generate, rw_dir, seed, keep)
    if not V:
        raise RuntimeError("no rewrite was kept: the shift diagnostic has nothing to measure")
    known_t, truth = [t for t, _ in world.known_test], [i for _, i in world.known_test]
    sets = {"cal": [t for t, _ in cal_a], "known": known_t, "unknown": list(world.unknown_test), "foreign": list(foreign), "shift": [t for t, _ in V]}
    shift_y = np.asarray([i for _, i in V])
    S: Dict[str, Dict[str, np.ndarray]] = {}
    rd = RD.CardReader(world.cards, with_exemplars=True, cache_dir=cache_dir, ask=ask)
    P = {k: rd.probabilities(v) for k, v in sets.items()}
    S["ask_exemplars"] = {k: RT.unclaimed_score(P[k]) for k in sets}
    X, y = [], []
    for i, n in enumerate(world.taught):
        X += world.teach[n]
        y += [i] * len(world.teach[n])
    Xv, y = encode(X), np.asarray(y)
    V_ = {k: encode(v) for k, v in sets.items()}
    for name in (r for r in readers if r != "ask_exemplars"):
        det = BL.make_baseline(name).fit(Xv, y)
        S[name] = {k: det.novelty_score(V_[k]) for k in sets}
    clf = LogisticRegression(max_iter=1000).fit(Xv, y)
    rec: dict = {"domain": world.name, "n": {k: len(v) for k, v in sets.items()}, "readers": {}, "systems": {},
                 "ident": {"ask_known": float(np.mean(P["known"][:, :-1].argmax(axis=1) == np.asarray(truth))), "ask_shift": float(np.mean(P["shift"][:, :-1].argmax(axis=1) == shift_y)),
                           "lr_known": float((clf.predict(V_["known"]) == np.asarray(truth)).mean()), "lr_shift": float((clf.predict(V_["shift"]) == shift_y).mean())}}
    flags = {}
    for r in readers:
        op = MS.operating_point(S[r]["cal"], S[r]["known"], S[r]["unknown"], S[r]["foreign"], q)
        op["shift_flag"] = MS.flag_rate(S[r]["shift"], op["cut"])
        rec["readers"][r] = op
        flags[r] = op["shift_flag"]
    pair = [r for r in PAIR if r in readers]
    rec["trusted"] = {"pair": SH.trusted({r: flags[r] for r in pair}, limit), "all": SH.trusted({r: flags[r] for r in readers}, limit)}
    rec["systems"] = {"either_fixed": fused_op(S, pair, "either", q), "both_fixed": fused_op(S, pair, "both", q),
                      "trusted_either": fused_op(S, rec["trusted"]["pair"], "either", q), "trusted_either_calV": fused_op(S, rec["trusted"]["pair"], "either", q, shift_in_cal=True),
                      "trusted_all_either": fused_op(S, rec["trusted"]["all"], "either", q)}
    return rec


def _mean(rows: List[dict], pick: Callable[[dict], float]) -> float:
    return float(np.nanmean([pick(r) for r in rows]))


def verdicts(per_domain: Dict[str, List[dict]], limit: float = LIMIT) -> dict:
    """The two rules declared before the run, evaluated on the seed means.
    C6a (the diagnostic): across all (domain, reader) pairs the Spearman correlation of shift flag with the true known-flagged rate is at least 0.6, every pair with a true rate above 0.15
    has a shift flag above the limit, and at least 80 % of the pairs with a true rate of at most 0.10 have a shift flag at or under the limit.
    C6b (the system): in EVERY domain the trusted-either system flags at most 0.15 of the real known test and finds at least (the best never-taught found among the readers that flag at most 0.15) - 0.05."""
    pairs = [(d, r, _mean(rows, lambda x: x["readers"][r]["shift_flag"]), _mean(rows, lambda x: x["readers"][r]["known_flagged"])) for d, rows in per_domain.items() for r in rows[0]["readers"]]
    rho = SH.spearman([p[2] for p in pairs], [p[3] for p in pairs])
    bad = [p for p in pairs if p[3] > TRUE_BAD]
    good = [p for p in pairs if p[3] <= TRUE_GOOD]
    bad_ok = all(p[2] > limit for p in bad)
    good_share = (sum(p[2] <= limit for p in good) / len(good)) if good else float("nan")
    a = {"rho": rho, "bad_untrusted": bad_ok, "n_bad": len(bad), "good_trusted_share": good_share, "n_good": len(good), "pass": bool(rho >= 0.6 and bad_ok and good_share >= 0.8)}
    b: Dict[str, dict] = {}
    for d, rows in per_domain.items():
        names = list(rows[0]["readers"])
        usable = [r for r in names if _mean(rows, lambda x: x["readers"][r]["known_flagged"]) <= TRUE_BAD]
        best = max((_mean(rows, lambda x: x["readers"][r]["unknown_found"]) for r in usable), default=float("nan"))
        kf = _mean(rows, lambda x: x["systems"]["trusted_either"]["known_flagged"])
        uf = _mean(rows, lambda x: x["systems"]["trusted_either"]["unknown_found"])
        b[d] = {"known_flagged": kf, "unknown_found": uf, "best_usable_found": best, "pass": bool(kf <= TRUE_BAD and (np.isnan(best) or uf >= best - 0.05))}
    return {"C6a": a, "C6b": b, "C6b_pass": all(v["pass"] for v in b.values())}


def report(per_domain: Dict[str, List[dict]], seeds: Sequence[int], limit: float = LIMIT) -> str:
    L = [f"# Lab G3, rewrites of known messages as a diagnostic of trust (seeds {seeds[0]}-{seeds[-1]})", "",
         f"Shift flag = share of rewrites of held-out known messages that a reader flags at its own cut (5 % of the other half of the held-out messages). True known-flagged = the same reader on the real "
         f"known test (arithmetic: new wordings; text domains: no shift). A reader is trusted when its shift flag is at most {limit}.", ""]
    for d, rows in per_domain.items():
        n = rows[0]["n"]
        L += [f"## {d} ({len(rows)} seeds; per seed {n['cal']} cut messages, {n['shift']} rewrites, {n['known']} known, {n['unknown']} never-taught, {n['foreign']} foreign)", "",
              "| reader | shift flag | TRUE known flagged | never-taught found | foreign found | AUROC known vs never-taught | trusted |", "|---|---|---|---|---|---|---|"]
        for r in rows[0]["readers"]:
            g = lambda k: _mean(rows, lambda x: x["readers"][r][k])
            sf = g("shift_flag")
            L.append(f"| {r} | {sf:.3f} | {g('known_flagged'):.3f} | {g('unknown_found'):.3f} | {g('foreign_found'):.3f} | {g('auroc_unknown'):.3f} | {'yes' if sf <= limit else 'no'} |")
        L += ["", "| system | known flagged | never-taught found | foreign found | AUROC known vs never-taught |", "|---|---|---|---|---|"]
        for s in rows[0]["systems"]:
            g = lambda k: _mean(rows, lambda x: x["systems"][s][k])
            L.append(f"| {s} | {g('known_flagged'):.3f} | {g('unknown_found'):.3f} | {g('foreign_found'):.3f} | {g('auroc_unknown'):.3f} |")
        counts: Dict[str, int] = {}
        for x in rows:
            counts[" + ".join(x["trusted"]["pair"])] = counts.get(" + ".join(x["trusted"]["pair"]), 0) + 1
        L += ["", f"Trusted pair readers per seed: {counts}. Who is it? top-1 on known / on rewrites: ask {_mean(rows, lambda x: x['ident']['ask_known']):.3f} / {_mean(rows, lambda x: x['ident']['ask_shift']):.3f}; "
              f"logistic regression on MiniLM {_mean(rows, lambda x: x['ident']['lr_known']):.3f} / {_mean(rows, lambda x: x['ident']['lr_shift']):.3f}.", ""]
    v = verdicts(per_domain, limit)
    a = v["C6a"]
    L += ["## Verdicts (rules fixed before the run)", "",
          f"- C6a (the diagnostic): Spearman {a['rho']:.3f} (>= 0.6); readers with a true known-flagged rate above {TRUE_BAD}: {a['n_bad']}, all untrusted: {a['bad_untrusted']}; readers at or under {TRUE_GOOD}: {a['n_good']}, "
          f"share trusted {a['good_trusted_share']:.2f} (>= 0.80). **{'PASS' if a['pass'] else 'FAIL'}**"]
    for d, x in v["C6b"].items():
        L.append(f"- C6b, {d}: trusted-either flags {x['known_flagged']:.3f} of known (<= {TRUE_BAD}) and finds {x['unknown_found']:.3f} (best usable reader {x['best_usable_found']:.3f}, need >= that - 0.05). **{'PASS' if x['pass'] else 'FAIL'}**")
    L += [f"- C6b overall (all four domains): **{'PASS' if v['C6b_pass'] else 'FAIL'}**", ""]
    return "\n".join(L)


def run_seed(domain: str, seed: int, out_dir, stamp: str, generate: Callable) -> dict:
    path = Path(out_dir) / f"{domain}_s{seed}.json"
    if runlib.cell_done(path, stamp):
        return json.loads(path.read_text(encoding="utf-8"))
    world, foreign, _ = LG.build_world(domain, seed, N_CAL_TEXT)
    rec = run_cell(world, foreign, generate, keep=KEEPS.get(domain, SH.keep_general), seed=seed)
    rec.update({"seed": seed, "code_stamp": stamp})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    runlib.atomic_write_text(path, json.dumps(LG._native(rec)))
    return rec


if __name__ == "__main__":
    lo, hi = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (SEEDS[0], SEEDS[-1])
    runlib.set_threads(3)
    stamp = runlib.code_stamp(STAMP_FILES)
    generate = RW.make_qwen_generator()
    per_domain: Dict[str, List[dict]] = {d: [] for d in LG.ALL_DOMAINS}
    for domain in LG.ALL_DOMAINS:
        for seed in range(lo, hi + 1):
            r = run_seed(domain, seed, OUT, stamp, generate)
            per_domain[domain].append(r)
            print(f"{domain} seed {seed}: rewrites {r['n']['shift']}; shift flag " + ", ".join(f"{k} {v['shift_flag']:.2f}" for k, v in r["readers"].items()) + f"; trusted {r['trusted']['pair']}", flush=True)
    text = report(per_domain, list(range(lo, hi + 1)))
    runlib.atomic_write_text(_KS / "results" / f"ROUTING_GENERAL_G3_{lo}_{hi}.md", text)
    print(text)
