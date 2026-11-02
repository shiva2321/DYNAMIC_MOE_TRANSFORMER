"""End to end: one stream over the three domains with everything put together (design: docs/superpowers/specs/2026-10-31-fsmoe-e2e-design.md).

The system is the pool with the closed-form head in every expert plus the global closed-form head (``GlobalFast``); verified items enter as interleaved chunks; what to verify is chosen by diversity-ranked
novelty. Verification has CHANNELS: code-looking inputs go to the tool (``fsmoe.sandbox``: run the function, read its behaviour; a taught family's behaviour gives that family's label, a new behaviour
opens a class with an id of the system's own, no human involved); every other input can only be checked by a human label, from a budget. The system never sees truth: for the classes the tool opens,
evaluation maps each id to the majority true skill of its items (the "cluster" reading).
``python -m fsmoe.e2e run --seeds 710 ...`` writes results/fsmoe/X/X_s<seed>.json.
"""

import argparse
import copy
import json
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from unimoe.systems2 import make_system2

from .bed import ROOT, build_bed
from .balance import BalancedLearner, DomainHead
from .detectors import Maha
from .domains import NAMES
from .domains import code as CO
from .domains.base import rng_for
from .learn_lab import GlobalFast, VerifiedLearner
from .sandbox import behaviour, find_function
from .system_lab import N_FAR, N_NOVEL, OUT as S_OUT, Prep, accuracy, build_stream, diverse_pick, teach, zlib_policy

OUT = ROOT / "results" / "fsmoe" / "X"
N_SLOTS = 20                                                                           # class slots on top of the bed's skills, for the ids the tool opens
ARMS = (("A0", dict(head=False, human=0.10, tool=None)), ("A1", dict(head=True, human=0.10, tool=None)), ("A2", dict(head=True, human=0.10, tool=0.10)),
        ("A3", dict(head=True, human=0.10, tool=1.0)), ("A4", dict(head=True, human=0.20, tool=1.0)))


class ToolChannel:
    """Runs code-looking texts in the sandbox. ``verify`` returns ("none",), ("known", global id of the taught family) or ("new", behaviour)."""

    def __init__(self, skills: Sequence[str], taught_code: Sequence[str]) -> None:
        rng = rng_for(0, "tool-refs")
        self.refs: Dict[tuple, int] = {}
        for s in taught_code:
            sigs = {behaviour(CO.render(s, style, rng)[0])[0] for style in CO.TEACH}
            if len(sigs) != 1 or None in sigs:
                raise AssertionError(f"the taught family {s} has no single behaviour: {sigs}")
            self.refs[next(iter(sigs))] = skills.index(f"code:{s}")
        self.calls = 0
        self.ran = 0

    def verify(self, text: str) -> Tuple[str, Optional[object], bool]:
        self.calls += 1
        sig, ran = behaviour(text)
        self.ran += int(ran)
        if sig is None:
            return "none", None, ran
        if sig in self.refs:
            return "known", self.refs[sig], ran
        return "new", sig, ran


class Clusters:
    """Ids the system gives to behaviours it has not met in teaching; ``mapping`` (evaluation only) gives each id the majority true skill of the items that received it (-1 = far text)."""

    def __init__(self, base: int, slots: int = N_SLOTS) -> None:
        self.base, self.slots = base, slots
        self.ids: Dict[tuple, int] = {}
        self.truth: Dict[int, Counter] = {}

    def assign(self, sig: tuple, true_global: int) -> Optional[int]:
        if sig not in self.ids:
            if len(self.ids) >= self.slots:
                return None
            self.ids[sig] = self.base + len(self.ids)
        cid = self.ids[sig]
        self.truth.setdefault(cid, Counter())[true_global] += 1
        return cid

    def mapping(self) -> Dict[int, int]:
        return {cid: c.most_common(1)[0][0] for cid, c in self.truth.items()}

    def purity(self) -> float:
        tot = sum(sum(c.values()) for c in self.truth.values())
        return float(sum(c.most_common(1)[0][1] for c in self.truth.values()) / tot) if tot else float("nan")


def build_stream_texts(prep: Prep, seed: int):
    """The stream of E-055 (same seeds, same order) with the text of every item, so the tool can read it. The system uses the text only through the tool and the vector only through the pool."""
    bed = prep.bed
    rng = np.random.default_rng([seed, 0x53545245])
    known = np.concatenate([prep.rows("test", None, "known_tw"), prep.rows("test", None, "known_ho")])
    nov_all = prep.rows("test", None, "novel")
    novel = rng.choice(nov_all, min(N_NOVEL, len(nov_all)), replace=False)
    far_texts = [t for _, v in sorted(bed.far.items()) for t in v[0]]
    far_i = rng.choice(len(prep.far_X), min(N_FAR, len(prep.far_X)), replace=False)
    X = torch.cat([prep.X[known], prep.X[novel], prep.far_X[far_i]])
    y = torch.cat([prep.y[known], prep.y[novel], torch.full((len(far_i),), -1, dtype=torch.long)])
    kind = np.asarray(["known"] * len(known) + ["novel"] * len(novel) + ["far"] * len(far_i))
    texts = [bed.text[i] for i in known] + [bed.text[i] for i in novel] + [far_texts[i] for i in far_i]
    order = rng.permutation(len(y))
    return X[order], y[order], kind[order], [texts[i] for i in order]


def evaluate(pool, prep: Prep, clusters: Clusters) -> dict:
    bed = prep.bed
    mapping = clusters.mapping()

    def preds(rows: np.ndarray) -> torch.Tensor:
        p = pool.predict(prep.X[rows]).probs.argmax(1).clone()
        for cid, t in mapping.items():
            p[p == cid] = t
        return p

    dom_of = lambda s: bed.skills[s].split(":")[0]
    cal_nov = prep.rows("cal", None, "novel")
    p, truth = preds(cal_nov), prep.y[cal_nov]
    per_skill = {int(s): float((p[truth == s] == s).float().mean()) for s in sorted({int(v) for v in truth.tolist()})}
    res = {"learned": {"all": float(np.mean([a >= 0.5 for a in per_skill.values()]))}, "n_withheld": {"all": len(per_skill)}, "novel_acc": {"all": float(np.mean(list(per_skill.values())))}}
    for d in NAMES:
        accs = [a for s, a in per_skill.items() if dom_of(s) == d]
        res["learned"][d], res["n_withheld"][d], res["novel_acc"][d] = float(np.mean([a >= 0.5 for a in accs])), len(accs), float(np.mean(accs))
    for kind in ("known_tw", "known_ho"):
        res[kind] = {}
        for d in (None,) + NAMES:
            rows = prep.rows("cal", d, kind)
            res[kind][d or "all"] = float((preds(rows) == prep.y[rows]).float().mean())
    return res


def deploy_e2e(pool, prep: Prep, seed: int, human: float, tool: Optional[float], tool_channel: ToolChannel, clusters: Clusters, texts_stream, known_rate: float = 1.0,
               balance: Optional[str] = None, domain_head: Optional[DomainHead] = None, new_cap: int = 8) -> dict:
    X, y, kind, texts = texts_stream
    bed = prep.bed
    rng = np.random.default_rng([seed, 0x453245, zlib_policy("e2e", human)])
    rate_rng = np.random.default_rng([seed, 0x4B52])                                          # only used when known_rate < 1, so the first five arms are unchanged
    dom_of = lambda s: bed.skills[s].split(":")[0]
    teach_rows = np.flatnonzero(bed.teaching())
    scorer = Maha().fit(prep.X[teach_rows], prep.y[teach_rows])
    if balance:
        learner = BalancedLearner(pool.observe, balance, {int(i) for i in np.unique(prep.y[teach_rows].numpy())}, np.random.default_rng([seed, 0x42414C]), new_cap=new_cap)
    else:
        learner = VerifiedLearner(pool.observe, "interleaved", set(), prep.X[teach_rows[:1]], prep.y[teach_rows[:1]], rng)
    code_dom = NAMES.index("code")
    head_right = [0, 0]

    def emit(xv: torch.Tensor, label: int, dom: int) -> None:
        learner.add(xv, label, dom) if balance else learner.add(xv, label)
    is_code = np.asarray([find_function(t) is not None for t in texts])
    before = evaluate(pool, prep, clusters)
    stats = {"human_checks": 0, "human_labels": 0, "tool_calls": 0, "tool_ran_far_noncode": 0, "tool_ran_far_code": 0, "tool_known": 0, "tool_new": 0, "tool_none": 0, "tool_slot_overflow": 0,
             "tool_known_trained": 0, "human_by_domain": {"arith": 0, "code": 0, "lang": 0}}
    n = len(y)
    for lo in range(0, n, 64):
        sl = slice(lo, min(lo + 64, n))
        x, yy, kk, code = X[sl], y[sl], kind[sl], is_code[sl]
        pool.predict(x)
        score = scorer.score(x)
        human_pick, tool_pick = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        if tool is None:
            human_pick = diverse_pick(score, x, int(round(human * len(x))))
        else:
            code_idx, non_idx = np.flatnonzero(code), np.flatnonzero(~code)
            if len(non_idx):
                human_pick = non_idx[diverse_pick(score[non_idx], x[non_idx], int(round(human * len(non_idx))))]
            if len(code_idx):
                tool_pick = code_idx if tool >= 1.0 else code_idx[diverse_pick(score[code_idx], x[code_idx], int(round(tool * len(code_idx))))]
        for j in human_pick:
            stats["human_checks"] += 1
            if int(yy[j]) >= 0:
                stats["human_labels"] += 1
                stats["human_by_domain"][dom_of(int(yy[j]))] += 1
                dom = 0
                if balance:
                    dom = int(domain_head(x[j])[0])
                    head_right[0] += int(dom == NAMES.index(dom_of(int(yy[j]))))
                    head_right[1] += 1
                emit(x[j], int(yy[j]), dom)
        for j in tool_pick:
            status, payload, ran = tool_channel.verify(texts[lo + int(j)])
            stats["tool_calls"] += 1
            if ran and kk[j] == "far":
                stats["tool_ran_far_code" if code[j] else "tool_ran_far_noncode"] += 1
            if status == "known":
                stats["tool_known"] += 1
                if known_rate >= 1.0 or rate_rng.random() < known_rate:
                    stats["tool_known_trained"] += 1
                    emit(x[j], int(payload), code_dom)
            elif status == "new":
                cid = clusters.assign(payload, int(yy[j]))
                if cid is None:
                    stats["tool_slot_overflow"] += 1
                    continue
                stats["tool_new"] += 1
                emit(x[j], cid, code_dom)
            else:
                stats["tool_none"] += 1
    learner.finish()
    after = evaluate(pool, prep, clusters)
    far_code = int(((kind == "far") & is_code).sum())
    extra = {}
    if balance:
        extra = {"trained_by_domain": {NAMES[d]: int(v) for d, v in learner.used.items()}, "new_trained_by_domain": {NAMES[d]: int(v) for d, v in learner.used_new.items()}, "chunks": learner.chunks,
                 "domain_head_accuracy": head_right[0] / head_right[1] if head_right[1] else float("nan")}
    return {"before": before, "after": after, **stats, **extra, "clusters_opened": len(clusters.ids), "cluster_purity": clusters.purity(), "far_code_in_stream": far_code,
            "stream_items": int(n), "code_items": int(is_code.sum()), "n_experts": int(pool.summary().get("n_experts", 1))}


def run_seed(seed: int, tuning: dict) -> dict:
    path = OUT / f"X_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    total = prep.n_classes + N_SLOTS
    texts_stream = build_stream_texts(prep, seed)
    y_ref = build_stream(prep, seed)[1]
    assert torch.equal(texts_stream[1], y_ref), "the end-to-end stream must be the stream of E-055"
    taught_code = [s for s in bed.domains["code"].taught]
    res = {"seed": seed, "arms": {}}
    alone = make_system2("ULDA", total, seed, tuning["FS_best"], d=prep.d)
    t0 = teach(alone, prep, seed)
    head_pool = GlobalFast(make_system2("ULDA", total, seed, tuning["FS_best"], d=prep.d), prep.d, total)
    t1 = teach(head_pool, prep, seed)
    res["teaching"] = {"pool_alone": {"mean_tw": t0["mean_tw"], "mean_ho": t0["mean_ho"]}, "pool_global_head": {"mean_tw": t1["mean_tw"], "mean_ho": t1["mean_ho"]}}
    for name, cfg in ARMS:
        pool = copy.deepcopy(head_pool if cfg["head"] else alone)
        tool_channel = ToolChannel(bed.skills, taught_code)
        r = deploy_e2e(pool, prep, seed, cfg["human"], cfg["tool"], tool_channel, Clusters(prep.n_classes), texts_stream)
        r["arm"], r["config"] = name, cfg
        res["arms"][name] = r
        a = r["after"]
        print(f"seed {seed} {name}: learned all {a['learned']['all']:.2f} (arith {a['learned']['arith']:.2f} code {a['learned']['code']:.2f} lang {a['learned']['lang']:.2f}) known_ho {a['known_ho']['all']:.3f} "
              f"human {r['human_labels']} tool {r['tool_calls']} clusters {r['clusters_opened']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


EXTRA_ARMS = (("A1r", dict(head=True, human=0.10, tool=None, known_rate=1.0)), ("A3r", dict(head=True, human=0.10, tool=1.0, known_rate=1.0)), ("A5", dict(head=True, human=0.10, tool=1.0, known_rate=0.1)))


def run_extra(seed: int, tuning: dict) -> dict:
    """A1 and A3 re-run with the counters that were missing (they must reproduce the first runs) and A5 (the tool's known items trained at 10 %). Writes X_extra_s<seed>.json."""
    path = OUT / f"X_extra_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    main = json.loads((OUT / f"X_s{seed}.json").read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    total = prep.n_classes + N_SLOTS
    texts_stream = build_stream_texts(prep, seed)
    taught_code = [s for s in bed.domains["code"].taught]
    head_pool = GlobalFast(make_system2("ULDA", total, seed, tuning["FS_best"], d=prep.d), prep.d, total)
    teach(head_pool, prep, seed)
    res = {"seed": seed, "arms": {}}
    for name, cfg in EXTRA_ARMS:
        r = deploy_e2e(copy.deepcopy(head_pool), prep, seed, cfg["human"], cfg["tool"], ToolChannel(bed.skills, taught_code), Clusters(prep.n_classes), texts_stream, cfg["known_rate"])
        r["arm"], r["config"] = name, cfg
        orig = main["arms"].get(name[:2])
        if orig is not None:
            r["reproduced"] = bool(abs(orig["after"]["learned"]["all"] - r["after"]["learned"]["all"]) < 1e-9 and abs(orig["after"]["known_ho"]["all"] - r["after"]["known_ho"]["all"]) < 1e-9)
        res["arms"][name] = r
        a = r["after"]
        print(f"seed {seed} {name}: learned all {a['learned']['all']:.2f} (arith {a['learned']['arith']:.2f} code {a['learned']['code']:.2f} lang {a['learned']['lang']:.2f}) known_ho {a['known_ho']['all']:.3f} "
              f"(code {a['known_ho']['code']:.3f}) human by domain {r['human_by_domain']} known trained {r['tool_known_trained']}", flush=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


BALANCE_ARMS = (("A3b", None), ("D1", "sub"), ("D2", "over"))
B_OUT = ROOT / "results" / "fsmoe" / "B"


def run_balance(seed: int, tuning: dict) -> dict:
    """The domain-balanced arms (design: docs/superpowers/specs/2026-10-31-fsmoe-balance-design.md): the E-057 A3 setting with arrival-order chunks (A3b, must reproduce A3), ``sub`` and ``over``."""
    path = B_OUT / f"B_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    main = json.loads((OUT / f"X_s{seed}.json").read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    total = prep.n_classes + N_SLOTS
    texts_stream = build_stream_texts(prep, seed)
    taught_code = [s for s in bed.domains["code"].taught]
    head_pool = GlobalFast(make_system2("ULDA", total, seed, tuning["FS_best"], d=prep.d), prep.d, total)
    teach(head_pool, prep, seed)
    teach_rows = np.flatnonzero(bed.teaching())
    dom_idx = torch.tensor([NAMES.index(d) for d in bed.domain[teach_rows]])
    dhead = DomainHead(prep.d, len(NAMES)).fit(prep.X[teach_rows], dom_idx)
    res = {"seed": seed, "arms": {}}
    for name, mode in BALANCE_ARMS:
        r = deploy_e2e(copy.deepcopy(head_pool), prep, seed, 0.10, 1.0, ToolChannel(bed.skills, taught_code), Clusters(prep.n_classes), texts_stream, 1.0, mode, dhead)
        r["arm"], r["balance"] = name, mode
        if mode is None:
            orig = main["arms"]["A3"]
            r["reproduced"] = bool(abs(orig["after"]["learned"]["all"] - r["after"]["learned"]["all"]) < 1e-9 and abs(orig["after"]["known_ho"]["all"] - r["after"]["known_ho"]["all"]) < 1e-9)
        res["arms"][name] = r
        a = r["after"]
        print(f"seed {seed} {name}: learned all {a['learned']['all']:.2f} (arith {a['learned']['arith']:.2f} code {a['learned']['code']:.2f} lang {a['learned']['lang']:.2f}) known_ho {a['known_ho']['all']:.3f} "
              f"(arith {a['known_ho']['arith']:.3f} code {a['known_ho']['code']:.3f} lang {a['known_ho']['lang']:.3f}) trained {r.get('trained_by_domain')} chunks {r.get('chunks')}", flush=True)
    B_OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


HYBRID_ARMS = (("D1r", "sub", 8), ("H4", "hybrid", 4), ("H8", "hybrid", 8), ("H12", "hybrid", 12))


def run_hybrid(seed: int, tuning: dict) -> dict:
    """The hybrid arms (design: docs/superpowers/specs/2026-10-31-fsmoe-hybrid-design.md): D1 again with the new counters (must reproduce D1) and ``hybrid`` with 4, 8 and 12 new-class items per chunk. Writes H_s<seed>.json."""
    path = B_OUT / f"H_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    total = prep.n_classes + N_SLOTS
    texts_stream = build_stream_texts(prep, seed)
    taught_code = [s for s in bed.domains["code"].taught]
    head_pool = GlobalFast(make_system2("ULDA", total, seed, tuning["FS_best"], d=prep.d), prep.d, total)
    teach(head_pool, prep, seed)
    teach_rows = np.flatnonzero(bed.teaching())
    dom_idx = torch.tensor([NAMES.index(d) for d in bed.domain[teach_rows]])
    dhead = DomainHead(prep.d, len(NAMES)).fit(prep.X[teach_rows], dom_idx)
    prev = json.loads((B_OUT / f"B_s{seed}.json").read_text(encoding="utf-8"))["arms"]["D1"] if (B_OUT / f"B_s{seed}.json").exists() else None
    res = {"seed": seed, "arms": {}}
    for name, mode, m in HYBRID_ARMS:
        r = deploy_e2e(copy.deepcopy(head_pool), prep, seed, 0.10, 1.0, ToolChannel(bed.skills, taught_code), Clusters(prep.n_classes), texts_stream, 1.0, mode, dhead, new_cap=m)
        r["arm"], r["balance"], r["new_cap"] = name, mode, m
        if name == "D1r" and prev is not None:
            r["reproduced"] = bool(abs(prev["after"]["learned"]["all"] - r["after"]["learned"]["all"]) < 1e-9 and abs(prev["after"]["known_ho"]["all"] - r["after"]["known_ho"]["all"]) < 1e-9)
        res["arms"][name] = r
        a = r["after"]
        print(f"seed {seed} {name}: learned all {a['learned']['all']:.2f} (arith {a['learned']['arith']:.2f} code {a['learned']['code']:.2f} lang {a['learned']['lang']:.2f}) known_ho {a['known_ho']['all']:.3f} "
              f"(arith {a['known_ho']['arith']:.3f} code {a['known_ho']['code']:.3f} lang {a['known_ho']['lang']:.3f}) trained {r.get('trained_by_domain')} new {r.get('new_trained_by_domain')} chunks {r.get('chunks')}", flush=True)
    B_OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


PRIOR_ARMS = (("A3u", None), ("D2u", "over"))


def run_priors(seed: int, tuning: dict) -> dict:
    """The uniform-prior arms (addendum of the balance design): the global head answers with the class counts removed from its logits. Writes B_priors_s<seed>.json."""
    path = B_OUT / f"B_priors_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    bed = build_bed(seed)
    prep = Prep(bed)
    total = prep.n_classes + N_SLOTS
    texts_stream = build_stream_texts(prep, seed)
    taught_code = [s for s in bed.domains["code"].taught]
    head_pool = GlobalFast(make_system2("ULDA", total, seed, tuning["FS_best"], d=prep.d), prep.d, total)
    t = teach(head_pool, prep, seed)
    uni = copy.deepcopy(head_pool)
    uni.uniform_priors = True
    teach_rows = np.flatnonzero(bed.teaching())
    dom_idx = torch.tensor([NAMES.index(d) for d in bed.domain[teach_rows]])
    dhead = DomainHead(prep.d, len(NAMES)).fit(prep.X[teach_rows], dom_idx)
    res = {"seed": seed, "teaching": {"prior_mean_tw": t["mean_tw"], "prior_mean_ho": t["mean_ho"], "uniform_mean_tw": float(np.mean([accuracy(uni, prep, prep.rows("test", d, "known_tw")) for d in NAMES])),
                                      "uniform_mean_ho": float(np.mean([accuracy(uni, prep, prep.rows("test", d, "known_ho")) for d in NAMES]))}, "arms": {}}
    for name, mode in PRIOR_ARMS:
        r = deploy_e2e(copy.deepcopy(uni), prep, seed, 0.10, 1.0, ToolChannel(bed.skills, taught_code), Clusters(prep.n_classes), texts_stream, 1.0, mode, dhead)
        r["arm"], r["balance"] = name, mode
        res["arms"][name] = r
        a = r["after"]
        print(f"seed {seed} {name}: learned all {a['learned']['all']:.2f} (arith {a['learned']['arith']:.2f} code {a['learned']['code']:.2f} lang {a['learned']['lang']:.2f}) known_ho {a['known_ho']['all']:.3f} "
              f"(arith {a['known_ho']['arith']:.3f} code {a['known_ho']['code']:.3f} lang {a['known_ho']['lang']:.3f})", flush=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(path)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "extra", "balance", "priors", "hybrid"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    a = ap.parse_args()
    tuning = json.loads((S_OUT / "tuning.json").read_text(encoding="utf-8"))
    for s in a.seeds:
        {"run": run_seed, "extra": run_extra, "balance": run_balance, "priors": run_priors, "hybrid": run_hybrid}[a.cmd](s, tuning)
