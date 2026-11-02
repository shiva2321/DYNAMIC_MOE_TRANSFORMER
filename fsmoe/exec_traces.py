"""Layer E: execution outcomes on real code (design: docs/superpowers/specs/2026-10-31-fsmoe-followup-design.md, part 1).

A function f on a list has a prefix outcome o_k = f(x[:k]); the step (o_{k-1}, x_k) -> o_k is what the code does to its state when one more element arrives, and it is observed by RUNNING the code, whatever
its style (loop, builtin, recursion, renamed, formatted, boilerplate, a one-liner from the standard library). One small network per taught family predicts o_k from (o_{k-1}, x_k), trained on executions
of the TEACHING styles only. A snippet is judged by the prediction errors of its executions in units of each expert's own teaching error. Signals: look (the Mahalanobis distance of the snippet text, the
Layer R signal), result_all (the smallest normalized error over the experts), cascade (the look names one expert and only its error counts) and the behaviour fingerprint (outputs on fixed probe inputs
against the taught families' reference outputs: a ceiling that learns nothing).
``python -m fsmoe.exec_traces run --seeds 710 ...`` writes results/fsmoe/E/E_s<seed>.json; ``python -m fsmoe.report_e`` builds the tables.
"""

import argparse
import contextlib
import io
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from unimoe.heads import LdaHead

from .bed import ROOT, _encode, build_bed, default_readers
from .detectors import Maha, auroc, conformal_cut
from .domains import code as CO
from .domains.base import rng_for
from .system_lab import Prep

OUT = ROOT / "results" / "fsmoe" / "E"
TAUGHT = ("sum_list", "max_list", "min_list", "count_even", "count_positive")
NEAR = ("product_list", "count_odd")
FAR_LIB = ("length", "median", "value_range", "second_largest", "mean", "last")
LO, HI = -3, 6
NV = HI - LO + 1
MS = (1, 3, 10)
Q, N_CAL, DRAWS = 0.05, 40, 30

LIB: Dict[str, List[str]] = {
    "sum_list": ["def {f}(x):\n    return sum(x)\n", "import math\ndef {f}(x):\n    return math.fsum(x)\n", "import functools, operator\ndef {f}(x):\n    return functools.reduce(operator.add, x)\n",
                 "def {f}(x):\n    return sum(map(int, x))\n"],
    "max_list": ["def {f}(x):\n    return max(x)\n", "def {f}(x):\n    return sorted(x)[-1]\n", "import functools\ndef {f}(x):\n    return functools.reduce(max, x)\n",
                 "import heapq\ndef {f}(x):\n    return heapq.nlargest(1, x)[0]\n"],
    "min_list": ["def {f}(x):\n    return min(x)\n", "def {f}(x):\n    return sorted(x)[0]\n", "import functools\ndef {f}(x):\n    return functools.reduce(min, x)\n",
                 "import heapq\ndef {f}(x):\n    return heapq.nsmallest(1, x)[0]\n"],
    "count_even": ["def {f}(x):\n    return len([v for v in x if not v & 1])\n", "def {f}(x):\n    return sum(v % 2 == 0 for v in x)\n", "def {f}(x):\n    return len(list(filter(lambda v: v % 2 == 0, x)))\n",
                   "def {f}(x):\n    return sum(1 for v in x if v % 2 == 0)\n"],
    "count_positive": ["def {f}(x):\n    return len([v for v in x if v > 0])\n", "def {f}(x):\n    return sum(v > 0 for v in x)\n", "def {f}(x):\n    return len(list(filter(lambda v: v > 0, x)))\n",
                       "def {f}(x):\n    return sum(1 for v in x if v > 0)\n"],
    "product_list": ["import math\ndef {f}(x):\n    return math.prod(x)\n", "import functools, operator\ndef {f}(x):\n    return functools.reduce(operator.mul, x)\n"],
    "count_odd": ["def {f}(x):\n    return len([v for v in x if v & 1])\n", "def {f}(x):\n    return sum(v % 2 for v in x)\n", "def {f}(x):\n    return sum(1 for v in x if v % 2 == 1)\n"],
    "length": ["def {f}(x):\n    return len(x)\n", "def {f}(x):\n    return sum(1 for _ in x)\n"],
    "median": ["import statistics\ndef {f}(x):\n    return statistics.median(x)\n",
               "def {f}(x):\n    s = sorted(x)\n    n = len(s)\n    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2\n"],
    "value_range": ["def {f}(x):\n    return max(x) - min(x)\n", "def {f}(x):\n    s = sorted(x)\n    return s[-1] - s[0]\n"],
    "second_largest": ["def {f}(x):\n    return sorted(x)[-2]\n", "import heapq\ndef {f}(x):\n    return heapq.nlargest(2, x)[1]\n"],
    "mean": ["def {f}(x):\n    return sum(x) / len(x)\n", "import statistics\ndef {f}(x):\n    return statistics.fmean(x)\n"],
    "last": ["def {f}(x):\n    return x[-1]\n", "def {f}(x):\n    return x[len(x) - 1]\n"],
}


# ---------------------------------------------------------------------------------------------------------------------------------- running code
def load_fn(code: str, fn: str):
    ns = {"__name__": "fsmoe_exec"}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(code, "<snippet>", "exec"), ns)
    return ns[fn]


def prefix_outcomes(fn, lists: Sequence[Sequence[int]]) -> Optional[List[List[float]]]:
    """o_k = f(x[:k]) for k = 2..n of every list; None if the code fails on any of them."""
    out = []
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            for x in lists:
                out.append([float(fn(list(x[:k]))) for k in range(2, len(x) + 1)])
    except Exception:
        return None
    return out


def steps(lists: Sequence[Sequence[int]], outs: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The transitions (o_{k-1}, x_k, o_k) for k = 3..n of every list."""
    prev, el, nxt = [], [], []
    for x, o in zip(lists, outs):
        for k in range(3, len(x) + 1):
            prev.append(o[k - 3])
            el.append(x[k - 1])
            nxt.append(o[k - 2])
    return np.asarray(prev), np.asarray(el), np.asarray(nxt)


def draw_lists(rng: np.random.Generator, n: int) -> List[List[int]]:
    return [[int(v) for v in rng.integers(LO, HI + 1, int(rng.integers(5, 9)))] for _ in range(n)]


def feat(prev: np.ndarray, el: np.ndarray) -> torch.Tensor:
    oh = np.zeros((len(el), NV), dtype=np.float32)
    oh[np.arange(len(el)), np.clip(el - LO, 0, NV - 1)] = 1.0
    return torch.from_numpy(np.concatenate([(prev / 10.0)[:, None].astype(np.float32), oh], axis=1))


class StepExpert:
    """Predicts the next outcome from (previous outcome, new element); floor = its mean squared error on its own teaching steps, so a score of about 1 means 'as good as on what it was taught'."""

    def __init__(self, seed: int, hidden: int = 32, epochs: int = 80, lr: float = 3e-3) -> None:
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(1 + NV, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.3 / np.sqrt(m.in_features), generator=g)
                nn.init.zeros_(m.bias)
        self.epochs, self.lr, self.seed, self.floor = epochs, lr, seed, 1.0

    def fit(self, prev: np.ndarray, el: np.ndarray, nxt: np.ndarray) -> "StepExpert":
        X, Y = feat(prev, el), torch.from_numpy((nxt / 10.0).astype(np.float32))[:, None]
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            order = torch.randperm(len(X), generator=g)
            for lo in range(0, len(X), 128):
                idx = order[lo:lo + 128]
                loss = ((self.net(X[idx]) - Y[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.floor = float(self.err(prev, el, nxt).mean()) + 1e-3
        return self

    @torch.no_grad()
    def err(self, prev: np.ndarray, el: np.ndarray, nxt: np.ndarray) -> np.ndarray:
        return ((self.net(feat(prev, el))[:, 0] - torch.from_numpy((nxt / 10.0).astype(np.float32))) ** 2).numpy()

    def score(self, prev: np.ndarray, el: np.ndarray, nxt: np.ndarray) -> float:
        return float(self.err(prev, el, nxt).mean() / self.floor)


# ---------------------------------------------------------------------------------------------------------------------------------- snippets
class Snip:
    def __init__(self, text: str, fn: str, family: str, role: str, kind: str) -> None:
        self.text, self.fn, self.family, self.role, self.kind = text, fn, family, role, kind        # role: known_tw / known_ho / near / far; kind: template / library


def _names(rng: np.random.Generator, k: int) -> List[str]:
    pool = CO.NEUTRAL["f"] + ["total", "calc_value", "summary", "helper_one", "reduce_all"]
    return [str(pool[int(i)]) for i in rng.permutation(len(pool))[:k]]


def build_snippets(seed: int, per_style: int = 4, names_per_lib: int = 3) -> List[Snip]:
    rng = rng_for(seed, "exec-snips")
    out: List[Snip] = []
    for fam in TAUGHT + NEAR:
        for style in CO.TEACH + CO.HELDOUT:
            for _ in range(per_style):
                text, fn = CO.render(fam, style, rng)
                role = ("known_tw" if style in CO.TEACH else "known_ho") if fam in TAUGHT else "near"
                out.append(Snip(text, fn, fam, role, "template"))
        for src in LIB[fam]:
            for nm in _names(rng, names_per_lib):
                out.append(Snip(src.format(f=nm), nm, fam, "known_ho" if fam in TAUGHT else "near", "library"))
    for fam in FAR_LIB:
        for src in LIB[fam]:
            for nm in _names(rng, names_per_lib):
                out.append(Snip(src.format(f=nm), nm, fam, "far", "library"))
    return out


def teaching_steps(seed: int, family: str, n_lists: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Steps of executions of the TEACHING styles (loop, builtin, recursion) of one taught family, on lists that no evaluation uses."""
    rng = rng_for(seed, "exec-train", family)
    lists = draw_lists(rng, n_lists)
    parts = []
    for style in CO.TEACH:
        for _ in range(4):
            text, fn = CO.render(family, style, rng)
            outs = prefix_outcomes(load_fn(text, fn), lists)
            assert outs is not None, (family, style)
            parts.append(steps(lists, outs))
    return tuple(np.concatenate([p[i] for p in parts]) for i in range(3))


# ---------------------------------------------------------------------------------------------------------------------------------- behaviour fingerprint (a ceiling)
def _ref_outputs(skill: str) -> np.ndarray:
    return np.asarray([float(CO._ref(skill, list(p))) for p in CO.PROBES["list"]])


def fingerprint_distance(fn, refs: Dict[str, np.ndarray]) -> float:
    """Smallest normalized distance between the snippet's outputs on the fixed probe lists and a taught family's reference outputs; 1e6 when the code fails."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            o = np.asarray([float(fn(list(p))) for p in CO.PROBES["list"]])
    except Exception:
        return 1e6
    return float(min(np.abs(o - r).sum() / (np.abs(r).sum() + 1.0) for r in refs.values()))


# ---------------------------------------------------------------------------------------------------------------------------------- the experiment
def run_seed(seed: int) -> dict:
    snips = build_snippets(seed)
    rng = rng_for(seed, "exec-eval-lists")
    lists = draw_lists(rng, max(MS))
    experts = {f: StepExpert(seed * 31 + i).fit(*teaching_steps(seed, f)) for i, f in enumerate(TAUGHT)}
    refs = {f: _ref_outputs(f) for f in TAUGHT}
    rows, per_snip = [], []
    for s in snips:
        try:
            fn = load_fn(s.text, s.fn)
        except Exception:
            continue
        outs = prefix_outcomes(fn, lists)
        if outs is None:
            continue
        rows.append(s)
        per_snip.append((outs, fingerprint_distance(fn, refs)))
    # the look: the same readers and frontend as the main experiments
    bed = build_bed(seed)
    prep = Prep(bed)
    raw = _encode(default_readers(), [s.text for s in rows], f"exec_s{seed}")
    Xs = prep.fe(raw)
    teach = np.flatnonzero(bed.teaching("code"))
    look = Maha().fit(prep.X[teach], prep.y[teach]).score(Xs)
    lda = LdaHead(prep.d, prep.n_classes, 0.1)
    lda.update(prep.X[teach], prep.y[teach])
    ids = [bed.skills.index(f"code:{f}") for f in TAUGHT]
    short = lda.proba(Xs)[:, ids].argmax(1).numpy()
    role = np.asarray([s.role for s in rows])
    kind = np.asarray([s.kind for s in rows])
    res: dict = {"seed": seed, "n": {r: int((role == r).sum()) for r in ("known_tw", "known_ho", "near", "far")},
                 "n_library": {r: int(((role == r) & (kind == "library")).sum()) for r in ("known_ho", "near", "far")}, "by_m": {}}
    known = np.flatnonzero((role == "known_tw") | (role == "known_ho"))
    cal_rng = np.random.default_rng([seed, 0x455843])
    draws = [cal_rng.choice(known, N_CAL, replace=False) for _ in range(DRAWS)]
    cats = {"known_ho_template": (role == "known_ho") & (kind == "template"), "known_ho_library": (role == "known_ho") & (kind == "library"), "known_tw": role == "known_tw",
            "near_template": (role == "near") & (kind == "template"), "near_library": (role == "near") & (kind == "library"), "far": role == "far"}
    ho, near, far = role == "known_ho", role == "near", role == "far"
    for M in MS:
        sig = {"look": look, "fingerprint": np.asarray([p[1] for p in per_snip])}
        ra, ca = np.zeros(len(rows)), np.zeros(len(rows))
        for i, (outs, _) in enumerate(per_snip):
            tr = steps(lists[:M], outs[:M])
            sc = [experts[f].score(*tr) for f in TAUGHT]
            ra[i], ca[i] = min(sc), sc[int(short[i])]
        sig["result_all"], sig["cascade"] = ra, ca
        cell = {}
        for name, s in sig.items():
            rates = {k: float(np.mean([(s[m] > conformal_cut(s[d], Q)).mean() for d in draws])) for k, m in cats.items()}
            cell[name] = {"auroc_near": auroc(s[ho], s[near]), "auroc_far": auroc(s[ho], s[far]), "auroc_tw_vs_ho": auroc(s[role == "known_tw"], s[ho]), "flag": rates}
        res["by_m"][str(M)] = cell
    res["steps_per_snippet"] = {str(M): int(len(steps(lists[:M], per_snip[0][0][:M])[0])) for M in MS}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.seeds:
        p = OUT / f"E_s{s}.json"
        if not p.exists():
            r = run_seed(s)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(r), encoding="utf-8")
            tmp.replace(p)
            print("done", p.name, r["n"], flush=True)
