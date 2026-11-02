"""Layer R: the recognizer lab. For one seed, one domain and one space (raw frontend vectors, or the learned invariance map on top of them) it fits the detectors on teaching rows only and measures,
on rows it never fitted: how often a known skill in a held-out wording is called new (the nuance error), how many withheld skills are found (near-twins and far), the cut-free AUROC, the group test
and the cost, under two calibrations of the same conformal rule:
* ``teach``  the cut from a held-out part of the TEACHING rows (what a system knows at the end of teaching: only the teaching wordings);
* ``stream`` the cut from n = 40 known inputs drawn from the stream's own wordings (what verified outcomes reveal; the outside anchor says they really are known), averaged over many draws.
Evaluation rows are split once into a calibration half and a test half (alternate rows within every skill x wording cell), so the cut never sees a test row.
``python -m fsmoe.recog_lab run --seeds 700`` writes one JSON per (seed, domain, space) into results/fsmoe/R/ (resumable); ``report`` builds the tables.
"""

import argparse
import json
import zlib
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from .bed import ROOT, Bed, build_bed
from .detectors import Knn, Maha, Vmf, auroc, conformal_cut, effective_rate
from .domains import NAMES
from .fastslow import FastSlow
from .frontend import Frontend
from .invariance import InvarianceMap

Q, N_CAL, DRAWS, GROUPS, D_FRONT, MODE = 0.05, 40, 30, (8, 16), 128, "pca"          # the frontend (PCA, 128 per reader) was chosen on the dev pilot, results/fsmoe/pilot_700.txt
SPACES = ("raw", "map")
FAR_FOR = {"arith": ("stories", "wiki", "science", "agnews", "newsgroups", "emotion"), "code": ("stdlib_code", "wiki", "stories"),
           "lang": ("stories", "wiki", "science", "agnews", "newsgroups", "emotion")}
OUT = ROOT / "results" / "fsmoe" / "R"
SIGNALS = ("knn", "maha", "vmf", "disagree", "conf")
FUSIONS = {"either(maha,disagree)": ("maha", "disagree"), "either(maha,knn)": ("maha", "knn"), "either(maha,conf)": ("maha", "conf"), "both(maha,disagree)": ("maha", "disagree")}


def _nan_mean(a: np.ndarray, m: np.ndarray) -> float:
    return float(a[m].mean()) if m.any() else float("nan")


def _parity(bed: Bed, ev: np.ndarray) -> np.ndarray:
    """True for the test half: alternate rows inside every (skill, family) cell."""
    count: Dict[tuple, int] = {}
    out = np.zeros(len(ev), dtype=bool)
    for j, i in enumerate(ev):
        key = (bed.skill[i], bed.family[i])
        c = count.get(key, 0)
        out[j] = bool(c % 2)
        count[key] = c + 1
    return out


class Lab:
    """Everything fitted for one (bed, domain, space)."""

    def __init__(self, bed: Bed, domain: str, space: str, seed: int, d: int = D_FRONT, mode: str = MODE) -> None:
        self.bed, self.domain, self.space, self.seed = bed, domain, space, seed
        tr_all = bed.teaching()
        self.fe = Frontend(d, mode, 0).fit(bed.raw[tr_all], np.ones(int(tr_all.sum()), dtype=bool))
        self.y = torch.from_numpy(bed.y)
        teach = np.flatnonzero(bed.teaching(domain))
        self.map: Optional[InvarianceMap] = None
        X = self.fe(bed.raw)
        if space == "map":
            idx = {s: i for i, s in enumerate(bed.skills)}
            twin = {idx[s]: idx[t] for s, t in bed.twin.items()}
            self.map = InvarianceMap(X.shape[1], 64, seed=seed).fit(X[teach], self.y[teach], twin, np.ones(len(teach), dtype=bool))
            X = self.map(X)
        self.X = X
        rng = np.random.default_rng([seed, 0x4C4142, zlib.crc32(f"{domain}{space}".encode())])
        perm = rng.permutation(teach)
        n_fit = int(0.7 * len(perm))
        self.fit_idx, self.calT_idx = perm[:n_fit], perm[n_fit:]
        nc = len(bed.skills)
        self.dets = {"knn": Knn(), "maha": Maha(), "vmf": Vmf()}
        for det in self.dets.values():
            det.fit(X[self.fit_idx], self.y[self.fit_idx])
        self.fs = FastSlow(X.shape[1], nc, seed=seed).fit(X[self.fit_idx], self.y[self.fit_idx])

    def scores(self, X: torch.Tensor) -> Dict[str, np.ndarray]:
        out = {n: det.score(X) for n, det in self.dets.items()}
        out["disagree"] = self.fs.disagreement(X)
        out["conf"] = self.fs.confidence(X)
        return out

    def far_scores(self) -> Dict[str, np.ndarray]:
        """Per signal, the scores of every far text relevant to this domain (concatenated)."""
        parts: Dict[str, List[np.ndarray]] = {s: [] for s in SIGNALS}
        for name in FAR_FOR[self.domain]:
            raw = self.bed.far[name][1]
            X = self.fe(raw)
            X = self.map(X) if self.map is not None else X
            for s, v in self.scores(X).items():
                parts[s].append(v)
        return {s: np.concatenate(v) for s, v in parts.items()}

    def cost(self) -> Dict[str, int]:
        m = self.fs.macs()
        return {"knn": self.dets["knn"].macs, "maha": self.dets["maha"].macs, "vmf": self.dets["vmf"].macs, "disagree": m["disagreement"], "conf": m["confidence"]}


def measure(lab: Lab) -> dict:
    bed, dom, seed = lab.bed, lab.domain, lab.seed
    ev = np.flatnonzero(bed.mask(domain=dom, split="eval"))
    test = _parity(bed, ev)
    sc = lab.scores(lab.X[ev])
    calT = lab.scores(lab.X[lab.calT_idx])
    far = lab.far_scores()
    taught, held = bed.taught[ev], bed.heldout[ev]
    far_set = np.asarray([s in bed.far_withheld for s in bed.skill[ev]])
    withheld = ~taught
    cats = {"known_tw": taught & ~held & test, "known_ho": taught & held & test, "novel_near": withheld & ~far_set & test, "novel_far": withheld & far_set & test}
    cal_known = np.flatnonzero(taught & ~test)
    rng = np.random.default_rng([seed, 0x434144, zlib.crc32(dom.encode())])
    draws = [rng.choice(cal_known, N_CAL, replace=False) for _ in range(DRAWS)]

    def rates(s: str, cut: float) -> dict:
        r = {k: _nan_mean(sc[s] > cut, m) for k, m in cats.items()}
        r["far_text"] = float((far[s] > cut).mean())
        return r

    res: dict = {"seed": seed, "domain": dom, "space": lab.space, "n": {k: int(m.sum()) for k, m in cats.items()}, "n_far_text": int(len(far["knn"])), "cost": lab.cost(),
                 "effective_stream_rate": effective_rate(N_CAL, Q), "signals": {}, "fusion": {}, "group": {}}
    for s in SIGNALS:
        sig = {"auroc": {"ho_vs_near": auroc(sc[s][cats["known_ho"]], sc[s][cats["novel_near"]]), "ho_vs_far": auroc(sc[s][cats["known_ho"]], sc[s][cats["novel_far"]]),
                         "ho_vs_far_text": auroc(sc[s][cats["known_ho"]], far[s]), "tw_vs_ho": auroc(sc[s][cats["known_tw"]], sc[s][cats["known_ho"]])}}
        sig["teach"] = rates(s, conformal_cut(calT[s], Q))
        acc = [rates(s, conformal_cut(sc[s][d], Q)) for d in draws]
        sig["stream"] = {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}
        res["signals"][s] = sig
    for name, parts in FUSIONS.items():
        mode = "both" if name.startswith("both") else "either"
        outs = []
        for d in draws:
            cuts = {s: conformal_cut(sc[s][d], Q) for s in parts}
            flag = lambda S, f: np.logical_and.reduce([S[s] > cuts[s] for s in parts]) if mode == "both" else np.logical_or.reduce([S[s] > cuts[s] for s in parts])
            fl = flag(sc, None)
            r = {k: _nan_mean(fl, m) for k, m in cats.items()}
            r["far_text"] = float(flag(far, None).mean())
            outs.append(r)
        res["fusion"][name] = {k: float(np.nanmean([o[k] for o in outs])) for k in outs[0]}
    # ---- the group test: the mean novelty score of g inputs from one source (one skill in one wording)
    cell_of = np.asarray([f"{a}|{b}" for a, b in zip(bed.skill[ev], bed.family[ev])])
    cells = sorted(set(cell_of))
    for g in GROUPS:
        out_g = {}
        for s in ("knn", "maha", "disagree"):
            cal_cells = [c for c in cells if taught[cell_of == c][0]]
            calg = []
            for _ in range(300):
                c = cal_cells[int(rng.integers(0, len(cal_cells)))]
                pool = np.flatnonzero((cell_of == c) & ~test)
                if len(pool):
                    calg.append(sc[s][rng.choice(pool, g, replace=len(pool) < g)].mean())
            cut = conformal_cut(np.asarray(calg), Q)
            hit = {k: [] for k in cats}
            for c in cells:
                pool = np.flatnonzero((cell_of == c) & test)
                if not len(pool):
                    continue
                for k, m in cats.items():
                    if m[pool[0]]:
                        for _ in range(4):
                            hit[k].append(sc[s][rng.choice(pool, g, replace=len(pool) < g)].mean() > cut)
            out_g[s] = {k: (float(np.mean(v)) if v else float("nan")) for k, v in hit.items()}
        res["group"][str(g)] = out_g
    p = lab.fs.p(lab.X[ev]).argmax(1).numpy()
    res["accuracy"] = {k: _nan_mean(p == bed.y[ev], cats[k]) for k in ("known_tw", "known_ho")}
    return res


def run(seeds: List[int], spaces=SPACES, domains=NAMES, out: Path = OUT) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        bed = None
        for dom in domains:
            for sp in spaces:
                path = out / f"R_s{seed}_{dom}_{sp}.json"
                if path.exists():
                    continue
                bed = bed or build_bed(seed)
                res = measure(Lab(bed, dom, sp, seed))
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps(res), encoding="utf-8")
                tmp.replace(path)
                print(f"done {path.name}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[700])
    a = ap.parse_args()
    run(a.seeds)
