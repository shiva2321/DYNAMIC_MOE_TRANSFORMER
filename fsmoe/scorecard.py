"""Scores every prediction declared in the design (sections 8, 8b, 8c) against the generated results of the report seeds.
``python -m fsmoe.scorecard --r-seeds 710 ... --s-seeds 710 ... --t-seeds 710 ...`` writes results/fsmoe/SCORECARD.md. Each prediction is a predicate on seed means; its measured value is printed next to it.
"""

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .bed import ROOT
from .domains import NAMES
from .recog_lab import FUSIONS, OUT as R_OUT, SIGNALS
from .system_lab import OUT as S_OUT
from .traces import OUT as T_OUT

LOOK = ("knn", "maha", "vmf")


class Data:
    def __init__(self, r_seeds: Sequence[int], s_seeds: Sequence[int], t_seeds: Sequence[int]) -> None:
        self.R = {(d, sp): [json.loads(p.read_text(encoding="utf-8")) for s in r_seeds if (p := R_OUT / f"R_s{s}_{d}_{sp}.json").exists()] for d in NAMES for sp in ("raw", "map")}
        self.S = [json.loads(p.read_text(encoding="utf-8")) for s in s_seeds if (p := S_OUT / f"S_s{s}.json").exists()]
        self.T = {(div, n): [json.loads(p.read_text(encoding="utf-8")) for s in t_seeds if (p := T_OUT / f"T_rec_s{s}_div{div}_n{n}.json").exists()] for div in (2, 6) for n in (0.05, 0.3, 1.0)}
        self.TD = [json.loads(p.read_text(encoding="utf-8")) for s in t_seeds if (p := T_OUT / f"T_depth_s{s}.json").exists()]

    def r(self, d: str, sp: str, s: str, cal: str, k: str) -> float:
        v = [run["signals"][s][cal][k] for run in self.R[(d, sp)]]
        return float(np.nanmean(v)) if v else float("nan")

    def r_auroc(self, d: str, sp: str, s: str, k: str) -> float:
        v = [run["signals"][s]["auroc"][k] for run in self.R[(d, sp)]]
        return float(np.nanmean(v)) if v else float("nan")

    def r_group(self, d: str, sp: str, g: str, s: str, k: str) -> float:
        v = [run["group"][g][s][k] for run in self.R[(d, sp)]]
        return float(np.nanmean(v)) if v else float("nan")

    def r_fusion(self, d: str, sp: str, f: str, k: str) -> float:
        v = [run["fusion"][f][k] for run in self.R[(d, sp)]]
        return float(np.nanmean(v)) if v else float("nan")

    def s_mean(self, f: Callable[[dict], float]) -> float:
        return float(np.nanmean([f(r) for r in self.S])) if self.S else float("nan")

    def s_deploy(self, policy: str, budget: float, key: str) -> float:
        v = [d[key] for r in self.S for d in r["deploy"] if d["policy"] == policy and abs(d["budget"] - budget) < 1e-9]
        return float(np.nanmean(v)) if v else float("nan")

    def t(self, div: int, noise: float, s: str, f: Callable[[dict], float]) -> float:
        runs = self.T[(div, noise)]
        return float(np.nanmean([f(r["signals"][s]) for r in runs])) if runs else float("nan")


def _fmt(*vals) -> str:
    return ", ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in vals)


def score(D: Data) -> List[Tuple[str, str, bool, str]]:
    out: List[Tuple[str, str, bool, str]] = []
    # ---- Layer R
    v = [D.r(d, "raw", s, "teach", "known_ho") for d in ("arith", "code") for s in LOOK]
    out.append(("R1", "raw look-based detectors, naive calibration: false flag on known skills in held-out wordings >= 0.20 in arithmetic and code", min(v) >= 0.20, f"min {min(v):.3f}"))
    v = [D.r(d, "map", s, "teach", "known_ho") for d in NAMES for s in LOOK]
    out.append(("R2", "the learned map alone does not bring that below 0.15 in any domain", min(v) >= 0.15, f"min {min(v):.3f}"))
    ff = [D.r(d, sp, s, "stream", "known_ho") for d in NAMES for sp in ("raw", "map") for s in LOOK]
    pw = [max(D.r(d, "raw", s, "stream", "novel_near") for s in SIGNALS) for d in NAMES]
    out.append(("R3", "stream calibration: false flag within 0.04-0.08 in all domains AND best power for withheld (near) skills 0.3-0.7", min(ff) >= 0.04 and max(ff) <= 0.08 and min(pw) >= 0.3 and max(pw) <= 0.7,
                f"false flag {min(ff):.3f}-{max(ff):.3f}; best near power per domain {_fmt(*pw)}"))
    ok, gaps = True, []
    for d in NAMES:
        dis = D.r_auroc(d, "raw", "disagree", "ho_vs_near")
        best = max(D.r_auroc(d, "raw", s, "ho_vs_near") for s in LOOK)
        gaps.append((dis, best))
        ok &= dis >= 0.80 and dis <= best + 0.03
    out.append(("R4", "fast-slow disagreement alone: AUROC >= 0.80 in every domain and never more than 0.03 above the best look-based score", ok, "; ".join(f"{d} {a:.3f} vs {b:.3f}" for d, (a, b) in zip(NAMES, gaps))))
    near = [max(D.r(d, sp, s, "stream", "novel_near") for s in SIGNALS) for d in ("code", "lang") for sp in ("raw", "map")]
    far = [max(D.r(d, "raw", s, "stream", "novel_far") for s in SIGNALS) for d in ("code", "lang")]
    out.append(("R5", "near-twin withheld skills found <= 0.50 by every single signal (code, language) AND far skills >= 0.90 by the best signal", max(near) <= 0.5 and min(far) >= 0.9,
                f"best near {max(near):.3f}; best far code {far[0]:.3f}, language {far[1]:.3f}"))
    gfar = [max(D.r_group(d, "raw", "16", s, "novel_far") for s in ("knn", "maha", "disagree")) for d in ("code", "lang")]
    gfa = [min(D.r_group(d, "raw", "16", s, "known_ho") for s in ("knn", "maha", "disagree")) for d in ("code", "lang")]
    gnear = [max(D.r_group(d, "raw", "16", s, "novel_near") for s in ("knn", "maha", "disagree")) for d in ("code", "lang")]
    out.append(("R6", "group test g = 16: far withheld >= 0.90 at false alarm <= 0.06 and near-twins <= 0.60 (code, language)", min(gfar) >= 0.9 and max(gfa) <= 0.06 and max(gnear) <= 0.6,
                f"far {_fmt(*gfar)}; lowest false alarm {_fmt(*gfa)}; near {_fmt(*gnear)}"))
    a = [D.r(d, "raw", "maha", "stream", "known_ho") for d in ("arith", "code")]
    b = [D.r(d, "raw", "maha", "teach", "known_ho") for d in ("arith", "code")]
    out.append(("R7", "calibration on verified outcomes controls the false flag better than teaching-wording calibration in arithmetic and code, and lands within 0.02 of q", all(x < y for x, y in zip(a, b)) and max(a) <= 0.07,
                f"stream {_fmt(*a)} vs teach {_fmt(*b)}"))
    bars = []
    for d in NAMES:
        passed = 0
        for s in SIGNALS:
            far_ = D.r(d, "raw", s, "stream", "novel_far")
            far_t = D.r(d, "raw", s, "stream", "far_text")
            far_ = far_t if far_ != far_ else (far_ + far_t) / 2
            passed += int(D.r(d, "raw", s, "stream", "known_ho") <= 0.05 and far_ >= 0.9 and D.r(d, "raw", s, "stream", "novel_near") >= 0.7)
        for f in FUSIONS:
            far_ = D.r_fusion(d, "raw", f, "novel_far")
            far_t = D.r_fusion(d, "raw", f, "far_text")
            far_ = far_t if far_ != far_ else (far_ + far_t) / 2
            passed += int(D.r_fusion(d, "raw", f, "known_ho") <= 0.05 and far_ >= 0.9 and D.r_fusion(d, "raw", f, "novel_near") >= 0.7)
        bars.append(passed)
    out.append(("R8", "the declared bars fail in at least two of three domains for every single signal and the best fusion", sum(1 for p in bars if p == 0) >= 2, f"signals/fusions passing the bars per domain: {dict(zip(NAMES, bars))}"))
    # ---- Layer S
    gap_tw = D.s_mean(lambda r: r["teaching"]["joint"]["mean_tw"] - r["teaching"]["FS"]["mean_tw"])
    gap_ho = D.s_mean(lambda r: r["teaching"]["joint"]["mean_ho"] - r["teaching"]["FS"]["mean_ho"])
    forg = D.s_mean(lambda r: max(r["teaching"]["FS"]["forgetting"].values()))
    tie = [D.s_mean(lambda r, n=n: r["teaching"]["FS"]["mean_tw"] - r["teaching"][n]["mean_tw"]) for n in ("replay_eqmem", "knn_eqmem")]
    nexp = D.s_mean(lambda r: r["teaching"]["FS"]["n_experts"])
    out.append(("S1", "the LDA-blended pool ends within 0.03 of the joint network on taught skills (teaching wording) and 0.05 on held-out wordings, forgetting <= 0.03, within +-0.03 of equal-memory replay and kNN, 3-12 experts",
                gap_tw <= 0.03 and gap_ho <= 0.05 and forg <= 0.03 and all(abs(t) <= 0.03 for t in tie) and 3 <= nexp <= 12,
                f"joint-FS gap tw {gap_tw:+.3f}, ho {gap_ho:+.3f}; worst forgetting {forg:.3f}; FS minus replay/kNN tw {_fmt(*tie)}; experts {nexp:.1f}"))
    prec_n, prec_r = D.s_deploy("novelty", 0.05, "precision_novel"), D.s_deploy("random", 0.05, "precision_novel")
    ln, lr = D.s_deploy("novelty", 0.05, "learned_fraction"), D.s_deploy("random", 0.05, "learned_fraction")
    out.append(("S2", "5 % budget: novelty-ranked precision >= 0.28 (twice the novel share), learned fraction >= 0.70 under novelty and <= 0.40 under random; none learns zero",
                prec_n >= 0.28 and ln >= 0.7 and lr <= 0.4 and D.s_deploy("none", 0.0, "learned_fraction") == 0.0, f"precision novelty {prec_n:.3f} vs random {prec_r:.3f}; learned fraction novelty {ln:.3f}, random {lr:.3f}, oracle {D.s_deploy('oracle', 0.05, 'learned_fraction'):.3f}"))
    d_top = D.s_mean(lambda r: r["team"]["team_ho"] - r["team"]["top1_ho"])
    size = D.s_mean(lambda r: r["team"]["mean_size"])
    out.append(("S3", "adaptive team within +-0.02 of top-1 on held-out wordings, mean size <= 1.3", abs(d_top) <= 0.02 and size <= 1.3, f"team minus top-1 {d_top:+.4f}; mean size {size:.2f}"))
    rises = {p: D.s_deploy(p, b, "known_ho_final") - D.s_deploy(p, b, "known_ho_before") for p, b in (("random", 0.05), ("novelty", 0.05), ("oracle", 0.05))}
    out.append(("S4", "held-out-wording accuracy of known skills RISES during deployment under every policy that verifies known items, more under novelty-ranked than random",
                rises["random"] > 0 and rises["novelty"] > rises["random"], "rise random {:+.3f}, novelty {:+.3f}, oracle {:+.3f} (oracle verifies no known item)".format(rises["random"], rises["novelty"], rises["oracle"])))
    # ---- Layer T
    p05 = D.t(6, 0.05, "result_all", lambda s: (s["flag"]["withheld_tw"] + s["flag"]["withheld_ho"]) / 2)
    p03 = D.t(6, 0.3, "result_all", lambda s: (s["flag"]["withheld_tw"] + s["flag"]["withheld_ho"]) / 2)
    p10 = D.t(6, 1.0, "result_all", lambda s: (s["flag"]["withheld_tw"] + s["flag"]["withheld_ho"]) / 2)
    a05 = D.t(6, 0.05, "result_all", lambda s: s["auroc_known_ho_vs_withheld"])
    lk = max(D.t(div, n, "look", lambda s: (s["flag"]["withheld_tw"] + s["flag"]["withheld_ho"]) / 2) for div in (2, 6) for n in (0.05, 0.3, 1.0))
    out.append(("T1", "result signal finds withheld regimes >= 0.90 / >= 0.70 / <= 0.50 at noise 0.05 / 0.3 / 1.0 (6 teaching surfaces), AUROC >= 0.98 at 0.05, look finds <= 0.15",
                p05 >= 0.9 and p03 >= 0.7 and p10 <= 0.5 and a05 >= 0.98 and lk <= 0.15, f"found {p05:.3f} / {p03:.3f} / {p10:.3f}; AUROC at 0.05 {a05:.3f}; look at most {lk:.3f}"))
    cost = [D.T[(6, 0.05)][0]["cost_macs"]["result_nearest"] / D.T[(6, 0.05)][0]["cost_macs"]["result_all"]] if D.T[(6, 0.05)] else [float("nan")]
    near_p = D.t(6, 0.3, "result_nearest", lambda s: (s["flag"]["withheld_tw"] + s["flag"]["withheld_ho"]) / 2)
    all_p = p03
    out.append(("T2", "the cascade (look shortlists one expert, result judges it) costs <= 0.4 of result_all and loses <= 0.05 of its power", cost[0] <= 0.4 and all_p - near_p <= 0.05, f"cost ratio {cost[0]:.2f}; power cascade {near_p:.3f} vs result_all {all_p:.3f} (noise 0.3)"))
    if D.TD:
        def grown(case, key="grown"):
            return [r["cases"][case][key] for r in D.TD if case in r["cases"]]
        def kinds(case, kind, key="grown"):
            return float(np.mean([sum(1 for e in g["events"] if e["kind"] == kind and e["gain_after"] >= 0.03) for g in grown(case, key)]))
        widen_all = all(kinds(c, "widen") > 0 for c in ("simple_shift", "simple_mix", "composed_3", "composed_5"))
        deep_comp = all(kinds(c, "deepen") > 0 for c in ("composed_3", "composed_5"))
        deep_simple = any(kinds(c, "deepen") > 0 for c in ("simple_shift", "simple_mix"))
        ratio = []
        for c in ("composed_3", "composed_5"):
            g = grown(c)
            best = np.mean([min(f["final_mse"] for f in r["cases"][c]["fixed"]) for r in D.TD if c in r["cases"]])
            ratio.append(float(np.mean([x["final_mse"] for x in g])) / best)
        out.append(("T3", "the stuck-then-grow rule widens on every case, deepens at least once on the composed cases and never on the simple ones, ends within 1.5x of the best fixed net's error (composed cases)",
                    widen_all and deep_comp and not deep_simple and max(ratio) <= 1.5,
                    f"widen on all four: {widen_all}; deepen on both composed: {deep_comp}; deepen on a simple one: {deep_simple}; error ratio to best fixed {_fmt(*ratio)} (declared threshold 0.2)"))
    return out


def build(r_seeds, s_seeds, t_seeds) -> str:
    rows = score(Data(r_seeds, s_seeds, t_seeds))
    held = sum(1 for r in rows if r[2])
    L = [f"# FS-MoE prediction scorecard (report seeds R {list(r_seeds)}, S {list(s_seeds)}, T {list(t_seeds)})", "",
         f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**. Each is a predicate on seed means, written in the design before the cells that test it (docs/superpowers/specs/2026-10-31-fsmoe-design.md, sections 8-8c).", "",
         "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    for pid, text, ok, val in rows:
        L.append(f"| {pid} | {text} | {'HELD' if ok else 'MISSED'} | {val} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-seeds", type=int, nargs="+", default=list(range(710, 720)))
    ap.add_argument("--s-seeds", type=int, nargs="+", default=list(range(710, 715)))
    ap.add_argument("--t-seeds", type=int, nargs="+", default=list(range(710, 715)))
    a = ap.parse_args()
    text = build(a.r_seeds, a.s_seeds, a.t_seeds)
    (ROOT / "results" / "fsmoe" / "SCORECARD.md").write_text(text, encoding="utf-8")
    print(text)
