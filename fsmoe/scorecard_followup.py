"""Scores the follow-up predictions (design: docs/superpowers/specs/2026-10-31-fsmoe-followup-design.md, E1-E6 and L0-L4) against the generated results of the report seeds.
``python -m fsmoe.scorecard_followup --seeds 710 ...`` writes results/fsmoe/SCORECARD_FOLLOWUP.md."""

import argparse
import json
from typing import List, Sequence, Tuple

import numpy as np

from .bed import ROOT
from .exec_traces import OUT as E_OUT
from .learn_lab import OUT as L_OUT


def build(seeds: Sequence[int]) -> str:
    E = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := E_OUT / f"E_s{s}.json").exists()]
    Lr = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := L_OUT / f"L_s{s}.json").exists()]
    rows: List[Tuple[str, str, bool, str]] = []
    mean = lambda v: float(np.mean(v))
    e = lambda M, sig, k: mean([r["by_m"][str(M)][sig][k] for r in E])
    ef = lambda M, sig, k: mean([r["by_m"][str(M)][sig]["flag"][k] for r in E])
    if E:
        ra, lk = e(10, "result_all", "auroc_near"), e(10, "look", "auroc_near")
        rows.append(("E1", "at M = 10 result_all separates known-in-held-out-styles from near-twins at AUROC >= 0.95 while the look stays <= 0.85", ra >= 0.95 and lk <= 0.85, f"result_all {ra:.3f}; look {lk:.3f}"))
        fl = ef(10, "result_all", "known_ho_library")
        rows.append(("E2", "real library idioms of known functions are flagged <= 0.10 at the stream cut (result_all)", fl <= 0.10, f"{fl:.3f} (look {ef(10, 'look', 'known_ho_library'):.3f})"))
        fo = ef(10, "result_all", "far")
        rows.append(("E3", "result_all finds >= 0.90 of the withheld real library functions", fo >= 0.9, f"{fo:.3f}"))
        gap = e(10, "result_all", "auroc_near") - e(10, "cascade", "auroc_near")
        rows.append(("E4", "the cascade is within 0.05 AUROC of result_all on near-twins", gap <= 0.05, f"gap {gap:.3f} (cascade {e(10, 'cascade', 'auroc_near'):.3f})"))
        a1 = e(1, "result_all", "auroc_near")
        rows.append(("E5", "at M = 1 result_all still reaches AUROC >= 0.85 on near-twins", a1 >= 0.85, f"{a1:.3f}"))
        fp = e(10, "fingerprint", "auroc_near")
        rows.append(("E6", "the behaviour fingerprint reaches AUROC 1.00", fp >= 0.999, f"{fp:.3f}"))
    if Lr:
        get = lambda policy, mode, key: [d[key] for r in Lr for d in r["deploy"] if d["policy"] == policy and d["mode"] == mode]
        fl_new = [d["flipped_to_new_skill"] / max(d["flipped"], 1) for r in Lr for d in r["deploy"] if d["mode"] == "interleaved"]
        rows.append(("L0", "of the known held-out-wording items that turn from right to wrong, at least half go to a newly learned skill (interleaved arms)", mean(fl_new) >= 0.5, f"{mean(fl_new):.3f}"))
        l1 = mean(get("diverse", "focused", "learned_fraction"))
        rows.append(("L1", "focused lifts the learned fraction under diverse 10 % from 0.28 to >= 0.50", l1 >= 0.5, f"{l1:.3f} (interleaved {mean(get('diverse', 'interleaved', 'learned_fraction')):.3f})"))
        l2a, l2b = mean(get("diverse", "focused_replay", "known_ho_final")), mean(get("diverse", "focused_replay", "learned_fraction"))
        rows.append(("L2", "focused_replay keeps known held-out-wording accuracy >= 0.85 while learning >= 0.40 (diverse 10 %)", l2a >= 0.85 and l2b >= 0.4, f"known ho after {l2a:.3f}; learned {l2b:.3f}"))
        l3 = mean(get("oracle", "focused_replay", "learned_fraction"))
        rows.append(("L3", "under the oracle at 10 %, focused_replay learns >= 0.70", l3 >= 0.7, f"{l3:.3f}"))
        tw = min(mean([d["known_tw_final"] for r in Lr for d in r["deploy"] if d["policy"] == p and d["mode"] == m]) for p in ("oracle", "diverse") for m in ("interleaved", "focused", "focused_replay"))
        rows.append(("L4", "known skills in taught wordings stay >= 0.95 in every arm", tw >= 0.95, f"lowest arm mean {tw:.3f}"))
    X = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := L_OUT / f"L_extra_s{s}.json").exists()]
    if X:
        gx = lambda mode, key: mean([d[key] for r in X for d in r["deploy"] if d["mode"] == mode])
        rows.append(("L5", "under balanced + focused_replay, known held-out-wording accuracy >= 0.85 while the learned fraction >= 0.40", gx("focused_replay", "known_ho_final") >= 0.85 and gx("focused_replay", "learned_fraction") >= 0.4,
                     f"known ho after {gx('focused_replay', 'known_ho_final'):.3f}; learned {gx('focused_replay', 'learned_fraction'):.3f}"))
        rows.append(("L6", "under balanced + interleaved, the learned fraction stays <= 0.30", gx("interleaved", "learned_fraction") <= 0.30, f"{gx('interleaved', 'learned_fraction'):.3f} (known ho after {gx('interleaved', 'known_ho_final'):.3f})"))
    Fa = [json.loads(p.read_text(encoding="utf-8")) for s in seeds if (p := L_OUT / f"L_fast_s{s}.json").exists()]
    if Fa:
        gf = lambda policy, key: mean([d[key] for r in Fa for d in r["deploy"] if d["policy"] == policy])
        rows.append(("L7", "global_fast + balanced: learned fraction >= 0.60 and known held-out-wording accuracy >= 0.85", gf("balanced", "learned_fraction") >= 0.6 and gf("balanced", "known_ho_final") >= 0.85,
                     f"learned {gf('balanced', 'learned_fraction'):.3f}; known ho after {gf('balanced', 'known_ho_final'):.3f}"))
        rows.append(("L8", "global_fast + diverse: learned fraction >= 0.60 and known held-out accuracy <= 0.82", gf("diverse", "learned_fraction") >= 0.6 and gf("diverse", "known_ho_final") <= 0.82,
                     f"learned {gf('diverse', 'learned_fraction'):.3f}; known ho after {gf('diverse', 'known_ho_final'):.3f}"))
        dtw = mean([r["teaching_mean_tw"] - r["pool_alone_mean_tw"] for r in Fa])
        dho = mean([r["teaching_mean_ho"] - r["pool_alone_mean_ho"] for r in Fa])
        rows.append(("L9", "after teaching only, global_fast reads taught skills within 0.01 of the pool alone (taught and held-out wordings)", abs(dtw) <= 0.01 and abs(dho) <= 0.01, f"difference tw {dtw:+.4f}, ho {dho:+.4f}"))
    held = sum(1 for r in rows if r[2])
    L = [f"# FS-MoE follow-up scorecard (seeds {list(seeds)})", "", f"{len(rows)} declared predictions: **{held} held, {len(rows) - held} missed**.", "", "| id | prediction | outcome | measured |", "|---|---|---|---|"]
    L += [f"| {i} | {t} | {'HELD' if ok else 'MISSED'} | {v} |" for i, t, ok, v in rows]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[710, 711, 712, 713, 714])
    a = ap.parse_args()
    text = build(a.seeds)
    (ROOT / "results" / "fsmoe" / "SCORECARD_FOLLOWUP.md").write_text(text, encoding="utf-8")
    print(text)
