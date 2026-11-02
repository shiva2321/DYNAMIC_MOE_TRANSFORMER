"""Standard checks of growmoe's ``transfer_chance`` flag on the real UniMoE suite (the streams of E-054: seven real text domains in three schedules, one process per seed):
``python -m fsmoe.chance_check run --seeds 500`` writes cells to results/unimoe/p4_chance (the E-054 cells in p4_main are only read), ``python -m fsmoe.chance_check table`` prints the comparison.

Arms (recipes from results/unimoe tuning, as in `unimoe.p4.main_run`): ``ULDA-off`` / ``ULDA-on`` the shipped pool with a closed-form head in every expert, flag None / 1.0; ``ULDA-onw`` / ``h16-onw`` the weak-parent-only rule (``transfer_chance_weak_only``); ``h16-off`` / ``h16-on`` the same with every
expert at 16 units and growth off (the small-start control of E-054, where weak experts matter). With the flag off every cell must equal the stored E-054 cell of the same system."""

import argparse
import json
from pathlib import Path

import numpy as np

from unimoe import experiment as E
from unimoe import p4 as P4
from unimoe import systems2 as S2
from unimoe.runner import run_cell

OUT = E.OUT / "p4_chance"
SCHEDULES = ("seq", "mix", "drift")
ARMS = ("ULDA-off", "ULDA-on", "ULDA-onw", "h16-off", "h16-on", "h16-onw")


def recipes(sch: str) -> dict:
    base = dict(E.load_tuning(sch)["U-real"])
    lam = P4.load_tuning()[sch]["lda_blend"]
    ulda = {**base, "lda_blend": lam}
    small = {**base, "hidden": 16, "grow": False}
    return {"ULDA-off": ("ULDA", ulda), "ULDA-on": ("ULDA", {**ulda, "transfer_chance": 1.0}), "h16-off": ("UG0", small), "h16-on": ("UG0", {**small, "transfer_chance": 1.0}),
            "ULDA-onw": ("ULDA", {**ulda, "transfer_chance": 1.0, "transfer_chance_weak_only": True}), "h16-onw": ("UG0", {**small, "transfer_chance": 1.0, "transfer_chance_weak_only": True})}


def run_seed(seed: int) -> None:
    for sch in SCHEDULES:
        st = E.streams(seed)[sch]
        blocks = [tuple(v) for v in st.label_blocks.values()]
        for arm, (name, recipe) in recipes(sch).items():
            rec = run_cell(OUT, arm, lambda name=name, recipe=recipe: S2.make_system2(name, st.n_classes, seed, recipe, blocks), st, seed, P4.stamp2())
            ev = rec.per_phase[-1]["events"]
            print(f"seed {seed} {sch:<5s} {arm:<9s} final mean {E.final_mean(rec):.4f} experts {rec.per_phase[-1]['n_experts']:>3d} spawn {ev.get('spawn', 0)} extend {ev.get('extend', 0)} params {rec.report['params']:>8d} {rec.seconds:5.0f}s", flush=True)


def _final(path: Path) -> float:
    return float(np.load(path)["R"][-1].mean())


def table(seeds=(500, 501, 502, 503, 504)) -> str:
    L = ["| schedule | arm | mean final accuracy | experts | spawns | extends | max gap to the stored E-054 cell (flag off) |", "|---|---|---|---|---|---|---|"]
    for sch in SCHEDULES:
        for arm in ARMS:
            vals, experts, sp, ex, gaps = [], [], [], [], []
            for s in seeds:
                base = OUT / f"unimoe-{sch}" / f"{arm}_s{s}"
                if not base.with_suffix(".npz").exists():
                    continue
                d = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
                vals.append(_final(base.with_suffix(".npz")))
                experts.append(d["per_phase"][-1]["n_experts"])
                ev = d["per_phase"][-1]["events"]
                sp.append(ev.get("spawn", 0))
                ex.append(ev.get("extend", 0))
                if arm == "ULDA-off":
                    old = E.OUT / "p4_main" / f"unimoe-{sch}" / f"ULDA_s{s}.npz"
                    if old.exists():
                        gaps.append(abs(vals[-1] - _final(old)))
            if vals:
                L.append(f"| {sch} | {arm} | {np.mean(vals):.4f} | {np.mean(experts):.2f} | {np.mean(sp):.2f} | {np.mean(ex):.2f} | {('%.2e' % max(gaps)) if gaps else '-'} |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "table"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[500])
    a = ap.parse_args()
    if a.cmd == "run":
        for s in a.seeds:
            run_seed(s)
    else:
        text = table(tuple(a.seeds))
        (E.OUT / "p4_chance" / "TABLE_CHANCE.md").write_text(text, encoding="utf-8")
        print(text)
