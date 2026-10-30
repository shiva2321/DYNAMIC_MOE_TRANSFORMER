"""The P4 tables (data only; the reading is in ANALYSIS_P4.md): final accuracy of every variant, paired differences against the control, structure and cost, the wrong-domain rate, and the
capacity-stream checks. Everything is computed from the cached cells; nothing is run here.

    python -m unimoe.p4_report            # writes results/unimoe/p4/TABLES.md
"""

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from . import analysis as AN
from . import experiment as E
from . import p4 as P4
from . import report as RP
from . import runlib

EVAL, CONFIRM = P4.EVAL, P4.CONFIRM
OUT = E.OUT / "p4"


def _recs(group: str) -> Dict[tuple, object]:
    d = E.OUT / group
    return RP.collect(d) if d.exists() else {}


def _fm(rec) -> float:
    return E.final_mean(rec)


def _ci(vals: Sequence[float], nd: int = 3) -> str:
    return RP._ci(list(vals), nd)


def vals(recs, system: str, sch: str, seeds) -> List[float]:
    return [_fm(recs[(sch, system, s)]) for s in seeds if (sch, system, s) in recs]


def paired(recs, a: str, b: str, sch: str, seeds, recs_b=None) -> List[float]:
    rb = recs if recs_b is None else recs_b
    return [_fm(recs[(sch, a, s)]) - _fm(rb[(sch, b, s)]) for s in seeds if (sch, a, s) in recs and (sch, b, s) in rb]


def wrong_domain(rec, stream_blocks: Dict[str, tuple]) -> float:
    """Share of test messages (all domains) answered with a label from another domain's block."""
    y, p = rec.arrays["test_y"], rec.arrays["test_pred"]
    starts = np.asarray(sorted(b[0] for b in stream_blocks.values()))
    dom = lambda lab: np.searchsorted(starts, lab, side="right") - 1
    return float((dom(y) != dom(p)).mean())


def main_table(seeds, title: str) -> List[str]:
    main, base = _recs("p4_main"), _recs("runs_v2")
    L = [f"## {title}", ""]
    for sch in P4.SCHEDULES:
        blocks = E.streams(seeds[0])[sch].label_blocks
        L += [f"### {sch}", "", "| system | final mean (mean [95% CI]) | per seed | vs UG0 (paired) | experts | grow events | params | wrong-domain |", "|---|---|---|---|---|---|---|---|"]
        rows = [(n, main, n) for n in ("UG0", "UG1", "UDOM", "UDOMS", "ULDA", "UPLUS")] + [(n, base, n) for n in ("U-real", "replay_eqmem", "knn_eqmem", "oracle_experts")]
        for label, pool, key in rows:
            v = vals(pool, key, sch, seeds)
            if not v:
                continue
            d = paired(pool, key, "UG0", sch, seeds, main) if label != "UG0" else []
            ex = [pool[(sch, key, s)].per_phase[-1]["n_experts"] for s in seeds if (sch, key, s) in pool]
            gr = [pool[(sch, key, s)].per_phase[-1]["events"].get("grow", 0) for s in seeds if (sch, key, s) in pool]
            pa = [pool[(sch, key, s)].report["params"] for s in seeds if (sch, key, s) in pool]
            wd = [wrong_domain(pool[(sch, key, s)], blocks) for s in seeds if (sch, key, s) in pool]
            L.append(f"| {label} | {_ci(v)} | {', '.join(f'{x:.3f}' for x in v)} | {_ci(d) if d else '-'} | {np.mean(ex):.1f} | {np.mean(gr):.1f} | {np.mean(pa):.0f} | {np.mean(wd):.3f} |")
        L.append("")
    return L


def capacity_tables() -> List[str]:
    L = ["## I1: the capacity-hungry stream (six arithmetic operators, 10,000 examples each)", ""]
    cap, grow, tune = _recs("p4_capacity"), _recs("p4_capacity_growth"), _recs("p4_capacity_tune")
    L += ["### Does width matter here? (fixed width, no growth, 20 inner steps)", "", "| schedule | " + " | ".join(f"hidden {w}" for w in (16, 128, 512)) + " | 512 minus 16 (paired) |", "|---|---|---|---|---|"]
    for sch in ("seq", "mix"):
        cells = [vals(cap, f"fixed-h{w}", sch, EVAL) for w in (16, 128, 512)]
        d = [b - a for a, b in zip(cells[0], cells[2])] if cells[0] and cells[2] else []
        L.append(f"| {sch} | " + " | ".join(_ci(c) if c else "-" for c in cells) + f" | {_ci(d) if d else '-'} |")
    L += ["", "### Threshold tuning on seed 499 (start at 16 units, one-step probe, sequential)", ""]
    for k, r in sorted(tune.items(), key=lambda kv: kv[0][1]):
        L.append(f"- `{k[1]}`: final mean {_fm(r):.3f}, experts {r.per_phase[-1]['n_experts']}, grow events {r.per_phase[-1]['events'].get('grow', 0)}")
    L += ["", "### Growth from 16 units (seeds 500-502)", "", "| schedule | system | final mean | per seed | grow events | experts | params |", "|---|---|---|---|---|---|---|"]
    for sch in ("seq", "mix"):
        for sysname in ("shipped-gap-fast-h16", "UG1-h16", "UG1-h16-s5", "fixed-h16-s5"):
            v = vals(grow, sysname, sch, EVAL)
            if v:
                rs = [grow[(sch, sysname, s)] for s in EVAL if (sch, sysname, s) in grow]
                L.append(f"| {sch} | {sysname} | {_ci(v)} | {', '.join(f'{x:.3f}' for x in v)} | {np.mean([r.per_phase[-1]['events'].get('grow', 0) for r in rs]):.1f} | "
                         f"{np.mean([r.per_phase[-1]['n_experts'] for r in rs]):.1f} | {np.mean([r.report['params'] for r in rs]):.0f} |")
    L.append("")
    return L


def small_start_table() -> List[str]:
    small, main = _recs("p4_main_small"), _recs("p4_main")
    L = ["## Growth from a too-small start (16 units) on the real suite, seeds 500-502", "", "| schedule | fixed 16 | grown, theta 0.2 (capacity stream's value) | grown, theta 0.3 (tuned, seed 499) | grow events (0.3) | params (0.3) | fixed 128 (UG0) |", "|---|---|---|---|---|---|---|"]
    for sch in P4.SCHEDULES:
        f, g2, g3 = (vals(small, t, sch, EVAL) for t in ("fixed-h16", "UG1-h16", "UG1-h16-tuned"))
        u = vals(main, "UG0", sch, EVAL)
        if not (f and g3):
            continue
        ev = [small[(sch, "UG1-h16-tuned", s)].per_phase[-1]["events"].get("grow", 0) for s in EVAL if (sch, "UG1-h16-tuned", s) in small]
        pa = [small[(sch, "UG1-h16-tuned", s)].report["params"] for s in EVAL if (sch, "UG1-h16-tuned", s) in small]
        L.append(f"| {sch} | {_ci(f)} | {_ci(g2)} | {_ci(g3)} | {', '.join(map(str, ev))} | {np.mean(pa):.0f} | {_ci(u)} |")
    L.append("")
    return L


def oracle_table() -> List[str]:
    rec = _recs("p4_oracle_lda")
    L = ["## I4: the LDA blend under oracle routing (experts told the regime)", "", "| schedule | seeds | " + " | ".join(f"lambda {l}" for l in (0.0, 0.3, 0.5, 0.7, 1.0)) + " |", "|---|---|---|---|---|---|---|"]
    for sch in P4.SCHEDULES:
        for tag, seeds in (("tuning 499", (499,)), ("evaluation 500-502", EVAL)):
            cells = [vals(rec, f"oracle_lda_{l}", sch, seeds) for l in (0.0, 0.3, 0.5, 0.7, 1.0)]
            if all(cells):
                L.append(f"| {sch} | {tag} | " + " | ".join(_ci(c) for c in cells) + " |")
    L.append("")
    return L


def main() -> None:
    L = ["# P4 tables (generated from the cached cells)", ""]
    L += capacity_tables()
    L += oracle_table()
    L += small_start_table()
    L += main_table(EVAL, "Main suite, evaluation seeds 500-502")
    if any(_recs("p4_main") and (sch, "UG0", s) in _recs("p4_main") for sch in P4.SCHEDULES for s in CONFIRM):
        L += main_table(CONFIRM, "Main suite, confirmation seeds 503-504")
    text = "\n".join(L)
    runlib.atomic_write_text(OUT / "TABLES.md", text)
    print(text)


if __name__ == "__main__":
    main()
