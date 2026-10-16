"""Build research/GROWMOE_MERGE_RESEARCH.md from report_template.md and the generated result files: every number that comes from a run is filled in here from the JSON files, none is typed.

    python build_report.py            # writes the report
    python build_report.py --check    # prints the computed numbers only
"""

import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import report_healing2 as R  # noqa: E402

TEMPLATE = HERE / "report_template.md"
OUT = ROOT / "research" / "GROWMOE_MERGE_RESEARCH.md"
DEV = ROOT / "research" / "growmoe_results" / "rv2d"


def runs(tag, arm):
    return [r for r in R.load(tag) if r["arm"] == arm]


def mean(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else float("nan")


def paired(tag, a, b):
    """mean and standard error over shared seeds of 100 * (acc_a - acc_b)"""
    ra, rb = {r["seed"]: r["acc"] for r in runs(tag, a)}, {r["seed"]: r["acc"] for r in runs(tag, b)}
    d = [100 * (ra[k] - rb[k]) for k in sorted(set(ra) & set(rb))]
    if not d:
        return float("nan"), float("nan")
    return float(np.mean(d)), float(np.std(d, ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0


def gap(tag, a, b):
    return paired(tag, a, b)[0]


def acc(tag, arm):
    return mean([100 * r["acc"] for r in runs(tag, arm)])


def merges(tag, arm):
    return mean([r["merges"] for r in runs(tag, arm)])


def injected(tag, arm):
    return mean([r["injected"] for r in runs(tag, arm)])


# ------------------------------------------------------------------------------------------------ tables
def md_section(path, heading_prefix):
    lines, grab, out = path.read_text(encoding="utf-8").splitlines(), False, []
    for ln in lines:
        if ln.startswith("## "):
            grab = ln.startswith(heading_prefix)
            continue
        if grab and ln.startswith("|"):
            out.append(ln)
        elif grab and out and ln.strip() == "":
            out.append("")
    return "\n".join(out).strip()


CONFIGS = [("n_a", "10 batches, 1 per regime"), ("n_a15", "15 batches, 1 per regime"), ("n_a20", "20 batches, 1 per regime"), ("n_m12", "20 batches, false spawn at batch 12"), ("n_b", "10 batches, 2 per regime")]
HEAL_ROWS = [("shipped", "the shipped merge"), ("preq", "prequential references"), ("agree", "label-free, agreement >= 0.8"), ("agree50", "label-free, agreement >= 0.5 (fitted to this task)"),
             ("rival", "rival tracking on live traffic (no sample withheld)"), ("hold", "held-out samples (1 in 8 withheld)"), ("ens", "answer-time averaging (no merge)")]


def heal_table():
    rows = ["| | " + " | ".join(name for _, name in CONFIGS) + " |", "|---|" + "---|" * len(CONFIGS)]
    rows.append("| accuracy without false spawns (points) | " + " | ".join(f"{acc(t, 'clean'):.1f}" if runs(t, "clean") else "-" for t, _ in CONFIGS) + " |")
    rows.append("| accuracy with false spawns, nothing done | " + " | ".join(f"{acc(t, 'none'):.1f}" if runs(t, "none") else "-" for t, _ in CONFIGS) + " |")
    rows.append("| experts without / with false spawns | " + " | ".join(f"{mean([r['final_experts'] for r in runs(t, 'clean')]):.0f} / {mean([r['final_experts'] for r in runs(t, 'none')]):.0f}" for t, _ in CONFIGS) + " |")
    for arm, label in HEAL_ROWS:
        cells = []
        for t, _ in CONFIGS:
            if not runs(t, arm):
                cells.append("-")
                continue
            g = paired(t, arm, "none")
            if arm == "ens":
                cells.append(f"-; {g[0]:+.1f} +-{g[1]:.1f}")
            else:
                cells.append(f"{merges(t, arm):.1f} of {injected(t, arm):.1f}; {g[0]:+.1f} +-{g[1]:.1f}")
        rows.append(f"| {label}: false spawns removed; accuracy change vs doing nothing | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def cost_table():
    rows = ["| stream | experts with false spawns | accuracy without false spawns | with false spawns | accuracy lost (paired, +-) | share of the questions answered by a false spawn |", "|---|---|---|---|---|---|"]
    for t, name in CONFIGS:
        if not runs(t, "none"):
            continue
        g = paired(t, "clean", "none")
        rows.append(f"| {name} | {mean([r['final_experts'] for r in runs(t, 'none')]):.0f} | {R.ms([100 * r['acc'] for r in runs(t, 'clean')])} | {R.ms([100 * r['acc'] for r in runs(t, 'none')])} | "
                    f"{g[0]:.1f} +-{g[1]:.1f} | {mean([100 * r['share_routed_to_injected'] for r in runs(t, 'none')]):.0f} % |")
    return "\n".join(rows)


CAPS = [("n_cap16", "limit 16, 12 false spawns (needs 24)"), ("n_cap20", "limit 20, 24 false spawns (needs 36)"), ("n_cap8clean", "limit 8, no false spawns (12 regimes need 12)")]
CAP_ROWS = [("cap_drop", "drop the cluster (today, silent)"), ("cap_extend", "force the cluster into the nearest expert"), ("cap_evict", "evict the most redundant expert (else the least recently used)"),
            ("cap_evict_lru", "evict the least recently used expert"), ("cap_smart", "evict the most redundant expert; if none is redundant, force an extend")]


def cap_summary():
    rows = ["| what a full pool does | " + " | ".join(name for _, name in CAPS) + " |", "|---|" + "---|" * len(CAPS)]
    rows.append("| (reference) the same stream without a limit | " + " | ".join(f"{acc(t, 'clean'):.1f}" if t != "n_cap8clean" else "-" for t, _ in CAPS) + " |")
    for arm, label in CAP_ROWS:
        cells = []
        for t, _ in CAPS:
            rs = runs(t, arm)
            if not rs:
                cells.append("-")
                continue
            late = mean([100 * r["acc_late"] for r in rs if r["acc_late"] is not None])
            noexp = mean([r["regimes_without_own_expert"] for r in rs])
            cells.append(f"{acc(t, arm):.1f}; regimes met after it filled: {late:.0f} %; {noexp:.1f} regimes without an expert")
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def alias_tables():
    rows = []
    for f in sorted(glob.glob(str(HERE / "alias_check_s3_*.json"))):
        rows += json.loads(Path(f).read_text(encoding="utf-8"))
    for r in rows:
        r.setdefault("arm", "S3")
    t1 = ["| stream | arm | stream seed | accuracy, pool seed = stream seed (as the dev runs) | accuracy, pool seed = stream seed + 7 | head start |", "|---|---|---|---|---|---|"]
    tab = {}
    for s in sorted({r["stream"] for r in rows}):
        for arm in ("S3", "S3+T"):
            for seed in (100, 101):
                a = [r for r in rows if r["stream"] == s and r["arm"] == arm and r["stream_seed"] == seed and r["pool_seed"] == seed]
                b = [r for r in rows if r["stream"] == s and r["arm"] == arm and r["stream_seed"] == seed and r["pool_seed"] == seed + 7]
                if a and b:
                    tab[(s, arm, seed)] = (100 * a[0]["acc"], 100 * b[0]["acc"])
                    t1.append(f"| {s} | {arm} | {seed} | {100 * a[0]['acc']:.1f} ({a[0]['experts']} experts) | {100 * b[0]['acc']:.1f} ({b[0]['experts']} experts) | {100 * (a[0]['acc'] - b[0]['acc']):+.1f} |")
    t2 = ["| stream | stream seed | transfer rule minus S3, as the dev runs measure it | the same without the head start |", "|---|---|---|---|"]
    for s in sorted({r["stream"] for r in rows}):
        for seed in (100, 101):
            if (s, "S3", seed) in tab and (s, "S3+T", seed) in tab:
                a3, b3 = tab[(s, "S3", seed)]
                at, bt = tab[(s, "S3+T", seed)]
                t2.append(f"| {s} | {seed} | {at - a3:+.1f} | {bt - b3:+.1f} |")
    return "\n".join(t1), "\n".join(t2), tab


TRUE_FUNCTIONS = {"arith-seq": 6, "s3:F-15": 15, "s3:F-30": 30, "s3:S2-30": 30, "s3:OV-0.6": 10, "s3:OV-0.85": 10, "s3:OV-0.3": 10, "s3:OV-0.0": 10, "s3:OV-0.3-MIXED": 10, "s3:S1": 7, "s3:S2-5": 5}


def dev_snapshot():
    cells = 0
    for f in (DEV / "cells").glob("*.json"):
        try:
            cells += 1 if "acc" in json.loads(f.read_text(encoding="utf-8")) else 0
        except Exception:
            pass
    rows = ["| unit | seeds | true number of functions | experts without the transfer rule (S3) | experts with it (S3+T) | accuracy difference T minus S3 (points, as measured with the head start where the stream has one) |", "|---|---|---|---|---|---|"]
    grab = False
    for ln in (DEV / "RESULTS_DEV.md").read_text(encoding="utf-8").splitlines():
        if ln.startswith("## "):
            grab = ln.startswith("## Transfer rule")
            continue
        if grab and ln.startswith("| ") and not ln.startswith("| unit") and not ln.startswith("|---"):
            c = [x.strip() for x in ln.strip("|").split("|")]
            m = re.search(r"experts ([\d.]+) -> ([\d.]+)", c[5])
            true = TRUE_FUNCTIONS.get(c[0])
            if m:
                rows.append(f"| {c[0]} | {c[1]} | {true if true is not None else 'n/a'} | {float(m.group(1)):.1f} | {float(m.group(2)):.1f} | {c[4]} |")
    return cells, "\n".join(rows)


def infant_table():
    p = HERE / "RESULTS_INFANT.md"
    return p.read_text(encoding="utf-8").split("\n\n", 1)[1].strip() if p.exists() else "(not run)"


def read_out(name):
    """The text of an evidence file with the library warnings removed (it is shown verbatim in the note)."""
    p = HERE / name
    if not p.exists():
        return f"({name} not found: run it)"
    keep = [ln.rstrip() for ln in p.read_text(encoding="utf-8", errors="replace").splitlines() if "Warning" not in ln and "pynvml" not in ln and ln.strip()]
    return "\n".join(keep)


def twin_diag_means():
    """Per run of twin_diag.py: its header line and its 'mean' line."""
    out, header = [], None
    for ln in read_out("twin_diag_output.txt").splitlines():
        if ln.startswith("n_train"):
            header = ln
        elif ln.startswith("mean"):
            out.append((header + "\n" if header else "") + ln)
    return "\n".join(out)


def twin_stats():
    """Signature overlap (both directions) and answer agreement of the (original, false spawn) pairs in healing2_diag_output.txt."""
    ov, ag = [], []
    for ln in read_out("healing2_diag_output.txt").splitlines():
        m = re.search(r"overlap ([\d.]+)/([\d.]+) agree ([\d.]+)", ln)
        if m:
            ov += [float(m.group(1)), float(m.group(2))]
            ag.append(float(m.group(3)))
    if not ov:
        return "?", "?"
    return f"{min(ov):.2f}-{max(ov):.2f}", f"{min(ag):.2f}-{max(ag):.2f}"


def infant_numbers():
    p = HERE / "infant_study.json"
    rs = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    shipped3 = [r["regimes_with_expert"] for r in rs if r["relax"] == 4.0 and r["length"] == 3]
    x1_10 = [r["experts"] for r in rs if r["relax"] == 1.0 and r["length"] == 10]
    return ", ".join(str(v) for v in shipped3[:-1]) + " and " + str(shipped3[-1]) if shipped3 else "?", ", ".join(str(v) for v in x1_10[:-1]) + " and " + str(x1_10[-1]) if x1_10 else "?"


def dev_numbers():
    """The arithmetic rows of the dev report (router, transfer rule, growth at the default gap)."""
    lines = (DEV / "RESULTS_DEV.md").read_text(encoding="utf-8").splitlines()
    sec, out = None, {}
    for ln in lines:
        if ln.startswith("## "):
            sec = ln
            continue
        c = [x.strip() for x in ln.strip("|").split("|")]
        if sec and sec.startswith("## Router") and c[0] == "arith-seq":
            out["router"] = c[4].split(" ")[0]
        if sec and sec.startswith("## Transfer rule") and c[0] == "arith-seq":
            out["transfer"] = c[4].split(" ")[0]
        if sec and sec.startswith("## Growth from the fit gap") and c[0] in ("arith-mixed16", "arith-mixed128"):
            out[c[0]] = c[4].split(" ")[0]
    return out


def compute():
    n_tags = ["n_a", "n_a15", "n_a20", "n_m12", "n_b"]
    ship = [r for t in n_tags for r in runs(t, "shipped")]
    all_runs = [r for t in n_tags + ["n_cap16", "n_cap20", "n_cap8clean"] for r in R.load(t)]
    cross = sum(r["merges_cross_regime"] for r in all_runs)
    total_merges = sum(r["merges"] for r in all_runs)
    gap_hold_b = paired("n_b", "none_h", "none")
    t_alias1, t_alias2, tab = alias_tables()
    cells, dev_counts = dev_snapshot()
    ph = {
        "SHIP_RUNS": str(len(ship)), "SHIP_MERGES": f"{sum(r['merges'] for r in ship)}", "SHIP_FS": f"{sum(r['injected'] for r in ship)}",
        "COST_A": f"{gap('n_a', 'clean', 'none'):.1f}", "COST_B": f"{gap('n_b', 'clean', 'none'):.1f}", "COST_A15": f"{gap('n_a15', 'clean', 'none'):.1f}",
        "COST_A20": f"{gap('n_a20', 'clean', 'none'):.1f}", "COST_M12": f"{gap('n_m12', 'clean', 'none'):.1f}",
        "SHARE_A": f"{mean([100 * r['share_routed_to_injected'] for r in runs('n_a', 'none')]):.0f}", "SHARE_A20": f"{mean([100 * r['share_routed_to_injected'] for r in runs('n_a20', 'none')]):.0f}",
        "CAP16_DROP": f"{acc('n_cap16', 'cap_drop'):.1f}", "CAP16_SMART": f"{acc('n_cap16', 'cap_smart'):.1f}", "CAP16_CLEAN": f"{acc('n_cap16', 'clean'):.1f}", "CAP16_EXTEND": f"{acc('n_cap16', 'cap_extend'):.1f}",
        "CAP16_LRU": f"{acc('n_cap16', 'cap_evict_lru'):.1f}", "CAP16_NOEXP": f"{mean([r['regimes_without_own_expert'] for r in runs('n_cap16', 'cap_drop')]):.0f}",
        "CAP16_LATE": f"{mean([100 * r['acc_late'] for r in runs('n_cap16', 'cap_drop') if r['acc_late'] is not None]):.0f}",
        "CAP8_DROP": f"{acc('n_cap8clean', 'cap_drop'):.1f}", "CAP8_EXTEND": f"{acc('n_cap8clean', 'cap_extend'):.1f}", "CAP8_EVICT": f"{acc('n_cap8clean', 'cap_evict'):.1f}",
        "CAP20_DROP": f"{acc('n_cap20', 'cap_drop'):.1f}", "CAP20_SMART": f"{acc('n_cap20', 'cap_smart'):.1f}",
        "RIVAL_A": f"{merges('n_a', 'rival'):.1f}", "RIVAL_A15": f"{merges('n_a15', 'rival'):.1f}", "RIVAL_A20": f"{merges('n_a20', 'rival'):.1f}", "RIVAL_B": f"{merges('n_b', 'rival'):.1f}",
        "RIVAL_M12": f"{merges('n_m12', 'rival'):.1f}", "RIVAL_GAIN_A20": f"{gap('n_a20', 'rival', 'none'):.1f}",
        "HOLD_A": f"{merges('n_a', 'hold'):.1f}", "HOLD_B": f"{merges('n_b', 'hold'):.1f}", "HOLD_GAIN_A": f"{gap('n_a', 'hold', 'none'):.1f}", "HOLD_NET_B": f"{gap('n_b', 'hold', 'none'):.1f}",
        "WITHHOLD_B": f"{-gap_hold_b[0]:.1f}", "WITHHOLD_CLEAN": f"{gap('n_a', 'clean', 'clean_h'):.1f}",
        "AGREE50_A": f"{merges('n_a', 'agree50'):.1f}", "AGREE50_B": f"{merges('n_b', 'agree50'):.1f}", "AGREE50_B_ACC": f"{acc('n_b', 'agree50'):.1f}", "NONE_B": f"{acc('n_b', 'none'):.1f}",
        "AGREE_A": f"{merges('n_a', 'agree'):.1f}", "ENS_A": f"{gap('n_a', 'ens', 'none'):.1f}", "ENS_B": f"{gap('n_b', 'ens', 'none'):.1f}", "ENS_A20": f"{gap('n_a20', 'ens', 'none'):.1f}", "ENS_M12": f"{gap('n_m12', 'ens', 'none'):.1f}",
        "CLEAN_ENS_DIFF": f"{max(abs(gap(t, 'clean_ens', 'clean')) for t in ('n_a', 'n_b')):.2f}",
        "CROSS": str(cross), "TOTAL_MERGES": str(total_merges),
        "ALIAS_F15": f"{np.mean([tab[('F-15', 'S3', s)][0] - tab[('F-15', 'S3', s)][1] for s in (100, 101)]):.1f}",
        "DEV_CELLS": str(cells),
        "HOLD_SOUP_A": f"{acc('n_a', 'hold_soup'):.1f}", "HOLD_DISTILL_A": f"{acc('n_a', 'hold_distill'):.1f}", "HOLD_ACC_A": f"{acc('n_a', 'hold'):.1f}", "CAP20_CLEAN": f"{acc('n_cap20', 'clean'):.1f}",
    }
    ph["INF_SHIPPED_3"], ph["INF_X1_10"] = infant_numbers()
    ph["TWIN_OVERLAP"], ph["TWIN_AGREE"] = twin_stats()
    dn = dev_numbers()
    ph.update({"DEV_ROUTER_ARITH": dn.get("router", "?"), "DEV_T_ARITH": dn.get("transfer", "?"), "DEV_G16": dn.get("arith-mixed16", "?"), "DEV_G128": dn.get("arith-mixed128", "?")})
    tables = {"PAIR_DECISIONS": md_section(HERE / "RESULTS_MERGE_STUDY.md", "## 1."), "PAIR_SWEEP": md_section(HERE / "RESULTS_MERGE_STUDY.md", "## 5."), "ARITH": md_section(HERE / "RESULTS_MERGE_STUDY.md", "## 6."),
              "COST_TABLE": cost_table(), "HEAL_TABLE": heal_table(), "CAP_TABLE": cap_summary(), "INFANT": infant_table(), "TWIN_DIAG": twin_diag_means(), "ALIAS_CHECK": read_out("alias_check_output.txt"), "IDSHIFT0": "pool seed offset 0 (the head start is present):\n" + read_out("idshift_check_offset0.txt"),
              "IDSHIFT7": "pool seed offset 7 (no head start):\n" + read_out("idshift_check_offset7.txt"), "ALIAS1": t_alias1, "ALIAS2": t_alias2, "DEV_COUNTS": dev_counts}
    return ph, tables


def main() -> None:
    ph, tables = compute()
    if "--check" in sys.argv:
        for k, v in ph.items():
            print(f"{k:16s} {v}")
        return
    text = TEMPLATE.read_text(encoding="utf-8")
    for k, v in {**tables, **ph}.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z0-9_]+\}\}", text)
    if left:
        print("UNFILLED:", sorted(set(left)))
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print("wrote", OUT, len(text.splitlines()), "lines")


if __name__ == "__main__":
    main()
