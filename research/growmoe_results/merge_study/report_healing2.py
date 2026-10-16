"""Tables for the end-to-end healing study (throwaway): every number in RESULTS_HEALING2.md is computed here from healing2_*.json.

    python report_healing2.py
"""

import json
import math
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

ARM_TEXT = {
    "clean": "no false spawns, no merge",
    "clean_h": "no false spawns, 1 in 8 samples withheld from training",
    "clean_ens": "no false spawns, answer-time twin averaging",
    "idshift": "no false spawns, expert ids advance as in the false-spawn pools (control for the id bookkeeping)",
    "none": "false spawns, no merge (today)",
    "none_h": "false spawns, no merge, 1 in 8 withheld (control for the hold arms)",
    "shipped": "false spawns, the shipped merge",
    "preq": "redirect merge, evidence = prequential references",
    "agree": "label-free merge, agreement >= 0.8 on stored prototypes",
    "agree50": "label-free merge, agreement >= 0.5 (a threshold fitted to this task)",
    "rival": "redirect merge, evidence = rival tracking on live traffic",
    "rival_ft": "rival + 20 fine-tuning steps of the survivor on the stored samples",
    "rival_g": "rival + the survivor must beat chance by 0.2 nats",
    "hold": "redirect merge, evidence = held-out samples (1 in 8)",
    "hold_g": "hold + the survivor must beat chance by 0.2 nats",
    "hold_soup": "hold + weight midpoint if not worse on held-out samples",
    "hold_distill": "hold + distillation on stored prototypes if not worse",
    "ens": "false spawns, no merge, answer-time twin averaging",
    "cap_drop": "full pool drops the cluster (what the shipped cap does)",
    "cap_extend": "full pool forces an extend into the nearest expert",
    "cap_evict": "full pool evicts the most redundant expert, then founds the new one",
    "cap_evict_lru": "full pool evicts the least recently used expert, then founds the new one",
    "cap_smart": "full pool evicts the most redundant expert if there is one, otherwise forces an extend",
    "hold_evict": "hold merge, plus eviction of the most redundant expert if the pool is still full",
}
ORDER = ["clean", "idshift", "clean_h", "clean_ens", "none", "none_h", "ens", "shipped", "preq", "agree", "agree50", "rival", "rival_ft", "rival_g", "hold", "hold_g", "hold_soup", "hold_distill",
         "cap_drop", "cap_extend", "cap_evict", "cap_evict_lru", "cap_smart", "hold_evict"]


def ms(v, nd=1):
    v = [x for x in v if x is not None]
    if not v:
        return "-"
    if len(v) == 1:
        return f"{np.mean(v):.{nd}f}"
    return f"{np.mean(v):.{nd}f} +-{np.std(v, ddof=1) / math.sqrt(len(v)):.{nd}f}"


def load(tag):
    p = HERE / f"healing2_{tag}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


def paired(res, arm, control, key="acc"):
    """Mean difference (points) arm - control over the seeds both have, with the standard error of the paired differences."""
    a = {r["seed"]: r[key] for r in res if r["arm"] == arm}
    c = {r["seed"]: r[key] for r in res if r["arm"] == control}
    seeds = sorted(set(a) & set(c))
    d = [100 * (a[s] - c[s]) for s in seeds]
    if len(d) < 2:
        return "-"
    return f"{np.mean(d):+.1f} +-{np.std(d, ddof=1) / math.sqrt(len(d)):.1f}"


def main_table(res, title, arms=None):
    out = [f"### {title}\n", "| arm | what it is | seeds | accuracy (points) | vs `none` | vs `none_h` | experts at the end | false spawns still alive | merges (of them wrong-regime) | regimes still served by several experts |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in (arms or ORDER):
        rs = [r for r in res if r["arm"] == arm]
        if not rs:
            continue
        injected = ms([r["injected"] for r in rs])
        out.append(f"| {arm} | {ARM_TEXT.get(arm, '')} | {len(rs)} | {ms([100 * r['acc'] for r in rs])} | {paired(res, arm, 'none') if arm != 'none' else ''} | {paired(res, arm, 'none_h') if arm not in ('none', 'none_h') else ''} | "
                   f"{ms([r['final_experts'] for r in rs])} | {ms([r['injected_alive'] for r in rs])} of {injected} | {ms([r['merges'] for r in rs])} ({sum(r['merges_cross_regime'] for r in rs)}) | {ms([r['regimes_served_by_several'] for r in rs])} |")
    return "\n".join(out) + "\n"


def cap_table(res, cap, need, arms=None):
    out = [f"### Pool limited to {cap} experts (12 regimes; without a limit this pool would hold {need})\n",
           "| arm | what it is | seeds | accuracy, all regimes | regimes that never got an expert of their own | accuracy on the regimes first met AFTER the pool filled up | seeds in which it filled up | clusters met while full | experts evicted (of them false spawns) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for arm in (arms or ORDER):
        rs = [r for r in res if r["arm"] == arm]
        if not rs:
            continue
        late = [100 * r["acc_late"] for r in rs if r["acc_late"] is not None]
        filled = sum(1 for r in rs if r["cap_reached_in_segment"] is not None)
        out.append(f"| {arm} | {ARM_TEXT.get(arm, '')} | {len(rs)} | {ms([100 * r['acc'] for r in rs])} | {ms([r['regimes_without_own_expert'] for r in rs])} | {ms(late) if late else '-'} | {filled} of {len(rs)} | "
                   f"{ms([r.get('cap_events') for r in rs])} | {ms([r.get('evicted') for r in rs])} ({ms([r.get('evicted_were_injected') for r in rs])}) |")
    return "\n".join(out) + "\n"


def main() -> None:
    md = ["# End-to-end healing study v2 (throwaway; generated by report_healing2.py from healing2_*.json; do not edit by hand)\n",
          "12 function regimes (4 classes, 32 inputs, 10 batches of 64 per regime unless stated, then two revisits); the pool is the stage-3 pool with the stage-4b transfer test on. Accuracy is the mean over the 12 regimes of "
          "the final held-out accuracy, in points. +-: standard error over seeds. `vs none` / `vs none_h`: mean and standard error of the paired difference over the seeds both arms share.\n",
          "False spawns are injected by handing a batch of a regime the pool already knows to the spawn routine with the extend option switched off (the way the merge audit forced one): "
          "config a injects one per regime (at batch 3), config b two (at batches 3 and 4). A duplicate is born a clone of its parent (lambda = 1.0, affinity 1.0).\n"]
    for tag, title in (("n_a", "Config a: one false spawn per regime (12 in total), batch 3 of 10"), ("n_b", "Config b: two false spawns per regime (24 in total), batches 3 and 4 of 10"),
                       ("n_a15", "Config a with regimes of 15 batches, false spawn at batch 3"),
                       ("n_a20", "Config a with regimes twice as long (20 batches of 64 per regime), false spawn at batch 3"), ("n_m12", "Regimes of 20 batches, false spawn at batch 12 (the original has seen 11 batches)"),
                       ("n_a4", "Config a, hold-out rate 1 in 4"), ("n_b4", "Config b, hold-out rate 1 in 4")):
        res = load(tag)
        if res:
            md.append(main_table(res, title))
    for tag, cap, need in (("n_cap16", 16, 24), ("n_cap20", 20, 36), ("n_cap8clean", 8, 12)):
        res = load(tag)
        if res:
            md.append(cap_table(res, cap, need))
    out = HERE / "RESULTS_HEALING2.md"
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(md))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
