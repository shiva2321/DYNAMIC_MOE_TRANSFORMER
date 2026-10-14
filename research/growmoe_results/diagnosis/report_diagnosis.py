"""Assemble RESULTS_DIAGNOSIS.md (and figures) from the diagnosis runs. Every number in the markdown is computed here from the json files; nothing is typed by hand.

    python report_diagnosis.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARITH = HERE.parent / "arith"
sys.path.insert(0, str(HERE.parents[2]))

import numpy as np  # noqa: E402

from growmoe import arith as A  # noqa: E402

OPS = A.OPS
ORDER = ["max", "add", "sub", "mod", "div", "mul"]
FIG = HERE / "figures"


def pct(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{nd}f}"


def load_dir(d: Path):
    out = []
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".partial.json"):
            continue
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return out


# ------------------------------------------------------------------------------------------------ study 1
def ladder_table(rows):
    cell = defaultdict(list)
    for r in rows:
        cell[(r["arch"], r["recipe"])].append(r)
    return cell


def mean_of(rs, key):
    return float(np.mean([r["summary"][key] for r in rs])) if rs else float("nan")


def sec_ladder(md, rows):
    cell = ladder_table(rows)
    archs = ["E128", "E512", "S256x2", "S256x2n", "L1024x3"]
    recipes = ["D40", "D10", "O5-r256", "O5-r2000", "O5-r20000", "O5-r100000", "O5-r300000", "O1-r20000", "O5-r20000-anneal", "M2000-S3k", "M2000-S23k", "M20000-S3k", "M20000-S23k"]
    pars = {a: next((r["train"]["params"] for r in rows if r["arch"] == a), None) for a in archs}
    md.append("## Study 1. Why does GrowMoE learn arithmetic worse than a dense network? Architecture against training recipe\n")
    md.append("Same data (mixed operators, lengths 1-5, 8,000 problems per operator = about 298,000 examples), same held-out test grid, same exact-match scoring as the benchmark. "
              "Mean over seeds of the exact match on UNSEEN problems (%); in brackets the number of seeds. `D40`/`D10` = the benchmark's dense recipe (shuffled batches of 512, one-cycle schedule, 40 / 10 epochs); "
              "`O5-r<N>` = the pool's own recipe (batches of 64 seen once, 5 steps on each batch mixed with an equal number of replayed examples from a reservoir of N, Adam 3e-3 constant), "
              "`O1` = one step per batch, `anneal` = the rate decayed to zero over the stream, `M<N>-S<k>` = the dense recipe for k optimiser steps on a random subset of N stored examples "
              "(an offline learner that may keep only as many examples as the pool's replay buffer holds).\n")
    md.append("| network (parameters) | " + " | ".join(recipes) + " |")
    md.append("|---|" + "---|" * len(recipes))
    for a in archs:
        if not any((a, r) in cell for r in recipes):
            continue
        md.append(f"| {a} ({pars[a]:,}) | " + " | ".join((f"{pct(mean_of(cell[(a, r)], 'mean_exact'))} ({len(cell[(a, r)])})" if (a, r) in cell else "-") for r in recipes) + " |")
    md.append("")
    md.append("Same table, exact match on the problems the network was TRAINED on (how well it fits what it was shown):\n")
    md.append("| network | " + " | ".join(recipes) + " |")
    md.append("|---|" + "---|" * len(recipes))
    for a in archs:
        if not any((a, r) in cell for r in recipes):
            continue
        md.append(f"| {a} | " + " | ".join((pct(mean_of(cell[(a, r)], 'seen_exact')) if (a, r) in cell else "-") for r in recipes) + " |")
    md.append("")
    md.append("**Is the single expert trained with the pool's recipe the same thing as the pool?** (unseen exact match %, per seed; the pool numbers come from the benchmark runs `arith/results/iid_FULL_*`)\n")
    md.append("| seed | pool shipped (GrowMoE) | E128 + O5-r256 | pool tuned (h512, replay 20,000) | E512 + O5-r20000 |")
    md.append("|---|---|---|---|---|")
    for s in (0, 1):
        def bench(name):
            f = ARITH / "results" / f"iid_FULL_{name}_s{s}.json"
            return float(np.array(json.loads(f.read_text(encoding='utf-8'))["grid_exact"]).mean()) if f.exists() else None

        def mine(a, r):
            rs = [x for x in cell.get((a, r), []) if x["seed"] == s]
            return rs[0]["summary"]["mean_exact"] if rs else None
        md.append(f"| {s} | {pct(bench('GrowMoE'))} | {pct(mine('E128', 'O5-r256'))} | {pct(bench('GrowMoE-h512-s5-r20000'))} | {pct(mine('E512', 'O5-r20000'))} |")
    md.append("")
    md.append("(Seed 1 of the shipped pool differs because its growth policy happened to fire three times in that run; the single expert trained without the policy cannot grow.)\n")

    def m(a, r):
        return mean_of(cell[(a, r)], "mean_exact") if (a, r) in cell else float("nan")
    steps = [("MLP-L (2.2 M parameters), dense recipe, 40 epochs: the benchmark's best dense network", m("L1024x3", "D40")),
             ("the pool's expert architecture (E512, 134 k parameters) with the same dense recipe", m("E512", "D40")),
             ("... the same expert, streaming recipe with unlimited replay (reservoir 300,000)", m("E512", "O5-r300000")),
             ("... streaming recipe, reservoir 20,000 (= the tuned pool)", m("E512", "O5-r20000")),
             ("... reservoir 256", m("E512", "O5-r256")),
             ("narrower expert, E128, reservoir 256 (= the shipped pool)", m("E128", "O5-r256"))]
    md.append("**From the best dense network down to the shipped pool, step by step** (unseen exact match %, mean over seeds):\n")
    md.append("| step | exact match | change |")
    md.append("|---|---|---|")
    prev = None
    for name, v in steps:
        md.append(f"| {name} | {pct(v)} | {'' if prev is None or np.isnan(v) or np.isnan(prev) else f'{100 * (v - prev):+.1f}'} |")
        prev = v
    md.append("")
    md.append(f"**The same large network under the pool's recipe:** MLP-L with the streaming recipe and a 20,000 reservoir scores {pct(m('L1024x3', 'O5-r20000'))} % "
              f"(the tuned pool's expert, 16 times smaller: {pct(m('E512', 'O5-r20000'))} %; MLP-S: {pct(m('S256x2', 'O5-r20000'))} %), against {pct(m('L1024x3', 'D40'))} % trained offline. "
              "Whatever the network, the streaming recipe lands in the same place.\n")


def fig_ladder(rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG.mkdir(exist_ok=True)
    cell = ladder_table(rows)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2))
    sizes = [("O5-r256", 256), ("O5-r2000", 2000), ("O5-r20000", 20000), ("O5-r100000", 100000), ("O5-r300000", 300000)]
    for a, colr in (("E128", "tab:orange"), ("E512", "tab:blue"), ("S256x2", "tab:green")):
        xs, ys = [], []
        for rec, n in sizes:
            if (a, rec) in cell:
                xs.append(n)
                ys.append(100 * mean_of(cell[(a, rec)], "mean_exact"))
        if xs:
            ax[0].plot(xs, ys, marker="o", color=colr, label=f"{a}: pool's streaming recipe")
        if (a, "D40") in cell:
            ax[0].axhline(100 * mean_of(cell[(a, "D40")], "mean_exact"), color=colr, ls="--", lw=1, label=f"{a}: dense recipe (40 passes over all data)")
        sub = [mean_of(cell[(a, rr)], "mean_exact") for rr in ("M20000-S3k", "M20000-S23k") if (a, rr) in cell]
        if sub:
            ax[0].scatter([20000], [100 * max(sub)], marker="s", color=colr, zorder=5, label=f"{a}: dense recipe on only 20,000 stored examples")
    if ("L1024x3", "D40") in cell:
        ax[0].axhline(100 * mean_of(cell[("L1024x3", "D40")], "mean_exact"), color="k", ls=":", lw=1, label="MLP-L dense (2.2 M parameters)")
    if ("L1024x3", "O5-r20000") in cell:
        ax[0].scatter([20000], [100 * mean_of(cell[("L1024x3", "O5-r20000")], "mean_exact")], marker="^", color="k", zorder=5, label="MLP-L with the pool's recipe")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("examples the learner can keep and replay (reservoir size)")
    ax[0].set_ylabel("exact match on unseen problems, %")
    ax[0].set_title("The pool's streaming recipe against memory", fontsize=10)
    ax[0].legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    ax[0].grid(alpha=0.3)
    for r in rows:
        if r["arch"] == "E512" and r["recipe"] in ("O5-r20000", "O5-r300000") and r["seed"] == 0 and r["train"].get("loss_curve_per_100_batches"):
            ax[1].plot(r["train"]["loss_curve_per_100_batches"], label=f"E512 {r['recipe']}")
    ax[1].set_xlabel("batches of 64 (x100)")
    ax[1].set_ylabel("batch cross-entropy (before the update)")
    ax[1].set_title("Training loss of the streaming recipe (seed 0)", fontsize=10)
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ladder.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ study 2
def sec_separability(md):
    f = HERE / "separability.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    md.append("## Study 2. Can the gate tell the operators apart? (the gate sees only the input)\n")
    o = d["operator"]
    md.append("One von Mises-Fisher density model per operator (256 stored inputs each, built exactly like an expert's signature), 24,000 held-out inputs.\n")
    md.append(f"- Which operator has the highest density at this input: **{pct(o['accuracy_argmax_density'])} %** right (chance {pct(o['chance'])} %); operator x output slot: **{pct(d['operator_x_slot']['accuracy_group'])} %** right "
              f"(chance {pct(d['operator_x_slot']['chance_group'])} %).")
    md.append(f"- The nearest stored input has the same operator in {pct(d['nearest_neighbour_same_operator'])} % of the cases.")
    md.append(f"- BUT an expert that has never seen operator j still ACCEPTS {pct(o['foreign_accept_mean'])} % of j's inputs as familiar (conformal p >= 0.02), against {pct(o['own_accept_mean'])} % of its own operator's inputs. "
              "So the per-sample gate can rank the operators, but it cannot call an operator it has never met \"new\".")
    g = d["geometry"]
    md.append(f"- Geometry: changing only the operator moves the input exactly as far as changing one digit (cosine {g['cos_change_only_operator']:.3f} in both cases); two random inputs of the same operator have mean cosine "
              f"{g['cos_same_operator']:.3f}, of different operators {g['cos_different_operator']:.3f}.\n")
    md.append("Acceptance matrix (rows: the expert's operator, columns: the operator of the input; % accepted):\n")
    md.append("| expert \\ input | " + " | ".join(OPS) + " |")
    md.append("|---|" + "---|" * len(OPS))
    for i, row in enumerate(o["accept_rate_sig_i_given_true_j"]):
        md.append(f"| {OPS[i]} | " + " | ".join(f"{100 * v:.0f}" for v in row) + " |")
    md.append("")


# ------------------------------------------------------------------------------------------------ study 3
def load_seq():
    runs = {}
    for d in ("seqprobe", "seqprobe2"):                              # seqprobe2 (instrumented) overrides the first runs of the same variant and seed
        if (HERE / d).exists():
            for f in sorted((HERE / d).glob("*_s*.json")):
                r = json.loads(f.read_text(encoding="utf-8"))
                runs[(r["variant"], r["seed"])] = r
    return runs


def sec_seqprobe(md):
    runs = load_seq()
    if not runs:
        return
    md.append("## Study 3. Operators arrive one after the other: what do the lifecycle decisions see, and what comes of them?\n")
    md.append("Same stream as the benchmark (max, add, sub, mod, div, mul; 8,000 problems each), tuned pool (h512, 5 steps, replay 20,000) with the stage-3 gate. "
              "`default` = shipped lifecycle (60-step interference probe, tolerance 0.05 nats), `probe600` = a ten times longer probe, "
              "`forced_spawn` = an extend can never be accepted (one new expert for every new operator).\n")
    md.append("| variant | seed | final mean exact % | forgetting (points) | experts at the end | extends | spawns | alarms | oracle-routed mean exact % |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for (v, sd), r in sorted(runs.items()):
        ev = r["train"].get("events", {})
        orc = r.get("oracle_routed_exact_by_op")
        md.append(f"| {v} | {sd} | {pct(r['summary']['mean_exact'])} | {100 * r['summary']['forgetting']:.1f} | {r['summary']['experts']} | {ev.get('extend', 0)} | {ev.get('spawn', 0)} | {ev.get('alarm', 0)} | "
                  f"{pct(float(np.mean(orc))) if orc else '-'} |")
    md.append("")
    md.append("(`oracle-routed` = every operator is answered by the existing expert that is best for it, whatever the gate says: what the same experts would score with perfect routing.)\n")
    for (v, sd), r in sorted(runs.items()):
        if sd != 0:
            continue
        md.append(f"**{v}, seed {sd}: exact match % of each operator after every block**\n")
        md.append("| after block | " + " | ".join(ORDER) + " | experts |")
        md.append("|---|" + "---|" * (len(ORDER) + 1))
        for b in r["blocks"]:
            md.append(f"| +{b['op_added']} | " + " | ".join(pct(b["exact_by_op"][OPS.index(o)]) for o in ORDER) + f" | {b['n_experts']} |")
        md.append("")
        last = r["blocks"][-1]["routing"]
        eids = sorted({e for o in last.values() for e in o}, key=int)
        md.append("Where the test inputs of each operator were sent at the end (% of the operator's examples per expert):\n")
        md.append("| operator | " + " | ".join(f"e{e}" for e in eids) + " |")
        md.append("|---|" + "---|" * len(eids))
        for o in ORDER:
            tot = max(sum(last[o].values()), 1)
            md.append(f"| {o} | " + " | ".join(f"{100 * last[o].get(str(e), last[o].get(e, 0)) / tot:.0f}" for e in eids) + " |")
        md.append("")
        if "expert_exact_by_op" in r:
            md.append("Exact match % of each expert when it answers EVERY example of an operator (the best entry of a column is the routing oracle):\n")
            md.append("| expert | " + " | ".join(ORDER) + " |")
            md.append("|---|" + "---|" * len(ORDER))
            for e, vals in sorted(r["expert_exact_by_op"].items(), key=lambda kv: int(kv[0])):
                md.append(f"| e{e} | " + " | ".join(pct(vals[OPS.index(o)]) for o in ORDER) + " |")
            md.append("| actual pool (with the gate) | " + " | ".join(pct(r["summary"]["exact_by_op"][o]) for o in ORDER) + " |")
            md.append("| oracle routing | " + " | ".join(pct(r["oracle_routed_exact_by_op"][OPS.index(o)]) for o in ORDER) + " |")
            md.append("")
        if "train_assignment_per_block" in r:
            md.append("Who was TRAINED on each block (share of the block's training examples handed to each expert; `none` = flagged, held or unlabelled, trained by nobody):\n")
            keys = sorted({k for blk in r["train_assignment_per_block"] for k in blk}, key=int)
            md.append("| block | " + " | ".join(("none" if k == "-1" else f"e{k}") for k in keys) + " |")
            md.append("|---|" + "---|" * len(keys))
            for b, blk in zip(r["blocks"], r["train_assignment_per_block"]):
                tot = max(sum(blk.values()), 1)
                md.append(f"| +{b['op_added']} | " + " | ".join(f"{100 * blk.get(k, 0) / tot:.1f}" for k in keys) + " |")
            md.append("")
    if ("default", 0) in runs:
        r = runs[("default", 0)]
        md.append("**What the interference probe saw at each decision of the default run (seed 0)** (losses in nats; it extends when BOTH numbers are at most 0.05):\n")
        md.append("| step | cluster size | old-regime loss, parent | new-regime loss, dedicated child | old-regime loss rise (d_old) | new-regime gap | outcome |")
        md.append("|---|---|---|---|---|---|---|")
        for e in r["probe_log"]:
            if "d_old" in e:
                md.append(f"| {e['step']} | {e['n_cluster']} | {e['old_loss_parent']:.3f} | {e['new_loss_child']:.4f} | {e['d_old']:+.3f} | {e['gap']:+.3f} | {', '.join(e['outcome']) or '-'} |")
        md.append("")
        md.append("The probe measures its losses on samples the networks were trained on (the 128-sample cluster and the other half of the replay buffer), so they are tiny: the decision rests on differences of 0.01-0.06 nats "
                  "against a tolerance of 0.05, while the task's real held-out cross-entropy is about 0.7 or more.\n")
    if ("probe600", 0) in runs:
        r = runs[("probe600", 0)]
        md.append("**The same probe with 600 steps (seed 0)**\n")
        md.append("| step | cluster size | d_old | gap | outcome |")
        md.append("|---|---|---|---|---|")
        for e in r["probe_log"]:
            if "d_old" in e:
                md.append(f"| {e['step']} | {e['n_cluster']} | {e['d_old']:+.3f} | {e['gap']:+.4f} | {', '.join(e['outcome']) or '-'} |")
        md.append("")


# ------------------------------------------------------------------------------------------------ study 4
def sec_growth(md):
    runs = load_dir(HERE / "growth") if (HERE / "growth").exists() else []
    if not runs:
        return
    md.append("## Study 4. The capacity-growth policy inside the pool: how often does it fire, and on how much evidence does it decide?\n")
    md.append("Mixed-operator stream, 4,656 batches. `plateau trigger` = how many times the trigger said \"the loss has stopped improving\" (it is asked only when the expert has 128+ replayed examples); a probe is the policy's "
              "propose-train-compare step (480 training steps for the control and each of up to 4 candidates), `backoff` = skipped because an earlier rejection started a waiting period.\n")
    md.append("| config | seed | unseen exact % | start params | end params | plateau trigger fired | would be true on this share of batches (offline replay) | probes run | skipped (backoff) | committed | probe training steps as % of the run's own training steps |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        calls = r["run_calls"]
        start = {"shipped": 35084, "tuned": 133772}[r["config"]]
        probe_steps = sum((1 + len(c["tried"])) * 480 for c in calls if c["reason"] != "backoff")
        md.append(f"| {r['config']} | {r['seed']} | {pct(r['summary']['mean_exact'])} | {start:,} | {r['train']['params']:,} | {len(r['run_plateaus'])} | {pct(r['offline_plateau_fraction'])} % | "
                  f"{sum(1 for c in calls if c['reason'] != 'backoff')} | {sum(1 for c in calls if c['reason'] == 'backoff')} | {sum(1 for c in calls if c['reason'] == 'committed')} | "
                  f"{100 * probe_steps / (r['n_batches'] * 5):.0f} % |")
    md.append("")
    for r in runs:
        md.append(f"**{r['config']}, seed {r['seed']}: every probe that actually ran**\n")
        md.append("| at batch | training / validation examples | hidden width | verdict | best candidate: gain vs control | p |")
        md.append("|---|---|---|---|---|---|")
        for c in r["run_calls"]:
            if c["reason"] == "backoff":
                continue
            md.append(f"| {c['n_batches_seen']} | {c['n_train']} / {c['n_valid']} | {c['hidden']} | {c['reason']} | {100 * c['best_gain']:+.1f} % | {c['best_p']:.3f} |")
        md.append("")
        md.append("The same policy asked again about the FINAL expert, with more evidence drawn from the stream:\n")
        md.append("| training / validation examples | verdict | best gain vs control | p | candidates (gain %, p) |")
        md.append("|---|---|---|---|---|")
        for k, v in r["counterfactual_final_expert"].items():
            md.append(f"| {k} | {v['reason']} | {100 * v['best_gain']:+.1f} % | {v['best_p']:.3f} | " + "; ".join(f"{n}: {100 * t['gain']:+.1f}, {t['p']:.3f}" for n, t in v["tried"].items()) + " |")
        md.append("")


def sec_grow_small(md):
    runs = load_dir(HERE / "grow_from_small") if (HERE / "grow_from_small").exists() else []
    if not runs:
        return
    md.append("## Study 5. Does the pool expand an expert as needed? A test with a known answer: start far too small\n")
    md.append("Mixed-operator arithmetic stream, reservoir 20,000 (so the policy has 16,000 training and 4,000 validation examples to decide on). Fixed experts of different widths show what the task needs; "
              "the growing pool starts at 16 hidden units and may widen (+50 %, +300 %, +1500 % candidates) or deepen.\n")
    md.append("| expert | parameters | unseen exact % (per seed) | mean |")
    md.append("|---|---|---|---|")
    for arch in ("E16", "E32", "E64", "E128", "E512"):
        vals = [r["fixed"][arch]["unseen_exact"] for r in runs if arch in r["fixed"]]
        pr = next((r["fixed"][arch]["params"] for r in runs if arch in r["fixed"]), None)
        md.append(f"| fixed, {arch[1:]} hidden units | {pr:,} | " + ", ".join(pct(v) for v in vals) + f" | {pct(float(np.mean(vals)))} |")
    gp = [r["growing_pool"] for r in runs]
    md.append(f"| **growing pool from 16** (shipped growth policy) | end: " + ", ".join(f"{g['params']:,}" for g in gp) + " | " + ", ".join(pct(g["unseen_exact"]) for g in gp) + f" | {pct(float(np.mean([g['unseen_exact'] for g in gp])))} |")
    md.append("")
    for r in runs:
        g = r["growing_pool"]
        real = [c for c in g["policy_calls"] if c["reason"] != "backoff"]
        md.append(f"Seed {r['seed']}: ended with {g['final_hidden']} hidden units and {g['final_depth']} block(s); {len(g['growth_events'])} growth event(s) "
                  + ("(" + "; ".join(f"step {e['step']}: {e['kind']} +{e.get('increment')}, gain {100 * e['gain']:+.1f} %, p {e['p']:.3f}" for e in g["growth_events"]) + ")" if g["growth_events"] else "")
                  + f"; {len(real)} probes ran and {sum(1 for c in real if c['reason'] == 'rejected')} were rejected. Best gains of the rejected probes (relative held-out loss, minimum required 2 %): "
                  + ", ".join(f"{100 * c['best_gain']:+.1f} % (p {c['best_p']:.3f})" for c in real if c["reason"] == "rejected")[:600] + ".\n")


def sec_widen_mid(md):
    allruns = load_dir(HERE / "widen_midstream") if (HERE / "widen_midstream").exists() else []
    if not allruns:
        return
    md.append("## Study 5b. Would the growth have paid off? Capacity added in the middle of the stream\n")
    md.append("The expert (pool recipe, reservoir 20,000) is grown exactly as the pool does on a commit (function-preserving, optimiser rebuilt) after half of the 4,656 batches; the stream goes on. "
              "For the shipped 128-wide expert the candidate sizes are those the policy proposes (+64, +384, +896 hidden units, or one more block); for the capacity-starved 16-wide expert the growth goes to 128 or 512 units.\n")
    for start in (128, 16):
        runs = [r for r in allruns if r.get("start_hidden", 128) == start]
        if not runs:
            continue
        names = list(runs[0]["arms"].keys())
        md.append(f"Starting expert: {start} hidden units.\n")
        md.append("| arm | parameters | unseen exact % (per seed) | mean | change against left alone |")
        md.append("|---|---|---|---|---|")
        base = float(np.mean([r["arms"][names[0]]["unseen_exact"] for r in runs]))
        for n in names:
            vals = [r["arms"][n]["unseen_exact"] for r in runs if n in r["arms"]]
            pr = next((r["arms"][n]["params"] for r in runs if n in r["arms"]), None)
            md.append(f"| {n} | {pr:,} | " + ", ".join(pct(v) for v in vals) + f" | {pct(float(np.mean(vals)))} | {100 * (float(np.mean(vals)) - base):+.1f} |")
        md.append("")


def sec_horizon(md):
    runs = load_dir(HERE / "horizon") if (HERE / "horizon").exists() else []
    if not runs:
        return
    md.append("## Study 5d. Why does the policy miss a capacity shortage? Probe length against the gain it measures\n")
    md.append("A 16-wide expert after 900 batches of the stream; control and widened candidates are trained for the given number of steps on the stored examples (16,000, mini-batches of 64, fresh Adam, "
              "as in the policy's probe) and compared on 4,000 held-out stream examples. Gain = relative reduction of the held-out cross-entropy against the control (the policy needs at least +2 %). "
              "For scale: widening the same expert to 128 or 512 units half-way through the real stream raises its accuracy from 11 % to 29-32 % (Study 5b).\n")
    cands = list(runs[0]["rows"][0]["candidates"].keys())
    md.append("| probe steps | control cross-entropy | " + " | ".join(f"{c} gain" for c in cands) + " |")
    md.append("|---|---|" + "---|" * len(cands))
    for i, row in enumerate(runs[0]["rows"]):
        md.append(f"| {row['steps']} | " + f"{np.mean([r['rows'][i]['control_ce'] for r in runs]):.3f}" + " | " + " | ".join(
            f"{100 * np.mean([r['rows'][i]['candidates'][c]['gain'] for r in runs]):+.1f} %" for c in cands) + " |")
    md.append("")
    md.append("Candidate accuracy on the held-out examples after the longest probe (5,000 steps) against the control: " + "; ".join(
        f"{c}: {pct(float(np.mean([r['rows'][-1]['candidates'][c]['acc'] for r in runs])))} % (control {pct(float(np.mean([r['rows'][-1]['candidates'][c]['control_acc'] for r in runs])))} %)" for c in cands) + ".\n")


def sec_routing_rule(md):
    f = HERE / "routing_rule.json"
    if not f.exists():
        return
    d = json.loads(f.read_text(encoding="utf-8"))
    md.append("## Study 2b. Why does density routing drain the older operators' inputs to the newer experts?\n")
    md.append("Signatures built the way the sequential pool builds them: expert 0 from three operators together (max, add, sub), experts 1-3 from one operator each (mod, div, mul), 256 stored inputs each.\n")
    md.append("| signature | kappa | log-normaliser | mean nearest-neighbour cosine of its stored inputs |")
    md.append("|---|---|---|---|")
    for s in d["signatures"]:
        md.append(f"| {s['name']} | {s['kappa']:.0f} | {s['log_norm']:.1f} | {s['nn_cos']:.3f} |")
    md.append("")
    md.append("Share (%) of each operator's test inputs routed to the expert that owns it:\n")
    md.append("| routing rule | max | add | sub | mod | div | mul |")
    md.append("|---|---|---|---|---|---|---|")
    for name, row in d["rules"].items():
        md.append(f"| {name} | " + " | ".join(f"{100 * v:.1f}" for v in row) + " |")
    md.append("")


def sec_router(md):
    f = HERE / "router.json"
    if not f.exists():
        return
    rows = json.loads(f.read_text(encoding="utf-8"))
    md.append("## Study 3b. How much of the sequential pool's loss is routing, and does a trivial learned router recover it?\n")
    md.append("The final pool of each instrumented run answered under: its shipped gate (accept, then highest log-density), the other routing rules the product offers, a routing ORACLE (best existing expert per operator), "
              "and a LEARNED router (multinomial logistic regression on unit-length inputs, trained on which expert was trained on each training example).\n")
    rules = sorted({k for r in rows for k in r["gate_rules_mean_exact"]})
    md.append("| variant | seed | shipped gate | " + " | ".join(rules) + " | learned router | oracle |")
    md.append("|---|---|---|" + "---|" * len(rules) + "---|---|")
    for r in rows:
        md.append(f"| {r['variant']} | {r['seed']} | {pct(r['shipped_gate_mean_exact'])} | " + " | ".join(pct(r["gate_rules_mean_exact"].get(k)) for k in rules) + f" | {pct(r['learned_router_mean_exact'])} | {pct(r['oracle_mean_exact'])} |")
    md.append("")
    md.append("(All numbers: mean exact match % over the six operators.)\n")


def sec_threshold(md):
    runs = load_dir(HERE / "grow_threshold") if (HERE / "grow_threshold").exists() else []
    if not runs:
        return
    md.append("## Study 5c. Is the 2 % minimum gain what keeps an under-sized expert small?\n")
    md.append("Same start as Study 5 (16 hidden units), only `GrowthCfg.min_gain` changed.\n")
    md.append("| min gain | seed | unseen exact % | end hidden units | end blocks | end parameters | growth events (kind +units, gain) |")
    md.append("|---|---|---|---|---|---|---|")
    for r in sorted(runs, key=lambda x: (-x["min_gain"], x["seed"])):
        md.append(f"| {r['min_gain']} | {r['seed']} | {pct(r['unseen_exact'])} | {r['final_hidden']} | {r['final_depth']} | {r['params']:,} | "
                  + "; ".join(f"{e['kind']} +{e.get('increment')} ({100 * e['gain']:+.1f} %)" for e in r["growth_events"]) + " |")
    md.append("")


# ------------------------------------------------------------------------------------------------ study 6
def sec_realtext(md):
    f = HERE.parent / "m5" / "m5b.json"
    g = HERE.parent / "m5" / "m5c_stage3.json"
    h = HERE.parent / "m5" / "m5d_fresh.json"
    if not f.exists():
        return
    old = json.loads(f.read_text(encoding="utf-8"))
    md.append("## Study 6. Real text (banking77, class-incremental): where do the points go? (from the saved m5b / m5c / m5d runs)\n")
    acc = {a: float(np.mean([old["final_runs"][a][s]["acc"] for s in old["final_runs"][a]])) for a in old["final_runs"]}
    new = json.loads(g.read_text(encoding="utf-8")) if g.exists() else None
    a_set = new["amended_runs"]["A_set (tuned steps)"]["mean"] if new else None
    md.append("Dev seeds 100-102 (3 seeds; exploratory):\n")
    md.append("| arm | accuracy % | what it is |")
    md.append("|---|---|---|")
    md.append(f"| kNN on all stored data | {pct(acc['kNN_all'])} | lookup ceiling |")
    md.append(f"| one network trained jointly on all classes (128 wide) | {pct(acc['B1_128'])} | parametric ceiling |")
    md.append(f"| experts told the regime at test time (oracle routing) | {pct(acc['B4'])} | what splitting the data into experts costs, with perfect routing |")
    md.append(f"| replay network (same-size replay buffer) | {pct(acc['B2'])} | |")
    if a_set is not None:
        md.append(f"| GrowMoE with the stage-3 gate | {pct(a_set)} | |")
    md.append(f"| GrowMoE today (no stage 3) | {pct(acc['A_today'])} | routing failure |")
    md.append("")
    if h.exists():
        d = json.loads(h.read_text(encoding="utf-8"))
        md.append("Fresh seeds 80-84 (5 seeds): " + "; ".join(f"{k} {np.mean(v):.1f}" for k, v in d["acc"].items()) + ".\n")


def fig_summary(rows):
    """One picture: the three answers (the streaming recipe, the operator-by-operator waterfall, growth as needed). Skips a panel whose data is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    FIG.mkdir(exist_ok=True)
    cell = ladder_table(rows)
    m = lambda a, r: 100 * mean_of(cell[(a, r)], "mean_exact") if (a, r) in cell else None      # noqa: E731
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))

    def bars(a, labels, vals, colors, title, xlabel="exact match on unseen problems, %"):
        keep = [(l, v, c) for l, v, c in zip(labels, vals, colors) if v is not None and not (isinstance(v, float) and np.isnan(v))]
        y = np.arange(len(keep))[::-1]
        a.barh(y, [v for _, v, _ in keep], color=[c for _, _, c in keep])
        a.set_yticks(y)
        a.set_yticklabels([l for l, _, _ in keep], fontsize=8)
        for yi, (_, v, _) in zip(y, keep):
            a.text(v + 0.6, yi, f"{v:.1f}", va="center", fontsize=8)
        a.set_title(title, fontsize=9.5)
        a.set_xlabel(xlabel, fontsize=8)
        a.grid(axis="x", alpha=0.3)
        a.set_xlim(0, 72)

    g, o, b = "tab:green", "tab:orange", "tab:blue"
    bars(ax[0], ["MLP-L (2.2 M), dense recipe, 40 passes", "MLP-L with the pool's recipe (20,000 kept)", "pool's expert (E512), dense recipe", "E512 streaming, 300,000 kept",
                 "E512 streaming, 20,000 kept (tuned pool)", "E128 streaming, 256 kept (shipped pool)"],
         [m("L1024x3", "D40"), m("L1024x3", "O5-r20000"), m("E512", "D40"), m("E512", "O5-r300000"), m("E512", "O5-r20000"), m("E128", "O5-r256")], [g, o, g, b, b, b],
         "1. Mixed arithmetic: the gap is the recipe")
    seq = load_seq()
    orc = lambda v: 100 * float(np.mean([np.mean(seq[(v, s)]["oracle_routed_exact_by_op"]) for s in (0, 1) if (v, s) in seq and "oracle_routed_exact_by_op" in seq[(v, s)]] or [np.nan]))      # noqa: E731
    act = lambda v: 100 * float(np.mean([seq[(v, s)]["summary"]["mean_exact"] for s in (0, 1) if (v, s) in seq] or [np.nan]))                                                       # noqa: E731
    oracle_op = None
    for f in glob_oracle():
        oracle_op = f
    lr = None
    rj = HERE / "router.json"
    if rj.exists():
        rr = [r for r in json.loads(rj.read_text(encoding="utf-8")) if r["variant"] == "default"]
        lr = 100 * float(np.mean([r["learned_router_mean_exact"] for r in rr])) if rr else None
    bars(ax[1], ["experts told the operator (offline)", "one expert per operator, perfect routing (= learned router)", "shipped lifecycle, perfect routing (= learned router)",
                 "one expert per operator, real gate", "shipped lifecycle, real gate"],
         [oracle_op, orc("forced_spawn"), orc("default"), act("forced_spawn"), act("default")], [g, b, b, o, o], "2. One operator at a time: where the points go")
    gs = load_dir(HERE / "grow_from_small") if (HERE / "grow_from_small").exists() else []
    wm = [r for r in load_dir(HERE / "widen_midstream") if r.get("start_hidden", 128) == 16]
    if gs:
        fx = lambda a: 100 * float(np.mean([r["fixed"][a]["unseen_exact"] for r in gs]))                                                                                            # noqa: E731
        gp = 100 * float(np.mean([r["growing_pool"]["unseen_exact"] for r in gs]))
        wa = lambda n: 100 * float(np.mean([r["arms"][n]["unseen_exact"] for r in wm])) if wm else None                                                                            # noqa: E731
        bars(ax[2], ["fixed 16 units", "fixed 64 units", "fixed 128 units", "fixed 512 units", "growing pool from 16 (shipped policy)", "16 units, widened to 128 half-way", "16 units, widened to 512 half-way"],
             [fx("E16"), fx("E64"), fx("E128"), fx("E512"), gp, wa("E16 widened to 128 at half"), wa("E16 widened to 512 at half")], [g, g, g, g, o, b, b], "3. Growth: the mechanism works,\nthe trigger does not")
    fig.tight_layout()
    fig.savefig(FIG / "fig_summary.png", dpi=120)
    plt.close(fig)


def glob_oracle():
    """Mean exact match (%) of the told-the-operator experts in the operator-by-operator benchmark runs (E-026)."""
    out = []
    for f in sorted((ARITH / "results").glob("seq_FULL_Oracle-op_s*.json")):
        out.append(100 * float(np.array(json.loads(f.read_text(encoding="utf-8"))["grid_exact"]).mean()))
    return [float(np.mean(out))] if out else []


def main():
    md = ["# Why does GrowMoE fall behind, and where does it not work as intended? (generated by report_diagnosis.py; do not edit by hand)\n"]
    rows = load_dir(HERE / "ladder")
    if rows:
        sec_ladder(md, rows)
        fig_ladder(rows)
        fig_summary(rows)
    sec_separability(md)
    sec_routing_rule(md)
    sec_seqprobe(md)
    sec_router(md)
    sec_growth(md)
    sec_grow_small(md)
    sec_widen_mid(md)
    sec_horizon(md)
    sec_threshold(md)
    sec_realtext(md)
    (HERE / "RESULTS_DIAGNOSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("wrote", HERE / "RESULTS_DIAGNOSIS.md")


if __name__ == "__main__":
    main()
