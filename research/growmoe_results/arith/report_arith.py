"""Report for the arithmetic benchmark: reads results/*.json and results/*.npz written by run_arith.py and writes RESULTS_ARITH.md plus figures. Every number is computed here.

    python research/growmoe_results/arith/report_arith.py [--dir results]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import arith_lib as L  # noqa: E402
from growmoe import arith as A  # noqa: E402

NAMES = list(A.OPS)


# ------------------------------------------------------------------------------------------------ small statistics (numpy only)
def contingency(a: np.ndarray, b: np.ndarray):
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    m = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(m, (ia, ib), 1)
    return m


def nmi(a, b) -> float:
    m = contingency(a, b).astype(np.float64)
    n = m.sum()
    pa, pb = m.sum(1) / n, m.sum(0) / n
    nz = m > 0
    mi = float((m[nz] / n * np.log(m[nz] * n / (m.sum(1)[:, None] * m.sum(0)[None, :])[nz])).sum())
    ha, hb = -float((pa[pa > 0] * np.log(pa[pa > 0])).sum()), -float((pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return 0.0 if ha * hb == 0 else mi / np.sqrt(ha * hb)


def ari(a, b) -> float:
    m = contingency(a, b).astype(np.float64)
    n = m.sum()
    comb = lambda x: x * (x - 1) / 2                                     # noqa: E731
    sij, sa, sb = comb(m).sum(), comb(m.sum(1)).sum(), comb(m.sum(0)).sum()
    exp = sa * sb / comb(n)
    return 0.0 if (sa + sb) / 2 == exp else float((sij - exp) / ((sa + sb) / 2 - exp))


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score of a random positive > score of a random negative), ties half (rank formula with average ranks)."""
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    uniq, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(uniq))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ------------------------------------------------------------------------------------------------ loading
def load(d: Path):
    runs = []
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        r["_stem"] = p.with_suffix("")
        runs.append(r)
    return runs


_TESTS = {}


def test_of(per_cell: int):
    if per_cell not in _TESTS:
        _TESTS[per_cell] = L.build_test(per_cell)
    return _TESTS[per_cell]


def per_problem(run: dict):
    """(test ProblemSet, per-problem exact-match bool, per-sample arrays dict) of one run."""
    ps = test_of(run["per_cell"])
    z = np.load(run["_stem"].with_suffix(".npz"))
    raw = {k: z[k] for k in z.files}
    wrong = np.bincount(ps.pi, weights=(~raw["correct"]).astype(np.float64), minlength=len(ps))
    return ps, wrong == 0, raw


def in_len(regime: str) -> np.ndarray:
    ok = np.zeros((5, 5), dtype=bool)
    for la in L.LENGTHS[regime]:
        for lb in L.LENGTHS[regime]:
            ok[la - 1, lb - 1] = True
    return ok


def mean_over(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}"


# ------------------------------------------------------------------------------------------------ sections
def sec_generalisation(runs, md):
    md.append("## 1. Does it generalise to unseen problems? (training lengths 1-5, mixed operators)\n")
    md.append("Exact match % (every answer digit right) on problems that were never trained on, by operator; `seen` = the learner's own training problems. Mean over seeds. "
              "`mod/div hard` = only problems with a >= b and b >= 2 (the rest are the copy-a / answer-0 shortcuts).\n")
    rows = defaultdict(list)
    for r in runs:
        if r["protocol"] == "iid" and r["regime"] == "FULL":
            rows[r["learner"]].append(r)
    md.append("| learner | seeds | seen | add | sub | mul | mod | div | max | mod hard | div hard | mean | digit acc | params | stored vecs | train s |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, rs in sorted(rows.items()):
        ex = np.mean([np.array(r["grid_exact"]) for r in rs], axis=0)                # (6, 5, 5)
        dg = np.mean([np.array(r["grid_digit"]) for r in rs], axis=0)
        hard = {"mod": [], "div": []}
        for r in rs:
            ps, exact, _ = per_problem(r)
            for o in ("mod", "div"):
                m = (ps.op == A.OPS.index(o)) & (ps.a >= ps.b) & (ps.b >= 2)
                hard[o].append(float(exact[m].mean()))
        seen = mean_over([np.mean(r["seen_exact_by_op"]) for r in rs])
        md.append(f"| {name} | {len(rs)} | {pct(seen)} | " + " | ".join(pct(ex[i].mean()) for i in range(6)) + f" | {pct(np.mean(hard['mod']))} | {pct(np.mean(hard['div']))} | "
                  f"{pct(ex.mean())} | {pct(dg.mean())} | {int(mean_over([r['train'].get('params') for r in rs])):,} | {int(mean_over([r['train'].get('stored_vectors') for r in rs])):,} | "
                  f"{mean_over([r['train'].get('seconds') for r in rs]):.0f} |")
    md.append("")
    md.append("**mod and div split by case** (exact match %, unseen problems, mean over seeds): when a < b the answer of `mod` is a itself and the answer of `div` is 0; when b = 1 the answer is a or 0; "
              "only a >= b with b >= 2 needs real division. A learner that is good at the first two and bad at the last has found shortcuts, not the operation.\n")
    md.append("| learner | mod: a<b | mod: b=1 | mod: a>=b, b>=2 | div: a<b | div: b=1 | div: a>=b, b>=2 |")
    md.append("|---|---|---|---|---|---|---|")
    for name, rs in sorted(rows.items()):
        cells = []
        for o in ("mod", "div"):
            for sel in ("lt", "one", "hard"):
                vals = []
                for r in rs:
                    ps, exact, _ = per_problem(r)
                    m = ps.op == A.OPS.index(o)
                    m &= {"lt": ps.a < ps.b, "one": ps.b == 1, "hard": (ps.a >= ps.b) & (ps.b >= 2)}[sel]
                    vals.append(float(exact[m].mean()))
                cells.append(pct(float(np.mean(vals))))
        md.append(f"| {name} | " + " | ".join(cells) + " |")
    md.append("")


def sec_by_length(runs, md, fig_dir):
    md.append("## 2. How accuracy falls with the length of the numbers (FULL training, unseen problems)\n")
    md.append("Exact match % by the length of the longer operand; mean over operators add, sub, max (the 'local-rule' operators) and over mul and mod/div separately, mean over seeds.\n")
    rows = defaultdict(list)
    for r in runs:
        if r["protocol"] == "iid" and r["regime"] == "FULL":
            rows[r["learner"]].append(np.array(r["grid_exact"]))
    maxlen = np.maximum.outer(np.arange(1, 6), np.arange(1, 6))                     # (5, 5)
    groups = {"add/sub/max": [0, 1, 5], "mul": [2], "mod/div": [3, 4]}
    md.append("| learner | operators | len 1 | len 2 | len 3 | len 4 | len 5 |")
    md.append("|---|---|---|---|---|---|---|")
    curves = {}
    for name, grids in sorted(rows.items()):
        g = np.mean(grids, axis=0)
        for gname, ops in groups.items():
            vals = [float(g[ops][:, maxlen == k].mean()) for k in range(1, 6)]
            curves[(name, gname)] = vals
            md.append(f"| {name} | {gname} | " + " | ".join(pct(v) for v in vals) + " |")
    md.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
        for k, gname in enumerate(groups):
            for name in sorted(rows):
                ax[k].plot(range(1, 6), [100 * v for v in curves[(name, gname)]], marker="o", label=name)
            ax[k].set_title(f"{gname}: exact match vs longer operand's length", fontsize=9)
            ax[k].set_xlabel("digits of the longer operand")
            ax[k].grid(alpha=0.3)
        ax[0].set_ylabel("exact match %")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_by_length.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        md.append(f"(no figure: {e})\n")


def sec_length_change(runs, md):
    md.append("## 3. Change the length of the numbers: train on some lengths, test on others\n")
    md.append("HOLE = trained with no operand of 3 digits (lengths 1, 2, 4, 5); SHORT = trained on lengths 1-3 only. Exact match % on unseen problems whose lengths were / were NOT in training "
              "(FULL = everything was in training, for reference). Mean over operators and seeds; per-operator numbers for the new lengths follow. "
              "`iid` = all operators mixed, `seq` = operators one after the other.\n")
    md.append("| learner | protocol | regime | in-training lengths | new lengths | new: add | sub | mul | mod | div | max | digit acc on new lengths |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows = defaultdict(list)
    for r in runs:
        rows[(r["learner"], r["protocol"], r["regime"])].append(r)
    order = lambda kv: (kv[0][0], kv[0][1], ["FULL", "HOLE", "SHORT"].index(kv[0][2]))          # noqa: E731
    for (name, proto, regime), rs in sorted(rows.items(), key=order):
        ok = in_len(regime)
        ex = np.mean([np.array(r["grid_exact"]) for r in rs], axis=0)
        dg = np.mean([np.array(r["grid_digit"]) for r in rs], axis=0)
        new = "-" if ok.all() else pct(ex[:, ~ok].mean())
        per = " | ".join("-" if ok.all() else pct(ex[i][~ok].mean()) for i in range(6))
        md.append(f"| {name} | {proto} | {regime} ({len(rs)} seeds) | {pct(ex[:, ok].mean())} | {new} | {per} | {'-' if ok.all() else pct(dg[:, ~ok].mean())} |")
    md.append("")
    md.append("**What the missing lengths cost** (same learner, same protocol, same test cells: *with* = those lengths were in training (the FULL run), *without* = they were left out (the HOLE or SHORT run, "
              "which has the same number of training problems, 8,000 per operator, drawn only from the allowed lengths). Exact match %, mean over seeds). "
              "The columns above compare cells of different difficulty (a 4-5 digit sum is harder than a 3-digit one), so this gap, not the difference between the two columns above, "
              "is the like-for-like measure of how far a learner gets on lengths it never saw.\n")
    md.append("| learner | protocol | regime (seeds without / with) | six operators: with | without | gap | add: with | without | sub: with | without | max: with | without |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (name, proto, regime), rs in sorted(rows.items(), key=order):
        ok = in_len(regime)
        full = rows.get((name, proto, "FULL"))
        if ok.all() or not full:
            continue
        ex = np.mean([np.array(r["grid_exact"]) for r in rs], axis=0)
        fx = np.mean([np.array(r["grid_exact"]) for r in full], axis=0)
        cells = []
        for i in (0, 1, 5):
            cells += [pct(fx[i][~ok].mean()), pct(ex[i][~ok].mean())]
        with_, without = fx[:, ~ok].mean(), ex[:, ~ok].mean()
        md.append(f"| {name} | {proto} | {regime} ({len(rs)} / {len(full)}) | {pct(with_)} | {pct(without)} | {100 * (with_ - without):.1f} | " + " | ".join(cells) + " |")
    md.append("")


def routing_stats(run):
    ps, exact, raw = per_problem(run)
    if "expert" not in raw:
        return None
    e = raw["expert"]
    op, pos = ps.op[ps.pi], ps.pos
    maxlen = np.maximum(ps.la[ps.pi], ps.lb[ps.pi])
    cm = contingency(e, op)
    out_g = {}
    if "g_expert" in raw:
        ge = raw["g_expert"]
        cmg = contingency(ge, op)
        out_g = {"g_e": ge, "g_nmi_op": nmi(ge, op), "g_nmi_slot": nmi(ge, pos), "g_purity": float(cmg.max(0).sum() / cmg.sum()), "g_used": int(len(np.unique(ge)))}
    return {**out_g, "n_experts_used": int(len(np.unique(e))), "nmi_op": nmi(e, op), "nmi_slot": nmi(e, pos), "nmi_len": nmi(e, maxlen), "purity_op": float(cm.max(0).sum() / cm.sum()),
            "unknown_rate": float(raw["unknown"].mean()) if "unknown" in raw else None, "e": e, "op": op, "pos": pos, "maxlen": maxlen, "ps": ps, "exact": exact, "raw": raw}


def sec_routing(runs, md, fig_dir):
    md.append("## 4. GrowMoE: does it route to specialised experts, and is the routing consistent?\n")
    pools = [r for r in runs if r["learner"].startswith("GrowMoE")]
    if not pools:
        md.append("(no GrowMoE runs yet)\n")
        return
    md.append("Per run on the held-out grid (every example is routed to one expert). NMI = how much knowing the expert tells you about the operator / output slot / length "
              "(0 = nothing, 1 = everything); purity = share of an operator's examples that go to that operator's most common expert; `experts` = experts alive at the end of training.\n")
    md.append("| learner | protocol | regime | seed | experts | used on test | NMI operator | NMI slot | NMI length | purity (operator) | abstain (unknown) % |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    stats = {}
    for r in sorted(pools, key=lambda x: (x["learner"], x["protocol"], x["regime"], x["seed"])):
        s = routing_stats(r)
        if s is None:
            continue
        stats[(r["learner"], r["protocol"], r["regime"], r["seed"])] = s
        md.append(f"| {r['learner']} | {r['protocol']} | {r['regime']} | {r['seed']} | {r['train'].get('experts')} | {s['n_experts_used']} | {s['nmi_op']:.2f} | {s['nmi_slot']:.2f} | "
                  f"{s['nmi_len']:.2f} | {pct(s['purity_op'])} | {pct(s['unknown_rate'])} |")
    md.append("")
    grouped = [(k, s) for k, s in sorted(stats.items()) if "g_e" in s]
    if grouped:
        md.append("**Reading the same trained pool out in groups**: the output slots of one problem share the operands and the operator, so they can be answered together by the expert most of "
                  "them chose on their own (the stage-3b group vote, with its per-input veto). Same trained pool, no retraining. Exact match % on unseen problems, individual vs grouped, and the "
                  "routing statistics of the grouped read-out.\n")
        md.append("| learner | protocol | regime | seed | exact: individual | exact: grouped | NMI operator (grouped) | NMI slot (grouped) | purity (grouped) | experts used (grouped) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for (name, proto, regime, seed), s in grouped:
            run = next(r for r in pools if (r["learner"], r["protocol"], r["regime"], r["seed"]) == (name, proto, regime, seed))
            ind, grp = np.array(run["grid_exact"]).mean(), np.array(run["grid_exact_group"]).mean()
            md.append(f"| {name} | {proto} | {regime} | {seed} | {pct(ind)} | {pct(grp)} | {s['g_nmi_op']:.2f} | {s['g_nmi_slot']:.2f} | {pct(s['g_purity'])} | {s['g_used']} |")
        md.append("")
    md.append("**Consistency** (adjusted Rand index between the routing of two runs on the same test examples; 1 = identical partition, 0 = unrelated): across seeds (same learner, protocol, regime) "
              "and across training-length regimes (same learner, protocol, seed).\n")
    md.append("| learner | protocol | comparison | ARI |")
    md.append("|---|---|---|---|")
    keys = sorted(stats)
    rows_written = 0
    for (n1, p1, g1, s1) in keys:
        for (n2, p2, g2, s2) in keys:
            if (n1, p1) != (n2, p2):
                continue
            if g1 == g2 and s1 < s2:
                md.append(f"| {n1} | {p1} | {g1}: seed {s1} vs seed {s2} | {ari(stats[(n1, p1, g1, s1)]['e'], stats[(n2, p2, g2, s2)]['e']):.2f} |")
                rows_written += 1
            if s1 == s2 and g1 == "FULL" and g2 != "FULL":
                md.append(f"| {n1} | {p1} | seed {s1}: FULL vs {g2} | {ari(stats[(n1, p1, g1, s1)]['e'], stats[(n2, p2, g2, s2)]['e']):.2f} |")
                rows_written += 1
    for (n1, p1, g1, s1) in keys:
        for (n2, p2, g2, s2) in keys:
            if (n1, p1, g1) == (n2, p2, g2) and s1 < s2 and "g_e" in stats[(n1, p1, g1, s1)] and "g_e" in stats[(n2, p2, g2, s2)]:
                md.append(f"| {n1} (grouped read-out) | {p1} | {g1}: seed {s1} vs seed {s2} | {ari(stats[(n1, p1, g1, s1)]['g_e'], stats[(n2, p2, g2, s2)]['g_e']):.2f} |")
                rows_written += 1
    if not rows_written:
        md.append("| - | - | (needs two seeds or two regimes) | - |")
    md.append("")
    md.append("**Specialist reuse for new lengths** (HOLE / SHORT runs): share of the new-length examples routed to the expert that handles the same (operator, output slot) most often on the "
              "in-training lengths; abstention rate on in-training vs new lengths; AUROC of the novelty p-value for telling new-length examples from in-length ones (0.5 = cannot tell).\n")
    md.append("| learner | protocol | regime | seed | reuse of the in-length specialist | abstain on in-length % | abstain on new length % | AUROC new vs in length |")
    md.append("|---|---|---|---|---|---|---|---|")
    for (name, proto, regime, seed), s in sorted(stats.items()):
        if regime == "FULL":
            continue
        ps, e = s["ps"], s["e"]
        ok = in_len(regime)[ps.la[ps.pi] - 1, ps.lb[ps.pi] - 1]
        cell = s["op"] * A.N_POS + s["pos"]
        modal = {}
        for c in np.unique(cell[ok]):
            vals, cnt = np.unique(e[ok & (cell == c)], return_counts=True)
            modal[c] = vals[cnt.argmax()]
        newm = ~ok
        reuse = float(np.mean([e[i] == modal.get(cell[i], -9) for i in np.flatnonzero(newm)]))
        unk, pn = s["raw"].get("unknown"), s["raw"].get("p_nov")
        au = auroc(1 - pn[newm], 1 - pn[ok]) if pn is not None else None
        md.append(f"| {name} | {proto} | {regime} | {seed} | {pct(reuse)} | {pct(float(unk[ok].mean())) if unk is not None else '-'} | {pct(float(unk[newm].mean())) if unk is not None else '-'} | "
                  f"{'-' if au is None else f'{au:.2f}'} |")
    md.append("")
    md.append("**Saying 'I don't know'**: a problem counts as answered when none of its output slots was flagged unknown by the pool's novelty gate. Share of problems answered, exact match among the "
              "answered ones, and exact match over all of them, separately for problems whose lengths were in training and for new lengths.\n")
    md.append("| learner | protocol | regime | seed | in-length: answered | in-length: exact if answered | in-length: exact overall | new length: answered | new length: exact if answered | new length: exact overall |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for (name, proto, regime, seed), s in sorted(stats.items()):
        unk = s["raw"].get("unknown")
        if unk is None:
            continue
        ps, exact = s["ps"], s["exact"]
        answered = np.bincount(ps.pi, weights=unk.astype(np.float64), minlength=len(ps)) == 0
        okp = in_len(regime)[ps.la - 1, ps.lb - 1]
        cells = []
        for m in (okp, ~okp):
            if m.any():
                a = answered[m]
                cells += [pct(float(a.mean())), pct(float(exact[m][a].mean())) if a.any() else "-", pct(float(exact[m].mean()))]
            else:
                cells += ["-", "-", "-"]
        md.append(f"| {name} | {proto} | {regime} | {seed} | " + " | ".join(cells) + " |")
    md.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        chosen = [k for k in keys if k[2] == "FULL" and k[3] == 0][:4]
        if chosen:
            fig, ax = plt.subplots(2, len(chosen), figsize=(4.2 * len(chosen), 7), squeeze=False)
            for j, k in enumerate(chosen):
                s = stats[k]
                for i, (lab, key, labels) in enumerate((("operator", "op", NAMES), ("output slot", "pos", [str(p) for p in range(A.N_POS)]))):
                    cm = contingency(s["e"], s[key]).astype(float)
                    cm = cm / cm.sum(0, keepdims=True)
                    ax[i][j].imshow(cm, aspect="auto", cmap="viridis", vmin=0, vmax=1)
                    ue, uk = np.unique(s["e"]), np.unique(s[key])
                    ax[i][j].set_yticks(range(len(ue)))
                    ax[i][j].set_yticklabels([f"e{v}" for v in ue], fontsize=6)
                    ax[i][j].set_xticks(range(len(uk)))
                    ax[i][j].set_xticklabels([labels[v] for v in uk], fontsize=7, rotation=45)
                    ax[i][j].set_title(f"{k[0]} {k[1]} {k[2]} s{k[3]}\nshare of each {lab} sent to each expert", fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / "fig_routing.png", dpi=120)
            plt.close(fig)
    except Exception as e:
        md.append(f"(no routing figure: {e})\n")


def sec_continual(runs, md, fig_dir):
    md.append("## 5. Learning the operators one after the other (continual protocol): who forgets?\n")
    seq = [r for r in runs if r["protocol"] == "seq" and r["regime"] == "FULL"]
    if not seq:
        md.append("(no sequential runs yet)\n")
        return
    md.append("Operators arrive in blocks: max, add, sub, mod, div, mul. Exact match % on unseen problems after the LAST block; `forgetting` = mean over operators of (best score after any block - final score), "
              "so 0 means nothing was lost.\n")
    md.append("| learner | seeds | add | sub | mul | mod | div | max | mean | forgetting | experts | params | stored vecs |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    by = defaultdict(list)
    for r in seq:
        by[r["learner"]].append(r)
    for name, rs in sorted(by.items()):
        fin = np.mean([np.array(r["grid_exact"]).mean(axis=(1, 2)) for r in rs], axis=0)
        forg = []
        for r in rs:
            m = np.array([b["exact_by_op"] for b in r["blocks"]])             # (blocks, ops)
            forg.append(float(np.mean(m.max(0) - m[-1])))
        md.append(f"| {name} | {len(rs)} | " + " | ".join(pct(fin[i]) for i in range(6)) + f" | {pct(fin.mean())} | {pct(np.mean(forg))} | "
                  f"{mean_over([r['train'].get('experts') for r in rs]):.1f} | {int(mean_over([r['train'].get('params') for r in rs])):,} | {int(mean_over([r['train'].get('stored_vectors') for r in rs])):,} |")
    md.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = sorted(by)
        fig, ax = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.4), sharey=True, squeeze=False)
        for j, name in enumerate(names):
            m = np.mean([[b["exact_by_op"] for b in r["blocks"]] for r in by[name]], axis=0)
            for oi in L.OP_ORDER:
                ax[0][j].plot(range(1, 7), 100 * m[:, oi], marker="o", label=A.OPS[oi])
            ax[0][j].set_title(name, fontsize=8)
            ax[0][j].set_xlabel("after block (+max +add +sub +mod +div +mul)", fontsize=6)
            ax[0][j].grid(alpha=0.3)
        ax[0][0].set_ylabel("exact match % (unseen)")
        ax[0][-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_forgetting.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        md.append(f"(no forgetting figure: {e})\n")


def carries_add(a, b):
    carry, n = np.zeros(len(a), dtype=np.int64), np.zeros(len(a), dtype=np.int64)
    for k in range(5):
        s = (a // 10 ** k) % 10 + (b // 10 ** k) % 10 + carry
        carry = (s >= 10).astype(np.int64)
        n += carry
    return n


def borrows_sub(a, b):
    """For a >= b: the number of digit positions that need a borrow (analysis only; no learner ever sees this)."""
    borrow, n = np.zeros(len(a), dtype=np.int64), np.zeros(len(a), dtype=np.int64)
    for k in range(5):
        d = (a // 10 ** k) % 10 - (b // 10 ** k) % 10 - borrow
        borrow = (d < 0).astype(np.int64)
        n += borrow
    return n


def sec_slots(runs, md):
    md.append("## 6. What exactly was learned: accuracy per answer digit, and per length of carry chain (FULL training, unseen problems)\n")
    rows = defaultdict(list)
    for r in runs:
        if r["protocol"] == "iid" and r["regime"] == "FULL":
            rows[r["learner"]].append(r)
    if not rows:
        md.append("(no iid FULL runs yet)\n")
        return
    md.append("Accuracy of one answer slot, counted only on problems whose true answer actually has that digit (slot 0 = units, 1 = tens, ...; `sign` = the minus sign of a subtraction), mean over seeds.\n")
    cache = {}
    for name, rs in rows.items():
        cache[name] = [per_problem(r) for r in rs]
    for oname in ("add", "sub", "mul"):
        oi, width = A.OPS.index(oname), A.WIDTH[oname]
        slots = list(range(width)) + ([A.SIGN_POS] if oname == "sub" else [])
        md.append(f"**{oname}**\n")
        md.append("| learner | " + " | ".join("sign" if p == A.SIGN_POS else f"slot {p}" for p in slots) + " |")
        md.append("|---|" + "---|" * len(slots))
        for name in sorted(rows):
            vals = []
            for p in slots:
                accs = []
                for ps, _, raw in cache[name]:
                    m = (ps.op[ps.pi] == oi) & (ps.pos == p)
                    if p != A.SIGN_POS:
                        m &= ps.y != A.PAD
                    accs.append(float(raw["correct"][m].mean()) if m.any() else float("nan"))
                vals.append(float(np.nanmean(accs)) if not np.all(np.isnan(accs)) else float("nan"))
            md.append(f"| {name} | " + " | ".join(pct(v) for v in vals) + " |")
        md.append("")
    md.append("Exact match on problems whose operands both have at least 4 digits, by the number of carries (addition) or borrows (subtraction with a >= b) in the true calculation; "
              "n = number of problems in that group per run.\n")
    for oname, fn in (("add", carries_add), ("sub", borrows_sub)):
        oi = A.OPS.index(oname)
        md.append(f"**{oname}** by number of {'carries' if oname == 'add' else 'borrows'}\n")
        md.append("| learner | " + " | ".join(str(k) for k in range(6)) + " |")
        md.append("|---|" + "---|" * 6)
        for name in sorted(rows):
            res, cnt = np.full((len(cache[name]), 6), np.nan), np.zeros(6)
            for j, (ps, exact, _) in enumerate(cache[name]):
                m = (ps.op == oi) & (ps.la >= 4) & (ps.lb >= 4)
                if oname == "sub":
                    m &= ps.a >= ps.b
                n = fn(ps.a[m], ps.b[m])
                for k in range(6):
                    sel = n == k
                    if sel.any():
                        res[j, k] = float(exact[m][sel].mean())
                    cnt[k] = sel.sum()
            md.append(f"| {name} | " + " | ".join(pct(v) + f" (n={int(cnt[k])})" for k, v in enumerate(np.nanmean(res, axis=0))) + " |")
        md.append("")


def sec_curve(md, fig_dir):
    cdir = HERE / "results_curve"
    if not cdir.exists():
        return
    md.append("## 7. How much data does it take? (FULL lengths, seed 0, same test grid)\n")
    md.append("Training problems per operator, exact match % on unseen problems by operator (all lengths), and digit accuracy.\n")
    md.append("| learner | problems per operator | add | sub | mul | mod | div | max | digit acc | seen |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    series = {}
    for d in sorted(cdir.glob("n*"), key=lambda q: int(q.name[1:])):
        n = int(d.name[1:])
        for r in load(d):
            ex = np.array(r["grid_exact"]).mean(axis=(1, 2))
            series.setdefault(r["learner"], []).append((n, ex, float(np.mean(r["grid_digit"])), float(np.mean(r["seen_exact_by_op"]))))
    main = {r["learner"]: r for r in load(HERE / "results") if r["protocol"] == "iid" and r["regime"] == "FULL" and r["seed"] == 0 and r["n_per_op"] == 8000}
    for name in sorted(series):
        if name in main:
            m = main[name]
            series[name].append((8000, np.array(m["grid_exact"]).mean(axis=(1, 2)), float(np.mean(m["grid_digit"])), float(np.mean(m["seen_exact_by_op"]))))
        for n, ex, dg, seen in sorted(series[name], key=lambda q: q[0]):
            md.append(f"| {name} | {n:,} | " + " | ".join(pct(v) for v in ex) + f" | {pct(dg)} | {pct(seen)} |")
    md.append("")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for k, name in enumerate(sorted(series)):
            pts = sorted(series[name], key=lambda q: q[0])
            for oi in (0, 1, 2, 5):
                ax[k].plot([q[0] for q in pts], [100 * q[1][oi] for q in pts], marker="o", label=NAMES[oi])
            ax[k].set_xscale("log")
            ax[k].set_title(f"{name}: unseen exact match vs training problems per operator", fontsize=8)
            ax[k].set_xlabel("training problems per operator")
            ax[k].grid(alpha=0.3)
        ax[0].set_ylabel("exact match %")
        ax[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_data_curve.png", dpi=120)
        plt.close(fig)
    except Exception as e:
        md.append(f"(no curve figure: {e})\n")


def sec_bar(runs, fig_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows = defaultdict(list)
        for r in runs:
            if r["protocol"] == "iid" and r["regime"] == "FULL":
                rows[r["learner"]].append(np.array(r["grid_exact"]).mean(axis=(1, 2)))
        if not rows:
            return
        fig, ax = plt.subplots(figsize=(10, 3.8))
        w = 0.8 / len(rows)
        for k, (name, v) in enumerate(sorted(rows.items())):
            ax.bar(np.arange(6) + k * w, 100 * np.mean(v, axis=0), w, label=name)
        ax.set_xticks(np.arange(6) + 0.4 - w / 2)
        ax.set_xticklabels(NAMES)
        ax.set_ylabel("exact match % on unseen problems")
        ax.set_title("Unseen problems, lengths 1-5 (mean over seeds)")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_unseen_by_operator.png", dpi=120)
        plt.close(fig)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results")
    a = ap.parse_args()
    d = HERE / a.dir
    runs = load(d)
    combos = sorted({r["protocol"] + "/" + r["regime"] + "/" + r["learner"] for r in runs})
    md = ["# Arithmetic benchmark: results (generated by report_arith.py; do not edit by hand)\n", f"{len(runs)} runs: " + ", ".join(combos) + "\n"]
    sec_generalisation(runs, md)
    sec_by_length(runs, md, d)
    sec_length_change(runs, md)
    sec_routing(runs, md, d)
    sec_continual(runs, md, d)
    sec_slots(runs, md)
    sec_curve(md, d)
    sec_bar(runs, d)
    out = HERE / ("RESULTS_ARITH.md" if a.dir == "results" else f"RESULTS_ARITH_{a.dir}.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
