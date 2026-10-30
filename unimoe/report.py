"""The evaluation report: tables, figures and the prediction scoreboard, generated from the run records and the streams; no number is typed.

``records`` maps (schedule, system, seed) to a ``RunRecord``; ``streams`` maps (schedule, seed) to the ``UStream`` it ran on (for the regimes' domains and the representation geometry).
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import torch

from . import analysis as AN
from .runner import RunRecord, load_record

Key = Tuple[str, str, int]
DOMAIN_ORDER = ("banking", "agnews", "newsgroups", "emotion", "sst2", "svamp", "arith")


def collect(runs_dir) -> Dict[Key, RunRecord]:
    out: Dict[Key, RunRecord] = {}
    for jp in sorted(Path(runs_dir).glob("*/*_s*.json")):
        stem = jp.stem
        system, seed = stem.rsplit("_s", 1)
        out[(jp.parent.name.replace("unimoe-", ""), system, int(seed))] = load_record(runs_dir, jp.parent.name, system, int(seed))
    return out


def _f(x, nd=3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(vals: Sequence[float], nd=3) -> str:
    vals = [v for v in vals if v is not None and not np.isnan(v)]
    if not vals:
        return "n/a"
    m, lo, hi = AN.over_seeds(vals)
    return f"{m:.{nd}f}" if len(vals) == 1 else f"{m:.{nd}f} [{max(lo, -9):.{nd}f}, {min(hi, 9):.{nd}f}]"


class View:
    """Convenience access: the per-seed values of a metric for one schedule and system."""

    def __init__(self, records: Dict[Key, RunRecord], streams: Dict[Tuple[str, int], object]) -> None:
        self.records, self.streams = records, streams
        self.schedules = sorted({k[0] for k in records}, key=("seq", "mix", "drift").index)
        self.systems = sorted({k[1] for k in records})
        self.seeds = sorted({k[2] for k in records})

    def recs(self, schedule: str, system: str) -> List[Tuple[int, RunRecord]]:
        return [(s, self.records[(schedule, system, s)]) for s in self.seeds if (schedule, system, s) in self.records]

    def vals(self, schedule: str, system: str, fn: Callable[[RunRecord, dict], float]) -> List[float]:
        return [fn(r, self.streams[(schedule, s)].regimes) for s, r in self.recs(schedule, system)]

    def mean(self, schedule: str, system: str, fn) -> float:
        v = [x for x in self.vals(schedule, system, fn) if x is not None and not np.isnan(x)]
        return float(np.mean(v)) if v else float("nan")


def _table(head: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def sections(v: View, tuning: dict) -> List[str]:
    L: List[str] = []
    fa = lambda r, rg: AN.final_accuracy(r, rg)["mean"]
    L += ["## A. Performance: final mean accuracy over the regimes (mean over seeds [95 % interval])", ""]
    for sch in v.schedules:
        rows = [[s, _ci(v.vals(sch, s, fa))] + [_f(v.mean(sch, s, lambda r, rg, d=d: AN.final_accuracy(r, rg)["per_domain"].get(d, float("nan"))), 2) for d in DOMAIN_ORDER] for s in v.systems if v.recs(sch, s)]
        L += [f"**{sch}** ({len(v.seeds)} seeds)", ""] + _table(["system", "final mean"] + list(DOMAIN_ORDER), rows)
    L += ["## B. Continual learning (sequential schedule)", ""]
    rows = []
    for s in v.systems:
        if not v.recs("seq", s):
            continue
        c = lambda key: v.vals("seq", s, lambda r, rg: float(np.nanmean(AN.continual(r, rg)[key])))
        rows.append([s, _ci(c("forgetting")), _ci(v.vals("seq", s, lambda r, rg: AN.continual(r, rg)["bwt"])), _ci(c("learning_acc")), _ci(c("first_visit_prequential"))])
    L += _table(["system", "mean forgetting", "backward transfer", "accuracy right after learning", "accuracy on a regime's first batch"], rows)
    L += ["## C. Structure: experts against regimes (UniMoE variants)", ""]
    rows = []
    for sch in v.schedules:
        for s in v.systems:
            if s in ("shipped", "U-real", "U-deep", "U-recipe") and v.recs(sch, s):
                g = lambda k, sch=sch, s=s: v.vals(sch, s, lambda r, rg: float(AN.structure(r, rg)[k]) if not isinstance(AN.structure(r, rg)[k], (list, dict)) else float(len(AN.structure(r, rg)[k])))
                rows.append([sch, s, _ci(g("n_experts"), 1), str(len(v.streams[(sch, v.seeds[0])].regimes)), _ci(g("regimes_with_exclusive_expert"), 1), _ci(g("merged_groups"), 1),
                             _ci(g("cross_domain_experts"), 1), _ci(g("split_regimes"), 1), _ci(g("purity"), 2), _ci(g("nmi_domain"), 2)])
    L += _table(["schedule", "system", "experts", "regimes", "regimes with own expert", "merged groups", "cross-domain experts", "split regimes", "purity", "NMI(expert, domain)"], rows)
    L += ["## D. Routing at answer time (accuracy lost against the owner expert's own answer)", ""]
    rows = []
    for sch in v.schedules:
        for s in ("shipped", "U-real", "U-deep", "U-recipe"):
            if v.recs(sch, s):
                g = lambda k, sch=sch, s=s: v.vals(sch, s, lambda r, rg: AN.routing(r, rg).get(k, float("nan")))
                rows.append([sch, s, _ci(g("routed_acc_on_oracle_samples")), _ci(g("oracle_acc")), _ci(g("gap")), _ci(g("owner_route_rate"))])
    L += _table(["schedule", "system", "routed accuracy", "owner-routed accuracy", "gap", "share sent to the regime's owner"], rows)
    L += ["## E. Novelty and out-of-distribution text (at the end of the sequential schedule)", ""]
    rows = []
    for s in v.systems:
        if v.recs("seq", s) and s not in ("oracle_experts",):
            n = lambda path: v.vals("seq", s, lambda r, rg: AN.novelty(r).get(path[0], {}).get(path[1], float("nan")) if len(path) == 2 else AN.novelty(r)[path[0]])
            rows.append([s, _ci(n(("false_novel_last",))), _ci(n(("stories", "auroc"))), _ci(n(("wiki", "auroc"))), _ci(n(("science_questions", "auroc"))), _ci(n(("stories", "unknown_rate")), 2),
                         _ci(n(("wiki", "unknown_rate")), 2), _ci(n(("science_questions", "unknown_rate")), 2)])
    L += _table(["system", "known flagged unknown", "AUROC stories", "AUROC wiki", "AUROC science", "found stories", "found wiki", "found science"], rows)
    L += ["## F. Cost", ""]
    rows = []
    for sch in v.schedules:
        for s in v.systems:
            if v.recs(sch, s):
                g = lambda k, sch=sch, s=s: v.vals(sch, s, lambda r, rg: float(AN.cost(r).get(k) or 0))
                rows.append([sch, s, _ci(g("params"), 0), _ci(g("memory_mb"), 2), _ci(g("predict_macs"), 0), _ci(g("seconds"), 0)])
    L += _table(["schedule", "system", "parameters", "memory (MB)", "multiply-adds per prediction", "seconds"], rows)
    L += ["## G. Arithmetic: exact match per problem (sequential schedule)", ""]
    ops = ["add", "sub", "mul", "mod", "div", "max"]
    rows = []
    for s in v.systems:
        if v.recs("seq", s):
            rows.append([s] + [_ci(v.vals("seq", s, lambda r, rg, o=o: AN.exact_match(r, rg).get(o, float("nan"))), 2) for o in ops])
    L += _table(["system"] + ops, rows)
    L += ["## H. Failure analysis of the final test (share of all test samples; sequential schedule)", ""]
    rows = []
    for s in ("shipped", "U-real", "U-deep", "U-recipe"):
        if v.recs("seq", s):
            for d in DOMAIN_ORDER:
                tot = lambda r, rg, d=d: sum(AN.failures(r, rg)["per_domain"].get(d, {}).values())
                share = lambda kind, s=s, d=d: v.vals("seq", s, lambda r, rg: AN.failures(r, rg)["per_domain"].get(d, {}).get(kind, 0) / max(sum(AN.failures(r, rg)["per_domain"].get(d, {}).values()), 1))
                rows.append([s, d, _ci(share("correct"), 2), _ci(share("routing_error"), 2), _ci(share("expert_error"), 2), _ci(share("unowned"), 2)])
    L += _table(["system", "domain", "correct", "routing error", "expert error", "regime without owner"], rows)
    L += ["## I. Representation geometry (training data on the sphere)", ""]
    seed0 = v.seeds[0]
    st = v.streams[("seq", seed0)]
    x = torch.cat([p.x for p in st.phases])
    reg = torch.cat([p.regime for p in st.phases])
    g = AN.representation(x, reg, st.regimes)
    L += [f"Seed {seed0}: mean cosine within a regime {g['within_regime_cos']:.3f}, between regimes of one domain {g['between_regimes_same_domain_cos']:.3f}, between domains {g['between_domains_cos']:.3f}.", ""]
    L += _table(["domain", "effective dimension (participation ratio)"], [[d, _f(g["effective_dim"].get(d), 1)] for d in DOMAIN_ORDER if d in g["effective_dim"]])
    L += ["## J. Tuned recipes (seed 499, sequential schedule)", "", "```", json.dumps(tuning, indent=1), "```", ""]
    return L


# ------------------------------------------------------------------------------------------------------------------------------ predictions
def scoreboard(v: View) -> List[str]:
    """Each prediction written in research/unimoe/PROGRESS.md before the evaluation, scored against the measurement (the bar is the one declared there)."""
    rows = []
    fa = lambda s, sch="seq": v.mean(sch, s, lambda r, rg: AN.final_accuracy(r, rg)["mean"])

    def add(text: str, value, ok) -> None:
        rows.append([text, _f(value) if isinstance(value, float) else str(value), "held" if ok else "MISSED"])

    ranges = {"joint": (0.68, 0.78), "oracle_experts": (0.62, 0.78), "replay_eqmem": (0.52, 0.68), "U-real": (0.52, 0.68), "shipped": (0.48, 0.62), "knn_eqmem": (0.52, 0.66), "lda": (0.42, 0.60), "finetune": (0.08, 0.25)}
    for s, (lo, hi) in ranges.items():
        if v.recs("seq", s):
            x = fa(s)
            add(f"sequential final mean of {s} in {lo}-{hi}", x, lo <= x <= hi)
    if v.recs("seq", "U-real") and v.recs("seq", "shipped"):
        d = fa("U-real") - fa("shipped")
        add("U-real above shipped by +0.01 to +0.05", d, 0.01 <= d <= 0.05)
    if v.recs("seq", "U-deep") and v.recs("seq", "U-real"):
        d = fa("U-deep") - fa("U-real")
        add("U-deep within 0.02 of U-real", d, abs(d) <= 0.02)
    if v.recs("seq", "U-recipe") and v.recs("seq", "U-real"):
        d = fa("U-recipe") - fa("U-real")
        add("U-recipe above U-real by at least 0.02", d, d >= 0.02)
    if v.recs("seq", "U-real") and v.recs("seq", "replay_eqmem"):
        d = fa("U-real") - fa("replay_eqmem")
        add("the pool beats the equal-memory replay network (predicted 40 %)", d, d > 0)
    for s in ("shipped", "U-real"):
        if v.recs("seq", s):
            n = v.mean("seq", s, lambda r, rg: float(AN.structure(r, rg)["n_experts"]))
            fewer = all(AN.structure(r, rg)["n_experts"] < len(rg) for _, r in v.recs("seq", s) for rg in [v.streams[("seq", _)].regimes])
            add(f"{s}: experts at the end in 9-16 and fewer than the 22 regimes in every seed", n, 9 <= n <= 16 and fewer)
    if v.recs("seq", "U-real"):
        cd = v.mean("seq", "U-real", lambda r, rg: float(AN.structure(r, rg)["cross_domain_experts"]))
        add("at least one expert holds regimes of two domains (U-real, mean over seeds >= 1)", cd, cd >= 1)
        gap = v.mean("seq", "U-real", lambda r, rg: AN.routing(r, rg).get("gap", float("nan")))
        add("oracle-routed accuracy exceeds routed accuracy by at least 0.03 (U-real)", gap, gap >= 0.03)
        for o in ("stories", "wiki", "science_questions"):
            a = v.mean("seq", "U-real", lambda r, rg, o=o: AN.novelty(r)[o]["auroc"])
            add(f"AUROC on {o} in 0.75-0.92 (U-real)", a, 0.75 <= a <= 0.92)
        found = max(v.mean("seq", "U-real", lambda r, rg, o=o: AN.novelty(r)[o]["unknown_rate"]) for o in ("stories", "wiki", "science_questions"))
        add("the unknown flag finds under 0.30 of every OOD set (U-real)", found, found < 0.30)
        fn = v.mean("seq", "U-real", lambda r, rg: AN.novelty(r)["false_novel_last"])
        add("false-novel rate on known data under 0.05 (U-real)", fn, fn < 0.05)
        fg = v.mean("seq", "U-real", lambda r, rg: AN.continual(r, rg)["mean_forgetting"])
        add("pool forgetting in 0.03-0.20 (U-real)", fg, 0.03 <= fg <= 0.20)
    if v.recs("seq", "finetune"):
        ft = v.mean("seq", "finetune", lambda r, rg: AN.continual(r, rg)["mean_forgetting"])
        add("finetune forgetting above 0.40", ft, ft > 0.40)
    if v.recs("mix", "U-real"):
        n = v.mean("mix", "U-real", lambda r, rg: float(AN.structure(r, rg)["n_experts"]))
        add("mixed schedule: the pool founds 1-4 experts (U-real)", n, 1 <= n <= 4)
        if v.recs("mix", "replay_eqmem"):
            d = abs(v.mean("mix", "U-real", lambda r, rg: AN.final_accuracy(r, rg)["mean"]) - v.mean("mix", "replay_eqmem", lambda r, rg: AN.final_accuracy(r, rg)["mean"]))
            add("mixed schedule: U-real within 0.03 of the equal-memory replay network (predicted 55 %)", d, d <= 0.03)
    if v.recs("seq", "U-real"):
        ex = {o: v.mean("seq", "U-real", lambda r, rg, o=o: AN.exact_match(r, rg).get(o, float("nan"))) for o in ("add", "sub", "max", "mul", "mod", "div")}
        ok = all(0.25 <= ex[o] <= 0.50 for o in ("add", "sub", "max")) and all(ex[o] < 0.15 for o in ("mul", "mod", "div"))
        add("exact match 0.25-0.50 for add/sub/max and under 0.15 for mul/mod/div (U-real)", "add %.2f sub %.2f max %.2f mul %.2f mod %.2f div %.2f" % tuple(ex[o] for o in ("add", "sub", "max", "mul", "mod", "div")), ok)
    return ["## Prediction scoreboard (predictions of `research/unimoe/PROGRESS.md`, written before the evaluation)", ""] + _table(["prediction", "measured", "verdict"], rows)


# ------------------------------------------------------------------------------------------------------------------------------ figures
def figures(v: View, out_dir) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    seed = v.seeds[0]
    show = [s for s in ("shipped", "U-real", "replay_eqmem") if v.recs("seq", s)]
    if show:
        fig, ax = plt.subplots(1, len(show), figsize=(5 * len(show), 4.5), squeeze=False)
        names = [r.name for r in v.streams[("seq", seed)].regimes.values()]
        for a, s in zip(ax[0], show):
            im = a.imshow(v.records[("seq", s, seed)].R, vmin=0, vmax=1, aspect="auto", cmap="viridis")
            a.set_title(f"{s} (seed {seed}, sequential)")
            a.set_xlabel("regime")
            a.set_ylabel("after phase")
            a.set_xticks(range(len(names)))
            a.set_xticklabels(names, rotation=90, fontsize=6)
        fig.colorbar(im, ax=ax.ravel().tolist(), shrink=0.8, label="accuracy")
        p = out_dir / "accuracy_matrices.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        made.append(p.name)
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in ("shipped", "U-real", "U-deep", "U-recipe"):
        rs = v.recs("seq", s)
        if rs:
            ax.plot(np.mean([[p["n_experts"] for p in r.per_phase] for _, r in rs], axis=0), label=s)
    st = v.streams[("seq", seed)]
    seen = np.cumsum([len(set(p.regime[p.first].tolist())) for p in st.phases])
    ax.plot(seen, "k--", label="regimes met so far")
    ax.set_xlabel("phase")
    ax.set_ylabel("experts")
    ax.legend()
    p = out_dir / "experts_over_time.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    made.append(p.name)
    fig, ax = plt.subplots(figsize=(9, 4))
    systems = [s for s in v.systems if v.recs("seq", s)]
    w = 0.8 / max(len(systems), 1)
    for i, s in enumerate(systems):
        ax.bar(np.arange(len(DOMAIN_ORDER)) + i * w, [v.mean("seq", s, lambda r, rg, d=d: AN.final_accuracy(r, rg)["per_domain"].get(d, float("nan"))) for d in DOMAIN_ORDER], w, label=s)
    ax.set_xticks(np.arange(len(DOMAIN_ORDER)) + 0.4)
    ax.set_xticklabels(DOMAIN_ORDER)
    ax.set_ylabel("final accuracy (sequential)")
    ax.legend(fontsize=6, ncol=2)
    p = out_dir / "accuracy_by_domain.png"
    fig.savefig(p, dpi=110, bbox_inches="tight")
    plt.close(fig)
    made.append(p.name)
    return made


def build(records: Dict[Key, RunRecord], streams: Dict[Tuple[str, int], object], tuning: dict, out_md, fig_dir, title: str = "UniMoE Stage 1: evaluation") -> str:
    v = View(records, streams)
    figs = figures(v, fig_dir)
    head = [f"# {title}", "", f"Seeds {v.seeds}; schedules {v.schedules}; systems {v.systems}. All seeds are dev seeds; nothing here is a pre-registered confirmation. Every number is generated from the run records. "
            "The seeds loop of the driver runs every seed and the tables show all of them (mean and interval), so the repository's best-of-seeds warning on it does not apply.", ""]
    text = "\n".join(head + scoreboard(v) + sections(v, tuning) + ["## Figures", ""] + [f"- `{Path(fig_dir).name}/{f}`" for f in figs] + [""])
    Path(out_md).write_text(text, encoding="utf-8")
    return text
