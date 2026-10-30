"""The evaluation battery (design section 6): every metric computed from a ``RunRecord`` (and the regimes of its stream), nothing re-run.

``regimes`` is the stream's ``regimes`` dict (each with ``domain``, ``kind``, ``name`` and, for arithmetic, ``meta["problem_id"]``). Seeds are aggregated with ``over_seeds``.
"""

from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import normalized_mutual_info_score

from growmoe.metrics import mean_ci

from .runner import RunRecord


def over_seeds(values: Sequence[float], level: float = 0.95) -> Tuple[float, float, float]:
    return mean_ci(values, level)


def _rid_order(regimes) -> List[int]:
    return sorted(regimes)


def final_accuracy(rec: RunRecord, regimes) -> dict:
    rids = _rid_order(regimes)
    last = rec.R[-1]
    per_domain: Dict[str, List[float]] = {}
    for i, r in enumerate(rids):
        per_domain.setdefault(regimes[r].domain, []).append(float(last[i]))
    return {"mean": float(np.mean(last)), "per_regime": [float(v) for v in last], "per_domain": {d: float(np.mean(v)) for d, v in per_domain.items()}}


def continual(rec: RunRecord, regimes, first_batch: int = 64) -> dict:
    """Forgetting, backward transfer and learning accuracy by the row after each regime's first visit; the prequential accuracy on the regime's first batch (before it was learned)."""
    rids = _rid_order(regimes)
    phase, regime = rec.arrays["train_phase"], rec.arrays["train_regime"]
    R = rec.R
    learned = [int(phase[regime == r].min()) if (regime == r).any() else R.shape[0] - 1 for r in rids]
    forgetting = [float(R[learned[i]:, i].max() - R[-1, i]) for i in range(len(rids))]
    bwt = float(np.mean([R[-1, i] - R[learned[i], i] for i in range(len(rids))]))
    pred, y = rec.arrays["train_pred"], rec.arrays["train_y"]
    first = []
    for r in rids:
        idx = np.flatnonzero(regime == r)[:first_batch]
        first.append(float((pred[idx] == y[idx]).mean()) if len(idx) else float("nan"))
    return {"forgetting": forgetting, "mean_forgetting": float(np.mean(forgetting)), "bwt": bwt, "learning_acc": [float(R[learned[i], i]) for i in range(len(rids))],
            "first_visit_prequential": first, "learned_row": learned}


def structure(rec: RunRecord, regimes, exclusive_share: float = 0.7, split_share: float = 0.2) -> dict:
    """How the experts divide the regimes, from the training assignments (samples no expert trained on do not count)."""
    reg, asg = rec.arrays["train_regime"], rec.arrays["train_assigned"]
    m = asg >= 0
    reg, asg = reg[m], asg[m]
    dom = np.array([regimes[int(r)].domain for r in reg]) if len(reg) else np.array([])
    owner: Dict[int, int] = {}
    for r in np.unique(reg):
        owner[int(r)] = Counter(asg[reg == r].tolist()).most_common(1)[0][0]
    n_of = Counter(asg.tolist())
    by_expert = {e: Counter(reg[asg == e].tolist()) for e in n_of}
    exclusive = [r for r, e in owner.items() if by_expert[e][r] / n_of[e] >= exclusive_share]
    owners_of: Dict[int, List[int]] = {}
    for r, e in owner.items():
        owners_of.setdefault(e, []).append(r)
    merged = sorted(sorted(rs) for rs in owners_of.values() if len(rs) >= 2)
    cross = 0
    for e in n_of:
        c = Counter(dom[asg == e].tolist())
        if sum(1 for v in c.values() if v / n_of[e] >= split_share) >= 2:
            cross += 1
    split = []
    for r in np.unique(reg):
        c = Counter(asg[reg == r].tolist())
        if sum(1 for v in c.values() if v / (reg == r).sum() >= split_share) >= 2:
            split.append(int(r))
    purity = float(sum(max(c.values()) for c in by_expert.values()) / max(len(asg), 1))
    return {"n_experts": int(rec.per_phase[-1]["n_experts"]) if rec.per_phase else len(n_of), "n_regimes": len(regimes), "owner": owner,
            "regimes_with_exclusive_expert": len(exclusive), "merged_groups": merged, "cross_domain_experts": cross, "split_regimes": sorted(split), "purity": purity,
            "nmi_regime": float(normalized_mutual_info_score(reg, asg)) if len(np.unique(asg)) > 1 or len(np.unique(reg)) > 1 else 0.0,
            "nmi_domain": float(normalized_mutual_info_score(dom, asg)) if len(dom) else 0.0}


def routing(rec: RunRecord, regimes) -> dict:
    """The accuracy lost at answer time: the routed answer against the answer of the regime's owner expert, on the samples where the owner exists."""
    t = rec.arrays
    y, pred, orc, exp, reg = t["test_y"], t["test_pred"], t["test_oracle_pred"], t["test_expert"], t["test_regime"]
    owner = {int(k): int(v) for k, v in rec.extra.get("owner_of_regime", {}).items()}
    has = orc >= 0
    own = np.array([owner.get(int(r), -2) for r in reg])
    out = {"routed_acc": float((pred == y).mean()), "n_oracle": int(has.sum())}
    if has.any():
        out.update({"routed_acc_on_oracle_samples": float((pred[has] == y[has]).mean()), "oracle_acc": float((orc[has] == y[has]).mean())})
        out["gap"] = out["oracle_acc"] - out["routed_acc_on_oracle_samples"]
    out["owner_route_rate"] = float((exp == own).mean())
    return out


def failures(rec: RunRecord, regimes) -> dict:
    """Every error of the final test classified: routed to a wrong expert whose regime owner would have been right (routing), routed to the owner and still wrong (expert), the regime has no owner."""
    t = rec.arrays
    y, pred, orc, exp, reg = t["test_y"], t["test_pred"], t["test_oracle_pred"], t["test_expert"], t["test_regime"]
    owner = {int(k): int(v) for k, v in rec.extra.get("owner_of_regime", {}).items()}
    counts = Counter()
    per_domain: Dict[str, Counter] = {}
    for i in range(len(y)):
        r = int(reg[i])
        if pred[i] == y[i]:
            kind = "correct"
        elif r not in owner:
            kind = "unowned"
        elif exp[i] != owner[r] and orc[i] == y[i]:
            kind = "routing_error"
        else:
            kind = "expert_error"
        counts[kind] += 1
        per_domain.setdefault(regimes[r].domain, Counter())[kind] += 1
    return {"counts": {k: counts.get(k, 0) for k in ("correct", "routing_error", "expert_error", "unowned")}, "n": int(len(y)),
            "false_refusals": int(t["test_unknown"].astype(bool).sum()), "per_domain": {d: dict(c) for d, c in per_domain.items()}}


def novelty(rec: RunRecord) -> dict:
    last = rec.ood[-1]
    out = {k[:-len("_unknown_rate")]: {"unknown_rate": last[k], "auroc": last[k[:-len("_unknown_rate")] + "_auroc"]} for k in last if k.endswith("_unknown_rate") and k != "known_unknown_rate"}
    rates = [o["known_unknown_rate"] for o in rec.ood]
    out["false_novel_last"], out["false_novel_max"] = rates[-1], max(rates)
    return out


def exact_match(rec: RunRecord, regimes) -> Dict[str, float]:
    """Per arithmetic operator: the share of problems whose every output slot is right (the metric of the arithmetic benchmark)."""
    t = rec.arrays
    out = {}
    for rid in _rid_order(regimes):
        rg = regimes[rid]
        if rg.kind != "digits" or "problem_id" not in rg.meta:
            continue
        m = t["test_regime"] == rid
        pid = np.asarray(rg.meta["problem_id"])
        wrong = np.bincount(np.unique(pid, return_inverse=True)[1], weights=(t["test_pred"][m] != t["test_y"][m]).astype(float))
        out[rg.name] = float((wrong == 0).mean())
    return out


def cost(rec: RunRecord) -> dict:
    r = rec.report
    return {"params": r.get("params"), "memory_mb": r.get("memory_bytes", 0) / 1e6, "stored_vectors": r.get("stored_vectors"), "predict_macs": r.get("predict_macs"),
            "gate_macs": r.get("gate_macs"), "expert_macs": r.get("expert_macs"), "seconds": rec.seconds}


def representation(x: torch.Tensor, regime: torch.Tensor, regimes, per_regime: int = 300) -> dict:
    """Geometry of the data on the sphere: cosine within a regime, between regimes of one domain, between domains; the effective dimension (participation ratio) of each domain."""
    g = torch.Generator().manual_seed(0)
    xs, rs = [], []
    for r in regime.unique().tolist():
        idx = (regime == r).nonzero().flatten()
        idx = idx[torch.randperm(len(idx), generator=g)[:per_regime]]
        xs.append(x[idx])
        rs.append(torch.full((len(idx),), r))
    x, regime = torch.cat(xs), torch.cat(rs)
    dom = np.array([regimes[int(r)].domain for r in regime])
    cos = (x @ x.T).numpy()
    same_r = (regime[:, None] == regime[None, :]).numpy()
    same_d = dom[:, None] == dom[None, :]
    eye = np.eye(len(x), dtype=bool)
    eff = {}
    for d in sorted(set(dom)):
        xd = x[torch.from_numpy(dom == d)]
        ev = torch.linalg.eigvalsh(torch.cov(xd.T)).clamp_min(0)
        eff[d] = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12))
    return {"within_regime_cos": float(cos[same_r & ~eye].mean()), "between_regimes_same_domain_cos": float(cos[same_d & ~same_r].mean()) if (same_d & ~same_r).any() else float("nan"),
            "between_domains_cos": float(cos[~same_d].mean()) if (~same_d).any() else float("nan"), "effective_dim": eff}
