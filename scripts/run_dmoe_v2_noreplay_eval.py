"""
`dmoe_v2_noreplay_routing`: novelty / routing / spawn audit and no-replay retention.
Protocol: results/dmoe_v2_noreplay_routing/PREREG.md (written before any run).

  --mode test  --steps N --seed-start a --seed-end b  : writes part_N_a_b.json
  --mode merge --steps N                              : writes results_N{N}.json (20 seeds)
"""
import argparse
import datetime
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_dmoe_v2_easy_eval as easy  # noqa: E402
import run_dmoe_v2_hard_control_eval as hc  # noqa: E402
from dmoe.continual import ContinualLearner  # noqa: E402

EXP_ID = "dmoe_v2_noreplay_routing"
OUT_DIR = ROOT / "results" / EXP_ID
LR = 4e-3
NO_MUTATION_INTERVAL = 10 ** 9


def _mean(v: List[float]) -> float:
    return sum(v) / len(v)


def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
    m = 0.5 * (p + q)
    return (0.5 * (p * torch.log((p + 1e-12) / (m + 1e-12))).sum()
            + 0.5 * (q * torch.log((q + 1e-12) / (m + 1e-12))).sum()).item()


@torch.no_grad()
def route_metrics(model: nn.Module, seed: int) -> Dict[str, float]:
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        return {}
    model.eval()
    n_layers = len(layers)
    feats: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]   # [layer][domain] -> [64, E]
    frozen_frac: List[List[float]] = [[] for _ in range(n_layers)]
    for d in range(3):
        x, _ = easy.generate_easy_domain_batch(d, 64, 16, seed * 100000 + 95000 + d)
        store: List[List[torch.Tensor]] = [[] for _ in range(n_layers)]
        hooks = [
            layer.moe.router_linear.register_forward_hook(lambda m, i, o, l=l: store[l].append(o.detach()))
            for l, layer in enumerate(layers)
        ]
        model(x, output_head="byte_chunk", update_stats=False)
        for h in hooks:
            h.remove()
        for l in range(n_layers):
            probs = torch.stack([F.softmax(o, dim=-1).mean(dim=1) for o in store[l]], dim=0).mean(dim=0)  # [64,E]
            feats[l].append(probs)
            fr = [layers[l].moe.expert_frozen[o.argmax(-1)].float().mean().item() for o in store[l]]
            frozen_frac[l].append(_mean(fr))
    between, floor, idacc = [], [], []
    for l in range(n_layers):
        means = [f.mean(dim=0) for f in feats[l]]
        pairs = [(0, 1), (0, 2), (1, 2)]
        between.append(_mean([_jsd(means[a], means[b]) for a, b in pairs]))
        floor.append(_mean([_jsd(f[:32].mean(0), f[32:].mean(0)) for f in feats[l]]))
        X = torch.cat(feats[l], dim=0)                      # [192,E]
        y = torch.arange(3).repeat_interleave(64)
        sums = torch.stack([X[y == c].sum(0) for c in range(3)])
        cnt = torch.tensor([64.0, 64.0, 64.0])
        correct = 0
        for i in range(X.size(0)):
            s2, c2 = sums.clone(), cnt.clone()
            s2[y[i]] -= X[i]
            c2[y[i]] -= 1
            cent = s2 / c2.unsqueeze(1)
            correct += int(((X[i].unsqueeze(0) - cent) ** 2).sum(1).argmin().item() == int(y[i]))
        idacc.append(correct / X.size(0))
    return {
        "route_domain_id_acc_mean_layers": _mean(idacc),
        "route_domain_id_acc_last_layer": idacc[-1],
        "route_jsd_between_mean_layers": _mean(between),
        "route_jsd_floor_mean_layers": _mean(floor),
        "route_jsd_gap_mean_layers": _mean(between) - _mean(floor),
        "route_jsd_gap_last_layer": between[-1] - floor[-1],
        "frac_top1_on_frozen_mean": _mean([_mean(f) for f in frozen_frac]),
        "n_layers_end": float(n_layers),
        "n_experts_total_end": float(sum(len(l.moe.experts) for l in layers)),
    }


def _total_experts(model: nn.Module) -> int:
    layers = getattr(model, "layers", [])
    return int(sum(len(l.moe.experts) for l in layers))


def _count_at(trace: List[Tuple[int, int]], initial: int, g: int) -> int:
    c = initial
    for step, cnt in trace:
        if step <= g:
            c = cnt
    return c


def train_run(model: nn.Module, seed: int, steps: int, schedule: List[Tuple[int, int]], fm: bool, replay_cap: int,
              mutation: bool = True, shuffle: bool = False, sequential_eval: bool = True):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    learner = ContinualLearner(
        model, opt, replay_buffer_capacity=replay_cap, replay_batch_size=8, distillation_weight=0.5,
        maturity_age=40, mutation_check_interval=25 if mutation else NO_MUTATION_INTERVAL,
        enable_fast_memory=fm, fast_memory_blend=0.55, fast_memory_reg=0.25,
    )
    initial_experts = _total_experts(model)
    novelty: List[float] = []
    count_trace: List[Tuple[int, int]] = []
    la_after_a = lb_after_b = acc_a_after_a = acc_b_after_b = 0.0
    for g, (d_id, st) in enumerate(schedule):
        x, y = easy.generate_easy_domain_batch(d_id, 16, 16, seed * 10000 + d_id * 1000 + st)
        if shuffle:
            pg = torch.Generator()
            pg.manual_seed(seed * 99999 + d_id * 1000 + st)
            y = y.reshape(-1)[torch.randperm(y.numel(), generator=pg)].reshape(y.shape)
        info = learner.step(x, y, output_head="byte_chunk")
        novelty.append(float(info["novelty_score"]))
        if info["mutation_report"] is not None:
            count_trace.append((g + 1, _total_experts(model)))
        if sequential_eval and g + 1 == steps:
            la_after_a, acc_a_after_a = easy.evaluate_domain(model, 0, seed, learner=learner)
        if sequential_eval and g + 1 == 2 * steps:
            lb_after_b, acc_b_after_b = easy.evaluate_domain(model, 1, seed, learner=learner)
    trace = {"novelty": novelty, "count_trace": count_trace, "initial_experts": initial_experts}
    return learner, trace, la_after_a, lb_after_b, acc_a_after_a, acc_b_after_b


def _metrics(model, seed, learner, la_a, lb_b, acc_a_after, acc_b_after) -> Dict[str, float]:
    la, aa = easy.evaluate_domain(model, 0, seed, learner=learner)
    lb, ab = easy.evaluate_domain(model, 1, seed, learner=learner)
    lc, ac = easy.evaluate_domain(model, 2, seed, learner=learner)
    return {
        "acc_A_final": aa, "acc_B_final": ab, "acc_C_final": ac,
        "avg_acc_final": (aa + ab + ac) / 3.0,
        "bwt_nats": 0.5 * ((la - la_a) + (lb - lb_b)),
        "acc_A_after_A": acc_a_after, "acc_B_after_B": acc_b_after,
        "retention_A": aa / max(acc_a_after, 1e-9),
        "retention_B": ab / max(acc_b_after, 1e-9),
    }


def _nov_jump(nov: List[float], steps: int) -> float:
    return _mean([_mean(nov[b:b + 5]) - _mean(nov[b - 10:b]) for b in (steps, 2 * steps)])


def _spawn_post_boundary(trace: dict, steps: int) -> float:
    init, tr = trace["initial_experts"], trace["count_trace"]
    return float(sum(_count_at(tr, init, b + 50) - _count_at(tr, init, b) for b in (steps, 2 * steps)))


def run_single_seed(seed: int, steps: int) -> Dict[str, Dict[str, float]]:
    seq = [(d, st) for d in range(3) for st in range(steps)]
    stat = [(0, st) for st in range(3 * steps)]
    joint = [(st % 3, st // 3) for st in range(3 * steps)]
    res: Dict[str, Dict[str, float]] = {}

    maj = _mean([torch.bincount(easy.generate_easy_domain_batch(d, 64, 16, seed * 100000 + 95000 + d)[1].reshape(-1),
                                minlength=256).max().item() / 1024.0 for d in range(3)])
    res["Reference_MajorityClass"] = {"avg_acc_final": maj}

    m = hc._make_dmoe_model(seed)
    nm = _metrics(m, seed, None, 0.0, 0.0, 0.0, 0.0)
    nm["bwt_nats"] = 0.0
    nm.update(route_metrics(m, seed))
    res["NullControl_Untrained"] = nm

    def run_cond(name, model, fm, replay_cap, mutation=True, shuffle=False, schedule=None, route=False, extra=False):
        learner, trace, la, lb, aa, ab = train_run(model, seed, steps, schedule or seq, fm, replay_cap,
                                                   mutation=mutation, shuffle=shuffle)
        r = _metrics(model, seed, learner, la, lb, aa, ab)
        if route:
            r.update(route_metrics(model, seed))
        if extra:
            r["novelty_jump"] = _nov_jump(trace["novelty"], steps)
            r["spawn_post_boundary"] = _spawn_post_boundary(trace, steps)
            r["experts_added_total"] = float(_count_at(trace["count_trace"], trace["initial_experts"], 3 * steps)
                                             - trace["initial_experts"])
        return learner, r

    _, res["Reference_Dense_JointMultitask"] = run_cond(
        "joint", easy._make_dense_for_learner(seed), False, 0, schedule=joint)
    _, res["Regular_NoReplay"] = run_cond("r0", easy._make_dense_for_learner(seed), False, 0)
    _, res["Regular_NoReplay_FM"] = run_cond("r1", easy._make_dense_for_learner(seed), True, 0)
    _, res["dMoE_NoReplay"] = run_cond("d0", hc._make_dmoe_model(seed), False, 0, route=True, extra=True)
    m_d1 = hc._make_dmoe_model(seed)
    l_d1, res["dMoE_NoReplay_FM"] = run_cond("d1", m_d1, True, 0)
    _, res["dMoE_NoReplay_NoMutation"] = run_cond("d2", hc._make_dmoe_model(seed), False, 0, mutation=False)
    _, res["Regular_Replay"] = run_cond("rr", easy._make_dense_for_learner(seed), False, 128)
    _, res["dMoE_Replay"] = run_cond("dr", hc._make_dmoe_model(seed), False, 128)
    _, res["NullControl_ShuffledTargets"] = run_cond("sh", hc._make_dmoe_model(seed), True, 0, shuffle=True)

    zs = [easy.evaluate_domain(m_d1, d, seed, learner=l_d1, blind_zero=True)[1] for d in range(3)]
    res["NullControl_BlindZeroInput"] = {"acc_A_final": zs[0], "acc_B_final": zs[1], "acc_C_final": zs[2],
                                         "avg_acc_final": _mean(zs)}

    m_s = hc._make_dmoe_model(seed)
    l_s, trace_s, *_ = train_run(m_s, seed, steps, stat, False, 0, sequential_eval=False)
    res["Control_Stationary_dMoE"] = {
        "novelty_jump": _nov_jump(trace_s["novelty"], steps),
        "spawn_post_boundary": _spawn_post_boundary(trace_s, steps),
        "experts_added_total": float(_count_at(trace_s["count_trace"], trace_s["initial_experts"], 3 * steps)
                                     - trace_s["initial_experts"]),
    }
    return res


def mode_test(a: int, b: int, steps: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for s in range(a, b):
        per_seed[str(s)] = run_single_seed(s, steps)
        print(f"N={steps} seed {s} done", flush=True)
    (out_dir / f"part_{steps}_{a}_{b}.json").write_text(
        json.dumps({"steps": steps, "per_seed": per_seed}, indent=2), encoding="utf-8")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        return f"unavailable: {err}"


def _paired(per_seed, seeds, ca, ka, cb, kb):
    diff = [per_seed[str(s)][ca][ka] - (per_seed[str(s)][cb][kb] if cb else 0.0) for s in seeds]
    n = len(diff)
    m = _mean(diff)
    sd = math.sqrt(sum((v - m) ** 2 for v in diff) / (n - 1))
    ci = 1.96 * sd / math.sqrt(n)
    return {"mean": round(m, 4), "ci95": round(ci, 4), "lo": round(m - ci, 4), "hi": round(m + ci, 4)}


def mode_merge(steps: int, command: str) -> None:
    parts = sorted(OUT_DIR.glob(f"part_{steps}_*.json"))
    merged: Dict[str, dict] = {}
    for p in parts:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["steps"] == steps
        merged.update(d["per_seed"])
    seeds = sorted(int(k) for k in merged)
    assert seeds == list(range(20)), f"expected seeds 0..19, got {seeds}"
    conds = list(merged["0"].keys())
    summary = {c: {k: hc._summarize([merged[str(s)][c][k] for s in seeds]) for k in merged["0"][c]} for c in conds}
    P = lambda *a: _paired(merged, seeds, *a)  # noqa: E731
    contrasts = {
        "Q1_novelty_jump_dMoE_minus_stationary": P("dMoE_NoReplay", "novelty_jump", "Control_Stationary_dMoE", "novelty_jump"),
        "Q2_spawn_post_boundary_dMoE_minus_stationary": P("dMoE_NoReplay", "spawn_post_boundary", "Control_Stationary_dMoE", "spawn_post_boundary"),
        "Q3a_route_domain_id_acc_trained_minus_untrained": P("dMoE_NoReplay", "route_domain_id_acc_mean_layers", "NullControl_Untrained", "route_domain_id_acc_mean_layers"),
        "Q3b_route_jsd_gap_trained": P("dMoE_NoReplay", "route_jsd_gap_mean_layers", None, None),
        "Q5_avgacc_dMoE_NoReplay_minus_Regular_NoReplay": P("dMoE_NoReplay", "avg_acc_final", "Regular_NoReplay", "avg_acc_final"),
        "Q6_avgacc_dMoE_Replay_minus_dMoE_NoReplay": P("dMoE_Replay", "avg_acc_final", "dMoE_NoReplay", "avg_acc_final"),
        "Q6_avgacc_Regular_Replay_minus_Regular_NoReplay": P("Regular_Replay", "avg_acc_final", "Regular_NoReplay", "avg_acc_final"),
        "Q7_avgacc_dMoE_NoReplay_minus_NoMutation": P("dMoE_NoReplay", "avg_acc_final", "dMoE_NoReplay_NoMutation", "avg_acc_final"),
        "extra_avgacc_dMoE_Replay_minus_Regular_Replay": P("dMoE_Replay", "avg_acc_final", "Regular_Replay", "avg_acc_final"),
    }
    q4 = {}
    for c in ("dMoE_NoReplay", "dMoE_NoReplay_FM", "Regular_NoReplay", "Regular_NoReplay_FM"):
        s = summary[c]
        ret, bwt, acc = s["retention_A"]["mean"], s["bwt_nats"]["mean"], s["avg_acc_final"]["mean"]
        q4[c] = {"retention_A": ret, "bwt_nats": bwt, "avg_acc": acc,
                 "pass_all": bool(ret >= 0.80 and bwt <= 0.10 and acc >= 0.80)}
    maj = summary["Reference_MajorityClass"]["avg_acc_final"]["mean"]
    nulls = {c: {"avg_acc": summary[c]["avg_acc_final"]["mean"], "limit": round(maj + 0.05, 4),
                 "pass": bool(summary[c]["avg_acc_final"]["mean"] <= maj + 0.05)}
             for c in ("NullControl_Untrained", "NullControl_ShuffledTargets", "NullControl_BlindZeroInput")}
    f1 = {c: summary[c]["avg_acc_final"]["max"] for c in conds if "avg_acc_final" in summary[c]}
    meta = {
        "exp_id": EXP_ID, "steps_per_domain": steps, "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__, "python": platform.python_version(), "command": command,
        "seeds": seeds, "lr": LR,
    }
    out = {"meta": meta, "summary": summary, "paired_contrasts": contrasts, "Q4_no_replay_criteria": q4,
           "null_gate_F2": nulls, "F1_max_acc_per_condition": f1}
    (OUT_DIR / f"results_N{steps}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"=== {EXP_ID} N={steps} (20 seeds) ===")
    for c in conds:
        s = summary[c]
        if "avg_acc_final" in s and "acc_A_final" in s:
            print(f"{c:36s} avg={s['avg_acc_final']['mean']:.4f} ±{s['avg_acc_final']['ci95']:.4f} "
                  f"A={s['acc_A_final']['mean']:.4f} B={s['acc_B_final']['mean']:.4f} C={s['acc_C_final']['mean']:.4f} "
                  f"retA={s['retention_A']['mean'] if 'retention_A' in s else float('nan'):.3f} "
                  f"BWT={s['bwt_nats']['mean'] if 'bwt_nats' in s else float('nan'):+.3f}")
    print("-- contrasts (mean [lo,hi]) --")
    for k, v in contrasts.items():
        print(f"{k}: {v['mean']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]")
    print("-- Q4 --", json.dumps(q4))
    print("-- null gate F2 --", json.dumps(nulls))
    for c in ("dMoE_NoReplay", "NullControl_Untrained", "Control_Stationary_dMoE"):
        s = summary[c]
        keys = [k for k in s if k.startswith(("route_", "frac_", "n_", "novelty", "spawn", "experts"))]
        print(c, {k: s[k]["mean"] for k in keys})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "merge"], required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-end", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    od = Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.mode == "test":
        mode_test(args.seed_start, args.seed_end, args.steps, od)
    else:
        mode_merge(args.steps, " ".join(sys.argv))

