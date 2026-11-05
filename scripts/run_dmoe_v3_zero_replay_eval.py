"""
`dmoe_v3_zero_replay`: Continual learning evaluation without replay.
Pre-registered protocol: results/dmoe_v3_zero_replay/PREREG.md.

Modes:
  --mode test --steps 240 --seed-start a --seed-end b : evaluates seeds [a, b) and writes part_240_a_b.json
  --mode merge --steps 240                            : merges parts into results.json and audits falsifiers
"""

import argparse
import datetime
import json
import math
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_dmoe_v2_easy_eval as easy
import run_dmoe_v2_hard_control_eval as hc
from dmoe.continual import ContinualLearner
from dmoe.model import SelfStabilizingDynamicMorphicMoE

EXP_ID = "dmoe_v3_zero_replay"
OUT_DIR = ROOT / "results" / EXP_ID
LR = 4e-3
DOMAINS = [0, 1, 2]
NO_MUTATION_INTERVAL = 10 ** 9


def _mean(v: List[float]) -> float:
    return sum(v) / len(v) if v else 0.0


def _summarize(vals: List[float]) -> Dict[str, float]:
    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": 0.0, "min": 0.0, "max": 0.0}
    m = _mean(vals)
    if n > 1:
        var = sum((x - m) ** 2 for x in vals) / (n - 1)
        sd = math.sqrt(max(var, 0.0))
        ci = 1.96 * sd / math.sqrt(n)
    else:
        sd = 0.0
        ci = 0.0
    return {
        "mean": round(m, 4),
        "std": round(sd, 4),
        "ci95": round(ci, 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        return f"unavailable: {err}"


def _make_dmoe_v3(seed: int) -> SelfStabilizingDynamicMorphicMoE:
    random.seed(seed)
    torch.manual_seed(seed)
    return SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=2,
        initial_experts=1,
        tokenizer_free=True,
        chunk_size=4,
        enable_local_lda=True,
        enable_setgate=True,
    )


def _make_dmoe_v2(seed: int) -> SelfStabilizingDynamicMorphicMoE:
    random.seed(seed)
    torch.manual_seed(seed)
    return SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=128,
        initial_layers=2,
        initial_experts=3,
        tokenizer_free=True,
        chunk_size=4,
        enable_local_lda=False,
        enable_setgate=False,
    )


@torch.no_grad()
def _eval_domain(
    model: nn.Module,
    learner: Optional[ContinualLearner],
    d: int,
    seed: int,
    blind_zero: bool = False,
) -> Tuple[float, float]:
    x, y = easy.generate_easy_domain_batch(d, batch_size=64, seq_len=16, seed=seed * 100000 + 95000 + d)
    if blind_zero:
        x = torch.zeros_like(x)

    model.eval()
    if learner is not None:
        logits = learner.predict_logits(x, output_head="byte_chunk")
    else:
        out = model(x, output_head="byte_chunk", update_stats=False)
        logits = out["output"]

    flat_logits = logits.reshape(-1, 256)
    flat_y = y.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_y).item()
    pred = flat_logits.argmax(dim=-1)
    acc = (pred == flat_y).float().mean().item()
    return loss, acc


def run_condition_seed(cond_name: str, seed: int, steps_per_domain: int) -> Dict[str, float]:
    if cond_name == "NullControl_Untrained":
        model = _make_dmoe_v3(seed)
        l_a, a_a = _eval_domain(model, None, 0, seed)
        l_b, a_b = _eval_domain(model, None, 1, seed)
        l_c, a_c = _eval_domain(model, None, 2, seed)
        avg_acc = (a_a + a_b + a_c) / 3.0
        return {
            "loss_A_final": l_a, "acc_A_final": a_a,
            "loss_B_final": l_b, "acc_B_final": a_b,
            "loss_C_final": l_c, "acc_C_final": a_c,
            "avg_acc_final": avg_acc, "retention_A": 1.0, "bwt_nats": 0.0,
        }

    if cond_name == "Baseline_RegularTransformer_NoReplay":
        model = easy._make_dense_for_learner(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        learner = ContinualLearner(model, opt, replay_buffer_capacity=0, mutation_check_interval=NO_MUTATION_INTERVAL)
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False)

    if cond_name == "Baseline_RegularTransformer_Replay128":
        model = easy._make_dense_for_learner(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        learner = ContinualLearner(
            model, opt, replay_buffer_capacity=128, replay_batch_size=8,
            distillation_weight=0.5, mutation_check_interval=NO_MUTATION_INTERVAL,
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False)

    if cond_name == "Baseline_dMoE_v2_NoReplay":
        model = _make_dmoe_v2(seed)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        learner = ContinualLearner(
            model, opt, replay_buffer_capacity=0, maturity_age=40,
            mutation_check_interval=25, enable_fast_memory=False,
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False)

    if cond_name == "NullControl_ShuffledTargets":
        model = _make_dmoe_v3(seed)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        learner = ContinualLearner(
            model, opt, replay_buffer_capacity=0, enable_local_lda=True,
            enable_setgate=True, freeze_backbone_on_domain=True,
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=True)

    if cond_name == "dMoE_v3_ZeroReplay":
        model = _make_dmoe_v3(seed)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        learner = ContinualLearner(
            model, opt, replay_buffer_capacity=0, enable_local_lda=True,
            enable_setgate=True, freeze_backbone_on_domain=True,
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False)

    if cond_name == "NullControl_BlindZeroInput":
        # Train v3 normally, then evaluate on zeroed inputs
        model = _make_dmoe_v3(seed)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        learner = ContinualLearner(
            model, opt, replay_buffer_capacity=0, enable_local_lda=True,
            enable_setgate=True, freeze_backbone_on_domain=True,
        )
        _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False)
        l_a, a_a = _eval_domain(model, learner, 0, seed, blind_zero=True)
        l_b, a_b = _eval_domain(model, learner, 1, seed, blind_zero=True)
        l_c, a_c = _eval_domain(model, learner, 2, seed, blind_zero=True)
        avg_acc = (a_a + a_b + a_c) / 3.0
        return {
            "loss_A_final": l_a, "acc_A_final": a_a,
            "loss_B_final": l_b, "acc_B_final": a_b,
            "loss_C_final": l_c, "acc_C_final": a_c,
            "avg_acc_final": avg_acc, "retention_A": 0.0, "bwt_nats": 0.0,
        }

    raise ValueError(f"Unknown condition: {cond_name}")


def _train_sequential(
    model: nn.Module,
    learner: ContinualLearner,
    seed: int,
    steps_per_domain: int,
    shuffle_targets: bool = False,
) -> Dict[str, float]:
    initial_metrics: Dict[int, Tuple[float, float]] = {}

    for d in DOMAINS:
        if d > 0 and hasattr(learner, "domain_transition"):
            learner.domain_transition(new_domain_id=d)

        for st in range(steps_per_domain):
            x, y = easy.generate_easy_domain_batch(d, batch_size=16, seq_len=16, seed=seed * 10000 + d * 1000 + st)
            if shuffle_targets:
                perm = torch.randperm(y.numel())
                y = y.reshape(-1)[perm].reshape(y.shape)
            learner.step(x, y, output_head="byte_chunk")

        # Evaluate right at domain completion
        with torch.no_grad():
            l_init, a_init = _eval_domain(model, learner, d, seed)
            initial_metrics[d] = (l_init, a_init)

    # Freeze final domain
    if hasattr(learner, "domain_transition"):
        learner.domain_transition(new_domain_id=len(DOMAINS), spawn_new_expert=False)

    # Evaluate all domains at completion
    final_losses = []
    final_accs = []
    for d in DOMAINS:
        l_fin, a_fin = _eval_domain(model, learner, d, seed)
        final_losses.append(l_fin)
        final_accs.append(a_fin)

    l_a_init, a_a_init = initial_metrics[0]
    l_b_init, a_b_init = initial_metrics[1]
    l_a_fin, a_a_fin = final_losses[0], final_accs[0]
    l_b_fin, a_b_fin = final_losses[1], final_accs[1]
    l_c_fin, a_c_fin = final_losses[2], final_accs[2]

    avg_acc_final = _mean(final_accs)
    retention_a = a_a_fin / max(a_a_init, 1e-6)
    bwt_nats = 0.5 * ((l_a_fin - l_a_init) + (l_b_fin - l_b_init))

    return {
        "acc_A_initial": a_a_init,
        "loss_A_initial": l_a_init,
        "acc_B_initial": a_b_init,
        "loss_B_initial": l_b_init,
        "acc_A_final": a_a_fin,
        "loss_A_final": l_a_fin,
        "acc_B_final": a_b_fin,
        "loss_B_final": l_b_fin,
        "acc_C_final": a_c_fin,
        "loss_C_final": l_c_fin,
        "avg_acc_final": avg_acc_final,
        "retention_A": retention_a,
        "bwt_nats": bwt_nats,
    }


CONDITIONS = [
    "NullControl_Untrained",
    "NullControl_ShuffledTargets",
    "NullControl_BlindZeroInput",
    "Baseline_RegularTransformer_NoReplay",
    "Baseline_RegularTransformer_Replay128",
    "Baseline_dMoE_v2_NoReplay",
    "dMoE_v3_ZeroReplay",
]


def mode_test(seed_start: int, seed_end: int, steps: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    for seed in range(seed_start, seed_end):
        per_seed[str(seed)] = {}
        print(f"--- Running Seed {seed} ---")
        for cond in CONDITIONS:
            t0 = datetime.datetime.now()
            res = run_condition_seed(cond, seed, steps)
            dur = (datetime.datetime.now() - t0).total_seconds()
            per_seed[str(seed)][cond] = res
            print(f"  [{dur:.1f}s] {cond:38s}: avg_acc={res['avg_acc_final']:.4f}, retA={res.get('retention_A', float('nan')):.4f}")

    part_file = out_dir / f"part_{steps}_{seed_start}_{seed_end}.json"
    part_file.write_text(json.dumps({"steps": steps, "per_seed": per_seed}, indent=2), encoding="utf-8")
    print(f"Wrote partition: {part_file}")


def mode_merge(steps: int, command: str) -> None:
    parts = sorted(OUT_DIR.glob(f"part_{steps}_*.json"))
    merged: Dict[str, dict] = {}
    for p in parts:
        d = json.loads(p.read_text(encoding="utf-8"))
        assert d["steps"] == steps
        merged.update(d["per_seed"])
    seeds = sorted(int(k) for k in merged)
    assert seeds == list(range(20)), f"expected seeds 0..19, got {seeds}"

    summary = {
        c: {k: _summarize([merged[str(s)][c][k] for s in seeds]) for k in merged["0"][c]}
        for c in CONDITIONS
    }

    # Audit Falsifiers
    v3_summary = summary["dMoE_v3_ZeroReplay"]
    dense_noreplay = summary["Baseline_RegularTransformer_NoReplay"]
    maj_baseline = 0.1722  # empirical majority class baseline
    null_threshold = 0.2222

    f1_val = v3_summary["avg_acc_final"]["mean"]
    f1_pass = f1_val >= 0.7000

    f2_val = v3_summary["retention_A"]["mean"]
    f2_pass = f2_val >= 0.7500

    null_accs = {
        c: summary[c]["avg_acc_final"]["mean"]
        for c in ("NullControl_Untrained", "NullControl_ShuffledTargets", "NullControl_BlindZeroInput")
    }
    f3_pass = all(acc <= null_threshold for acc in null_accs.values())

    dense_acc = dense_noreplay["avg_acc_final"]["mean"]
    f4_margin = f1_val - dense_acc
    f4_pass = f4_margin > 0.2500

    max_any_acc = max(summary[c]["avg_acc_final"]["max"] for c in CONDITIONS)
    f5_pass = max_any_acc <= 0.9700

    falsifiers = {
        "F1_accuracy_gte_0.70": {"value": f1_val, "threshold": 0.7000, "passed": f1_pass},
        "F2_retention_gte_0.75": {"value": f2_val, "threshold": 0.7500, "passed": f2_pass},
        "F3_null_controls_lte_0.2222": {"null_accs": null_accs, "threshold": null_threshold, "passed": f3_pass},
        "F4_margin_over_dense_gt_0.25": {"margin": round(f4_margin, 4), "dense_acc": dense_acc, "v3_acc": f1_val, "passed": f4_pass},
        "F5_ceiling_lte_0.97": {"max_acc": max_any_acc, "threshold": 0.9700, "passed": f5_pass},
        "ALL_PASSED": bool(f1_pass and f2_pass and f3_pass and f4_pass and f5_pass),
    }

    meta = {
        "exp_id": EXP_ID,
        "steps_per_domain": steps,
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "python": platform.python_version(),
        "command": command,
        "seeds": seeds,
        "lr": LR,
    }

    results = {
        "meta": meta,
        "falsifiers": falsifiers,
        "summary": summary,
        "per_seed": merged,
    }

    results_file = OUT_DIR / "results.json"
    results_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"=== {EXP_ID} FINAL RESULTS (N={steps}, 20 SEEDS) ===")
    print("=" * 70)
    for c in CONDITIONS:
        s = summary[c]
        print(f"{c:38s}: avg={s['avg_acc_final']['mean']:.4f} ±{s['avg_acc_final']['ci95']:.4f} "
              f"A={s['acc_A_final']['mean']:.4f} B={s['acc_B_final']['mean']:.4f} C={s['acc_C_final']['mean']:.4f} "
              f"retA={s['retention_A']['mean']:.4f} BWT={s['bwt_nats']['mean']:+.4f}")

    print("\n--- FALSIFIER AUDIT ---")
    for k, v in falsifiers.items():
        print(f"  {k}: {v}")
    print(f"\nFinal results saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "merge"], required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=20)
    parser.add_argument("--out-dir", type=str, default="")
    args = parser.parse_args()

    od = Path(args.out_dir) if args.out_dir else OUT_DIR
    if args.mode == "test":
        mode_test(args.seed_start, args.seed_end, args.steps, od)
    elif args.mode == "merge":
        mode_merge(args.steps, " ".join(sys.argv))

