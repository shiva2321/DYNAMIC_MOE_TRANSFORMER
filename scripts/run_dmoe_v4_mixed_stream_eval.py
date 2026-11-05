"""
`dmoe_v4_mixed_stream`: Evaluation of per-sequence content routing and always-plastic
null-space projection on interleaved mixed streams (B2) with zero replay.
Pre-registered protocol: results/dmoe_v4_mixed_stream/PREREG.md.

Modes:
  --mode test --steps 240 --seed-start a --seed-end b : evaluates seeds [a, b) and writes part_{steps}_{a}_{b}.json
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
from dmoe.continual import ContinualLearner
from dmoe.model import SelfStabilizingDynamicMorphicMoE

EXP_ID = "dmoe_v4_mixed_stream"
OUT_DIR = ROOT / "results" / EXP_ID
LR = 4e-3
DOMAINS = [0, 1, 2]
NO_MUTATION_INTERVAL = 10 ** 9
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _make_dmoe_v4(seed: int) -> SelfStabilizingDynamicMorphicMoE:
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


def generate_mixed_stream_batch(
    batch_size: int = 64, seq_len: int = 16, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate an interleaved mixed-stream batch of sequences from Domains 0, 1, and 2.
    Returns:
        x: [B, seq_len] input sequences
        y: [B, seq_len] target sequences
        domain_labels: [B] true domain index for each sequence
    """
    gen = torch.Generator().manual_seed(int(seed))
    domain_labels = torch.randint(0, 3, (batch_size,), generator=gen, dtype=torch.long)
    xs, ys = [], []
    for i in range(batch_size):
        d = int(domain_labels[i].item())
        sub_seed = seed * 1000 + i * 17 + d
        x_i, y_i = easy.generate_easy_domain_batch(d, batch_size=1, seq_len=seq_len, seed=sub_seed)
        xs.append(x_i)
        ys.append(y_i)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), domain_labels


@torch.no_grad()
def _eval_mixed_stream(
    model: nn.Module,
    learner: Optional[ContinualLearner],
    seed: int,
    blind_zero: bool = False,
) -> Dict[str, float]:
    eval_seed = seed * 100000 + 88888
    x, y, labels = generate_mixed_stream_batch(batch_size=64, seq_len=16, seed=eval_seed)
    if blind_zero:
        x = torch.zeros_like(x)

    x, y = x.to(DEVICE), y.to(DEVICE)
    labels = labels.to(DEVICE)

    model.eval()
    if learner is not None:
        logits = learner.predict_logits(x, output_head="byte_chunk")
    else:
        out = model(x, output_head="byte_chunk", update_stats=False)
        logits = out["output"]

    flat_logits = logits.reshape(x.size(0), -1, 256)
    n_tokens = flat_logits.size(1)
    flat_y = y.reshape(x.size(0), -1)[:, :n_tokens]

    flat_logits_2d = flat_logits.reshape(-1, 256)
    flat_y_1d = flat_y.reshape(-1)
    loss_mixed = F.cross_entropy(flat_logits_2d, flat_y_1d).item()

    preds = flat_logits.argmax(dim=-1)
    acc_mixed = (preds == flat_y).float().mean().item()

    # Per-domain accuracies within the mixed stream
    mask_a = labels == 0
    mask_b = labels == 1
    mask_c = labels == 2
    acc_a = (preds[mask_a] == flat_y[mask_a]).float().mean().item() if mask_a.any() else 0.0
    acc_b = (preds[mask_b] == flat_y[mask_b]).float().mean().item() if mask_b.any() else 0.0
    acc_c = (preds[mask_c] == flat_y[mask_c]).float().mean().item() if mask_c.any() else 0.0

    # Routing accuracy (if model routes per-sequence)
    assigned = getattr(model, "last_assigned_experts", None)
    if assigned is not None and assigned.numel() == x.size(0):
        acc_route = (assigned.to(DEVICE) == labels).float().mean().item()
    else:
        acc_route = 0.0

    return {
        "loss_mixed_final": loss_mixed,
        "acc_mixed_final": acc_mixed,
        "acc_route_final": acc_route,
        "acc_A_final": acc_a,
        "acc_B_final": acc_b,
        "acc_C_final": acc_c,
    }


def _train_sequential(
    model: nn.Module,
    learner: ContinualLearner,
    seed: int,
    steps_per_domain: int,
    shuffle_targets: bool = False,
    is_dmoe: bool = False,
    blind_zero: bool = False,
) -> Dict[str, float]:
    acc_a_init = 0.0

    for d in DOMAINS:
        if d > 0 and hasattr(learner, "domain_transition"):
            learner.domain_transition(new_domain_id=d)

        for st in range(steps_per_domain):
            x, y = easy.generate_easy_domain_batch(
                d, batch_size=16, seq_len=16, seed=seed * 10000 + d * 1000 + st
            )
            x, y = x.to(DEVICE), y.to(DEVICE)
            if shuffle_targets:
                perm = torch.randperm(y.numel(), device=DEVICE)
                y = y.reshape(-1)[perm].reshape(y.shape)
            learner.step(x, y, output_head="byte_chunk")

        # Snapshot domain A performance at domain A completion
        if d == 0:
            with torch.no_grad():
                x_a_test, y_a_test = easy.generate_easy_domain_batch(
                    0, batch_size=64, seq_len=16, seed=seed * 100000 + 95000
                )
                x_a_test, y_a_test = x_a_test.to(DEVICE), y_a_test.to(DEVICE)
                logits_a = learner.predict_logits(x_a_test, output_head="byte_chunk")
                flat_l = logits_a.reshape(64, -1, 256)
                flat_t = y_a_test.reshape(64, -1)[:, : flat_l.size(1)]
                acc_a_init = (flat_l.argmax(dim=-1) == flat_t).float().mean().item()

    # Freeze final domain
    if hasattr(learner, "domain_transition"):
        learner.domain_transition(new_domain_id=len(DOMAINS), spawn_new_expert=False)

    # Evaluate on held-out mixed stream
    mixed_metrics = _eval_mixed_stream(model, learner, seed, blind_zero=blind_zero)
    retention_a = mixed_metrics["acc_A_final"] / max(acc_a_init, 1e-6)

    return {
        **mixed_metrics,
        "acc_A_initial": acc_a_init,
        "retention_A": retention_a,
    }


def run_condition_seed(cond_name: str, seed: int, steps_per_domain: int) -> Dict[str, float]:
    if cond_name == "NullControl_Untrained":
        model = _make_dmoe_v4(seed).to(DEVICE)
        metrics = _eval_mixed_stream(model, None, seed)
        return {
            **metrics,
            "acc_A_initial": metrics["acc_A_final"],
            "retention_A": 1.0,
        }

    if cond_name == "Baseline_RegularTransformer_NoReplay":
        model = easy._make_dense_for_learner(seed).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        learner = ContinualLearner(
            model, opt, replay_buffer_capacity=0, mutation_check_interval=NO_MUTATION_INTERVAL
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False)

    if cond_name == "Baseline_RegularTransformer_Replay128":
        model = easy._make_dense_for_learner(seed).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR)
        learner = ContinualLearner(
            model,
            opt,
            replay_buffer_capacity=128,
            replay_batch_size=8,
            distillation_weight=0.5,
            mutation_check_interval=NO_MUTATION_INTERVAL,
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False)

    if cond_name == "NullControl_ShuffledTargets":
        model = _make_dmoe_v4(seed).to(DEVICE)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        learner = ContinualLearner(
            model,
            opt,
            replay_buffer_capacity=0,
            enable_local_lda=True,
            enable_setgate=True,
            freeze_backbone_on_domain=False,
            enable_null_space=True,
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=True, is_dmoe=True)

    if cond_name == "NullControl_BlindZeroInput":
        model = _make_dmoe_v4(seed).to(DEVICE)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        learner = ContinualLearner(
            model,
            opt,
            replay_buffer_capacity=0,
            enable_local_lda=True,
            enable_setgate=True,
            freeze_backbone_on_domain=False,
            enable_null_space=True,
        )
        return _train_sequential(
            model, learner, seed, steps_per_domain, shuffle_targets=False, is_dmoe=True, blind_zero=True
        )

    if cond_name == "dMoE_v4_MixedStream_ZeroReplay":
        model = _make_dmoe_v4(seed).to(DEVICE)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        learner = ContinualLearner(
            model,
            opt,
            replay_buffer_capacity=0,
            enable_local_lda=True,
            enable_setgate=True,
            freeze_backbone_on_domain=False,
            enable_null_space=True,
            null_space_energy_threshold=0.98,
            null_space_max_rank_ratio=0.6,
        )
        return _train_sequential(model, learner, seed, steps_per_domain, shuffle_targets=False, is_dmoe=True)

    raise ValueError(f"Unknown condition: {cond_name}")


CONDITIONS = [
    "NullControl_Untrained",
    "NullControl_ShuffledTargets",
    "NullControl_BlindZeroInput",
    "Baseline_RegularTransformer_NoReplay",
    "Baseline_RegularTransformer_Replay128",
    "dMoE_v4_MixedStream_ZeroReplay",
]


def run_test(steps_per_domain: int, seed_start: int, seed_end: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part_path = OUT_DIR / f"part_{steps_per_domain}_{seed_start}_{seed_end}.json"
    per_condition: Dict[str, Dict[str, List[float]]] = {c: {} for c in CONDITIONS}

    seeds = list(range(seed_start, seed_end))
    print(f"[{datetime.datetime.now().isoformat()}] Starting seeds {seeds} for {steps_per_domain} steps/domain on {DEVICE}", flush=True)

    for cond in CONDITIONS:
        t0 = datetime.datetime.now()
        for seed in seeds:
            res = run_condition_seed(cond, seed, steps_per_domain)
            for k, v in res.items():
                per_condition[cond].setdefault(k, []).append(v)
        dt = (datetime.datetime.now() - t0).total_seconds()
        accs = per_condition[cond].get("acc_mixed_final", [0.0])
        print(f"  {cond:36s} | seeds {seed_start}..{seed_end-1} | mean acc_mixed={_mean(accs):.4f} | elapsed {dt:.1f}s", flush=True)

    payload = {
        "exp_id": EXP_ID,
        "steps_per_domain": steps_per_domain,
        "seed_start": seed_start,
        "seed_end": seed_end,
        "device": str(DEVICE),
        "conditions": per_condition,
    }
    part_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {part_path}", flush=True)
    return part_path


def run_merge(steps_per_domain: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUT_DIR / "results.json"

    pattern = f"part_{steps_per_domain}_*.json"
    parts = sorted(OUT_DIR.glob(pattern))
    if not parts:
        raise FileNotFoundError(f"No parts matching {pattern} found in {OUT_DIR}")

    merged_per_cond: Dict[str, Dict[str, List[float]]] = {c: {} for c in CONDITIONS}
    covered_seeds = set()

    for p in parts:
        data = json.loads(p.read_text())
        s_start, s_end = data["seed_start"], data["seed_end"]
        covered_seeds.update(range(s_start, s_end))
        for cond, metrics in data["conditions"].items():
            for k, vals in metrics.items():
                merged_per_cond[cond].setdefault(k, []).extend(vals)

    expected_seeds = set(range(20))
    missing = expected_seeds - covered_seeds
    if missing:
        print(f"WARNING: Missing seeds {sorted(missing)}! Covered: {sorted(covered_seeds)}")

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for cond, metrics in merged_per_cond.items():
        summary[cond] = {}
        for k, vals in metrics.items():
            summary[cond][k] = _summarize(vals)

    # Pre-registered Falsifiers Audit
    v4 = summary.get("dMoE_v4_MixedStream_ZeroReplay", {})
    base_no_rep = summary.get("Baseline_RegularTransformer_NoReplay", {})
    v4_acc = v4.get("acc_mixed_final", {}).get("mean", 0.0)
    v4_ret = v4.get("retention_A", {}).get("mean", 0.0)
    v4_route = v4.get("acc_route_final", {}).get("mean", 0.0)
    base_acc = base_no_rep.get("acc_mixed_final", {}).get("mean", 0.0)

    falsifiers = {
        "F1_mixed_acc_ge_0.70": {
            "threshold": 0.7000,
            "measured": v4_acc,
            "passed": bool(v4_acc >= 0.7000),
            "claim": "dMoE v4 mixed stream accuracy must be >= 0.7000",
        },
        "F2_retention_A_ge_0.75": {
            "threshold": 0.7500,
            "measured": v4_ret,
            "passed": bool(v4_ret >= 0.7500),
            "claim": "Domain A retention ratio must be >= 0.7500",
        },
        "F3_routing_acc_ge_0.90": {
            "threshold": 0.9000,
            "measured": v4_route,
            "passed": bool(v4_route >= 0.9000),
            "claim": "Per-sequence routing accuracy must be >= 0.9000",
        },
        "F4_margin_over_no_replay_gt_0.25": {
            "threshold": 0.2500,
            "measured": round(v4_acc - base_acc, 4),
            "passed": bool((v4_acc - base_acc) > 0.2500),
            "claim": "dMoE v4 mixed stream gain over dense un-replayed transformer must be > +0.2500",
        },
        "F5_ceiling_le_0.97": {
            "threshold": 0.9700,
            "measured": max(
                c_sum.get("acc_mixed_final", {}).get("max", 0.0) for c_sum in summary.values()
            ),
            "passed": bool(
                all(c_sum.get("acc_mixed_final", {}).get("max", 0.0) <= 0.9700 for c_sum in summary.values())
            ),
            "claim": "No condition may exceed 0.9700 on 5% noise data (data leakage guard)",
        },
    }

    all_passed = all(f["passed"] for f in falsifiers.values())

    report = {
        "exp_id": EXP_ID,
        "generated_at": datetime.datetime.now().isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": str(DEVICE),
        },
        "steps_per_domain": steps_per_domain,
        "num_seeds": len(covered_seeds),
        "seeds": sorted(covered_seeds),
        "summary": summary,
        "falsifiers": falsifiers,
        "all_falsifiers_passed": all_passed,
    }

    results_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {results_path}")

    # Print summary table
    print("\n" + "=" * 95)
    print(f"dMoE v4 Mixed-Stream Benchmark Summary (N={len(covered_seeds)} seeds, steps={steps_per_domain}/domain):")
    print("=" * 95)
    print(f"{'Condition':38s} | {'Acc Mixed':15s} | {'Retention A':12s} | {'Routing Acc':12s} | {'Loss Mixed':10s}")
    print("-" * 95)
    for cond in CONDITIONS:
        c_sum = summary.get(cond, {})
        acc = c_sum.get("acc_mixed_final", {})
        ret = c_sum.get("retention_A", {})
        route = c_sum.get("acc_route_final", {})
        loss = c_sum.get("loss_mixed_final", {})
        acc_str = f"{acc.get('mean', 0.0):.4f} +/- {acc.get('ci95', 0.0):.4f}"
        ret_str = f"{ret.get('mean', 0.0):.4f}"
        route_str = f"{route.get('mean', 0.0):.4f}" if cond.startswith("dMoE") else "N/A"
        loss_str = f"{loss.get('mean', 0.0):.4f}"
        print(f"{cond:38s} | {acc_str:15s} | {ret_str:12s} | {route_str:12s} | {loss_str:10s}")
    print("=" * 95)

    print("\nPre-Registered Falsifier Audit:")
    for fid, f in falsifiers.items():
        status = "PASSED" if f["passed"] else "FAILED"
        print(f"  [{status}] {fid}: measured={f['measured']} (threshold {f['threshold']}) -> {f['claim']}")
    print(f"\nFinal Verdict: {'ALL FALSIFIERS PASSED - CLAIM CONFIRMED' if all_passed else 'FALSIFIED'}")
    return results_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "merge"], required=True)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "test":
        run_test(args.steps, args.seed_start, args.seed_end)
    elif args.mode == "merge":
        run_merge(args.steps)


if __name__ == "__main__":
    main()
