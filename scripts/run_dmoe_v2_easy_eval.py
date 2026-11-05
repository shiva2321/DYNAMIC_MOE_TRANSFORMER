"""
Learnable-task matched-baseline continual-learning experiment `dmoe_v2_continual_easy`.
Protocol: results/dmoe_v2_continual_easy/PREREG.md (frozen before any run).

Modes:
  tune  : validation-seed (100..104) learnability ladder for the dense baseline (logs tuning_log.json)
  test  : one frozen test run on seeds [seed-start, seed-end) with --steps N (writes a part file)
  merge : combine part files for seeds 0..19 into results.json
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
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_dmoe_v2_hard_control_eval as hc  # noqa: E402
from dmoe.continual import ContinualLearner  # noqa: E402

EXP_ID = "dmoe_v2_continual_easy"


class DenseForLearner(hc.RegularDenseTransformerMatched):
    """Same dense baseline; adds only the no-op topology hooks ContinualLearner expects (no new parameters,
    no change to forward/learning). A dense model has no topology to mutate."""

    layers: list = []

    def execute_graph_mutation(self, novelty_score=None):
        return {"mutation_logs": []}


def _make_dense_for_learner(seed: int) -> "DenseForLearner":
    random.seed(seed)
    torch.manual_seed(seed)
    return DenseForLearner(d_model=32, num_heads=2, d_ff=128, num_layers=2, chunk_size=4)
OUT_DIR = ROOT / "results" / EXP_ID
LR = 4e-3
TUNE_SEEDS = [100, 101, 102, 103, 104]
TUNE_LADDER = [60, 120, 240, 480]
LEARNABLE_THRESHOLD = 0.80
NOISE_RATE = 0.05


def generate_easy_domain_batch(
    domain_id: int, batch_size: int = 16, seq_len: int = 16, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    payload = torch.randint(32, 48, (batch_size, seq_len), generator=gen, dtype=torch.long)
    prefix = torch.randint(10 + 2 * domain_id, 12 + 2 * domain_id, (batch_size, seq_len), generator=gen, dtype=torch.long)
    slot = (torch.arange(seq_len, dtype=torch.long) % 4).unsqueeze(0).expand(batch_size, seq_len)
    x = torch.where(slot == 0, prefix, payload)
    v = x - 32

    if domain_id == 0:
        body = (v + slot) % 16
        head = (x * 3 + 0) % 16
    elif domain_id == 1:
        body = (v ^ (slot + 5)) % 16
        head = (x * 5 + 1) % 16
    elif domain_id == 2:
        body = (3 * v + 7 - slot) % 16
        head = (x * 7 + 2) % 16
    else:
        raise ValueError(f"Unknown domain_id: {domain_id}")

    clean_y = 32 + torch.where(slot == 0, head, body)
    noise_mask = torch.rand((batch_size, seq_len), generator=gen) < NOISE_RATE
    random_y = torch.randint(32, 48, (batch_size, seq_len), generator=gen, dtype=torch.long)
    y = torch.where(noise_mask, random_y, clean_y)
    return x, y


@torch.no_grad()
def evaluate_domain(
    model: nn.Module,
    domain_id: int,
    seed: int,
    learner: ContinualLearner = None,
    blind_zero: bool = False,
) -> Tuple[float, float]:
    eval_seed = seed * 100000 + 95000 + domain_id
    x, y = generate_easy_domain_batch(domain_id, batch_size=64, seq_len=16, seed=eval_seed)
    if blind_zero:
        x = torch.zeros_like(x)
    model.eval()
    if learner is not None and learner.enable_fast_memory and learner.fast_memory is not None:
        logits = learner.predict_logits(x, output_head="byte_chunk")
    else:
        logits = model(x, output_head="byte_chunk", update_stats=False)["output"]
    flat_logits = logits.reshape(-1, 256)
    flat_y = y.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_y).item()
    acc = (flat_logits.argmax(dim=-1) == flat_y).float().mean().item()
    return loss, acc


def _plain_train(model: nn.Module, domains: List[int], seed: int, steps: int, frozen_after_first: bool = False,
                 stage_eval: bool = True):
    """Plain sequential AdamW training (no wrapper). Returns stage-eval losses for BWT."""
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    la_after_a, lb_after_b = 0.0, 0.0
    for idx, d_id in enumerate(domains):
        if frozen_after_first and idx == 1:
            model.freeze_backbone()
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        model.train()
        for st in range(steps):
            x, y = generate_easy_domain_batch(d_id, 16, 16, seed * 10000 + d_id * 1000 + st)
            opt.zero_grad()
            out = model(x, output_head="byte_chunk")
            loss = F.cross_entropy(out["output"].reshape(-1, 256), y.reshape(-1))
            loss.backward()
            opt.step()
        if stage_eval and idx == 0:
            la_after_a, _ = evaluate_domain(model, domains[0], seed)
        elif stage_eval and idx == 1:
            lb_after_b, _ = evaluate_domain(model, domains[1], seed)
    return la_after_a, lb_after_b


def _final_metrics(model: nn.Module, seed: int, la_after_a: float, lb_after_b: float,
                   learner: ContinualLearner = None) -> Dict[str, float]:
    la, aa = evaluate_domain(model, 0, seed, learner=learner)
    lb, ab = evaluate_domain(model, 1, seed, learner=learner)
    lc, ac = evaluate_domain(model, 2, seed, learner=learner)
    return {
        "acc_A_final": aa,
        "acc_B_final": ab,
        "acc_C_final": ac,
        "avg_acc_final": (aa + ab + ac) / 3.0,
        "bwt_nats": 0.5 * ((la - la_after_a) + (lb - lb_after_b)),
        "loss_A_final": la,
        "loss_B_final": lb,
        "loss_C_final": lc,
        "loss_A_after_A": la_after_a,
        "loss_B_after_B": lb_after_b,
    }


def _learner_run(model: nn.Module, seed: int, steps: int, enable_fm: bool, shuffle_targets: bool = False):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    learner = ContinualLearner(
        model, opt, replay_buffer_capacity=128, replay_batch_size=8, distillation_weight=0.5,
        maturity_age=40, mutation_check_interval=25, enable_fast_memory=enable_fm,
        fast_memory_blend=0.55, fast_memory_reg=0.25,
    )
    la_after_a, lb_after_b = 0.0, 0.0
    for d_id in range(3):
        for st in range(steps):
            x, y = generate_easy_domain_batch(d_id, 16, 16, seed * 10000 + d_id * 1000 + st)
            if shuffle_targets:
                pg = torch.Generator()
                pg.manual_seed(seed * 99999 + d_id * 1000 + st)
                y = y.reshape(-1)[torch.randperm(y.numel(), generator=pg)].reshape(y.shape)
            learner.step(x, y, output_head="byte_chunk")
        if d_id == 0:
            la_after_a, _ = evaluate_domain(model, 0, seed, learner=learner)
        elif d_id == 1:
            lb_after_b, _ = evaluate_domain(model, 1, seed, learner=learner)
    return learner, _final_metrics(model, seed, la_after_a, lb_after_b, learner=learner)


def run_single_seed(seed: int, steps: int) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}

    m = hc._make_dmoe_model(seed)
    res["NullControl_Untrained"] = _final_metrics(m, seed, 0.0, 0.0)
    res["NullControl_Untrained"]["bwt_nats"] = 0.0

    m = hc._make_dense_baseline(seed)
    a, b = _plain_train(m, [0, 1, 2], seed, steps)
    res["Baseline_RegularTransformer_Unfrozen"] = _final_metrics(m, seed, a, b)

    m = hc._make_dense_baseline(seed)
    a, b = _plain_train(m, [0, 1, 2], seed, steps, frozen_after_first=True)
    res["Baseline_RegularTransformer_FrozenBackbone"] = _final_metrics(m, seed, a, b)

    _, res["Baseline_RegularTransformer_WithReplay"] = _learner_run(_make_dense_for_learner(seed), seed, steps, False)
    _, res["Baseline_RegularTransformer_WithReplayAndFastMemory"] = _learner_run(
        _make_dense_for_learner(seed), seed, steps, True)
    _, res["Ablation_dMoE_NoFastMemory"] = _learner_run(hc._make_dmoe_model(seed), seed, steps, False)
    _, res["NullControl_ShuffledTargets"] = _learner_run(hc._make_dmoe_model(seed), seed, steps, True, shuffle_targets=True)
    m_full = hc._make_dmoe_model(seed)
    l_full, res["dMoE_v2_Full"] = _learner_run(m_full, seed, steps, True)

    zs = [evaluate_domain(m_full, d, seed, learner=l_full, blind_zero=True)[1] for d in range(3)]
    res["NullControl_BlindZeroInput"] = {
        "acc_A_final": zs[0], "acc_B_final": zs[1], "acc_C_final": zs[2],
        "avg_acc_final": sum(zs) / 3.0, "bwt_nats": 0.0,
    }
    return res


def mode_tune() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log = {"seeds": TUNE_SEEDS, "ladder": TUNE_LADDER, "threshold": LEARNABLE_THRESHOLD, "configs": [],
           "selected_steps": None}
    for n in TUNE_LADDER:
        dense_accs, dmoe_accs = [], []
        for s in TUNE_SEEDS:
            for d_id in range(3):
                md = hc._make_dense_baseline(s)
                _plain_train(md, [d_id], s, n, stage_eval=False)
                dense_accs.append(evaluate_domain(md, d_id, s)[1])
                mm = hc._make_dmoe_model(s)
                _plain_train(mm, [d_id], s, n, stage_eval=False)
                dmoe_accs.append(evaluate_domain(mm, d_id, s)[1])
        dense_mean = sum(dense_accs) / len(dense_accs)
        dmoe_mean = sum(dmoe_accs) / len(dmoe_accs)
        entry = {"steps_per_domain": n, "dense_single_domain_acc_mean": round(dense_mean, 4),
                 "dmoe_single_domain_acc_mean_(logged_not_used)": round(dmoe_mean, 4),
                 "dense_per_run": [round(a, 4) for a in dense_accs],
                 "dmoe_per_run": [round(a, 4) for a in dmoe_accs],
                 "meets_threshold": dense_mean >= LEARNABLE_THRESHOLD}
        log["configs"].append(entry)
        print(f"N={n}: dense={dense_mean:.4f} dmoe={dmoe_mean:.4f} meets={entry['meets_threshold']}", flush=True)
        (OUT_DIR / "tuning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        if dense_mean >= LEARNABLE_THRESHOLD:
            log["selected_steps"] = n
            break
    (OUT_DIR / "tuning_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("selected_steps:", log["selected_steps"])


def mode_test(a: int, b: int, steps: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed = {}
    for s in range(a, b):
        per_seed[str(s)] = run_single_seed(s, steps)
        print(f"seed {s} done", flush=True)
    (OUT_DIR / f"part_{a}_{b}.json").write_text(
        json.dumps({"steps": steps, "per_seed": per_seed}, indent=2), encoding="utf-8")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        return f"unavailable: {err}"


def mode_merge(command: str) -> None:
    parts = sorted(OUT_DIR.glob("part_*.json"))
    merged: Dict[str, Dict[str, dict]] = {}
    steps = None
    for p in parts:
        d = json.loads(p.read_text(encoding="utf-8"))
        steps = d["steps"] if steps is None else steps
        assert steps == d["steps"], "inconsistent steps across parts"
        merged.update(d["per_seed"])
    seeds = sorted(int(k) for k in merged)
    assert seeds == list(range(20)), f"expected seeds 0..19, got {seeds}"
    conds = list(merged["0"].keys())
    summary = {}
    for c in conds:
        summary[c] = {}
        for k in merged["0"][c]:
            summary[c][k] = hc._summarize([merged[str(s)][c][k] for s in seeds])
    sd, sr = hc._make_dmoe_model(0), hc._make_dense_baseline(0)
    meta = {
        "exp_id": EXP_ID, "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "torch_version": torch.__version__, "python": platform.python_version(),
        "command": command, "num_seeds": 20, "seeds": seeds, "steps_per_domain": steps,
        "lr": LR, "bayes_ceiling_approx": round(0.95 + 0.05 / 16, 4),
        "dmoe_v2_params": sum(p.numel() for p in sd.parameters()),
        "regular_transformer_matched_params": sum(p.numel() for p in sr.parameters()),
    }
    (OUT_DIR / "results.json").write_text(
        json.dumps({"meta": meta, "continual_learning_20_seeds": summary}, indent=2), encoding="utf-8")
    print(f"=== {EXP_ID} 20-Seed Summary (steps_per_domain={steps}) ===")
    for c, s in summary.items():
        print(f"{c:52s} | AvgAcc={s['avg_acc_final']['mean']:.4f} ± {s['avg_acc_final']['ci95']:.4f} "
              f"| A={s['acc_A_final']['mean']:.4f} B={s['acc_B_final']['mean']:.4f} C={s['acc_C_final']['mean']:.4f} "
              f"| BWT={s['bwt_nats']['mean']:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "test", "merge"], required=True)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--seed-end", type=int, default=20)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.mode == "tune":
        mode_tune()
    elif args.mode == "test":
        assert args.steps > 0, "--steps required"
        mode_test(args.seed_start, args.seed_end, args.steps)
    else:
        mode_merge(" ".join(sys.argv))

