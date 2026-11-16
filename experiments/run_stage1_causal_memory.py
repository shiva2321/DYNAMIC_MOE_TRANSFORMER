"""
Stage 1 Empirical Verification Suite (`stage1_causal_memory_v8`):
Strictly causal evaluation against matched RoPE Transformer, GRU, ablations, and 3 null controls
per `results/stage1_causal_memory/PREREG_v8.md` and `research-protocol.md`.

Usage:
  python experiments/run_stage1_causal_memory.py --mode tune
  python experiments/run_stage1_causal_memory.py --mode test
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
import pathlib
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.real_model import RealSelfStabilizingDMoE
from dmoe.stage1_baselines import CausalDenseRoPEBaseline, GRUBaseline, count_parameters


RESULTS_DIR = ROOT / "results" / "stage1_causal_memory"

DEV_SEEDS = [100, 101, 102, 103, 104]
TEST_SEEDS = list(range(20))

VOCAB_START = 32
VOCAB_SIZE = 32  # Shared tokens [32..63]
QUERY_MARK = 64
RULE_MARK = 65

# Fixed leak-free corpus splits from PREREG.md
TRAIN_CODE_FILES = sorted(p.name for p in (ROOT / "dmoe").glob("*.py"))
VAL_CODE_FILES = [
    "test_attention.py",
    "test_cognitive_architecture.py",
    "test_complex_ops.py",
    "test_continual_learner.py",
    "test_continual_learning.py",
    "test_decoder.py",
    "test_encoder.py",
    "test_end_to_end.py",
    "test_experts.py",
    "test_jepa.py",
    "test_layer.py",
    "test_manifold.py",
]
TEST_CODE_FILES = [
    "test_autonomous_lifecycle.py",
    "test_dmoe_v2_easy_task.py",
    "test_dmoe_v2_upgrades.py",
    "test_dmoe_v3_setgate.py",
    "test_dmoe_v4_mixed_stream.py",
    "test_model_mutation.py",
    "test_real_model_regression_gate.py",
    "test_replay_buffer.py",
    "test_simulation_loop.py",
    "test_surprise_thinking.py",
    "test_tokenizer_free.py",
    "test_unified_novelty.py",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_warmup_cosine_lr(
    opt: torch.optim.Optimizer,
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 40,
) -> None:
    w = min(warmup_steps, max(1, total_steps // 5))
    if step < w:
        lr = base_lr * float(step + 1) / float(w)
    else:
        prog = float(step - w) / float(max(1, total_steps - w))
        lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
    for pg in opt.param_groups:
        pg["lr"] = lr


def forward_logits(model: nn.Module, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    out = model(x, **kwargs)["output"]
    if out.dim() == 4:
        return out.squeeze(2)
    return out


def build_model(name: str, device: torch.device) -> nn.Module:
    if name in ("Cog_Mem_Hier", "Null_Untrained", "Null_ShuffledTargets", "Null_BlindZeroInput"):
        m = RealSelfStabilizingDMoE(
            d_model=48,
            num_heads=4,
            d_ff=74,
            initial_layers=2,
            initial_experts=1,
            chunk_size=1,
            enable_memory=True,
            num_ltm_slots=64,
            enable_three_level_attn=True,
            local_window=8,
            chunk_stride=4,
            enable_layer_stm=True,
            signed_beta=True,
            use_rope=True,
            causal_attn=True,
            auto_write_ltm=False,
        )
    elif name == "Cog_Mem_Flat":
        m = RealSelfStabilizingDMoE(
            d_model=48,
            num_heads=4,
            d_ff=244,
            initial_layers=2,
            initial_experts=1,
            chunk_size=1,
            enable_memory=True,
            num_ltm_slots=64,
            enable_three_level_attn=False,
            enable_layer_stm=True,
            signed_beta=True,
            use_rope=True,
            causal_attn=True,
            auto_write_ltm=False,
        )
    elif name == "Cog_NoSignedBeta":
        m = RealSelfStabilizingDMoE(
            d_model=48,
            num_heads=4,
            d_ff=74,
            initial_layers=2,
            initial_experts=1,
            chunk_size=1,
            enable_memory=True,
            num_ltm_slots=64,
            enable_three_level_attn=True,
            local_window=8,
            chunk_stride=4,
            enable_layer_stm=True,
            signed_beta=False,
            use_rope=True,
            causal_attn=True,
            auto_write_ltm=False,
        )
    elif name == "Cog_NoSTM_Hier":
        m = RealSelfStabilizingDMoE(
            d_model=48,
            num_heads=4,
            d_ff=352,
            initial_layers=2,
            initial_experts=1,
            chunk_size=1,
            enable_memory=False,
            enable_three_level_attn=True,
            local_window=8,
            chunk_stride=4,
            enable_layer_stm=False,
            use_rope=True,
            causal_attn=True,
            auto_write_ltm=False,
        )
    elif name == "DenseRoPE_Baseline":
        m = CausalDenseRoPEBaseline(d_model=64, num_heads=4, d_ff=192, num_layers=3)
    elif name == "GRU_Baseline":
        m = GRUBaseline(d_model=96, num_layers=2)
    else:
        raise ValueError(f"Unknown model name: {name}")
    return m.to(device)


def summarize(vals: List[float]) -> Dict[str, Any]:
    n = len(vals)
    mean = sum(vals) / max(1, n)
    sd = statistics.stdev(vals) if n > 1 else 0.0
    se = sd / math.sqrt(max(1, n))
    # t_0.975 for df=19 is 2.093, df=4 is 2.776
    t_crit = 2.093 if n == 20 else (2.776 if n == 5 else 1.96)
    return {
        "n": n,
        "mean": round(mean, 6),
        "std": round(sd, 6),
        "ci95_half": round(t_crit * se, 6),
        "min": round(min(vals), 6) if vals else 0.0,
        "max": round(max(vals), 6) if vals else 0.0,
        "per_seed": [round(v, 6) for v in vals],
    }


def paired_stats(a: List[float], b: List[float]) -> Dict[str, float]:
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = sum(diffs) / max(1, n)
    sd_d = statistics.stdev(diffs) if n > 1 else 0.0
    t_stat = mean_d / (sd_d / math.sqrt(n)) if (n > 1 and sd_d > 1e-12) else 0.0
    return {
        "mean_diff": round(mean_d, 6),
        "std_diff": round(sd_d, 6),
        "t_stat": round(t_stat, 4),
    }


def verify_causality_gate(device: torch.device) -> Dict[str, float]:
    models_to_check = [
        "Cog_Mem_Hier",
        "Cog_Mem_Flat",
        "Cog_NoSignedBeta",
        "Cog_NoSTM_Hier",
        "DenseRoPE_Baseline",
        "GRU_Baseline",
    ]
    report: Dict[str, float] = {}
    set_all_seeds(999)
    x1 = torch.randint(VOCAB_START, VOCAB_START + VOCAB_SIZE, (2, 32), device=device)
    x2 = x1.clone()
    x2[:, 16:] = torch.randint(VOCAB_START, VOCAB_START + VOCAB_SIZE, (2, 16), device=device)
    for name in models_to_check:
        m = build_model(name, device)
        m.eval()
        if hasattr(m, "memory") and m.memory is not None:
            m.memory.long_term.write_kv(
                torch.randn(4, m.d_model, device=device),
                torch.randn(4, m.d_model, device=device),
            )
        with torch.no_grad():
            y1 = forward_logits(m, x1)
            y2 = forward_logits(m, x2)
        diff = float((y1[:, :16, :] - y2[:, :16, :]).abs().max().item())
        report[name] = diff
        if diff > 1e-5:
            raise RuntimeError(f"Causality gate F8 violated by {name}: diff={diff}")
    return report


# ---------------------------------------------------------------------------
# Task 1A: Shared-Vocab MQAR + OOD Length Generalization
# ---------------------------------------------------------------------------
def gen_mqar_batch(
    batch_size: int,
    gap_len: int,
    k_pairs: int,
    n_queries: int,
    rng: random.Random,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vocab = list(range(VOCAB_START, VOCAB_START + VOCAB_SIZE))
    seq_len = 2 * k_pairs + gap_len + 2 * n_queries
    xs = []
    q_pos_all = []
    q_tgt_all = []
    for _ in range(batch_size):
        keys = rng.sample(vocab, k_pairs)
        key_set = set(keys)
        rest = [v for v in vocab if v not in key_set]
        vals = [rng.choice(rest) for _ in range(k_pairs)]
        seq = []
        for k, v in zip(keys, vals):
            seq.append(k)
            seq.append(v)
        for _ in range(gap_len):
            seq.append(rng.choice(rest))
        q_idxs = rng.sample(range(k_pairs), n_queries)
        qp = []
        qt = []
        for idx in q_idxs:
            seq.append(QUERY_MARK)
            seq.append(keys[idx])
            qp.append(len(seq) - 1)
            qt.append(vals[idx])
        xs.append(seq)
        q_pos_all.append(qp)
        q_tgt_all.append(qt)
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(q_pos_all, dtype=torch.long, device=device),
        torch.tensor(q_tgt_all, dtype=torch.long, device=device),
    )


def eval_mqar(
    model: nn.Module,
    eval_batches: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    blind_zero: bool = False,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, q_pos, q_tgt in eval_batches:
            xin = torch.zeros_like(x) if blind_zero else x
            logits = forward_logits(model, xin)  # [B, L, 256]
            B = x.size(0)
            q_logits = torch.gather(
                logits, 1, q_pos.unsqueeze(-1).expand(-1, -1, logits.size(-1))
            )
            preds = q_logits.argmax(dim=-1)
            correct += int((preds == q_tgt).sum().item())
            total += int(q_tgt.numel())
    return correct / max(1, total)


def run_task_1a(
    model_name: str,
    seed: int,
    device: torch.device,
    lr: float,
    steps: int,
    batch_size: int = 64,
    also_eval_blind: bool = False,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    tr_rng = random.Random(seed * 10007 + 11)
    ev_rng = random.Random(seed * 10007 + 9011)

    model = build_model(model_name, device)
    is_shuffled = model_name == "Null_ShuffledTargets"
    is_untrained = model_name == "Null_Untrained"

    if not is_untrained:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(device.type == "cuda"))
        for st in range(steps):
            set_warmup_cosine_lr(opt, st, steps, lr)
            gap = tr_rng.randint(6, 18)
            x, q_pos, q_tgt = gen_mqar_batch(batch_size, gap, k_pairs=5, n_queries=4, rng=tr_rng, device=device)
            if is_shuffled:
                perm = torch.randperm(batch_size, device=device)
                q_tgt = q_tgt[perm]
            logits = forward_logits(model, x)
            q_logits = torch.gather(
                logits, 1, q_pos.unsqueeze(-1).expand(-1, -1, logits.size(-1))
            )
            loss = F.cross_entropy(q_logits.reshape(-1, logits.size(-1)), q_tgt.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    eval_id = [gen_mqar_batch(64, 12, 5, 4, ev_rng, device) for _ in range(4)]
    eval_ood = [gen_mqar_batch(64, 64, 5, 4, ev_rng, device) for _ in range(4)]

    blind = model_name == "Null_BlindZeroInput"
    res: Dict[str, Any] = {
        "id_acc": eval_mqar(model, eval_id, blind_zero=blind),
        "ood64_acc": eval_mqar(model, eval_ood, blind_zero=blind),
    }
    if also_eval_blind:
        res["blind"] = {
            "id_acc": eval_mqar(model, eval_id, blind_zero=True),
            "ood64_acc": eval_mqar(model, eval_ood, blind_zero=True),
        }
    return res


# ---------------------------------------------------------------------------
# Task 1B: 2-Channel Parity State Tracking (M22)
# ---------------------------------------------------------------------------
def gen_parity_batch(
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = [[rng.randint(32, 39) for _ in range(seq_len)] for _ in range(batch_size)]
    x = torch.tensor(xs, dtype=torch.long, device=device)
    op = (x - 32) % 4
    f0 = ((op == 1) | (op == 3)).long()
    f1 = ((op == 2) | (op == 3)).long()
    p0 = torch.cumsum(f0, dim=1) % 2
    p1 = torch.cumsum(f1, dim=1) % 2
    y = 40 + 2 * p0 + p1
    return x, y


def eval_parity(model: nn.Module, x: torch.Tensor, y: torch.Tensor, blind_zero: bool = False) -> float:
    model.eval()
    with torch.no_grad():
        xin = torch.zeros_like(x) if blind_zero else x
        logits = forward_logits(model, xin)
        preds = logits.argmax(dim=-1)
        return float((preds == y).float().mean().item())


def run_task_1b(
    model_name: str,
    seed: int,
    device: torch.device,
    lr: float,
    steps: int,
    batch_size: int = 64,
    also_eval_blind: bool = False,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    tr_rng = random.Random(seed * 10007 + 22)
    ev_rng = random.Random(seed * 10007 + 9022)

    model = build_model(model_name, device)
    is_shuffled = model_name == "Null_ShuffledTargets"
    is_untrained = model_name == "Null_Untrained"

    if not is_untrained:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(device.type == "cuda"))
        for st in range(steps):
            set_warmup_cosine_lr(opt, st, steps, lr)
            x, y = gen_parity_batch(batch_size, 20, tr_rng, device)
            if is_shuffled:
                perm = torch.randperm(batch_size, device=device)
                y = y[perm]
            logits = forward_logits(model, x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    x_id, y_id = gen_parity_batch(256, 20, ev_rng, device)
    x_ood, y_ood = gen_parity_batch(256, 64, ev_rng, device)
    blind = model_name == "Null_BlindZeroInput"
    res: Dict[str, Any] = {
        "id_acc": eval_parity(model, x_id, y_id, blind_zero=blind),
        "ood64_acc": eval_parity(model, x_ood, y_ood, blind_zero=blind),
    }
    if also_eval_blind:
        res["blind"] = {
            "id_acc": eval_parity(model, x_id, y_id, blind_zero=True),
            "ood64_acc": eval_parity(model, x_ood, y_ood, blind_zero=True),
        }
    return res


# ---------------------------------------------------------------------------
# Task 1C: Cross-Episode LTM + Distractor Survival + Null-Sink Non-Interference
# ---------------------------------------------------------------------------
def gen_episode_pair(
    batch_size: int,
    k_facts: int,
    rng: random.Random,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ep_1: learnable background cycle `(prev + 1) mod 32 + 32` punctuated by `k_facts`
          surprising bindings `[k_i, v_i]` where `v_i != (k_i + 1) mod 32 + 32`.
    ep_2: queries `[QUERY_MARK, k_i] -> v_i`.
    """
    vocab = list(range(VOCAB_START, VOCAB_START + VOCAB_SIZE))
    ep1_list, ep2_list, qp_list, qt_list = [], [], [], []
    for _ in range(batch_size):
        keys = rng.sample(vocab, k_facts)
        key_set = set(keys)
        vals = []
        for k in keys:
            # Ensure novel value differs from background cycle successor
            bg_next = VOCAB_START + ((k - VOCAB_START + 1) % VOCAB_SIZE)
            cand = [t for t in vocab if t != bg_next and t not in key_set]
            vals.append(rng.choice(cand))

        # Construct ep1 cleanly: interleaved background segments + novel pairs
        cur = rng.choice([t for t in vocab if t not in key_set])
        ep1 = []
        for k, v in zip(keys, vals):
            for _ in range(2):
                ep1.append(cur)
                nxt = VOCAB_START + ((cur - VOCAB_START + 1) % VOCAB_SIZE)
                if nxt in key_set:
                    nxt = VOCAB_START + ((nxt - VOCAB_START + 1) % VOCAB_SIZE)
                cur = nxt
            ep1.append(k)
            ep1.append(v)
            cur = VOCAB_START + ((v - VOCAB_START + 1) % VOCAB_SIZE)
            if cur in key_set:
                cur = VOCAB_START + ((cur - VOCAB_START + 1) % VOCAB_SIZE)

        # ep2: short neutral prefix + queries
        ep2 = []
        q_c = rng.choice([t for t in vocab if t not in key_set])
        for _ in range(4):
            ep2.append(q_c)
            q_c = VOCAB_START + ((q_c - VOCAB_START + 1) % VOCAB_SIZE)
        qp, qt = [], []
        order = list(range(k_facts))
        rng.shuffle(order)
        for idx in order:
            ep2.append(QUERY_MARK)
            ep2.append(keys[idx])
            qp.append(len(ep2) - 1)
            qt.append(vals[idx])
        ep1_list.append(ep1)
        ep2_list.append(ep2)
        qp_list.append(qp)
        qt_list.append(qt)

    return (
        torch.tensor(ep1_list, dtype=torch.long, device=device),
        torch.tensor(ep2_list, dtype=torch.long, device=device),
        torch.tensor(qp_list, dtype=torch.long, device=device),
        torch.tensor(qt_list, dtype=torch.long, device=device),
    )


def gen_rule_batch(
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parametric rule probes: [RULE_MARK, tok] -> (tok - 32 + 7) % 32 + 32 inside background."""
    vocab = list(range(VOCAB_START, VOCAB_START + VOCAB_SIZE))
    xs, rp_all, rt_all = [], [], []
    for _ in range(batch_size):
        seq, rp, rt = [], [], []
        cur = rng.choice(vocab)
        for _ in range(4):
            seq.append(cur)
            cur = VOCAB_START + ((cur - VOCAB_START + 1) % VOCAB_SIZE)
            tok = rng.choice(vocab)
            seq.append(RULE_MARK)
            seq.append(tok)
            rp.append(len(seq) - 1)
            rt.append(VOCAB_START + ((tok - VOCAB_START + 7) % VOCAB_SIZE))
        xs.append(seq)
        rp_all.append(rp)
        rt_all.append(rt)
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(rp_all, dtype=torch.long, device=device),
        torch.tensor(rt_all, dtype=torch.long, device=device),
    )


def gen_distractor_episode(k_facts: int, rng: random.Random, device: torch.device) -> torch.Tensor:
    vocab = list(range(VOCAB_START, VOCAB_START + VOCAB_SIZE))
    ep = []
    cur = rng.choice(vocab)
    for _ in range(k_facts):
        for _ in range(2):
            ep.append(cur)
            cur = VOCAB_START + ((cur - VOCAB_START + 1) % VOCAB_SIZE)
        k = rng.choice(vocab)
        v = rng.choice([t for t in vocab if t != VOCAB_START + ((k - VOCAB_START + 1) % VOCAB_SIZE)])
        ep.extend([k, v])
        cur = rng.choice(vocab)
    return torch.tensor([ep], dtype=torch.long, device=device)


def run_task_1c(
    model_name: str,
    seed: int,
    device: torch.device,
    lr: float,
    steps: int,
    min_nll: float = 0.7,
    batch_size: int = 32,
    also_eval_blind: bool = False,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    tr_rng = random.Random(seed * 10007 + 33)
    ev_rng = random.Random(seed * 10007 + 9033)

    model = build_model(model_name, device)
    is_shuffled = model_name == "Null_ShuffledTargets"
    is_untrained = model_name == "Null_Untrained"
    has_ltm = hasattr(model, "extract_episode_kv") and getattr(model, "enable_memory", False)

    if not is_untrained:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(device.type == "cuda"))
        for st in range(steps):
            set_warmup_cosine_lr(opt, st, steps, lr)
            ep1, ep2, qp, qt = gen_episode_pair(batch_size, k_facts=4, rng=tr_rng, device=device)
            xr, rp, rt = gen_rule_batch(batch_size, rng=tr_rng, device=device)
            ep1_tgt = ep1[:, 1:]
            if is_shuffled:
                perm = torch.randperm(batch_size, device=device)
                qt = qt[perm]
                rt = rt[perm]
                ep1_tgt = ep1_tgt[perm]

            # 1. Single forward pass on ep1 teaches routine background & feeds surprise-based KV extraction
            out_ep1 = model(ep1, update_stats=False) if has_ltm else None
            lg_ep1 = (
                out_ep1["output"].squeeze(2) if out_ep1["output"].dim() == 4 else out_ep1["output"]
            ) if out_ep1 is not None else forward_logits(model, ep1)
            loss_bg = F.cross_entropy(
                lg_ep1[:, :-1, :].reshape(-1, lg_ep1.size(-1)),
                ep1_tgt.reshape(-1),
            )

            # 2. Cross-episode LTM retrieval (only available on memory-enabled architectures)
            ep_mem = (
                model.extract_episode_kv(ep1, top_k=14, min_nll=min_nll, cached_out=out_ep1)
                if has_ltm
                else None
            )
            lg_ep2 = (
                forward_logits(model, ep2, update_stats=False, episode_memory=ep_mem)
                if has_ltm
                else forward_logits(model, ep2)
            )
            q_lg = torch.gather(lg_ep2, 1, qp.unsqueeze(-1).expand(-1, -1, lg_ep2.size(-1)))
            loss_ep = F.cross_entropy(q_lg.reshape(-1, lg_ep2.size(-1)), qt.reshape(-1))

            # 3. Parametric rule: train with populated unrelated distractor LTM on 50% of steps
            #    so contextual ltm_gate & null-sink learn non-interference cleanly.
            xr_mem = None
            if has_ltm and ep_mem is not None and (st % 2 == 1):
                xr_mem = {
                    "ep_keys": torch.roll(ep_mem["ep_keys"].detach(), shifts=1, dims=0),
                    "ep_vals": torch.roll(ep_mem["ep_vals"].detach(), shifts=1, dims=0),
                    "ep_mask": torch.roll(ep_mem["ep_mask"], shifts=1, dims=0),
                }
            lg_r = (
                forward_logits(model, xr, update_stats=False, episode_memory=xr_mem)
                if has_ltm
                else forward_logits(model, xr)
            )
            r_lg = torch.gather(lg_r, 1, rp.unsqueeze(-1).expand(-1, -1, lg_r.size(-1)))
            loss_rule = F.cross_entropy(r_lg.reshape(-1, lg_r.size(-1)), rt.reshape(-1))

            loss = 0.25 * loss_bg + 0.50 * loss_ep + 0.25 * loss_rule
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    # Frozen evaluation across independent multi-episode trials
    model.eval()
    blind = model_name == "Null_BlindZeroInput"
    rule_eval_x, rule_eval_p, rule_eval_t = gen_rule_batch(64, ev_rng, device)

    def _eval_rule(force_blind: bool = False) -> float:
        with torch.no_grad():
            xin = torch.zeros_like(rule_eval_x) if force_blind else rule_eval_x
            lg = forward_logits(model, xin)
            rl = torch.gather(lg, 1, rule_eval_p.unsqueeze(-1).expand(-1, -1, lg.size(-1)))
            return float((rl.argmax(dim=-1) == rule_eval_t).float().mean().item())

    if hasattr(model, "reset_ltm"):
        model.reset_ltm()
    clean_rule_acc = _eval_rule(force_blind=blind)

    # Test 1: Cross-episode LTM survival after 4 noisy distractor episodes
    ltm_correct = 0
    ltm_total = 0
    blind_ltm_correct = 0
    n_episodes = 24
    for _ in range(n_episodes):
        ep1, ep2, qp, qt = gen_episode_pair(1, k_facts=4, rng=ev_rng, device=device)
        if hasattr(model, "reset_ltm"):
            model.reset_ltm()
        if has_ltm and not blind:
            model.consolidate_ltm(ep1, top_k=14, min_nll=min_nll)
            for _ in range(4):
                ep_d = gen_distractor_episode(k_facts=2, rng=ev_rng, device=device)
                model.consolidate_ltm(ep_d, top_k=6, min_nll=min_nll)
        with torch.no_grad():
            xin2 = torch.zeros_like(ep2) if blind else ep2
            lg2 = forward_logits(model, xin2)
            ql2 = torch.gather(lg2, 1, qp.unsqueeze(-1).expand(-1, -1, lg2.size(-1)))
            ltm_correct += int((ql2.argmax(dim=-1) == qt).sum().item())
            ltm_total += int(qt.numel())
            if also_eval_blind:
                lg2_b = forward_logits(model, torch.zeros_like(ep2))
                ql2_b = torch.gather(lg2_b, 1, qp.unsqueeze(-1).expand(-1, -1, lg2_b.size(-1)))
                blind_ltm_correct += int((ql2_b.argmax(dim=-1) == qt).sum().item())

    # Test 2: Fill LTM bank to all 64 unrelated slots and measure parametric non-interference
    if hasattr(model, "reset_ltm"):
        model.reset_ltm()
    if has_ltm and not blind:
        for _ in range(16):
            ep_d = gen_distractor_episode(k_facts=4, rng=ev_rng, device=device)
            model.consolidate_ltm(ep_d, top_k=12, min_nll=0.0)
    post64_rule_acc = _eval_rule(force_blind=blind)
    blind_post64 = _eval_rule(force_blind=True) if also_eval_blind else 0.0
    if hasattr(model, "reset_ltm"):
        model.reset_ltm()
    blind_clean = _eval_rule(force_blind=True) if also_eval_blind else 0.0

    res: Dict[str, Any] = {
        "ltm_recall_acc": ltm_correct / max(1, ltm_total),
        "known_rule_clean_acc": clean_rule_acc,
        "known_rule_post64_acc": post64_rule_acc,
        "non_interference_drop": clean_rule_acc - post64_rule_acc,
    }
    if also_eval_blind:
        res["blind"] = {
            "ltm_recall_acc": blind_ltm_correct / max(1, ltm_total),
            "known_rule_clean_acc": blind_clean,
            "known_rule_post64_acc": blind_post64,
            "non_interference_drop": blind_clean - blind_post64,
        }
    return res


# ---------------------------------------------------------------------------
# Task 1D: Real Workspace Python Corpus Causal Next-Byte Compression (BPB)
# ---------------------------------------------------------------------------
def load_byte_windows(folder: str, filenames: List[str], window_len: int) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for fn in filenames:
        raw = (ROOT / folder / fn).read_bytes()
        data = torch.tensor(list(raw), dtype=torch.long)
        stride = window_len
        for start in range(0, data.numel() - (window_len + 1) + 1, stride):
            chunks.append(data[start : start + window_len + 1])
    return torch.stack(chunks, dim=0)


def eval_bpb(
    model: nn.Module,
    windows: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
    blind_zero: bool = False,
) -> float:
    model.eval()
    total_nll = 0.0
    total_toks = 0
    with torch.no_grad():
        for i in range(0, windows.size(0), batch_size):
            w = windows[i : i + batch_size].to(device)
            x, y = w[:, :-1], w[:, 1:]
            xin = torch.zeros_like(x) if blind_zero else x
            lg = forward_logits(model, xin)
            nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1), reduction="sum")
            total_nll += float(nll.item())
            total_toks += int(y.numel())
    return (total_nll / max(1, total_toks)) / math.log(2.0)


def run_task_1d(
    model_name: str,
    seed: int,
    device: torch.device,
    lr: float,
    steps: int,
    train_w: torch.Tensor,
    eval_w64: torch.Tensor,
    eval_w128: torch.Tensor,
    batch_size: int = 64,
    also_eval_blind: bool = False,
) -> Dict[str, Any]:
    set_all_seeds(seed)
    model = build_model(model_name, device)
    is_shuffled = model_name == "Null_ShuffledTargets"
    is_untrained = model_name == "Null_Untrained"

    train_ms_per_step = 0.0
    if not is_untrained:
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(device.type == "cuda"))
        n_tr = train_w.size(0)
        g = torch.Generator().manual_seed(seed * 10007 + 44)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for st in range(steps):
            set_warmup_cosine_lr(opt, st, steps, lr)
            idx = torch.randint(0, n_tr, (batch_size,), generator=g)
            w = train_w[idx].to(device)
            x, y = w[:, :-1], w[:, 1:]
            if is_shuffled:
                perm = torch.randperm(batch_size, device=device)
                y = y[perm]
            lg = forward_logits(model, x)
            loss = F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_ms_per_step = ((time.perf_counter() - t0) * 1000.0) / max(1, steps)

    blind = model_name == "Null_BlindZeroInput"
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf0 = time.perf_counter()
    bpb_64 = eval_bpb(model, eval_w64, device, blind_zero=blind)
    if device.type == "cuda":
        torch.cuda.synchronize()
    inf_dt = max(time.perf_counter() - t_inf0, 1e-6)
    toks_per_sec = float(eval_w64.size(0) * 64) / inf_dt
    bpb_128 = eval_bpb(model, eval_w128, device, blind_zero=blind)

    res: Dict[str, Any] = {
        "bpb_64": bpb_64,
        "bpb_128": bpb_128,
        "train_ms_per_step": train_ms_per_step,
        "eval_toks_per_sec": toks_per_sec,
    }
    if also_eval_blind:
        res["blind"] = {
            "bpb_64": eval_bpb(model, eval_w64, device, blind_zero=True),
            "bpb_128": eval_bpb(model, eval_w128, device, blind_zero=True),
            "train_ms_per_step": 0.0,
            "eval_toks_per_sec": toks_per_sec,
        }
    return res


def _resolve_lr(model_name: str, args: argparse.Namespace) -> float:
    if model_name == "DenseRoPE_Baseline":
        return float(args.lr_dense)
    if model_name == "GRU_Baseline":
        return float(args.lr_gru)
    return float(args.lr_cog)


def _aggregate_seed_dicts(
    t1a: List[Dict[str, Any]],
    t1b: List[Dict[str, Any]],
    t1c: List[Dict[str, Any]],
    t1d: List[Dict[str, Any]],
    dt_m: float,
) -> Dict[str, Any]:
    return {
        "wall_clock_sec": round(dt_m, 2),
        "1a_id_acc": summarize([r["id_acc"] for r in t1a]),
        "1a_ood64_acc": summarize([r["ood64_acc"] for r in t1a]),
        "1b_id_acc": summarize([r["id_acc"] for r in t1b]),
        "1b_ood64_acc": summarize([r["ood64_acc"] for r in t1b]),
        "1c_ltm_recall": summarize([r["ltm_recall_acc"] for r in t1c]),
        "1c_rule_clean": summarize([r["known_rule_clean_acc"] for r in t1c]),
        "1c_rule_post64": summarize([r["known_rule_post64_acc"] for r in t1c]),
        "1c_ni_drop": summarize([r["non_interference_drop"] for r in t1c]),
        "1d_bpb_64": summarize([r["bpb_64"] for r in t1d]),
        "1d_bpb_128": summarize([r["bpb_128"] for r in t1d]),
        "1d_train_ms_step": summarize([r["train_ms_per_step"] for r in t1d]),
        "1d_eval_toks_sec": summarize([r["eval_toks_per_sec"] for r in t1d]),
    }


def _collect_run_metadata() -> Dict[str, Any]:
    """Capture reproducibility metadata per research-protocol §6."""
    git_commit = "UNKNOWN"
    git_dirty: Optional[bool] = None
    if shutil.which("git") is not None and (ROOT / ".git").exists():
        res_head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), capture_output=True, text=True
        )
        res_dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(ROOT), capture_output=True, text=True
        )
        if res_head.returncode == 0 and res_dirty.returncode == 0:
            git_commit = res_head.stdout.strip()
            git_dirty = len(res_dirty.stdout.strip()) > 0
        else:
            git_commit = f"ERROR(rc={res_head.returncode})"
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "command": " ".join(sys.argv),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 Causal Memory Experimental Suite (v8)")
    parser.add_argument("--mode", choices=["tune", "test"], required=True)
    parser.add_argument("--lr_cog", type=float, default=3.5e-3)
    parser.add_argument("--lr_dense", type=float, default=4.5e-3)
    parser.add_argument("--lr_gru", type=float, default=4.5e-3)
    parser.add_argument("--steps_1a", type=int, default=700)
    parser.add_argument("--steps_1b", type=int, default=500)
    parser.add_argument("--steps_1c", type=int, default=350)
    parser.add_argument("--steps_1d", type=int, default=300)
    parser.add_argument("--min_nll", type=float, default=0.7)
    parser.add_argument("--models", type=str, default="")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    causality = verify_causality_gate(device)
    print(f"Causality gate F8 PASSED on all 6 models: max_diff={max(causality.values()):.2e}")

    train_w64 = load_byte_windows("dmoe", TRAIN_CODE_FILES, 64)
    val_w64 = load_byte_windows("tests", VAL_CODE_FILES, 64)
    val_w128 = load_byte_windows("tests", VAL_CODE_FILES, 128)
    test_w64 = load_byte_windows("tests", TEST_CODE_FILES, 64)
    test_w128 = load_byte_windows("tests", TEST_CODE_FILES, 128)

    if args.mode == "tune":
        # Strictly dev seeds [100..104] and Val code files
        models = (
            [m.strip() for m in args.models.split(",") if m.strip()]
            if args.models.strip()
            else ["Cog_Mem_Hier", "Cog_Mem_Flat", "Cog_NoSignedBeta", "DenseRoPE_Baseline", "GRU_Baseline"]
        )
        lrs = [3.0e-3, 4.5e-3]
        records = []
        for lr in lrs:
            res_lr: Dict[str, Any] = {"lr": lr, "seeds": DEV_SEEDS, "models": {}}
            for m in models:
                t1a = [run_task_1a(m, s, device, lr=lr, steps=args.steps_1a) for s in DEV_SEEDS]
                t1b = [run_task_1b(m, s, device, lr=lr, steps=args.steps_1b) for s in DEV_SEEDS]
                t1c = [run_task_1c(m, s, device, lr=lr, steps=args.steps_1c, min_nll=args.min_nll) for s in DEV_SEEDS]
                t1d = [run_task_1d(m, s, device, lr=lr, steps=args.steps_1d, train_w=train_w64, eval_w64=val_w64, eval_w128=val_w128) for s in DEV_SEEDS]
                res_lr["models"][m] = {
                    "1a_id": summarize([r["id_acc"] for r in t1a]),
                    "1a_ood64": summarize([r["ood64_acc"] for r in t1a]),
                    "1b_id": summarize([r["id_acc"] for r in t1b]),
                    "1b_ood64": summarize([r["ood64_acc"] for r in t1b]),
                    "1c_ltm_recall": summarize([r["ltm_recall_acc"] for r in t1c]),
                    "1c_ni_drop": summarize([r["non_interference_drop"] for r in t1c]),
                    "1d_val_bpb64": summarize([r["bpb_64"] for r in t1d]),
                }
                print(
                    f"[TUNE v8 lr={lr:.1e}] {m:18s} | "
                    f"1A(id={res_lr['models'][m]['1a_id']['mean']:.3f}, ood={res_lr['models'][m]['1a_ood64']['mean']:.3f}) | "
                    f"1B(ood={res_lr['models'][m]['1b_ood64']['mean']:.3f}) | "
                    f"1C(rec={res_lr['models'][m]['1c_ltm_recall']['mean']:.3f}, drop={res_lr['models'][m]['1c_ni_drop']['mean']:.3f}) | "
                    f"1D(val_bpb={res_lr['models'][m]['1d_val_bpb64']['mean']:.3f})"
                )
            records.append(res_lr)

        out_path = RESULTS_DIR / "tuning_log_v8.json"
        out_path.write_text(json.dumps({"metadata": _collect_run_metadata(), "config": vars(args), "trials": records}, indent=2))
        print(f"Saved tuning log to {out_path}")
        return

    # --mode test: Frozen 20-seed verification across ALL models, ablations & null controls
    train_models = [
        "Cog_Mem_Hier",
        "Cog_Mem_Flat",
        "Cog_NoSignedBeta",
        "Cog_NoSTM_Hier",
        "DenseRoPE_Baseline",
        "GRU_Baseline",
        "Null_Untrained",
        "Null_ShuffledTargets",
    ]
    param_counts = {
        m: count_parameters(build_model(m, torch.device("cpu")))
        for m in ["Cog_Mem_Hier", "Cog_Mem_Flat", "Cog_NoSignedBeta", "Cog_NoSTM_Hier", "DenseRoPE_Baseline", "GRU_Baseline"]
    }
    print("Parameter counts:", param_counts)

    results: Dict[str, Any] = {
        "experiment_id": "stage1_causal_memory_v8",
        "metadata": _collect_run_metadata(),
        "device": str(device),
        "hyperparams": vars(args),
        "seeds": TEST_SEEDS,
        "parameter_counts": param_counts,
        "causality_gate_max_diff": causality,
        "corpus_windows": {
            "train_w64": int(train_w64.size(0)),
            "test_w64": int(test_w64.size(0)),
            "test_w128": int(test_w128.size(0)),
        },
        "models": {},
    }

    for m in train_models:
        lr_m = _resolve_lr(m, args)
        blind_too = m == "Cog_Mem_Hier"
        t_start = time.perf_counter()
        t1a = [run_task_1a(m, s, device, lr=lr_m, steps=args.steps_1a, also_eval_blind=blind_too) for s in TEST_SEEDS]
        t1b = [run_task_1b(m, s, device, lr=lr_m, steps=args.steps_1b, also_eval_blind=blind_too) for s in TEST_SEEDS]
        t1c = [run_task_1c(m, s, device, lr=lr_m, steps=args.steps_1c, min_nll=args.min_nll, also_eval_blind=blind_too) for s in TEST_SEEDS]
        t1d = [run_task_1d(m, s, device, lr=lr_m, steps=args.steps_1d, train_w=train_w64, eval_w64=test_w64, eval_w128=test_w128, also_eval_blind=blind_too) for s in TEST_SEEDS]
        dt_m = time.perf_counter() - t_start

        summary_m = _aggregate_seed_dicts(t1a, t1b, t1c, t1d, dt_m)
        results["models"][m] = summary_m
        print(
            f"[TEST 20s {dt_m:5.1f}s] {m:20s} | "
            f"1A({summary_m['1a_id_acc']['mean']:.3f}/{summary_m['1a_ood64_acc']['mean']:.3f}) | "
            f"1B({summary_m['1b_id_acc']['mean']:.3f}/{summary_m['1b_ood64_acc']['mean']:.3f}) | "
            f"1C(rec={summary_m['1c_ltm_recall']['mean']:.3f}, drop={summary_m['1c_ni_drop']['mean']:.3f}) | "
            f"1D(bpb64={summary_m['1d_bpb_64']['mean']:.3f}, bpb128={summary_m['1d_bpb_128']['mean']:.3f})"
        )
        if blind_too:
            b_summary = _aggregate_seed_dicts(
                [r["blind"] for r in t1a],
                [r["blind"] for r in t1b],
                [r["blind"] for r in t1c],
                [r["blind"] for r in t1d],
                0.0,
            )
            results["models"]["Null_BlindZeroInput"] = b_summary
            print(
                f"[TEST 20s eval ] {'Null_BlindZeroInput':20s} | "
                f"1A({b_summary['1a_id_acc']['mean']:.3f}/{b_summary['1a_ood64_acc']['mean']:.3f}) | "
                f"1B({b_summary['1b_id_acc']['mean']:.3f}/{b_summary['1b_ood64_acc']['mean']:.3f}) | "
                f"1C(rec={b_summary['1c_ltm_recall']['mean']:.3f}, drop={b_summary['1c_ni_drop']['mean']:.3f}) | "
                f"1D(bpb64={b_summary['1d_bpb_64']['mean']:.3f}, bpb128={b_summary['1d_bpb_128']['mean']:.3f})"
            )

    # Evaluate PREREG falsifiers F1-F8 programmatically
    hier = results["models"]["Cog_Mem_Hier"]
    nostm = results["models"]["Cog_NoSTM_Hier"]
    nosig = results["models"]["Cog_NoSignedBeta"]
    drope = results["models"]["DenseRoPE_Baseline"]
    gru = results["models"]["GRU_Baseline"]

    results["paired_comparisons"] = {
        "1a_ood64_hier_vs_nostm": paired_stats(hier["1a_ood64_acc"]["per_seed"], nostm["1a_ood64_acc"]["per_seed"]),
        "1a_ood64_hier_vs_gru": paired_stats(hier["1a_ood64_acc"]["per_seed"], gru["1a_ood64_acc"]["per_seed"]),
        "1a_ood64_hier_vs_denserope": paired_stats(hier["1a_ood64_acc"]["per_seed"], drope["1a_ood64_acc"]["per_seed"]),
        "1b_ood64_hier_vs_nosigned": paired_stats(hier["1b_ood64_acc"]["per_seed"], nosig["1b_ood64_acc"]["per_seed"]),
        "1b_ood64_hier_vs_denserope": paired_stats(hier["1b_ood64_acc"]["per_seed"], drope["1b_ood64_acc"]["per_seed"]),
        "1b_ood64_hier_vs_gru": paired_stats(hier["1b_ood64_acc"]["per_seed"], gru["1b_ood64_acc"]["per_seed"]),
        "1c_rec_hier_vs_denserope": paired_stats(hier["1c_ltm_recall"]["per_seed"], drope["1c_ltm_recall"]["per_seed"]),
        "1d_bpb64_hier_vs_denserope": paired_stats(hier["1d_bpb_64"]["per_seed"], drope["1d_bpb_64"]["per_seed"]),
    }

    falsifiers = {
        "F1_mqar_acc_and_stability": bool(
            hier["1a_id_acc"]["mean"] >= 0.90
            and hier["1a_id_acc"]["min"] >= 0.75
            and hier["1a_id_acc"]["std"] <= 0.08
            and hier["1a_ood64_acc"]["mean"] >= 0.85
            and hier["1a_ood64_acc"]["min"] >= 0.65
        ),
        "F2_mqar_beats_nostm_and_gru": bool(
            (hier["1a_ood64_acc"]["mean"] - nostm["1a_ood64_acc"]["mean"]) >= 0.40
            and (hier["1a_ood64_acc"]["mean"] - gru["1a_ood64_acc"]["mean"]) >= 0.10
        ),
        "F3_parity_beats_nosigned_and_denserope": bool(
            hier["1b_ood64_acc"]["mean"] >= 0.90
            and hier["1b_ood64_acc"]["min"] >= 0.75
            and (hier["1b_ood64_acc"]["mean"] - nosig["1b_ood64_acc"]["mean"]) >= 0.15
            and (hier["1b_ood64_acc"]["mean"] - drope["1b_ood64_acc"]["mean"]) >= 0.15
        ),
        "F4_ltm_survival": bool(
            hier["1c_ltm_recall"]["mean"] >= 0.75
            and hier["1c_ltm_recall"]["min"] >= 0.50
            and (hier["1c_ltm_recall"]["mean"] - drope["1c_ltm_recall"]["mean"]) >= 0.65
            and (hier["1c_ltm_recall"]["mean"] - gru["1c_ltm_recall"]["mean"]) >= 0.65
        ),
        "F5_non_interference": bool(
            hier["1c_ni_drop"]["mean"] <= 0.03 and hier["1c_ni_drop"]["max"] <= 0.08
        ),
        "F6_real_code_bpb": bool(
            hier["1d_bpb_64"]["mean"] <= (drope["1d_bpb_64"]["mean"] + 0.05)
        ),
        "F7_null_controls": bool(
            results["models"]["Null_Untrained"]["1a_id_acc"]["max"] <= 0.08
            and results["models"]["Null_Untrained"]["1b_id_acc"]["max"] <= 0.35
            and results["models"]["Null_Untrained"]["1d_bpb_64"]["min"] >= 6.5
            and results["models"]["Null_ShuffledTargets"]["1a_id_acc"]["max"] <= 0.08
            and results["models"]["Null_ShuffledTargets"]["1b_id_acc"]["max"] <= 0.35
            and results["models"]["Null_ShuffledTargets"]["1d_bpb_64"]["min"] >= 4.5
            and (results["models"]["Null_ShuffledTargets"]["1d_bpb_64"]["min"] - hier["1d_bpb_64"]["max"]) >= 1.5
            and results["models"]["Null_BlindZeroInput"]["1a_id_acc"]["max"] <= 0.08
            and results["models"]["Null_BlindZeroInput"]["1b_id_acc"]["max"] <= 0.35
            and results["models"]["Null_BlindZeroInput"]["1d_bpb_64"]["min"] >= 4.5
        ),
        "F8_causality_gate": all(v <= 1e-5 for v in causality.values()),
    }
    results["falsifiers"] = falsifiers
    results["all_passed"] = all(falsifiers.values())

    out_path = RESULTS_DIR / "results_v8.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("\nFalsifiers:", json.dumps(falsifiers, indent=2))
    print(f"ALL PASSED: {results['all_passed']} -> saved to {out_path}")


if __name__ == "__main__":
    main()
