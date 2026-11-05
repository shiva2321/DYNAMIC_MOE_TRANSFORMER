import copy
import datetime
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.chunker import BYTE_PAD_ID, MultiScaleByteChunker
from dmoe.continual import ContinualLearner
from dmoe.experts import DynamicMorphicMoE
from dmoe.fast_memory import MultiSlotExactLdaMemory
from dmoe.model import SelfStabilizingDynamicMorphicMoE


class RegularDenseTransformerBaseline(nn.Module):
    """
    Parameter-matched standard real-valued Dense Transformer baseline using the
    exact same MultiScaleByteChunker frontend and byte_chunk_head readout as dMoE.
    """

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 2,
        d_ff: int = 128,
        num_layers: int = 2,
        chunk_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.tokenizer_free = True
        self.byte_chunker = MultiScaleByteChunker(d_model=d_model, chunk_size=chunk_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.byte_chunk_head = nn.Linear(d_model, chunk_size * 256)

    def freeze_backbone(self) -> None:
        for p in self.byte_chunker.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        for p in self.final_norm.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor, output_head: str = "byte_chunk", update_stats: bool = True) -> Dict[str, torch.Tensor]:
        chunks = self.byte_chunker(x)
        seq_len = chunks.size(1)
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        chunks = chunks + 0.05 * torch.sin(pos)
        hidden = self.final_norm(self.encoder(chunks))
        bsz, s_c, _ = hidden.shape
        logits = self.byte_chunk_head(hidden).view(bsz, s_c, self.chunk_size, 256)
        return {
            "output": logits,
            "decoder_latent": hidden,
            "byte_residuals": self.byte_chunker.last_byte_features,
            "novelty_score": torch.tensor(0.0, device=x.device),
        }


def generate_domain_batch(
    domain_id: int, batch_size: int = 16, seq_len: int = 16, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Frozen protocol data generator (see results/dmoe_v2_continual/PREREG.md §3).
    Generates raw byte sequences [B, 16] and target bytes [B, 16] for Domain 0 (A),
    Domain 1 (B), or Domain 2 (C), including 25% shared structural delimiter bytes
    [0..31] whose target depends on the domain context.
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
    slot = pos % 4

    if domain_id == 0:
        # Domain A: Arithmetic Expressions [32..63]
        base_bytes = torch.randint(32, 64, (batch_size, seq_len), generator=gen, dtype=torch.long)
        shared_delim = torch.randint(0, 16, (batch_size, seq_len), generator=gen, dtype=torch.long)
        x = torch.where(slot == 0, shared_delim, base_bytes)
        y = torch.where(
            slot == 0,
            32 + (x * 2 + 1) % 32,
            32 + ((x - 32) + 3 * slot + 1) % 32,
        )
    elif domain_id == 1:
        # Domain B: Code Identifiers / Syntax [64..95]
        base_bytes = torch.randint(64, 96, (batch_size, seq_len), generator=gen, dtype=torch.long)
        shared_delim = torch.randint(0, 16, (batch_size, seq_len), generator=gen, dtype=torch.long)
        x = torch.where(slot == 0, shared_delim, base_bytes)
        y = torch.where(
            slot == 0,
            64 + (x * 3 + 5) % 32,
            64 + (((x - 64) ^ (slot + 5))) % 32,
        )
    elif domain_id == 2:
        # Domain C: Structured Key-Value Records [96..127]
        base_bytes = torch.randint(96, 128, (batch_size, seq_len), generator=gen, dtype=torch.long)
        shared_delim = torch.randint(0, 16, (batch_size, seq_len), generator=gen, dtype=torch.long)
        x = torch.where(slot == 0, shared_delim, base_bytes)
        y = torch.where(
            slot == 0,
            96 + (x * 5 + 9) % 32,
            96 + (3 * (x - 96) + 7 - slot) % 32,
        )
    else:
        raise ValueError(f"Unknown domain_id: {domain_id}")

    return x, y


@torch.no_grad()
def evaluate_domain(
    model: nn.Module,
    domain_id: int,
    seed: int,
    learner: ContinualLearner = None,
    blind_zero: bool = False,
) -> Tuple[float, float]:
    """
    Returns (cross_entropy_loss_in_nats, byte_accuracy) on held-out test batch (B=64, L=16).
    """
    eval_seed = seed * 100000 + 90000 + domain_id
    x, y = generate_domain_batch(domain_id=domain_id, batch_size=64, seq_len=16, seed=eval_seed)
    if blind_zero:
        x = torch.zeros_like(x)

    model.eval()
    if learner is not None and learner.enable_fast_memory and learner.fast_memory is not None:
        logits = learner.predict_logits(x, output_head="byte_chunk")
    else:
        out = model(x, output_head="byte_chunk", update_stats=False)
        logits = out["output"]

    flat_logits = logits.reshape(-1, 256)
    flat_y = y.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_y).item()
    preds = flat_logits.argmax(dim=-1)
    acc = (preds == flat_y).float().mean().item()
    return loss, acc


def _make_dmoe_model(seed: int) -> SelfStabilizingDynamicMorphicMoE:
    random.seed(seed)
    torch.manual_seed(seed)
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=32,
        num_heads=2,
        d_ff=64,
        initial_layers=2,
        initial_experts=2,
        theta_up=0.85,
        theta_down=0.02,
        theta_novelty_vertical=3.0,
        min_layers=1,
        max_layers=4,
        enable_simulation_loop=True,
        max_simulation_steps=2,
        theta_converge=0.25,
        skip_threshold=0.5,
        tokenizer_free=True,
        chunk_size=4,
    )
    for layer in model.layers:
        layer.moe.top_k = 2
        layer.moe.spawn_patience = 2
    return model


def _make_dense_baseline(seed: int) -> RegularDenseTransformerBaseline:
    random.seed(seed)
    torch.manual_seed(seed)
    return RegularDenseTransformerBaseline(d_model=32, num_heads=2, d_ff=128, num_layers=2, chunk_size=4)


def run_single_seed_continual(seed: int, steps_per_domain: int = 50) -> Dict[str, Dict[str, float]]:
    results_seed: Dict[str, Dict[str, float]] = {}

    # 1. NullControl_Untrained
    m_untrained = _make_dmoe_model(seed)
    la_u, aa_u = evaluate_domain(m_untrained, 0, seed)
    lb_u, ab_u = evaluate_domain(m_untrained, 1, seed)
    lc_u, ac_u = evaluate_domain(m_untrained, 2, seed)
    results_seed["NullControl_Untrained"] = {
        "acc_A_final": aa_u,
        "acc_B_final": ab_u,
        "acc_C_final": ac_u,
        "avg_acc_final": (aa_u + ab_u + ac_u) / 3.0,
        "bwt_nats": 0.0,
    }

    # 2. Baseline_RegularTransformer_Unfrozen
    m_dense = _make_dense_baseline(seed)
    opt_dense = torch.optim.AdamW(m_dense.parameters(), lr=4e-3)
    loss_a_after_a_dense, loss_b_after_b_dense = 0.0, 0.0
    for d_id in range(3):
        m_dense.train()
        for st in range(steps_per_domain):
            x, y = generate_domain_batch(d_id, batch_size=16, seq_len=16, seed=seed * 10000 + d_id * 1000 + st)
            opt_dense.zero_grad()
            out = m_dense(x)
            loss = F.cross_entropy(out["output"].reshape(-1, 256), y.reshape(-1))
            loss.backward()
            opt_dense.step()
        if d_id == 0:
            loss_a_after_a_dense, _ = evaluate_domain(m_dense, 0, seed)
        elif d_id == 1:
            loss_b_after_b_dense, _ = evaluate_domain(m_dense, 1, seed)

    la_d, aa_d = evaluate_domain(m_dense, 0, seed)
    lb_d, ab_d = evaluate_domain(m_dense, 1, seed)
    lc_d, ac_d = evaluate_domain(m_dense, 2, seed)
    bwt_dense = 0.5 * ((la_d - loss_a_after_a_dense) + (lb_d - loss_b_after_b_dense))
    results_seed["Baseline_RegularTransformer_Unfrozen"] = {
        "acc_A_final": aa_d,
        "acc_B_final": ab_d,
        "acc_C_final": ac_d,
        "avg_acc_final": (aa_d + ab_d + ac_d) / 3.0,
        "bwt_nats": bwt_dense,
    }

    # 3. Baseline_RegularTransformer_FrozenBackbone (Frozen after Domain A)
    m_frozen = _make_dense_baseline(seed)
    opt_frozen = torch.optim.AdamW(m_frozen.parameters(), lr=4e-3)
    loss_a_after_a_frz, loss_b_after_b_frz = 0.0, 0.0
    for d_id in range(3):
        if d_id == 1:
            m_frozen.freeze_backbone()
            opt_frozen = torch.optim.AdamW([p for p in m_frozen.parameters() if p.requires_grad], lr=4e-3)
        m_frozen.train()
        for st in range(steps_per_domain):
            x, y = generate_domain_batch(d_id, batch_size=16, seq_len=16, seed=seed * 10000 + d_id * 1000 + st)
            opt_frozen.zero_grad()
            out = m_frozen(x)
            loss = F.cross_entropy(out["output"].reshape(-1, 256), y.reshape(-1))
            loss.backward()
            opt_frozen.step()
        if d_id == 0:
            loss_a_after_a_frz, _ = evaluate_domain(m_frozen, 0, seed)
        elif d_id == 1:
            loss_b_after_b_frz, _ = evaluate_domain(m_frozen, 1, seed)

    la_f, aa_f = evaluate_domain(m_frozen, 0, seed)
    lb_f, ab_f = evaluate_domain(m_frozen, 1, seed)
    lc_f, ac_f = evaluate_domain(m_frozen, 2, seed)
    bwt_frz = 0.5 * ((la_f - loss_a_after_a_frz) + (lb_f - loss_b_after_b_frz))
    results_seed["Baseline_RegularTransformer_FrozenBackbone"] = {
        "acc_A_final": aa_f,
        "acc_B_final": ab_f,
        "acc_C_final": ac_f,
        "avg_acc_final": (aa_f + ab_f + ac_f) / 3.0,
        "bwt_nats": bwt_frz,
    }

    # 4. Baseline_StaticMoE_Naive (sequential Complex MoE without replay or fast memory)
    m_static = _make_dmoe_model(seed)
    opt_static = torch.optim.AdamW(m_static.parameters(), lr=4e-3)
    loss_a_after_a_st, loss_b_after_b_st = 0.0, 0.0
    for d_id in range(3):
        m_static.train()
        for st in range(steps_per_domain):
            x, y = generate_domain_batch(d_id, batch_size=16, seq_len=16, seed=seed * 10000 + d_id * 1000 + st)
            opt_static.zero_grad()
            out = m_static(x, output_head="byte_chunk")
            loss = F.cross_entropy(out["output"].reshape(-1, 256), y.reshape(-1))
            loss.backward()
            opt_static.step()
        if d_id == 0:
            loss_a_after_a_st, _ = evaluate_domain(m_static, 0, seed)
        elif d_id == 1:
            loss_b_after_b_st, _ = evaluate_domain(m_static, 1, seed)

    la_s, aa_s = evaluate_domain(m_static, 0, seed)
    lb_s, ab_s = evaluate_domain(m_static, 1, seed)
    lc_s, ac_s = evaluate_domain(m_static, 2, seed)
    bwt_static = 0.5 * ((la_s - loss_a_after_a_st) + (lb_s - loss_b_after_b_st))
    results_seed["Baseline_StaticMoE_Naive"] = {
        "acc_A_final": aa_s,
        "acc_B_final": ab_s,
        "acc_C_final": ac_s,
        "avg_acc_final": (aa_s + ab_s + ac_s) / 3.0,
        "bwt_nats": bwt_static,
    }

    # Helper for ContinualLearner variants
    def _run_learner_variant(
        enable_fast_memory: bool,
        replay_cap: int,
        shuffle_targets: bool = False,
    ) -> Tuple[SelfStabilizingDynamicMorphicMoE, ContinualLearner, Dict[str, float]]:
        m = _make_dmoe_model(seed)
        opt = torch.optim.AdamW(m.parameters(), lr=4e-3)
        learner = ContinualLearner(
            m,
            opt,
            replay_buffer_capacity=replay_cap,
            replay_batch_size=8,
            distillation_weight=0.5,
            maturity_age=40,
            mutation_check_interval=25,
            enable_fast_memory=enable_fast_memory,
            fast_memory_blend=0.55,
            fast_memory_reg=0.25,
        )
        la_after_a, lb_after_b = 0.0, 0.0
        for d_id in range(3):
            for st in range(steps_per_domain):
                x, y = generate_domain_batch(d_id, batch_size=16, seq_len=16, seed=seed * 10000 + d_id * 1000 + st)
                if shuffle_targets:
                    perm_gen = torch.Generator()
                    perm_gen.manual_seed(seed * 99999 + d_id * 1000 + st)
                    perm = torch.randperm(y.numel(), generator=perm_gen)
                    y = y.reshape(-1)[perm].reshape(y.shape)
                learner.step(x, y, output_head="byte_chunk")
            if d_id == 0:
                la_after_a, _ = evaluate_domain(m, 0, seed, learner=learner)
            elif d_id == 1:
                lb_after_b, _ = evaluate_domain(m, 1, seed, learner=learner)

        la_end, aa_end = evaluate_domain(m, 0, seed, learner=learner)
        lb_end, ab_end = evaluate_domain(m, 1, seed, learner=learner)
        lc_end, ac_end = evaluate_domain(m, 2, seed, learner=learner)
        bwt = 0.5 * ((la_end - la_after_a) + (lb_end - lb_after_b))
        metrics = {
            "acc_A_final": aa_end,
            "acc_B_final": ab_end,
            "acc_C_final": ac_end,
            "avg_acc_final": (aa_end + ab_end + ac_end) / 3.0,
            "bwt_nats": bwt,
        }
        return m, learner, metrics

    # 5. Ablation_dMoE_NoFastMemory
    _, _, results_seed["Ablation_dMoE_NoFastMemory"] = _run_learner_variant(
        enable_fast_memory=False, replay_cap=128
    )

    # 6. Ablation_dMoE_FastMemoryOnly_NoReplay
    _, _, results_seed["Ablation_dMoE_FastMemoryOnly_NoReplay"] = _run_learner_variant(
        enable_fast_memory=True, replay_cap=0
    )

    # 7. NullControl_ShuffledTargets
    _, _, results_seed["NullControl_ShuffledTargets"] = _run_learner_variant(
        enable_fast_memory=True, replay_cap=128, shuffle_targets=True
    )

    # 8. dMoE_v2_Full (Headline Method) + NullControl_BlindZeroInput
    m_full, learner_full, results_seed["dMoE_v2_Full"] = _run_learner_variant(
        enable_fast_memory=True, replay_cap=128, shuffle_targets=False
    )

    _, aa_blind = evaluate_domain(m_full, 0, seed, learner=learner_full, blind_zero=True)
    _, ab_blind = evaluate_domain(m_full, 1, seed, learner=learner_full, blind_zero=True)
    _, ac_blind = evaluate_domain(m_full, 2, seed, learner=learner_full, blind_zero=True)
    results_seed["NullControl_BlindZeroInput"] = {
        "acc_A_final": aa_blind,
        "acc_B_final": ab_blind,
        "acc_C_final": ac_blind,
        "avg_acc_final": (aa_blind + ab_blind + ac_blind) / 3.0,
        "bwt_nats": 0.0,
    }

    return results_seed


def run_dispatch_efficiency_benchmark(num_seeds: int = 20) -> Dict[str, float]:
    dense_tok_per_s: List[float] = []
    sorted_topk_tok_per_s: List[float] = []
    speedups: List[float] = []

    for s in range(num_seeds):
        torch.manual_seed(s)
        moe_dense = DynamicMorphicMoE(d_model=64, d_ff=256, initial_experts=8, top_k=None)
        moe_topk = copy.deepcopy(moe_dense)
        moe_topk.top_k = 2
        moe_dense.eval()
        moe_topk.eval()

        z_r = torch.randn(16, 64, 64)
        z_i = torch.randn(16, 64, 64)
        n_tokens = 16 * 64 * 20

        # Warmup
        with torch.no_grad():
            _ = moe_dense(z_r, z_i, update_stats=False)
            _ = moe_topk(z_r, z_i, update_stats=False)

        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = moe_dense(z_r, z_i, update_stats=False)
        dt_dense = max(time.perf_counter() - t0, 1e-9)

        t1 = time.perf_counter()
        with torch.no_grad():
            for _ in range(20):
                _ = moe_topk(z_r, z_i, update_stats=False)
        dt_topk = max(time.perf_counter() - t1, 1e-9)

        tps_dense = n_tokens / dt_dense
        tps_topk = n_tokens / dt_topk
        dense_tok_per_s.append(tps_dense)
        sorted_topk_tok_per_s.append(tps_topk)
        speedups.append(tps_topk / tps_dense)

    sp_t = torch.tensor(speedups, dtype=torch.float64)
    d_t = torch.tensor(dense_tok_per_s, dtype=torch.float64)
    k_t = torch.tensor(sorted_topk_tok_per_s, dtype=torch.float64)
    return {
        "dense_tok_per_sec_mean": round(float(d_t.mean().item()), 1),
        "sorted_topk2_tok_per_sec_mean": round(float(k_t.mean().item()), 1),
        "speedup_mean": round(float(sp_t.mean().item()), 4),
        "speedup_std": round(float(sp_t.std().item()), 4),
        "speedup_ci95": round(float(1.96 * sp_t.std().item() / math.sqrt(num_seeds)), 4),
        "speedup_min": round(float(sp_t.min().item()), 4),
        "speedup_max": round(float(sp_t.max().item()), 4),
    }


def run_certified_unlearning_benchmark(num_seeds: int = 20) -> Dict[str, float]:
    max_diffs: List[float] = []
    agreements: List[float] = []

    for s in range(num_seeds):
        torch.manual_seed(s + 500)
        slot_dim = 64
        feats_a = torch.randn(16, 4, 4, slot_dim) + 0.8
        _, y_a = generate_domain_batch(0, batch_size=16, seq_len=16, seed=s * 100 + 1)
        feats_b = torch.randn(16, 4, 4, slot_dim) - 0.8
        _, y_b = generate_domain_batch(1, batch_size=16, seq_len=16, seed=s * 100 + 2)
        feats_c = torch.randn(16, 4, 4, slot_dim) + 0.2
        _, y_c = generate_domain_batch(2, batch_size=16, seq_len=16, seed=s * 100 + 3)

        mem_unlearned = MultiSlotExactLdaMemory(d_in=slot_dim, num_classes=256, num_slots=4, reg=0.5)
        mem_unlearned.absorb_slots(feats_a, y_a)
        mem_unlearned.absorb_slots(feats_b, y_b)
        mem_unlearned.absorb_slots(feats_c, y_c)
        mem_unlearned.downdate_slots(feats_b, y_b)

        mem_retrained = MultiSlotExactLdaMemory(d_in=slot_dim, num_classes=256, num_slots=4, reg=0.5)
        mem_retrained.absorb_slots(feats_a, y_a)
        mem_retrained.absorb_slots(feats_c, y_c)

        q = torch.randn(8, 4, 4, slot_dim)
        logits_u = mem_unlearned.predict_slot_logits(q)
        logits_r = mem_retrained.predict_slot_logits(q)

        diff = float((logits_u.double() - logits_r.double()).abs().max().item())
        agree = float((logits_u.argmax(dim=-1) == logits_r.argmax(dim=-1)).float().mean().item())
        max_diffs.append(diff)
        agreements.append(agree)

    return {
        "max_logit_diff_across_20_seeds": max(max_diffs),
        "mean_argmax_agreement": sum(agreements) / len(agreements),
        "min_argmax_agreement": min(agreements),
    }


def _summarize_metric(vals: List[float]) -> Dict[str, float]:
    t = torch.tensor(vals, dtype=torch.float64)
    n = len(vals)
    mean = float(t.mean().item())
    std = float(t.std().item()) if n > 1 else 0.0
    ci95 = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci95": round(ci95, 4),
        "min": round(float(t.min().item()), 4),
        "max": round(float(t.max().item()), 4),
        "per_seed": [round(float(v), 4) for v in vals],
    }


def main() -> None:
    num_seeds = 20
    all_seed_runs = [run_single_seed_continual(seed=s, steps_per_domain=50) for s in range(num_seeds)]
    methods = list(all_seed_runs[0].keys())

    summary_by_method = {}
    for m_name in methods:
        summary_by_method[m_name] = {
            "avg_acc_final": _summarize_metric([r[m_name]["avg_acc_final"] for r in all_seed_runs]),
            "acc_A_final": _summarize_metric([r[m_name]["acc_A_final"] for r in all_seed_runs]),
            "acc_B_final": _summarize_metric([r[m_name]["acc_B_final"] for r in all_seed_runs]),
            "acc_C_final": _summarize_metric([r[m_name]["acc_C_final"] for r in all_seed_runs]),
            "bwt_nats": _summarize_metric([r[m_name]["bwt_nats"] for r in all_seed_runs]),
        }

    # Parameter counts comparison
    dmoe_sample = _make_dmoe_model(0)
    dense_sample = _make_dense_baseline(0)

    git_commit = (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT))
        .decode("utf-8")
        .strip()
    )

    payload = {
        "meta": {
            "exp_id": "dmoe_v2_continual",
            "git_commit": git_commit,
            "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "num_seeds": num_seeds,
            "seeds": list(range(num_seeds)),
            "steps_per_domain": 50,
            "dmoe_v2_params": sum(p.numel() for p in dmoe_sample.parameters()),
            "regular_transformer_params": sum(p.numel() for p in dense_sample.parameters()),
        },
        "continual_learning_20_seeds": summary_by_method,
        "dispatch_efficiency_20_seeds": run_dispatch_efficiency_benchmark(num_seeds=num_seeds),
        "certified_unlearning_20_seeds": run_certified_unlearning_benchmark(num_seeds=num_seeds),
    }

    out_dir = ROOT / "results" / "dmoe_v2_continual"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Print concise summary table
    print("=== dmoe_v2_continual 20-Seed Summary ===")
    for m_name, stats in summary_by_method.items():
        acc_m = stats["avg_acc_final"]["mean"]
        acc_ci = stats["avg_acc_final"]["ci95"]
        a_m = stats["acc_A_final"]["mean"]
        bwt_m = stats["bwt_nats"]["mean"]
        print(
            f"{m_name:44s} | AvgAcc={acc_m:.4f} ± {acc_ci:.4f} | "
            f"TaskA_Retained={a_m:.4f} | BWT={bwt_m:+.4f} nats"
        )
    print("Dispatch Efficiency:", json.dumps(payload["dispatch_efficiency_20_seeds"]))
    print("Certified Unlearning:", json.dumps(payload["certified_unlearning_20_seeds"]))


if __name__ == "__main__":
    main()

