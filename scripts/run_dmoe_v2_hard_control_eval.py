import datetime
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmoe.chunker import MultiScaleByteChunker
from dmoe.continual import ContinualLearner
from dmoe.decoder import UniversalProjectiveDecoder
from dmoe.model import SelfStabilizingDynamicMorphicMoE


class RegularDenseTransformerMatched(nn.Module):
    """
    Parameter-matched and readout-matched real-valued Dense Transformer baseline.
    Uses the exact same MultiScaleByteChunker frontend and UniversalProjectiveDecoder
    byte_chunk_head + extract_slot_features readout path as dMoE v2 (research-protocol.md §7).
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
        self.decoder = UniversalProjectiveDecoder(
            d_model=d_model,
            vocab_size=256,
            continuous_dim=32,
            chunk_size=chunk_size,
            tokenizer_free=True,
        )

    def freeze_backbone(self) -> None:
        for p in self.byte_chunker.parameters():
            p.requires_grad_(False)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, output_head: str = "byte_chunk", update_stats: bool = True
    ) -> Dict[str, torch.Tensor]:
        chunks = self.byte_chunker(x)
        seq_len = chunks.size(1)
        pos = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
        chunks = chunks + 0.05 * torch.sin(pos)
        hidden = self.encoder(chunks)
        z_imag = torch.zeros_like(hidden)
        byte_residuals = self.byte_chunker.last_byte_features
        out, latent = self.decoder.forward_with_latent(
            hidden, z_imag, head=output_head, byte_residuals=byte_residuals
        )
        return {
            "output": out,
            "decoder_latent": latent,
            "byte_residuals": byte_residuals,
            "novelty_score": torch.tensor(0.0, device=x.device),
        }


def generate_hard_domain_batch(
    domain_id: int, batch_size: int = 16, seq_len: int = 16, seed: int = 0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Frozen protocol data generator (see results/dmoe_v2_continual_hard_control/PREREG.md §3).
    All 3 domains share the exact same payload byte vocabulary [32..63] (eliminating
    any disjoint-vocabulary shortcut). Position 0 carries a domain-family prefix
    that must be propagated across chunks via attention/bus, and each target byte y[:, t]
    depends on x[:, 0], x[:, t], and x[:, t-1], with 5% symmetric label corruption
    bounding the Bayes-optimal accuracy ceiling at ~0.9516.
    """
    gen = torch.Generator()
    gen.manual_seed(int(seed))

    x = torch.randint(32, 64, (batch_size, seq_len), generator=gen, dtype=torch.long)
    prefix_base = 10 + 2 * domain_id
    prefix = torch.randint(prefix_base, prefix_base + 2, (batch_size,), generator=gen, dtype=torch.long)
    x[:, 0] = prefix

    pos = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len)
    slot = pos % 4
    prev_x = torch.roll(x, shifts=1, dims=1)
    prev_x[:, 0] = prefix

    if domain_id == 0:
        clean_y = 32 + ((x - 32) + (prev_x % 8) + slot) % 32
        clean_y[:, 0] = 32 + (prefix * 2 + 1) % 32
    elif domain_id == 1:
        clean_y = 32 + (((x - 32) ^ (prev_x % 8)) + 2 * slot + 7) % 32
        clean_y[:, 0] = 32 + (prefix * 3 + 5) % 32
    elif domain_id == 2:
        clean_y = 32 + (3 * (x - 32) - (prev_x % 8) + slot + 13) % 32
        clean_y[:, 0] = 32 + (prefix * 5 + 9) % 32
    else:
        raise ValueError(f"Unknown domain_id: {domain_id}")

    noise_mask = torch.rand((batch_size, seq_len), generator=gen) < 0.05
    random_y = torch.randint(32, 64, (batch_size, seq_len), generator=gen, dtype=torch.long)
    y = torch.where(noise_mask, random_y, clean_y)
    return x, y


@torch.no_grad()
def evaluate_hard_domain(
    model: nn.Module,
    domain_id: int,
    seed: int,
    learner: ContinualLearner = None,
    blind_zero: bool = False,
) -> Tuple[float, float]:
    eval_seed = seed * 100000 + 95000 + domain_id
    x, y = generate_hard_domain_batch(domain_id=domain_id, batch_size=64, seq_len=16, seed=eval_seed)
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


def _make_dense_baseline(seed: int) -> RegularDenseTransformerMatched:
    random.seed(seed)
    torch.manual_seed(seed)
    return RegularDenseTransformerMatched(d_model=32, num_heads=2, d_ff=128, num_layers=2, chunk_size=4)


def run_single_seed_hard(seed: int, steps_per_domain: int = 60) -> Dict[str, Dict[str, float]]:
    results_seed: Dict[str, Dict[str, float]] = {}

    # 1. NullControl_Untrained
    m_untrained = _make_dmoe_model(seed)
    _, aa_u = evaluate_hard_domain(m_untrained, 0, seed)
    _, ab_u = evaluate_hard_domain(m_untrained, 1, seed)
    _, ac_u = evaluate_hard_domain(m_untrained, 2, seed)
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
    la_after_a_d, lb_after_b_d = 0.0, 0.0
    for d_id in range(3):
        m_dense.train()
        for st in range(steps_per_domain):
            x, y = generate_hard_domain_batch(d_id, batch_size=16, seq_len=16, seed=seed * 10000 + d_id * 1000 + st)
            opt_dense.zero_grad()
            out = m_dense(x)
            loss = F.cross_entropy(out["output"].reshape(-1, 256), y.reshape(-1))
            loss.backward()
            opt_dense.step()
        if d_id == 0:
            la_after_a_d, _ = evaluate_hard_domain(m_dense, 0, seed)
        elif d_id == 1:
            lb_after_b_d, _ = evaluate_hard_domain(m_dense, 1, seed)

    la_d, aa_d = evaluate_hard_domain(m_dense, 0, seed)
    lb_d, ab_d = evaluate_hard_domain(m_dense, 1, seed)
    lc_d, ac_d = evaluate_hard_domain(m_dense, 2, seed)
    results_seed["Baseline_RegularTransformer_Unfrozen"] = {
        "acc_A_final": aa_d,
        "acc_B_final": ab_d,
        "acc_C_final": ac_d,
        "avg_acc_final": (aa_d + ab_d + ac_d) / 3.0,
        "bwt_nats": 0.5 * ((la_d - la_after_a_d) + (lb_d - lb_after_b_d)),
    }

    # 3. Baseline_RegularTransformer_FrozenBackbone
    m_frozen = _make_dense_baseline(seed)
    opt_frozen = torch.optim.AdamW(m_frozen.parameters(), lr=4e-3)
    la_after_a_f, lb_after_b_f = 0.0, 0.0
    for d_id in range(3):
        if d_id == 1:
            m_frozen.freeze_backbone()
            opt_frozen = torch.optim.AdamW([p for p in m_frozen.parameters() if p.requires_grad], lr=4e-3)
        m_frozen.train()
        for st in range(steps_per_domain):
            x, y = generate_hard_domain_batch(d_id, batch_size=16, seq_len=16, seed=seed * 10000 + d_id * 1000 + st)
            opt_frozen.zero_grad()
            out = m_frozen(x)
            loss = F.cross_entropy(out["output"].reshape(-1, 256), y.reshape(-1))
            loss.backward()
            opt_frozen.step()
        if d_id == 0:
            la_after_a_f, _ = evaluate_hard_domain(m_frozen, 0, seed)
        elif d_id == 1:
            lb_after_b_f, _ = evaluate_hard_domain(m_frozen, 1, seed)

    la_f, aa_f = evaluate_hard_domain(m_frozen, 0, seed)
    lb_f, ab_f = evaluate_hard_domain(m_frozen, 1, seed)
    lc_f, ac_f = evaluate_hard_domain(m_frozen, 2, seed)
    results_seed["Baseline_RegularTransformer_FrozenBackbone"] = {
        "acc_A_final": aa_f,
        "acc_B_final": ab_f,
        "acc_C_final": ac_f,
        "avg_acc_final": (aa_f + ab_f + ac_f) / 3.0,
        "bwt_nats": 0.5 * ((la_f - la_after_a_f) + (lb_f - lb_after_b_f)),
    }

    def _run_learner_on_model(
        model: nn.Module,
        enable_fast_memory: bool,
        replay_cap: int,
        shuffle_targets: bool = False,
    ) -> Tuple[nn.Module, ContinualLearner, Dict[str, float]]:
        opt = torch.optim.AdamW(model.parameters(), lr=4e-3)
        learner = ContinualLearner(
            model,
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
                x, y = generate_hard_domain_batch(
                    d_id, batch_size=16, seq_len=16, seed=seed * 10000 + d_id * 1000 + st
                )
                if shuffle_targets:
                    perm_gen = torch.Generator()
                    perm_gen.manual_seed(seed * 99999 + d_id * 1000 + st)
                    perm = torch.randperm(y.numel(), generator=perm_gen)
                    y = y.reshape(-1)[perm].reshape(y.shape)
                learner.step(x, y, output_head="byte_chunk")
            if d_id == 0:
                la_after_a, _ = evaluate_hard_domain(model, 0, seed, learner=learner)
            elif d_id == 1:
                lb_after_b, _ = evaluate_hard_domain(model, 1, seed, learner=learner)

        la_end, aa_end = evaluate_hard_domain(model, 0, seed, learner=learner)
        lb_end, ab_end = evaluate_hard_domain(model, 1, seed, learner=learner)
        lc_end, ac_end = evaluate_hard_domain(model, 2, seed, learner=learner)
        bwt = 0.5 * ((la_end - la_after_a) + (lb_end - lb_after_b))
        metrics = {
            "acc_A_final": aa_end,
            "acc_B_final": ab_end,
            "acc_C_final": ac_end,
            "avg_acc_final": (aa_end + ab_end + ac_end) / 3.0,
            "bwt_nats": bwt,
        }
        return model, learner, metrics

    # 4. Baseline_RegularTransformer_WithReplay
    _, _, results_seed["Baseline_RegularTransformer_WithReplay"] = _run_learner_on_model(
        _make_dense_baseline(seed), enable_fast_memory=False, replay_cap=128
    )

    # 5. Baseline_RegularTransformer_WithReplayAndFastMemory
    _, _, results_seed["Baseline_RegularTransformer_WithReplayAndFastMemory"] = _run_learner_on_model(
        _make_dense_baseline(seed), enable_fast_memory=True, replay_cap=128
    )

    # 6. Ablation_dMoE_NoFastMemory
    _, _, results_seed["Ablation_dMoE_NoFastMemory"] = _run_learner_on_model(
        _make_dmoe_model(seed), enable_fast_memory=False, replay_cap=128
    )

    # 7. NullControl_ShuffledTargets
    _, _, results_seed["NullControl_ShuffledTargets"] = _run_learner_on_model(
        _make_dmoe_model(seed), enable_fast_memory=True, replay_cap=128, shuffle_targets=True
    )

    # 8. dMoE_v2_Full
    m_full, l_full, results_seed["dMoE_v2_Full"] = _run_learner_on_model(
        _make_dmoe_model(seed), enable_fast_memory=True, replay_cap=128
    )

    # 9. NullControl_BlindZeroInput
    _, aa_z = evaluate_hard_domain(m_full, 0, seed, learner=l_full, blind_zero=True)
    _, ab_z = evaluate_hard_domain(m_full, 1, seed, learner=l_full, blind_zero=True)
    _, ac_z = evaluate_hard_domain(m_full, 2, seed, learner=l_full, blind_zero=True)
    results_seed["NullControl_BlindZeroInput"] = {
        "acc_A_final": aa_z,
        "acc_B_final": ab_z,
        "acc_C_final": ac_z,
        "avg_acc_final": (aa_z + ab_z + ac_z) / 3.0,
        "bwt_nats": 0.0,
    }

    return results_seed


def _summarize(values: List[float]) -> Dict[str, object]:
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / max(1, n - 1)
    sd = math.sqrt(var)
    ci95 = 1.96 * sd / math.sqrt(n)
    return {
        "mean": round(mean, 4),
        "std": round(sd, 4),
        "ci95": round(ci95, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "per_seed": [round(v, 4) for v in values],
    }


def main() -> None:
    seeds = list(range(20))
    steps_per_domain = 60

    all_continual: Dict[str, Dict[str, List[float]]] = {}
    for s in seeds:
        seed_res = run_single_seed_hard(s, steps_per_domain=steps_per_domain)
        for cond, metrics in seed_res.items():
            if cond not in all_continual:
                all_continual[cond] = {k: [] for k in metrics}
            for k, val in metrics.items():
                all_continual[cond][k].append(val)

    summary_continual: Dict[str, Dict[str, object]] = {}
    for cond, metric_lists in all_continual.items():
        summary_continual[cond] = {
            "avg_acc_final": _summarize(metric_lists["avg_acc_final"]),
            "acc_A_final": _summarize(metric_lists["acc_A_final"]),
            "acc_B_final": _summarize(metric_lists["acc_B_final"]),
            "acc_C_final": _summarize(metric_lists["acc_C_final"]),
            "bwt_nats": _summarize(metric_lists["bwt_nats"]),
        }

    git_commit = "unknown"
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT)
            .decode("utf-8")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as err:
        git_commit = f"unavailable: {err}"

    sample_dmoe = _make_dmoe_model(0)
    sample_dense = _make_dense_baseline(0)
    dmoe_params = sum(p.numel() for p in sample_dmoe.parameters())
    dense_params = sum(p.numel() for p in sample_dense.parameters())

    out_dir = ROOT / "results" / "dmoe_v2_continual_hard_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"

    payload = {
        "meta": {
            "exp_id": "dmoe_v2_continual_hard_control",
            "git_commit": git_commit,
            "utc_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "torch_version": torch.__version__,
            "num_seeds": len(seeds),
            "seeds": seeds,
            "steps_per_domain": steps_per_domain,
            "theoretical_bayes_ceiling": 0.9516,
            "dmoe_v2_params": dmoe_params,
            "regular_transformer_matched_params": dense_params,
        },
        "continual_learning_20_seeds": summary_continual,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== dmoe_v2_continual_hard_control 20-Seed Summary ===")
    for cond, s_dict in summary_continual.items():
        avg = s_dict["avg_acc_final"]
        a_ret = s_dict["acc_A_final"]
        bwt = s_dict["bwt_nats"]
        print(
            f"{cond:50s} | AvgAcc={avg['mean']:.4f} ± {avg['ci95']:.4f} "
            f"| TaskA_Retained={a_ret['mean']:.4f} | BWT={bwt['mean']:+.4f} nats"
        )


if __name__ == "__main__":
    main()
