import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.jepa import JEPAModule, MASK_TOKEN_ID
from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.continual import ContinualLearner
from tests.test_jepa import _train_model, _train_and_eval_probe
from tests.test_simulation_loop import _sim_model
from tests.test_tokenizer_free import _structured_byte_batch
from tests.test_continual_learning import (
    _retention_model,
    _task_a_batch,
    _task_b_batch,
    _eval_task_a_loss,
)


def eval_phase2a_retention() -> dict:
    random.seed(0)
    torch.manual_seed(0)
    mitigated = _retention_model()
    torch.manual_seed(0)
    baseline = _retention_model()

    eval_x, eval_targets = _task_a_batch(batch_size=16)

    mitigated_optimizer = torch.optim.AdamW(mitigated.parameters(), lr=5e-3)
    learner = ContinualLearner(
        mitigated,
        mitigated_optimizer,
        replay_buffer_capacity=200,
        replay_batch_size=8,
        distillation_weight=0.5,
        mutation_check_interval=1000,
    )
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=5e-3)

    train_steps_per_task = 40

    for _ in range(train_steps_per_task):
        x, targets = _task_a_batch()
        learner.step(x, targets, output_head="token")

        baseline_optimizer.zero_grad()
        out = baseline(x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss.backward()
        baseline_optimizer.step()

    mitigated_after_a = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    baseline_after_a = _eval_task_a_loss(baseline, eval_x, eval_targets)

    for _ in range(train_steps_per_task):
        x, targets = _task_b_batch()
        learner.step(x, targets, output_head="token")

        baseline_optimizer.zero_grad()
        out = baseline(x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss.backward()
        baseline_optimizer.step()

    mitigated_after_b = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    baseline_after_b = _eval_task_a_loss(baseline, eval_x, eval_targets)

    mitigated_forgetting = mitigated_after_b - mitigated_after_a
    baseline_forgetting = baseline_after_b - baseline_after_a

    return {
        "mitigated_after_a": round(mitigated_after_a, 4),
        "mitigated_after_b": round(mitigated_after_b, 4),
        "mitigated_forgetting": round(mitigated_forgetting, 4),
        "baseline_after_a": round(baseline_after_a, 4),
        "baseline_after_b": round(baseline_after_b, 4),
        "baseline_forgetting": round(baseline_forgetting, 4),
    }


def eval_phase2b_i_jepa() -> dict:
    torch.manual_seed(0)
    encoder = UniversalMultimodalEncoder(d_model=16)
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, mask_ratio=0.3, ema_decay=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(jepa.trainable_parameters()), lr=5e-3
    )
    losses = []
    for _ in range(60):
        token_ids = torch.randint(0, 50, (8, 10), dtype=torch.long)
        optimizer.zero_grad()
        loss = jepa.compute_loss(token_ids)
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()
        losses.append(loss.item())

    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10

    with torch.no_grad():
        inputs = torch.randint(0, 50, (5, 10), dtype=torch.long)
        target_r, _ = jepa.target_encoder(inputs)
        pooled = target_r.mean(dim=1)
    pairwise_dists = torch.cdist(pooled, pooled)
    off_diagonal_mean = pairwise_dists[~torch.eye(5, dtype=torch.bool)].mean().item()

    model_jepa = _train_model(use_jepa=True, steps=80, seed=0)
    model_baseline = _train_model(use_jepa=False, steps=80, seed=0)
    jepa_probe_loss = _train_and_eval_probe(model_jepa, seed=1)
    baseline_probe_loss = _train_and_eval_probe(model_baseline, seed=1)

    torch.manual_seed(0)
    enc_grad = UniversalMultimodalEncoder(d_model=16)
    jepa_grad = JEPAModule(enc_grad, predictor_hidden_dim=32)
    x_grad = torch.randint(0, 50, (4, 8))
    loss_grad = jepa_grad.compute_loss(x_grad)
    loss_grad.backward()
    g = enc_grad.text_embed.weight.grad
    max_grad_real = g[:MASK_TOKEN_ID].abs().max().item()
    grad_mask = g[MASK_TOKEN_ID].abs().max().item()

    return {
        "early_avg": round(early_avg, 4),
        "late_avg": round(late_avg, 4),
        "off_diagonal_mean": round(off_diagonal_mean, 4),
        "jepa_probe_loss": round(jepa_probe_loss, 4),
        "baseline_probe_loss": round(baseline_probe_loss, 4),
        "max_grad_real_tokens": round(max_grad_real, 4),
        "grad_mask_token": round(grad_mask, 4),
    }


def eval_phase2b_ii_simulation_loop() -> dict:
    passed = 0
    max_norms = []
    steps_list = []
    for s in range(20):
        torch.manual_seed(s)
        m = _sim_model(
            max_simulation_steps=20,
            theta_converge=-1000.0,
            min_improvement=-1000.0,
            skip_threshold=-100.0,
        )
        m.eval()
        x = torch.randint(0, 100, (2, 6), dtype=torch.long)
        out = m(x)
        mr = out["z_real"].abs().max().item()
        mi = out["z_imag"].abs().max().item()
        st = out["simulation_steps_used"]
        ok = (st == 20) and (mr < 1e4) and (mi < 1e4)
        passed += int(ok)
        max_norms.append(max(mr, mi))
        steps_list.append(st)

    return {
        "passed_seeds": passed,
        "total_seeds": 20,
        "steps_min": min(steps_list),
        "steps_max": max(steps_list),
        "max_norm_min": round(min(max_norms), 4),
        "max_norm_median": round(sorted(max_norms)[10], 4),
        "max_norm_max": round(max(max_norms), 4),
    }


def eval_phase3_tokenizer_free() -> dict:
    tokenized_model = SelfStabilizingDynamicMorphicMoE(
        d_model=32, num_heads=2, d_ff=64, initial_layers=1, initial_experts=2, tokenizer_free=False
    )
    byte_model = SelfStabilizingDynamicMorphicMoE(
        d_model=32, num_heads=2, d_ff=64, initial_layers=1, initial_experts=2, tokenizer_free=True, chunk_size=4
    )
    n_tokenized = sum(p.numel() for p in tokenized_model.parameters())
    n_byte = sum(p.numel() for p in byte_model.parameters())

    def run_seed(seed: int):
        def train_m(use_jepa: bool):
            torch.manual_seed(seed)
            m = SelfStabilizingDynamicMorphicMoE(
                d_model=16,
                num_heads=2,
                d_ff=32,
                initial_layers=1,
                initial_experts=2,
                theta_up=100.0,
                theta_down=-1.0,
                tokenizer_free=True,
                chunk_size=4,
            )
            jepa_mod = JEPAModule(m.encoder, predictor_hidden_dim=32, mask_ratio=0.25, ema_decay=0.95)
            params = list(m.parameters()) + (list(jepa_mod.trainable_parameters()) if use_jepa else [])
            opt = torch.optim.AdamW(params, lr=5e-3)
            for step in range(80):
                torch.manual_seed(seed * 10000 + 1000 + step)
                b_in, b_coarse_tgt, _ = _structured_byte_batch(batch_size=16)
                opt.zero_grad()
                out = m(b_in)
                t_loss = F.cross_entropy(out["output"].reshape(-1, 256), b_coarse_tgt.reshape(-1))
                if use_jepa:
                    torch.manual_seed(seed * 10000 + 5000 + step)
                    t_loss = t_loss + jepa_mod.compute_loss(b_in)
                t_loss.backward()
                opt.step()
                if use_jepa:
                    jepa_mod.update_target_encoder()
            return m

        def eval_p(m):
            torch.manual_seed(seed + 100)
            for p in m.encoder.parameters():
                p.requires_grad_(False)
            probe = torch.nn.Linear(m.d_model * 2, 16)
            p_opt = torch.optim.AdamW(probe.parameters(), lr=2e-2)
            for step in range(80):
                torch.manual_seed(seed * 10000 + 9000 + step)
                b_in, _, topics = _structured_byte_batch(batch_size=16)
                with torch.no_grad():
                    r, i = m.encoder(b_in)
                    pooled = torch.cat([r.mean(dim=1), i.mean(dim=1)], dim=-1)
                p_opt.zero_grad()
                F.cross_entropy(probe(pooled), topics).backward()
                p_opt.step()
            torch.manual_seed(seed * 10000 + 9999)
            eval_in, _, eval_topics = _structured_byte_batch(batch_size=64)
            with torch.no_grad():
                r, i = m.encoder(eval_in)
                pooled = torch.cat([r.mean(dim=1), i.mean(dim=1)], dim=-1)
                return F.cross_entropy(probe(pooled), eval_topics).item()

        return eval_p(train_m(True)), eval_p(train_m(False))

    j_losses, b_losses = [], []
    wins = 0
    for s in range(20):
        jl, bl = run_seed(s)
        j_losses.append(jl)
        b_losses.append(bl)
        wins += int(jl < bl)
    jt = torch.tensor(j_losses)
    bt = torch.tensor(b_losses)

    return {
        "tokenized_params": n_tokenized,
        "tokenizer_free_params": n_byte,
        "seed0_jepa_probe_loss": round(j_losses[0], 4),
        "seed0_baseline_probe_loss": round(b_losses[0], 4),
        "wins_20_seeds": wins,
        "total_seeds": 20,
        "jepa_mean": round(jt.mean().item(), 4),
        "jepa_std": round(jt.std().item(), 4),
        "jepa_median": round(jt.median().item(), 4),
        "baseline_mean": round(bt.mean().item(), 4),
        "baseline_std": round(bt.std().item(), 4),
        "baseline_median": round(bt.median().item(), 4),
    }


def main() -> None:
    results = {
        "phase2a_retention": eval_phase2a_retention(),
        "phase2b_i_jepa": eval_phase2b_i_jepa(),
        "phase2b_ii_simulation_loop": eval_phase2b_ii_simulation_loop(),
        "phase3_tokenizer_free": eval_phase3_tokenizer_free(),
    }
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "phase2_phase3_eval.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
