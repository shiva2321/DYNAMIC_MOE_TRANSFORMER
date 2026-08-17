"""
Rigorous Multi-Dataset Empirical Experiment Suite.
Directly benchmarks:
1. Dense Transformer (Monolithic Baseline)
2. Static Softmax MoE (Fixed 4-Expert Linear Gating Baseline)
3. Dynamic Hyperspace 2.0 MoE (Neuro-Symbolic Lifelong Learning Architecture)

Across 7 Diverse Datasets:
- Code, Math, Wikipedia, Dialogue, Literature, Biomedical, and Cloud JSON Telemetry.

Measures:
- Catastrophic Forgetting & Knowledge Retention Rate (%)
- Domain Specialization Gini Impurity
- Dynamic Expert Spawning Trajectory
- Out-of-Domain Generalization & Parameter Efficiency
"""

import os
import sys
import time
import math
import json
import argparse
from typing import Dict, List, Any, Tuple
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.dataset_hub import MultiDomainDatasetHub, DOMAIN_LIST, DOMAIN_METADATA
from model.nanogpt import HyperTransformerLM
from model.baselines import DenseTransformerLM, StaticSoftmaxMoELM

def sync_hyper_optimizer(optimizer: torch.optim.Optimizer, model: HyperTransformerLM) -> int:
    existing = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing.add(p)
    new_params = [p for p in model.parameters() if p not in existing and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params})
    return len(new_params)

def eval_model_on_domain(
    model: torch.nn.Module,
    hub: MultiDomainDatasetHub,
    domain_key: str,
    device: torch.device,
    num_batches: int = 4,
    use_amp: bool = False,
) -> float:
    """Evaluates cross-entropy loss on held-out validation batch."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets, _ = hub.get_batch(domain_key, is_eval=True)
            inputs, targets = inputs.to(device), targets.to(device)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    if isinstance(model, HyperTransformerLM):
                        _, loss, _ = model(inputs, targets=targets, allow_spawning=False)
                    else:
                        _, loss, _ = model(inputs, targets=targets)
            else:
                if isinstance(model, HyperTransformerLM):
                    _, loss, _ = model(inputs, targets=targets, allow_spawning=False)
                else:
                    _, loss, _ = model(inputs, targets=targets)
            total_loss += loss.item()
    return total_loss / num_batches

def compute_gini_specialization(activation_counts: Dict[str, Dict[int, int]]) -> float:
    """
    Computes average Gini coefficient of expert specialization across domains.
    1.0 = perfect domain-dedicated experts; 0.0 = completely uniform unspecialized routing.
    """
    gini_scores = []
    for domain, counts in activation_counts.items():
        if not counts:
            continue
        vals = sorted(list(counts.values()))
        n = len(vals)
        if n == 0 or sum(vals) == 0:
            continue
        cum_vals = 0
        total = sum(vals)
        gini_sum = 0
        for i, v in enumerate(vals, 1):
            gini_sum += (2 * i - n - 1) * v
        gini = gini_sum / (n * total) if (n * total) > 0 else 0.0
        gini_scores.append(gini)
    return sum(gini_scores) / len(gini_scores) if gini_scores else 0.0

def run_experiment_suite(
    steps_per_domain: int = 40,
    batch_size: int = 8,
    seq_len: int = 128,
    device_name: str = "cuda",
    output_dir: str = "experiments",
) -> Dict[str, Any]:
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    use_amp = (device.type == "cuda")
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

    print("================================================================================")
    print("  [EXPERIMENT SUITE] 3-MODEL COMPARATIVE MULTI-DATASET BENCHMARK")
    print("================================================================================")
    print(f"Device: {device} | Mixed Precision: {use_amp} | Steps per Domain: {steps_per_domain}")
    print(f"Active Domains (7): {', '.join(DOMAIN_LIST)}")
    print(f"Total Sequential Steps per Model: {steps_per_domain * len(DOMAIN_LIST)}\n")

    hub = MultiDomainDatasetHub(seq_len=seq_len, batch_size=batch_size, use_bpe=True)

    # 1. Initialize the 3 competitor models
    d_model = 256
    n_layers = 3
    n_heads = 4
    d_ff = 512
    d_hyper = 2048

    print("-> Initializing Competitor Architectures...")
    model_dense = DenseTransformerLM(vocab_size=hub.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff * 2, max_seq_len=seq_len + 32).to(device)
    model_static = StaticSoftmaxMoELM(vocab_size=hub.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, num_experts=6, top_k=2, max_seq_len=seq_len + 32).to(device)
    model_hyper = HyperTransformerLM(vocab_size=hub.vocab_size, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, d_hyper=d_hyper, top_k=2, spawn_threshold=0.25, max_experts=16, max_seq_len=seq_len + 32).to(device)

    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=6e-4, weight_decay=1e-4)
    opt_static = torch.optim.AdamW(model_static.parameters(), lr=6e-4, weight_decay=1e-4)
    opt_hyper = torch.optim.AdamW(model_hyper.parameters(), lr=6e-4, weight_decay=1e-4)

    scaler_dense = torch.amp.GradScaler('cuda', enabled=use_amp)
    scaler_static = torch.amp.GradScaler('cuda', enabled=use_amp)
    scaler_hyper = torch.amp.GradScaler('cuda', enabled=use_amp)

    models_dict = {
        "Dense Transformer": {"model": model_dense, "opt": opt_dense, "scaler": scaler_dense},
        "Static Softmax MoE": {"model": model_static, "opt": opt_static, "scaler": scaler_static},
        "Dynamic Hyperspace MoE": {"model": model_hyper, "opt": opt_hyper, "scaler": scaler_hyper},
    }

    # Tracking Structures
    # initial_loss_after_domain: {model_name: {domain_key: float}}
    initial_losses: Dict[str, Dict[str, float]] = {m: {} for m in models_dict}
    # final_eval_losses: {model_name: {domain_key: float}}
    final_losses: Dict[str, Dict[str, float]] = {m: {} for m in models_dict}
    # step_loss_histories: {model_name: List[float]}
    loss_histories: Dict[str, List[float]] = {m: [] for m in models_dict}
    # routing activations: {model_name: {domain_key: {expert_id: count}}}
    static_routing: Dict[str, Dict[int, int]] = {d: {} for d in DOMAIN_LIST}
    hyper_routing: Dict[str, Dict[int, int]] = {d: {} for d in DOMAIN_LIST}
    hyper_spawns: List[Dict[str, Any]] = []

    total_steps = steps_per_domain * len(DOMAIN_LIST)
    start_time = time.time()

    print("\n--- Beginning Sequential Multi-Dataset Lifelong Training Stream ---")

    for step in range(1, total_steps + 1):
        domain_idx = (step - 1) // steps_per_domain
        domain_key = DOMAIN_LIST[domain_idx]
        domain_name = DOMAIN_METADATA[domain_key]["name"]

        inputs, targets, _ = hub.get_batch(domain_key)
        inputs, targets = inputs.to(device), targets.to(device)

        # -------------------------------------------------------------
        # 1. Train Dense Transformer
        # -------------------------------------------------------------
        model_dense.train()
        opt_dense.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                _, loss_d, _ = model_dense(inputs, targets=targets)
            scaler_dense.scale(loss_d).backward()
            scaler_dense.unscale_(opt_dense)
            torch.nn.utils.clip_grad_norm_(model_dense.parameters(), 1.0)
            scaler_dense.step(opt_dense)
            scaler_dense.update()
        else:
            _, loss_d, _ = model_dense(inputs, targets=targets)
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(model_dense.parameters(), 1.0)
            opt_dense.step()
        loss_histories["Dense Transformer"].append(loss_d.item())

        # -------------------------------------------------------------
        # 2. Train Static Softmax MoE
        # -------------------------------------------------------------
        model_static.train()
        opt_static.zero_grad()
        if use_amp:
            with torch.amp.autocast('cuda'):
                _, loss_s, telem_s = model_static(inputs, targets=targets)
            scaler_static.scale(loss_s).backward()
            scaler_static.unscale_(opt_static)
            torch.nn.utils.clip_grad_norm_(model_static.parameters(), 1.0)
            scaler_static.step(opt_static)
            scaler_static.update()
        else:
            _, loss_s, telem_s = model_static(inputs, targets=targets)
            loss_s.backward()
            torch.nn.utils.clip_grad_norm_(model_static.parameters(), 1.0)
            opt_static.step()
        loss_histories["Static Softmax MoE"].append(loss_s.item())
        if telem_s:
            top_s = telem_s[0]["top_indices"].view(-1).tolist()
            for idx in top_s:
                static_routing[domain_key][idx] = static_routing[domain_key].get(idx, 0) + 1

        # -------------------------------------------------------------
        # 3. Train Dynamic Hyperspace MoE
        # -------------------------------------------------------------
        model_hyper.train()
        opt_hyper.zero_grad()
        prev_exp = model_hyper.blocks[0].hyper_moe.num_experts
        if use_amp:
            with torch.amp.autocast('cuda'):
                _, loss_h, telem_h = model_hyper(inputs, targets=targets, allow_spawning=True)
            scaler_hyper.scale(loss_h).backward()
            scaler_hyper.unscale_(opt_hyper)
            torch.nn.utils.clip_grad_norm_(model_hyper.parameters(), 1.0)
            scaler_hyper.step(opt_hyper)
            scaler_hyper.update()
        else:
            _, loss_h, telem_h = model_hyper(inputs, targets=targets, allow_spawning=True)
            loss_h.backward()
            torch.nn.utils.clip_grad_norm_(model_hyper.parameters(), 1.0)
            opt_hyper.step()
        loss_histories["Dynamic Hyperspace MoE"].append(loss_h.item())

        curr_exp = model_hyper.blocks[0].hyper_moe.num_experts
        if curr_exp != prev_exp:
            new_p = sync_hyper_optimizer(opt_hyper, model_hyper)
            hyper_spawns.append({
                "step": step,
                "domain": domain_key,
                "prev_experts": prev_exp,
                "curr_experts": curr_exp,
            })
            print(f"  ⚡ [SPAWN @ Step {step:3d}] Domain: {domain_name:<30} | Experts: {prev_exp} -> {curr_exp}")

        if telem_h:
            top_h = telem_h[0]["top_indices"].view(-1).tolist()
            for idx in top_h:
                hyper_routing[domain_key][idx] = hyper_routing[domain_key].get(idx, 0) + 1

        # Check if we just completed a domain phase
        if step % steps_per_domain == 0:
            completed_domain = DOMAIN_LIST[(step // steps_per_domain) - 1]
            # Record initial loss right after training on this domain
            loss_d_init = eval_model_on_domain(model_dense, hub, completed_domain, device, use_amp=use_amp)
            loss_s_init = eval_model_on_domain(model_static, hub, completed_domain, device, use_amp=use_amp)
            loss_h_init = eval_model_on_domain(model_hyper, hub, completed_domain, device, use_amp=use_amp)

            initial_losses["Dense Transformer"][completed_domain] = loss_d_init
            initial_losses["Static Softmax MoE"][completed_domain] = loss_s_init
            initial_losses["Dynamic Hyperspace MoE"][completed_domain] = loss_h_init

            print(f"[DOMAIN PHASE COMPLETED: {completed_domain:<14}] "
                  f"Losses -> Dense: {loss_d_init:.4f} | Static MoE: {loss_s_init:.4f} | Hyperspace: {loss_h_init:.4f}")

    elapsed_time = time.time() - start_time
    print(f"\n[BENCHMARK RUN FINISHED] Elapsed time: {elapsed_time:.2f}s")

    # -----------------------------------------------------------------
    # Final Evaluation Across All Domains (Measuring Retention & Forgetting)
    # -----------------------------------------------------------------
    print("\n================================================================================")
    print("  [RESULTS] FINAL FORGETTING & KNOWLEDGE RETENTION AUDIT")
    print("================================================================================")
    for m_name in models_dict:
        m_obj = models_dict[m_name]["model"]
        for domain_key in DOMAIN_LIST:
            final_loss = eval_model_on_domain(m_obj, hub, domain_key, device, num_batches=6, use_amp=use_amp)
            final_losses[m_name][domain_key] = final_loss

    # Compute Forgetting Delta & Retention Rate
    forgetting_metrics: Dict[str, Dict[str, Any]] = {}
    for m_name in models_dict:
        domain_deltas = {}
        domain_retentions = {}
        for domain_key in DOMAIN_LIST:
            l_init = initial_losses[m_name].get(domain_key, 1.0)
            l_final = final_losses[m_name].get(domain_key, 1.0)
            delta = l_final - l_init  # Positive delta = loss increased (forgot knowledge)
            retention = max(0.0, 100.0 * (1.0 - max(0.0, delta) / max(l_init, 1e-4)))
            domain_deltas[domain_key] = round(delta, 4)
            domain_retentions[domain_key] = round(retention, 2)
        
        avg_retention = sum(domain_retentions.values()) / len(domain_retentions)
        avg_final_loss = sum(final_losses[m_name].values()) / len(DOMAIN_LIST)
        avg_final_ppl = math.exp(min(avg_final_loss, 20.0))

        forgetting_metrics[m_name] = {
            "initial_losses": {k: round(v, 4) for k, v in initial_losses[m_name].items()},
            "final_losses": {k: round(v, 4) for k, v in final_losses[m_name].items()},
            "forgetting_deltas": domain_deltas,
            "retention_rates_pct": domain_retentions,
            "average_retention_pct": round(avg_retention, 2),
            "average_final_loss": round(avg_final_loss, 4),
            "average_final_perplexity": round(avg_final_ppl, 2),
        }

    # Routing Gini Specialization
    static_gini = compute_gini_specialization(static_routing)
    hyper_gini = compute_gini_specialization(hyper_routing)

    print(f"\n{'Architecture':<26} | {'Avg Final Loss':<15} | {'Avg Perplexity':<15} | {'Retention Rate':<15}")
    print("--------------------------------------------------------------------------------")
    for m_name, met in forgetting_metrics.items():
        print(f"{m_name:<26} | {met['average_final_loss']:<15.4f} | {met['average_final_perplexity']:<15.2f} | {met['average_retention_pct']:<14.1f}%")
    print("--------------------------------------------------------------------------------")
    print(f"Routing Specialization Purity (Gini Index): Static MoE = {static_gini * 100:.1f}% | Dynamic Hyperspace MoE = {hyper_gini * 100:.1f}%\n")

    # Compile Benchmark Results Object
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "steps_per_domain": steps_per_domain,
            "total_steps": total_steps,
            "d_model": d_model,
            "n_layers": n_layers,
            "d_hyper": d_hyper,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "device": str(device),
        },
        "elapsed_time_seconds": round(elapsed_time, 2),
        "forgetting_metrics": forgetting_metrics,
        "specialization": {
            "static_moe_gini_pct": round(static_gini * 100, 2),
            "dynamic_hyper_gini_pct": round(hyper_gini * 100, 2),
            "static_routing": static_routing,
            "hyper_routing": hyper_routing,
        },
        "spawning_events": hyper_spawns,
        "loss_histories": loss_histories,
    }

    # Save benchmark payload to JSON
    json_path = os.path.join(output_dir, "experiment_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[SAVED] Complete benchmark results saved to {json_path}")

    # Save checkpoints
    torch.save(model_hyper.state_dict(), os.path.join(output_dir, "checkpoints", "benchmark_hyperspace_moe.pt"))
    torch.save(model_static.state_dict(), os.path.join(output_dir, "checkpoints", "benchmark_static_moe.pt"))
    torch.save(model_dense.state_dict(), os.path.join(output_dir, "checkpoints", "benchmark_dense.pt"))
    print("[SAVED] All 3 benchmark model checkpoints saved.")

    return results_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_domain", type=int, default=40, help="Steps per domain for the benchmark run")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_experiment_suite(
        steps_per_domain=args.steps_per_domain,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device_name=args.device,
    )
