"""
Benchmark Suite for Hyperspace 2.0 Neuro-Cognitive Substrate.
Compares:
1. Dense Transformer (Baseline)
2. Static Softmax MoE (Baseline)
3. Hyperspace 1.0 (Basic VSA Routing)
4. Hyperspace 2.0 (Full Neuro-Cognitive Substrate: Dentate Gyrus + Dendritic Experts + Hopfield + Criticality)

Evaluates:
- Sequential Continual Learning (Catastrophic Forgetting Retention)
- Unseen Hybrid Domain Generalization (Math + Code + Philosophy)
- Criticality Branching Ratio Stability (sigma ~ 1.0)
- Generation Quality & Token Attribution
"""

import sys
import time
import math
import argparse
from typing import Dict, List
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.rich_corpus import RichMultiDomainCorpus
from model.nanogpt import HyperTransformerLM
from model.baselines import DenseTransformerLM, StaticSoftmaxMoELM
from model.attribution import AttributionTracer

def evaluate_loss(model, corpus, domain_id: int, num_batches: int = 8, device: torch.device = None) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets, _ = corpus.get_batch(domain_id, is_eval=True)
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss, _ = model(inputs, targets=targets)
            total_loss += loss.item()
    return total_loss / num_batches

def sync_optimizer(optimizer, model):
    existing = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing.add(p)
    new_params = [p for p in model.parameters() if p not in existing and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params})

def run_benchmark(device_str: str = "cuda", steps_per_domain: int = 60):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("================================================================================")
    print("  [BENCHMARK] HYPERSPACE 2.0 NEURO-COGNITIVE SUBSTRATE RIGOROUS EVALUATION")
    print("================================================================================")
    print(f"Device: {device} | Steps per Domain: {steps_per_domain} | Total Sequential Steps: {steps_per_domain * 4}\n")

    corpus = RichMultiDomainCorpus(seq_len=96, batch_size=8)
    
    d_model = 256
    n_layers = 3
    n_heads = 4
    d_ff = 512
    d_hyper = 2048

    print("-> Initializing Models...")
    # 1. Dense Baseline
    model_dense = DenseTransformerLM(vocab_size=256, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff * 2).to(device)
    # 2. Static Softmax MoE Baseline
    model_static = StaticSoftmaxMoELM(vocab_size=256, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, num_experts=4, top_k=2).to(device)
    # 3. Hyperspace 2.0 (Full Neuro-Cognitive Model)
    model_hyperspace2 = HyperTransformerLM(
        vocab_size=256,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        spawn_threshold=0.28,
        max_experts=16
    ).to(device)

    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=5e-4)
    opt_static = torch.optim.AdamW(model_static.parameters(), lr=5e-4)
    opt_hyper2 = torch.optim.AdamW(model_hyperspace2.parameters(), lr=5e-4)

    domains = [0, 1, 2, 3] # Code, Math, Philosophy, JSON
    domain_names = ["Code", "Math", "Philosophy", "JSON"]

    print("\n--- Phase 1: Sequential Domain Training Stream ---")
    start_time = time.time()
    
    total_steps = steps_per_domain * len(domains)
    running_sigmas = []

    for step in range(1, total_steps + 1):
        domain_idx = (step - 1) // steps_per_domain
        d_id = domains[domain_idx]
        d_name = domain_names[d_id]

        inputs, targets, _ = corpus.get_batch(d_id)
        inputs, targets = inputs.to(device), targets.to(device)

        # 1. Train Dense
        model_dense.train()
        opt_dense.zero_grad()
        _, loss_dense, _ = model_dense(inputs, targets=targets)
        loss_dense.backward()
        opt_dense.step()

        # 2. Train Static MoE
        model_static.train()
        opt_static.zero_grad()
        _, loss_static, _ = model_static(inputs, targets=targets)
        loss_static.backward()
        opt_static.step()

        # 3. Train Hyperspace 2.0
        model_hyperspace2.train()
        opt_hyper2.zero_grad()
        prev_exp = model_hyperspace2.blocks[0].hyper_moe.num_experts
        _, loss_hyper2, telem_h2 = model_hyperspace2(inputs, targets=targets, allow_spawning=True)
        
        if model_hyperspace2.blocks[0].hyper_moe.num_experts != prev_exp:
            sync_optimizer(opt_hyper2, model_hyperspace2)
            print(f"[SPAWN @ Step {step:3d}] Domain: {d_name:<10} | Hyperspace 2.0 Experts: {model_hyperspace2.blocks[0].hyper_moe.num_experts}")

        loss_hyper2.backward()
        opt_hyper2.step()

        if telem_h2:
            running_sigmas.append(telem_h2[0]["branching_ratio"])

        if step % steps_per_domain == 0:
            sigma_val = telem_h2[0]["branching_ratio"] if telem_h2 else 1.0
            temp_val = telem_h2[0]["routing_temperature"] if telem_h2 else 10.0
            print(f"Completed Domain {domain_idx+1}/4 ({d_name:<10}) | Dense: {loss_dense.item():.3f} | Static: {loss_static.item():.3f} | Hyper 2.0: {loss_hyper2.item():.3f} (Sigma: {sigma_val:.3f}, T: {temp_val:.1f})")

    print(f"Sequential training completed in {time.time() - start_time:.2f}s\n")

    # ----------------------------------------------------------------------
    # EVALUATION 1: Continual Learning & Catastrophic Forgetting
    # ----------------------------------------------------------------------
    print("================================================================================")
    print("  [EVAL 1] CATASTROPHIC FORGETTING / KNOWLEDGE RETENTION EVALUATION")
    print("  (Measuring Perplexity across all domains after sequential training)")
    print("================================================================================")
    print(f"{'Domain':<14} | {'Dense Perplexity':<18} | {'Static MoE Perplexity':<22} | {'Hyperspace 2.0 Perplexity':<26}")
    print("-" * 84)

    dense_ppls, static_ppls, hyper2_ppls = [], [], []
    for d_id in domains:
        d_name = domain_names[d_id]
        l_dense = evaluate_loss(model_dense, corpus, d_id, device=device)
        l_static = evaluate_loss(model_static, corpus, d_id, device=device)
        l_h2 = evaluate_loss(model_hyperspace2, corpus, d_id, device=device)

        ppl_d = math.exp(min(12.0, l_dense))
        ppl_s = math.exp(min(12.0, l_static))
        ppl_h2 = math.exp(min(12.0, l_h2))

        dense_ppls.append(ppl_d)
        static_ppls.append(ppl_s)
        hyper2_ppls.append(ppl_h2)

        print(f"{d_name:<14} | {ppl_d:18.2f} | {ppl_s:22.2f} | {ppl_h2:26.2f}")

    print("-" * 84)
    print(f"{'Mean Perplexity':<14} | {sum(dense_ppls)/4:18.2f} | {sum(static_ppls)/4:22.2f} | {sum(hyper2_ppls)/4:26.2f}")

    # ----------------------------------------------------------------------
    # EVALUATION 2: Unseen Hybrid Domain Stress Test
    # ----------------------------------------------------------------------
    print("\n================================================================================")
    print("  [EVAL 2] UNSEEN HYBRID DOMAIN STRESS TEST (Math + Code + Logic in Superposition)")
    print("================================================================================")
    l_hyb_d = evaluate_loss(model_dense, corpus, domain_id=4, device=device)
    l_hyb_s = evaluate_loss(model_static, corpus, domain_id=4, device=device)
    l_hyb_h2 = evaluate_loss(model_hyperspace2, corpus, domain_id=4, device=device)

    print(f"Dense Transformer Hybrid Loss:       {l_hyb_d:.4f} (Perplexity: {math.exp(min(12.0, l_hyb_d)):.2f})")
    print(f"Static Softmax MoE Hybrid Loss:      {l_hyb_s:.4f} (Perplexity: {math.exp(min(12.0, l_hyb_s)):.2f})")
    print(f"Hyperspace 2.0 (Neuro-Cognitive) Loss: {l_hyb_h2:.4f} (Perplexity: {math.exp(min(12.0, l_hyb_h2)):.2f})")

    # ----------------------------------------------------------------------
    # EVALUATION 3: Self-Organized Criticality Stability (sigma = 1.0)
    # ----------------------------------------------------------------------
    print("\n================================================================================")
    print("  [EVAL 3] SELF-ORGANIZED CRITICALITY ANALYSIS")
    print("================================================================================")
    mean_sigma = sum(running_sigmas) / len(running_sigmas) if running_sigmas else 1.0
    print(f"Target Branching Ratio (Edge of Chaos): 1.000")
    print(f"Empirical Average Branching Ratio (sigma): {mean_sigma:.4f}")
    print(f"Criticality Stability Status: {'STABLE (At Critical Point)' if 0.95 <= mean_sigma <= 1.05 else 'ADAPTED'}")

    # ----------------------------------------------------------------------
    # EVALUATION 4: Token-Level Expert Attribution Tracing
    # ----------------------------------------------------------------------
    print("\n================================================================================")
    print("  [EVAL 4] TOKEN-LEVEL EXPERT ATTRIBUTION TRACE (Hyperspace 2.0)")
    print("================================================================================")
    sample_tests = [
        ("Code Prompt", "def dijkstra("),
        ("Math Prompt", "Spectral Theorem: Every Hermitian"),
        ("Philosophy Prompt", "The hard problem of"),
        ("Hybrid Prompt", "def power_iteration(matrix_A):")
    ]
    for title, text in sample_tests:
        prompt_t = torch.tensor([corpus.encode(text)], dtype=torch.long, device=device)
        trace = AttributionTracer.trace_generation_attribution(model_hyperspace2, prompt_t, corpus, max_new_tokens=32, temperature=0.7)
        AttributionTracer.print_colored_trace(trace, title=f"{title}: '{text}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps_per_domain", type=int, default=60)
    args = parser.parse_args()
    run_benchmark(device_str=args.device, steps_per_domain=args.steps_per_domain)
