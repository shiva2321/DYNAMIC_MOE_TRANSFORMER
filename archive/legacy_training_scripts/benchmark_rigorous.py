"""
Rigorous Comparative Benchmark:
Dense Transformer vs Static Softmax MoE vs Dynamic Hyperspace MoE.
Evaluates:
1. Continual Learning & Catastrophic Forgetting (Sequential Domain Training)
2. Domain Specialization Purity (Gini Index)
3. Unseen Hybrid Domain Generalization (Cross-Expert Collaboration)
4. Fine-Grained Token Attribution Visualizer
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

def evaluate_model_loss(model, corpus, domain_id: int, num_batches: int = 8, device: torch.device = None) -> float:
    """Computes average cross-entropy loss and perplexity on a given domain."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets, _ = corpus.get_batch(domain_id, is_eval=True)
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss, _ = model(inputs, targets=targets)
            total_loss += loss.item()
    return total_loss / num_batches

def sync_hyper_optimizer(optimizer, model):
    existing_params = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing_params.add(p)
    new_params = [p for p in model.parameters() if p not in existing_params and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params})

def run_benchmark(device_str: str = "cuda", steps_per_domain: int = 60):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("================================================================================")
    print("  [BENCHMARK] RIGOROUS EVALUATION: DENSE vs STATIC MoE vs DYNAMIC HYPER-MoE")
    print("================================================================================")
    print(f"Device: {device} | Steps per Domain: {steps_per_domain} (Total Sequential Steps: {steps_per_domain * 4})\n")

    corpus = RichMultiDomainCorpus(seq_len=96, batch_size=8)
    
    # 1. Instantiate the 3 competitor models with matched parameters
    d_model = 256
    n_layers = 3
    n_heads = 4
    d_ff = 512
    d_hyper = 2048

    print("-> Initializing Models...")
    model_dense = DenseTransformerLM(vocab_size=256, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff * 2).to(device)
    model_static_moe = StaticSoftmaxMoELM(vocab_size=256, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, num_experts=4, top_k=2).to(device)
    model_hyper_moe = HyperTransformerLM(vocab_size=256, d_model=d_model, n_layers=n_layers, n_heads=n_heads, d_ff=d_ff, d_hyper=d_hyper, top_k=2, spawn_threshold=0.28, max_experts=16).to(device)

    opt_dense = torch.optim.AdamW(model_dense.parameters(), lr=5e-4)
    opt_static = torch.optim.AdamW(model_static_moe.parameters(), lr=5e-4)
    opt_hyper = torch.optim.AdamW(model_hyper_moe.parameters(), lr=5e-4)

    # ----------------------------------------------------------------------
    # EXPERIMENT 1: Sequential Lifelong Learning Stream (Domain 0 -> 1 -> 2 -> 3)
    # ----------------------------------------------------------------------
    domains = [0, 1, 2, 3] # Code, Math, Philosophy, JSON
    domain_names = ["Code", "Math", "Philosophy", "JSON"]
    
    print("\n--- Phase 1: Sequential Domain Training Stream ---")
    start_time = time.time()
    
    # Track domain expert activations for Hyper-MoE
    hyper_expert_activations = {d: {} for d in domains}
    static_expert_activations = {d: {} for d in domains}

    total_steps = steps_per_domain * len(domains)
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
        model_static_moe.train()
        opt_static.zero_grad()
        _, loss_static, telem_static = model_static_moe(inputs, targets=targets)
        loss_static.backward()
        opt_static.step()

        # 3. Train Dynamic Hyper-MoE
        model_hyper_moe.train()
        opt_hyper.zero_grad()
        prev_exp = model_hyper_moe.blocks[0].hyper_moe.num_experts
        _, loss_hyper, telem_hyper = model_hyper_moe(inputs, targets=targets, allow_spawning=True)
        if model_hyper_moe.blocks[0].hyper_moe.num_experts != prev_exp:
            sync_hyper_optimizer(opt_hyper, model_hyper_moe)
            print(f"[SPAWN @ Step {step:3d}] Domain: {d_name:<10} | Hyper-MoE Experts: {model_hyper_moe.blocks[0].hyper_moe.num_experts}")

        loss_hyper.backward()
        opt_hyper.step()

        # Record expert routing stats
        if telem_hyper:
            for exp in telem_hyper[0]["top_indices"].view(-1).tolist():
                hyper_expert_activations[d_id][exp] = hyper_expert_activations[d_id].get(exp, 0) + 1
        if telem_static:
            for exp in telem_static[0]["top_indices"].view(-1).tolist():
                static_expert_activations[d_id][exp] = static_expert_activations[d_id].get(exp, 0) + 1

        if step % steps_per_domain == 0:
            print(f"Completed Domain {domain_idx+1}/4 ({d_name:<10}) | Dense Loss: {loss_dense.item():.3f} | Static MoE Loss: {loss_static.item():.3f} | Hyper-MoE Loss: {loss_hyper.item():.3f}")

    print(f"Sequential training completed in {time.time() - start_time:.2f}s\n")

    # ----------------------------------------------------------------------
    # EXPERIMENT 2: Catastrophic Forgetting & Domain Retention Evaluation
    # ----------------------------------------------------------------------
    print("================================================================================")
    print("  [EVAL 1] CATASTROPHIC FORGETTING / KNOWLEDGE RETENTION EVALUATION")
    print("  (Measuring Perplexity across all domains after sequential training)")
    print("================================================================================")
    
    print(f"{'Domain':<14} | {'Dense Perplexity':<18} | {'Static MoE Perplexity':<22} | {'Hyper-MoE Perplexity':<20}")
    print("-" * 80)
    
    dense_ppls, static_ppls, hyper_ppls = [], [], []
    for d_id in domains:
        d_name = domain_names[d_id]
        l_dense = evaluate_model_loss(model_dense, corpus, d_id, device=device)
        l_static = evaluate_model_loss(model_static_moe, corpus, d_id, device=device)
        l_hyper = evaluate_model_loss(model_hyper_moe, corpus, d_id, device=device)
        
        ppl_dense = math.exp(min(12.0, l_dense))
        ppl_static = math.exp(min(12.0, l_static))
        ppl_hyper = math.exp(min(12.0, l_hyper))
        
        dense_ppls.append(ppl_dense)
        static_ppls.append(ppl_static)
        hyper_ppls.append(ppl_hyper)
        
        print(f"{d_name:<14} | {ppl_dense:18.2f} | {ppl_static:22.2f} | {ppl_hyper:20.2f}")

    print("-" * 80)
    print(f"{'Mean Perplexity':<14} | {sum(dense_ppls)/4:18.2f} | {sum(static_ppls)/4:22.2f} | {sum(hyper_ppls)/4:20.2f}")

    # ----------------------------------------------------------------------
    # EXPERIMENT 3: Domain Specialization Purity (Gini Impurity Index)
    # ----------------------------------------------------------------------
    print("\n================================================================================")
    print("  [EVAL 2] DOMAIN SPECIALIZATION PURITY & GINI IMPURITY")
    print("  (0.00 = Perfectly Pure Specialist, 0.75 = Uniform Non-Specialized)")
    print("================================================================================")
    total_hyper_experts = model_hyper_moe.blocks[0].hyper_moe.num_experts
    print(f"Total Spawned Hyper-MoE Experts in Layer 0: {total_hyper_experts}")
    
    for exp_id in range(total_hyper_experts):
        dist = [hyper_expert_activations[d_id].get(exp_id, 0) for d_id in domains]
        total_act = sum(dist)
        if total_act > 0:
            gini = AttributionTracer.compute_gini_impurity(dist)
            top_domain = domain_names[dist.index(max(dist))]
            top_pct = (max(dist) / total_act) * 100.0
            print(f"Expert {exp_id:2d} -> Primary Domain: {top_domain:<12} ({top_pct:5.1f}% of tokens) | Gini Impurity: {gini:.3f} | Total Tokens: {total_act}")

    # ----------------------------------------------------------------------
    # EXPERIMENT 4: Unseen Hybrid Domain Compositional Generalization
    # ----------------------------------------------------------------------
    print("\n================================================================================")
    print("  [EVAL 3] UNSEEN HYBRID DOMAIN STRESS TEST (Math + Code in Superposition)")
    print("================================================================================")
    l_hybrid_dense = evaluate_model_loss(model_dense, corpus, domain_id=4, device=device)
    l_hybrid_static = evaluate_model_loss(model_static_moe, corpus, domain_id=4, device=device)
    l_hybrid_hyper = evaluate_model_loss(model_hyper_moe, corpus, domain_id=4, device=device)
    
    print(f"Dense Transformer Hybrid Loss:       {l_hybrid_dense:.4f} (Perplexity: {math.exp(min(12.0, l_hybrid_dense)):.2f})")
    print(f"Static Softmax MoE Hybrid Loss:      {l_hybrid_static:.4f} (Perplexity: {math.exp(min(12.0, l_hybrid_static)):.2f})")
    print(f"Dynamic Hyper-MoE + Bus Hybrid Loss: {l_hybrid_hyper:.4f} (Perplexity: {math.exp(min(12.0, l_hybrid_hyper)):.2f})")

    # ----------------------------------------------------------------------
    # EXPERIMENT 5: Fine-Grained Token-by-Token Attribution Tracing
    # ----------------------------------------------------------------------
    print("\n================================================================================")
    print("  [EVAL 4] TOKEN-LEVEL EXPERT ATTRIBUTION TRACE (Who wrote what?)")
    print("================================================================================")
    
    sample_tests = [
        ("Code Prompt", "def dijkstra("),
        ("Math Prompt", "Theorem: Spectral"),
        ("Philosophy Prompt", "The hard problem of"),
        ("Hybrid Math+Code Prompt", "def power_iteration(matrix_A):")
    ]
    
    for prompt_title, text in sample_tests:
        prompt_tensor = torch.tensor([corpus.encode(text)], dtype=torch.long, device=device)
        trace_log = AttributionTracer.trace_generation_attribution(
            model_hyper_moe, prompt_tensor, corpus, max_new_tokens=32, temperature=0.7
        )
        AttributionTracer.print_colored_trace(trace_log, title=f"Attribution for: '{text}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps_per_domain", type=int, default=60)
    args = parser.parse_args()
    run_benchmark(device_str=args.device, steps_per_domain=args.steps_per_domain)
