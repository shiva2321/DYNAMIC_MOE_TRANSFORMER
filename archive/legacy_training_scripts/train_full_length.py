"""
Full-Length Multi-Domain Training & Generation Engine.
Trains Hyperspace 2.0 with BPE Sub-Word Tokenization across 5 diverse domains:
1. Python Algorithms & Systems Code
2. Mathematics & Quantum Physics
3. Philosophy of Mind & Epistemology
4. Cloud Architecture & JSON Telemetry
5. Narrative Literature & Science Fiction

Generates full-length, multi-paragraph, coherent text and code blocks with token-level expert attribution.
"""

import sys
import time
import math
import argparse
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from data.bpe_multidomain import BPEMultiDomainCorpus
from model.nanogpt import HyperTransformerLM

def sync_optimizer(optimizer, model):
    existing = set()
    for group in optimizer.param_groups:
        for p in group['params']:
            existing.add(p)
    new_params = [p for p in model.parameters() if p not in existing and p.requires_grad]
    if new_params:
        optimizer.add_param_group({'params': new_params})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=600, help="Total training steps")
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--d_hyper", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print("================================================================================")
    print("  [FULL-LENGTH ENGINE] MULTI-DOMAIN BPE TRAINING & COHERENT GENERATION")
    print("================================================================================")
    print(f"Device: {device} | Steps: {args.steps} | Embedding Dim: {args.d_model} | Layers: {args.n_layers}")
    print(f"BPE Sub-word Tokenizer: GPT-2 Encoding (Vocab size: 50,304)\n")

    corpus = BPEMultiDomainCorpus(seq_len=args.seq_len, batch_size=args.batch_size)

    # Initialize Hyperspace 2.0 Model
    model = HyperTransformerLM(
        vocab_size=corpus.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        d_hyper=args.d_hyper,
        top_k=2,
        spawn_threshold=0.26,
        max_experts=16,
        max_seq_len=256
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    domains = [0, 1, 2, 3, 4]
    steps_per_domain = 40

    print("--- Beginning Multi-Domain BPE Training Stream ---")
    start_time = time.time()
    
    # Track domain expert affinities
    domain_expert_counts = {d: {} for d in domains}

    for step in range(1, args.steps + 1):
        domain_idx = (step - 1) // steps_per_domain
        d_id = domains[domain_idx % len(domains)]
        
        inputs, targets, d_name = corpus.get_batch(d_id)
        inputs, targets = inputs.to(device), targets.to(device)

        model.train()
        optimizer.zero_grad()
        
        prev_exp = model.blocks[0].hyper_moe.num_experts
        logits, loss, telemetries = model(inputs, targets=targets, allow_spawning=True)
        
        if model.blocks[0].hyper_moe.num_experts != prev_exp:
            sync_optimizer(optimizer, model)
            print(f"[SPAWN @ Step {step:3d}] Domain: {d_name:<16} | Spawned Experts: {model.blocks[0].hyper_moe.num_experts}")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Track expert activations in Layer 0
        if telemetries:
            for exp in telemetries[0]["top_indices"].view(-1).tolist():
                domain_expert_counts[d_id][exp] = domain_expert_counts[d_id].get(exp, 0) + 1

        if step % 50 == 0 or step == 1 or step == args.steps:
            sigma = telemetries[0]["branching_ratio"] if telemetries else 1.0
            temp = telemetries[0]["routing_temperature"] if telemetries else 10.0
            print(f"Step {step:4d}/{args.steps} | Domain: {d_name:<16} | Loss: {loss.item():.4f} | Perplexity: {math.exp(min(12.0, loss.item())):.2f} | Sigma: {sigma:.3f} | Temp: {temp:.1f}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f}s ({args.steps / total_time:.1f} steps/sec)\n")

    # ==================================================================
    # DOMAIN SPECIALIZATION MATRIX
    # ==================================================================
    print("================================================================================")
    print("  [STATS] DOMAIN SPECIALIZATION DISTRIBUTION (Layer 0)")
    print("================================================================================")
    total_exp = model.blocks[0].hyper_moe.num_experts
    header = f"{'Domain':<18} | " + " | ".join(f"Exp {i:2d}" for i in range(total_exp))
    print(header)
    print("-" * len(header))
    for d_id in domains:
        d_name, _ = corpus.domain_map[d_id]
        counts = domain_expert_counts[d_id]
        total_tokens = sum(counts.values()) or 1
        row = f"{d_name:<18} | "
        for i in range(total_exp):
            pct = (counts.get(i, 0) / total_tokens) * 100.0
            row += f"{pct:5.1f}% | "
        print(row)

    # ==================================================================
    # FULL-LENGTH GENERATIONS ACROSS ALL 5 DOMAINS
    # ==================================================================
    print("\n================================================================================")
    print("  [GENERATION] FULL-LENGTH MULTI-PARAGRAPH GENERATIONS ACROSS 5 DOMAINS")
    print("================================================================================")

    test_prompts = [
        ("Domain 0: Python Systems", "def dijkstra_shortest_path(graph: Dict[str, Dict[str, float]], start: str) -> Dict[str, float]:\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0.0\n    pq = MinHeapPriorityQueue()"),
        ("Domain 1: Mathematics", "Spectral Theorem for Hermitian Operators:\nLet H be a complex Hilbert space and let A: H -> H be a bounded self-adjoint linear operator.\nThere exists a unique spectral measure E"),
        ("Domain 2: Philosophy", "The Hard Problem of Consciousness and Phenomenal Qualia:\nThe distinction between the functional mechanisms of cognitive processing and subjective qualitative experience represents the central debate"),
        ("Domain 3: Cloud JSON", '{\n  "system_telemetry": {\n    "cluster_name": "us-west-prod-hyper-01",\n    "architecture": "neuro_symbolic_hyperspace",\n    "status": "HEALTHY",\n    "metrics": {'),
        ("Domain 4: Sci-Fi Narrative", "The starship Prometheus drifted in the silent vacuum beyond the rings of Saturn.\nInside the central nexus, the quantum navigation core pulsed with a deep, rhythmic sapphire glow, calculating hyper-dimensional trajectories")
    ]

    model.eval()
    for domain_title, prompt_str in test_prompts:
        print(f"\n================================================================================")
        print(f"  {domain_title}")
        print(f"================================================================================")
        print(f"[PROMPT]:\n{prompt_str}\n")
        
        prompt_tokens = corpus.encode(prompt_str)
        curr_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        # Generate 120 new BPE tokens (several complete paragraphs or full function blocks!)
        max_gen = 120
        temperature = 0.5
        top_k = 30
        
        generated_segments = []
        for _ in range(max_gen):
            idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
            with torch.no_grad():
                logits, _, telemetries = model(idx_cond, allow_spawning=False)
                
            last_logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(last_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)
            
            top_exp = telemetries[0]["top_indices"][0, -1, 0].item() if telemetries else 0
            tok_text = corpus.decode([next_tok.item()])
            generated_segments.append((tok_text, top_exp))

        # Reconstruct and format full output with expert attribution highlights
        full_text = corpus.decode(curr_ids[0].tolist())
        print(f"[FULL-LENGTH GENERATION RESULT]:\n{full_text}\n")
        
        # Attribution trace summary
        exp_usage = {}
        for _, exp in generated_segments:
            exp_usage[exp] = exp_usage.get(exp, 0) + 1
        summary_str = ", ".join(f"Expert {e}: {c} tokens ({c/max_gen*100:.1f}%)" for e, c in sorted(exp_usage.items()))
        print(f"[EXPERT ATTRIBUTION SUMMARY]: {summary_str}")
        print("-" * 80)

if __name__ == "__main__":
    main()
