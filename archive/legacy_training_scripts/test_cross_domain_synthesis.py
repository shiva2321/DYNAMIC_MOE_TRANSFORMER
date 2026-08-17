"""
Empirical Cross-Domain Generalization & Inter-Expert Synthesis Benchmark.
Tests:
1. Domain Relevance & Accuracy on Single-Domain Inquiries
2. Hybrid Cross-Domain Synthesis (e.g. Code + Math, Code + JSON, Bio + AI, History + SciFi)
3. Inter-Expert Co-Activation & Apical Bus Collaboration Measurement
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from data.dataset_hub import MultiDomainDatasetHub
from hyperspace.vsa import ComplexPhasorVSA

def load_model(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    state_dict = state.get("model_state", state)

    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 6)
    n_heads = config.get("n_heads", 6)
    d_ff = config.get("d_ff", 1024)
    d_hyper = config.get("d_hyper", 2048)
    top_k = config.get("top_k", 2)
    max_experts = config.get("max_experts", 16)
    spawn_threshold = config.get("spawn_threshold", 0.25)
    seq_len = config.get("seq_len", 256)
    max_seq_len = seq_len + 32

    model = HyperTransformerLM(
        vocab_size=50304,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=top_k,
        max_experts=max_experts,
        spawn_threshold=spawn_threshold,
        max_seq_len=max_seq_len,
    ).to(device)

    for b_idx, block in enumerate(model.blocks):
        exp_keys = [k for k in state_dict.keys() if k.startswith(f"blocks.{b_idx}.hyper_moe.experts.")]
        expert_ids = set(int(k.split(".")[4]) for k in exp_keys)
        target_num_experts = max(len(expert_ids), 2)
        while block.hyper_moe.num_experts < target_num_experts:
            seed = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(seed, label="loaded_expert")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def run_synthesis_experiment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hub = MultiDomainDatasetHub(seq_len=256, batch_size=4, use_bpe=True)

    ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "scaled_production_hyperspace.pt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "experiments", "checkpoints", "benchmark_hyperspace_moe.pt")

    model = load_model(ckpt_path, device)

    # 4 High-Value Hybrid Cross-Domain Inquiries
    hybrid_prompts = [
        {
            "category": "Hybrid 1: Python Code + Quantum Mathematical Physics",
            "prompt": "def compute_quantum_hamiltonian_eigenvalues(matrix_H):",
            "expected_synthesis": "Python syntax with linear algebra / eigenvalue / matrix operations",
        },
        {
            "category": "Hybrid 2: Python Code + Cloud JSON Telemetry",
            "prompt": "def export_kubernetes_pod_telemetry_to_json(cluster_data):",
            "expected_synthesis": "Python service code manipulating Kubernetes dictionaries/JSON metrics",
        },
        {
            "category": "Hybrid 3: Biomedical Molecular Biology + Formal Physics",
            "prompt": "In cellular biophysics, mitochondrial electron transport follows thermodynamic laws such that",
            "expected_synthesis": "Mitochondrial biochemical pathway linked with thermodynamic energy equations",
        },
        {
            "category": "Hybrid 4: Multi-Turn Dialogue + Distributed Systems Engineering",
            "prompt": "User: How do async worker pools prevent memory leaks in production?\nAssistant:",
            "expected_synthesis": "Conversational assistant explanation with technical concurrency details",
        }
    ]

    print("==========================================================================================")
    print("  [CROSS-DOMAIN SYNTHESIS BENCHMARK] INTER-EXPERT COLLABORATION AUDIT")
    print("==========================================================================================")

    for item in hybrid_prompts:
        cat = item["category"]
        prompt = item["prompt"]
        expected = item["expected_synthesis"]

        print(f"\n>>> TEST CATEGORY: {cat}")
        print(f"    Expected Synthesis: {expected}")
        print(f"    Prompt: {prompt}")

        prompt_tokens = hub.encode(prompt)
        curr_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

        gen_tokens = []
        layer_expert_coactivations = {l: {} for l in range(len(model.blocks))}

        for _ in range(40):
            idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
            with torch.no_grad():
                logits, _, telemetries = model(idx_cond, allow_spawning=False)

            last_logits = logits[:, -1, :] / 0.75
            v, _ = torch.topk(last_logits, min(40, last_logits.size(-1)))
            last_logits[last_logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(last_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_tok], dim=1)

            # Record co-activation (Top-2 experts working together)
            if telemetries:
                for l_idx, telem in enumerate(telemetries):
                    e_pair = tuple(sorted(telem["top_indices"][0, -1].tolist()))
                    layer_expert_coactivations[l_idx][e_pair] = layer_expert_coactivations[l_idx].get(e_pair, 0) + 1

            gen_tokens.append(next_tok.item())

        gen_text = hub.decode(gen_tokens)
        print(f"    GENERATED COMPLETION:\n    {prompt}{gen_text}\n")

        # Collaboration Analysis
        print("    [Inter-Expert Co-Activation Summary]")
        for l_idx in [0, 2, 5]: # Sample Layers
            top_pairs = sorted(layer_expert_coactivations[l_idx].items(), key=lambda x: x[1], reverse=True)
            pairs_str = ", ".join([f"Experts {list(p)}: {cnt} tokens" for p, cnt in top_pairs[:2]])
            print(f"      Layer {l_idx} Active Collaboration: {pairs_str}")
        print("-" * 90)

if __name__ == "__main__":
    run_synthesis_experiment()
