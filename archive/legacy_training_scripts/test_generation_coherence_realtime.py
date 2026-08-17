"""
Real-time Generation Coherence & Sensibility Inspector on the Redesigned Dynamic Engine.
Evaluates actual generated text across diverse sampling strategies (Greedy, Low-Temp, Top-p).
"""

import os
import sys
import torch
import torch.nn.functional as F
import tiktoken

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model.nanogpt import HyperTransformerLM
from hyperspace.vsa import ComplexPhasorVSA

def inspect_generation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "experiments/checkpoints/hyperspace_dynamic_cloned_engine.pt"

    enc = tiktoken.get_encoding("gpt2")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    state_dict = checkpoint.get("model_state", checkpoint)

    vocab_size = config.get("vocab_size", 50304)
    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 4)
    n_heads = config.get("n_heads", 6)
    d_ff = config.get("d_ff", 768)
    d_hyper = config.get("d_hyper", 2048)

    per_layer_experts = []
    for l in range(n_layers):
        exp_keys = set()
        for k in state_dict.keys():
            if f"blocks.{l}.hyper_moe.experts." in k:
                exp_keys.add(int(k.split(".")[4]))
        per_layer_experts.append(max(2, len(exp_keys)))

    print("\n" + "=" * 90)
    print("  [GENERATION COHERENCE & SENSIBILITY INSPECTION]")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Architecture: {n_layers} Layers, Per-Layer Experts: {per_layer_experts} (Total: {sum(per_layer_experts)})")
    print("=" * 90)

    model = HyperTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=0.28,
        max_experts=16,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=576,
        dropout=0.0,
    ).to(device)

    # Spawn to match checkpoint
    for l_idx, block in enumerate(model.blocks):
        needed = per_layer_experts[l_idx] - block.hyper_moe.num_experts
        for exp_i in range(needed):
            dummy_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(dummy_key, label=f"restored_L{l_idx}_E{exp_i}")

    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    prompts = [
        {
            "domain": "Python Code & Algorithms",
            "prompt": "def fibonacci_sequence(n):\n    \"\"\"Generates Fibonacci numbers.\"\"\"\n",
            "max_tokens": 50,
            "strategies": ["Greedy (Argmax)", "Temperature 0.6", "Nucleus Top-p (0.85)"]
        },
        {
            "domain": "Corporate Law & Contracts",
            "prompt": "Under Section 2.01 of this Agreement, the Buyer agrees to purchase and the Seller agrees to",
            "max_tokens": 50,
            "strategies": ["Greedy (Argmax)", "Temperature 0.6", "Nucleus Top-p (0.85)"]
        },
        {
            "domain": "Formal Mathematics & Matrices",
            "prompt": "Let A be an n-by-n symmetric matrix. Then the eigenvalues of A are",
            "max_tokens": 50,
            "strategies": ["Greedy (Argmax)", "Temperature 0.6", "Nucleus Top-p (0.85)"]
        }
    ]

    for p_info in prompts:
        prompt_str = p_info["prompt"]
        p_toks = enc.encode(prompt_str)
        
        print("\n" + "#" * 80)
        print(f"  DOMAIN: {p_info['domain']}")
        print(f"  PROMPT: {prompt_str.strip()}")
        print("#" * 80)

        # 1. Greedy Decoding
        curr = torch.tensor([p_toks], dtype=torch.long, device=device)
        gen_greedy = []
        for _ in range(p_info["max_tokens"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    l_out, _, _ = model(curr, allow_spawning=False)
            nxt = torch.argmax(l_out[0, -1, :vocab_size]).item()
            gen_greedy.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
        print(f"\n[1. Greedy (Argmax) Output]:\n{prompt_str}{enc.decode(gen_greedy)}")

        # 2. Temperature 0.6 Sampling
        curr = torch.tensor([p_toks], dtype=torch.long, device=device)
        gen_temp = []
        for _ in range(p_info["max_tokens"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    l_out, _, _ = model(curr, allow_spawning=False)
            scaled_logits = l_out[0, -1, :vocab_size] / 0.6
            v, top_idx = torch.topk(scaled_logits, 40)
            probs = F.softmax(v, dim=-1)
            nxt = top_idx[torch.multinomial(probs, 1)].item()
            gen_temp.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
        print(f"\n[2. Temperature 0.6 Output]:\n{prompt_str}{enc.decode(gen_temp)}")

        # 3. Nucleus Top-p (0.85) Sampling
        curr = torch.tensor([p_toks], dtype=torch.long, device=device)
        gen_topp = []
        for _ in range(p_info["max_tokens"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    l_out, _, _ = model(curr, allow_spawning=False)
            logits = l_out[0, -1, :vocab_size]
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > 0.85
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')
            probs = F.softmax(logits / 0.7, dim=-1)
            nxt = torch.multinomial(probs, 1).item()
            gen_topp.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
        print(f"\n[3. Nucleus Top-p (0.85) Output]:\n{prompt_str}{enc.decode(gen_topp)}")

if __name__ == "__main__":
    inspect_generation()
