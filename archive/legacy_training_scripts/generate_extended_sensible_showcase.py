"""
Generate Extended Sensible Text Showcase from the Interleaved Dynamic Engine.
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

def run_extended_showcase():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "experiments/checkpoints/hyperspace_interleaved_dynamic_engine.pt"

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
    print("  [EXTENDED SENSIBLE GENERATION SHOWCASE]")
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
        spawn_threshold=0.30,
        max_experts=16,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=576,
        dropout=0.0,
    ).to(device)

    for l_idx, block in enumerate(model.blocks):
        needed = per_layer_experts[l_idx] - block.hyper_moe.num_experts
        for exp_i in range(needed):
            dummy_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(dummy_key, label=f"restored_L{l_idx}_E{exp_i}")

    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    test_cases = [
        {
            "domain": "Statutory Law & Contracts (FreeLaw)",
            "prompt": "Section 8.01 Indemnification. From and after the Closing, the Seller shall indemnify, defend and hold harmless the Buyer and its",
            "max_tokens": 80,
            "temp": 0.6
        },
        {
            "domain": "Representations and Warranties (FreeLaw)",
            "prompt": "Section 3.05 Financial Statements. The Seller has delivered to the Buyer true and complete copies of the audited balance sheets and the",
            "max_tokens": 80,
            "temp": 0.6
        },
        {
            "domain": "Corporate Governance & Definitions (FreeLaw)",
            "prompt": "Section 1.01 Defined Terms. As used in this Agreement, the following terms have the meanings specified below:\n\"Affiliate\" means,",
            "max_tokens": 80,
            "temp": 0.6
        }
    ]

    for tc in test_cases:
        p_str = tc["prompt"]
        p_toks = enc.encode(p_str)
        curr = torch.tensor([p_toks], dtype=torch.long, device=device)
        gen = []

        print("\n" + "#" * 80)
        print(f"  DOMAIN: {tc['domain']}")
        print(f"  PROMPT: {p_str.strip()}")
        print("#" * 80)

        for _ in range(tc["max_tokens"]):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    l_out, _, telem = model(curr, allow_spawning=False)
            
            scaled_logits = l_out[0, -1, :vocab_size] / tc["temp"]
            v, top_idx = torch.topk(scaled_logits, 40)
            probs = F.softmax(v, dim=-1)
            nxt = top_idx[torch.multinomial(probs, 1)].item()
            gen.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)

        print("\n[ACTUAL MODEL GENERATION]:")
        print(p_str + enc.decode(gen))
        print("-" * 80)

if __name__ == "__main__":
    run_extended_showcase()
