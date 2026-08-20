"""
Load the final, fully-trained (10-domain, seed 1337) spawning checkpoint and
generate real completions for a handful of prompts spanning different trained
domains. No cherry-picking: every prompt below is run once, output printed
raw, first attempt.
"""
import os
import sys
import torch
import tiktoken

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.nanogpt import HyperTransformerLM


def load_dynamic_hyper_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    layer_experts = {}
    for k in sd.keys():
        if "hyper_moe.experts." in k:
            parts = k.split(".")
            layer_idx = int(parts[1])
            exp_idx = int(parts[4])
            layer_experts[layer_idx] = max(layer_experts.get(layer_idx, 0), exp_idx + 1)

    print(f"[CHECKPOINT LOAD] {ckpt_path}")
    print(f"[CHECKPOINT LOAD] Discovered layer expert topology: {layer_experts}")

    model = HyperTransformerLM(
        vocab_size=50304,
        d_model=384,
        n_layers=4,
        n_heads=6,
        d_ff=768,
        d_hyper=2048,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=1.0,
        max_experts=32,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=576,
        dropout=0.0,
        use_bus=True,
    ).to(device)

    for layer_idx, num_exp in layer_experts.items():
        moe = model.blocks[layer_idx].hyper_moe
        current_exp = len(moe.experts)
        for i in range(current_exp, num_exp):
            dummy_key = torch.randn(1, moe.d_hyper, dtype=torch.cfloat, device=device)
            dummy_key = dummy_key / (torch.abs(dummy_key) + 1e-8)
            moe._spawn_expert(dummy_key, label=f"restored_{i}")

    model.load_state_dict(sd)
    model.eval()
    print("[CHECKPOINT LOAD] OK — weights loaded into frozen eval model.\n")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    ckpt_path = "experiments/checkpoint_10domain_hyperspace_budgeted_spawn.pt"
    model = load_dynamic_hyper_model(ckpt_path, device)

    enc = tiktoken.get_encoding("gpt2")

    prompts = [
        ("github_code (systems/algo code)", "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"),
        ("python_code", "import numpy as np\n\ndef normalize(vector):\n"),
        ("gutenberg_literature", "It was the best of times, and yet"),
        ("wikitext_facts", "The history of the Roman Empire begins with"),
        ("freelaw_legal", "This Agreement is entered into by and between"),
        ("openweb_math", "Theorem. Let n be a positive integer. Then"),
        ("financial_market", "The Federal Reserve announced today that interest rates"),
        ("out-of-distribution (never trained on)", "The recipe for a traditional Sunday roast begins with"),
    ]

    torch.manual_seed(1337)

    print("=" * 100)
    for label, prompt in prompts:
        ids = torch.tensor([enc.encode(prompt)], dtype=torch.long, device=device)
        out_ids = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=40)
        text = enc.decode(out_ids[0].tolist())
        print(f"[{label}]")
        print(f"PROMPT:     {prompt!r}")
        print(f"COMPLETION: {text!r}")
        print("-" * 100)
    print("=" * 100)


if __name__ == "__main__":
    main()
