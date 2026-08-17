"""
Inspect raw token predictions, top-1/top-5 token accuracy, and greedy decoding on real dataset batches.
"""

import os
import sys
import json
import numpy as np
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

def test_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = "experiments/checkpoints/hyperspace_deep_trained_25m.pt"
    
    enc = tiktoken.get_encoding("gpt2")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    state_dict = state.get("model_state", state)

    vocab_size = config.get("vocab_size", 50304)
    d_model = config.get("d_model", 384)
    n_layers = config.get("n_layers", 6)
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

    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    max_seq_len = clean_sd.get("pos_embeddings.weight", torch.zeros(1088, 1)).shape[0]

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
        spawn_threshold=0.35,
        max_experts=32,
        initial_experts=2,
        use_sparse_attn=True,
        foveal_window=128,
        num_landmarks=4,
        num_sinks=4,
        max_seq_len=max_seq_len,
        dropout=0.0,
    ).to(device)

    for l_idx, block in enumerate(model.blocks):
        needed = per_layer_experts[l_idx] - block.hyper_moe.num_experts
        for exp_i in range(needed):
            dummy_key = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper), device=device)
            block.hyper_moe._spawn_expert(dummy_key, label=f"restored_L{l_idx}_E{exp_i}")

    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    # Load real validation data
    meta_file = "data/real_blend_cache/metadata_real_blend.json"
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for domain in ["github_code", "fineweb_edu", "freelaw_legal", "gutenberg_literature"]:
        shards = np.memmap(meta["domains"][domain]["val_file"], dtype=np.uint16, mode='r')
        # Take a slice of 128 tokens
        raw_tokens = shards[1000:1128].astype(np.int64)
        x = torch.from_numpy(raw_tokens[:-1]).unsqueeze(0).to(device)
        y = torch.from_numpy(raw_tokens[1:]).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                logits, loss, _ = model(x, targets=y, allow_spawning=False)
        
        preds = torch.argmax(logits, dim=-1)
        top5 = torch.topk(logits, k=5, dim=-1).indices
        
        correct_top1 = (preds == y).float().mean().item()
        correct_top5 = torch.any(top5 == y.unsqueeze(-1), dim=-1).float().mean().item()
        
        ground_truth_text = enc.decode(raw_tokens)
        prompt_text = enc.decode(raw_tokens[:30])
        
        # Test greedy generation from prompt
        curr = x[:, :30]
        gen = []
        for _ in range(60):
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    l_out, _, _ = model(curr, allow_spawning=False)
            nxt = torch.argmax(l_out[0, -1, :vocab_size]).item()
            gen.append(nxt)
            curr = torch.cat([curr, torch.tensor([[nxt]], device=device)], dim=1)
        
        gen_greedy = enc.decode(gen)

        print(f"\n==================== Domain: {domain} ====================")
        print(f"Validation Loss: {loss.item():.4f} | Top-1 Acc: {correct_top1*100:.2f}% | Top-5 Acc: {correct_top5*100:.2f}%")
        print(f"\n[Ground Truth Excerpt]:\n{ground_truth_text[:200]}...")
        print(f"\n[Prompt (30 tokens)]:\n{prompt_text}")
        print(f"\n[Greedy Generated Continuation (60 tokens)]:\n{gen_greedy}")

if __name__ == "__main__":
    test_predictions()
