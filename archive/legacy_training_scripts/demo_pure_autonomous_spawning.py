"""
Pure Autonomous Spawning Demonstration.
Shows that the neural network spawns new experts 100% on its own inside model.forward(),
without ANY human intervention or manual spawn function calls.
"""

import os
import sys
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.nanogpt import HyperTransformerLM
from data.dataset_hub import MultiDomainDatasetHub

def demonstrate_pure_autonomous_spawning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hub = MultiDomainDatasetHub(seq_len=64, batch_size=2, use_bpe=True)

    print("=" * 85)
    print("  [DEMO] PURE AUTONOMOUS EXPERT SPAWNING INSIDE PYTORCH FORWARD PASS")
    print("=" * 85)

    # 1. Initialize a model with 2 initial bootstrap experts and sensitive novelty threshold
    model = HyperTransformerLM(
        vocab_size=hub.vocab_size,
        d_model=256,
        n_layers=3,
        n_heads=4,
        d_ff=512,
        d_hyper=2048,
        top_k=2,
        spawn_threshold=0.85, # Triggers when query resonance with existing keys is not high
        max_experts=16,
    ).to(device)
    model.train()

    print(f"\n[STEP 1: INITIAL STATE BEFORE ANY FORWARD PASS]")
    print(f"  Layer 0 Experts Count: {model.blocks[0].hyper_moe.num_experts}")
    print(f"  Layer 1 Experts Count: {model.blocks[1].hyper_moe.num_experts}")
    print(f"  Layer 2 Experts Count: {model.blocks[2].hyper_moe.num_experts}")
    print("  Notice: Exactly 2 experts exist per layer.")

    # 2. Prepare random novel domain input tokens
    novel_text = "IN WITNESS WHEREOF the parties execute this Delaware corporate merger agreement."
    tokens = hub.encode(novel_text)
    input_tensor = torch.tensor([tokens[:64] + [0] * max(0, 64 - len(tokens))], dtype=torch.long, device=device)

    print(f"\n[STEP 2: RUNNING STANDARD model(input_tensor)]")
    print("  Executing: logits, loss, telemetries = model(input_tensor)")
    print("  (Zero manual spawn calls. Zero human intervention. Pure model.forward())")

    # Standard PyTorch forward pass - the model makes all decisions internally
    logits, loss, telemetries = model(input_tensor, allow_spawning=True)

    print(f"\n[STEP 3: INSPECTING MODEL AFTER A SINGLE FORWARD PASS]")
    print(f"  Layer 0 Experts Count: {model.blocks[0].hyper_moe.num_experts}")
    print(f"  Layer 1 Experts Count: {model.blocks[1].hyper_moe.num_experts}")
    print(f"  Layer 2 Experts Count: {model.blocks[2].hyper_moe.num_experts}")

    if model.blocks[0].hyper_moe.num_experts > 2:
        print("\n  >>> RESULT: The neural network autonomously detected novelty in C^D,")
        print("      instantiated a new TwoCompartmentDendriticExpert on the GPU,")
        print("      and registered its complex address key completely on its own inside forward()!")
    print("=" * 85)

if __name__ == "__main__":
    demonstrate_pure_autonomous_spawning()
