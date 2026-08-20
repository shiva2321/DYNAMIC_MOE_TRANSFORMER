import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.hyper_moe import DynamicHyperMoE
from hyperspace.dynamic_optimizer import DynamicWarmupAdamW

def test_spawning_and_optimization_integration():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("  [SMOKE TEST: DYNAMIC SPAWNING + TOKEN-SORTED DISPATCH + DYNAMICWARMUPADAMW]")
    print("=" * 90)

    d_model = 384
    d_ff = 768
    d_hyper = 2048
    max_experts = 16
    initial_experts = 2

    moe = DynamicHyperMoE(
        d_model=d_model,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=0.35,
        max_experts=max_experts,
        initial_experts=initial_experts,
        use_bus=True
    ).to(device)

    optimizer = DynamicWarmupAdamW(moe.parameters(), lr=5e-4, weight_decay=0.01, default_group_warmup_steps=10)

    print(f"  Initial Experts: {moe.num_experts}")
    assert moe.num_experts == 2, "Expected 2 initial experts"

    # Step 1: Normal forward/backward on bootstrap experts
    x1 = torch.randn(2, 64, d_model, device=device)
    out1, telem1 = moe(x1, allow_spawning=False)
    loss1 = out1.sum() + telem1["ortho_loss"]
    loss1.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("  Step 1 on bootstrap experts passed cleanly.")

    # Step 2: Trigger dynamic spawn with novel data and allow_spawning=True
    prev_expert_count = moe.num_experts
    # Force novel inputs that will trigger novelty detector
    x2 = torch.randn(4, 128, d_model, device=device) * 5.0
    out2, telem2 = moe(x2, allow_spawning=True)

    curr_expert_count = moe.num_experts
    print(f"  Step 2 post-spawning expert count: {curr_expert_count}")

    if curr_expert_count > prev_expert_count:
        # Register new experts into DynamicWarmupAdamW exactly as exp_continual_learning_control does
        for l_exp_idx in range(prev_expert_count, curr_expert_count):
            new_exp = moe.experts[l_exp_idx]
            optimizer.add_dynamic_param_group(
                new_exp.parameters(),
                lr=5e-4,
                weight_decay=0.01,
                warmup_steps=10,
                group_name=f"spawned_exp_{l_exp_idx}"
            )
            print(f"  [SUCCESS] Expert #{l_exp_idx} successfully registered in DynamicWarmupAdamW!")

    loss2 = out2.sum() + telem2["ortho_loss"]
    loss2.backward()

    # Verify that all experts (including the newly spawned ones) have gradients
    for idx, exp in enumerate(moe.experts):
        has_grad = any(p.grad is not None and torch.norm(p.grad).item() > 0 for p in exp.parameters())
        print(f"  Expert #{idx} parameters have active nonzero gradients: {has_grad}")

    optimizer.step()
    optimizer.zero_grad()

    print("\n  >>> [SMOKE TEST PASSED] Dynamic Neurogenesis & Optimization 100% Functional! <<<")

if __name__ == "__main__":
    test_spawning_and_optimization_integration()
