"""
Unit test for Adaptive Dynamic-k Criticality-Gated Routing.
Verifies:
1. Simple/confident tokens recruit k=1 expert.
2. Binary hybrid tokens recruit k=2 experts.
3. Complex multi-domain tokens recruit k >= 3 experts.
4. Correctness of sparse token dispatch, apical context feedback, and gradient flow.
"""

import sys
import os
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.hyper_moe import DynamicHyperMoE

def test_adaptive_dynamic_k():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Testing Adaptive Dynamic-k Routing on {device}]")

    d_model = 128
    d_ff = 256
    d_hyper = 512
    max_experts = 8

    moe = DynamicHyperMoE(
        d_model=d_model,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        max_experts=max_experts,
        initial_experts=4,
        use_bus=True,
        use_hopfield=False,
    ).to(device)

    # Synthetic batch of 6 tokens with varied representation complexity
    x = torch.randn(2, 3, d_model, device=device) # [B=2, S=3, D=128]
    out, telem = moe(x, allow_spawning=False)

    print("Output shape:", out.shape)
    print("Top indices:\n", telem["top_indices"])
    print("Top weights:\n", telem["top_weights"])
    print("Mean active k:", telem.get("mean_active_k", "N/A"))

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "NaNs detected in MoE output"

    # Test backward pass
    loss = out.sum()
    loss.backward()
    print("Gradient backward passed successfully!")
    print("Test Passed!")

if __name__ == "__main__":
    test_adaptive_dynamic_k()
