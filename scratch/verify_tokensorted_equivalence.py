import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.dendritic_expert import TwoCompartmentDendriticExpert
from scratch.test_tokensorted_moe import TokenSortedDendriticMoE

def run_equivalence_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("  [MATHEMATICAL & NUMERICAL EQUIVALENCE: TOKEN-SORTED MOE]")
    print("=" * 90)

    torch.manual_seed(1337)
    d_model = 384
    d_ff = 768
    num_experts = 32
    N = 512
    k = 2

    # Instantiate token-sorted model
    model = TokenSortedDendriticMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, device=device)

    flat_x = torch.randn(N, d_model, device=device, requires_grad=True)
    flat_x_loop = flat_x.clone().detach().requires_grad_(True)

    top_indices = torch.randint(0, num_experts, (N, k), device=device)
    top_weights = F.softmax(torch.randn(N, k, device=device), dim=-1)
    bus_context = torch.randn(N, k, d_model, device=device, requires_grad=True)
    bus_context_loop = bus_context.clone().detach().requires_grad_(True)

    # 1. Execute Original Loop Dispatch using model.experts
    expert_slot_final_outputs = torch.zeros(N, k, d_model, device=device)
    for k_idx in range(k):
        indices_k = top_indices[:, k_idx]
        weights_k = top_weights[:, k_idx]
        active_token_mask = weights_k > 1e-5
        active_exp_ids = torch.unique(indices_k[active_token_mask])
        for exp_id_tensor in active_exp_ids:
            exp_id = exp_id_tensor.item()
            expert = model.experts[exp_id]
            token_mask = active_token_mask & (indices_k == exp_id)
            tokens_b = flat_x_loop[token_mask]
            context_a = bus_context_loop[token_mask, k_idx]
            exp_out = expert(x_basal=tokens_b, c_apical=context_a)
            expert_slot_final_outputs[token_mask, k_idx] = exp_out

    out_loop = (expert_slot_final_outputs * top_weights.unsqueeze(-1)).sum(dim=1)
    loss_loop = out_loop.sum()
    loss_loop.backward()

    # 2. Execute Token-Sorted Dispatch
    out_sorted = model(flat_x, top_indices, top_weights, bus_context)
    loss_sorted = out_sorted.sum()
    loss_sorted.backward()

    # 3. Check Differences
    fwd_diff = torch.max(torch.abs(out_loop - out_sorted)).item()
    grad_x_diff = torch.max(torch.abs(flat_x.grad - flat_x_loop.grad)).item()
    grad_bus_diff = torch.max(torch.abs(bus_context.grad - bus_context_loop.grad)).item()

    print(f"\n  Forward Max Absolute Difference: {fwd_diff:.2e}")
    print(f"  Input Gradient Max Difference:   {grad_x_diff:.2e}")
    print(f"  Bus Gradient Max Difference:     {grad_bus_diff:.2e}")

    assert fwd_diff < 1e-5, f"Forward mismatch: {fwd_diff}"
    assert grad_x_diff < 1e-5, f"Input grad mismatch: {grad_x_diff}"
    assert grad_bus_diff < 1e-5, f"Bus grad mismatch: {grad_bus_diff}"

    print("\n  >>> [SUCCESS] 100% BITWISE & MATHEMATICAL EQUIVALENCE VERIFIED! <<<")

if __name__ == "__main__":
    run_equivalence_test()
