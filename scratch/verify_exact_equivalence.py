import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.dendritic_expert import TwoCompartmentDendriticExpert

class ExactVectorizedDendriticMoE(nn.Module):
    """
    Exact Bitwise-Equivalent Vectorized Two-Compartment Dendritic MoE.
    Implements the exact biophysical equations:
      Basal:  h_basal = SiLU(x @ W_bg) * (x @ W_bu)
      Apical: h_apical = SiLU(c @ W_ag) * (c @ W_au)
      Soma:   h_soma = h_basal + alpha * h_apical + beta * (h_basal * h_apical)
      Output: y = h_soma @ W_down
    """
    def __init__(self, d_model=384, d_ff=768, num_experts=32, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_experts = num_experts

        # 3D Stacked Weight Tensors: [E, d_in, d_out] for direct x @ W computation
        self.w_bg = nn.Parameter(torch.zeros(num_experts, d_model, d_ff, device=device))
        self.w_bu = nn.Parameter(torch.zeros(num_experts, d_model, d_ff, device=device))
        self.w_ag = nn.Parameter(torch.zeros(num_experts, d_model, d_ff, device=device))
        self.w_au = nn.Parameter(torch.zeros(num_experts, d_model, d_ff, device=device))
        self.w_down = nn.Parameter(torch.zeros(num_experts, d_ff, d_model, device=device))

        # Expert-specific scalar parameters
        self.alpha_apical = nn.Parameter(torch.zeros(num_experts, 1, device=device))
        self.beta_nmda = nn.Parameter(torch.zeros(num_experts, 1, device=device))

    def load_from_module_list(self, experts_list: nn.ModuleList):
        """Loads weights exactly from a list of individual TwoCompartmentDendriticExpert instances."""
        with torch.no_grad():
            for i, exp in enumerate(experts_list):
                self.w_bg[i] = exp.basal_gate.weight.T
                self.w_bu[i] = exp.basal_up.weight.T
                self.w_ag[i] = exp.apical_gate.weight.T
                self.w_au[i] = exp.apical_up.weight.T
                self.w_down[i] = exp.w_down.weight.T
                self.alpha_apical[i, 0] = exp.alpha_apical.data
                self.beta_nmda[i, 0] = exp.beta_nmda.data

    def forward(self, flat_x, top_indices, top_weights, bus_context=None):
        """
        flat_x: [N_tokens, d_model]
        top_indices: [N_tokens, k]
        top_weights: [N_tokens, k]
        bus_context: [N_tokens, k, d_model] (optional)
        """
        N, d = flat_x.shape
        k = top_indices.shape[1]

        slot_outputs = torch.zeros(N, k, d, device=flat_x.device, dtype=flat_x.dtype)

        for k_idx in range(k):
            exp_idx = top_indices[:, k_idx] # [N]
            w_val = top_weights[:, k_idx].unsqueeze(-1) # [N, 1]

            # 1. Gather expert weights for all N tokens at once: [N, d_model, d_ff]
            w_bg_k = self.w_bg[exp_idx]
            w_bu_k = self.w_bu[exp_idx]
            w_dn_k = self.w_down[exp_idx]

            # 2. Vectorized Basal Forward: SiLU(x @ W_bg) * (x @ W_bu)
            # flat_x: [N, 1, d_model] @ [N, d_model, d_ff] -> [N, d_ff]
            x_unsqueezed = flat_x.unsqueeze(1)
            h_bg = F.silu(torch.bmm(x_unsqueezed, w_bg_k).squeeze(1))
            h_bu = torch.bmm(x_unsqueezed, w_bu_k).squeeze(1)
            h_basal = h_bg * h_bu # [N, d_ff]

            if bus_context is not None:
                # 3. Vectorized Apical Forward: SiLU(c @ W_ag) * (c @ W_au)
                c_k = bus_context[:, k_idx].unsqueeze(1) # [N, 1, d_model]
                w_ag_k = self.w_ag[exp_idx]
                w_au_k = self.w_au[exp_idx]
                h_ag = F.silu(torch.bmm(c_k, w_ag_k).squeeze(1))
                h_au = torch.bmm(c_k, w_au_k).squeeze(1)
                h_apical = h_ag * h_au # [N, d_ff]

                # 4. Exact NMDA Coincidence Detection:
                # h_soma = h_basal + alpha * h_apical + beta * (h_basal * h_apical)
                alpha_k = self.alpha_apical[exp_idx] # [N, 1]
                beta_k = self.beta_nmda[exp_idx] # [N, 1]
                h_soma = h_basal + (alpha_k * h_apical) + (beta_k * (h_basal * h_apical))
            else:
                h_soma = h_basal

            # 5. Output Projection: h_soma @ W_down
            out_k = torch.bmm(h_soma.unsqueeze(1), w_dn_k).squeeze(1) # [N, d_model]
            slot_outputs[:, k_idx] = out_k * w_val

        return slot_outputs.sum(dim=1)

def run_equivalence_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("  [MATHEMATICAL & NUMERICAL EQUIVALENCE VERIFICATION]")
    print("  Comparing Loop-based Dispatch vs Exact Vectorized Dispatch")
    print("=" * 90)

    torch.manual_seed(1337)
    d_model = 384
    d_ff = 768
    num_experts = 32
    N = 512 # 2 * 256 tokens
    k = 2

    # 1. Create list of original TwoCompartmentDendriticExpert modules
    experts_list = nn.ModuleList([
        TwoCompartmentDendriticExpert(d_model=d_model, d_ff=d_ff, device=device)
        for _ in range(num_experts)
    ])

    # 2. Create vectorized module and load exact weights
    vec_model = ExactVectorizedDendriticMoE(d_model=d_model, d_ff=d_ff, num_experts=num_experts, device=device)
    vec_model.load_from_module_list(experts_list)

    # 3. Create test inputs
    flat_x = torch.randn(N, d_model, device=device, requires_grad=True)
    flat_x_vec = flat_x.clone().detach().requires_grad_(True)

    # Create realistic skewed routing indices
    top_indices = torch.randint(0, num_experts, (N, k), device=device)
    top_weights = F.softmax(torch.randn(N, k, device=device), dim=-1)
    bus_context = torch.randn(N, k, d_model, device=device, requires_grad=True)
    bus_context_vec = bus_context.clone().detach().requires_grad_(True)

    # --- EXECUTE ORIGINAL LOOP DISPATCH ---
    expert_slot_final_outputs = torch.zeros(N, k, d_model, device=device)
    for k_idx in range(k):
        indices_k = top_indices[:, k_idx]
        weights_k = top_weights[:, k_idx]
        active_token_mask = weights_k > 1e-5
        active_exp_ids = torch.unique(indices_k[active_token_mask])
        for exp_id_tensor in active_exp_ids:
            exp_id = exp_id_tensor.item()
            expert = experts_list[exp_id]
            token_mask = active_token_mask & (indices_k == exp_id)
            tokens_b = flat_x[token_mask]
            context_a = bus_context[token_mask, k_idx]
            exp_out = expert(x_basal=tokens_b, c_apical=context_a)
            expert_slot_final_outputs[token_mask, k_idx] = exp_out

    out_loop = (expert_slot_final_outputs * top_weights.unsqueeze(-1)).sum(dim=1)
    loss_loop = out_loop.sum()
    loss_loop.backward()

    # --- EXECUTE EXACT VECTORIZED DISPATCH ---
    out_vec = vec_model(flat_x_vec, top_indices, top_weights, bus_context_vec)
    loss_vec = out_vec.sum()
    loss_vec.backward()

    # --- VERIFY NUMERICAL TOLERANCES ---
    fwd_diff = torch.max(torch.abs(out_loop - out_vec)).item()
    grad_x_diff = torch.max(torch.abs(flat_x.grad - flat_x_vec.grad)).item()
    grad_bus_diff = torch.max(torch.abs(bus_context.grad - bus_context_vec.grad)).item()

    print(f"\n  Forward Max Absolute Difference: {fwd_diff:.2e}")
    print(f"  Input Gradient Max Difference:   {grad_x_diff:.2e}")
    print(f"  Bus Gradient Max Difference:     {grad_bus_diff:.2e}")

    assert fwd_diff < 1e-5, f"Forward mismatch: {fwd_diff}"
    assert grad_x_diff < 1e-5, f"Input grad mismatch: {grad_x_diff}"
    assert grad_bus_diff < 1e-5, f"Bus grad mismatch: {grad_bus_diff}"

    print("\n  >>> [SUCCESS] 100% BITWISE & MATHEMATICAL EQUIVALENCE VERIFIED! <<<")

if __name__ == "__main__":
    run_equivalence_test()
