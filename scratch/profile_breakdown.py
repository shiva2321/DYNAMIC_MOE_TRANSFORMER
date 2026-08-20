import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.hyper_moe import DynamicHyperMoE

def profile_layer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 90)
    print("  [COMPONENT-LEVEL TIME PROFILER FOR DYNAMICHYPERMOE (32 EXPERTS)]")
    print("=" * 90)

    d_model = 384
    d_ff = 768
    d_hyper = 2048
    b, s = 2, 256
    num_tokens = b * s

    moe = DynamicHyperMoE(
        d_model=d_model,
        d_ff=d_ff,
        d_hyper=d_hyper,
        top_k=2,
        max_k=4,
        top_p=0.85,
        dynamic_k=True,
        spawn_threshold=1.0,
        max_experts=32,
        initial_experts=32,
        use_bus=True
    ).to(device)

    x = torch.randn(b, s, d_model, device=device, requires_grad=True)

    # Warmup
    for _ in range(5):
        out, telem = moe(x, allow_spawning=False)
        loss = out.sum() + telem["ortho_loss"] + telem["load_balance_loss"]
        loss.backward()
        moe.zero_grad()
    torch.cuda.synchronize()

    # Now instrument sub-steps with CUDA events
    starts = {k: torch.cuda.Event(enable_timing=True) for k in ["total", "proj", "res", "topk", "ortho", "pass1", "bus", "pass2", "backward"]}
    ends = {k: torch.cuda.Event(enable_timing=True) for k in starts.keys()}

    steps = 50
    times = {k: 0.0 for k in starts.keys()}

    for _ in range(steps):
        torch.cuda.synchronize()
        starts["total"].record()

        flat_x = x.view(-1, d_model)
        
        # 1. Proj
        starts["proj"].record()
        norm_x = F.normalize(flat_x, p=2, dim=-1)
        r_part = moe.proj_r(norm_x).float()
        i_part = moe.proj_i(norm_x).float()
        z = torch.complex(r_part, i_part)
        token_phasors = z / (torch.abs(z) + 1e-8)
        ends["proj"].record()

        # 2. Resonance
        starts["res"].record()
        resonance = moe.memory.compute_resonance(token_phasors)
        ends["res"].record()

        # 3. Top-k & dynamic k
        starts["topk"].record()
        scaled_logits = resonance * 10.0
        router_probs = F.softmax(scaled_logits, dim=-1)
        top_weights, top_indices = torch.topk(router_probs, k=2, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        ends["topk"].record()

        # 4. Ortho loss
        starts["ortho"].record()
        ortho_loss = moe.memory.compute_orthogonality_loss()
        ends["ortho"].record()

        # 5. Pass 1 Basal Dispatch
        starts["pass1"].record()
        k_eval = 2
        expert_slot_basal_outputs = torch.zeros(num_tokens, k_eval, d_model, device=device)
        for k_idx in range(k_eval):
            indices_k = top_indices[:, k_idx]
            weights_k = top_weights[:, k_idx]
            active_token_mask = weights_k > 1e-5
            active_exp_ids = torch.unique(indices_k[active_token_mask])
            for exp_id_tensor in active_exp_ids:
                exp_id = exp_id_tensor.item()
                expert = moe.experts[exp_id]
                token_mask = active_token_mask & (indices_k == exp_id)
                selected_tokens = flat_x[token_mask]
                exp_out = expert(x_basal=selected_tokens, c_apical=None)
                expert_slot_basal_outputs[token_mask, k_idx] = exp_out
        ends["pass1"].record()

        # 6. Global Bus
        starts["bus"].record()
        selected_keys = moe.memory.expert_keys[top_indices]
        bus_context = moe.bus.broadcast_and_listen(
            expert_outputs=expert_slot_basal_outputs,
            expert_keys=selected_keys,
            expert_weights=top_weights.unsqueeze(-1)
        )
        ends["bus"].record()

        # 7. Pass 2 Apical Dispatch
        starts["pass2"].record()
        expert_slot_final_outputs = torch.zeros(num_tokens, k_eval, d_model, device=device)
        for k_idx in range(k_eval):
            indices_k = top_indices[:, k_idx]
            weights_k = top_weights[:, k_idx]
            active_token_mask = weights_k > 1e-5
            active_exp_ids = torch.unique(indices_k[active_token_mask])
            for exp_id_tensor in active_exp_ids:
                exp_id = exp_id_tensor.item()
                expert = moe.experts[exp_id]
                token_mask = active_token_mask & (indices_k == exp_id)
                tokens_b = flat_x[token_mask]
                context_a = bus_context[token_mask, k_idx]
                exp_out = expert(x_basal=tokens_b, c_apical=context_a)
                expert_slot_final_outputs[token_mask, k_idx] = exp_out
        ends["pass2"].record()

        # 8. Backward pass
        out = (expert_slot_final_outputs * top_weights.unsqueeze(-1)).sum(dim=1)
        loss = out.sum() + ortho_loss
        starts["backward"].record()
        loss.backward()
        moe.zero_grad()
        ends["backward"].record()

        ends["total"].record()
        torch.cuda.synchronize()

        for k in starts.keys():
            times[k] += starts[k].elapsed_time(ends[k])

    print(f"{'Component':<35} | {'Mean Time (ms)':<15} | {'% of Forward+Backward'}")
    print("-" * 75)
    total_ms = times["total"] / steps
    for k in ["proj", "res", "topk", "ortho", "pass1", "bus", "pass2", "backward"]:
        comp_ms = times[k] / steps
        pct = (comp_ms / total_ms) * 100.0
        print(f"{k:<35} | {comp_ms:8.2f} ms     | {pct:5.1f}%")
    print("-" * 75)
    print(f"{'Total Layer Forward+Backward':<35} | {total_ms:8.2f} ms     | 100.0%")

if __name__ == "__main__":
    profile_layer()
