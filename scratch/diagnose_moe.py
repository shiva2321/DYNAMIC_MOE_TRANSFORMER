import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hyperspace.vsa import ComplexPhasorVSA

class ComplexLinearPhasorProjection(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.proj_r = nn.Linear(d_in, d_out, bias=False)
        self.proj_i = nn.Linear(d_in, d_out, bias=False)
        nn.init.orthogonal_(self.proj_r.weight)
        nn.init.orthogonal_(self.proj_i.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real = self.proj_r(x)
        imag = self.proj_i(x)
        z = torch.complex(real, imag)
        return z / (torch.abs(z) + 1e-8)

def test_spawning_progression():
    d_model = 128
    d_hyper = 1024
    batch_size = 8
    seq_len = 32
    spawn_threshold = 0.20

    projector = ComplexLinearPhasorProjection(d_model, d_hyper)
    
    # Start with 2 bootstrap random keys
    expert_keys = ComplexPhasorVSA.random_hyperspace_vector((2, d_hyper))
    
    torch.manual_seed(42)
    domain_directions = F.normalize(torch.randn(4, d_model), p=2, dim=-1)
    
    for d_idx in range(4):
        domain_center = domain_directions[d_idx]
        noise = torch.randn(batch_size, seq_len, d_model) * 0.1
        batch = F.normalize(domain_center.unsqueeze(0).unsqueeze(0) + noise, p=2, dim=-1)
        
        flat_x = batch.view(-1, d_model)
        query_phasors = projector(flat_x) # [N, D]
        
        # Compute resonance with existing keys
        res = ComplexPhasorVSA.batch_similarity_matrix(query_phasors, expert_keys)
        max_sim = res.max(dim=-1).values.mean().item()
        
        should_spawn = max_sim < spawn_threshold
        if should_spawn:
            # Centroid of domain queries
            new_key = query_phasors.mean(dim=0, keepdim=True)
            new_key = new_key / torch.abs(new_key)
            expert_keys = torch.cat([expert_keys, new_key], dim=0)
            status = f"SPAWNED Expert #{expert_keys.shape[0]-1}"
        else:
            best_exp = res.mean(dim=0).argmax().item()
            status = f"ROUTED to Expert #{best_exp}"
            
        print(f"Domain {d_idx} -> Max Sim: {max_sim:.4f} -> {status} (Total Experts: {expert_keys.shape[0]})")

    # Now re-feed Domain 0 to verify it routes back to its spawned specialist!
    d0_test = F.normalize(domain_directions[0].unsqueeze(0).unsqueeze(0) + torch.randn(batch_size, seq_len, d_model)*0.1, p=2, dim=-1)
    q0 = projector(d0_test.view(-1, d_model))
    res0 = ComplexPhasorVSA.batch_similarity_matrix(q0, expert_keys)
    best_d0 = res0.mean(dim=0).argmax().item()
    max_sim_d0 = res0.mean(dim=0).max().item()
    print(f"\n[Verification] Re-feeding Domain 0 -> Max Sim: {max_sim_d0:.4f} -> Routed to Specialist: Expert #{best_d0} (Correct: Expert #2)")

if __name__ == "__main__":
    test_spawning_progression()
