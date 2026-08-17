"""
Dynamic Hyperspace 2.0 Mixture-of-Experts (Hyper-MoE) Layer.
Integrates:
1. Dentate Gyrus Hyper-Sparse Pattern Separation (WTA orthogonalization)
2. Toroidal Grid Coordinate Map
3. Two-Compartment Dendritic Micro-Experts (Apical Bus + Basal Feedforward with NMDA spikes)
4. Self-Organized Criticality Controller (Edge of Chaos branching ratio sigma = 1.0)
5. Modern Hopfield Dense Associative Memory clean-up (2^{D/2} capacity)
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from hyperspace.vsa import ComplexPhasorVSA
from hyperspace.memory import SemanticHyperspaceMemory
from hyperspace.bus import HyperspaceGlobalBus
from hyperspace.dentate_grid import DentateGyrusPatternSeparator, ToroidalGridEncoder
from hyperspace.hopfield import ModernHopfieldMemory
from hyperspace.criticality import SelfOrganizedCriticalityController
from model.dendritic_expert import TwoCompartmentDendriticExpert

class DynamicHyperMoE(nn.Module):
    """
    Hyperspace 2.0 Neuro-Cognitive Mixture-of-Experts Layer.
    """
    def __init__(
        self,
        d_model: int = 256,
        d_ff: int = 512,
        d_hyper: int = 2048,
        top_k: int = 2,
        max_k: int = 4,
        top_p: float = 0.85,
        dynamic_k: bool = False,
        spawn_threshold: float = 0.30,
        max_experts: int = 32,
        initial_experts: int = 2,
        use_bus: bool = True,
        use_dentate: bool = True,
        use_hopfield: bool = True,
        use_criticality: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_hyper = d_hyper
        self.top_k = top_k
        self.max_k = max_k
        self.top_p = top_p
        self.dynamic_k = dynamic_k
        self.use_bus = use_bus
        self.use_dentate = use_dentate
        self.use_hopfield = use_hopfield
        self.use_criticality = use_criticality
        self.max_experts = max_experts
        
        # 1. Complex Linear Phasor Projection Layers (C^{D x d_model})
        self.proj_r = nn.Linear(d_model, d_hyper, bias=False)
        self.proj_i = nn.Linear(d_model, d_hyper, bias=False)
        nn.init.orthogonal_(self.proj_r.weight)
        nn.init.orthogonal_(self.proj_i.weight)

        # 2. Semantic Hyperspace Item Memory for Expert Addresses in C^D
        self.memory = SemanticHyperspaceMemory(
            d_hyper=d_hyper,
            spawn_threshold=spawn_threshold,
            max_experts=max_experts
        )
        
        # 3. Dynamic Container for Two-Compartment Dendritic Micro-Experts
        self.experts = nn.ModuleList()
        
        # 4. Inter-Expert Global Workspace Bus
        if self.use_bus:
            self.bus = HyperspaceGlobalBus(d_model=d_model, d_hyper=d_hyper)
        else:
            self.bus = None
            
        # 5. Modern Hopfield Dense Associative Clean-Up Memory
        if self.use_hopfield:
            self.hopfield_cleaner = ModernHopfieldMemory(d_dim=d_model, num_patterns=max_experts, beta=6.0, max_iter=2)
        else:
            self.hopfield_cleaner = None

        # 6. Self-Organized Criticality Controller
        if self.use_criticality:
            self.criticality_controller = SelfOrganizedCriticalityController(target_branching_ratio=1.0, base_temperature=10.0)
        else:
            self.criticality_controller = None

        # Telemetry usage buffer
        self.register_buffer("expert_usage_counts", torch.zeros(max_experts, dtype=torch.long))

        # Bootstrap with initial random experts
        for i in range(initial_experts):
            seed = ComplexPhasorVSA.random_hyperspace_vector((1, d_hyper))
            self._spawn_expert(seed_key=seed, label=f"bootstrap_{i}")

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _spawn_expert(self, seed_key: torch.Tensor, label: str = "auto_spawned") -> int:
        """
        Spawns a new Two-Compartment Dendritic Micro-Expert via Clonal Mitosis
        (inheriting weights from the closest parent expert) and binds its learnable address in C^D.
        """
        if len(self.experts) >= self.max_experts:
            return len(self.experts) - 1
        device = seed_key.device
        new_expert = TwoCompartmentDendriticExpert(self.d_model, self.d_ff, device=device)
        
        # Biological Clonal Mitosis: warm-initialize from nearest parent expert if available
        if len(self.experts) > 0 and self.memory.num_experts > 0:
            with torch.no_grad():
                res = self.memory.compute_resonance(seed_key) # [1, N]
                parent_idx = torch.argmax(res[0]).item()
                parent_idx = min(parent_idx, len(self.experts) - 1)
                parent_sd = self.experts[parent_idx].state_dict()
                cloned_sd = {}
                for k, v in parent_sd.items():
                    noise = torch.randn_like(v) * 0.01
                    cloned_sd[k] = v.clone() + noise
                new_expert.load_state_dict(cloned_sd)
        
        self.experts.append(new_expert)
        expert_id = self.memory.register_expert_address(seed_key, label=label)
        return expert_id

    def forward(
        self, x: torch.Tensor, allow_spawning: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        x: [Batch, SeqLen, d_model]
        """
        b, s, d = x.shape
        flat_x = x.view(-1, d) # [N_tokens, d_model]
        num_tokens = flat_x.shape[0]
        
        # 1. Project Tokens to Unit Complex Phasor Hyperspace
        norm_flat_x = F.normalize(flat_x, p=2, dim=-1)
        r_part = self.proj_r(norm_flat_x).float()
        i_part = self.proj_i(norm_flat_x).float()
        z = torch.complex(r_part, i_part)
        query_phasors = z / (torch.abs(z) + 1e-8)
        
        # 2. Novelty Evaluation & Dynamic Spawning
        if self.training and allow_spawning and len(self.experts) < self.max_experts:
            resonance, max_sim, should_spawn, seed_vector = self.memory.evaluate_novelty(query_phasors)
            if should_spawn and seed_vector is not None:
                new_id = self._spawn_expert(seed_vector, label=f"spawned_t{len(self.experts)}")
                resonance = self.memory.compute_resonance(query_phasors)
        else:
            resonance = self.memory.compute_resonance(query_phasors)
            
        # 3. Homeostatic Habituation (Synaptic Fatigue / BCM Rule)
        if self.training and self.num_experts > 1:
            total_usage = self.expert_usage_counts[:self.num_experts].sum().float() + 1e-6
            relative_usage = self.expert_usage_counts[:self.num_experts].float() / total_usage
            fatigue_penalty = relative_usage.unsqueeze(0) * 0.25
            effective_resonance = resonance - fatigue_penalty.to(resonance.device)
        else:
            effective_resonance = resonance

        # 4. Criticality-Regulated Adaptive Multi-Expert Routing
        k_eval = min(self.max_k if self.dynamic_k else self.top_k, self.num_experts)
        if self.criticality_controller is not None:
            raw_logits = effective_resonance * 10.0
            temp, branching_ratio = self.criticality_controller.update_criticality(raw_logits, effective_resonance)
            scaled_logits = effective_resonance * temp
        else:
            scaled_logits = effective_resonance * 10.0
            branching_ratio = 1.0
            temp = 10.0

        router_probs = F.softmax(scaled_logits, dim=-1) # [N_tokens, num_experts]
        top_weights, top_indices = torch.topk(router_probs, k=k_eval, dim=-1) # [N_tokens, k_eval]

        if self.dynamic_k and k_eval > 1:
            # 1. Routing Shannon Entropy per Token (Context Ambiguity/Breadth)
            entropy = -torch.sum(router_probs * torch.log2(router_probs + 1e-12), dim=-1) # [N_tokens]
            max_entropy = math.log2(max(2, self.num_experts))
            norm_entropy = (entropy / max_entropy).clamp(0.0, 1.0) # [N_tokens] in [0, 1]

            # 2. Phasor Resonance Spectral Bandwidth per Token
            max_res_per_token, _ = effective_resonance.max(dim=-1, keepdim=True) # [N_tokens, 1]
            res_cutoff = max_res_per_token * 0.70
            resonant_counts = (effective_resonance >= res_cutoff).float().sum(dim=-1) # [N_tokens]

            # 3. Contextual Dynamic k*(x) Target
            # When input is sharp: norm_entropy -> 0 => k_target -> 1
            # When input is multi-domain: norm_entropy -> 1 => k_target scales up with resonant experts
            k_target = torch.clamp(
                torch.round(1.0 + (k_eval - 1.0) * norm_entropy).long(),
                min=1,
                max=k_eval
            ) # [N_tokens]

            # 4. Construct adaptive token mask
            slot_ranks = torch.arange(k_eval, device=x.device).unsqueeze(0).expand(num_tokens, -1) # [N_tokens, k_eval]
            rank_mask = slot_ranks < k_target.unsqueeze(1)

            # 5. Dual-Gating: Top-p Cumulative Mass Gating
            cumsum_weights = torch.cumsum(top_weights, dim=-1)
            prob_mask = (cumsum_weights - top_weights) < self.top_p
            prob_mask[:, 0] = True

            final_mask = rank_mask & prob_mask
            final_mask[:, 0] = True # Always keep Top-1 primary specialist

            top_weights = top_weights * final_mask.float()
            top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        else:
            top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Auxiliary MoE Load-Balancing Loss
        if self.training and self.num_experts > 1:
            expert_freq = torch.zeros(self.num_experts, device=x.device)
            for exp_id in range(self.num_experts):
                expert_freq[exp_id] = (top_indices == exp_id).float().mean()
            avg_router_prob = router_probs.mean(dim=0) # [num_experts]
            load_balance_loss = float(self.num_experts) * torch.sum(expert_freq * avg_router_prob)
        else:
            load_balance_loss = torch.tensor(0.0, device=x.device)

        # 5. Pass 1: Sparse Token-Dispatched Feedforward Basal Computation
        expert_slot_basal_outputs = torch.zeros(num_tokens, k_eval, d, device=x.device, dtype=flat_x.dtype)
        for k_idx in range(k_eval):
            indices_k = top_indices[:, k_idx]
            weights_k = top_weights[:, k_idx]
            active_token_mask = weights_k > 1e-5
            if not active_token_mask.any():
                continue
            active_exp_ids = torch.unique(indices_k[active_token_mask])
            for exp_id_tensor in active_exp_ids:
                exp_id = exp_id_tensor.item()
                expert = self.experts[exp_id]
                token_mask = active_token_mask & (indices_k == exp_id)
                selected_tokens = flat_x[token_mask]
                exp_out = expert(x_basal=selected_tokens, c_apical=None)
                expert_slot_basal_outputs[token_mask, k_idx] = exp_out.to(expert_slot_basal_outputs.dtype)
                
                if self.training:
                    self.expert_usage_counts[exp_id] += token_mask.sum().item()

        # 5. Pass 2: Hyperspace Global Bus & Apical Context Feedback
        if self.bus is not None and k_eval > 1:
            selected_keys = self.memory.expert_keys[top_indices] # [N_tokens, k_eval, d_hyper]
            bus_context = self.bus.broadcast_and_listen(
                expert_outputs=expert_slot_basal_outputs,
                expert_keys=selected_keys,
                expert_weights=top_weights.unsqueeze(-1).to(expert_slot_basal_outputs.dtype)
            ) # [N_tokens, k_eval, d_model]
            
            # Optional Modern Hopfield Associative Clean-Up
            if self.hopfield_cleaner is not None:
                bus_context_flat = bus_context.view(-1, d)
                cleaned_context, _ = self.hopfield_cleaner(bus_context_flat)
                bus_context = cleaned_context.view(num_tokens, k_eval, d)

            # Sparse Token-Dispatched Somatic Integration with Apical Context
            expert_slot_final_outputs = torch.zeros(num_tokens, k_eval, d, device=x.device, dtype=flat_x.dtype)
            for k_idx in range(k_eval):
                indices_k = top_indices[:, k_idx]
                weights_k = top_weights[:, k_idx]
                active_token_mask = weights_k > 1e-5
                if not active_token_mask.any():
                    continue
                active_exp_ids = torch.unique(indices_k[active_token_mask])
                for exp_id_tensor in active_exp_ids:
                    exp_id = exp_id_tensor.item()
                    expert = self.experts[exp_id]
                    token_mask = active_token_mask & (indices_k == exp_id)
                    tokens_b = flat_x[token_mask]
                    context_a = bus_context[token_mask, k_idx]
                    exp_out = expert(x_basal=tokens_b, c_apical=context_a)
                    expert_slot_final_outputs[token_mask, k_idx] = exp_out.to(expert_slot_final_outputs.dtype)
        else:
            expert_slot_final_outputs = expert_slot_basal_outputs

        # 6. Output Synthesis
        out = (expert_slot_final_outputs * top_weights.unsqueeze(-1).to(expert_slot_final_outputs.dtype)).sum(dim=1)
        ortho_loss = self.memory.compute_orthogonality_loss()
        
        telemetry = {
            "num_experts": self.num_experts,
            "mean_resonance": resonance.mean().item(),
            "max_resonance": resonance.max(dim=-1)[0].mean().item(),
            "branching_ratio": branching_ratio,
            "routing_temperature": temp,
            "ortho_loss": ortho_loss,
            "load_balance_loss": load_balance_loss,
            "mean_active_k": (top_weights > 1e-5).float().sum(dim=-1).mean().item(),
            "top_indices": top_indices.detach().view(b, s, k_eval),
            "top_weights": top_weights.detach().view(b, s, k_eval),
        }

        return out.view(b, s, d), telemetry
