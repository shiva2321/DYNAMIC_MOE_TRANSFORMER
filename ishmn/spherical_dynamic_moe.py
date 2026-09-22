"""
Dynamic Spherical Hyperspace Mixture-of-Experts (SphericalDynamicMoE).
Adapted from Universal SubStrait for Infinite Spherical Hyperspace.

Features:
1. Experts anchored in S^{D-1} x R+ hyperspace coordinates.
2. Spherical resonance routing (geodesic angle alignment + radial proximity).
3. Autonomous Neurogenesis via Clonal Mitosis upon novelty (resonance < tau_spawn).
4. DynamicWarmupAdamW integration for shock-free continuous parameter registration.
5. Sleep-phase barycentric weight merging and apoptotic pruning.
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import (
    project_to_sphere,
    geodesic_distance_sphere,
    log_map_sphere,
    exp_map_sphere,
)
from ishmn.neuro.dendritic_expert import TwoCompartmentDendriticExpert
from ishmn.dynamic_optimizer import DynamicWarmupAdamW

class SphericalDynamicMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        d_ff: int = 64,
        d_context: int = 32,
        hyperspace_dim: int = 8,
        top_k: int = 2,
        initial_experts: int = 1,
        max_experts: Optional[int] = None,
        spawn_threshold: float = 0.40,
        merge_angle_threshold: float = 0.15,
        min_experts: int = 1,
        kappa_route: float = 4.0,
        beta_route: float = 0.8,
        enable_shared_expert: bool = False,
        enable_aux_loss_free_balancing: bool = False,
        bias_gamma: float = 0.05,
        enable_gwt_coalition: bool = False,
        gwt_context_weight: float = 0.3,
        split_lambda_threshold: float = 0.08,
        split_anisotropy_ratio: float = 2.0,
        split_kurtosis_threshold: float = -0.5,
        split_min_count: int = 64,
        split_lockout_sleeps: int = 3,
        scatter_momentum: float = 0.05,
        reservoir_size: int = 256,
        use_drift_guard: bool = False,
        drift_sigma: float = 2.0,
        orthogonalize_on_spawn: bool = False,
        enable_online_spawning: Optional[bool] = None,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_context = d_context
        self.hyperspace_dim = hyperspace_dim
        self.top_k = top_k
        self.max_experts = max_experts
        self.spawn_threshold = spawn_threshold
        self.merge_angle_threshold = merge_angle_threshold
        self.min_experts = min_experts
        self.kappa_route = kappa_route
        self.beta_route = beta_route
        self.enable_shared_expert = enable_shared_expert
        self.enable_aux_loss_free_balancing = enable_aux_loss_free_balancing
        self.bias_gamma = bias_gamma
        self.enable_gwt_coalition = enable_gwt_coalition
        self.gwt_context_weight = gwt_context_weight
        self.use_drift_guard = use_drift_guard
        self.drift_sigma = drift_sigma
        self.orthogonalize_on_spawn = orthogonalize_on_spawn
        self.enable_online_spawning = enable_online_spawning

        # Gem 3: GWT Coalition Proposal Projector (maps context to hyperspace anchor bids)
        if enable_gwt_coalition:
            self.context_to_proposal = nn.Linear(self.d_context, self.hyperspace_dim, bias=False, device=device)
            nn.init.xavier_uniform_(self.context_to_proposal.weight)
        else:
            self.context_to_proposal = None

        # Anisotropic mitosis controls (see mitose_anisotropic)
        self.split_lambda_threshold = split_lambda_threshold
        self.split_anisotropy_ratio = split_anisotropy_ratio
        self.split_kurtosis_threshold = split_kurtosis_threshold
        self.split_min_count = split_min_count
        self.split_lockout_sleeps = split_lockout_sleeps
        self.scatter_momentum = scatter_momentum
        self.reservoir_size = reservoir_size

        # Statistical Outlier Drift-Guard Buffers (Adapted from Universal SubStrait)
        buf_cap = max_experts if max_experts is not None else 64
        self.register_buffer("running_sim_mean", torch.zeros(buf_cap, device=device), persistent=False)
        self.register_buffer("running_sim_var", torch.ones(buf_cap, device=device) * 0.04, persistent=False)
        self.register_buffer("usage_counts", torch.zeros(buf_cap, dtype=torch.long, device=device), persistent=False)

        # Shared Expert Isolation (DeepSeek-style): always active for all tokens
        if enable_shared_expert:
            self.shared_expert = TwoCompartmentDendriticExpert(
                d_model=self.d_model,
                d_ff=self.d_ff,
                d_context=self.d_context,
                device=device
            )
        else:
            self.shared_expert = None


        # ModuleList container for dendritic experts
        self.experts = nn.ModuleList()

        # Expert hyperspace anchors: u_anchors in S^{D-1}, r_anchors in [0, inf)
        self.register_buffer("expert_u", torch.empty(0, hyperspace_dim, device=device))
        self.register_buffer("expert_r", torch.empty(0, 1, device=device))
        self.register_buffer("expert_usage", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("expert_bias", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("routed_load_ema", torch.empty(0, dtype=torch.float32, device=device))

        # Receptive-field second moment in T_{u_e} S^{D-1}, evidence count, and post-split
        # merge immunity. Drive anisotropic mitosis (see mitose_anisotropic).
        self.register_buffer("expert_scatter", torch.empty(0, hyperspace_dim, hyperspace_dim, device=device))
        self.register_buffer("expert_route_count", torch.empty(0, dtype=torch.float32, device=device))
        self.register_buffer("expert_lockout", torch.empty(0, dtype=torch.float32, device=device))

        # Reservoir of recent tangent vectors per expert. The second moment alone cannot
        # tell a bimodal field from a merely stretched one, so the split test needs the
        # projected shape, not just its variance (see evaluate_split_criterion).
        self.register_buffer("expert_reservoir", torch.empty(0, reservoir_size, hyperspace_dim, device=device))
        self.register_buffer("expert_reservoir_fill", torch.empty(0, dtype=torch.float32, device=device))

        # Bootstrap initial experts
        for i in range(initial_experts):
            seed_u = project_to_sphere(torch.randn(1, hyperspace_dim, device=device))
            seed_r = torch.tensor([[1.0 + i * 1.5]], device=device)
            self._spawn_expert_raw(seed_u, seed_r, parent_idx=None)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Backward compatibility: checkpoints written before receptive-field tracking existed
        # carry none of the mitosis statistics buffers. Seed them at their current shape so
        # strict loading of an older checkpoint still succeeds; the statistics simply start
        # empty and re-accumulate.
        for buf_name in ["expert_scatter", "expert_route_count", "expert_lockout",
                         "expert_reservoir", "expert_reservoir_fill"]:
            key = prefix + buf_name
            if key not in state_dict:
                current = getattr(self, buf_name, None)
                if current is not None:
                    state_dict[key] = torch.zeros_like(current)

        # Gem 3: Seed missing GWT context projection if loading older checkpoint
        if self.enable_gwt_coalition and self.context_to_proposal is not None:
            ctx_key = prefix + "context_to_proposal.weight"
            if ctx_key not in state_dict:
                state_dict[ctx_key] = self.context_to_proposal.weight.data.clone()

        # Dynamically instantiate missing expert containers if checkpoint contains dynamically spawned experts
        exp_indices = set()
        for k in state_dict.keys():
            if k.startswith(prefix + "experts."):
                parts = k[len(prefix + "experts."):].split(".")
                if parts[0].isdigit():
                    exp_indices.add(int(parts[0]))
        if exp_indices:
            target_count = max(exp_indices) + 1
            device = self.expert_u.device if self.expert_u.numel() > 0 else torch.device("cpu")
            while len(self.experts) < target_count:
                new_u = torch.zeros(1, self.hyperspace_dim, device=device)
                new_r = torch.ones(1, 1, device=device)
                self._spawn_expert_raw(new_u, new_r)

            # Adjust buffers to match target count if present in checkpoint
            for buf_name in ["expert_u", "expert_r", "expert_usage", "expert_bias", "routed_load_ema",
                             "expert_scatter", "expert_route_count", "expert_lockout",
                             "expert_reservoir", "expert_reservoir_fill",
                             "running_sim_mean", "running_sim_var", "usage_counts"]:
                state_key = prefix + buf_name
                if state_key in state_dict:
                    setattr(self, buf_name, state_dict[state_key].clone().to(device))

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _spawn_expert_raw(
        self,
        u_coord: torch.Tensor,
        r_coord: torch.Tensor,
        parent_idx: Optional[int] = None
    ) -> int:
        """Internal helper to instantiate an expert and bind anchor coordinates."""
        device = u_coord.device
        new_expert = TwoCompartmentDendriticExpert(
            d_model=self.d_model,
            d_ff=self.d_ff,
            d_context=self.d_context,
            device=device
        )

        # Clonal Mitosis: clone parent weights + small noise perturbation
        if parent_idx is not None and 0 <= parent_idx < len(self.experts):
            with torch.no_grad():
                parent_sd = self.experts[parent_idx].state_dict()
                cloned_sd = {}
                for k, v in parent_sd.items():
                    noise = torch.randn_like(v) * 0.01
                    cloned_sd[k] = v.clone() + noise
                new_expert.load_state_dict(cloned_sd)

        # Gram-Schmidt Key Orthogonalization (Adapted from Universal SubStrait)
        if self.orthogonalize_on_spawn and self.expert_u.numel() > 0:
            with torch.no_grad():
                D = self.hyperspace_dim
                N = self.expert_u.size(0)
                u_f = u_coord.float()
                if N < D:
                    Q, _ = torch.linalg.qr(self.expert_u.t().float()) # [D, N]
                    u_proj = torch.mm(torch.mm(u_f, Q), Q.t())
                    ortho = u_f - u_proj
                    norm = torch.norm(ortho, dim=-1, keepdim=True)
                    if norm.item() < 1e-4:
                        rand_v = torch.randn_like(u_f)
                        rand_proj = torch.mm(torch.mm(rand_v, Q), Q.t())
                        ortho = rand_v - rand_proj
                        norm = torch.norm(ortho, dim=-1, keepdim=True)
                    u_coord = (ortho / norm.clamp(min=1e-8)).to(u_coord.dtype)
                else:
                    # Hyperspace is fully spanned (N >= D). Find direction of maximal angular separation
                    candidates = torch.randn(64, D, device=u_coord.device)
                    candidates = candidates / candidates.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    cosines = torch.mm(candidates, self.expert_u.t().float()) # [64, N]
                    max_cosines = cosines.abs().max(dim=-1).values
                    best_idx = torch.argmin(max_cosines).item()
                    u_coord = candidates[best_idx:best_idx+1].to(u_coord.dtype)

        self.experts.append(new_expert)

        # Update anchor buffers
        self.expert_u = torch.cat([self.expert_u, u_coord], dim=0)
        self.expert_r = torch.cat([self.expert_r, r_coord], dim=0)
        new_usage = torch.tensor([1.0], device=device)
        self.expert_usage = torch.cat([self.expert_usage, new_usage], dim=0)
        self.expert_bias = torch.cat([self.expert_bias, torch.zeros(1, device=device)], dim=0)
        self.routed_load_ema = torch.cat([self.routed_load_ema, torch.zeros(1, device=device)], dim=0)
        self.expert_scatter = torch.cat(
            [self.expert_scatter, torch.zeros(1, self.hyperspace_dim, self.hyperspace_dim, device=device)], dim=0
        )
        self.expert_route_count = torch.cat([self.expert_route_count, torch.zeros(1, device=device)], dim=0)
        self.expert_lockout = torch.cat([self.expert_lockout, torch.zeros(1, device=device)], dim=0)
        self.expert_reservoir = torch.cat(
            [self.expert_reservoir, torch.zeros(1, self.reservoir_size, self.hyperspace_dim, device=device)], dim=0
        )
        self.expert_reservoir_fill = torch.cat([self.expert_reservoir_fill, torch.zeros(1, device=device)], dim=0)

        return len(self.experts) - 1

    @torch.no_grad()
    def observe_receptive_field(self, u_x: torch.Tensor, assign_idx: torch.Tensor) -> None:
        """
        Streaming update of each expert's receptive-field second moment in its own tangent space.

        For inputs u routed to expert e with anchor mu_e, accumulates
            Sigma_e <- (1 - eta) * Sigma_e + eta * E[ log_{mu_e}(u) log_{mu_e}(u)^T ]
        Sigma_e is rank <= D-1 (it always has a null direction along mu_e), and its leading
        eigenpair describes the dominant elongation of the region the expert actually serves.
        """
        if self.expert_scatter.numel() == 0 or u_x.numel() == 0:
            return
        eta = self.scatter_momentum
        for e in torch.unique(assign_idx).tolist():
            if e >= self.num_experts:
                continue
            members = u_x[assign_idx == e]
            if members.numel() == 0:
                continue
            mu = self.expert_u[e].unsqueeze(0).expand_as(members)
            tangents = log_map_sphere(mu, members)                       # [n, D]
            batch_scatter = (tangents.t() @ tangents) / members.size(0)  # [D, D]
            self.expert_scatter[e] = (1.0 - eta) * self.expert_scatter[e] + eta * batch_scatter
            self.expert_route_count[e] = self.expert_route_count[e] + members.size(0)

            # Reservoir: fill sequentially, then replace uniformly at random.
            n_new = members.size(0)
            fill = int(self.expert_reservoir_fill[e].item())
            if fill < self.reservoir_size:
                take = min(n_new, self.reservoir_size - fill)
                self.expert_reservoir[e, fill:fill + take] = tangents[:take]
                self.expert_reservoir_fill[e] = fill + take
                tangents, n_new = tangents[take:], n_new - take
            if n_new > 0:
                slots = torch.randint(0, self.reservoir_size, (n_new,), device=tangents.device)
                self.expert_reservoir[e, slots] = tangents

    def evaluate_split_criterion(self, expert_idx: int) -> Dict[str, Any]:
        """
        Second-moment split test for expert `expert_idx`.

        An expert is split when the region it serves has become large, elongated AND
        genuinely two-lobed:
            lambda_1 > tau_split           (the field has spread)
            lambda_1 / lambda_2 > rho      (it has stretched along one axis)
            excess kurtosis < kappa < 0    (that axis is bimodal, not just long)
            n_e >= n_min                   (enough routed evidence to trust the estimate)

        The kurtosis term is load-bearing. Anisotropy alone cannot separate a bimodal
        field from a unimodal stretched one -- both are elongated -- and splitting the
        latter fragments a coherent region, which measurably worsens load balance
        (Gini 0.509 -> 0.627 on the live brain when this gate was absent). Excess
        kurtosis is ~0 for a Gaussian, -1.2 for uniform, and -2 for a clean two-point
        mixture, so a negative threshold admits only actually-separated lobes.
        """
        if expert_idx >= self.num_experts or self.expert_scatter.numel() == 0:
            return {
                "should_split": False,
                "reason": "no_such_expert",
                "route_count": 0.0,
                "excess_kurtosis": 0.0,
                "kurtosis": 0.0,
                "anisotropy": 1.0,
                "lambda_1": 0.0,
                "lambda_2": 0.0,
                "principal_direction": torch.zeros(self.hyperspace_dim, device=self.expert_u.device)
            }

        sigma = self.expert_scatter[expert_idx].float()
        sigma = 0.5 * (sigma + sigma.t())  # symmetrise against accumulated float drift
        evals, evecs = torch.linalg.eigh(sigma)
        lam1 = float(evals[-1].item())
        lam2 = float(evals[-2].clamp(min=1e-8).item())
        ratio = lam1 / lam2
        count = float(self.expert_route_count[expert_idx].item())
        w1 = evecs[:, -1]

        kurt = self._projected_excess_kurtosis(expert_idx, w1)

        should = (
            lam1 > self.split_lambda_threshold
            and ratio > self.split_anisotropy_ratio
            and kurt is not None
            and kurt < self.split_kurtosis_threshold
            and count >= self.split_min_count
            and ((self.max_experts is None) or (self.num_experts < self.max_experts))
        )
        return {
            "should_split": bool(should),
            "lambda_1": lam1,
            "lambda_2": lam2,
            "anisotropy": ratio,
            "excess_kurtosis": kurt,
            "kurtosis": kurt,
            "route_count": count,
            "principal_direction": w1.clone(),
        }

    @torch.no_grad()
    def _projected_excess_kurtosis(self, expert_idx: int, direction: torch.Tensor) -> Optional[float]:
        """
        Excess kurtosis of the reservoir projected onto `direction`.

        Returns None when the reservoir holds too few samples to estimate a 4th moment.
        """
        if self.expert_reservoir.numel() == 0:
            return None
        fill = int(self.expert_reservoir_fill[expert_idx].item())
        n = self.reservoir_size if fill >= self.reservoir_size else fill
        if n < 32:
            return None

        s = self.expert_reservoir[expert_idx, :n].float() @ direction.float()
        s = s - s.mean()
        var = s.pow(2).mean()
        if var <= 1e-12:
            return None
        return float((s.pow(4).mean() / var.pow(2) - 3.0).item())

    def estimate_marginal_kurtosis(self, expert_idx: int, direction: torch.Tensor) -> Optional[float]:
        """Alias for backward compatibility."""
        return self._projected_excess_kurtosis(expert_idx, direction)


    @torch.no_grad()
    def mitose_anisotropic(self, expert_idx: int) -> Optional[int]:
        """
        Splits an over-stretched expert into two daughters along its principal tangent axis.
        """
        crit = self.evaluate_split_criterion(expert_idx)
        if not crit["should_split"]:
            return None

        mu = self.expert_u[expert_idx].unsqueeze(0)
        w1 = crit["principal_direction"].unsqueeze(0).to(mu.dtype)
        step = math.sqrt(max(crit["lambda_1"], 1e-8))
        u_plus = exp_map_sphere(mu, step * w1)
        u_minus = exp_map_sphere(mu, -step * w1)

        child_idx = self._spawn_expert_raw(u_minus, self.expert_r[expert_idx].clone().unsqueeze(0), parent_idx=None)

        parent_sd = self.experts[expert_idx].state_dict()
        child_sd = self.experts[child_idx].state_dict()
        for key in parent_sd:
            if not torch.is_floating_point(parent_sd[key]):
                child_sd[key] = parent_sd[key].clone()
                continue
            delta = torch.randn_like(parent_sd[key]) * 0.01
            child_sd[key] = parent_sd[key] - delta
            parent_sd[key] = parent_sd[key] + delta
        self.experts[expert_idx].load_state_dict(parent_sd)
        self.experts[child_idx].load_state_dict(child_sd)

        self.expert_u[expert_idx] = u_plus.squeeze(0)

        half_usage = self.expert_usage[expert_idx] * 0.5
        self.expert_usage[expert_idx] = half_usage
        self.expert_usage[child_idx] = half_usage
        self.expert_scatter[expert_idx].zero_()
        self.expert_scatter[child_idx].zero_()
        self.expert_route_count[expert_idx] = 0.0
        self.expert_route_count[child_idx] = 0.0
        self.expert_reservoir[expert_idx].zero_()
        self.expert_reservoir[child_idx].zero_()
        self.expert_reservoir_fill[expert_idx] = 0.0
        self.expert_reservoir_fill[child_idx] = 0.0
        self.expert_lockout[expert_idx] = float(self.split_lockout_sleeps)
        self.expert_lockout[child_idx] = float(self.split_lockout_sleeps)
        return child_idx

    @torch.no_grad()
    def mitose_all_eligible(self, max_splits: int = 2) -> List[Dict[str, Any]]:
        """Applies anisotropic mitosis to the most over-stretched experts (largest lambda_1 first)."""
        scored = []
        for e in range(self.num_experts):
            crit = self.evaluate_split_criterion(e)
            if crit["should_split"]:
                scored.append((crit["lambda_1"], e))
        scored.sort(reverse=True)

        events = []
        for _, e in scored[:max_splits]:
            crit = self.evaluate_split_criterion(e)
            child = self.mitose_anisotropic(e)
            if child is not None:
                events.append({
                    "parent": e,
                    "child": child,
                    "lambda_1": crit["lambda_1"],
                    "anisotropy": crit["anisotropy"],
                    "separation_rad": 2.0 * math.sqrt(max(crit["lambda_1"], 1e-8)),
                })
        return events

    def compute_resonance(
        self,
        u_x: torch.Tensor,
        r_x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        apply_bias: bool = True
    ) -> torch.Tensor:
        """
        Computes spherical resonance between input coordinate and all expert anchors.
        Args:
            u_x: [B, D] in S^{D-1}
            r_x: [B, 1] in [0, inf)
            context: Optional [B, d_context] context for GWT coalition proposals
            apply_bias: Whether to add aux-loss-free expert bias for balanced routing
        Returns:
            resonance: [B, E]
        """
        # u_x: [B, D], expert_u: [E, D] -> cos_sim: [B, E]
        cos_sim = torch.mm(u_x, self.expert_u.t())

        # Radial distance: [B, 1] vs [E, 1] -> [B, E]
        dr_sq = (r_x - self.expert_r.t()) ** 2

        resonance = self.kappa_route * cos_sim - self.beta_route * dr_sq

        # Gem 3: GWT Coalition Proposal Integration
        if self.enable_gwt_coalition and context is not None and self.context_to_proposal is not None and self.expert_u.numel() > 0:
            u_ctx = F.normalize(self.context_to_proposal(context), p=2, dim=-1) # [B, D]
            context_bids = torch.mm(u_ctx, self.expert_u.t()) # [B, E]
            resonance = resonance + self.gwt_context_weight * context_bids

        if apply_bias and self.enable_aux_loss_free_balancing and self.expert_bias.numel() == resonance.size(1):
            resonance = resonance + self.expert_bias.unsqueeze(0)
        return resonance

    def forward(
        self,
        x: torch.Tensor,
        u_x: torch.Tensor,
        r_x: torch.Tensor,
        context: torch.Tensor = None,
        optimizer: Optional[DynamicWarmupAdamW] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x: [B, d_model] Input representations
            u_x: [B, hyperspace_dim] S^{D-1} coordinates
            r_x: [B, 1] Radial coordinates
            context: [B, d_context] Optional memory context
            optimizer: Optional DynamicWarmupAdamW for online neurogenesis registration
        """
        B = x.size(0)
        E = self.num_experts

        # 1. Compute Resonance (including Gem 3 GWT Coalition proposals if enabled)
        resonance = self.compute_resonance(u_x, r_x, context=context, apply_bias=self.enable_aux_loss_free_balancing) # [B, E]

        # 2. Check for Novelty & Autonomous Neurogenesis
        max_res, closest_expert = torch.max(resonance, dim=-1) # [B]
        spawned_expert_idx = None
        can_spawn = (self.max_experts is None) or (E < self.max_experts)
        should_check_online = self.enable_online_spawning if self.enable_online_spawning is not None else (optimizer is not None)
        if self.training and can_spawn and should_check_online:
            # If batch contains inputs with low resonance across all experts
            min_max_res = max_res.min().item()
            if min_max_res < self.spawn_threshold:
                novel_b = torch.argmin(max_res).item()
                parent = closest_expert[novel_b].item()
                new_u = u_x[novel_b:novel_b+1].detach()
                new_r = r_x[novel_b:novel_b+1].detach()

                spawned_expert_idx = self._spawn_expert_raw(new_u, new_r, parent_idx=parent)

                # Register in DynamicWarmupAdamW or PyTorch optimizer without gradient shock
                if optimizer is not None:
                    if hasattr(optimizer, "add_dynamic_param_group"):
                        optimizer.add_dynamic_param_group(
                            list(self.experts[spawned_expert_idx].parameters()),
                            group_name=f"expert_{spawned_expert_idx}"
                        )
                    elif hasattr(optimizer, "add_param_group"):
                        optimizer.add_param_group({"params": list(self.experts[spawned_expert_idx].parameters())})

                # Recompute resonance with newly spawned expert
                E = self.num_experts
                resonance = self.compute_resonance(u_x, r_x, context=context, apply_bias=self.enable_aux_loss_free_balancing)

        # Statistical Outlier Drift-Guard (Adapted from Universal SubStrait)
        if self.use_drift_guard and E > 0:
            if self.training:
                with torch.no_grad():
                    active_E = min(E, self.running_sim_mean.numel())
                    for e in range(active_E):
                        batch_res_e = resonance[:, e].detach()
                        b_mean = batch_res_e.mean()
                        b_var = batch_res_e.var().clamp(min=1e-4, max=1.0) if B > 1 else torch.tensor(0.04, device=x.device)
                        alpha = 0.95
                        self.running_sim_mean[e] = alpha * self.running_sim_mean[e] + (1.0 - alpha) * b_mean
                        self.running_sim_var[e] = (alpha * self.running_sim_var[e] + (1.0 - alpha) * b_var).clamp(min=1e-4, max=1.0)
                        self.usage_counts[e] += B
            active_E = min(E, self.running_sim_mean.numel())
            std_expanded = torch.sqrt(self.running_sim_var[:active_E] + 1e-6).unsqueeze(0)
            mean_expanded = self.running_sim_mean[:active_E].unsqueeze(0)
            is_outlier = torch.abs(resonance[:, :active_E] - mean_expanded) > (self.drift_sigma * std_expanded)
            guarded_res = torch.where(is_outlier, resonance[:, :active_E].detach(), resonance[:, :active_E])
            if active_E < E:
                resonance = torch.cat([guarded_res, resonance[:, active_E:]], dim=-1)
            else:
                resonance = guarded_res

        # 3. Top-k Routing
        k = min(self.top_k, E)
        topk_scores, topk_indices = torch.topk(resonance, k=k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1) # [B, k]

        # Aux-Loss-Free Dynamic Balancing (DeepSeek-style EMA bias update outside autograd)
        if self.training and self.enable_aux_loss_free_balancing and E > 1:
            with torch.no_grad():
                counts_routed = torch.bincount(topk_indices.view(-1), minlength=E).float()
                batch_load = counts_routed / float(B * k)
                if self.routed_load_ema.numel() == E:
                    self.routed_load_ema.mul_(0.9).add_(batch_load, alpha=0.1)
                    target_load = 1.0 / float(E)
                    self.expert_bias.data.add_(self.routed_load_ema - target_load, alpha=-self.bias_gamma)
                    self.expert_bias.data.clamp_(-2.0, 2.0)

        # Accumulate receptive-field geometry for the primary assignment of each token.
        if self.training and self.expert_scatter.numel() > 0:
            self.observe_receptive_field(u_x.detach(), topk_indices[:, 0].detach())

        # 4. Token-Sorted Grouped Feedforward Dispatch (GPU Optimized)
        B, D = x.shape
        flat_exp_idx = topk_indices.view(-1) # [B * k]
        tokens_expanded = x.unsqueeze(1).expand(B, k, D).reshape(B * k, D)
        if context is not None:
            ctx_expanded = context.unsqueeze(1).expand(B, k, -1).reshape(B * k, -1)
        else:
            ctx_expanded = None

        counts = torch.bincount(flat_exp_idx, minlength=E)
        sort_indices = torch.argsort(flat_exp_idx)
        sorted_tokens = tokens_expanded[sort_indices]
        sorted_ctx = ctx_expanded[sort_indices] if ctx_expanded is not None else None

        chunk_outputs = []
        counts_cpu = counts.tolist()
        start_idx = 0
        for exp_id, count in enumerate(counts_cpu):
            if count == 0:
                continue
            end_idx = start_idx + count
            exp_tokens = sorted_tokens[start_idx:end_idx]
            exp_c = sorted_ctx[start_idx:end_idx] if sorted_ctx is not None else None
            exp_out = self.experts[exp_id](exp_tokens, context=exp_c)
            chunk_outputs.append(exp_out)
            if self.training:
                self.expert_usage = self.expert_usage.clone()
                self.expert_usage[exp_id] = self.expert_usage[exp_id] + count
            start_idx = end_idx

        if len(chunk_outputs) > 0:
            sorted_outputs = torch.cat(chunk_outputs, dim=0)
        else:
            sorted_outputs = torch.zeros(B * k, D, device=x.device, dtype=x.dtype)

        # Invert sorting to reconstruct [B, k, D]
        inv_sort_indices = torch.empty_like(sort_indices)
        inv_sort_indices[sort_indices] = torch.arange(len(sort_indices), device=x.device)
        expert_slot_outputs = sorted_outputs[inv_sort_indices].view(B, k, D)

        # Weighted combination across top-k
        out = (expert_slot_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)

        # 5. DeepSeek-Style Shared Expert Isolation
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x, context=context)
            out = shared_out + out

        routing_entropy = -(topk_weights * torch.log(topk_weights.clamp(min=1e-8))).sum(dim=-1).mean().item()

        metadata = {
            "num_experts": E,
            "spawned_expert": spawned_expert_idx,
            "topk_indices": topk_indices.detach().cpu().tolist(),
            "topk_weights": topk_weights.detach().cpu().tolist(),
            "routing_entropy": routing_entropy,
            "expert_usage": self.expert_usage.detach().cpu().tolist(),
            "expert_anchors_u": self.expert_u.detach().cpu().tolist(),
            "expert_anchors_r": self.expert_r.squeeze(-1).detach().cpu().tolist()
        }

        return out, metadata

    def sleep_consolidate_and_prune(
        self,
        prune_usage_threshold: float = 0.02,
        optimizer: Optional[DynamicWarmupAdamW] = None
    ) -> Dict[str, int]:
        """
        Barycentric Merging of co-resonant experts and apoptosis of dead/dormant experts.
        Adapted from Universal SubStrait sleep consolidation.
        """
        E = self.num_experts
        if E <= self.min_experts:
            return {"merged": 0, "pruned": 0, "final_experts": E}

        merged_count = 0
        pruned_count = 0

        with torch.no_grad():
            # 1. Geodesic Barycentric Merging for co-resonant experts
            cos_matrix = torch.mm(self.expert_u, self.expert_u.t()).clamp(-1.0, 1.0)
            angles = torch.acos(cos_matrix)

            to_merge = set()
            for i in range(E):
                if i in to_merge:
                    continue
                for j in range(i + 1, E):
                    if j in to_merge:
                        continue
                    # Freshly split daughters are immune: without this the barycentric
                    # merge re-absorbs them before they can differentiate.
                    if self.expert_lockout.numel() > max(i, j) and (
                        self.expert_lockout[i] > 0 or self.expert_lockout[j] > 0
                    ):
                        continue
                    if angles[i, j].item() < self.merge_angle_threshold:
                        # Merge j into i using usage-weighted barycentric interpolation
                        u_i = max(1.0, float(self.expert_usage[i].item()))
                        u_j = max(1.0, float(self.expert_usage[j].item()))
                        w_i = u_i / (u_i + u_j)
                        w_j = u_j / (u_i + u_j)

                        sd_i = self.experts[i].state_dict()
                        sd_j = self.experts[j].state_dict()
                        for key in sd_i:
                            sd_i[key] = w_i * sd_i[key] + w_j * sd_j[key]
                        self.experts[i].load_state_dict(sd_i)

                        # Update anchor towards spherical geodesic midpoint
                        self.expert_u[i] = project_to_sphere(w_i * self.expert_u[i] + w_j * self.expert_u[j])
                        self.expert_r[i] = w_i * self.expert_r[i] + w_j * self.expert_r[j]
                        self.expert_usage[i] += self.expert_usage[j]

                        to_merge.add(j)
                        merged_count += 1

            surviving_after_merge = [idx for idx in range(E) if idx not in to_merge]

            # 2. Apoptotic Pruning of dormant / unutilized experts
            to_prune = set()
            total_usage = float(self.expert_usage[surviving_after_merge].sum().item())
            if total_usage > 0 and len(surviving_after_merge) > self.min_experts:
                # Sort from lowest to highest usage
                usage_sorted = sorted(surviving_after_merge, key=lambda idx: float(self.expert_usage[idx].item()))
                for idx in usage_sorted:
                    # Guard minimum expert floor
                    if (len(surviving_after_merge) - len(to_prune)) <= self.min_experts:
                        break
                    rel_usage = float(self.expert_usage[idx].item()) / total_usage
                    if rel_usage < prune_usage_threshold:
                        to_prune.add(idx)
                        pruned_count += 1

            # 3. Rebuild surviving expert modules and coordinate buffers
            final_surviving = [idx for idx in surviving_after_merge if idx not in to_prune]
            if len(final_surviving) >= self.min_experts and len(final_surviving) < E:
                new_experts_list = nn.ModuleList([self.experts[i] for i in final_surviving])
                self.experts = new_experts_list
                self.expert_u = self.expert_u[final_surviving].clone()
                self.expert_r = self.expert_r[final_surviving].clone()
                self.expert_usage = self.expert_usage[final_surviving].clone()
                if self.expert_bias.numel() == E:
                    self.expert_bias = self.expert_bias[final_surviving].clone()
                if self.routed_load_ema.numel() == E:
                    self.routed_load_ema = self.routed_load_ema[final_surviving].clone()
                if self.expert_scatter.size(0) == E:
                    self.expert_scatter = self.expert_scatter[final_surviving].clone()
                if self.expert_route_count.numel() == E:
                    self.expert_route_count = self.expert_route_count[final_surviving].clone()
                if self.expert_lockout.numel() == E:
                    self.expert_lockout = self.expert_lockout[final_surviving].clone()
                if self.expert_reservoir.size(0) == E:
                    self.expert_reservoir = self.expert_reservoir[final_surviving].clone()
                if self.expert_reservoir_fill.numel() == E:
                    self.expert_reservoir_fill = self.expert_reservoir_fill[final_surviving].clone()

        # Decay post-split merge immunity by one sleep cycle.
        if self.expert_lockout.numel() > 0:
            self.expert_lockout.clamp_(min=0.0).sub_(1.0).clamp_(min=0.0)

        # 4. Synchronize with DynamicWarmupAdamW if provided
        if optimizer is not None and (merged_count > 0 or pruned_count > 0):
            optimizer.prune_invalid_params(set(self.parameters()))

        return {"merged": merged_count, "pruned": pruned_count, "final_experts": self.num_experts}

