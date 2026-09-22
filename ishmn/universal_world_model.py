"""
Universal Latent World Model (Predictive JEPA on Riemannian Manifolds).

Completely domain-agnostic: No arithmetic, no base-10 digits, no hardcoded dictionaries.
Operates purely on continuous latent states z in S^{D-1} x R+ and context tokens c.

Core Principles:
1. Forward Latent Transition: z_{t+1} = W_fwd(z_t, c_t)
2. Inverse Latent Transition: z_t = W_bwd(z_{t+1}, c_t)
3. Self-Supervised Geodesic Reversibility (Pearl Level 3 Causal Consistency):
   L_causal = d_geodesic( W_bwd(W_fwd(z, c), c), z )
4. Adaptive Computation Time (ACT / System 1 vs System 2 Thinking):
   Circulates endogenous latent steps on the manifold until phase-lock consensus
   crosses the emission threshold (tau >= 0.70).
"""

import math
from typing import Optional, Any, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ishmn.geometry import (
    project_to_sphere,
    geodesic_distance_sphere,
    exp_map_sphere
)
from ishmn.spherical_attention import (
    RiemannianSphericalAttention,
    LoopedRiemannianTransformer,
    DualStreamRiemannianAttention
)
from ishmn.neuro.predictive_coding import HierarchicalRiemannianPredictiveCoding

class PonderHalting(nn.Module):
    """
    PonderNet-style Probabilistic Halting Mechanism with Geometric Prior.
    Reference: Banino et al. (DeepMind 2021), 'PonderNet: Learning to Ponder'

    Formulates dynamic computation as a proper hazard probability distribution p(n)
    over thinking steps n in {1 ... N}, regularized against a Geometric prior Geom(lambda_prior).
    """
    def __init__(
        self,
        d_model: int = 32,
        hyperspace_dim: int = 8,
        prior_lambda: float = 0.35,
        cum_threshold: float = 0.85,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.prior_lambda = prior_lambda
        self.cum_threshold = cum_threshold
        self.device = device if device is not None else torch.device("cpu")
        self.gate = nn.Sequential(
            nn.Linear(d_model + hyperspace_dim + 1, 32, device=self.device),
            nn.SiLU(),
            nn.Linear(32, 1, device=self.device),
            nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor, u: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """Emits hazard rate lambda_n in (0, 1) for the current deliberation step."""
        feat = torch.cat([z, u, r], dim=-1)
        return torch.clamp(self.gate(feat), 1e-4, 1.0 - 1e-4)

    def compute_step_distribution(self, hazards: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes step probabilities p(n) = lambda_n * prod_{j=1}^{n-1} (1 - lambda_j)
        and cumulative probabilities P(<= n) = 1 - prod_{j=1}^n (1 - lambda_j).
        """
        p_list = []
        unhalted = torch.ones_like(hazards[0])
        for lam in hazards:
            p_n = lam * unhalted
            p_list.append(p_n)
            unhalted = unhalted * (1.0 - lam)
        p_dist = torch.stack(p_list, dim=-1)
        cum_dist = 1.0 - torch.cumprod(1.0 - torch.stack(hazards, dim=-1), dim=-1)
        return p_dist, cum_dist

    def kl_divergence_prior(self, p_dist: torch.Tensor, max_steps: int) -> torch.Tensor:
        """
        Computes KL divergence D_KL(p(n) || Geom(lambda_prior)) across steps.
        """
        steps = torch.arange(1, max_steps + 1, device=p_dist.device, dtype=torch.float32)
        p_geom = self.prior_lambda * ((1.0 - self.prior_lambda) ** (steps - 1.0))
        p_geom = p_geom / p_geom.sum()
        p_geom = p_geom.view(1, 1, -1)
        kl = (p_dist * (torch.log(p_dist + 1e-8) - torch.log(p_geom + 1e-8))).sum(dim=-1)
        return kl.mean()


class UniversalWorldModel(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        hyperspace_dim: int = 8,
        consensus_threshold: float = 0.70,
        max_thinking_steps: int = 8,
        use_dynamic_moe: bool = False,
        moe_instance: Optional[Any] = None,
        use_working_memory: bool = False,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.consensus_threshold = consensus_threshold
        self.max_thinking_steps = max_thinking_steps
        self.device = device if device is not None else torch.device("cpu")

        # 1. Forward Latent Transition Net (Predicts state progression in tangent space)
        self.w_fwd = nn.Sequential(
            nn.Linear(d_model + d_model, 64, device=self.device),
            nn.SiLU(),
            nn.Linear(64, d_model, device=self.device)
        )

        # 2. Inverse Latent Transition Net (Learns causal rollback dynamically)
        self.w_bwd = nn.Sequential(
            nn.Linear(d_model + d_model, 64, device=self.device),
            nn.SiLU(),
            nn.Linear(64, d_model, device=self.device)
        )

        # 3. Riemannian Spherical Attention over Deliberation Trajectories
        self.deliberation_attn = RiemannianSphericalAttention(
            d_model=d_model,
            num_heads=4,
            hyperspace_dim=hyperspace_dim,
            device=self.device
        )
        # Dual-Stream Trajectory Attention with magnitude-modulated phase concentration
        self.dual_stream_attn = DualStreamRiemannianAttention(
            d_model=d_model,
            num_heads=4,
            hyperspace_dim=hyperspace_dim,
            device=self.device
        )

        # 4. Projection to S^{D-1} x R+
        self.to_sphere = nn.Linear(d_model, hyperspace_dim, bias=False, device=self.device)
        self.to_radius = nn.Linear(d_model, 1, device=self.device)

        # 5. Looped Riemannian Transformer for Turing-Complete Sequential Deliberation
        self.deliberation_loop = LoopedRiemannianTransformer(
            d_model=d_model,
            num_heads=4,
            hyperspace_dim=hyperspace_dim,
            d_ff=64,
            max_loops=max_thinking_steps,
            halting_threshold=consensus_threshold,
            use_dynamic_moe=use_dynamic_moe,
            moe_instance=moe_instance,
            use_working_memory=use_working_memory,
            device=self.device
        )

        # 6. Metacognitive Consensus & Halting Gate (PonderNet Calibrated System 1 vs System 2)
        self.ponder_halting = PonderHalting(
            d_model=d_model,
            hyperspace_dim=hyperspace_dim,
            prior_lambda=0.35,
            cum_threshold=0.85,
            device=self.device
        )
        self.halting_gate = self.ponder_halting.gate

        # 7. Hierarchical Riemannian Predictive Coding (HRPC) Engine
        self.hrpc = HierarchicalRiemannianPredictiveCoding(
            num_layers=3,
            d_model=d_model,
            hyperspace_dim=hyperspace_dim,
            input_dim=d_model,
            device=self.device
        )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Standard nn.Module forward pass."""
        return self.forward_transition(z, c)

    def forward_transition(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Predicts next latent state: z_{t+1} = W_fwd(z_t, c)
        """
        combined = torch.cat([z, c], dim=-1)
        delta = self.w_fwd(combined)
        return z + delta

    def backward_transition(self, z_next: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs previous latent state: z_t = W_bwd(z_{t+1}, c)
        """
        combined = torch.cat([z_next, c], dim=-1)
        delta = self.w_bwd(combined)
        return z_next + delta

    def project_manifold(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Maps continuous hidden state to (u in S^{D-1}, r in R+)
        """
        u = project_to_sphere(self.to_sphere(h))
        r = F.softplus(self.to_radius(h)) + 1.0
        return u, r

    def causal_residual(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        True Pearl-L3 round-trip residual on S^{D-1} x R+.
        d_M(z, W_bwd(W_fwd(z, c), c)) = d_S(u0, ub) + 0.5 * |r0 - rb|
        Returns tensor [B] of geodesic displacements.
        """
        z_fwd = self.forward_transition(z, c)
        z_back = self.backward_transition(z_fwd, c)
        u0, r0 = self.project_manifold(z)
        ub, rb = self.project_manifold(z_back)
        d_angular = geodesic_distance_sphere(u0, ub).squeeze(-1)
        d_radial = torch.abs(r0 - rb).squeeze(-1)
        return d_angular + 0.5 * d_radial

    def compute_causal_inconsistency(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Pure self-supervised causal loss:
        Measures geodesic distance on S^{D-1} x R+ between original state z
        and its counterfactual round-trip reconstruction.
        """
        # Forward prediction
        z_next = self.forward_transition(z, c)
        # Backward rollback
        z_reconstructed = self.backward_transition(z_next, c)

        # Manifold projections
        u_orig, r_orig = self.project_manifold(z)
        u_recon, r_recon = self.project_manifold(z_reconstructed)

        # Riemannian distance on S^{D-1} x R+
        d_angular = geodesic_distance_sphere(u_orig, u_recon)
        d_radial = torch.abs(r_orig - r_recon).squeeze(-1)
        total_causal_surprise = d_angular + 0.5 * d_radial

        return total_causal_surprise.mean()

    def compute_predictive_free_energy(self, z: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes joint Causal Inconsistency and Hierarchical Variational Free Energy.
        Balances top-down prior beliefs and bottom-up sensory predictions.
        """
        # Causal surprise from forward-backward loop
        causal_surprise = self.compute_causal_inconsistency(z, c)

        # Predictive coding free energy
        relaxed_states, traj = self.hrpc.infer_states(z, num_steps=4, step_size=0.05)
        energy_loss, meta = self.hrpc.compute_free_energy(relaxed_states, z)

        total_loss = causal_surprise + 0.5 * energy_loss
        meta["causal_surprise"] = causal_surprise.item()
        meta["total_predictive_energy"] = total_loss.item()
        return total_loss, meta

    def deliberate_and_halting_loop(
        self,
        z_init: torch.Tensor,
        context: torch.Tensor,
        moe_layer: nn.Module = None,
        use_dual_stream: bool = True
    ) -> dict:
        """
        Adaptive Computation Time (ACT) with Riemannian Spherical Trajectory Attention:
        Dynamically chooses between System 1 (1 tick) and System 2 (multi-tick deliberation).
        Maintains an endogenous thinking trajectory attended via RSA or Dual-Stream Attention with SO(2) phase winding.
        """
        z = z_init
        c = context
        thinking_trajectory = []
        traj_latents = [z_init.unsqueeze(1)]
        halted = False
        steps_taken = 0
        unhalted = torch.ones(z_init.size(0), 1, device=z_init.device)
        cum_halting_prob = 0.0

        for step in range(1, self.max_thinking_steps + 1):
            steps_taken = step
            # 1. Forward evolution step
            z_cand = self.forward_transition(z, c)

            # Predictive Coding State Relaxation (balancing top-down prior & candidate observation)
            if self.training and torch.is_grad_enabled():
                with torch.enable_grad():
                    pred_states, _ = self.hrpc.infer_states(z_cand, num_steps=2, step_size=0.04)
                    z_cand = 0.85 * z_cand + 0.15 * pred_states[0]
            else:
                with torch.no_grad():
                    pred_states = self.hrpc.forward_relaxation_step(z_cand, num_steps=2, alpha=0.15)
                    z_cand = 0.85 * z_cand + 0.15 * pred_states[0]

            # 2. RSA Trajectory Attention over historical deliberation steps
            traj_latents.append(z_cand.unsqueeze(1))
            traj_seq = torch.cat(traj_latents, dim=1) # [B, step+1, d_model]
            if use_dual_stream and hasattr(self, "dual_stream_attn"):
                attended_seq, _, ds_meta = self.dual_stream_attn(traj_seq)
                z_cand = z_cand + 0.25 * attended_seq[:, -1, :]
                attn_weights = ds_meta["phase_weights"]
            else:
                attended_seq, attn_weights = self.deliberation_attn(traj_seq)
                z_cand = z_cand + 0.25 * attended_seq[:, -1, :]

            # 3. Optional enrichment through Dendritic MoE
            u, r = self.project_manifold(z_cand)
            if moe_layer is not None:
                moe_out, meta = moe_layer(z_cand, u, r)
                z_cand = z_cand + 0.3 * moe_out
                u, r = self.project_manifold(z_cand)

            # 4. Counterfactual verification (True Pearl-L3 round-trip)
            z_fwd = self.forward_transition(z_cand, c)
            z_recon = self.backward_transition(z_fwd, c)
            u0, r0 = self.project_manifold(z_cand)
            ub, rb = self.project_manifold(z_recon)
            d_angular = geodesic_distance_sphere(u0, ub).squeeze(-1)
            d_radial = torch.abs(r0 - rb).squeeze(-1)
            causal_error = (d_angular + 0.5 * d_radial).mean().item()

            # 5. Measure calibrated internal halting consensus via PonderNet hazard rate
            hazard_rate = self.ponder_halting(z_cand, u, r)
            consensus = hazard_rate.mean().item()
            unhalted = unhalted * (1.0 - hazard_rate)
            cum_halting_prob = (1.0 - unhalted).mean().item()

            thinking_trajectory.append({
                "step": step,
                "consensus": consensus,
                "cum_halting_prob": cum_halting_prob,
                "causal_error": causal_error,
                "u": u.detach().cpu().numpy(),
                "r": r.mean().item(),
                "attn_weights": attn_weights.detach().cpu().numpy()
            })

            # Check calibrated PonderNet stopping condition or consensus threshold
            if (cum_halting_prob >= self.ponder_halting.cum_threshold or consensus >= self.consensus_threshold) and causal_error <= 0.35:
                halted = True
                z = z_cand
                break
            else:
                # Deliberate: feedback causal correction into candidate state
                z = z_cand - 0.2 * (z_recon - z_cand)
                # Soft norm bounding to prevent numerical runaway while preserving native embedding scale
                norm_z = torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-6
                if norm_z.max() > 15.0:
                    z = torch.clamp(norm_z, max=15.0) * (z / norm_z)

        # Final projection and manifold bounding
        norm_z = torch.norm(z, p=2, dim=-1, keepdim=True) + 1e-6
        if norm_z.max() > 15.0:
            z = torch.clamp(norm_z, max=15.0) * (z / norm_z)
        u, r = self.project_manifold(z)

        return {
            "final_latent": z,
            "final_u": u,
            "final_r": r,
            "steps_taken": steps_taken,
            "system_used": "System 1 (Instant Intuition)" if steps_taken == 1 else f"System 2 ({steps_taken} Thinking Ticks)",
            "consensus": consensus,
            "cum_halting_prob": cum_halting_prob,
            "causal_error": causal_error,
            "thinking_trajectory": thinking_trajectory
        }
