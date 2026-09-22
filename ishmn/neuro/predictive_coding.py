"""
Hierarchical Riemannian Predictive Coding (HRPC) on Manifolds S^{D-1} x R+.

Grounded in:
1. Karl Friston's Variational Free Energy Principle (FEP).
2. Continuous Attractor Neural Networks (CANNs) & Neural Manifold Geometry.
3. Two-Compartment Cortical Pyramidal Neurons (Apical Prior vs Basal Sensory).
4. Joint Embedding Predictive Architecture (JEPA) with Manifold VICReg Regularization.
5. Theta-Gamma Oscillatory Clocking & Self-Organized Criticality (Branching Ratio ~ 1.0).

Completely domain-agnostic: Operates on any input modality (continuous trajectories,
high-dimensional dense latents, or discrete sequence embeddings).
"""

import math
from typing import Dict, Any, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from ishmn.geometry import (
    project_to_sphere,
    geodesic_distance_sphere,
    exp_map_sphere
)


class DendriticCompartment(nn.Module):
    """
    Two-compartment pyramidal neuron model:
    - Basal Compartment: Integrates bottom-up feedforward sensory inputs.
    - Apical Compartment: Integrates top-down contextual priors, predictive feedback,
      and associative memory signals via local non-Hebbian NMDA coincidence detection.
    - Somatic Integration: Combines basal and apical compartments, computing local
      prediction error and modulating firing output.
    """
    def __init__(self, d_model: int, d_apical: int, device: torch.device = None):
        super().__init__()
        self.d_model = d_model
        self.device = device if device is not None else torch.device("cpu")

        # Basal feedforward pathway (linear AMPA-like reception)
        self.w_basal = nn.Linear(d_model, d_model, bias=False, device=self.device)

        # Apical feedback pathway (contextual NMDA plateau detector)
        self.w_apical = nn.Linear(d_apical, d_model, bias=False, device=self.device)
        self.nmda_threshold = nn.Parameter(torch.tensor(0.5, device=self.device))
        self.apical_gain = nn.Parameter(torch.tensor(1.0, device=self.device))

        # Somatic integration gate
        self.soma_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, device=self.device),
            nn.SiLU(),
            nn.Linear(d_model, d_model, device=self.device)
        )

    def forward(self, basal_input: torch.Tensor, apical_input: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            soma_activation: Integrated somatic potential.
            dendritic_discrepancy: Difference between bottom-up and top-down potentials.
        """
        # 1. Basal drive (feedforward)
        v_basal = self.w_basal(basal_input)

        # 2. Apical drive (feedback prior with non-linear NMDA plateau gating)
        if apical_input is not None:
            raw_apical = self.w_apical(apical_input)
            # NMDA plateau: non-linear sigmoidal amplification past threshold
            plateau = torch.sigmoid(raw_apical - self.nmda_threshold) * self.apical_gain
            v_apical = raw_apical * (1.0 + plateau)
        else:
            v_apical = torch.zeros_like(v_basal)

        # 3. Dendritic discrepancy (prediction error at cellular compartment level)
        dendritic_discrepancy = v_basal - v_apical

        # 4. Somatic integration
        combined = torch.cat([v_basal, v_apical], dim=-1)
        soma_activation = self.soma_gate(combined)

        return soma_activation, dendritic_discrepancy


class RiemannianPredictiveCodingLayer(nn.Module):
    """
    Single layer in the Hierarchical Riemannian Predictive Coding hierarchy.
    Maintains a latent representation h_l, projected to (u_l in S^{D-1}, r_l in R+).
    Computes top-down predictions of lower layers, geodesic errors in tangent space,
    and precision-weighted free energy.
    """
    def __init__(
        self,
        level_idx: int,
        d_model: int,
        hyperspace_dim: int,
        d_child: Optional[int] = None,
        device: torch.device = None
    ):
        super().__init__()
        self.level_idx = level_idx
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.d_child = d_child if d_child is not None else d_model
        self.device = device if device is not None else torch.device("cpu")

        # Dendritic compartmental unit
        self.dendrite = DendriticCompartment(
            d_model=d_model,
            d_apical=d_model,
            device=self.device
        )

        # Manifold coordinate projections: h -> (u in S^{D-1}, r in R+)
        self.to_sphere = nn.Linear(d_model, hyperspace_dim, bias=False, device=self.device)
        self.to_radius = nn.Linear(d_model, 1, device=self.device)

        # Top-down generative network: predicts child layer's manifold coordinates from current state
        self.gen_child_sphere = nn.Sequential(
            nn.Linear(d_model, d_model, device=self.device),
            nn.SiLU(),
            nn.Linear(d_model, hyperspace_dim, bias=False, device=self.device)
        )
        self.gen_child_radius = nn.Sequential(
            nn.Linear(d_model, 32, device=self.device),
            nn.SiLU(),
            nn.Linear(32, 1, device=self.device)
        )
        self.gen_child_latent = nn.Linear(d_model, self.d_child, device=self.device)

        # Dynamic precision parameters (confidence / inverse variance)
        # Log-precision parameterized to guarantee positivity: Pi = exp(gamma)
        self.log_precision_sphere = nn.Parameter(torch.tensor(1.0, device=self.device))
        self.log_precision_radius = nn.Parameter(torch.tensor(0.5, device=self.device))

    def project_manifold(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps hidden vector h to unit sphere u in S^{D-1} and radius r in R+."""
        u = project_to_sphere(self.to_sphere(h))
        r = F.softplus(self.to_radius(h)) + 1.0
        return u, r

    def generate_child_prediction(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-down generative model:
        Given current layer's hidden state h_l, generate prior expectation of child:
        mu_{u, l-1} in S^{D-1}, mu_{r, l-1} in R+, and mu_{h, l-1}.
        """
        mu_u = project_to_sphere(self.gen_child_sphere(h))
        mu_r = F.softplus(self.gen_child_radius(h)) + 1.0
        mu_h = self.gen_child_latent(h)
        return mu_u, mu_r, mu_h

    def compute_layer_prediction_error(
        self,
        u_actual: torch.Tensor,
        r_actual: torch.Tensor,
        u_pred: torch.Tensor,
        r_pred: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes precision-weighted prediction errors on Riemannian manifold S^{D-1} x R+.
        Returns:
            error_energy: Scaled free-energy scalar for this layer.
            d_angular: Geodesic distances.
            tangent_error: Error vector in tangent space T_{u_pred} S^{D-1}.
            radial_error: Logarithmic scale error.
        """
        cos_sim = torch.sum(u_actual * u_pred, dim=-1, keepdim=True)
        cos_sim = torch.clamp(cos_sim, -0.9999, 0.9999)

        # Geodesic chordal distance squared on S^{D-1}: 2 * (1 - cos_sim)
        # Mathematically equivalent to ||u - mu||^2.
        # Smooth, convex, gradient is -2 * u_pred, ZERO possibility of acos gradient singularity.
        chordal_dist_sq = 2.0 * (1.0 - cos_sim)

        # Tangent error vector in T_{u_pred} S^{D-1}
        tangent_error = u_actual - cos_sim * u_pred

        # Angular metric for logging / display (detached from autograd graph)
        with torch.no_grad():
            d_angular = torch.acos(cos_sim).squeeze(-1)

        # Logarithmic radial error (hyperbolic scale distance)
        radial_error = torch.log(r_actual + 1e-5) - torch.log(r_pred + 1e-5)  # [B, 1]

        # Precision weights: Pi = exp(clamp(log_pi))
        pi_u = torch.exp(torch.clamp(self.log_precision_sphere, -3.0, 5.0))
        pi_r = torch.exp(torch.clamp(self.log_precision_radius, -3.0, 5.0))

        # Precision-weighted free energy:
        energy_u = 0.5 * pi_u * chordal_dist_sq.mean()
        energy_r = 0.5 * pi_r * (radial_error ** 2).mean()
        complexity = -0.5 * (torch.log(pi_u) + torch.log(pi_r))

        layer_free_energy = energy_u + energy_r + complexity

        return layer_free_energy, d_angular, tangent_error, radial_error.squeeze(-1)


class HierarchicalRiemannianPredictiveCoding(nn.Module):
    """
    Hierarchical Riemannian Predictive Coding (HRPC) Architecture.

    Structure:
    - L layers (l = 0, ..., L-1).
    - Level 0: Sensory interface layer (receives external inputs).
    - Level 1..L-1: Hierarchical latent generative layers.
    - Bi-directional message passing:
      - Top-down: Generative predictions mu_{l-1} = g_l(z_l) flow downward.
      - Bottom-up: Precision-weighted prediction errors epsilon_l flow upward.
    - Fast Inference: Continuous state relaxation minimizing Variational Free Energy F.
    - Slow Learning: Local synaptic plasticity and AdamW optimization.
    - Representation Collapse Protection: Hyperspherical VICReg on S^{D-1}.
    - Criticality Control: Maintains branching ratio sigma ~ 1.0.
    """
    def __init__(
        self,
        num_layers: int = 3,
        d_model: int = 32,
        hyperspace_dim: int = 8,
        input_dim: int = 32,
        vicreg_lambda: float = 0.05,
        criticality_lambda: float = 0.02,
        device: torch.device = None
    ):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.input_dim = input_dim
        self.vicreg_lambda = vicreg_lambda
        self.criticality_lambda = criticality_lambda
        self.device = device if device is not None else torch.device("cpu")

        # Sensory input projector to Layer 0
        self.sensory_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model, device=self.device),
            nn.SiLU(),
            nn.Linear(d_model, d_model, device=self.device)
        )

        # Predictive coding hierarchy
        self.layers = nn.ModuleList([
            RiemannianPredictiveCodingLayer(
                level_idx=l,
                d_model=d_model,
                hyperspace_dim=hyperspace_dim,
                d_child=d_model,
                device=self.device
            )
            for l in range(num_layers)
        ])

        # Top-level prior (unconditional root prior on top layer)
        self.root_prior_u = nn.Parameter(
            project_to_sphere(torch.randn(1, hyperspace_dim, device=self.device))
        )
        self.root_prior_r = nn.Parameter(torch.tensor(2.0, device=self.device))

        # Criticality controller (tunes dynamic gain to keep branching ratio ~ 1.0)
        self.neural_gain = nn.Parameter(torch.tensor(1.0, device=self.device))

    def init_hidden_states(self, batch_size: int) -> List[torch.Tensor]:
        """Initializes latent hidden vectors for all layers."""
        return [
            torch.zeros(batch_size, self.d_model, device=self.device, requires_grad=True)
            for _ in range(self.num_layers)
        ]

    def compute_free_energy(
        self,
        h_states: List[torch.Tensor],
        sensory_input: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Computes Total Variational Free Energy F:
        F = sum_{l=0}^{L-1} F_l + lambda_vic * L_VIC + lambda_crit * L_crit

        Args:
            h_states: List of hidden state tensors [h_0, h_1, ..., h_{L-1}].
            sensory_input: External sensory vector [B, input_dim].
        """
        batch_size = sensory_input.shape[0]
        total_fep = torch.tensor(0.0, device=self.device)
        layer_metrics = []

        # 1. Layer 0 sensory ground truth
        x_encoded = self.sensory_encoder(sensory_input)
        u_sensory = project_to_sphere(self.layers[0].to_sphere(x_encoded))
        r_sensory = F.softplus(self.layers[0].to_radius(x_encoded)) + 1.0

        # Project all layer states to manifold
        manifold_states = []
        for l, h in enumerate(h_states):
            u, r = self.layers[l].project_manifold(h)
            manifold_states.append((u, r))

        # 2. Compute prediction errors across hierarchy
        for l in range(self.num_layers):
            u_curr, r_curr = manifold_states[l]

            if l == self.num_layers - 1:
                # Root layer: compare with unconditional root prior
                root_u = project_to_sphere(self.root_prior_u).expand(batch_size, -1)
                root_r = F.softplus(self.root_prior_r).expand(batch_size, 1) + 1.0
                f_layer, d_ang, tang_err, rad_err = self.layers[l].compute_layer_prediction_error(
                    u_actual=u_curr,
                    r_actual=r_curr,
                    u_pred=root_u,
                    r_pred=root_r
                )
            else:
                # Compare with top-down prediction from layer above (l+1)
                h_above = h_states[l + 1]
                pred_u, pred_r, _ = self.layers[l + 1].generate_child_prediction(h_above)
                f_layer, d_ang, tang_err, rad_err = self.layers[l].compute_layer_prediction_error(
                    u_actual=u_curr,
                    r_actual=r_curr,
                    u_pred=pred_u,
                    r_pred=pred_r
                )

            total_fep = total_fep + f_layer
            layer_metrics.append({
                "level": l,
                "free_energy": f_layer.item(),
                "geodesic_error": d_ang.mean().item(),
                "radial_error": rad_err.abs().mean().item()
            })

        # 3. Layer 0 sensory prediction error:
        # How well h_0 explains the incoming sensory observation x_encoded
        u_0, r_0 = manifold_states[0]
        f_sensory, d_sens, _, _ = self.layers[0].compute_layer_prediction_error(
            u_actual=u_sensory,
            r_actual=r_sensory,
            u_pred=u_0,
            r_pred=r_0
        )
        total_fep = total_fep + f_sensory

        # 4. Manifold VICReg Loss (JEPA-inspired Variance & Covariance regularization on S^{D-1})
        vic_loss = self.compute_vicreg_loss([u for u, _ in manifold_states])

        # 5. Self-Organized Criticality Loss (Branching ratio sigma ~ 1.0)
        crit_loss, branching_ratio = self.compute_criticality_loss(h_states)

        total_loss = total_fep + self.vicreg_lambda * vic_loss + self.criticality_lambda * crit_loss

        return total_loss, {
            "total_free_energy": total_loss.item(),
            "fep_pure": total_fep.item(),
            "sensory_error": d_sens.mean().item(),
            "vicreg_loss": vic_loss.item(),
            "criticality_loss": crit_loss.item(),
            "branching_ratio": branching_ratio,
            "layer_metrics": layer_metrics
        }

    def compute_vicreg_loss(self, sphere_states: List[torch.Tensor]) -> torch.Tensor:
        """
        Variance-Invariance-Covariance Regularization adapted for Hyperspherical Manifolds:
        - Variance penalty: Prevents directional collapse onto a single point.
        - Covariance penalty: Decorrelates hyperspace dimensions, maximizing representational capacity.
        """
        total_vic = torch.tensor(0.0, device=self.device)
        target_std = 0.5

        for u in sphere_states:
            if u.shape[0] <= 1:
                continue
            # Variance term: penalize dimensions with standard deviation < target_std
            std = torch.sqrt(u.var(dim=0) + 1e-4)
            var_penalty = torch.mean(F.relu(target_std - std))

            # Covariance term: penalize off-diagonal correlations
            u_centered = u - u.mean(dim=0)
            cov = (u_centered.T @ u_centered) / (u.shape[0] - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            cov_penalty = (off_diag ** 2).sum() / self.hyperspace_dim

            total_vic = total_vic + var_penalty + cov_penalty

        return total_vic

    def compute_criticality_loss(self, h_states: List[torch.Tensor]) -> Tuple[torch.Tensor, float]:
        """
        Measures the dynamic branching ratio sigma across layers:
        sigma = E[ ||h_{l+1}|| / (||h_l|| + eps) ].
        The critical point is at sigma = 1.0 (edge of chaos: maximal information flow and dynamic range).
        """
        ratios = []
        for l in range(len(h_states) - 1):
            norm_curr = torch.norm(h_states[l], p=2, dim=-1) + 1e-6
            norm_next = torch.norm(h_states[l + 1], p=2, dim=-1)
            ratio = (norm_next / norm_curr).mean()
            ratios.append(ratio)

        if not ratios:
            return torch.tensor(0.0, device=self.device), 1.0

        branching_ratio = torch.stack(ratios).mean()
        # Loss penalizes deviation from critical point 1.0
        crit_loss = (branching_ratio - 1.0) ** 2
        return crit_loss, branching_ratio.item()

    def infer_states(
        self,
        sensory_input: torch.Tensor,
        num_steps: int = 10,
        step_size: float = 0.05,
        adapt_learning_rate: bool = True
    ) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
        """
        Fast Perceptual Inference (Continuous Attractor Relaxation).
        Simulates the continuous descent down the Variational Free Energy landscape.
        Neurons adjust their hidden state h_l to minimize prediction errors before decisions.

        Returns:
            optimal_states: List of relaxed hidden states [h_0, ..., h_{L-1}].
            trajectory: Step-by-step audit of energy minimization.
        """
        batch_size = sensory_input.shape[0]
        # Initial feedforward sweep as starting hypothesis
        h_0 = self.sensory_encoder(sensory_input)
        h_states = [h_0.clone().detach().requires_grad_(True)]
        for l in range(1, self.num_layers):
            # Prior projection from layer below
            h_next = F.silu(self.layers[l].dendrite.w_basal(h_states[l - 1].detach()))
            h_states.append(h_next.clone().detach().requires_grad_(True))

        trajectory = []

        # Relaxation loop (Perceptual Settling)
        for step in range(num_steps):
            # Compute free energy for current state configuration
            energy, meta = self.compute_free_energy(h_states, sensory_input)

            # Compute gradients of free energy with respect to states: grad_h = dF/dh
            grads = torch.autograd.grad(
                energy,
                h_states,
                create_graph=False,
                retain_graph=False
            )

            # Update states via descent on free energy manifold
            new_h_states = []
            with torch.no_grad():
                for l in range(self.num_layers):
                    # Gradient descent step on state
                    g = torch.nan_to_num(grads[l], nan=0.0, posinf=5.0, neginf=-5.0)
                    g = torch.clamp(g, -5.0, 5.0)
                    h_updated = h_states[l] - step_size * g
                    h_updated = h_updated.detach().requires_grad_(True)
                    new_h_states.append(h_updated)

            h_states = new_h_states
            meta["step"] = step
            trajectory.append(meta)

        # Final evaluation of relaxed states
        final_energy, final_meta = self.compute_free_energy(h_states, sensory_input)
        final_meta["step"] = num_steps
        trajectory.append(final_meta)

        return h_states, trajectory

    def forward_relaxation_step(
        self,
        sensory_input: torch.Tensor,
        num_steps: int = 2,
        alpha: float = 0.15
    ) -> List[torch.Tensor]:
        """
        Autograd-free forward relaxation step for real-time inference and token deliberation.
        Balances top-down apical expectation and bottom-up basal sensory inputs
        via closed-form dendritic integration without autograd computation graphs.

        Args:
            sensory_input: Bottom-up sensory input tensor [B, input_dim].
            num_steps: Number of top-down / bottom-up relaxation sweeps.
            alpha: Relaxation blending rate between current state and dendritic integration.

        Returns:
            relaxed_states: List of hidden state tensors [h_0, ..., h_{L-1}].
        """
        # 1. Feedforward hypothesis sweep
        h_0 = self.sensory_encoder(sensory_input)
        h_states = [h_0]
        for l in range(1, self.num_layers):
            h_next = F.silu(self.layers[l].dendrite.w_basal(h_states[l - 1]))
            h_states.append(h_next)

        # 2. Fast forward relaxation across dendritic compartments
        for _ in range(num_steps):
            for l in reversed(range(self.num_layers - 1)):
                # Layer l+1 predicts child layer l
                _, _, mu_h = self.layers[l + 1].generate_child_prediction(h_states[l + 1])
                # Cellular dendritic integration: combine basal drive (h_l) with apical prior (mu_h)
                soma_act, _ = self.layers[l].dendrite(h_states[l], mu_h)
                h_states[l] = (1.0 - alpha) * h_states[l] + alpha * soma_act

        return h_states

    def theta_cycle_step(
        self,
        sensory_input: torch.Tensor,
        phase: float,  # Phase in [0, 2*pi]
        num_fast_steps: int = 5
    ) -> Tuple[List[torch.Tensor], Dict[str, Any]]:
        """
        Theta-Oscillatory Coordinated Inference Step:
        - Phase in [0, pi] (Theta Valley): Sensory ingestion / bottom-up error propagation.
        - Phase in [pi, 2*pi] (Theta Peak): Top-down generative recall / consolidation.
        """
        # Modulation coefficients based on theta rhythm
        sensory_weight = 0.5 * (1.0 + math.cos(phase))       # High at 0, low at pi
        prior_weight = 0.5 * (1.0 - math.cos(phase))         # High at pi, low at 0

        # Adjust layer precision dynamically to reflect oscillatory phase
        with torch.no_grad():
            self.layers[0].log_precision_sphere.data.add_(0.1 * (sensory_weight - 0.5))
            for l in range(1, self.num_layers):
                self.layers[l].log_precision_sphere.data.add_(0.1 * (prior_weight - 0.5))

        states, trajectory = self.infer_states(
            sensory_input=sensory_input,
            num_steps=num_fast_steps,
            step_size=0.08
        )

        return states, {
            "phase": phase,
            "phase_deg": math.degrees(phase),
            "sensory_weight": sensory_weight,
            "prior_weight": prior_weight,
            "final_free_energy": trajectory[-1]["total_free_energy"]
        }

    def train_predictive_step(
        self,
        sensory_input: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        inference_steps: int = 6,
        step_size: float = 0.05
    ) -> Dict[str, Any]:
        """
        Full Hierarchical Predictive Coding training step:
        1. Fast state relaxation to find low-energy hypothesis states h*.
        2. Slow synaptic parameter update minimizing free energy at h*.
        """
        # 1. Inference phase (settle state attractors)
        with torch.enable_grad():
            relaxed_states, trajectory = self.infer_states(
                sensory_input=sensory_input,
                num_steps=inference_steps,
                step_size=step_size
            )

        # 2. Learning phase (update synaptic weights)
        optimizer.zero_grad()
        # Evaluate energy at relaxed state configuration
        relaxed_no_grad = [h.detach() for h in relaxed_states]
        energy, meta = self.compute_free_energy(relaxed_no_grad, sensory_input)

        energy.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=2.0)
        optimizer.step()

        meta["initial_free_energy"] = trajectory[0]["total_free_energy"]
        meta["final_free_energy"] = trajectory[-1]["total_free_energy"]
        meta["energy_reduction"] = meta["initial_free_energy"] - meta["final_free_energy"]

        return meta
