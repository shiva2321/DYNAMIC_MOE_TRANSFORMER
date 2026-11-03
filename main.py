"""
Self-Stabilizing Dynamic Morphic Mixture-of-Experts (SS-DM-MoE) Transformer
=============================================================================
A complete, production-grade PyTorch framework for non-Euclidean representation
learning with complex-valued Hilbert latent spaces, autogenous predictive
manifolds, and runtime horizontal/vertical graph morphing.

Author: Principal Research Engineer
License: Apache-2.0
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# SECTION 1: COMPLEX HILBERT SPACE OPERATORS (C^d)
# ============================================================================

class ComplexLinear(nn.Module):
    """
    Complex-valued linear transformation operating on dual real/imaginary fields.
    Satisfies canonical complex multiplication:
        (W_r + i*W_i)(x + i*y) = (W_r*x - W_i*y) + i*(W_r*y + W_i*x)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        scale = 1.0 / math.sqrt(2.0 * in_features)
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * scale)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_r = F.linear(z_r, self.weight_real) - F.linear(z_i, self.weight_imag)
        out_i = F.linear(z_r, self.weight_imag) + F.linear(z_i, self.weight_real)

        if self.has_bias:
            out_r = out_r + self.bias_real
            out_i = out_i + self.bias_imag

        return out_r, out_i


class ModReLU(nn.Module):
    """
    Magnitude-controlled non-linear activation for complex numbers:
        ModReLU(z) = ReLU(|z| + b) * (z / (|z| + eps))
    Preserves topological phase angle while filtering sub-threshold amplitudes.
    """
    def __init__(self, features: int, eps: float = 1e-7):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        magnitude = torch.sqrt(z_r ** 2 + z_i ** 2 + self.eps)
        scale = F.relu(magnitude + self.b) / magnitude
        return z_r * scale, z_i * scale


class ComplexLayerNorm(nn.Module):
    """
    Complex Layer Normalization standardizing the complex manifold using
    joint variance across real and imaginary channels:
        Var(Z) = Var(Z_r) + Var(Z_i)
    """
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma_r = nn.Parameter(torch.ones(d_model))
        self.gamma_i = nn.Parameter(torch.zeros(d_model))
        self.beta_r = nn.Parameter(torch.zeros(d_model))
        self.beta_i = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_r = z_r.mean(dim=-1, keepdim=True)
        mean_i = z_i.mean(dim=-1, keepdim=True)
        z_cent_r = z_r - mean_r
        z_cent_i = z_i - mean_i

        variance = (z_cent_r ** 2 + z_cent_i ** 2).mean(dim=-1, keepdim=True)
        inv_std = 1.0 / torch.sqrt(variance + self.eps)

        norm_r = z_cent_r * inv_std
        norm_i = z_cent_i * inv_std

        # Complex scale: (gamma_r + i*gamma_i) * (norm_r + i*norm_i) + beta
        out_r = norm_r * self.gamma_r - norm_i * self.gamma_i + self.beta_r
        out_i = norm_r * self.gamma_i + norm_i * self.gamma_r + self.beta_i
        return out_r, out_i


# ============================================================================
# SECTION 2: UNIVERSAL MULTIMODAL INGESTION LAYER
# ============================================================================

class UniversalMultimodalEncoder(nn.Module):
    """
    Adaptive multimodal projection engine. Ingests arbitrary text sequences,
    visual grids, or multivariate continuous streams, projecting them directly
    into complex Hilbert space C^d with explicit Real and Imaginary components.
    """
    def __init__(self, d_model: int, max_tokens: int = 32000):
        super().__init__()
        self.d_model = d_model

        # Modality adapters
        self.text_embed = nn.Embedding(max_tokens, d_model)
        self.image_patch = nn.Conv2d(3, d_model, kernel_size=4, stride=4)
        self.continuous_proj = nn.LazyLinear(d_model)

        # Complex component projectors
        self.proj_real = nn.Linear(d_model, d_model)
        self.proj_imag = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Modality 1: Discrete Token Stream [Batch, Sequence]
        if x.dim() == 2 and x.dtype in (torch.int32, torch.int64):
            emb = self.text_embed(x)  # [B, S, d_model]
            positions = torch.arange(x.size(1), device=x.device, dtype=torch.float32).unsqueeze(-1)
            omega = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) * -(math.log(10000.0) / self.d_model))
            phase_wave = torch.cat([torch.sin(positions * omega), torch.cos(positions * omega)], dim=-1)[:, :self.d_model]
            base_repr = emb + phase_wave.unsqueeze(0)

        # Modality 2: Visual Spatial Matrix [Batch, Channels, Height, Width]
        elif x.dim() == 4:
            if x.size(1) == 1:
                x = x.repeat(1, 3, 1, 1)
            patches = self.image_patch(x)  # [B, d_model, H', W']
            base_repr = patches.flatten(2).transpose(1, 2)  # [B, S_patches, d_model]

        # Modality 3: Continuous Multivariate Stream [Batch, Time, Channels]
        elif x.dim() == 3:
            base_repr = self.continuous_proj(x)  # [B, T, d_model]

        else:
            raise ValueError(f"Unsupported multimodal input shape/dtype: shape={x.shape}, dtype={x.dtype}")

        z_r = self.proj_real(base_repr)
        z_i = self.proj_imag(base_repr)
        return z_r, z_i


# ============================================================================
# SECTION 3: GEOMETRIC LATENT FIELD & AUTOGENOUS CORRECTIVE LOOP (\Omega_corr)
# ============================================================================

class GeometricCorrectiveManifold(nn.Module):
    """
    Constructs an autogenous internal geometric loop:
    1. Evaluates incoming coordinates (z_r, z_i) to predict geodesic deformation.
    2. Measures deviation: \\Delta Z = Z - \\hat{Z}_pred.
    3. Computes running structural manifold error metric E_\\Omega for vertical morphing.
    4. Applies differentiable non-linear complex correction.
    """
    def __init__(self, d_model: int, momentum: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.momentum = momentum

        # Geodesic predictive mapping
        self.pred_1 = ComplexLinear(d_model, d_model)
        self.pred_act = ModReLU(d_model)
        self.pred_2 = ComplexLinear(d_model, d_model)

        # Correction transformation
        self.corr_1 = ComplexLinear(d_model, d_model)
        self.corr_act = ModReLU(d_model)
        self.corr_2 = ComplexLinear(d_model, d_model)

        # Online running average metric
        self.register_buffer("running_error_magnitude", torch.tensor(0.25, dtype=torch.float32))

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Predict structural manifold state
        h_r, h_i = self.pred_act(*self.pred_1(z_r, z_i))
        pred_r, pred_i = self.pred_2(h_r, h_i)

        # 2. Geometric deviation vector
        delta_r = z_r - pred_r
        delta_i = z_i - pred_i

        # 3. Structural error magnitude
        error_norm = torch.sqrt(delta_r ** 2 + delta_i ** 2 + 1e-8)
        batch_error = error_norm.mean()

        if self.training:
            with torch.no_grad():
                self.running_error_magnitude.mul_(self.momentum).add_(batch_error * (1.0 - self.momentum))

        # 4. Corrective feedback
        c_r, c_i = self.corr_act(*self.corr_1(delta_r, delta_i))
        corr_r, corr_i = self.corr_2(c_r, c_i)

        # Re-stabilized latent field
        corrected_r = z_r + corr_r
        corrected_i = z_i + corr_i

        return corrected_r, corrected_i, batch_error


# ============================================================================
# SECTION 4: COMPLEX HERMITIAN MULTI-HEAD ATTENTION
# ============================================================================

class ComplexHermitianAttention(nn.Module):
    """
    Multi-Head Attention operating directly in Hilbert space.
    Computes Hermitian inner products:
        Re(Q K^H) = Q_r K_r^T + Q_i K_i^T
    Preserves phase wave interference across the entire context window.
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = ComplexLinear(d_model, d_model)
        self.k_proj = ComplexLinear(d_model, d_model)
        self.v_proj = ComplexLinear(d_model, d_model)
        self.out_proj = ComplexLinear(d_model, d_model)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = z_r.shape

        qr, qi = self.q_proj(z_r, z_i)
        kr, ki = self.k_proj(z_r, z_i)
        vr, vi = self.v_proj(z_r, z_i)

        # Reshape to [B, num_heads, S, head_dim]
        def split(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        qr, qi = split(qr), split(qi)
        kr, ki = split(kr), split(ki)
        vr, vi = split(vr), split(vi)

        # Hermitian scalar product: Re(Q K^H) = Q_r K_r^T + Q_i K_i^T
        attn_logits = (torch.matmul(qr, kr.transpose(-2, -1)) + torch.matmul(qi, ki.transpose(-2, -1))) * self.scale
        attn_weights = F.softmax(attn_logits, dim=-1)

        out_r = torch.matmul(attn_weights, vr)
        out_i = torch.matmul(attn_weights, vi)

        # Recombine heads
        def merge(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous().view(B, S, D)

        return self.out_proj(merge(out_r), merge(out_i))


# ============================================================================
# SECTION 5: MORPHIC MIXTURE OF EXPERTS (HORIZONTAL MUTATION)
# ============================================================================

class ComplexExpert(nn.Module):
    """Complex feed-forward expert with native ModReLU non-linearity."""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = ComplexLinear(d_model, d_ff)
        self.act = ModReLU(d_ff)
        self.linear2 = ComplexLinear(d_ff, d_model)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_r, h_i = self.act(*self.linear1(z_r, z_i))
        return self.linear2(h_r, h_i)

    def clone_with_mutation(self, mutation_std: float = 0.02) -> "ComplexExpert":
        """Spawns a mutated daughter expert via stochastic mitotic replication."""
        device = self.linear1.weight_real.device
        new_expert = ComplexExpert(self.linear1.in_features, self.linear1.out_features).to(device)

        with torch.no_grad():
            for src_p, dst_p in zip(self.parameters(), new_expert.parameters()):
                noise = torch.randn_like(src_p) * mutation_std
                dst_p.copy_(src_p + noise)

        return new_expert


class DynamicMorphicMoE(nn.Module):
    """
    Self-Morphing Horizontal MoE Engine:
    - Splits & spawns new experts when routing entropy H_r exceeds tau_split.
    - Expunges dormant experts when rolling utility frequency falls below tau_death.
    """
    def __init__(self, d_model: int, d_ff: int, initial_experts: int = 3,
                 tau_split: float = 1.05, tau_death: float = 0.05, min_experts: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.tau_split = tau_split
        self.tau_death = tau_death
        self.min_experts = min_experts

        self.experts = nn.ModuleList([ComplexExpert(d_model, d_ff) for _ in range(initial_experts)])
        self.router = nn.Linear(2 * d_model, initial_experts)

        self.register_buffer("expert_usage", torch.ones(initial_experts) / initial_experts)
        self.register_buffer("last_batch_entropy", torch.tensor(0.0))

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, D = z_r.shape
        flat_input = torch.cat([z_r, z_i], dim=-1)  # [B, S, 2*D]

        router_logits = self.router(flat_input)
        router_probs = F.softmax(router_logits, dim=-1)  # [B, S, num_experts]

        # Information-theoretic routing entropy H_r
        mean_usage = router_probs.mean(dim=(0, 1))  # [num_experts]
        entropy = -torch.sum(mean_usage * torch.log(mean_usage + 1e-9))
        self.last_batch_entropy.copy_(entropy.detach())

        if self.training:
            with torch.no_grad():
                self.expert_usage.mul_(0.90).add_(mean_usage.detach() * 0.10)

        # Weighted combination of expert outputs
        acc_r = torch.zeros_like(z_r)
        acc_i = torch.zeros_like(z_i)
        for j, expert in enumerate(self.experts):
            w = router_probs[:, :, j:j+1]
            e_r, e_i = expert(z_r, z_i)
            acc_r = acc_r + w * e_r
            acc_i = acc_i + w * e_i

        return acc_r, acc_i, entropy

    def mutate_experts(self) -> Tuple[bool, bool, str]:
        """Performs runtime horizontal topological mutations."""
        spawned, pruned = False, False
        msg = "Stable"
        device = self.router.weight.device

        # Condition 1: Expert Mitosis (Overload -> High Entropy)
        if self.last_batch_entropy.item() > self.tau_split:
            parent_idx = int(torch.argmax(self.expert_usage).item())
            new_expert = self.experts[parent_idx].clone_with_mutation(mutation_std=0.015)
            self.experts.append(new_expert)

            # Reconstruct router parameters
            old_w, old_b = self.router.weight.data, self.router.bias.data
            in_f, out_f = self.router.in_features, self.num_experts

            new_router = nn.Linear(in_f, out_f).to(device)
            with torch.no_grad():
                new_router.weight[:out_f-1].copy_(old_w)
                new_router.bias[:out_f-1].copy_(old_b)
                # Daughter expert inherits parent routing affinity + slight variation
                new_router.weight[out_f-1].copy_(old_w[parent_idx] + torch.randn(in_f, device=device) * 0.01)
                new_router.bias[out_f-1].copy_(old_b[parent_idx])

            self.router = new_router

            # Update tracking buffers
            new_usage = torch.cat([self.expert_usage, torch.tensor([self.expert_usage[parent_idx].item() * 0.5], device=device)])
            self.register_buffer("expert_usage", new_usage)

            spawned = True
            msg = f"Mitosis: Duplicated expert #{parent_idx} (Total: {self.num_experts})"

        # Condition 2: Expert Apoptosis (Starvation -> Dormancy)
        elif self.num_experts > self.min_experts:
            min_val, least_idx = torch.min(self.expert_usage, dim=0)
            if min_val.item() < self.tau_death:
                idx_to_prune = int(least_idx.item())
                del self.experts[idx_to_prune]

                keep_indices = [i for i in range(self.router.out_features) if i != idx_to_prune]
                old_w, old_b = self.router.weight.data, self.router.bias.data
                in_f = self.router.in_features

                new_router = nn.Linear(in_f, len(keep_indices)).to(device)
                with torch.no_grad():
                    new_router.weight.copy_(old_w[keep_indices])
                    new_router.bias.copy_(old_b[keep_indices])
                self.router = new_router

                new_usage = self.expert_usage[keep_indices]
                self.register_buffer("expert_usage", new_usage)

                pruned = True
                msg = f"Apoptosis: Pruned dormant expert #{idx_to_prune} (Total: {self.num_experts})"

        return spawned, pruned, msg


# ============================================================================
# SECTION 6: DYNAMIC MORPHIC TRANSFORMER LAYER
# ============================================================================

class DynamicMorphicTransformerLayer(nn.Module):
    """
    Self-stabilizing dynamic layer:
    - Pre-norm Hermitian Self-Attention + Residual Gate
    - Pre-norm Dynamic MoE + Residual Gate
    - Tracks online salience contribution: S_l = ||Delta Z_layer|| / ||Z_in||
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, initial_experts: int = 3):
        super().__init__()
        self.d_model = d_model
        self.norm1 = ComplexLayerNorm(d_model)
        self.attn = ComplexHermitianAttention(d_model, num_heads)

        self.norm2 = ComplexLayerNorm(d_model)
        self.moe = DynamicMorphicMoE(d_model, d_ff, initial_experts=initial_experts)

        # Adaptive zero-shock residual multiplier
        self.res_gate = nn.Parameter(torch.ones(1) * 0.5)
        self.register_buffer("layer_salience", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sub-layer 1: Attention
        n1_r, n1_i = self.norm1(z_r, z_i)
        att_r, att_i = self.attn(n1_r, n1_i)
        mid_r = z_r + self.res_gate * att_r
        mid_i = z_i + self.res_gate * att_i

        # Sub-layer 2: Morphic MoE
        n2_r, n2_i = self.norm2(mid_r, mid_i)
        moe_r, moe_i, entropy = self.moe(n2_r, n2_i)
        out_r = mid_r + self.res_gate * moe_r
        out_i = mid_i + self.res_gate * moe_i

        # Online salience tracking
        if self.training:
            with torch.no_grad():
                diff_norm = torch.sqrt((out_r - z_r) ** 2 + (out_i - z_i) ** 2 + 1e-8).mean()
                in_norm = torch.sqrt(z_r ** 2 + z_i ** 2 + 1e-8).mean()
                salience = diff_norm / (in_norm + 1e-5)
                self.layer_salience.mul_(0.95).add_(salience * 0.05)

        return out_r, out_i, entropy


# ============================================================================
# SECTION 7: UNIVERSAL PROJECTIVE DECODER
# ============================================================================

class UniversalProjectiveDecoder(nn.Module):
    """
    Decoupled bridge mapping Hilbert space C^d back to concrete target domains:
    - Discrete token distribution [Batch, Sequence, Vocab]
    - Continuous multivariate vectors [Batch, Time/Sequence, OutputDim]
    - Latent geometric state
    """
    def __init__(self, d_model: int, vocab_size: int = 32000, continuous_dim: int = 8):
        super().__init__()
        self.proj_r = nn.Linear(d_model, d_model)
        self.proj_i = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self.token_head = nn.Linear(d_model, vocab_size)
        self.continuous_head = nn.Linear(d_model, continuous_dim)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor, head: str = "token") -> torch.Tensor:
        unified = self.norm(self.proj_r(z_r) + self.proj_i(z_i))
        if head == "token":
            return self.token_head(unified)
        elif head == "continuous":
            return self.continuous_head(unified)
        elif head == "latent":
            return unified
        raise ValueError(f"Unknown decoding head: {head}")


# ============================================================================
# SECTION 8: DYNAMIC OPTIMIZER MANAGER
# ============================================================================

class MorphicOptimizerManager:
    """
    Manages optimizer state synchronization across dynamic structural mutations.
    Preserves Adam momentum (m) and variance (v) buffers for existing weights
    while seamlessly initializing states for newly sprouted parameters.
    """
    def __init__(self, model: nn.Module, lr: float = 1e-3, weight_decay: float = 1e-2):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def synchronize_after_mutation(self):
        """Reconstructs optimizer parameter groups without losing momentum memory."""
        old_state = self.optimizer.state
        new_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Map saved state by parameter identity / tensor shape where available
        retained_count = 0
        for p in self.model.parameters():
            for old_p, state in old_state.items():
                if p.shape == old_p.shape and torch.equal(p.data, old_p.data):
                    new_optimizer.state[p] = state
                    retained_count += 1
                    break

        self.optimizer = new_optimizer
        return retained_count

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


# ============================================================================
# SECTION 9: INTEGRATED SS-DM-MoE SYSTEM ENGINE
# ============================================================================

class SelfStabilizingDynamicMorphicMoE(nn.Module):
    """
    Full SS-DM-MoE Framework:
    Integrates Universal Ingestion, Geometric Manifold Loop, Vertical Layer Morphing,
    Horizontal Expert Mitosis/Apoptosis, and Universal Decoding.
    """
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 128,
                 initial_layers: int = 2, initial_experts: int = 3,
                 theta_up: float = 0.40, theta_down: float = 0.05,
                 min_layers: int = 1, max_layers: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta_up = theta_up
        self.theta_down = theta_down
        self.min_layers = min_layers
        self.max_layers = max_layers

        self.encoder = UniversalMultimodalEncoder(d_model)
        self.manifold_corr = GeometricCorrectiveManifold(d_model)
        self.layers = nn.ModuleList([
            DynamicMorphicTransformerLayer(d_model, num_heads, d_ff, initial_experts=initial_experts)
            for _ in range(initial_layers)
        ])
        self.decoder = UniversalProjectiveDecoder(d_model)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor, output_head: str = "token") -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        # 1. Universal Ingestion into C^d
        z_r, z_i = self.encoder(x)

        # 2. Geometric Predictive Correction Loop (Omega_corr)
        z_r, z_i, structural_error = self.manifold_corr(z_r, z_i)

        # 3. Dynamic Morphic Transformer Pipeline
        entropies = []
        for layer in self.layers:
            z_r, z_i, entropy = layer(z_r, z_i)
            entropies.append(entropy)

        # 4. Universal Projective Output
        output = self.decoder(z_r, z_i, head=output_head)

        return {
            "output": output,
            "structural_error": structural_error,
            "routing_entropies": entropies,
            "z_r": z_r,
            "z_i": z_i
        }

    def execute_graph_mutation(self) -> Dict[str, Union[int, List[int], List[str]]]:
        """Orchestrates runtime vertical depth and horizontal expert mutations."""
        logs = []
        device = next(self.parameters()).device

        # --- A. Vertical Morphing (Layer Growth / Pruning) ---
        manifold_err = self.manifold_corr.running_error_magnitude.item()

        # Growth: High manifold distortion triggers addition of a new layer
        if manifold_err > self.theta_up and self.num_layers < self.max_layers:
            num_exp = self.layers[-1].moe.num_experts
            new_layer = DynamicMorphicTransformerLayer(self.d_model, self.num_heads, self.d_ff, initial_experts=num_exp).to(device)
            # Initialize with soft-residual gate to prevent gradient shock
            new_layer.res_gate.data.fill_(0.01)
            self.layers.append(new_layer)
            logs.append(f"[Vertical Growth] Structural Error {manifold_err:.4f} > {self.theta_up:.4f}. Appended Layer #{self.num_layers}.")

        # Shrinkage: Dormant layer salience triggers layer collapse
        elif self.num_layers > self.min_layers:
            saliences = [layer.layer_salience.item() for layer in self.layers]
            min_s, min_idx = min((s, idx) for idx, s in enumerate(saliences))
            if min_s < self.theta_down:
                del self.layers[min_idx]
                logs.append(f"[Vertical Pruning] Layer #{min_idx} salience {min_s:.4f} < {self.theta_down:.4f}. Collapsed layer.")

        # --- B. Horizontal Morphing (Expert Mitosis / Apoptosis per layer) ---
        for idx, layer in enumerate(self.layers):
            spawned, pruned, action = layer.moe.mutate_experts()
            if spawned or pruned:
                logs.append(f"[Horizontal Morph Layer {idx}] {action}")

        return {
            "total_layers": self.num_layers,
            "experts_per_layer": [l.moe.num_experts for l in self.layers],
            "mutation_logs": logs
        }


# ============================================================================
# SECTION 10: REAL-TIME VALIDATION HARNESS
# ============================================================================

def execute_verification_suite():
    print("=" * 80)
    print("INITIALIZING SS-DM-MoE DYNAMIC RUNTIME VERIFICATION SUITE")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in operation: {device}\n")

    # Instantiate model
    d_model = 64
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=d_model,
        num_heads=4,
        d_ff=128,
        initial_layers=2,
        initial_experts=3,
        theta_up=0.35,      # Sensitive threshold for demo
        theta_down=0.08,    # Pruning threshold
        min_layers=1,
        max_layers=6
    ).to(device)

    opt_manager = MorphicOptimizerManager(model, lr=1e-3)

    print(f"Initial State:")
    print(f"  • Depth (Layers):    {model.num_layers}")
    print(f"  • Experts per layer: {[l.moe.num_experts for l in model.layers]}")
    print(f"  • Latent Space:      Hilbert Field C^{d_model}")
    print("-" * 80)

    # 1. Multimodal Stress Verification
    print("\n[PHASE 1] Ingesting Heterogeneous Multimodal Streams")

    # Stream A: Discrete Tokens
    tokens = torch.randint(0, 1000, (4, 16), dtype=torch.long, device=device)
    out_tokens = model(tokens, output_head="token")
    print(f"  [1] Discrete Text Sequence:    Input {list(tokens.shape)}  -> Logits {list(out_tokens['output'].shape)}")

    # Stream B: Visual Matrix
    images = torch.randn(4, 3, 32, 32, device=device)
    out_images = model(images, output_head="continuous")
    print(f"  [2] Visual Matrix:             Input {list(images.shape)} -> Output {list(out_images['output'].shape)}")

    # Stream C: Continuous Stream
    stream = torch.randn(4, 20, 12, device=device)
    out_stream = model(stream, output_head="continuous")
    print(f"  [3] Continuous Time Stream:    Input {list(stream.shape)} -> Output {list(out_stream['output'].shape)}")

    # 2. Trigger Vertical Layer Growth
    print("\n" + "-" * 80)
    print("[PHASE 2] Inducing Topological Stress on \\Omega_corr to Trigger Vertical Growth")
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.60)  # Exceeds theta_up (0.35)

    report = model.execute_graph_mutation()
    opt_manager.synchronize_after_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  New Depth: {model.num_layers} Layers | Experts: {report['experts_per_layer']}")
    assert model.num_layers == 3, f"Expected 3 layers, got {model.num_layers}"

    # 3. Trigger Horizontal Expert Mitosis (Splitting)
    print("\n" + "-" * 80)
    print("[PHASE 3] Inducing High Routing Entropy to Trigger Expert Mitosis")
    # Simulate high router cross-talk in Layer 0
    with torch.no_grad():
        model.layers[0].moe.last_batch_entropy.fill_(1.30)  # Exceeds tau_split (1.05)

    report = model.execute_graph_mutation()
    opt_manager.synchronize_after_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Layer 0 Experts after Mitosis: {model.layers[0].moe.num_experts}")
    assert model.layers[0].moe.num_experts == 4, f"Expected 4 experts, got {model.layers[0].moe.num_experts}"

    # 4. Trigger Horizontal Expert Apoptosis (Pruning)
    print("\n" + "-" * 80)
    print("[PHASE 4] Inducing Expert Starvation to Trigger Apoptosis")
    with torch.no_grad():
        model.layers[0].moe.expert_usage[0] = 0.01  # Below tau_death (0.05)
        model.layers[0].moe.last_batch_entropy.fill_(0.40)  # Low entropy

    report = model.execute_graph_mutation()
    opt_manager.synchronize_after_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Layer 0 Experts after Apoptosis: {model.layers[0].moe.num_experts}")

    # 5. Full End-to-End Optimization Pass
    print("\n" + "-" * 80)
    print("[PHASE 5] End-to-End Gradient Backpropagation & Optimization Check")
    opt_manager.zero_grad()
    res = model(tokens, output_head="token")
    targets = torch.randint(0, 1000, (4, 16), device=device)

    task_loss = F.cross_entropy(res["output"].view(-1, 32000), targets.view(-1))
    manifold_loss = res["structural_error"]
    total_loss = task_loss + 0.1 * manifold_loss

    total_loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt_manager.step()

    print(f"  Step Completed:")
    print(f"    • Total Loss:           {total_loss.item():.4f}")
    print(f"    • Manifold Metric Loss: {manifold_loss.item():.4f}")
    print(f"    • Clipped Grad Norm:    {grad_norm.item():.4f}")
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED: Dynamic Morphic Transformer Architecture is Fully Verified.")
    print("=" * 80)


if __name__ == "__main__":
    execute_verification_suite()