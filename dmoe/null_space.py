"""
Gradient Projection Memory (GPM) / AlphaEdit null-space gradient projector.

Tracks activation covariances across sequential domains for monitored linear and complex-linear
layers, computes orthonormal projection bases via SVD, and projects optimizer updates
onto the orthogonal complement (null space) of previous domain subspaces:
    P_null = I - U U^T
    Delta W_proj = Delta W @ P_null

This allows all shared backbone parameters to stay continuously trainable (always-plastic,
requires_grad=True) without causing catastrophic forgetting on prior domains.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from dmoe.complex_ops import ComplexLinear


class NullSpaceProjector:
    """
    Continual Learning Null-Space Projector for shared backbone parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        energy_threshold: float = 0.98,
        max_rank_ratio: float = 0.6,
        track_attention: bool = True,
        track_encoder: bool = True,
        track_decoder: bool = True,
        min_singular_value: float = 1e-4,
    ):
        self.model = model
        self.energy_threshold = energy_threshold
        self.max_rank_ratio = max_rank_ratio
        self.track_attention = track_attention
        self.track_encoder = track_encoder
        self.track_decoder = track_decoder
        self.min_singular_value = min_singular_value

        self.monitored_modules: Dict[str, nn.Module] = {}
        self.covariances: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self.bases: Dict[str, torch.Tensor] = {}
        self.p_null: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.is_tracking: bool = True

        self._discover_and_register()

    def _discover_and_register(self) -> None:
        """Find target linear/complex-linear modules and register forward hooks."""
        self.remove_hooks()
        self.monitored_modules.clear()

        # 1. Encoder modules
        if self.track_encoder and hasattr(self.model, "encoder"):
            enc = self.model.encoder
            if hasattr(enc, "byte_chunker") and hasattr(enc.byte_chunker, "chunk_proj"):
                self.monitored_modules["encoder.byte_chunker.chunk_proj"] = enc.byte_chunker.chunk_proj
            if hasattr(enc, "unify_real"):
                self.monitored_modules["encoder.unify_real"] = enc.unify_real
            if hasattr(enc, "unify_imag"):
                self.monitored_modules["encoder.unify_imag"] = enc.unify_imag

        # 2. Attention modules in each layer
        if self.track_attention and hasattr(self.model, "layers"):
            for l_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, "attn"):
                    attn = layer.attn
                    for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        if hasattr(attn, proj_name):
                            self.monitored_modules[f"layers.{l_idx}.attn.{proj_name}"] = getattr(attn, proj_name)

        # 3. Decoder modules
        if self.track_decoder and hasattr(self.model, "decoder"):
            dec = self.model.decoder
            if hasattr(dec, "proj_real"):
                self.monitored_modules["decoder.proj_real"] = dec.proj_real
            if hasattr(dec, "proj_imag"):
                self.monitored_modules["decoder.proj_imag"] = dec.proj_imag

        # Register forward hooks on all monitored modules
        for name, mod in self.monitored_modules.items():
            is_complex = isinstance(mod, ComplexLinear)
            handle = mod.register_forward_hook(self._make_hook(name, is_complex))
            self.hooks.append(handle)

    def _make_hook(self, name: str, is_complex: bool):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not self.is_tracking or not module.training:
                return
            with torch.no_grad():
                if is_complex:
                    # inputs is (z_real, z_imag)
                    zr, zi = inputs[0], inputs[1]
                    zr_flat = zr.reshape(-1, zr.size(-1)).detach().float()
                    zi_flat = zi.reshape(-1, zi.size(-1)).detach().float()
                    cov = zr_flat.t() @ zr_flat + zi_flat.t() @ zi_flat
                    count = (zr_flat.size(0) + zi_flat.size(0))
                else:
                    x = inputs[0]
                    x_flat = x.reshape(-1, x.size(-1)).detach().float()
                    cov = x_flat.t() @ x_flat
                    count = x_flat.size(0)

                if name not in self.covariances:
                    self.covariances[name] = cov
                    self.counts[name] = count
                else:
                    self.covariances[name] += cov
                    self.counts[name] += count

        return hook

    def update_basis_for_domain(self, domain_id: int) -> Dict[str, int]:
        """
        Compute or expand the principal subspace basis using accumulated covariances.
        Returns the rank of the basis added per module.
        """
        ranks_added: Dict[str, int] = {}
        with torch.no_grad():
            for name, mod in self.monitored_modules.items():
                if name not in self.covariances or self.counts.get(name, 0) == 0:
                    continue

                C = self.covariances[name] / max(self.counts[name], 1)
                d_in = C.size(0)
                max_rank = max(1, int(d_in * self.max_rank_ratio))

                if name in self.bases and self.bases[name] is not None:
                    U_prev = self.bases[name]
                    P_prev = torch.eye(d_in, device=C.device) - U_prev @ U_prev.t()
                    C_res = P_prev @ C @ P_prev
                else:
                    U_prev = None
                    C_res = C

                # SVD of covariance
                try:
                    U_mat, S, _ = torch.linalg.svd(C_res, full_matrices=False)
                except torch.cuda.CudaError:
                    # Fallback to CPU SVD if CUDA SVD hits numerical corner case
                    U_mat, S, _ = torch.linalg.svd(C_res.cpu(), full_matrices=False)
                    U_mat = U_mat.to(C.device)
                    S = S.to(C.device)

                valid_mask = S > self.min_singular_value
                if not valid_mask.any():
                    ranks_added[name] = 0
                    continue

                U_mat = U_mat[:, valid_mask]
                S = S[valid_mask]

                total_energy = S.sum()
                if total_energy > 0:
                    cum_energy = S.cumsum(dim=0) / total_energy
                    k = (cum_energy < self.energy_threshold).sum().item() + 1
                else:
                    k = 0

                k_prev = U_prev.size(1) if U_prev is not None else 0
                k_allowed = max(0, max_rank - k_prev)
                k = min(k, k_allowed, U_mat.size(1))

                if k > 0:
                    new_basis = U_mat[:, :k]
                    if U_prev is not None:
                        combined = torch.cat([U_prev, new_basis], dim=1)
                        Q, _ = torch.linalg.qr(combined)
                        self.bases[name] = Q[:, : k_prev + k]
                    else:
                        self.bases[name] = new_basis
                    ranks_added[name] = k
                else:
                    ranks_added[name] = 0

                # Cache projection matrix
                if name in self.bases:
                    U = self.bases[name]
                    self.p_null[name] = torch.eye(d_in, device=U.device) - U @ U.t()

                # Reset accumulator for next domain
                self.covariances[name].zero_()
                self.counts[name] = 0

        return ranks_added

    def project_gradients(self) -> None:
        """
        Project gradients of all monitored modules onto the null space of prior domains.
        Guarantees Delta W @ U_prior = 0.
        """
        for name, mod in self.monitored_modules.items():
            if name not in self.p_null:
                continue
            P = self.p_null[name]
            if isinstance(mod, ComplexLinear):
                if mod.weight_real.grad is not None:
                    P_c = P.to(device=mod.weight_real.grad.device, dtype=mod.weight_real.grad.dtype)
                    mod.weight_real.grad.data.copy_(mod.weight_real.grad.data @ P_c)
                if mod.weight_imag.grad is not None:
                    P_c = P.to(device=mod.weight_imag.grad.device, dtype=mod.weight_imag.grad.dtype)
                    mod.weight_imag.grad.data.copy_(mod.weight_imag.grad.data @ P_c)
                # Freeze bias updates after first domain to prevent uniform output drift
                if mod.bias_real is not None and mod.bias_real.grad is not None:
                    mod.bias_real.grad.zero_()
                if mod.bias_imag is not None and mod.bias_imag.grad is not None:
                    mod.bias_imag.grad.zero_()
            elif isinstance(mod, nn.Linear):
                if mod.weight.grad is not None:
                    P_c = P.to(device=mod.weight.grad.device, dtype=mod.weight.grad.dtype)
                    mod.weight.grad.data.copy_(mod.weight.grad.data @ P_c)
                if mod.bias is not None and mod.bias.grad is not None:
                    mod.bias.grad.zero_()

    def remove_hooks(self) -> None:
        """Remove all registered PyTorch hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()
