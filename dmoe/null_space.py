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
        track_memory: bool = True,
        track_thinker: bool = False,
        track_embeddings: bool = False,
        freeze_shared_norms_on_update: bool = False,
        min_singular_value: float = 1e-4,
    ):
        self.model = model
        self.energy_threshold = energy_threshold
        self.max_rank_ratio = max_rank_ratio
        self.track_attention = track_attention
        self.track_encoder = track_encoder
        self.track_decoder = track_decoder
        self.track_memory = track_memory
        self.track_thinker = track_thinker
        self.track_embeddings = track_embeddings
        self.freeze_shared_norms_on_update = freeze_shared_norms_on_update
        self.min_singular_value = min_singular_value

        self.monitored_modules: Dict[str, nn.Module] = {}
        self.shared_norms: List[nn.Module] = []
        self.cov_alias: Dict[str, str] = {}
        self._parent_hooks_spec: List[Tuple[str, nn.Module, str]] = []
        self.covariances: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self.bases: Dict[str, torch.Tensor] = {}
        self.p_null: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.is_tracking: bool = True

        self._discover_and_register()

    def _accumulate_cov(self, cov_key: str, tensor: torch.Tensor) -> None:
        x_flat = tensor.reshape(-1, tensor.size(-1)).detach().float()
        if x_flat.size(0) == 0:
            return
        cov = x_flat.t() @ x_flat
        count = x_flat.size(0)
        if cov_key not in self.covariances:
            self.covariances[cov_key] = cov
            self.counts[cov_key] = count
        else:
            self.covariances[cov_key] += cov
            self.counts[cov_key] += count

    def _register_fused_stm_aliases(self, prefix: str, stm_mod: nn.Module) -> None:
        self._parent_hooks_spec.append((prefix, stm_mod, "stm"))
        for proj_name in ("w_k", "w_q", "w_v", "w_g", "w_beta", "w_alpha", "w_k2", "w_v2", "w_beta2"):
            full_name = f"{prefix}.{proj_name}"
            if full_name in self.monitored_modules:
                self.cov_alias[full_name] = f"{prefix}.__x__"

    def _register_fused_hier_aliases(self, prefix: str, hier_mod: nn.Module) -> None:
        self._parent_hooks_spec.append((prefix, hier_mod, "hier_attn"))
        for proj_name in ("q_local", "k_local", "v_local", "q_mid", "q_glob"):
            full_name = f"{prefix}.{proj_name}"
            if full_name in self.monitored_modules:
                self.cov_alias[full_name] = f"{prefix}.__x__"
        for proj_name in ("k_mid", "v_mid"):
            full_name = f"{prefix}.{proj_name}"
            if full_name in self.monitored_modules:
                self.cov_alias[full_name] = f"{prefix}.__mid__"
        for proj_name in ("k_glob", "v_glob"):
            full_name = f"{prefix}.{proj_name}"
            if full_name in self.monitored_modules:
                self.cov_alias[full_name] = f"{prefix}.__mem__"

    def _make_parent_hook(self, prefix: str, kind: str):
        def hook(module: nn.Module, args: Tuple[Any, ...], kwargs: Dict[str, Any], output: Any) -> None:
            if not self.is_tracking or not module.training:
                return
            with torch.no_grad():
                x = args[0] if len(args) > 0 else kwargs.get("x")
                if x is None or not isinstance(x, torch.Tensor):
                    return
                self._accumulate_cov(f"{prefix}.__x__", x)
                if kind == "hier_attn":
                    s = int(getattr(module, "chunk_stride", 0))
                    L = x.size(1)
                    n_full = (L // s) if s > 0 else 0
                    if n_full > 0 and hasattr(module, "mid_pool_proj") and hasattr(module, "mid_bos"):
                        B, _, D = x.shape
                        chunks = x[:, : n_full * s, :].view(B, n_full, s, D)
                        macro = module.mid_pool_proj(chunks.mean(dim=2))
                        macro_with_bos = torch.cat([module.mid_bos.expand(B, 1, D), macro], dim=1)
                        self._accumulate_cov(f"{prefix}.__mid__", macro_with_bos)
                    mem_ctx = args[1] if len(args) > 1 else kwargs.get("memory_context")
                    if mem_ctx is not None and isinstance(mem_ctx, torch.Tensor) and mem_ctx.numel() > 0:
                        self._accumulate_cov(f"{prefix}.__mem__", mem_ctx)

        return hook

    def _discover_and_register(self) -> None:
        """Find target linear/complex-linear/embedding modules and register forward hooks."""
        self.remove_hooks()
        self.monitored_modules.clear()
        self.shared_norms.clear()
        self.cov_alias.clear()
        self._parent_hooks_spec.clear()

        # 1. Encoder modules (legacy SelfStabilizingDynamicMorphicMoE + RealSelfStabilizingDMoE)
        if self.track_encoder:
            if hasattr(self.model, "encoder"):
                enc = self.model.encoder
                if hasattr(enc, "byte_chunker") and hasattr(enc.byte_chunker, "chunk_proj"):
                    self.monitored_modules["encoder.byte_chunker.chunk_proj"] = enc.byte_chunker.chunk_proj
                if hasattr(enc, "unify_real"):
                    self.monitored_modules["encoder.unify_real"] = enc.unify_real
                if hasattr(enc, "unify_imag"):
                    self.monitored_modules["encoder.unify_imag"] = enc.unify_imag
            if hasattr(self.model, "chunk_proj") and isinstance(self.model.chunk_proj, nn.Linear):
                self.monitored_modules["chunk_proj"] = self.model.chunk_proj
        if self.track_embeddings and hasattr(self.model, "byte_embed") and isinstance(self.model.byte_embed, nn.Embedding):
            self.monitored_modules["byte_embed"] = self.model.byte_embed

        # 2. Attention & STM modules in each layer (skip per-expert MoE branches)
        if self.track_attention and hasattr(self.model, "layers"):
            for l_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, "norm1") and isinstance(layer.norm1, nn.LayerNorm):
                    self.shared_norms.append(layer.norm1)
                if hasattr(layer, "norm2") and isinstance(layer.norm2, nn.LayerNorm):
                    self.shared_norms.append(layer.norm2)
                if hasattr(layer, "stm_norm") and isinstance(layer.stm_norm, nn.LayerNorm):
                    self.shared_norms.append(layer.stm_norm)
                if hasattr(layer, "attn"):
                    attn = layer.attn
                    for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        if hasattr(attn, proj_name):
                            self.monitored_modules[f"layers.{l_idx}.attn.{proj_name}"] = getattr(attn, proj_name)
                if getattr(layer, "hier_attn", None) is not None:
                    ha_prefix = f"layers.{l_idx}.hier_attn"
                    for sub_name, sub_mod in layer.hier_attn.named_modules():
                        if isinstance(sub_mod, nn.Linear):
                            self.monitored_modules[f"{ha_prefix}.{sub_name}"] = sub_mod
                    self._register_fused_hier_aliases(ha_prefix, layer.hier_attn)
                else:
                    for proj_name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                        sub = getattr(layer, proj_name, None)
                        if isinstance(sub, nn.Linear):
                            self.monitored_modules[f"layers.{l_idx}.{proj_name}"] = sub
                if getattr(layer, "layer_stm", None) is not None:
                    stm_prefix = f"layers.{l_idx}.layer_stm"
                    for sub_name, sub_mod in layer.layer_stm.named_modules():
                        if isinstance(sub_mod, nn.Linear):
                            self.monitored_modules[f"{stm_prefix}.{sub_name}"] = sub_mod
                    self._register_fused_stm_aliases(stm_prefix, layer.layer_stm)
                if getattr(layer, "stm_gate", None) is not None and isinstance(layer.stm_gate, nn.Linear):
                    self.monitored_modules[f"layers.{l_idx}.stm_gate"] = layer.stm_gate

        # 2b. Dual-timescale memory & thinker (optional, shared across experts)
        if self.track_memory and getattr(self.model, "memory", None) is not None:
            for sub_name, sub_mod in self.model.memory.named_modules():
                if isinstance(sub_mod, nn.Linear):
                    self.monitored_modules[f"memory.{sub_name}"] = sub_mod
                elif isinstance(sub_mod, nn.LayerNorm):
                    self.shared_norms.append(sub_mod)
            if getattr(self.model.memory, "short_term", None) is not None:
                self._register_fused_stm_aliases("memory.short_term", self.model.memory.short_term)
        if self.track_thinker and getattr(self.model, "thinker", None) is not None:
            for sub_name, sub_mod in self.model.thinker.named_modules():
                if isinstance(sub_mod, nn.Linear):
                    self.monitored_modules[f"thinker.{sub_name}"] = sub_mod
                elif isinstance(sub_mod, nn.LayerNorm):
                    self.shared_norms.append(sub_mod)

        # 3. Decoder modules (legacy proj_real/imag; RealDecoder expert_heads are expert-private)
        if self.track_decoder and hasattr(self.model, "decoder"):
            dec = self.model.decoder
            if hasattr(dec, "proj_real"):
                self.monitored_modules["decoder.proj_real"] = dec.proj_real
            if hasattr(dec, "proj_imag"):
                self.monitored_modules["decoder.proj_imag"] = dec.proj_imag
            if hasattr(dec, "final_norm") and isinstance(dec.final_norm, nn.LayerNorm):
                self.shared_norms.append(dec.final_norm)

        # Register forward hooks on non-aliased monitored modules + parent fused modules
        for name, mod in self.monitored_modules.items():
            if name in self.cov_alias:
                continue
            is_complex = isinstance(mod, ComplexLinear)
            is_embed = isinstance(mod, nn.Embedding)
            handle = mod.register_forward_hook(self._make_hook(name, is_complex, is_embed=is_embed))
            self.hooks.append(handle)
        for prefix, p_mod, kind in self._parent_hooks_spec:
            try:
                handle = p_mod.register_forward_hook(self._make_parent_hook(prefix, kind), with_kwargs=True)
            except TypeError:
                p_fn = self._make_parent_hook(prefix, kind)
                handle = p_mod.register_forward_hook(lambda m, args, out, _fn=p_fn: _fn(m, args, {}, out))
            self.hooks.append(handle)

    def _make_hook(self, name: str, is_complex: bool, is_embed: bool = False):
        def hook(module: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            if not self.is_tracking or not module.training:
                return
            with torch.no_grad():
                if is_complex:
                    zr, zi = inputs[0], inputs[1]
                    zr_flat = zr.reshape(-1, zr.size(-1)).detach().float()
                    zi_flat = zi.reshape(-1, zi.size(-1)).detach().float()
                    cov = zr_flat.t() @ zr_flat + zi_flat.t() @ zi_flat
                    count = (zr_flat.size(0) + zi_flat.size(0))
                elif is_embed:
                    out_flat = output.reshape(-1, output.size(-1)).detach().float()
                    cov = out_flat.t() @ out_flat
                    count = out_flat.size(0)
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
        src_ranks: Dict[str, int] = {}
        with torch.no_grad():
            for cov_key in list(self.covariances.keys()):
                if self.counts.get(cov_key, 0) == 0:
                    continue

                C = self.covariances[cov_key] / max(self.counts[cov_key], 1)
                d_in = C.size(0)
                max_rank = max(1, int(d_in * self.max_rank_ratio))

                if cov_key in self.bases and self.bases[cov_key] is not None:
                    U_prev = self.bases[cov_key]
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
                    src_ranks[cov_key] = 0
                    self.covariances[cov_key].zero_()
                    self.counts[cov_key] = 0
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
                        self.bases[cov_key] = Q[:, : k_prev + k]
                    else:
                        self.bases[cov_key] = new_basis
                    src_ranks[cov_key] = k
                else:
                    src_ranks[cov_key] = 0

                if cov_key in self.bases:
                    U = self.bases[cov_key]
                    self.p_null[cov_key] = torch.eye(d_in, device=U.device) - U @ U.t()

                self.covariances[cov_key].zero_()
                self.counts[cov_key] = 0

            for name in self.monitored_modules:
                cov_key = self.cov_alias.get(name, name)
                if cov_key in self.bases:
                    self.bases[name] = self.bases[cov_key]
                if cov_key in self.p_null:
                    self.p_null[name] = self.p_null[cov_key]
                if cov_key in src_ranks:
                    ranks_added[name] = src_ranks[cov_key]

        if self.freeze_shared_norms_on_update and self.p_null:
            for norm in self.shared_norms:
                for p in norm.parameters():
                    p.requires_grad_(False)

        return ranks_added

    def project_gradients(self) -> None:
        """
        Project gradients of all monitored modules onto the null space of prior domains.
        Guarantees Delta W @ U_prior = 0 (and for embeddings, Delta E @ P_out_null).
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
            elif isinstance(mod, nn.Embedding):
                if mod.weight.grad is not None:
                    P_c = P.to(device=mod.weight.grad.device, dtype=mod.weight.grad.dtype)
                    mod.weight.grad.data.copy_(mod.weight.grad.data @ P_c)

    @torch.no_grad()
    def snapshot_monitored_weights(self) -> Dict[str, Any]:
        """Snapshot weights/biases of monitored layers for AdamW-safe delta projection."""
        saved: Dict[str, Any] = {}
        for name, mod in self.monitored_modules.items():
            if name not in self.p_null:
                continue
            if isinstance(mod, ComplexLinear):
                saved[name] = (
                    mod.weight_real.data.clone(),
                    mod.weight_imag.data.clone(),
                    mod.bias_real.data.clone() if mod.bias_real is not None else None,
                    mod.bias_imag.data.clone() if mod.bias_imag is not None else None,
                )
            elif isinstance(mod, (nn.Linear, nn.Embedding)):
                bias = getattr(mod, "bias", None)
                saved[name] = (
                    mod.weight.data.clone(),
                    bias.data.clone() if bias is not None else None,
                )
        return saved

    @torch.no_grad()
    def project_parameter_deltas(self, saved: Dict[str, Any]) -> None:
        """Project optimizer parameter updates Delta W onto P_null and restore frozen biases."""
        for name, mod in self.monitored_modules.items():
            if name not in self.p_null or name not in saved:
                continue
            P = self.p_null[name]
            if isinstance(mod, ComplexLinear):
                w0_r, w0_i, b0_r, b0_i = saved[name]
                P_r = P.to(device=mod.weight_real.device, dtype=mod.weight_real.dtype)
                mod.weight_real.data.copy_(w0_r + (mod.weight_real.data - w0_r) @ P_r)
                mod.weight_imag.data.copy_(w0_i + (mod.weight_imag.data - w0_i) @ P_r)
                if b0_r is not None and mod.bias_real is not None:
                    mod.bias_real.data.copy_(b0_r)
                if b0_i is not None and mod.bias_imag is not None:
                    mod.bias_imag.data.copy_(b0_i)
            elif isinstance(mod, (nn.Linear, nn.Embedding)):
                w0, b0 = saved[name]
                P_w = P.to(device=mod.weight.device, dtype=mod.weight.dtype)
                mod.weight.data.copy_(w0 + (mod.weight.data - w0) @ P_w)
                bias = getattr(mod, "bias", None)
                if b0 is not None and bias is not None:
                    bias.data.copy_(b0)

    def remove_hooks(self) -> None:
        """Remove all registered PyTorch hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()
