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


_PRIVATE_MARKERS = (
    "experts",
    "expert_heads",
    "attn_adapters",
    "stm_adapters",
    "expert_adapters",
    "expert_energy",
    "lora_a",
    "lora_b",
    "shared_base",
)

_READOUT_HEAD_NAMES = (
    "rel_pool_proj",
    "wm_ret_proj",
    "i7_struct_head",
    "expr_proj",
    "expr_head",
    "ctx_head",
    "fsp_head",
    "abstain_head",
)


def _is_private_module_name(name: str) -> bool:
    return any(tok in name for tok in _PRIVATE_MARKERS)


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
        track_thinker: bool = True,
        track_world_model: bool = True,
        track_embeddings: bool = False,
        freeze_shared_norms_on_update: bool = False,
        min_singular_value: float = 1e-4,
        freeze_unprojected_shared: bool = False,
        max_step_rank_ratio: Optional[float] = None,
        step_rank_decay: float = 1.0,
        defer_cov_commit: bool = False,
        exempt_readout_heads: bool = False,
    ):
        self.model = model
        self.exempt_readout_heads = bool(exempt_readout_heads)
        self.freeze_unprojected_shared = bool(freeze_unprojected_shared)
        self.max_step_rank_ratio = (
            float(max_step_rank_ratio) if max_step_rank_ratio is not None else None
        )
        self.step_rank_decay = float(step_rank_decay)
        self._consolidations: int = 0
        self.defer_cov_commit = bool(defer_cov_commit)
        self.energy_threshold = energy_threshold
        self.max_rank_ratio = max_rank_ratio
        self.track_attention = track_attention
        self.track_encoder = track_encoder
        self.track_decoder = track_decoder
        self.track_memory = track_memory
        self.track_thinker = track_thinker
        self.track_world_model = track_world_model
        self.track_embeddings = track_embeddings
        self.freeze_shared_norms_on_update = freeze_shared_norms_on_update
        self.min_singular_value = min_singular_value

        self.monitored_modules: Dict[str, nn.Module] = {}
        self.shared_norms: List[nn.Module] = []
        self.cov_alias: Dict[str, str] = {}
        self._parent_hooks_spec: List[Tuple[str, nn.Module, str]] = []
        self.covariances: Dict[str, torch.Tensor] = {}
        self.counts: Dict[str, int] = {}
        self._pending_cov: Dict[str, torch.Tensor] = {}
        self._pending_counts: Dict[str, int] = {}
        self.bases: Dict[str, torch.Tensor] = {}
        self.p_null: Dict[str, torch.Tensor] = {}
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.is_tracking: bool = True
        self._proj_cache_sig: Optional[Tuple[Tuple[str, int], ...]] = None
        self._dense_buckets: List[Tuple[List[torch.Tensor], torch.Tensor]] = []
        self._factored_items: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._monitored_biases: List[torch.Tensor] = []
        self._complex_items: List[Tuple[str, ComplexLinear]] = []

        self._discover_and_register()
        if hasattr(self.model, "attach_null_space_projector"):
            self.model.attach_null_space_projector(self)
        elif hasattr(self.model, "null_space_projector"):
            self.model.null_space_projector = self

    def _accumulate_cov(self, cov_key: str, tensor: torch.Tensor) -> None:
        x_flat = tensor.reshape(-1, tensor.size(-1)).detach().float()
        if x_flat.size(0) == 0:
            return
        self._accumulate_cov_flat(cov_key, x_flat)

    def _accumulate_cov_flat(self, cov_key: str, x_flat: torch.Tensor, weight_count: Optional[int] = None) -> None:
        count = x_flat.size(0) if weight_count is None else int(weight_count)
        if count <= 0:
            return
        with torch.amp.autocast(x_flat.device.type, enabled=False):
            xf = x_flat.detach().float()
            target_cov = self._pending_cov if self.defer_cov_commit else self.covariances
            target_cnt = self._pending_counts if self.defer_cov_commit else self.counts
            if cov_key not in target_cov:
                target_cov[cov_key] = xf.t() @ xf
                target_cnt[cov_key] = count
            else:
                target_cov[cov_key].addmm_(xf.t(), xf)
                target_cnt[cov_key] += count

    @torch.no_grad()
    def commit_pending(self) -> None:
        """Commit staged forward-pass activation covariances after an accepted update."""
        for k, cov in self._pending_cov.items():
            cnt = self._pending_counts.get(k, 0)
            if cnt <= 0:
                continue
            if k not in self.covariances:
                self.covariances[k] = cov
                self.counts[k] = cnt
            else:
                self.covariances[k] += cov
                self.counts[k] += cnt
        self._pending_cov.clear()
        self._pending_counts.clear()

    def discard_pending(self) -> None:
        """Drop staged forward-pass activation covariances (e.g. on boundary/deferred steps)."""
        self._pending_cov.clear()
        self._pending_counts.clear()

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
            if getattr(self.model, "ms_chunker", None) is not None:
                msc = self.model.ms_chunker
                if hasattr(msc, "chunk_proj") and isinstance(msc.chunk_proj, nn.Linear):
                    self.monitored_modules["ms_chunker.chunk_proj"] = msc.chunk_proj
                if hasattr(msc, "scale_gate") and isinstance(msc.scale_gate, nn.Linear):
                    self.monitored_modules["ms_chunker.scale_gate"] = msc.scale_gate
                if hasattr(msc, "chunk_norm") and isinstance(msc.chunk_norm, nn.LayerNorm):
                    self.shared_norms.append(msc.chunk_norm)
                if self.track_embeddings and hasattr(msc, "byte_embed") and isinstance(msc.byte_embed, nn.Embedding):
                    self.monitored_modules["ms_chunker.byte_embed"] = msc.byte_embed
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
                moe_mod = getattr(layer, "moe", None)
                if moe_mod is not None and getattr(moe_mod, "shared_expert", None) is not None:
                    sh = moe_mod.shared_expert
                    if hasattr(sh, "linear1") and isinstance(sh.linear1, nn.Linear):
                        self.monitored_modules[f"layers.{l_idx}.moe.shared_expert.linear1"] = sh.linear1
                    if hasattr(sh, "linear2") and isinstance(sh.linear2, nn.Linear):
                        self.monitored_modules[f"layers.{l_idx}.moe.shared_expert.linear2"] = sh.linear2

        # 2a. v7 GPM-monitored contextual router & dual meaning projections (excludes NeedHead/experts)
        cr = getattr(self.model, "context_router", None)
        if cr is not None:
            if hasattr(cr, "q_proj") and isinstance(cr.q_proj, nn.Linear):
                self.monitored_modules["context_router.q_proj"] = cr.q_proj
            if hasattr(cr, "ctx_proj") and isinstance(cr.ctx_proj, nn.Linear):
                self.monitored_modules["context_router.ctx_proj"] = cr.ctx_proj
            if hasattr(cr, "route_norm") and isinstance(cr.route_norm, nn.LayerNorm):
                self.shared_norms.append(cr.route_norm)
        for head_name in ("expr_head", "ctx_head", "fsp_head", "abstain_head", "rel_pool_proj", "wm_ret_proj", "i7_struct_head"):
            h_mod = getattr(self.model, head_name, None)
            if isinstance(h_mod, nn.Linear):
                self.monitored_modules[head_name] = h_mod
        if getattr(self.model, "expr_proj", None) is not None:
            for s_idx, sub_m in enumerate(self.model.expr_proj):
                if isinstance(sub_m, nn.Linear):
                    self.monitored_modules[f"expr_proj.{s_idx}"] = sub_m

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
                if _is_private_module_name(sub_name):
                    continue
                if isinstance(sub_mod, nn.Linear):
                    self.monitored_modules[f"thinker.{sub_name}"] = sub_mod
                elif isinstance(sub_mod, nn.LayerNorm):
                    self.shared_norms.append(sub_mod)
        if self.track_world_model and getattr(self.model, "world_model", None) is not None:
            for sub_name, sub_mod in self.model.world_model.named_modules():
                if isinstance(sub_mod, nn.Linear) and any(p.requires_grad for p in sub_mod.parameters()):
                    self.monitored_modules[f"world_model.{sub_name}"] = sub_mod
                elif isinstance(sub_mod, nn.LayerNorm):
                    self.shared_norms.append(sub_mod)
            if getattr(self.model, "wm_imagine_proj", None) is not None and isinstance(self.model.wm_imagine_proj, nn.Linear):
                self.monitored_modules["wm_imagine_proj"] = self.model.wm_imagine_proj

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
                    self._accumulate_cov_flat(name, zr_flat, weight_count=zr_flat.size(0) + zi_flat.size(0))
                    with torch.amp.autocast(zi_flat.device.type, enabled=False):
                        target_cov = self._pending_cov if self.defer_cov_commit else self.covariances
                        target_cov[name].addmm_(zi_flat.detach().float().t(), zi_flat.detach().float())
                elif is_embed:
                    out_flat = output.reshape(-1, output.size(-1)).detach().float()
                    self._accumulate_cov_flat(name, out_flat)
                else:
                    x = inputs[0]
                    x_flat = x.reshape(-1, x.size(-1)).detach().float()
                    self._accumulate_cov_flat(name, x_flat)

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
                if self.exempt_readout_heads and any(
                    cov_key == h or cov_key.startswith(f"{h}.") for h in _READOUT_HEAD_NAMES
                ):
                    self.covariances[cov_key].zero_()
                    self.counts[cov_key] = 0
                    continue

                C = (self.covariances[cov_key] / max(self.counts[cov_key], 1)).float()
                d_in = C.size(0)
                max_rank = max(1, int(d_in * self.max_rank_ratio))

                with torch.amp.autocast(device_type=C.device.type, enabled=False):
                    if cov_key in self.bases and self.bases[cov_key] is not None:
                        U_prev = self.bases[cov_key].float()
                        P_prev = torch.eye(d_in, device=C.device, dtype=torch.float32) - U_prev @ U_prev.t()
                        C_res = P_prev @ C @ P_prev
                    else:
                        U_prev = None
                        C_res = C

                    # SVD of covariance strictly in FP32
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
                    if self.max_step_rank_ratio is not None:
                        eff_step = self.max_step_rank_ratio * (self.step_rank_decay ** self._consolidations)
                        k_step_cap = max(1, int(d_in * eff_step))
                        k_allowed = min(k_allowed, k_step_cap)
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
                if self.exempt_readout_heads and any(
                    name == h or name.startswith(f"{h}.") for h in _READOUT_HEAD_NAMES
                ):
                    continue
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
            if getattr(self.model, "wm_imagine_gate", None) is not None and isinstance(
                self.model.wm_imagine_gate, nn.Parameter
            ):
                self.model.wm_imagine_gate.requires_grad_(False)

        if self.freeze_unprojected_shared and self.p_null:
            # Preflight audit C5: causal convs, output-norm scales, bos/null keys and similar small shared
            # tensors are on the old domains' data path but are neither projected nor norm-frozen.
            covered = set()
            for mod in self.monitored_modules.values():
                for p in mod.parameters():
                    covered.add(id(p))
            for nm, p in self.model.named_parameters():
                if _is_private_module_name(nm):
                    continue
                if id(p) in covered:
                    continue
                p.requires_grad_(False)

        if src_ranks:
            self._consolidations += 1
        return ranks_added

    def _ensure_proj_cache(self) -> None:
        """Group active dense null-space projectors by weight shape for batched bmm."""
        sig = tuple((name, id(p)) for name, p in self.p_null.items() if name in self.monitored_modules)
        if self._proj_cache_sig == sig:
            return

        groups: Dict[Tuple[Tuple[int, ...], str, str], Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        factored: List[Tuple[torch.Tensor, torch.Tensor]] = []
        biases: List[torch.Tensor] = []
        cplx: List[Tuple[str, ComplexLinear]] = []

        for name, mod in self.monitored_modules.items():
            if name not in self.p_null:
                continue
            if isinstance(mod, ComplexLinear):
                cplx.append((name, mod))
                continue
            if not isinstance(mod, (nn.Linear, nn.Embedding)):
                continue
            w = mod.weight
            if isinstance(mod, nn.Linear) and mod.bias is not None:
                biases.append(mod.bias)
            U = self.bases.get(name)
            if U is not None and U.ndim == 2 and U.size(0) > 384 and U.size(1) * 2 < U.size(0):
                factored.append((w, U.to(device=w.device, dtype=w.dtype)))
                continue
            P = self.p_null[name].to(device=w.device, dtype=w.dtype)
            key = (tuple(w.shape), str(w.device), str(w.dtype))
            if key not in groups:
                groups[key] = ([], [])
            w_list, p_list = groups[key]
            w_list.append(w)
            p_list.append(P)

        self._dense_buckets = [(w_list, torch.stack(p_list, dim=0)) for w_list, p_list in groups.values()]
        self._factored_items = factored
        self._monitored_biases = biases
        self._complex_items = cplx
        self._proj_cache_sig = sig

    def _project_right(self, mat: torch.Tensor, name: str) -> torch.Tensor:
        """Compute mat @ P_null; use factored U only when d > 384 and 2*r < d to avoid small-matmul launch overhead."""
        U = self.bases.get(name)
        if U is not None and U.ndim == 2 and U.size(0) > 384 and U.size(1) * 2 < U.size(0):
            U_c = U.to(device=mat.device, dtype=mat.dtype)
            return mat - (mat @ U_c) @ U_c.t()
        P = self.p_null[name].to(device=mat.device, dtype=mat.dtype)
        return mat @ P

    @torch.no_grad()
    def project_gradients(self) -> None:
        """
        Project gradients of all monitored modules onto the null space of prior domains.
        Guarantees Delta W @ U_prior = 0 (and for embeddings, Delta E @ P_out_null).
        """
        if not self.p_null:
            return
        self._ensure_proj_cache()

        for w_list, P_stack in self._dense_buckets:
            grads = [w.grad for w in w_list if w.grad is not None]
            if not grads:
                continue
            if len(grads) == len(w_list):
                P_use = P_stack
            else:
                idx = [i for i, w in enumerate(w_list) if w.grad is not None]
                P_use = P_stack[idx]
            g_proj = torch.bmm(torch.stack(grads, dim=0), P_use)
            torch._foreach_copy_(grads, list(g_proj.unbind(0)))

        for w, U_c in self._factored_items:
            if w.grad is not None:
                g = w.grad
                w.grad.copy_(g - (g @ U_c) @ U_c.t())

        if self._monitored_biases:
            b_grads = [b.grad for b in self._monitored_biases if b.grad is not None]
            if b_grads:
                torch._foreach_zero_(b_grads)

        for name, mod in self._complex_items:
            if mod.weight_real.grad is not None:
                mod.weight_real.grad.copy_(self._project_right(mod.weight_real.grad, name))
            if mod.weight_imag.grad is not None:
                mod.weight_imag.grad.copy_(self._project_right(mod.weight_imag.grad, name))
            if mod.bias_real is not None and mod.bias_real.grad is not None:
                mod.bias_real.grad.zero_()
            if mod.bias_imag is not None and mod.bias_imag.grad is not None:
                mod.bias_imag.grad.zero_()

    @torch.no_grad()
    def snapshot_monitored_weights(self) -> Dict[str, Any]:
        """Snapshot weights/biases of monitored layers for AdamW-safe delta projection."""
        if not self.p_null:
            return {}
        self._ensure_proj_cache()
        bucket_snaps = [torch.stack([w.detach() for w in w_list], dim=0).clone() for w_list, _ in self._dense_buckets]
        fac_snaps = [w.detach().clone() for w, _ in self._factored_items]
        bias_snaps = [b.detach().clone() for b in self._monitored_biases]
        cplx_snaps = [
            (
                name,
                mod.weight_real.detach().clone(),
                mod.weight_imag.detach().clone(),
                mod.bias_real.detach().clone() if mod.bias_real is not None else None,
                mod.bias_imag.detach().clone() if mod.bias_imag is not None else None,
            )
            for name, mod in self._complex_items
        ]
        return {
            "__batched__": True,
            "cache_sig": self._proj_cache_sig,
            "buckets": bucket_snaps,
            "dense_ids": [[id(w) for w in w_list] for w_list, _ in self._dense_buckets],
            "factored": fac_snaps,
            "factored_ids": [id(w) for w, _ in self._factored_items],
            "biases": bias_snaps,
            "bias_ids": [id(b) for b in self._monitored_biases],
            "complex": cplx_snaps,
        }

    @torch.no_grad()
    def project_parameter_deltas(self, saved: Dict[str, Any]) -> None:
        """Project optimizer parameter updates Delta W onto P_null and restore frozen biases."""
        if not saved or not self.p_null:
            return
        if not saved.get("__batched__", False):
            for name, mod in self.monitored_modules.items():
                if name not in self.p_null or name not in saved:
                    continue
                if isinstance(mod, ComplexLinear):
                    w0_r, w0_i, b0_r, b0_i = saved[name]
                    mod.weight_real.data.copy_(w0_r + self._project_right(mod.weight_real.data - w0_r, name))
                    mod.weight_imag.data.copy_(w0_i + self._project_right(mod.weight_imag.data - w0_i, name))
                    if b0_r is not None and mod.bias_real is not None:
                        mod.bias_real.data.copy_(b0_r)
                    if b0_i is not None and mod.bias_imag is not None:
                        mod.bias_imag.data.copy_(b0_i)
                elif isinstance(mod, (nn.Linear, nn.Embedding)):
                    w0, b0 = saved[name]
                    mod.weight.data.copy_(w0 + self._project_right(mod.weight.data - w0, name))
                    bias = getattr(mod, "bias", None)
                    if b0 is not None and bias is not None:
                        bias.data.copy_(b0)
            return

        self._ensure_proj_cache()
        if saved.get("cache_sig") != self._proj_cache_sig:
            # Mid-step consolidation altered bucket membership; resolve snapshots safely by param id.
            snap_by_id: Dict[int, torch.Tensor] = {}
            for ids_l, b_snap in zip(saved.get("dense_ids", []), saved["buckets"]):
                for wid, w0_t in zip(ids_l, b_snap.unbind(0)):
                    snap_by_id[wid] = w0_t
            for wid, w0_t in zip(saved.get("factored_ids", []), saved["factored"]):
                snap_by_id[wid] = w0_t
            for w_list, P_stack in self._dense_buckets:
                for b_idx, w in enumerate(w_list):
                    w0_t = snap_by_id.get(id(w))
                    if w0_t is not None:
                        w.copy_(w0_t + (w.detach() - w0_t) @ P_stack[b_idx])
            for (w, U_c) in self._factored_items:
                w0_t = snap_by_id.get(id(w))
                if w0_t is not None:
                    dw = w - w0_t
                    w.copy_(w0_t + (dw - (dw @ U_c) @ U_c.t()))
            bias_by_id = dict(zip(saved.get("bias_ids", []), saved.get("biases", [])))
            for b in self._monitored_biases:
                b0 = bias_by_id.get(id(b))
                if b0 is not None:
                    b.copy_(b0)
        else:
            for (w_list, P_stack), w0_stack in zip(self._dense_buckets, saved["buckets"]):
                w_cur = torch.stack([w.detach() for w in w_list], dim=0)
                w_proj = w0_stack + torch.bmm(w_cur - w0_stack, P_stack)
                torch._foreach_copy_(w_list, list(w_proj.unbind(0)))

            for (w, U_c), w0 in zip(self._factored_items, saved["factored"]):
                dw = w - w0
                w.copy_(w0 + (dw - (dw @ U_c) @ U_c.t()))

            if self._monitored_biases and saved["biases"]:
                torch._foreach_copy_(self._monitored_biases, saved["biases"])

        for (name, mod), (_, w0_r, w0_i, b0_r, b0_i) in zip(self._complex_items, saved["complex"]):
            mod.weight_real.copy_(w0_r + self._project_right(mod.weight_real - w0_r, name))
            mod.weight_imag.copy_(w0_i + self._project_right(mod.weight_imag - w0_i, name))
            if b0_r is not None and mod.bias_real is not None:
                mod.bias_real.copy_(b0_r)
            if b0_i is not None and mod.bias_imag is not None:
                mod.bias_imag.copy_(b0_i)

    def remove_hooks(self) -> None:
        """Remove all registered PyTorch hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()

    def __del__(self) -> None:
        self.remove_hooks()
