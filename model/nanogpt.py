"""
HyperTransformerLM: NanoGPT Architecture powered by Dynamic Hyper-MoE Layers.
"""

import math
from typing import Optional, Tuple, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.hyper_moe import DynamicHyperMoE

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class CausalSelfAttention(nn.Module):
    """Standard Multi-Head Causal Self-Attention."""
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 512, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Causal mask buffer
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        qkv = self.qkv_proj(x).view(b, s, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [b, n_heads, s, head_dim]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_mask[:, :, :s, :s] == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v) # [b, n_heads, s, head_dim]
        context = context.permute(0, 2, 1, 3).contiguous().view(b, s, d)
        return self.out_proj(context)

from model.hyper_moe import DynamicHyperMoE
from model.dynamic_sparse_attention import DynamicSparseAttention

class HyperTransformerBlock(nn.Module):
    """Transformer Block combining Causal Attention with a Dynamic Hyper-MoE FFN."""
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        d_hyper: int,
        top_k: int = 2,
        max_k: int = 4,
        top_p: float = 0.85,
        dynamic_k: bool = False,
        spawn_threshold: float = 0.35,
        max_experts: int = 32,
        initial_experts: int = 2,
        use_sparse_attn: bool = False,
        foveal_window: int = 128,
        num_landmarks: int = 4,
        num_sinks: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        use_bus: bool = True,
        use_hopfield: bool = True,
        use_criticality: bool = True,
        use_drift_guard: bool = False,
        drift_sigma: float = 2.0,
        use_context_binding: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.use_sparse_attn = use_sparse_attn
        if use_sparse_attn:
            self.attn = DynamicSparseAttention(
                d_model=d_model,
                n_heads=n_heads,
                foveal_window=foveal_window,
                num_sinks=num_sinks,
                num_landmarks=num_landmarks,
                dropout=dropout
            )
        else:
            self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        
        self.norm2 = RMSNorm(d_model)
        self.hyper_moe = DynamicHyperMoE(
            d_model=d_model,
            d_ff=d_ff,
            d_hyper=d_hyper,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            dynamic_k=dynamic_k,
            spawn_threshold=spawn_threshold,
            max_experts=max_experts,
            initial_experts=initial_experts,
            use_bus=use_bus,
            use_hopfield=use_hopfield,
            use_criticality=use_criticality,
            use_drift_guard=use_drift_guard,
            drift_sigma=drift_sigma,
            use_context_binding=use_context_binding,
        )

    def forward(self, x: torch.Tensor, allow_spawning: bool = True, is_replay: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Pre-LN Self-Attention
        if self.use_sparse_attn:
            attn_out, _ = self.attn(self.norm1(x))
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x))
            
        # Pre-LN Hyper-MoE
        moe_out, telemetry = self.hyper_moe(self.norm2(x), allow_spawning=allow_spawning, is_replay=is_replay)
        x = x + moe_out
        return x, telemetry

class HyperTransformerLM(nn.Module):
    """
    Causal Language Model with Dynamic Hyperspace Mixture-of-Experts.
    """
    def __init__(
        self,
        vocab_size: int = 256, # Default to byte-level or small char vocab for zero-dep test
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        d_hyper: int = 2048,
        top_k: int = 2,
        max_k: int = 4,
        top_p: float = 0.85,
        dynamic_k: bool = False,
        spawn_threshold: float = 0.35,
        max_experts: int = 32,
        initial_experts: int = 2,
        use_sparse_attn: bool = False,
        foveal_window: int = 128,
        num_landmarks: int = 4,
        num_sinks: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.0,
        use_bus: bool = True,
        use_hopfield: bool = True,
        use_criticality: bool = True,
        use_drift_guard: Union[bool, List[bool]] = False,
        drift_sigma: float = 2.0,
        use_context_binding: bool = False,
        ortho_loss_weight: float = 0.005,
        load_bal_weight: float = 0.01,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dynamic_k = dynamic_k
        self.max_k = max_k
        self.top_p = top_p
        self.ortho_loss_weight = ortho_loss_weight
        self.load_bal_weight = load_bal_weight
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
        drift_guards = use_drift_guard if isinstance(use_drift_guard, (list, tuple)) else [use_drift_guard] * n_layers
        
        self.blocks = nn.ModuleList([
            HyperTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                d_hyper=d_hyper,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                dynamic_k=dynamic_k,
                spawn_threshold=spawn_threshold,
                max_experts=max_experts,
                initial_experts=initial_experts,
                use_sparse_attn=use_sparse_attn,
                foveal_window=foveal_window,
                num_landmarks=num_landmarks,
                num_sinks=num_sinks,
                max_seq_len=max_seq_len,
                dropout=dropout,
                use_bus=use_bus,
                use_hopfield=use_hopfield,
                use_criticality=use_criticality,
                use_drift_guard=drift_guards[l],
                drift_sigma=drift_sigma,
                use_context_binding=use_context_binding,
            )
            for l in range(n_layers)
        ])
        
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie token embeddings and LM head weights
        self.lm_head.weight = self.token_embeddings.weight

        # Standard GPT-2/LLaMA weight initialization
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith('out_proj.weight') or name.endswith('w_down.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def configure_optimizers(self, lr: float = 6e-4, weight_decay: float = 0.01, betas: Tuple[float, float] = (0.9, 0.95)):
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        return torch.optim.AdamW(optim_groups, lr=lr, betas=betas)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        allow_spawning: bool = True,
        is_replay: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Dict[str, Any]]]:
        """
        input_ids: [Batch, SeqLen]
        targets: Optional [Batch, SeqLen] for calculating Cross-Entropy Loss
        """
        b, s = input_ids.shape
        device = input_ids.device
        
        x = self.token_embeddings(input_ids)
        x = self.dropout(x)
        
        layer_telemetries = []
        for l_idx, block in enumerate(self.blocks):
            block_spawning = allow_spawning[l_idx] if isinstance(allow_spawning, (list, tuple)) else allow_spawning
            x, telem = block(x, allow_spawning=block_spawning, is_replay=is_replay)
            layer_telemetries.append(telem)
            
        x = self.final_norm(x)
        logits = self.lm_head(x) # [Batch, SeqLen, vocab_size]
        
        loss = None
        if targets is not None:
            # Cross-entropy token loss
            lm_loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            # Auxiliary soft orthogonality loss & load-balancing loss from active MoE layers
            ortho_loss = sum(telem.get("ortho_loss", 0.0) for telem in layer_telemetries) if layer_telemetries else 0.0
            load_bal_loss = sum(telem.get("load_balance_loss", 0.0) for telem in layer_telemetries) if layer_telemetries else 0.0
            
            if isinstance(ortho_loss, torch.Tensor):
                ortho_loss = torch.clamp(ortho_loss, max=5.0)
            else:
                ortho_loss = min(5.0, float(ortho_loss))
                
            if isinstance(load_bal_loss, torch.Tensor):
                load_bal_loss = torch.clamp(load_bal_loss, max=5.0)
            else:
                load_bal_loss = min(5.0, float(load_bal_loss))
                
            loss = lm_loss + (self.ortho_loss_weight * ortho_loss) + (self.load_bal_weight * load_bal_loss)
            
        return logits, loss, layer_telemetries

    @torch.no_grad()
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """Autoregressive text generation with fixed experts (allow_spawning=False)."""
        curr_ids, _ = self.generate_with_telemetry(prompt_ids, max_new_tokens, temperature, top_k)
        return curr_ids

    @torch.no_grad()
    def generate_with_telemetry(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        Autoregressive text generation tracking per-token expert routing decisions and telemetry.
        """
        self.eval()
        curr_ids = prompt_ids
        step_telemetries = []

        for step_idx in range(max_new_tokens):
            idx_cond = curr_ids if curr_ids.size(1) <= self.max_seq_len else curr_ids[:, -self.max_seq_len:]
            logits, _, layer_telems = self(idx_cond, allow_spawning=False)
            logits = logits[:, -1, :] / max(1e-4, temperature)
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

            # Record per-layer routing decisions for the last position
            step_record = {
                "step": step_idx,
                "token_id": next_token.item(),
                "layer_routings": []
            }
            for l_idx, telem in enumerate(layer_telems):
                top_idx = telem["top_indices"][:, -1, :].cpu().numpy().tolist()[0]
                top_w = telem["top_weights"][:, -1, :].cpu().numpy().tolist()[0]
                max_res = telem.get("max_resonance", 0.0)
                step_record["layer_routings"].append({
                    "layer": l_idx,
                    "top_indices": top_idx,
                    "top_weights": top_w,
                    "max_resonance": max_res
                })

            step_telemetries.append(step_record)
            
        return curr_ids, step_telemetries
