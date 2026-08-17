"""
Baseline Model Architectures for Rigorous Comparative Benchmarking:
1. DenseTransformerLM: Standard monolithic Transformer (LLaMA-style).
2. StaticSoftmaxMoELM: Standard fixed-expert MoE with linear softmax gating (Switch/Mixtral style).
"""

import math
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nanogpt import RMSNorm, CausalSelfAttention
from model.expert import MicroExpert

# ----------------------------------------------------------------------
# 1. Standard Dense Transformer
# ----------------------------------------------------------------------
class DenseTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = MicroExpert(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class DenseTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 3, n_heads: int = 4, d_ff: int = 1024, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            DenseTransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

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

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Dict[str, Any]]]:
        b, s = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, s, device=device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.pos_embeddings(positions)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss, []

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 40, temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        self.eval()
        curr_ids = prompt_ids
        for _ in range(max_new_tokens):
            idx_cond = curr_ids if curr_ids.size(1) <= self.max_seq_len else curr_ids[:, -self.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
        return curr_ids

# ----------------------------------------------------------------------
# 2. Static Softmax Mixture of Experts (Fixed Experts + Linear Gate)
# ----------------------------------------------------------------------
class StaticSoftmaxMoELayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([MicroExpert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, s, d = x.shape
        flat_x = x.view(-1, d)
        
        # Linear Softmax Gate
        gate_logits = self.gate(flat_x)
        top_weights, top_indices = torch.topk(F.softmax(gate_logits, dim=-1), k=self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        out = torch.zeros_like(flat_x)
        for k_idx in range(self.top_k):
            indices_k = top_indices[:, k_idx]
            weights_k = top_weights[:, k_idx].unsqueeze(-1)
            
            for exp_id, expert in enumerate(self.experts):
                mask = (indices_k == exp_id)
                if mask.any():
                    out[mask] += weights_k[mask] * expert(flat_x[mask])
                    
        telemetry = {
            "top_indices": top_indices.detach().view(b, s, self.top_k),
            "top_weights": top_weights.detach().view(b, s, self.top_k),
        }
        return out.view(b, s, d), telemetry

class StaticSoftmaxMoEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int = 4, top_k: int = 2, max_seq_len: int = 512):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.moe = StaticSoftmaxMoELayer(d_model, d_ff, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = x + self.attn(self.norm1(x))
        moe_out, telem = self.moe(self.norm2(x))
        x = x + moe_out
        return x, telem

class StaticSoftmaxMoELM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 256, n_layers: int = 3, n_heads: int = 4, d_ff: int = 512, num_experts: int = 4, top_k: int = 2, max_seq_len: int = 512):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(max_seq_len, d_model)
        
        self.blocks = nn.ModuleList([
            StaticSoftmaxMoEBlock(d_model, n_heads, d_ff, num_experts, top_k, max_seq_len)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

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

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Dict[str, Any]]]:
        b, s = input_ids.shape
        device = input_ids.device
        positions = torch.arange(0, s, device=device).unsqueeze(0)
        x = self.token_embeddings(input_ids) + self.pos_embeddings(positions)
        
        layer_telemetries = []
        for block in self.blocks:
            x, telem = block(x)
            layer_telemetries.append(telem)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
            
        return logits, loss, layer_telemetries

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 40, temperature: float = 0.8, top_k: int = 40) -> torch.Tensor:
        self.eval()
        curr_ids = prompt_ids
        for _ in range(max_new_tokens):
            idx_cond = curr_ids if curr_ids.size(1) <= self.max_seq_len else curr_ids[:, -self.max_seq_len:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
        return curr_ids

# ----------------------------------------------------------------------
# 3. Strictly Capacity-Matched Static MoE (Exact 128.95M Parameter Match)
# ----------------------------------------------------------------------
from model.dendritic_expert import TwoCompartmentDendriticExpert

class DendriticStaticSoftmaxMoELayer(nn.Module):
    """Static Softmax MoE using identical TwoCompartmentDendriticExperts (Exact Parameter Match)."""
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 16, top_k: int = 2, d_gate: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_gate, bias=False),
            nn.SiLU(),
            nn.Linear(d_gate, num_experts, bias=False)
        )
        self.experts = nn.ModuleList([TwoCompartmentDendriticExpert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, s, d = x.shape
        flat_x = x.view(-1, d)
        gate_logits = self.gate(flat_x)
        top_weights, top_indices = torch.topk(F.softmax(gate_logits, dim=-1), k=self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        out = torch.zeros_like(flat_x)
        for k_idx in range(self.top_k):
            indices_k = top_indices[:, k_idx]
            weights_k = top_weights[:, k_idx].unsqueeze(-1)
            for exp_id, expert in enumerate(self.experts):
                mask = (indices_k == exp_id)
                if mask.any():
                    out[mask] += weights_k[mask] * expert(flat_x[mask])
                    
        return out.view(b, s, d), {"top_indices": top_indices.detach().view(b, s, self.top_k)}

class DendriticStaticSoftmaxMoEBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, num_experts: int = 16, top_k: int = 2, max_seq_len: int = 576):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.norm2 = RMSNorm(d_model)
        self.moe = DendriticStaticSoftmaxMoELayer(d_model, d_ff, num_experts, top_k)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = x + self.attn(self.norm1(x))
        moe_out, telem = self.moe(self.norm2(x))
        x = x + moe_out
        return x, telem

class DendriticStaticSoftmaxMoELM(nn.Module):
    """
    Strictly Capacity-Matched Static MoE Baseline:
    Identical 16 TwoCompartmentDendriticExperts per layer (128.95M parameters),
    using standard linear Softmax Gating instead of Complex Phasor Hyperspace Gating.
    """
    def __init__(self, vocab_size: int = 50304, d_model: int = 384, n_layers: int = 4, n_heads: int = 6, d_ff: int = 768, num_experts: int = 16, top_k: int = 2, max_seq_len: int = 576):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            DendriticStaticSoftmaxMoEBlock(d_model, n_heads, d_ff, num_experts, top_k, max_seq_len)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embeddings.weight

    def forward(self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Dict[str, Any]]]:
        x = self.token_embeddings(input_ids)
        telemetries = []
        for block in self.blocks:
            x, telem = block(x)
            telemetries.append(telem)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return logits, loss, telemetries
