"""
The 5 Required Parameter- and FLOP-Matched Baselines for Track K.

Baselines:
1. Dense_NoReplay: Plain dense Transformer trained sequentially with AdamW, zero replay.
2. Dense_Replay128: Plain dense Transformer with reservoir buffer of 128 samples.
3. Dense_LoRA_Router: Dense Transformer with per-domain low-rank adapters and sequence router.
4. Dense_InContext: Dense Transformer given recent evidence pairs in its context prompt.
5. NonNeural_Dictionary_Counters: Exact non-neural key-value counting store.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseTransformerBaseline(nn.Module):
    """
    Parameter-matched dense Transformer baseline without dynamic MoE or complex numbers.
    Uses standard real-valued weights, MultiHeadAttention, and feedforward blocks.
    """

    def __init__(
        self,
        d_model: int = 32,
        num_heads: int = 2,
        d_ff: int = 128,
        num_layers: int = 2,
        vocab_size: int = 256,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.chunk_size = chunk_size

        self.byte_embed = nn.Embedding(258, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(d_model),
                "q_proj": nn.Linear(d_model, d_model),
                "k_proj": nn.Linear(d_model, d_model),
                "v_proj": nn.Linear(d_model, d_model),
                "out_proj": nn.Linear(d_model, d_model),
                "norm2": nn.LayerNorm(d_model),
                "ffn1": nn.Linear(d_model, d_ff),
                "ffn2": nn.Linear(d_ff, d_model),
            })
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, chunk_size * vocab_size)

    def get_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """Compute final hidden states: [B, L, d_model]."""
        B, L = x.shape
        h = self.byte_embed(x) + self.pos_embed[:, :L, :]

        for layer in self.layers:
            h_norm = layer["norm1"](h)
            q = layer["q_proj"](h_norm)
            k = layer["k_proj"](h_norm)
            v = layer["v_proj"](h_norm)

            scores = (q @ k.transpose(-2, -1)) / (self.d_model ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            attn_out = layer["out_proj"](attn_weights @ v)
            h = h + attn_out

            h_ffn = layer["norm2"](h)
            ffn_out = layer["ffn2"](F.gelu(layer["ffn1"](h_ffn)))
            h = h + ffn_out

        return self.final_norm(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L] byte tokens in [32..47]
        Returns logits: [B, L, 256]
        """
        B, L = x.shape
        h = self.get_hidden(x)
        if self.chunk_size == 1:
            return self.out_head(h)
        n_chunks = L // self.chunk_size
        h_chunks = h.view(B, n_chunks, self.chunk_size, self.d_model).mean(dim=2)
        logits = self.out_head(h_chunks).view(B, n_chunks, self.chunk_size, 256)
        return logits.view(B, L, 256)


class DenseLoRAExpertBaseline(nn.Module):
    """
    Mainstream parameter-isolation baseline:
    Frozen Dense Transformer base + per-domain low-rank LoRA adapter + domain router.
    """

    def __init__(
        self,
        base_model: DenseTransformerBaseline,
        d_model: int = 32,
        lora_rank: int = 4,
        max_domains: int = 8,
    ):
        super().__init__()
        self.base_model = base_model
        # Freeze base model parameters
        for p in self.base_model.parameters():
            p.requires_grad_(False)

        self.d_model = d_model
        self.lora_rank = lora_rank
        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, lora_rank) * 0.02)
            for _ in range(max_domains)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(lora_rank, d_model))
            for _ in range(max_domains)
        ])
        self.lora_head = nn.ModuleList([
            nn.Linear(d_model, 256)
            for _ in range(max_domains)
        ])
        self.active_domain = 0

    def forward(self, x: torch.Tensor, domain_id: Optional[int] = None) -> torch.Tensor:
        d = domain_id if domain_id is not None else self.active_domain
        h = self.base_model.get_hidden(x)
        delta_h = (h @ self.lora_A[d]) @ self.lora_B[d]
        base_logits = self.base_model.out_head(h)
        adapter_logits = self.lora_head[d](h + delta_h)
        return base_logits + adapter_logits


class NonNeuralDictionaryBaseline:
    """
    Baseline 5: Exact Non-Neural Key-Value Store with Counters.
    The algorithmic integrity check:
    Maintains a prior-count table: (context, input) -> {target: count, sources: set()}.
    - Prior-count policy:
      - If no prior count: learns target immediately with count=1.
      - If contradictory target arrives: increments tentative count for contradictory target.
      - If tentative count > established prior count: flips the primary belief.
    """

    def __init__(self, flip_threshold_ratio: float = 1.0):
        # primary_store: (x_{t-1}, x_t) -> (primary_y, prior_count)
        self.primary_store: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # tentative_store: (x_{t-1}, x_t) -> {candidate_y: (count, sources_set)}
        self.tentative_store: Dict[Tuple[int, int], Dict[int, Tuple[int, set]]] = defaultdict(dict)
        self.flip_threshold_ratio = flip_threshold_ratio

    def update(self, x: torch.Tensor, y: torch.Tensor, source_id: str = "default") -> None:
        B, L = x.shape
        x_cpu = x.detach().cpu().tolist()
        y_cpu = y.detach().cpu().tolist()

        for b in range(B):
            for t in range(1, L):
                prev_x = x_cpu[b][t - 1]
                cur_x = x_cpu[b][t]
                target_y = y_cpu[b][t]
                key = (prev_x, cur_x)

                if key not in self.primary_store:
                    # Brand-new fact: learn immediately at low confidence (count=1)
                    self.primary_store[key] = (target_y, 1)
                else:
                    cur_y, prior_cnt = self.primary_store[key]
                    if target_y == cur_y:
                        # Reinforce existing belief
                        self.primary_store[key] = (cur_y, prior_cnt + 1)
                    else:
                        # Contradictory observation: register in tentative store
                        cand_dict = self.tentative_store[key]
                        c_cnt, c_sources = cand_dict.get(target_y, (0, set()))
                        c_sources.add(source_id)
                        cand_dict[target_y] = (c_cnt + 1, c_sources)

                        # Flip check: if consistent counter-evidence outweighs prior
                        # and originates from multiple sources or sustained observations
                        if (c_cnt + 1) >= prior_cnt * self.flip_threshold_ratio:
                            # Flip belief!
                            self.primary_store[key] = (target_y, c_cnt + 1)
                            del cand_dict[target_y]

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        x_cpu = x.detach().cpu().tolist()
        preds = torch.zeros(B, L, dtype=torch.long, device=x.device)

        for b in range(B):
            preds[b, 0] = x[b, 0]
            for t in range(1, L):
                prev_x = x_cpu[b][t - 1]
                cur_x = x_cpu[b][t]
                key = (prev_x, cur_x)
                if key in self.primary_store:
                    preds[b, t] = self.primary_store[key][0]
                else:
                    preds[b, t] = cur_x  # Fallback to identity
        return preds

    def storage_bytes(self) -> int:
        """Calculate exact memory usage of dictionary stores."""
        # Key: 2 ints (16 bytes), Value: 2 ints (16 bytes) -> 32 bytes per entry
        primary_bytes = len(self.primary_store) * 32
        tentative_entries = sum(len(c) for c in self.tentative_store.values())
        tentative_bytes = tentative_entries * 40
        return primary_bytes + tentative_bytes
