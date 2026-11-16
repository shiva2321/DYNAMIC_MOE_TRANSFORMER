"""
Fair, strictly causal Stage 2 reference baselines for comparing against
Dynamic Thinking + Latent World Model architectures without strawmen:
  1. Re-exports Stage 1's CausalDenseRoPEBaseline, GRUBaseline, count_parameters.
  2. CausalLoopedTransformerBaseline: Weight-tied Causal RoPE Transformer iterated K steps
     (Universal Transformer / Looped Transformer comparator with NO hardcoded z_{t-1} shift),
     supporting deep supervision across steps k = 0..K.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.stage1_baselines import (
    CausalDenseRoPEBaseline,
    CausalRoPEBlock,
    GRUBaseline,
    count_parameters,
)


class CausalLoopedTransformerBaseline(nn.Module):
    """
    Generic weight-tied Causal RoPE Transformer looped up to K steps.
    Serves as a fair looped-compute comparator without hand-crafted neighbor-shift operators.
    """

    def __init__(
        self,
        d_model: int = 48,
        num_heads: int = 4,
        d_ff: int = 128,
        num_prelude_layers: int = 1,
        max_steps: int = 5,
        vocab_size: int = 258,
        out_vocab: int = 256,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        self.embed = nn.Embedding(vocab_size, d_model)
        self.prelude = nn.ModuleList([
            CausalRoPEBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope_base=rope_base)
            for _ in range(num_prelude_layers)
        ])
        self.loop_block = CausalRoPEBlock(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope_base=rope_base
        )
        self.step_embed = nn.Embedding(max(16, max_steps + 1), d_model)
        self.anchor_scale = nn.Parameter(torch.tensor(0.5))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_vocab)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(self.final_norm(z))

    def forward(
        self,
        x: torch.Tensor,
        force_thinking_steps: Optional[int] = None,
    ) -> Dict[str, object]:
        B, L = x.shape
        h_anchor = self.embed(x)
        for blk in self.prelude:
            h_anchor = blk(h_anchor)

        steps = self.max_steps if force_thinking_steps is None else int(force_thinking_steps)
        z = h_anchor
        step_logits: List[torch.Tensor] = [self._decode(z)]
        alpha = torch.sigmoid(self.anchor_scale)

        for k in range(steps):
            s_idx = torch.full((1, 1), min(k, self.step_embed.num_embeddings - 1), device=x.device, dtype=torch.long)
            s_vec = self.step_embed(s_idx)
            # Blend current state with anchor input before weight-tied causal RoPE attention + FFN
            z_in = (1.0 - alpha) * z + alpha * h_anchor + s_vec
            z = self.loop_block(z_in)
            step_logits.append(self._decode(z))

        return {
            "output": step_logits[-1],
            "latent": z,
            "step_logits": step_logits,
            "steps_used": steps,
            "active_token_fraction": 1.0 if steps > 0 else 0.0,
        }

    def compute_deep_loss(self, out: Dict[str, object], targets: torch.Tensor) -> torch.Tensor:
        """Step-weighted deep supervision matching SurpriseDrivenLatentThinker training."""
        step_logits: List[torch.Tensor] = out["step_logits"]  # type: ignore[assignment]
        weights = [float(k + 1) for k in range(len(step_logits))]
        w_sum = sum(weights)
        total = step_logits[0].sum() * 0.0
        for w, lg in zip(weights, step_logits):
            ce_k = F.cross_entropy(lg.reshape(-1, lg.size(-1)), targets.reshape(-1))
            total = total + (w / w_sum) * ce_k
        return total


__all__ = [
    "CausalDenseRoPEBaseline",
    "CausalLoopedTransformerBaseline",
    "GRUBaseline",
    "count_parameters",
]
