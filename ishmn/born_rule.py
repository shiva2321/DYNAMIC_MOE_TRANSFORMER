"""
Rectified Born Rule Generation Filter & Logits Processor.
Adapted from Resonance X / Quantum Probability & Vector Symbolic Architectures.

Provides interference-based decoding:
1. Computes similarity of candidate vocabulary tokens against recent trailing tokens.
2. Applies destructive quantum interference:
   adjusted_i = sim_i - alpha * rep_penalty_i
   raw_i = max(0, adjusted_i)^2   (Born quadratic probability amplitude)
3. Normalizes to valid probability distribution with margin & gate diagnostics.
4. Operates as a Hugging Face LogitsProcessor to prevent degenerative token looping
   without heuristic logit mutilation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LogitsProcessor


@dataclass
class BornResult:
    probs: torch.Tensor          # (V,) normalized probabilities
    best_idx: int                # Winning token index (-1 if none)
    best_prob: float             # Winning token probability
    entropy: float               # Shannon entropy of distribution
    margin: float                # Margin between top-1 and top-2
    gate_passed: bool            # True if best_prob >= gate_threshold


def rectified_born(
    similarity: torch.Tensor,
    *,
    history_embeds: Optional[torch.Tensor] = None,
    vocab_matrix: Optional[torch.Tensor] = None,
    alpha: float = 0.20,
    gate_threshold: float = 0.01,
    is_logits: bool = False,
    temperature: float = 1.0,
) -> BornResult:
    """
    Computes Rectified Born Rule probabilities for one decoding step.

    Parameters
    ----------
    similarity : torch.Tensor, shape (V,)
        Real-valued cosine-similarity-scale scores in [-1, 1] or raw logits.
    history_embeds : Optional[torch.Tensor], shape (H, D)
        Recent token embeddings for computing interference penalties.
    vocab_matrix : Optional[torch.Tensor], shape (V, D)
        Vocabulary embedding table (normalized).
    alpha : float
        Repetition interference weight.
    gate_threshold : float
        Minimum winning probability required to pass emission gate.
    is_logits : bool
        If True, normalizes logits via softmax to ensure strict shift-invariance.
    temperature : float
        Softmax temperature when is_logits=True.
    """
    sims = similarity.float()
    V = sims.shape[0]
    device = sims.device

    # Shift-Invariance Guard: If inputs are uncalibrated raw logits, map via softmax
    # so adding an arbitrary constant c to all logits preserves exact output probabilities.
    if is_logits or (sims.numel() > 0 and (sims.max() > 1.05 or sims.min() < -1.05)):
        temp = max(1e-4, temperature)
        sims = F.softmax(sims / temp, dim=-1)

    rep_penalty = torch.zeros(V, device=device, dtype=torch.float32)
    if history_embeds is not None and vocab_matrix is not None:
        if history_embeds.numel() > 0:
            # history_embeds: (H, D), vocab_matrix: (V, D)
            # Cosine similarity matrix between vocabulary and recent history
            hist_norm = F.normalize(history_embeds.float(), p=2, dim=-1)
            v_norm = F.normalize(vocab_matrix.float(), p=2, dim=-1)
            hist_sims = torch.matmul(v_norm, hist_norm.T)  # (V, H)
            rep_penalty = torch.clamp(hist_sims.max(dim=-1).values, min=0.0)

    adjusted = sims - alpha * rep_penalty
    raw = torch.clamp(adjusted, min=0.0) ** 2
    raw_total = raw.sum()

    if raw_total <= 1e-14:
        return BornResult(
            probs=torch.zeros(V, device=device),
            best_idx=-1,
            best_prob=0.0,
            entropy=1.0,
            margin=0.0,
            gate_passed=False,
        )

    probs = raw / raw_total
    max_penalty_value = rep_penalty.max().item()
    abstain_mass = (alpha * max_penalty_value) ** 2
    total = raw_total.item() + abstain_mass

    best_idx = int(torch.argmax(raw).item())
    best_prob = float(raw[best_idx].item() / total)

    lp = torch.log(torch.clamp(probs, min=1e-16))
    entropy = float(-(probs * lp).sum().item())

    if V >= 2:
        top2 = torch.topk(raw, 2).values
        margin = float((top2[0] - top2[1]).item() / total)
    else:
        margin = best_prob

    return BornResult(
        probs=probs,
        best_idx=best_idx,
        best_prob=best_prob,
        entropy=entropy,
        margin=margin,
        gate_passed=best_prob >= gate_threshold,
    )


class BornRuleLogitsProcessor(LogitsProcessor):
    """
    HuggingFace LogitsProcessor that applies the Rectified Born Rule
    to suppress degenerative repetition loops during autoregressive generation.
    Applies quadratic destructive interference against trailing token embeddings.
    """
    def __init__(
        self,
        embed_tokens: nn.Embedding,
        alpha: float = 3.5,
        window_size: int = 32
    ):
        self.embed_tokens = embed_tokens
        self.alpha = alpha
        self.window_size = window_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: (B, T)
        scores: (B, V) raw logits
        """
        B, T = input_ids.shape
        if T <= 1 or self.alpha <= 0.0:
            return scores

        device = scores.device
        vocab_weight = self.embed_tokens.weight.data.to(device)  # (V, D)
        vocab_norm = F.normalize(vocab_weight.float(), p=2, dim=-1)

        modified_scores = scores.clone()
        hist_window = min(T, self.window_size)
        recent_ids = input_ids[:, -hist_window:]  # (B, H)

        for b in range(B):
            b_ids = recent_ids[b]
            b_embeds = self.embed_tokens(b_ids).detach().float()  # (H, D)
            b_embeds_norm = F.normalize(b_embeds, p=2, dim=-1)

            # Cosine similarity of all vocabulary items to recent emitted tokens
            # (V, D) @ (D, H) -> (V, H)
            hist_sims = torch.matmul(vocab_norm, b_embeds_norm.T)

            # Rectified Born quadratic interference: max(0, sim)^2
            rect_sims = torch.clamp(hist_sims, min=0.0) ** 2  # (V, H)
            rep_penalty = rect_sims.max(dim=-1).values        # (V,) in [0, 1]

            # Apply destructive interference to logits
            modified_scores[b] = modified_scores[b] - self.alpha * rep_penalty

            # Damping exact token repeats
            modified_scores[b, b_ids] -= (self.alpha * 0.5)

        return modified_scores
