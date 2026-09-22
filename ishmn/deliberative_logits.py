"""
Deliberative Concept Logits Grounding.
Connects UCE's crystallized semantic concepts and invariant attractors
with autoregressive token generation to steer vocabulary selection toward
physically/factually grounded attractors.
"""

from typing import Optional, List, Dict, Any
import torch
from transformers import LogitsProcessor


class DeliberativeConceptLogitsProcessor(LogitsProcessor):
    """
    Monitors recent generation context and cross-references UCE semantic invariants.
    Adds bounded additive logit priors for verified conceptual attractors.
    """
    def __init__(
        self,
        uce: Any,
        tokenizer: Any,
        boost_weight: float = 2.0,
        max_boost_tokens: int = 3,
        enabled: bool = True
    ):
        self.uce = uce
        self.tokenizer = tokenizer
        self.boost_weight = boost_weight
        self.max_boost_tokens = max_boost_tokens
        self.enabled = enabled

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if not self.enabled:
            return scores

        # Decode recent tokens to extract context words
        seq_len = input_ids.shape[1]
        recent_ids = input_ids[0, max(0, seq_len - 32):]
        if self.tokenizer is not None:
            recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=True).lower()
        else:
            recent_text = "".join(chr(min(max(t.item(), 0), 127)) for t in recent_ids).lower()

        # Check UCE semantic concepts
        recalled_words = []
        if hasattr(self.uce, "memory") and hasattr(self.uce.memory, "semantic_concepts"):
            for sc in self.uce.memory.semantic_concepts:
                meta = sc.get("metadata", {})
                w = meta.get("word", "").strip().lower()
                kws = [k.lower() for k in meta.get("keywords", [])]
                if w and (w in recent_text or any(kw in recent_text for kw in kws if len(kw) > 2)):
                    if w not in recalled_words:
                        recalled_words.append(w)
                    if len(recalled_words) >= self.max_boost_tokens:
                        break

        # Check Hopfield stored concept names as fallback
        if len(recalled_words) < self.max_boost_tokens and hasattr(self.uce, "hopfield"):
            for w in self.uce.hopfield.concept_names:
                if len(w) > 2 and w in recent_text and w not in recalled_words:
                    recalled_words.append(w)
                    if len(recalled_words) >= self.max_boost_tokens:
                        break

        # Apply gentle prior boost for recalled words
        if recalled_words and self.tokenizer is not None:
            device = scores.device
            for word in recalled_words:
                for prefix in [" ", ""]:
                    tok_ids = self.tokenizer.encode(prefix + word, add_special_tokens=False)
                    if tok_ids:
                        first_id = tok_ids[0]
                        if first_id < scores.shape[-1]:
                            scores[:, first_id] = scores[:, first_id] + self.boost_weight

        return scores
