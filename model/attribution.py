"""
Token-Level Expert Attribution & Specialization Analytics Suite.
Quantifies:
1. Expert Contribution per Token (who generated what?)
2. Specialization Purity Index (Gini Coefficient across domains)
3. Inter-Expert Co-Activation Graph (who collaborates with whom?)
"""

import math
from typing import List, Dict, Tuple, Any
import torch
import torch.nn.functional as F

class AttributionTracer:
    """
    Traces and formats token-level expert assignments and collaboration statistics.
    """
    
    @staticmethod
    def compute_gini_impurity(distribution: List[float]) -> float:
        """
        Computes Gini index for domain specialization:
        Gini = 1 - sum(p_i^2).
        0.0 = Perfectly specialized to a single domain.
        Higher = Uniformly distributed across all domains.
        """
        total = sum(distribution)
        if total == 0:
            return 0.0
        probs = [x / total for x in distribution]
        return 1.0 - sum(p ** 2 for p in probs)

    @staticmethod
    def trace_generation_attribution(
        model,
        prompt_tensor: torch.Tensor,
        corpus,
        max_new_tokens: int = 40,
        temperature: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Generates text token-by-token while recording the exact expert routing at each step.
        """
        model.eval()
        device = prompt_tensor.device
        curr_ids = prompt_tensor
        trace_log = []

        for _ in range(max_new_tokens):
            idx_cond = curr_ids if curr_ids.size(1) <= model.max_seq_len else curr_ids[:, -model.max_seq_len:]
            
            with torch.no_grad():
                logits, _, telemetries = model(idx_cond, allow_spawning=False)
                
            last_logits = logits[:, -1, :] / temperature
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)
            
            # Extract top experts for the newly generated token from Layer 0
            if telemetries:
                layer0_telem = telemetries[0]
                # top_indices: [Batch, SeqLen, K]
                token_top_experts = layer0_telem["top_indices"][0, -1].tolist()
                token_top_weights = layer0_telem["top_weights"][0, -1].tolist()
            else:
                token_top_experts = [-1]
                token_top_weights = [1.0]

            token_char = corpus.decode([next_token.item()])
            trace_log.append({
                "token_id": next_token.item(),
                "token_char": token_char,
                "experts": token_top_experts,
                "weights": token_top_weights,
            })

        return trace_log

    @staticmethod
    def print_colored_trace(trace_log: List[Dict[str, Any]], title: str = "Attribution Trace"):
        """
        Prints formatted text with expert IDs tagged inline.
        """
        print(f"\n=== {title} ===")
        formatted_str = ""
        for item in trace_log:
            exp_str = ",".join(str(e) for e in item["experts"])
            char = item["token_char"]
            # Represent newlines nicely
            if char == "\n":
                char = "\\n\n"
            formatted_str += f"{char}[E:{exp_str}]"
        print(formatted_str)
        print("-" * 60)
