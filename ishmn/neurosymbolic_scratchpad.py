"""
Prefrontal Neuro-Symbolic Scratchpad Interception.
Connects UCE's 100% exact arithmetic engine with SLM token generation
to verify equations and eliminate multi-step arithmetic dropouts.
"""

import re
import torch
from typing import Optional
from transformers import LogitsProcessor

from ishmn.universal_cognitive_engine import UniversalCognitiveEngine


class NeuroSymbolicScratchpadLogitsProcessor(LogitsProcessor):
    """
    Monitors recently generated tokens for arithmetic transitions (e.g. 'A + B =', 'A * B =').
    Computes exact ground-truth via UCE's dual-process arithmetic solver,
    and boosts logits for the exact numerical tokens.
    """
    def __init__(
        self,
        uce: UniversalCognitiveEngine,
        tokenizer,
        boost_weight: float = 2.5
    ):
        self.uce = uce
        self.tokenizer = tokenizer
        self.boost_weight = boost_weight

        # Pattern for simple binary arithmetic: e.g. "8 * 5 =", "16 - 3 =", "80 + 50 ="
        self.eq_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:\.\d+)?)\s*=\s*\$?$")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Decode the last 12 tokens to inspect recent generation context
        seq_len = input_ids.shape[1]
        recent_ids = input_ids[0, max(0, seq_len - 12):]
        recent_text = self.tokenizer.decode(recent_ids, skip_special_tokens=True)

        match = self.eq_pattern.search(recent_text)
        if match:
            a_str, op, b_str = match.groups()
            try:
                a_val = float(a_str)
                b_val = float(b_str)

                # Solve via UCE dual-process arithmetic solver
                res = self.uce.solve_with_dual_process(f"{a_val} {op} {b_val}", enable_system2=True)
                exact_ans = res.get("system2_exact_result")

                if exact_ans is not None:
                    # Format as clean integer if whole number
                    if float(exact_ans).is_integer():
                        ans_str = str(int(exact_ans))
                    else:
                        ans_str = f"{exact_ans:.2f}".rstrip("0").rstrip(".")

                    # Add bounded additive prior for the exact answer token (System 2 guidance, no hard override)
                    for prefix in ["", " "]:
                        tok_ids = self.tokenizer.encode(prefix + ans_str, add_special_tokens=False)
                        if tok_ids:
                            first_id = tok_ids[0]
                            if first_id < scores.shape[-1]:
                                scores[:, first_id] = scores[:, first_id] + self.boost_weight
            except Exception:
                pass


        return scores
