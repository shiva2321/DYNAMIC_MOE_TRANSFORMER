"""
Causal World Model with Bidirectional Counterfactual Inversion (Pearl Level 3).

Enforces mathematical reversibility and self-correction:
Every forward hypothesis must verify its own counterfactual rollback:
f(a, op, b) = c  <==>  f(c, inv(op), b) = a

Components:
1. Bidirectional Inversion Operator Mapping
2. Causal Inconsistency / Surprise Metric (S = ||a_recon - a||^2)
3. Iterative Mental Self-Correction Loop (recovers from noise or misprediction)
4. Structured Causal Verification & Natural Language Explanation Generator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ishmn.grounded_perceptor import OPERATOR_SEMANTICS, GroundedPerceptor

class CausalWorldModel(nn.Module):
    def __init__(
        self,
        forward_model: nn.Module,
        perceptor: GroundedPerceptor,
        surprise_threshold: float = 1.0,
        max_correction_steps: int = 5,
        correction_lr: float = 0.5,
        device: torch.device = None
    ):
        super().__init__()
        self.forward_model = forward_model
        self.perceptor = perceptor
        self.surprise_threshold = surprise_threshold
        self.max_correction_steps = max_correction_steps
        self.correction_lr = correction_lr
        self.device = device if device is not None else torch.device("cpu")

    def forward_eval(self, a: torch.Tensor, op: str, b: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Executes forward perception + model pass.
        Returns: predicted_val (unnormalized float tensor), metadata
        """
        grounded = self.perceptor.ground_problem(a, [op] * len(a), b)
        pred, meta = self.forward_model(grounded["X"], grounded["U"], grounded["R"])
        # Denormalize output
        pred_val = pred * 50.0
        return pred_val, meta

    def verify_counterfactual(
        self,
        a_val: float,
        op: str,
        b_val: float,
        candidate_c: float
    ) -> tuple[float, float, bool]:
        """
        Executes counterfactual rollback:
        a_recon = forward(candidate_c, inv(op), b_val)
        Returns: a_recon, surprise, is_consistent
        """
        inv_op = OPERATOR_SEMANTICS[op]["inverse"]
        
        # Invert operation: candidate_c inv_op b_val
        c_tensor = torch.tensor([candidate_c], dtype=torch.float32, device=self.device)
        b_tensor = torch.tensor([b_val], dtype=torch.float32, device=self.device)
        
        a_recon_tensor, _ = self.forward_eval(c_tensor, inv_op, b_tensor)
        a_recon = a_recon_tensor[0, 0].item()
        
        surprise = abs(a_recon - a_val)
        is_consistent = surprise <= self.surprise_threshold

        return a_recon, surprise, is_consistent

    def evaluate_with_self_correction(
        self,
        a_val: float,
        op: str,
        b_val: float,
        injected_noise: float = 0.0
    ) -> dict:
        """
        Evaluates problem (a, op, b), checks counterfactual reversibility,
        and if inconsistent (e.g. due to noise or difficult edge cases),
        executes iterative mental self-correction until consensus is achieved.
        """
        a_tensor = torch.tensor([a_val], dtype=torch.float32, device=self.device)
        b_tensor = torch.tensor([b_val], dtype=torch.float32, device=self.device)

        # 1. System 1 Forward Intuition
        initial_c_tensor, meta = self.forward_eval(a_tensor, op, b_tensor)
        initial_c = initial_c_tensor[0, 0].item() + injected_noise

        # 2. Counterfactual Verification
        current_c = initial_c
        correction_trace = []
        
        a_recon, surprise, is_consistent = self.verify_counterfactual(a_val, op, b_val, current_c)
        correction_trace.append({
            "step": 0,
            "candidate_c": current_c,
            "a_recon": a_recon,
            "surprise": surprise,
            "consistent": is_consistent
        })

        # 3. Iterative Self-Correction Loop
        step = 0
        inv_op = OPERATOR_SEMANTICS[op]["inverse"]
        while not is_consistent and step < self.max_correction_steps:
            step += 1
            # Compute causal gradient:
            # If a_recon < a_val: candidate_c was too small (for +) or too large (for -)
            # General rule: shift candidate_c by error in reconstructed input
            if op == "+":
                # c = a + b => a = c - b => if a_recon < a, c is too small
                error_delta = (a_val - a_recon)
                current_c = current_c + self.correction_lr * error_delta
            elif op == "-":
                # c = a - b => a = c + b => if a_recon < a, c is too small
                error_delta = (a_val - a_recon)
                current_c = current_c + self.correction_lr * error_delta
            elif op == "*":
                # c = a * b => a = c / b
                if abs(b_val) > 1e-4:
                    current_c = current_c + self.correction_lr * (a_val - a_recon) * b_val
            elif op == "%":
                # discrete cyclic step
                break

            a_recon, surprise, is_consistent = self.verify_counterfactual(a_val, op, b_val, current_c)
            correction_trace.append({
                "step": step,
                "candidate_c": current_c,
                "a_recon": a_recon,
                "surprise": surprise,
                "consistent": is_consistent
            })

        # 4. Generate Causal Explanation
        explanation = self.generate_causal_explanation(
            a_val, op, b_val, initial_c, current_c, a_recon, surprise, step
        )

        return {
            "initial_candidate": initial_c,
            "final_output": current_c,
            "ground_truth": self._ground_truth(a_val, op, b_val),
            "a_reconstructed": a_recon,
            "final_surprise": surprise,
            "is_verified": is_consistent,
            "correction_steps": step,
            "correction_trace": correction_trace,
            "explanation": explanation
        }

    def generate_causal_explanation(
        self,
        a: float,
        op: str,
        b: float,
        init_c: float,
        final_c: float,
        a_recon: float,
        surprise: float,
        steps: int
    ) -> str:
        """
        Generates structured causal natural language justification.
        """
        inv_op = OPERATOR_SEMANTICS[op]["inverse"]
        if steps == 0 and surprise <= self.surprise_threshold:
            return (
                f"Verified: {a:.2f} {op} {b:.2f} = {final_c:.2f} is causally sound. "
                f"Counterfactual rollback ({final_c:.2f} {inv_op} {b:.2f}) returned {a_recon:.2f} "
                f"(Inversion Surprise: {surprise:.4f} <= threshold {self.surprise_threshold})."
            )
        elif surprise <= self.surprise_threshold:
            return (
                f"Self-Corrected in {steps} mental steps: Initial hypothesis was {init_c:.2f}, "
                f"which failed rollback ({init_c:.2f} {inv_op} {b:.2f} != {a:.2f}). "
                f"Adjusted candidate to {final_c:.2f}, successfully restoring causal reversibility "
                f"(Inversion Surprise: {surprise:.4f})."
            )
        else:
            return (
                f"Unresolved Causal Paradox: Hypothesized {final_c:.2f} for {a:.2f} {op} {b:.2f}, "
                f"but inverse rollback yielded {a_recon:.2f} (Residual surprise: {surprise:.4f}). "
                f"Recommend System 2 algorithmic decomposition."
            )

    def _ground_truth(self, a: float, op: str, b: float) -> float:
        if op == "+":
            return a + b
        elif op == "-":
            return a - b
        elif op == "*":
            return a * b
        elif op == "%":
            return a % 5
        elif op == "/":
            return a / (b if b != 0 else 1.0)
        return 0.0
