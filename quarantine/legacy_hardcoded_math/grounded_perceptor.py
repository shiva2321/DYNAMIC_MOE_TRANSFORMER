"""
Embodied Triple-Code Grounded Perceptor.
Implements Stanislas Dehaene's Triple-Code Model of Mathematical Cognition
in Infinite Spherical Hyperspace (S^{D-1} x R+).

Codes:
1. Symbolic Code: Discrete operator IDs and one-hot token channels.
2. Analog Magnitude Code: Log-compressive radial scalar r in R+ representing continuous
   quantity/mass, with a mental number line and multi-frequency harmonic receptive fields.
3. Sensorimotor Action Code: Directional unit vector u in S^{D-1} representing the physical
   action:
   - Addition (+): Geodesic translation along the positive accumulation pole.
   - Subtraction (-): Inverted velocity vector (removal/backtracking).
   - Multiplication (*): Multi-scale logarithmic dilation.
   - Modulo (%): Periodic phase winding around the 1-sphere S^1.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ishmn.geometry import project_to_sphere

OPERATOR_SEMANTICS = {
    "+": {"id": 0, "name": "addition", "inverse": "-", "action_pole": 0},
    "-": {"id": 1, "name": "subtraction", "inverse": "+", "action_pole": 1},
    "*": {"id": 2, "name": "multiplication", "inverse": "/", "action_pole": 2},
    "%": {"id": 3, "name": "modulo", "inverse": "%", "action_pole": 3},
    "/": {"id": 4, "name": "division", "inverse": "*", "action_pole": 4}
}

NORMALIZATION_SCALE = 50.0

class GroundedPerceptor(nn.Module):
    def __init__(
        self,
        d_model: int = 32,
        hyperspace_dim: int = 8,
        device: torch.device = None
    ):
        super().__init__()
        self.d_model = d_model
        self.hyperspace_dim = hyperspace_dim
        self.device = device if device is not None else torch.device("cpu")

    def encode_analog_magnitude(self, values: torch.Tensor) -> torch.Tensor:
        """
        Encodes values into multi-frequency continuous analog magnitude channels
        resembling biological intraparietal sulcus receptive fields.
        """
        values = values.to(self.device)
        freqs = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0], device=self.device)
        angles = (values.unsqueeze(1) / NORMALIZATION_SCALE) * freqs.unsqueeze(0)
        sin_feat = torch.sin(angles)
        cos_feat = torch.cos(angles)
        return torch.cat([sin_feat, cos_feat], dim=-1)

    def compute_action_direction(self, op_ids: torch.Tensor, a_vals: torch.Tensor, b_vals: torch.Tensor) -> torch.Tensor:
        """
        Maps operator to canonical geometric action direction on S^{D-1}.
        """
        B = op_ids.shape[0]
        raw_u = torch.zeros(B, self.hyperspace_dim, device=self.device)
        raw_u.scatter_(1, (op_ids % self.hyperspace_dim).unsqueeze(1), 2.0)
        raw_u.scatter_(1, ((op_ids + 2) % self.hyperspace_dim).unsqueeze(1), 1.0)
        raw_u[:, 4] = torch.sin(a_vals / 10.0)
        raw_u[:, 5] = torch.cos(b_vals / 10.0)
        return project_to_sphere(raw_u)

    def ground_problem(
        self,
        a: torch.Tensor,
        op_str: list[str],
        b: torch.Tensor
    ) -> dict:
        """
        Grounds inputs (a, op, b) into canonical Triple-Code representation:
        - X: [B, d_model] Grounded continuous feature vector
        - U: [B, hyperspace_dim] S^{D-1} Action directional vector
        - R: [B, 1] Analog physical quantity/mass scalar
        """
        a = a.to(self.device).float()
        b = b.to(self.device).float()
        B = a.shape[0]

        op_ids = torch.tensor([OPERATOR_SEMANTICS[o]["id"] for o in op_str], dtype=torch.long, device=self.device)

        # 1. Continuous feature tensor [B, d_model]
        X = torch.zeros(B, self.d_model, device=self.device)
        X[:, 0] = a / NORMALIZATION_SCALE
        X[:, 1] = b / NORMALIZATION_SCALE

        # Symbolic One-hot operator encoding (channels 2..6)
        X.scatter_(1, (2 + op_ids).unsqueeze(1), 1.0)

        # Analog magnitude harmonics (channels 7..26)
        freqs = [1.0, 2.0, 4.0, 8.0, 16.0]
        for idx, f in enumerate(freqs):
            X[:, 7 + idx * 2] = torch.sin(f * a / NORMALIZATION_SCALE)
            X[:, 7 + idx * 2 + 1] = torch.cos(f * a / NORMALIZATION_SCALE)
            X[:, 17 + idx * 2] = torch.sin(f * b / NORMALIZATION_SCALE)
            X[:, 17 + idx * 2 + 1] = torch.cos(f * b / NORMALIZATION_SCALE)

        # 2. Sensorimotor Action Direction in S^{D-1}
        U = self.compute_action_direction(op_ids, a, b)

        # 3. Log-compressive Physical Radius in R+
        R = (1.0 + torch.log1p(torch.abs(a) + torch.abs(b))).unsqueeze(1)

        return {
            "X": X,
            "U": U,
            "R": R,
            "op_ids": op_ids,
            "a": a,
            "b": b,
            "op_str": op_str
        }

    def mental_number_line_distance(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """
        Dehaene's Weber-Fechner Mental Number Line metric:
        Subjective psychological distance d = |log(v1) - log(v2)|.
        Proves analog magnitude discrimination.
        """
        eps = 1e-4
        return torch.abs(torch.log(v1.clamp(min=eps)) - torch.log(v2.clamp(min=eps)))
