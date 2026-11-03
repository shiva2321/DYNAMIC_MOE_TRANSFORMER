from typing import Tuple

import torch
import torch.nn as nn

from dmoe.complex_ops import ComplexLinear, ModReLU


class GeometricCorrectiveManifold(nn.Module):
    def __init__(self, d_model: int, momentum: float = 0.95):
        super().__init__()
        self.d_model = d_model
        self.momentum = momentum

        self.predictor_stage1 = ComplexLinear(d_model, d_model)
        self.predictor_stage2 = ComplexLinear(d_model, d_model)

        self.corrector_linear1 = ComplexLinear(d_model, d_model)
        self.act = ModReLU(d_model)
        self.corrector_linear2 = ComplexLinear(d_model, d_model)

        self.register_buffer("running_error_magnitude", torch.tensor(0.5, dtype=torch.float32))

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1_r, p1_i = self.predictor_stage1(z_r, z_i)
        pred_r, pred_i = self.predictor_stage2(p1_r, p1_i)

        delta_r = z_r - pred_r
        delta_i = z_i - pred_i

        error_norm = torch.sqrt(delta_r ** 2 + delta_i ** 2 + 1e-8)
        batch_mean_error = error_norm.mean()

        if self.training:
            with torch.no_grad():
                self.running_error_magnitude.mul_(self.momentum).add_(
                    batch_mean_error.detach() * (1.0 - self.momentum)
                )

        c1_r, c1_i = self.corrector_linear1(delta_r, delta_i)
        c1_act_r, c1_act_i = self.act(c1_r, c1_i)
        corr_r, corr_i = self.corrector_linear2(c1_act_r, c1_act_i)

        corrected_r = z_r + corr_r
        corrected_i = z_i + corr_i

        return corrected_r, corrected_i, batch_mean_error
