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
        self.register_buffer("running_error_variance", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer("last_novelty_score", torch.tensor(0.0, dtype=torch.float32))

    def forward(
        self, z_r: torch.Tensor, z_i: torch.Tensor, update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p1_r, p1_i = self.predictor_stage1(z_r, z_i)
        pred_r, pred_i = self.predictor_stage2(p1_r, p1_i)

        delta_r = z_r - pred_r
        delta_i = z_i - pred_i

        error_norm = torch.sqrt(delta_r ** 2 + delta_i ** 2 + 1e-8)
        batch_mean_error = error_norm.mean()

        prior_mean = self.running_error_magnitude.detach().clone()
        prior_var = self.running_error_variance.detach().clone()
        novelty_score = (batch_mean_error.detach() - prior_mean) / (torch.sqrt(prior_var) + 1e-8)

        if update_stats:
            with torch.no_grad():
                self.last_novelty_score.copy_(novelty_score)
                if self.training:
                    deviation = batch_mean_error.detach() - prior_mean
                    self.running_error_magnitude.mul_(self.momentum).add_(
                        batch_mean_error.detach() * (1.0 - self.momentum)
                    )
                    self.running_error_variance.mul_(self.momentum).add_(
                        (deviation ** 2) * (1.0 - self.momentum)
                    )

        c1_r, c1_i = self.corrector_linear1(delta_r, delta_i)
        c1_act_r, c1_act_i = self.act(c1_r, c1_i)
        corr_r, corr_i = self.corrector_linear2(c1_act_r, c1_act_i)

        corrected_r = z_r + corr_r
        corrected_i = z_i + corr_i

        return corrected_r, corrected_i, batch_mean_error, novelty_score
