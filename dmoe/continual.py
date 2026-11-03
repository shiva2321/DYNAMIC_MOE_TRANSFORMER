from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.model import rebuild_optimizer_preserving_state
from dmoe.replay_buffer import ReplayBuffer


class ContinualLearner:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer_capacity: int = 500,
        replay_batch_size: int = 8,
        distillation_weight: float = 0.5,
        maturity_age: Optional[int] = None,
        tau_novelty: Optional[float] = None,
        entropy_floor: Optional[float] = None,
        theta_novelty_vertical: Optional[float] = None,
        mutation_check_interval: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.replay_batch_size = replay_batch_size
        self.distillation_weight = distillation_weight
        self.maturity_age = maturity_age
        self.tau_novelty = tau_novelty
        self.entropy_floor = entropy_floor
        self.theta_novelty_vertical = theta_novelty_vertical
        self.mutation_check_interval = mutation_check_interval
        self.global_step = 0
        self._apply_thresholds()

    def _apply_thresholds(self) -> None:
        if self.theta_novelty_vertical is not None and hasattr(self.model, "theta_novelty_vertical"):
            self.model.theta_novelty_vertical = self.theta_novelty_vertical
        if hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if self.maturity_age is not None:
                    layer.moe.maturity_age = self.maturity_age
                if self.tau_novelty is not None:
                    layer.moe.tau_novelty = self.tau_novelty
                if self.entropy_floor is not None:
                    layer.moe.entropy_floor = self.entropy_floor

    def step(self, x: torch.Tensor, targets: torch.Tensor, output_head: str = "token") -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(x, output_head=output_head)
        task_loss = self._task_loss(out["output"], targets, output_head)
        with torch.no_grad():
            latent_snapshot = self.model.decoder(out["z_real"], out["z_imag"], head="latent").detach().clone()

        rehearsal_loss = torch.tensor(0.0, device=x.device)
        distillation_loss = torch.tensor(0.0, device=x.device)
        replay_samples = self.replay_buffer.sample(self.replay_batch_size)
        if replay_samples:
            rehearsal_losses = []
            distillation_losses = []
            for replay_x, replay_targets, replay_head, replay_latent in replay_samples:
                replay_out = self.model(replay_x, output_head=replay_head)
                rehearsal_losses.append(self._task_loss(replay_out["output"], replay_targets, replay_head))
                current_latent = self.model.decoder(replay_out["z_real"], replay_out["z_imag"], head="latent")
                distillation_losses.append(F.mse_loss(current_latent, replay_latent))
            rehearsal_loss = torch.stack(rehearsal_losses).mean()
            distillation_loss = torch.stack(distillation_losses).mean()

        total_loss = task_loss + rehearsal_loss + self.distillation_weight * distillation_loss
        total_loss.backward()
        self.optimizer.step()

        self.replay_buffer.add((x.detach().clone(), targets.detach().clone(), output_head, latent_snapshot))

        self.global_step += 1
        mutation_report = None
        if self.global_step % self.mutation_check_interval == 0:
            novelty_score = self.model.manifold_corr.last_novelty_score.item()
            mutation_report = self.model.execute_graph_mutation(novelty_score=novelty_score)
            self._apply_thresholds()
            if mutation_report["mutation_logs"]:
                self.optimizer = rebuild_optimizer_preserving_state(
                    self.optimizer,
                    self.model,
                    lr=self.optimizer.defaults["lr"],
                    weight_decay=self.optimizer.defaults.get("weight_decay", 0.01),
                )
            for layer in self.model.layers:
                layer.moe.freeze_mature_experts()

        return {
            "task_loss": task_loss.item(),
            "rehearsal_loss": rehearsal_loss.item(),
            "distillation_loss": distillation_loss.item(),
            "total_loss": total_loss.item(),
            "novelty_score": self.model.manifold_corr.last_novelty_score.item(),
            "mutation_report": mutation_report,
        }

    @staticmethod
    def _task_loss(output: torch.Tensor, targets: torch.Tensor, output_head: str) -> torch.Tensor:
        if output_head == "token":
            return F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))
        return F.mse_loss(output, targets)
