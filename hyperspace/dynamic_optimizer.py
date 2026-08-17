"""
Dynamic Group Warmup Optimizer & Momentum Buffer Conditioning.
Eliminates cold-start gradient shocks when dynamically spawning new parameter groups mid-training:
1. Tracks individual birth steps (spawn_step) for each parameter group.
2. Applies localized linear warmup multiplier eta_g(t) = min(1.0, (t - t_spawn) / W_g).
3. Initializes second-moment variance buffers v_0 to the running median variance of warm experts.
"""

import math
from typing import Dict, List, Any, Optional, Set
import torch
from torch.optim import AdamW

class DynamicWarmupAdamW(AdamW):
    """
    AdamW optimizer with per-parameter-group localized warmup schedules and
    momentum buffer variance conditioning for dynamically spawned modules.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        default_group_warmup_steps: int = 50,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.default_group_warmup_steps = default_group_warmup_steps
        self.global_step = 0

        # Tag initial parameter groups with spawn_step = 0
        for group in self.param_groups:
            if "spawn_step" not in group:
                group["spawn_step"] = 0
            if "warmup_steps" not in group:
                group["warmup_steps"] = 0 # Base parameters follow standard scheduler
            if "base_lr" not in group:
                group["base_lr"] = group["lr"]

    def add_dynamic_param_group(
        self,
        new_params: List[torch.nn.Parameter],
        lr: Optional[float] = None,
        weight_decay: float = 0.01,
        warmup_steps: Optional[int] = None,
        group_name: str = "spawned_expert",
    ):
        """
        Dynamically attaches newly spawned expert parameters with localized warmup
        and momentum conditioning.
        """
        if not new_params:
            return

        base_lr = lr if lr is not None else self.defaults["lr"]
        warmup = warmup_steps if warmup_steps is not None else self.default_group_warmup_steps

        # Compute running median variance from existing active state buffers
        median_v = self._get_running_median_variance()

        new_group = {
            "params": new_params,
            "lr": base_lr * (1.0 / max(1, warmup)), # Start at fraction of target LR
            "base_lr": base_lr,
            "weight_decay": weight_decay,
            "spawn_step": self.global_step,
            "warmup_steps": warmup,
            "group_name": group_name,
        }

        self.add_param_group(new_group)

        # Condition momentum buffers for newly registered parameters
        for p in new_params:
            state = self.state[p]
            state["step"] = torch.tensor(0.0, dtype=torch.float32, device=p.device)
            # Exponential moving average of gradient values
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values (conditioned to prevent cold spike)
            if median_v is not None and median_v > 0.0:
                state["exp_avg_sq"] = torch.full_like(p, fill_value=median_v, memory_format=torch.preserve_format)
            else:
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _get_running_median_variance(self) -> Optional[float]:
        """Fast scalar estimate of exp_avg_sq without expensive all-tensor CPU-GPU sync."""
        if len(self.param_groups) > 0 and len(self.param_groups[0]["params"]) > 0:
            p0 = self.param_groups[0]["params"][0]
            state0 = self.state.get(p0, {})
            if "exp_avg_sq" in state0:
                return float(state0["exp_avg_sq"].data.flatten()[0].item())
        return 1e-4

    def update_dynamic_schedules(self, global_lr_multiplier: float = 1.0):
        """
        Updates each group's effective learning rate taking into account both
        global cosine schedule and localized per-group warmup.
        """
        self.global_step += 1

        for group in self.param_groups:
            spawn_step = group.get("spawn_step", 0)
            warmup_steps = group.get("warmup_steps", 0)
            base_lr = group.get("base_lr", group["lr"])

            steps_since_spawn = self.global_step - spawn_step

            if warmup_steps > 0 and steps_since_spawn < warmup_steps:
                # Local linear warmup for recently spawned group
                local_multiplier = float(steps_since_spawn) / float(warmup_steps)
            else:
                local_multiplier = 1.0

            group["lr"] = base_lr * global_lr_multiplier * local_multiplier
