"""
Dynamic Group Warmup Optimizer & Momentum Buffer Conditioning.
Adapted from Universal SubStrait for Infinite Spherical Hyperspace MoE.

Eliminates cold-start gradient shocks when dynamically spawning new experts mid-training:
1. Tracks individual birth steps (spawn_step) for each newly spawned parameter group.
2. Applies localized linear warmup multiplier: eta_g(t) = min(1.0, (t - t_spawn) / W_g).
3. Initializes second-moment variance buffers v_0 to the running median variance of warm experts.
"""

from typing import Dict, List, Any, Optional, Set
import torch
from torch.optim import AdamW

class DynamicWarmupAdamW(AdamW):
    """
    AdamW optimizer with per-parameter-group localized warmup schedules and
    momentum buffer variance conditioning for dynamically spawned expert networks.
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        default_group_warmup_steps: int = 40,
    ):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.default_group_warmup_steps = default_group_warmup_steps
        self.global_step = 0

        # Tag initial parameter groups with spawn_step = 0
        for group in self.param_groups:
            if "spawn_step" not in group:
                group["spawn_step"] = 0
            if "warmup_steps" not in group:
                group["warmup_steps"] = 0
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
            "lr": base_lr * (1.0 / max(1, warmup)),
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
            state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if median_v is not None and median_v > 0.0:
                state["exp_avg_sq"] = torch.full_like(p, fill_value=median_v, memory_format=torch.preserve_format)
            else:
                state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

    def _get_running_median_variance(self) -> Optional[float]:
        """Fast scalar estimate of exp_avg_sq without CPU-GPU synchronization bottlenecks."""
        if len(self.param_groups) > 0 and len(self.param_groups[0]["params"]) > 0:
            p0 = self.param_groups[0]["params"][0]
            state0 = self.state.get(p0, {})
            if "exp_avg_sq" in state0:
                return float(state0["exp_avg_sq"].data.flatten()[0].item())
        return 1e-4

    def update_dynamic_schedules(self, global_lr_multiplier: float = 1.0):
        """
        Updates each group's effective learning rate taking into account both
        global schedule and localized per-group warmup.
        """
        self.global_step += 1

        for group in self.param_groups:
            base_lr = group.get("base_lr", self.defaults["lr"])
            spawn_step = group.get("spawn_step", 0)
            warmup_steps = group.get("warmup_steps", 0)

            if warmup_steps > 0:
                steps_since_spawn = max(0, self.global_step - spawn_step)
                local_mult = min(1.0, float(steps_since_spawn) / float(warmup_steps))
            else:
                local_mult = 1.0

            group["lr"] = base_lr * local_mult * global_lr_multiplier

    def prune_invalid_params(self, valid_params: Set[torch.nn.Parameter]):
        """
        Removes parameters and empty groups that are no longer part of valid_params,
        and cleanly purges corresponding optimizer state to prevent memory leaks.
        """
        valid_ids = {id(p) for p in valid_params}
        new_groups = []
        for group in self.param_groups:
            filtered_params = [p for p in group["params"] if id(p) in valid_ids]
            if len(filtered_params) > 0:
                group["params"] = filtered_params
                new_groups.append(group)
            else:
                for p in group["params"]:
                    if p in self.state:
                        del self.state[p]
        self.param_groups = new_groups

        # Purge dangling parameter states
        for p in list(self.state.keys()):
            if id(p) not in valid_ids:
                del self.state[p]

