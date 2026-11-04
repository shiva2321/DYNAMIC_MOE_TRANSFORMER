from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.complex_ops import ComplexLinear, ModReLU


def first_triggered_reason(triggers: Sequence[Tuple[bool, str]]) -> Optional[str]:
    for fired, reason in triggers:
        if fired:
            return reason
    return None


class ComplexExpert(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = ComplexLinear(d_model, d_ff)
        self.act = ModReLU(d_ff)
        self.linear2 = ComplexLinear(d_ff, d_model)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h_r, h_i = self.linear1(z_r, z_i)
        h_r, h_i = self.act(h_r, h_i)
        return self.linear2(h_r, h_i)

    def clone_with_mutation(self, mutation_std: float = 0.02) -> "ComplexExpert":
        d_model = self.linear1.in_features
        d_ff = self.linear1.out_features
        new_expert = ComplexExpert(d_model, d_ff).to(self.linear1.weight_real.device)

        with torch.no_grad():
            new_expert.linear1.weight_real.copy_(
                self.linear1.weight_real + torch.randn_like(self.linear1.weight_real) * mutation_std
            )
            new_expert.linear1.weight_imag.copy_(
                self.linear1.weight_imag + torch.randn_like(self.linear1.weight_imag) * mutation_std
            )
            new_expert.linear1.bias_real.copy_(
                self.linear1.bias_real + torch.randn_like(self.linear1.bias_real) * mutation_std
            )
            new_expert.linear1.bias_imag.copy_(
                self.linear1.bias_imag + torch.randn_like(self.linear1.bias_imag) * mutation_std
            )

            new_expert.act.b.copy_(self.act.b + torch.randn_like(self.act.b) * mutation_std)

            new_expert.linear2.weight_real.copy_(
                self.linear2.weight_real + torch.randn_like(self.linear2.weight_real) * mutation_std
            )
            new_expert.linear2.weight_imag.copy_(
                self.linear2.weight_imag + torch.randn_like(self.linear2.weight_imag) * mutation_std
            )
            new_expert.linear2.bias_real.copy_(
                self.linear2.bias_real + torch.randn_like(self.linear2.bias_real) * mutation_std
            )
            new_expert.linear2.bias_imag.copy_(
                self.linear2.bias_imag + torch.randn_like(self.linear2.bias_imag) * mutation_std
            )

        return new_expert


class DynamicMorphicMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        initial_experts: int = 4,
        tau_split: float = 1.25,
        tau_death: float = 0.04,
        min_experts: int = 2,
        maturity_age: int = 200,
        tau_novelty: float = 2.5,
        entropy_floor: float = 0.3,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.tau_split = tau_split
        self.tau_death = tau_death
        self.min_experts = min_experts
        self.maturity_age = maturity_age
        self.tau_novelty = tau_novelty
        self.entropy_floor = entropy_floor

        self.experts = nn.ModuleList([ComplexExpert(d_model, d_ff) for _ in range(initial_experts)])
        self.router_linear = nn.Linear(2 * d_model, initial_experts)
        self.register_buffer("expert_usage", torch.ones(initial_experts) / initial_experts)
        self.register_buffer("expert_age", torch.zeros(initial_experts))
        self.register_buffer("expert_frozen", torch.zeros(initial_experts, dtype=torch.bool))
        self._last_entropy = torch.tensor(0.0)

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def _replace_buffer(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._buffers:
            del self._buffers[name]
        self.register_buffer(name, tensor)

    def _set_expert_buffers(self, usage: torch.Tensor, age: torch.Tensor, frozen: torch.Tensor) -> None:
        self._replace_buffer("expert_usage", usage)
        self._replace_buffer("expert_age", age)
        self._replace_buffer("expert_frozen", frozen)

    def forward(
        self, z_r: torch.Tensor, z_i: torch.Tensor, update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input = torch.cat([z_r, z_i], dim=-1)

        router_logits = self.router_linear(flat_input)
        router_probs = F.softmax(router_logits, dim=-1)

        mean_probs = router_probs.mean(dim=(0, 1))
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9))

        if update_stats:
            self._last_entropy = entropy.detach()
            if self.training:
                with torch.no_grad():
                    self.expert_usage.mul_(0.9).add_(mean_probs.detach() * 0.1)
                    self.expert_age.add_(1.0)

        acc_r = torch.zeros_like(z_r)
        acc_i = torch.zeros_like(z_i)

        for j, expert in enumerate(self.experts):
            weight_j = router_probs[:, :, j : j + 1]
            exp_out_r, exp_out_i = expert(z_r, z_i)
            acc_r = acc_r + weight_j * exp_out_r
            acc_i = acc_i + weight_j * exp_out_i

        return acc_r, acc_i, entropy

    def freeze_mature_experts(self) -> int:
        newly_frozen = 0
        for idx, expert in enumerate(self.experts):
            if not bool(self.expert_frozen[idx]) and self.expert_age[idx].item() >= self.maturity_age:
                for p in expert.parameters():
                    p.requires_grad_(False)
                with torch.no_grad():
                    self.expert_frozen[idx] = True
                newly_frozen += 1
        return newly_frozen

    def mutate_topology(
        self, last_entropy: Optional[float] = None, novelty_score: Optional[float] = None
    ) -> Tuple[bool, bool, str]:
        if last_entropy is None:
            last_entropy = self._last_entropy.item()

        spawned = False
        pruned = False
        action_log = []
        device = self.router_linear.weight.device

        spawn_reason = first_triggered_reason(
            [
                (
                    last_entropy > self.tau_split,
                    f"Routing entropy={last_entropy:.4f} > {self.tau_split:.4f}",
                ),
                (
                    novelty_score is not None
                    and novelty_score > self.tau_novelty
                    and last_entropy > self.entropy_floor,
                    f"Novelty={novelty_score if novelty_score is not None else 0.0:.4f} > {self.tau_novelty:.4f}",
                ),
            ]
        )

        if spawn_reason is not None:
            most_used_idx = int(torch.argmax(self.expert_usage).item())
            new_expert = self.experts[most_used_idx].clone_with_mutation(mutation_std=0.015).to(device)
            self.experts.append(new_expert)

            old_w = self.router_linear.weight.data
            old_b = self.router_linear.bias.data
            in_f = self.router_linear.in_features

            new_router = nn.Linear(in_f, len(self.experts)).to(device)
            with torch.no_grad():
                new_router.weight[: old_w.size(0)].copy_(old_w)
                new_router.bias[: old_b.size(0)].copy_(old_b)
                new_router.weight[old_w.size(0)].copy_(
                    old_w[most_used_idx] + torch.randn(in_f, device=device) * 0.01
                )
                new_router.bias[old_b.size(0)].copy_(old_b[most_used_idx])
            self.router_linear = new_router

            self._set_expert_buffers(
                usage=torch.cat(
                    [
                        self.expert_usage,
                        torch.tensor([self.expert_usage[most_used_idx].item() * 0.5], device=device),
                    ]
                ),
                age=torch.cat([self.expert_age, torch.zeros(1, device=device)]),
                frozen=torch.cat([self.expert_frozen, torch.zeros(1, dtype=torch.bool, device=device)]),
            )

            spawned = True
            action_log.append(
                f"Split parent expert #{most_used_idx} ({spawn_reason}) -> New Total: {len(self.experts)}"
            )

        elif len(self.experts) > self.min_experts:
            min_val, least_used_idx = torch.min(self.expert_usage, dim=0)
            if min_val.item() < self.tau_death:
                least_idx = int(least_used_idx.item())
                del self.experts[least_idx]

                keep_indices = [i for i in range(self.router_linear.out_features) if i != least_idx]
                old_w = self.router_linear.weight.data
                old_b = self.router_linear.bias.data
                in_f = self.router_linear.in_features

                new_router = nn.Linear(in_f, len(keep_indices)).to(device)
                with torch.no_grad():
                    new_router.weight.copy_(old_w[keep_indices])
                    new_router.bias.copy_(old_b[keep_indices])
                self.router_linear = new_router

                self._set_expert_buffers(
                    usage=self.expert_usage[keep_indices].clone(),
                    age=self.expert_age[keep_indices].clone(),
                    frozen=self.expert_frozen[keep_indices].clone(),
                )

                pruned = True
                action_log.append(f"Expunged dormant expert #{least_idx} -> New Total: {len(self.experts)}")

        log_str = "; ".join(action_log) if action_log else "Stable"
        return spawned, pruned, log_str
