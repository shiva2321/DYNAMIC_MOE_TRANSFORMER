from collections import deque
from typing import Deque, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.complex_ops import ComplexLinear, ModReLU
from dmoe.fast_memory import ExactLdaFastMemory, MultiSlotExactLdaMemory
from dmoe.routing import SetGate


def first_triggered_reason(triggers: Sequence[Tuple[bool, str]]) -> Optional[str]:
    for fired, reason in triggers:
        if fired:
            return reason
    return None


class WithinBatchGapMonitor:
    """
    Tracks within-batch fit gap g = 1 - sum(L_post) / sum(L_pre) and
    pre-step loss plateau (adapted from spherical_hyperspace_memory/growmoe/gapgrow.py).
    """

    def __init__(self, window: int = 4):
        if window < 2:
            raise ValueError("window must be at least 2")
        self.window = window
        self.pairs: Deque[Tuple[float, float]] = deque(maxlen=window)
        self.pre_losses: Deque[float] = deque(maxlen=2 * window)
        self.age = 0

    def add(self, pre_loss: float, post_loss: float) -> None:
        self.pairs.append((float(pre_loss), float(post_loss)))
        self.pre_losses.append(float(pre_loss))
        self.age += 1

    def gap(self) -> Optional[float]:
        if len(self.pairs) < self.window:
            return None
        s_pre = sum(p for p, _ in self.pairs)
        s_post = sum(q for _, q in self.pairs)
        return None if s_pre <= 0.0 else 1.0 - s_post / s_pre

    def reset(self) -> None:
        self.pairs.clear()
        self.pre_losses.clear()
        self.age = 0


def _svd_factors(delta: torch.Tensor, energy_keep: float, max_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    u, s, vh = torch.linalg.svd(delta.double(), full_matrices=False)
    total_energy = float((s * s).sum().item())
    if total_energy <= 1e-18:
        r = 1
    else:
        cum_energy = torch.cumsum(s * s, dim=0) / total_energy
        r = int(torch.searchsorted(cum_energy, torch.tensor(energy_keep, dtype=torch.float64)).item()) + 1
    r = max(1, min(r, max_rank, s.numel()))
    a = (u[:, :r] * s[:r].unsqueeze(0)).to(dtype=delta.dtype)
    b = vh[:r, :].to(dtype=delta.dtype)
    return a, b


class DeltaComplexLinear(nn.Module):
    """
    Copy-on-Write ComplexLinear layer sharing a frozen parent ComplexLinear
    and learning/compressing a delta (DeltaW_r, DeltaW_i) via truncated SVD.
    """

    def __init__(self, base: ComplexLinear, mutation_std: float = 0.015):
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.has_bias = base.has_bias
        self.__dict__["_base"] = base

        self.delta_real: Optional[nn.Parameter] = nn.Parameter(
            torch.randn_like(base.weight_real) * mutation_std
        )
        self.delta_imag: Optional[nn.Parameter] = nn.Parameter(
            torch.randn_like(base.weight_imag) * mutation_std
        )
        self.a_real: Optional[nn.Parameter] = None
        self.b_real: Optional[nn.Parameter] = None
        self.a_imag: Optional[nn.Parameter] = None
        self.b_imag: Optional[nn.Parameter] = None

        if self.has_bias:
            self.bias_real = nn.Parameter(
                base.bias_real.detach().clone() + torch.randn_like(base.bias_real) * mutation_std
            )
            self.bias_imag = nn.Parameter(
                base.bias_imag.detach().clone() + torch.randn_like(base.bias_imag) * mutation_std
            )
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    @property
    def weight_real(self) -> torch.Tensor:
        base_wr = self.__dict__["_base"].weight_real.detach()
        if self.delta_real is not None:
            return base_wr + self.delta_real
        return base_wr + (self.a_real @ self.b_real)

    @property
    def weight_imag(self) -> torch.Tensor:
        base_wi = self.__dict__["_base"].weight_imag.detach()
        if self.delta_imag is not None:
            return base_wi + self.delta_imag
        return base_wi + (self.a_imag @ self.b_imag)

    def compress_svd(self, energy_keep: float = 0.99, max_rank: int = 8) -> None:
        if self.delta_real is None or self.delta_imag is None:
            return
        with torch.no_grad():
            ar, br = _svd_factors(self.delta_real.data, energy_keep, max_rank)
            ai, bi = _svd_factors(self.delta_imag.data, energy_keep, max_rank)
        self.delta_real = None
        self.delta_imag = None
        self.a_real = nn.Parameter(ar)
        self.b_real = nn.Parameter(br)
        self.a_imag = nn.Parameter(ai)
        self.b_imag = nn.Parameter(bi)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        wr = self.weight_real
        wi = self.weight_imag
        out_real = F.linear(z_real, wr) - F.linear(z_imag, wi)
        out_imag = F.linear(z_real, wi) + F.linear(z_imag, wr)
        if self.has_bias:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        return out_real, out_imag


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


class LocalLdaExpert(nn.Module):
    """
    Expert with an integrated, private closed-form LDA head and exemplar signature buffer.
    Adapted from spherical_hyperspace_memory (unimoe/pools.py UniPool & unimoe/heads.py).
    
    Each expert maintains:
      - Neural complex feedforward layers
      - An isolated MultiSlotExactLdaMemory or ExactLdaFastMemory instance (zero cross-talk with other experts)
      - An exemplar buffer of reference representations for SetGate exchangeability testing
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        enable_lda_head: bool = True,
        num_classes: int = 256,
        num_slots: int = 4,
        lda_shrink: float = 0.25,
        max_exemplars: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.enable_lda_head = enable_lda_head
        self.num_classes = num_classes
        self.num_slots = num_slots
        self.max_exemplars = max_exemplars

        self.neural_expert = ComplexExpert(d_model, d_ff)
        if enable_lda_head:
            # Slot dim is 2 * d_model when byte_residuals are present, or d_model
            self.lda_head = MultiSlotExactLdaMemory(
                d_in=2 * d_model,
                num_classes=num_classes,
                num_slots=num_slots,
                reg=lda_shrink,
            )
        else:
            self.lda_head = None

        self.register_buffer("exemplars", torch.zeros(0, d_model, dtype=torch.float32))

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.neural_expert(z_r, z_i)

    @torch.no_grad()
    def update_exemplars(self, latents: torch.Tensor) -> None:
        """Keep a rolling window of recent representations to serve as reference samples for SetGate."""
        flat = latents.detach().reshape(-1, self.d_model).float()
        if self.exemplars.numel() == 0:
            self.exemplars = flat[: self.max_exemplars].clone()
        else:
            combined = torch.cat([self.exemplars, flat], dim=0)
            if combined.size(0) > self.max_exemplars:
                combined = combined[-self.max_exemplars :]
            self.exemplars = combined.clone()

    @torch.no_grad()
    def absorb(self, slot_features: torch.Tensor, targets: torch.Tensor, ignore_index: int = 256) -> None:
        """Absorb training targets in closed form directly into this expert's private LDA head."""
        if self.lda_head is not None:
            self.lda_head.absorb_slots(slot_features, targets, ignore_index=ignore_index)

    @torch.no_grad()
    def predict_logits(self, slot_features: torch.Tensor) -> Optional[torch.Tensor]:
        """Predict closed-form logits from this expert's local LDA statistics."""
        if self.lda_head is not None:
            return self.lda_head.predict_slot_logits(slot_features)
        return None

    def clone_with_mutation(self, mutation_std: float = 0.02) -> "LocalLdaExpert":
        new_exp = LocalLdaExpert(
            d_model=self.d_model,
            d_ff=self.d_ff,
            enable_lda_head=self.enable_lda_head,
            num_classes=self.num_classes,
            num_slots=self.num_slots,
            max_exemplars=self.max_exemplars,
        ).to(self.neural_expert.linear1.weight_real.device)
        new_exp.neural_expert = self.neural_expert.clone_with_mutation(mutation_std=mutation_std)
        if self.exemplars.numel() > 0:
            new_exp.exemplars = self.exemplars.clone()
        return new_exp


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
        top_k: Optional[int] = None,
        spawn_patience: int = 1,
        enable_local_lda: bool = False,
        enable_setgate: bool = False,
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
        self.top_k = top_k
        self.spawn_patience = max(1, int(spawn_patience))
        self.enable_local_lda = enable_local_lda
        self.enable_setgate = enable_setgate
        self.novelty_streak = 0
        self.gap_monitor = WithinBatchGapMonitor(window=4)

        if enable_setgate:
            self.setgate = SetGate(d_model=d_model)
        else:
            self.setgate = None

        if enable_local_lda:
            self.experts = nn.ModuleList([
                LocalLdaExpert(d_model, d_ff, enable_lda_head=True)
                for _ in range(initial_experts)
            ])
        else:
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

    def _dispatch_sorted_topk(
        self, z_r: torch.Tensor, z_i: torch.Tensor, router_probs: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Token-sorted contiguous active-expert dispatch (adapted from universal_substrait/model/hyper_moe.py).
        Only tokens routed to expert j execute expert j's ComplexLinear weights.
        """
        bsz, seq_len, d_model = z_r.shape
        n_tokens = bsz * seq_len
        flat_r = z_r.reshape(n_tokens, d_model)
        flat_i = z_i.reshape(n_tokens, d_model)
        flat_probs = router_probs.reshape(n_tokens, -1)

        top_vals, top_idx = torch.topk(flat_probs, k=k, dim=-1)
        top_weights = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-9)

        flat_expert_ids = top_idx.reshape(-1)
        flat_weights = top_weights.reshape(-1, 1)
        token_indices = torch.arange(n_tokens, device=z_r.device).unsqueeze(1).expand(-1, k).reshape(-1)

        perm = torch.argsort(flat_expert_ids)
        sorted_experts = flat_expert_ids[perm]
        sorted_tokens = token_indices[perm]
        sorted_weights = flat_weights[perm]

        counts = torch.bincount(sorted_experts, minlength=len(self.experts)).tolist()
        out_r = torch.zeros_like(flat_r)
        out_i = torch.zeros_like(flat_i)

        cursor = 0
        for exp_id, count in enumerate(counts):
            if count == 0:
                continue
            tok_slice = sorted_tokens[cursor : cursor + count]
            w_slice = sorted_weights[cursor : cursor + count]
            exp_r, exp_i = self.experts[exp_id](flat_r[tok_slice], flat_i[tok_slice])
            out_r.index_add_(0, tok_slice, exp_r * w_slice)
            out_i.index_add_(0, tok_slice, exp_i * w_slice)
            cursor += count

        return out_r.reshape(bsz, seq_len, d_model), out_i.reshape(bsz, seq_len, d_model)

    def spawn_expert(self, source_idx: Optional[int] = None) -> int:
        device = self.expert_frozen.device
        if source_idx is None:
            source_idx = int(torch.argmax(self.expert_usage).item()) if len(self.experts) > 0 else 0

        if self.enable_local_lda:
            new_expert = LocalLdaExpert(
                d_model=self.d_model,
                d_ff=self.d_ff,
                enable_lda_head=True,
                num_classes=self.experts[source_idx].num_classes if hasattr(self.experts[source_idx], "num_classes") else 256,
                num_slots=self.experts[source_idx].num_slots if hasattr(self.experts[source_idx], "num_slots") else 4,
            ).to(device)
        else:
            new_expert = self.experts[source_idx].clone_with_mutation(mutation_std=0.015).to(device)

        self.experts.append(new_expert)
        new_idx = len(self.experts) - 1

        old_w = self.router_linear.weight.data
        old_b = self.router_linear.bias.data
        in_f = self.router_linear.in_features
        new_router = nn.Linear(in_f, len(self.experts)).to(device)
        with torch.no_grad():
            new_router.weight[: old_w.size(0)].copy_(old_w)
            new_router.bias[: old_b.size(0)].copy_(old_b)
            new_router.weight[old_w.size(0)].copy_(
                old_w[source_idx] + torch.randn(in_f, device=device) * 0.01
            )
            new_router.bias[old_b.size(0)].copy_(old_b[source_idx])
        self.router_linear = new_router

        self._set_expert_buffers(
            usage=torch.cat([self.expert_usage, torch.tensor([0.1], device=device)]),
            age=torch.cat([self.expert_age, torch.zeros(1, device=device)]),
            frozen=torch.cat([self.expert_frozen, torch.zeros(1, dtype=torch.bool, device=device)]),
        )
        return new_idx

    def freeze_expert(self, expert_idx: int) -> None:
        if 0 <= expert_idx < len(self.experts):
            for p in self.experts[expert_idx].parameters():
                p.requires_grad_(False)
            with torch.no_grad():
                self.expert_frozen[expert_idx] = True

    def forward(
        self, z_r: torch.Tensor, z_i: torch.Tensor, update_stats: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if getattr(self, "force_routed_expert", None) is not None:
            chosen_idx = min(self.force_routed_expert, len(self.experts) - 1)
            self.last_routed_expert = chosen_idx
            if update_stats and self.training:
                with torch.no_grad():
                    self.expert_usage.mul_(0.9)
                    self.expert_usage[chosen_idx].add_(0.1)
                    self.expert_age[chosen_idx].add_(1.0)
            exp_out_r, exp_out_i = self.experts[chosen_idx](z_r, z_i)
            return exp_out_r, exp_out_i, torch.tensor(0.0, device=z_r.device)

        if self.enable_setgate and self.setgate is not None:
            seq_lat = torch.sqrt(z_r**2 + z_i**2).mean(dim=1)
            expert_exemplars = [e.exemplars if hasattr(e, "exemplars") else None for e in self.experts]
            step_val = int(self.expert_age.sum().item())
            verdict, owner_idx, p_val = self.setgate.route_batch(seq_lat, expert_exemplars, step=step_val)

            if verdict == "known" and owner_idx >= 0:
                chosen_idx = owner_idx
            else:
                unfrozen = [i for i, f in enumerate(self.expert_frozen) if not f]
                if unfrozen:
                    chosen_idx = unfrozen[0]
                elif self.training:
                    chosen_idx = self.spawn_expert()
                else:
                    candidates = []
                    batch_cent = seq_lat.mean(dim=0)
                    for e_idx, e in enumerate(self.experts):
                        if hasattr(e, "exemplars") and e.exemplars.numel() > 0:
                            d = (e.exemplars.mean(dim=0) - batch_cent).norm().item()
                            candidates.append((d, e_idx))
                    chosen_idx = min(candidates, key=lambda c: c[0])[1] if candidates else 0

            self.last_routed_expert = chosen_idx
            self.last_routing_verdict = verdict
            self.last_routing_p_val = p_val

            if update_stats:
                self._last_entropy = torch.tensor(0.0, device=z_r.device)
                if self.training:
                    with torch.no_grad():
                        self.expert_usage.mul_(0.9)
                        self.expert_usage[chosen_idx].add_(0.1)
                        self.expert_age[chosen_idx].add_(1.0)
                        if not bool(self.expert_frozen[chosen_idx]) and hasattr(self.experts[chosen_idx], "update_exemplars"):
                            self.experts[chosen_idx].update_exemplars(seq_lat)

            exp_out_r, exp_out_i = self.experts[chosen_idx](z_r, z_i)
            return exp_out_r, exp_out_i, torch.tensor(0.0, device=z_r.device)

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

        if self.top_k is not None and 0 < self.top_k < len(self.experts):
            acc_r, acc_i = self._dispatch_sorted_topk(z_r, z_i, router_probs, k=self.top_k)
            return acc_r, acc_i, entropy

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
        self,
        last_entropy: Optional[float] = None,
        novelty_score: Optional[float] = None,
        task_loss: Optional[float] = None,
        tau_loss: Optional[float] = None,
    ) -> Tuple[bool, bool, str]:
        if last_entropy is None:
            last_entropy = self._last_entropy.item()

        spawned = False
        pruned = False
        action_log = []
        device = self.router_linear.weight.device

        candidate_reason = first_triggered_reason(
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
                (
                    task_loss is not None
                    and tau_loss is not None
                    and task_loss > tau_loss
                    and last_entropy > self.entropy_floor,
                    f"DualGate TaskLoss={task_loss if task_loss is not None else 0.0:.4f} > {tau_loss if tau_loss is not None else 0.0:.4f}",
                ),
            ]
        )

        if candidate_reason is not None:
            self.novelty_streak += 1
        else:
            self.novelty_streak = 0

        spawn_reason = candidate_reason if self.novelty_streak >= self.spawn_patience else None

        if spawn_reason is not None:
            self.novelty_streak = 0
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
