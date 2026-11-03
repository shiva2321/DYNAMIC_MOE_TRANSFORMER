# SS-DM-MoE Phase 2a: Continual Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add catastrophic-forgetting mitigation (reservoir replay with distillation anchors + expert-maturity freezing) and novelty-aware expert/layer spawning to the existing Phase 1 `dmoe/` package, proven by a real retention test comparing against a no-mitigation baseline.

**Architecture:** Extend `GeometricCorrectiveManifold` with a novelty z-score (mean+variance EMA), extend `DynamicMorphicMoE` with expert aging/freezing and a second additive spawn trigger, extend `SelfStabilizingDynamicMorphicMoE.execute_graph_mutation` with the matching vertical trigger, then layer a new `ReplayBuffer` + `ContinualLearner` on top — all changes to existing files are additive/backward-compatible (default parameters preserve exact Phase 1 behavior).

**Tech Stack:** Python 3.14, PyTorch 2.15, pytest — same as Phase 1. Project root: `D:\Dynamic_MOE_Transformer`. Test commands use `./.venv/Scripts/python.exe` (git-bash style; swap to `.\.venv\Scripts\python.exe` in PowerShell).

Spec: `docs/superpowers/specs/2026-11-04-dmoe-phase2a-continual-learning-design.md`

**Note on scope vs. the spec's phrasing:** the spec's testing-plan section mentioned updating `test_model_mutation.py` and `test_end_to_end.py` because of the manifold interface change. On closer inspection of the actual Phase 1 code (not just the spec), only `tests/test_manifold.py` calls `GeometricCorrectiveManifold` directly and needs updating for the 3-tuple→4-tuple change; `test_model_mutation.py` and `test_end_to_end.py` only ever call the full model (`model(x)`), whose public `forward()` dict keeps all its existing keys and just gains one (`"novelty_score"`) — so they need new tests added (for the new novelty-driven trigger), not fixes for breakage. This plan reflects that; no task "fixes" either file, Task 3 just adds to `test_model_mutation.py`.

---

### Task 1: Novelty z-score in `GeometricCorrectiveManifold`

**Files:**
- Modify: `dmoe/manifold.py`
- Modify: `tests/test_manifold.py`

- [ ] **Step 1: Update the test file first (TDD — these will fail against the current 3-tuple-returning implementation)**

Replace the full contents of `tests/test_manifold.py`:

```python
import math

import torch

from dmoe.manifold import GeometricCorrectiveManifold


def _zero_predictor(m: GeometricCorrectiveManifold) -> None:
    with torch.no_grad():
        for p in [
            m.predictor_stage1.weight_real, m.predictor_stage1.weight_imag,
            m.predictor_stage1.bias_real, m.predictor_stage1.bias_imag,
            m.predictor_stage2.weight_real, m.predictor_stage2.weight_imag,
            m.predictor_stage2.bias_real, m.predictor_stage2.bias_imag,
        ]:
            p.zero_()


def test_output_shape_matches_input():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out_r, out_i, err, novelty = m(z_r, z_i)
    assert out_r.shape == z_r.shape
    assert out_i.shape == z_i.shape
    assert err.dim() == 0
    assert novelty.dim() == 0


def test_running_error_updates_only_in_training_mode():
    m = GeometricCorrectiveManifold(d_model=8)
    initial_mean = m.running_error_magnitude.item()
    initial_var = m.running_error_variance.item()

    m.eval()
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    m(z_r, z_i)
    assert m.running_error_magnitude.item() == initial_mean
    assert m.running_error_variance.item() == initial_var

    m.train()
    m(z_r, z_i)
    assert m.running_error_magnitude.item() != initial_mean
    assert m.running_error_variance.item() != initial_var


def test_error_is_non_negative():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(4, 2, 8) * 100
    z_i = torch.randn(4, 2, 8) * 100
    _, _, err, _ = m(z_r, z_i)
    assert err.item() >= 0.0


def test_gradients_flow_through_corrective_path():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    out_r, out_i, err, novelty = m(z_r, z_i)
    (out_r.sum() + out_i.sum() + err).backward()
    assert z_r.grad is not None
    assert not torch.isnan(z_r.grad).any()
    assert m.predictor_stage1.weight_real.grad is not None
    assert not torch.isnan(m.predictor_stage1.weight_real.grad).any()
    assert m.corrector_linear1.weight_real.grad is not None
    assert not torch.isnan(m.corrector_linear1.weight_real.grad).any()


def test_predictor_stages_are_named_clearly():
    m = GeometricCorrectiveManifold(d_model=8)
    assert hasattr(m, "predictor_stage1")
    assert hasattr(m, "predictor_stage2")


def test_novelty_score_is_detached_from_computation_graph():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    _, _, _, novelty = m(z_r, z_i)
    assert novelty.requires_grad is False


def test_novelty_score_matches_z_score_formula():
    m = GeometricCorrectiveManifold(d_model=4)
    _zero_predictor(m)
    with torch.no_grad():
        m.running_error_magnitude.fill_(0.5)
        m.running_error_variance.fill_(0.04)

    z_r = torch.full((1, 1, 4), 0.6)
    z_i = torch.zeros(1, 1, 4)

    _, _, batch_mean_error, novelty_score = m(z_r, z_i)

    expected_novelty = (batch_mean_error.detach() - 0.5) / (math.sqrt(0.04) + 1e-8)
    assert torch.allclose(novelty_score, expected_novelty, atol=1e-4)


def test_novelty_score_is_near_zero_when_batch_matches_running_mean():
    m = GeometricCorrectiveManifold(d_model=4)
    _zero_predictor(m)
    with torch.no_grad():
        error_norm_value = math.sqrt(0.6 ** 2 + 1e-8)
        m.running_error_magnitude.fill_(error_norm_value)
        m.running_error_variance.fill_(0.04)

    z_r = torch.full((1, 1, 4), 0.6)
    z_i = torch.zeros(1, 1, 4)

    _, _, _, novelty_score = m(z_r, z_i)
    assert abs(novelty_score.item()) < 1e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_manifold.py -v`
Expected: FAIL — `ValueError: too many values to unpack (expected 3)` on the tests that unpack 4 values from the current 3-tuple-returning `forward()`.

- [ ] **Step 3: Replace the implementation**

Replace the full contents of `dmoe/manifold.py`:

```python
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
        self.last_novelty_score = torch.tensor(0.0)

    def forward(
        self, z_r: torch.Tensor, z_i: torch.Tensor
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
        self.last_novelty_score = novelty_score

        if self.training:
            with torch.no_grad():
                deviation = batch_mean_error.detach() - prior_mean
                self.running_error_magnitude.mul_(self.momentum).add_(
                    batch_mean_error.detach() * (1.0 - self.momentum)
                )
                self.running_error_variance.mul_(self.momentum).add_((deviation ** 2) * (1.0 - self.momentum))

        c1_r, c1_i = self.corrector_linear1(delta_r, delta_i)
        c1_act_r, c1_act_i = self.act(c1_r, c1_i)
        corr_r, corr_i = self.corrector_linear2(c1_act_r, c1_act_i)

        corrected_r = z_r + corr_r
        corrected_i = z_i + corr_i

        return corrected_r, corrected_i, batch_mean_error, novelty_score
```

Note the `.detach().clone()` on `prior_mean`/`prior_var`: `.detach()` alone returns a tensor sharing the *same storage* as the buffer, so the in-place `.mul_()/.add_()` updates below it would silently corrupt the already-captured "prior" values if they weren't cloned.

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_manifold.py -v`
Expected: `9 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/manifold.py tests/test_manifold.py
git commit -m "feat: add novelty z-score to GeometricCorrectiveManifold

Extends the EMA mean of structural error with an EMA variance, giving
a real novelty signal (how surprising is this batch relative to
recent normal variation) instead of a blunt fixed-magnitude threshold.
Breaking change to forward()'s return signature (3-tuple -> 4-tuple),
isolated to this file and its direct test; dmoe/model.py and other
tests only consume the model-level dict and are unaffected."
```

---

### Task 2: Expert maturity, freezing, and novelty-aware spawning in `DynamicMorphicMoE`

**Files:**
- Modify: `dmoe/experts.py`
- Modify: `tests/test_experts.py`

- [ ] **Step 1: Add the new tests (append to the existing file — the existing tests stay and must keep passing unmodified, since this change is additive)**

Append to `tests/test_experts.py` (keep everything already in the file above this):

```python
def test_expert_age_increments_in_training_mode_only():
    moe = DynamicMorphicMoE(d_model=8, d_ff=16, initial_experts=3)
    moe.eval()
    z_r = torch.randn(2, 4, 8)
    z_i = torch.randn(2, 4, 8)
    moe(z_r, z_i)
    assert torch.equal(moe.expert_age, torch.zeros(3))

    moe.train()
    moe(z_r, z_i)
    assert torch.equal(moe.expert_age, torch.ones(3))
    moe(z_r, z_i)
    assert torch.equal(moe.expert_age, torch.full((3,), 2.0))


def test_freeze_mature_experts_freezes_only_experts_past_maturity_age():
    moe = DynamicMorphicMoE(d_model=8, d_ff=16, initial_experts=3, maturity_age=2)
    with torch.no_grad():
        moe.expert_age.copy_(torch.tensor([0.0, 2.0, 5.0]))

    newly_frozen = moe.freeze_mature_experts()

    assert newly_frozen == 2
    assert bool(moe.expert_frozen[0]) is False
    assert bool(moe.expert_frozen[1]) is True
    assert bool(moe.expert_frozen[2]) is True
    assert moe.experts[0].linear1.weight_real.requires_grad is True
    assert moe.experts[1].linear1.weight_real.requires_grad is False
    assert moe.experts[2].linear1.weight_real.requires_grad is False

    assert moe.freeze_mature_experts() == 0


def test_frozen_expert_parameters_do_not_update_after_optimizer_step():
    moe = DynamicMorphicMoE(d_model=8, d_ff=16, initial_experts=2, maturity_age=1)
    with torch.no_grad():
        moe.expert_age.copy_(torch.tensor([5.0, 5.0]))
    moe.freeze_mature_experts()

    before = moe.experts[0].linear1.weight_real.clone()
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-1)

    z_r = torch.randn(2, 4, 8)
    z_i = torch.randn(2, 4, 8)
    acc_r, acc_i, _ = moe(z_r, z_i)
    (acc_r.sum() + acc_i.sum()).backward()
    optimizer.step()

    after = moe.experts[0].linear1.weight_real
    assert torch.equal(before, after)


def test_mutate_topology_novelty_score_none_preserves_phase1_behavior():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=3, tau_split=1.0, tau_death=0.01, min_experts=2
    )
    with torch.no_grad():
        moe.router_linear.weight.zero_()
        moe.router_linear.bias.zero_()
    moe.train()
    z_r = torch.randn(2, 4, 8)
    z_i = torch.randn(2, 4, 8)
    moe(z_r, z_i)

    spawned, pruned, _ = moe.mutate_topology(novelty_score=None)

    assert spawned is True
    assert pruned is False
    assert moe.num_experts == 4


def test_mutate_topology_spawns_on_high_novelty_even_with_moderate_entropy():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=3,
        tau_split=10.0, tau_death=0.01, min_experts=2,
        tau_novelty=2.0, entropy_floor=0.1,
    )
    moe._last_entropy = torch.tensor(0.5)  # below tau_split, above entropy_floor

    spawned, pruned, _ = moe.mutate_topology(novelty_score=3.0)

    assert spawned is True
    assert pruned is False
    assert moe.num_experts == 4


def test_mutate_topology_high_novelty_but_entropy_below_floor_does_not_spawn():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=3,
        tau_split=10.0, tau_death=10.0, min_experts=2,
        tau_novelty=2.0, entropy_floor=0.5,
    )
    moe._last_entropy = torch.tensor(0.1)  # below entropy_floor

    spawned, pruned, _ = moe.mutate_topology(novelty_score=5.0)

    assert spawned is False
    assert moe.num_experts == 3


def test_mutate_topology_resizes_age_and_frozen_buffers_on_spawn_and_death():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=3, tau_split=1.0, tau_death=0.9, min_experts=2
    )
    moe._last_entropy = torch.tensor(5.0)
    moe.mutate_topology()
    assert moe.expert_age.shape == (4,)
    assert moe.expert_frozen.shape == (4,)
    assert moe.expert_age[-1].item() == 0.0
    assert bool(moe.expert_frozen[-1]) is False

    with torch.no_grad():
        moe.expert_usage[0] = 0.0001
    moe._last_entropy = torch.tensor(0.1)
    moe.mutate_topology()
    assert moe.expert_age.shape == (moe.num_experts,)
    assert moe.expert_frozen.shape == (moe.num_experts,)
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_experts.py -v`
Expected: the 7 pre-existing tests still PASS; the 7 new tests FAIL (`AttributeError: 'DynamicMorphicMoE' object has no attribute 'expert_age'` / unexpected keyword argument `maturity_age` / etc.)

- [ ] **Step 3: Replace the implementation**

Replace the full contents of `dmoe/experts.py`:

```python
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.complex_ops import ComplexLinear, ModReLU


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

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_input = torch.cat([z_r, z_i], dim=-1)

        router_logits = self.router_linear(flat_input)
        router_probs = F.softmax(router_logits, dim=-1)

        mean_probs = router_probs.mean(dim=(0, 1))
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-9))
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

        entropy_trigger = last_entropy > self.tau_split
        novelty_trigger = (
            novelty_score is not None
            and novelty_score > self.tau_novelty
            and last_entropy > self.entropy_floor
        )

        if entropy_trigger or novelty_trigger:
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

            new_usage = torch.cat(
                [self.expert_usage, torch.tensor([self.expert_usage[most_used_idx].item() * 0.5], device=device)]
            )
            self._replace_buffer("expert_usage", new_usage)

            new_age = torch.cat([self.expert_age, torch.zeros(1, device=device)])
            self._replace_buffer("expert_age", new_age)

            new_frozen = torch.cat([self.expert_frozen, torch.zeros(1, dtype=torch.bool, device=device)])
            self._replace_buffer("expert_frozen", new_frozen)

            spawned = True
            action_log.append(f"Split parent expert #{most_used_idx} -> New Total: {len(self.experts)}")

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

                new_usage = self.expert_usage[keep_indices].clone()
                self._replace_buffer("expert_usage", new_usage)

                new_age = self.expert_age[keep_indices].clone()
                self._replace_buffer("expert_age", new_age)

                new_frozen = self.expert_frozen[keep_indices].clone()
                self._replace_buffer("expert_frozen", new_frozen)

                pruned = True
                action_log.append(f"Expunged dormant expert #{least_idx} -> New Total: {len(self.experts)}")

        log_str = "; ".join(action_log) if action_log else "Stable"
        return spawned, pruned, log_str
```

- [ ] **Step 4: Run to verify all tests pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_experts.py -v`
Expected: `14 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/experts.py tests/test_experts.py
git commit -m "feat: add expert maturity freezing and novelty-aware spawning

Experts past maturity_age get requires_grad_(False) on every optimizer
cycle, so new patterns go to fresh capacity instead of slowly
overwriting settled experts (architecture-based forgetting defense).
mutate_topology gains an optional novelty_score argument opening a
second, additive spawn trigger (novel-and-confused, not just
overloaded) -- default None preserves exact Phase 1 behavior."
```

---

### Task 3: Novelty-driven vertical growth and `forward()` novelty passthrough in the full model

**Files:**
- Modify: `dmoe/model.py`
- Modify: `tests/test_model_mutation.py` (additions only — no existing test changes)

- [ ] **Step 1: Append new tests to `tests/test_model_mutation.py`**

Append to `tests/test_model_mutation.py` (keep everything already in the file above this):

```python
def test_vertical_growth_triggers_from_novelty_score_alone():
    model = _tiny_model(theta_up=10.0)  # make the error-magnitude trigger unreachable
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.0)
        model.manifold_corr.last_novelty_score = torch.tensor(5.0)
    report = model.execute_graph_mutation()
    assert model.num_layers == 3
    assert any("Novelty Z-Score" in log for log in report["mutation_logs"])


def test_execute_graph_mutation_accepts_explicit_novelty_override():
    model = _tiny_model(theta_up=10.0)
    report = model.execute_graph_mutation(novelty_score=5.0)
    assert model.num_layers == 3
    assert any("Novelty Z-Score" in log for log in report["mutation_logs"])


def test_forward_returns_novelty_score_in_output_dict():
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert "novelty_score" in out
    assert out["novelty_score"].dim() == 0
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_model_mutation.py -v`
Expected: the 9 pre-existing tests (8 passed + 1 skipped) are unaffected; the 3 new tests FAIL (`KeyError: 'novelty_score'` / `AttributeError` / assertion failures since the novelty trigger path doesn't exist yet).

- [ ] **Step 3: Replace the implementation**

Replace the full contents of `dmoe/model.py`:

```python
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.manifold import GeometricCorrectiveManifold
from dmoe.layer import DynamicMorphicTransformerLayer
from dmoe.decoder import UniversalProjectiveDecoder
from dmoe import checkpoint as checkpoint_module


class SelfStabilizingDynamicMorphicMoE(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        initial_layers: int = 2,
        initial_experts: int = 3,
        theta_up: float = 0.45,
        theta_down: float = 0.05,
        theta_novelty_vertical: float = 2.5,
        min_layers: int = 1,
        max_layers: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta_up = theta_up
        self.theta_down = theta_down
        self.theta_novelty_vertical = theta_novelty_vertical
        self.min_layers = min_layers
        self.max_layers = max_layers

        self.encoder = UniversalMultimodalEncoder(d_model)
        self.manifold_corr = GeometricCorrectiveManifold(d_model)

        self.layers = nn.ModuleList(
            [
                DynamicMorphicTransformerLayer(d_model, num_heads, d_ff, initial_experts=initial_experts)
                for _ in range(initial_layers)
            ]
        )

        self.decoder = UniversalProjectiveDecoder(d_model)

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def forward(self, x: torch.Tensor, output_head: str = "token") -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        z_r, z_i = self.encoder(x)
        z_r, z_i, structural_error, novelty_score = self.manifold_corr(z_r, z_i)

        entropy_traces = []
        for layer in self.layers:
            z_r, z_i, entropy = layer(z_r, z_i)
            entropy_traces.append(entropy)

        output = self.decoder(z_r, z_i, head=output_head)

        return {
            "output": output,
            "structural_error": structural_error,
            "novelty_score": novelty_score,
            "routing_entropies": entropy_traces,
            "z_real": z_r,
            "z_imag": z_i,
        }

    def execute_graph_mutation(
        self, novelty_score: Optional[float] = None
    ) -> Dict[str, Union[int, List[int], List[str]]]:
        mutation_logs = []
        current_error = self.manifold_corr.running_error_magnitude.item()
        current_novelty = (
            novelty_score if novelty_score is not None else self.manifold_corr.last_novelty_score.item()
        )
        device = next(self.parameters()).device

        vertical_error_trigger = current_error > self.theta_up
        vertical_novelty_trigger = current_novelty > self.theta_novelty_vertical

        if (vertical_error_trigger or vertical_novelty_trigger) and len(self.layers) < self.max_layers:
            new_layer = DynamicMorphicTransformerLayer(
                self.d_model,
                self.num_heads,
                self.d_ff,
                initial_experts=self.layers[-1].moe.num_experts,
            ).to(device)
            self.layers.append(new_layer)
            if vertical_error_trigger:
                mutation_logs.append(
                    f"[Vertical Growth] Structural Error={current_error:.4f} > {self.theta_up:.4f}. "
                    f"Appended Layer #{len(self.layers)}."
                )
            else:
                mutation_logs.append(
                    f"[Vertical Growth] Novelty Z-Score={current_novelty:.4f} > "
                    f"{self.theta_novelty_vertical:.4f}. Appended Layer #{len(self.layers)}."
                )

        elif len(self.layers) > self.min_layers:
            salience_values = [l.layer_salience.item() for l in self.layers]
            min_salience, min_idx = min((val, idx) for idx, val in enumerate(salience_values))

            if min_salience < self.theta_down:
                del self.layers[min_idx]
                mutation_logs.append(
                    f"[Vertical Pruning] Layer #{min_idx} salience={min_salience:.4f} < "
                    f"{self.theta_down:.4f}. Collapsed layer."
                )

        for idx, layer in enumerate(self.layers):
            spawned, pruned, log_msg = layer.moe.mutate_topology(novelty_score=current_novelty)
            if spawned or pruned:
                mutation_logs.append(f"[Horizontal Morph Layer {idx}] {log_msg}")

        return {
            "total_layers": len(self.layers),
            "experts_per_layer": [l.moe.num_experts for l in self.layers],
            "mutation_logs": mutation_logs,
        }

    def save_checkpoint(self, path: str) -> None:
        checkpoint_module.save_checkpoint(self, path)

    @classmethod
    def load_checkpoint(cls, path: str, device=None) -> "SelfStabilizingDynamicMorphicMoE":
        return checkpoint_module.load_checkpoint(path, cls, device=device)


def rebuild_optimizer_preserving_state(
    old_optimizer: torch.optim.Optimizer, model: nn.Module, **adamw_kwargs
) -> torch.optim.AdamW:
    id_to_state = {}
    for group in old_optimizer.param_groups:
        for p in group["params"]:
            if p in old_optimizer.state:
                id_to_state[id(p)] = old_optimizer.state[p]

    new_optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
    for group in new_optimizer.param_groups:
        for p in group["params"]:
            saved_state = id_to_state.get(id(p))
            if saved_state is not None:
                new_optimizer.state[p] = saved_state

    return new_optimizer
```

Why passing `novelty_score=current_novelty` (a real float, possibly `0.0`) to every layer's `mutate_topology` call doesn't break Phase 1 tests: those tests never call `model(x)` before forcing a mutation, so `last_novelty_score` stays at its `__init__` default of `0.0` throughout, and `0.0 > tau_novelty` (default `2.5`) is always `False` — the novelty trigger path stays closed exactly like it does when the argument is omitted entirely.

- [ ] **Step 4: Run to verify all tests pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_model_mutation.py tests/test_end_to_end.py -v`
Expected: `11 passed, 1 skipped` for `test_model_mutation.py` (8 original + 3 new, plus the CUDA skip) and `3 passed` for `test_end_to_end.py` (unchanged).

- [ ] **Step 5: Commit**

```bash
git add dmoe/model.py tests/test_model_mutation.py
git commit -m "feat: add novelty-driven vertical growth trigger to execute_graph_mutation

Adds an additive OR-condition alongside the existing theta_up error
check, and threads the manifold's novelty score into every layer's
horizontal mutate_topology call. forward()'s returned dict gains a
novelty_score key; existing keys and behavior are unchanged."
```

---

### Task 4: Reservoir-sampling `ReplayBuffer`

**Files:**
- Create: `tests/test_replay_buffer.py`
- Create: `dmoe/replay_buffer.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_replay_buffer.py`:

```python
import random

from dmoe.replay_buffer import ReplayBuffer


def test_buffer_keeps_all_items_under_capacity():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.add(i)
    assert len(buf) == 5


def test_buffer_caps_at_capacity():
    buf = ReplayBuffer(capacity=5)
    for i in range(100):
        buf.add(i)
    assert len(buf) == 5


def test_sample_returns_requested_count_of_distinct_items():
    buf = ReplayBuffer(capacity=20)
    for i in range(20):
        buf.add(i)
    sampled = buf.sample(5)
    assert len(sampled) == 5
    assert len(set(sampled)) == 5


def test_sample_returns_all_items_when_k_exceeds_buffer_size():
    buf = ReplayBuffer(capacity=20)
    for i in range(3):
        buf.add(i)
    sampled = buf.sample(10)
    assert sorted(sampled) == [0, 1, 2]


def test_reservoir_sampling_is_not_biased_toward_recency():
    random.seed(0)
    trials = 2000
    capacity = 10
    stream_length = 100
    early_item_survival = 0
    late_item_survival = 0

    for _ in range(trials):
        buf = ReplayBuffer(capacity=capacity)
        for i in range(stream_length):
            buf.add(i)
        if 0 in buf._items:
            early_item_survival += 1
        if (stream_length - 1) in buf._items:
            late_item_survival += 1

    early_rate = early_item_survival / trials
    late_rate = late_item_survival / trials
    assert abs(early_rate - late_rate) < 0.05
    assert abs(early_rate - capacity / stream_length) < 0.05
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_replay_buffer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.replay_buffer'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/replay_buffer.py`:

```python
import random
from typing import Any, List


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._items: List[Any] = []
        self._seen_count = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: Any) -> None:
        self._seen_count += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return
        replace_idx = random.randrange(self._seen_count)
        if replace_idx < self.capacity:
            self._items[replace_idx] = item

    def sample(self, k: int) -> List[Any]:
        if k >= len(self._items):
            return list(self._items)
        return random.sample(self._items, k)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_replay_buffer.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/replay_buffer.py tests/test_replay_buffer.py
git commit -m "feat: add reservoir-sampling replay buffer

Classic Algorithm R: the first `capacity` items are kept
unconditionally, and item n thereafter replaces a uniformly-random
slot with probability capacity/n -- a uniform sample over the whole
stream history, not a sliding window biased toward recent data."
```

---

### Task 5: `ContinualLearner`

**Files:**
- Create: `dmoe/continual.py`
- Create: `tests/test_continual_learner.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_continual_learner.py`:

```python
import torch

from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.continual import ContinualLearner


def _tiny_model():
    return SelfStabilizingDynamicMorphicMoE(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=1,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        min_layers=1,
        max_layers=2,
    )


def test_step_runs_and_returns_expected_metric_keys():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, optimizer, mutation_check_interval=100)

    x = torch.randint(0, 50, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 50, (2, 4))
    metrics = learner.step(x, targets, output_head="token")

    for key in ["task_loss", "rehearsal_loss", "distillation_loss", "total_loss", "novelty_score", "mutation_report"]:
        assert key in metrics
    assert metrics["mutation_report"] is None


def test_step_populates_replay_buffer():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, optimizer, replay_buffer_capacity=5, mutation_check_interval=100)

    for _ in range(3):
        x = torch.randint(0, 50, (2, 4), dtype=torch.long)
        targets = torch.randint(0, 50, (2, 4))
        learner.step(x, targets, output_head="token")

    assert len(learner.replay_buffer) == 3


def test_replay_loss_is_nonzero_once_buffer_has_items():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(
        model, optimizer, replay_buffer_capacity=10, replay_batch_size=2, mutation_check_interval=100
    )

    x = torch.randint(0, 50, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 50, (2, 4))
    learner.step(x, targets, output_head="token")
    metrics = learner.step(x, targets, output_head="token")

    assert metrics["rehearsal_loss"] >= 0.0


def test_mutation_check_only_fires_on_interval_boundary():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    learner = ContinualLearner(model, optimizer, mutation_check_interval=3)

    x = torch.randint(0, 50, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 50, (2, 4))

    reports = []
    for _ in range(6):
        metrics = learner.step(x, targets, output_head="token")
        reports.append(metrics["mutation_report"])

    assert reports[0] is None
    assert reports[1] is None
    assert reports[2] is not None
    assert reports[3] is None
    assert reports[4] is None
    assert reports[5] is not None
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_continual_learner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.continual'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/continual.py`:

```python
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.replay_buffer import ReplayBuffer


class ContinualLearner:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer_capacity: int = 500,
        replay_batch_size: int = 8,
        distillation_weight: float = 0.5,
        mutation_check_interval: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)
        self.replay_batch_size = replay_batch_size
        self.distillation_weight = distillation_weight
        self.mutation_check_interval = mutation_check_interval
        self.global_step = 0

    def step(self, x: torch.Tensor, targets: torch.Tensor, output_head: str = "token") -> Dict[str, Any]:
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(x, output_head=output_head)
        task_loss = self._task_loss(out["output"], targets, output_head)
        with torch.no_grad():
            latent_snapshot = self.model.decoder(out["z_real"], out["z_imag"], head="latent").detach().clone()

        rehearsal_loss = torch.tensor(0.0)
        distillation_loss = torch.tensor(0.0)
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
```

Note the efficiency choice: the latent snapshot (both for the newly-added sample and for the distillation target during replay) is computed by re-running only the cheap decoder `"latent"` head on `z_real`/`z_imag` that the main forward pass already produced — not a second full encoder+manifold+layers forward pass.

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_continual_learner.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/continual.py tests/test_continual_learner.py
git commit -m "feat: add ContinualLearner composing replay, distillation, and mutation checks

One step() call does: forward the new batch, sample and replay past
batches with a rehearsal loss plus a Dark-Experience-Replay-style
distillation loss against each sample's stored latent snapshot, one
combined backward/optimizer step, insert the new batch into the
reservoir buffer, and periodically run mutation + expert-freezing
checks."
```

---

### Task 6: Retention test — proving the mitigation actually works

**Files:**
- Create: `tests/test_continual_learning.py`

- [ ] **Step 1: Write the test**

Create `tests/test_continual_learning.py`:

```python
import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE
from dmoe.continual import ContinualLearner


def _retention_model():
    return SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=1,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        min_layers=1,
        max_layers=2,
    )


def _task_a_batch(batch_size=8, seq_len=6):
    tokens = torch.randint(0, 50, (batch_size, seq_len), dtype=torch.long)
    targets = (tokens + 1) % 50
    return tokens, targets


def _task_b_batch(batch_size=8, seq_len=6):
    tokens = torch.randint(100, 150, (batch_size, seq_len), dtype=torch.long)
    targets = 100 + (tokens * 2) % 50
    return tokens, targets


def _eval_task_a_loss(model, eval_x, eval_targets):
    model.eval()
    with torch.no_grad():
        out = model(eval_x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), eval_targets.view(-1))
    model.train()
    return loss.item()


def test_continual_learner_retains_task_a_better_than_naive_baseline():
    torch.manual_seed(0)
    mitigated = _retention_model()
    torch.manual_seed(0)
    baseline = _retention_model()

    eval_x, eval_targets = _task_a_batch(batch_size=16)

    mitigated_optimizer = torch.optim.AdamW(mitigated.parameters(), lr=5e-3)
    learner = ContinualLearner(
        mitigated,
        mitigated_optimizer,
        replay_buffer_capacity=200,
        replay_batch_size=8,
        distillation_weight=0.5,
        mutation_check_interval=1000,  # far beyond this test's step count: isolates the retention
    )                                   # effect from architecture mutation, which Task 3 already covers
    baseline_optimizer = torch.optim.AdamW(baseline.parameters(), lr=5e-3)

    train_steps_per_task = 40

    for _ in range(train_steps_per_task):
        x, targets = _task_a_batch()
        learner.step(x, targets, output_head="token")

        baseline_optimizer.zero_grad()
        out = baseline(x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss.backward()
        baseline_optimizer.step()

    mitigated_a_loss_after_a = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    baseline_a_loss_after_a = _eval_task_a_loss(baseline, eval_x, eval_targets)

    for _ in range(train_steps_per_task):
        x, targets = _task_b_batch()
        learner.step(x, targets, output_head="token")

        baseline_optimizer.zero_grad()
        out = baseline(x, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss.backward()
        baseline_optimizer.step()

    mitigated_a_loss_after_b = _eval_task_a_loss(mitigated, eval_x, eval_targets)
    baseline_a_loss_after_b = _eval_task_a_loss(baseline, eval_x, eval_targets)

    mitigated_forgetting = mitigated_a_loss_after_b - mitigated_a_loss_after_a
    baseline_forgetting = baseline_a_loss_after_b - baseline_a_loss_after_a

    assert mitigated_forgetting < baseline_forgetting
```

- [ ] **Step 2: Run it**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_continual_learning.py -v`
Expected: `1 passed`. This is a real trained-model comparison, not a forced/mocked one, so if it fails on the first run, that is genuine evidence the mechanism isn't yet strong enough at these settings — the fix is tuning `train_steps_per_task`, `lr`, or `distillation_weight` in this test (all three affect how much signal the mitigated model gets to actually learn and retain), not loosening the assertion. Re-run after any such change to confirm.

- [ ] **Step 3: Commit**

```bash
git add tests/test_continual_learning.py
git commit -m "test: add retention test proving replay+distillation reduces forgetting

Trains a ContinualLearner-driven model and a plain-optimizer-loop
baseline (same seed, same data) on two disjoint synthetic text rule
tasks; asserts the mitigated model's Task-A loss increase after
Task-B training is smaller than the baseline's -- a measured
comparison, not an assumption that forgetting is eliminated."
```

---

### Task 7: Export new symbols and final verification

**Files:**
- Modify: `dmoe/__init__.py`

- [ ] **Step 1: Update the package `__init__.py`**

Replace the full contents of `dmoe/__init__.py`:

```python
from dmoe.model import SelfStabilizingDynamicMorphicMoE, rebuild_optimizer_preserving_state
from dmoe.layer import DynamicMorphicTransformerLayer
from dmoe.experts import ComplexExpert, DynamicMorphicMoE
from dmoe.complex_ops import ComplexLinear, ModReLU, ComplexLayerNorm
from dmoe.attention import ComplexMultiHeadAttention
from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.manifold import GeometricCorrectiveManifold
from dmoe.decoder import UniversalProjectiveDecoder
from dmoe.checkpoint import save_checkpoint, load_checkpoint
from dmoe.replay_buffer import ReplayBuffer
from dmoe.continual import ContinualLearner

__all__ = [
    "SelfStabilizingDynamicMorphicMoE",
    "rebuild_optimizer_preserving_state",
    "DynamicMorphicTransformerLayer",
    "ComplexExpert",
    "DynamicMorphicMoE",
    "ComplexLinear",
    "ModReLU",
    "ComplexLayerNorm",
    "ComplexMultiHeadAttention",
    "UniversalMultimodalEncoder",
    "GeometricCorrectiveManifold",
    "UniversalProjectiveDecoder",
    "save_checkpoint",
    "load_checkpoint",
    "ReplayBuffer",
    "ContinualLearner",
]
```

- [ ] **Step 2: Sanity-check the package imports cleanly**

Run: `./.venv/Scripts/python.exe -c "import dmoe; print(dmoe.ContinualLearner, dmoe.ReplayBuffer)"`
Expected: prints both classes with no errors.

- [ ] **Step 3: Run the full suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all Phase 1 tests still pass unchanged, plus every test added in Tasks 1–6, 1 skipped (CUDA). No failures.

- [ ] **Step 4: Run the existing Phase 1 validation harness once more as a regression check**

Run: `./.venv/Scripts/python.exe main.py`
Expected: same five-phase trace as before, completing without errors — confirms Task 1–3's changes to `manifold.py`/`experts.py`/`model.py` didn't regress the Phase 1 demo.

- [ ] **Step 5: Final commit**

```bash
git add dmoe/__init__.py
git commit -m "$(cat <<'EOF'
Phase 2a complete: continual learning without catastrophic forgetting

Novelty z-score in Omega_corr, expert maturity freezing, novelty-aware
spawning, reservoir replay buffer, and ContinualLearner are all in
place and tested, including a real retention test comparing against
a no-mitigation baseline on disjoint synthetic tasks.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Plan self-review notes

- **Spec coverage:** novelty z-score (Task 1), expert maturity/freezing (Task 2), novelty-aware spawning for both horizontal (Task 2) and vertical (Task 3), `ReplayBuffer` (Task 4), `ContinualLearner` (Task 5), retention test (Task 6). The pinned Option C (router-specific EWC) and Phase 2b are explicitly out of scope per the spec and not included here.
- **Correction from the original task framing:** verified against the actual Phase 1 code (not just the spec) that `test_model_mutation.py` and `test_end_to_end.py` don't need breaking-change fixes — only additions. Documented at the top of this plan so whoever executes it doesn't go looking for breakage that isn't there.
- **Type/signature consistency:** `mutate_topology(last_entropy=None, novelty_score=None)` (Task 2) matches every call site — `execute_graph_mutation`'s `layer.moe.mutate_topology(novelty_score=current_novelty)` (Task 3) and the direct test calls in Task 2. `execute_graph_mutation(novelty_score=None)` (Task 3) matches `ContinualLearner.step()`'s `self.model.execute_graph_mutation(novelty_score=novelty_score)` (Task 5). `GeometricCorrectiveManifold.forward()`'s 4-tuple (Task 1) matches `model.py`'s unpacking (Task 3). `ReplayBuffer.sample()`'s return shape (list of 4-tuples) matches `ContinualLearner.step()`'s unpacking (Task 5).
- **No placeholders:** every step has complete, runnable code.
