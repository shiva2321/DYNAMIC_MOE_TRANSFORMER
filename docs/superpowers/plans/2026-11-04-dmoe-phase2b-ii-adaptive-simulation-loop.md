# SS-DM-MoE Phase 2b-ii: Adaptive Internal Simulation Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, stability-hardened simulation loop to `SelfStabilizingDynamicMorphicMoE` that re-runs `manifold_corr` + the last layer for extra iterations — randomized depth during training, energy-based halting during inference — with zero effect on any model built with the feature left off (the default).

**Architecture:** All changes live in `dmoe/model.py` and `dmoe/checkpoint.py`. No changes to `layer.py`, `manifold.py`, `experts.py`, `encoder.py`, `decoder.py`, or `complex_ops.py` — the loop is pure orchestration reusing those modules' existing public forward passes, stabilized by a learned damped (under-relaxed) fixed-point update.

**Tech Stack:** Python 3.14, PyTorch 2.15, pytest. Project root: `D:\Dynamic_MOE_Transformer`. Test commands use `./.venv/Scripts/python.exe` (git-bash style; swap to `.\.venv\Scripts\python.exe` in PowerShell).

Spec: `docs/superpowers/specs/2026-11-04-dmoe-phase2b-ii-adaptive-simulation-loop-design.md`

**Concurrency note:** Phase 2b-i's execution (`dmoe/jepa.py`, `tests/test_jepa.py`, `dmoe/__init__.py`) may still be landing in parallel. This plan touches `dmoe/model.py`, `dmoe/checkpoint.py`, `tests/test_simulation_loop.py` (new), and `tests/test_model_mutation.py` — none of which Phase 2b-i's plan touches, so there's no file-level conflict. This plan does **not** touch `dmoe/__init__.py` at all: no new top-level class or function is introduced (`enable_simulation_loop` and friends are just new constructor parameters on the existing, already-exported `SelfStabilizingDynamicMorphicMoE`).

---

### Task 1: Core loop — constructor flag, damped update, training/eval behavior, skip-hint

**Files:**
- Create: `tests/test_simulation_loop.py`
- Modify: `dmoe/model.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_simulation_loop.py`:

```python
import torch

from dmoe.model import SelfStabilizingDynamicMorphicMoE


def _sim_model(**overrides):
    kwargs = dict(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=2,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        min_layers=1,
        max_layers=2,
        enable_simulation_loop=True,
        max_simulation_steps=4,
        theta_converge=0.1,
        min_improvement=0.01,
        skip_threshold=0.0,
    )
    kwargs.update(overrides)
    return SelfStabilizingDynamicMorphicMoE(**kwargs)


def test_disabled_by_default_matches_phase1_behavior():
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=8, num_heads=2, d_ff=16, initial_layers=1, initial_experts=2
    )
    assert model.enable_simulation_loop is False
    assert not hasattr(model, "loop_damping_logit")

    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 0


def test_training_mode_uses_randomized_step_count_within_bounds():
    # skip_threshold set absurdly low so the skip-hint never fires here --
    # this test is isolating randomized-depth behavior, not the skip-hint.
    model = _sim_model(max_simulation_steps=4, skip_threshold=-100.0)
    model.train()

    seen_counts = set()
    x = torch.randint(0, 100, (2, 4), dtype=torch.long)
    for _ in range(30):
        out = model(x)
        count = out["simulation_steps_used"]
        assert 1 <= count <= 4
        seen_counts.add(count)

    assert len(seen_counts) > 1


def test_eval_mode_halts_immediately_when_error_already_below_convergence_threshold():
    # theta_converge set absurdly high so the very first iteration's error
    # is guaranteed to be below it, regardless of the actual numeric value.
    model = _sim_model(max_simulation_steps=4, theta_converge=1000.0, skip_threshold=-100.0)
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 1


def test_eval_mode_runs_to_max_steps_when_error_stays_high_and_does_not_improve_enough():
    # theta_converge and min_improvement set unreachable so neither halting
    # condition can fire -- the loop must run the full max_simulation_steps.
    model = _sim_model(
        max_simulation_steps=4, theta_converge=-1000.0, min_improvement=-1000.0, skip_threshold=-100.0
    )
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 4


def test_skip_hint_prevents_loop_even_when_enabled():
    # skip_threshold set absurdly high so any real computed novelty score
    # falls at or below it, regardless of the actual numeric value.
    model = _sim_model(skip_threshold=1000.0)
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)
    assert out["simulation_steps_used"] == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_simulation_loop.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'enable_simulation_loop'`

- [ ] **Step 3: Replace the implementation**

Replace the full contents of `dmoe/model.py`:

```python
import random
from typing import Dict, List, Optional, Tuple, Union

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
        enable_simulation_loop: bool = False,
        max_simulation_steps: int = 4,
        theta_converge: float = 0.1,
        min_improvement: float = 0.01,
        skip_threshold: float = 0.0,
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
        self.enable_simulation_loop = enable_simulation_loop
        self.max_simulation_steps = max_simulation_steps
        self.theta_converge = theta_converge
        self.min_improvement = min_improvement
        self.skip_threshold = skip_threshold

        self.encoder = UniversalMultimodalEncoder(d_model)
        self.manifold_corr = GeometricCorrectiveManifold(d_model)

        self.layers = nn.ModuleList(
            [
                DynamicMorphicTransformerLayer(d_model, num_heads, d_ff, initial_experts=initial_experts)
                for _ in range(initial_layers)
            ]
        )

        self.decoder = UniversalProjectiveDecoder(d_model)

        if self.enable_simulation_loop:
            self.loop_damping_logit = nn.Parameter(torch.tensor(0.0))

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def _run_simulation_loop(
        self, z_r: torch.Tensor, z_i: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.manifold_corr.last_novelty_score.item() <= self.skip_threshold:
            return z_r, z_i, 0

        damping = torch.sigmoid(self.loop_damping_logit)
        steps_used = 0

        if self.training:
            steps = random.randint(1, self.max_simulation_steps)
            for _ in range(steps):
                corrected_r, corrected_i, _, _ = self.manifold_corr(z_r, z_i)
                candidate_r, candidate_i, _ = self.layers[-1](corrected_r, corrected_i)
                z_r = z_r + damping * (candidate_r - z_r)
                z_i = z_i + damping * (candidate_i - z_i)
                steps_used += 1
        else:
            previous_error = None
            for _ in range(self.max_simulation_steps):
                corrected_r, corrected_i, structural_error, _ = self.manifold_corr(z_r, z_i)
                candidate_r, candidate_i, _ = self.layers[-1](corrected_r, corrected_i)
                z_r = z_r + damping * (candidate_r - z_r)
                z_i = z_i + damping * (candidate_i - z_i)
                steps_used += 1

                current_error = structural_error.item()
                if current_error < self.theta_converge:
                    break
                if previous_error is not None and (previous_error - current_error) < self.min_improvement:
                    break
                previous_error = current_error

        return z_r, z_i, steps_used

    def forward(self, x: torch.Tensor, output_head: str = "token") -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        z_r, z_i = self.encoder(x)
        z_r, z_i, structural_error, novelty_score = self.manifold_corr(z_r, z_i)

        entropy_traces = []
        for layer in self.layers:
            z_r, z_i, entropy = layer(z_r, z_i)
            entropy_traces.append(entropy)

        simulation_steps_used = 0
        if self.enable_simulation_loop:
            z_r, z_i, simulation_steps_used = self._run_simulation_loop(z_r, z_i)

        output = self.decoder(z_r, z_i, head=output_head)

        return {
            "output": output,
            "structural_error": structural_error,
            "novelty_score": novelty_score,
            "routing_entropies": entropy_traces,
            "z_real": z_r,
            "z_imag": z_i,
            "simulation_steps_used": simulation_steps_used,
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

The only changes from the current file: `import random`, the five new
constructor parameters and their storage, the conditional
`loop_damping_logit` parameter, the new `_run_simulation_loop` method, and
the three new lines in `forward()` (the `simulation_steps_used`
initialization, the conditional call, and the new dict key). Everything
else — `execute_graph_mutation`, `save_checkpoint`, `load_checkpoint`,
`rebuild_optimizer_preserving_state` — is unchanged from Phase 2a.

- [ ] **Step 4: Run to verify all tests pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_simulation_loop.py -v`
Expected: `5 passed`

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: every pre-existing test (Phase 1, 2a, 2b-i) still passes
unchanged, plus the 5 new ones here. `enable_simulation_loop` defaults to
`False` everywhere else, so nothing about this change should touch any
existing test's behavior.

- [ ] **Step 6: Commit**

```bash
git add dmoe/model.py tests/test_simulation_loop.py
git commit -m "feat: add opt-in adaptive internal simulation loop

Re-runs manifold_corr + the last layer for extra iterations via a
learned, sigmoid-bounded damping factor (under-relaxed fixed-point
update -- the Fixed-Point-Reasoners stability fix), with randomized
iteration depth during training and energy-based halting (using
Omega_corr's own structural_error as the convergence signal) during
inference. Off by default (enable_simulation_loop=False); when off,
no new parameters are created and forward() is byte-for-byte
unchanged from Phase 2a."
```

---

### Task 2: Stability under forced excessive iterations, and gradient flow

**Files:**
- Modify: `tests/test_simulation_loop.py` (append only)

- [ ] **Step 1: Append the stability and gradient tests**

Append to `tests/test_simulation_loop.py` (keep everything already there):

```python
def test_damped_loop_stays_bounded_under_forced_excessive_iterations():
    # Forces far more iterations than any sane trained cap (20, vs. the
    # 4-ish range this phase trains with) and asserts the damping
    # mechanism keeps the latent state finite and bounded -- direct
    # evidence against the divergence this whole design exists to avoid.
    model = _sim_model(
        max_simulation_steps=20, theta_converge=-1000.0, min_improvement=-1000.0, skip_threshold=-100.0
    )
    model.eval()
    x = torch.randint(0, 100, (2, 6), dtype=torch.long)
    out = model(x)

    assert out["simulation_steps_used"] == 20
    assert torch.isfinite(out["z_real"]).all()
    assert torch.isfinite(out["z_imag"]).all()
    assert out["z_real"].abs().max().item() < 1e4
    assert out["z_imag"].abs().max().item() < 1e4


def test_gradients_flow_through_damped_loop_including_damping_parameter():
    model = _sim_model(max_simulation_steps=3, skip_threshold=-100.0)
    model.train()
    x = torch.randint(0, 100, (2, 4), dtype=torch.long)

    out = model(x, output_head="token")
    loss = out["output"].sum()
    loss.backward()

    assert model.loop_damping_logit.grad is not None
    assert not torch.isnan(model.loop_damping_logit.grad).any()
    assert model.layers[-1].attn.q_proj.weight_real.grad is not None
    assert not torch.isnan(model.layers[-1].attn.q_proj.weight_real.grad).any()
```

- [ ] **Step 2: Run to verify both pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_simulation_loop.py -v`
Expected: `7 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation_loop.py
git commit -m "test: verify the simulation loop stays bounded under forced excessive iterations

Direct evidence the damped fixed-point update prevents divergence,
plus confirmation gradients (including the damping parameter itself)
flow correctly through a training-mode loop."
```

---

### Task 3: Checkpoint persistence for the new config and the damping parameter

**Files:**
- Modify: `dmoe/checkpoint.py`
- Modify: `tests/test_model_mutation.py` (append only)

- [ ] **Step 1: Append the checkpoint round-trip test**

Append to `tests/test_model_mutation.py` (keep everything already there —
`SelfStabilizingDynamicMorphicMoE` is already imported at the top of this
file from Phase 1):

```python
def test_checkpoint_round_trip_with_simulation_loop_enabled(tmp_path):
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=2,
        initial_experts=2,
        theta_up=100.0,
        theta_down=-1.0,
        min_layers=1,
        max_layers=2,
        enable_simulation_loop=True,
        max_simulation_steps=3,
        theta_converge=0.1,
        min_improvement=0.01,
        skip_threshold=-100.0,
    )
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out_before = model(x)["output"]

    path = tmp_path / "ckpt_sim_loop.pt"
    model.save_checkpoint(str(path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(path))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)["output"]

    assert loaded.enable_simulation_loop is True
    assert hasattr(loaded, "loop_damping_logit")
    assert torch.equal(model.loop_damping_logit, loaded.loop_damping_logit)
    assert torch.allclose(out_before, out_after, atol=1e-6)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_model_mutation.py -k simulation_loop -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'enable_simulation_loop'`
(the same error as Task 1, since `checkpoint.py` doesn't yet pass these
through, but this time surfacing from inside `load_checkpoint`'s model
reconstruction).

- [ ] **Step 3: Replace the implementation**

Replace the full contents of `dmoe/checkpoint.py`:

```python
from typing import Optional

import torch
import torch.nn as nn

from dmoe.layer import DynamicMorphicTransformerLayer


def save_checkpoint(model: nn.Module, path: str) -> None:
    config = {
        "d_model": model.d_model,
        "num_heads": model.num_heads,
        "d_ff": model.d_ff,
        "theta_up": model.theta_up,
        "theta_down": model.theta_down,
        "theta_novelty_vertical": model.theta_novelty_vertical,
        "min_layers": model.min_layers,
        "max_layers": model.max_layers,
        "enable_simulation_loop": model.enable_simulation_loop,
        "max_simulation_steps": model.max_simulation_steps,
        "theta_converge": model.theta_converge,
        "min_improvement": model.min_improvement,
        "skip_threshold": model.skip_threshold,
    }
    experts_per_layer = [layer.moe.num_experts for layer in model.layers]
    numeric_projector_dims = [int(k) for k in model.encoder.numeric_projectors.keys()]
    torch.save(
        {
            "config": config,
            "experts_per_layer": experts_per_layer,
            "numeric_projector_dims": numeric_projector_dims,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model_cls, device: Optional[torch.device] = None) -> nn.Module:
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    config = payload["config"]
    experts_per_layer = payload["experts_per_layer"]
    numeric_projector_dims = payload.get("numeric_projector_dims", [])

    model = model_cls(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        initial_layers=len(experts_per_layer),
        initial_experts=experts_per_layer[0] if experts_per_layer else 1,
        theta_up=config["theta_up"],
        theta_down=config["theta_down"],
        theta_novelty_vertical=config.get("theta_novelty_vertical", 2.5),
        min_layers=config["min_layers"],
        max_layers=config["max_layers"],
        enable_simulation_loop=config.get("enable_simulation_loop", False),
        max_simulation_steps=config.get("max_simulation_steps", 4),
        theta_converge=config.get("theta_converge", 0.1),
        min_improvement=config.get("min_improvement", 0.01),
        skip_threshold=config.get("skip_threshold", 0.0),
    )

    rebuilt_layers = nn.ModuleList(
        [
            DynamicMorphicTransformerLayer(config["d_model"], config["num_heads"], config["d_ff"], initial_experts=n)
            for n in experts_per_layer
        ]
    )
    model.layers = rebuilt_layers
    for f_dim in numeric_projector_dims:
        model.encoder._get_numeric_projector(f_dim, torch.device("cpu"))
    model.load_state_dict(payload["state_dict"])

    for layer in model.layers:
        for idx, expert in enumerate(layer.moe.experts):
            if bool(layer.moe.expert_frozen[idx]):
                for p in expert.parameters():
                    p.requires_grad_(False)

    if device is not None:
        model = model.to(device)

    return model
```

The only changes from the current file: five new `config` entries in
`save_checkpoint`, and the matching five new constructor keyword arguments
(each with a `.get(..., default)` fallback so checkpoints saved *before*
this phase still load correctly) in `load_checkpoint`. Passing
`enable_simulation_loop=True` through the constructor is what makes
`loop_damping_logit` exist on the reconstructed model *before*
`load_state_dict` runs — without this, loading a checkpoint saved with the
loop enabled would fail with a missing-key error.

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_model_mutation.py -v`
Expected: all pre-existing tests in this file still pass, plus the new one
(one more than Phase 2a's count).

- [ ] **Step 5: Commit**

```bash
git add dmoe/checkpoint.py tests/test_model_mutation.py
git commit -m "feat: persist simulation-loop config and damping parameter in checkpoints

Without this, a checkpoint saved with enable_simulation_loop=True
would fail to load: the reconstructed model needs the flag passed
through its constructor so loop_damping_logit exists before
load_state_dict runs. .get(..., default) fallbacks keep checkpoints
saved before this phase loading correctly."
```

---

### Task 4: Full verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete pytest suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: every test across Phase 1, Phase 2a, Phase 2b-i, and this phase
passes, 1 skipped for CUDA. No failures or collection errors.

- [ ] **Step 2: Run the Phase 1 validation harness as a final sanity check**

Run: `./.venv/Scripts/python.exe main.py`
Expected: the same five-phase trace as always, completing without errors —
confirms this phase's changes to `model.py`/`checkpoint.py` (both used by
every existing model instance, even with the new feature off) didn't
regress anything.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "$(cat <<'EOF'
Phase 2b-ii complete: adaptive internal simulation loop

Opt-in, off by default. Re-runs Omega_corr + the last layer for extra
iterations with a learned damped fixed-point update for stability,
randomized depth during training, and energy-based halting (driven
by Omega_corr's own structural_error, not a separate learned halting
network) during inference. Validated: regression guard against the
disabled default, randomized training depth, both eval-mode halting
paths, the novelty skip-hint, stability under forced excessive
iterations, gradient flow, and checkpoint round-trip.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Plan self-review notes

- **Spec coverage:** constructor flag and new parameters, damped loop,
  training-mode randomized depth, eval-mode energy-based halting, novelty
  skip-hint (Task 1); stability-under-excess and gradient-flow validation
  (Task 2); checkpoint persistence (Task 3). Every item in the spec's
  testing plan has a corresponding test. The explicitly deferred items
  (default-on, learned halting network, `ContinualLearner`/`JEPAModule`
  integration) are correctly absent from every task.
- **Type/signature consistency:** `_run_simulation_loop`'s return type
  (`Tuple[torch.Tensor, torch.Tensor, int]`, Task 1) matches its one call
  site in `forward()`. `save_checkpoint`/`load_checkpoint`'s new config
  keys (Task 3) match the constructor parameter names introduced in Task 1
  exactly, including using `.get(..., default)` so pre-this-phase
  checkpoints still load.
- **No placeholders:** every step has complete, runnable code.
