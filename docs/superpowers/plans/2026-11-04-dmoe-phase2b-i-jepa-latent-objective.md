# SS-DM-MoE Phase 2b-i: JEPA Latent Objective Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `JEPAModule` that trains a predictor to predict a masked span's latent representation (via an EMA target encoder) instead of raw tokens, and prove it actually does something useful with a loss-decrease check, an anti-collapse check, and a downstream linear-probe comparison against a task-loss-only baseline.

**Architecture:** One new, fully isolated file (`dmoe/jepa.py`) wrapping an *existing* `UniversalMultimodalEncoder` instance by reference — no changes to any Phase 1 or Phase 2a file. The predictor reuses the existing `ComplexExpert` module as-is.

**Tech Stack:** Python 3.14, PyTorch 2.15, pytest. Project root: `D:\Dynamic_MOE_Transformer`. Test commands use `./.venv/Scripts/python.exe` (git-bash style; swap to `.\.venv\Scripts\python.exe` in PowerShell).

Spec: `docs/superpowers/specs/2026-11-04-dmoe-phase2b-i-jepa-latent-objective-design.md`

**Concurrency note:** Phase 2a's implementation may be landing in parallel (touching `dmoe/continual.py`, `dmoe/replay_buffer.py`, `dmoe/__init__.py`, and their tests). This plan creates only new files (`dmoe/jepa.py`, `tests/test_jepa.py`) in Tasks 1–3 — those cannot conflict. Task 4 touches `dmoe/__init__.py`, which Phase 2a's own plan also touches; Task 4's step is written as a **targeted addition** (read the file, add two lines) rather than a full-file replacement, specifically so it doesn't clobber whatever Phase 2a's execution has already added there. Do not run Task 4 until Phase 2a's execution has finished and its own `__init__.py` change has landed.

---

### Task 1: `JEPAModule` core — masking, loss, EMA update

**Files:**
- Create: `tests/test_jepa.py`
- Create: `dmoe/jepa.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_jepa.py`:

```python
import torch

from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.jepa import JEPAModule, MASK_TOKEN_ID


def _small_encoder():
    return UniversalMultimodalEncoder(d_model=16)


def test_mask_token_id_is_reserved_top_of_vocab():
    assert MASK_TOKEN_ID == 31999


def test_compute_loss_returns_scalar_and_is_finite():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    token_ids = torch.randint(0, 1000, (4, 10), dtype=torch.long)
    loss = jepa.compute_loss(token_ids)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_compute_loss_masks_a_contiguous_span_not_the_whole_sequence():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, mask_ratio=0.3)
    seq_len = 10
    start, end = jepa._masked_span(seq_len)
    assert 0 <= start < end <= seq_len
    assert end - start == max(1, int(seq_len * 0.3))


def test_gradients_flow_to_predictor_and_context_encoder_only():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    token_ids = torch.randint(0, 1000, (4, 10), dtype=torch.long)

    loss = jepa.compute_loss(token_ids)
    loss.backward()

    assert jepa.predictor.linear1.weight_real.grad is not None
    assert not torch.isnan(jepa.predictor.linear1.weight_real.grad).any()
    assert encoder.text_embed.weight.grad is not None
    for p in jepa.target_encoder.parameters():
        assert p.grad is None


def test_target_encoder_starts_as_exact_copy_of_context_encoder():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    assert torch.equal(jepa.target_encoder.text_embed.weight, jepa.context_encoder.text_embed.weight)
    assert jepa.target_encoder.text_embed.weight is not jepa.context_encoder.text_embed.weight


def test_update_target_encoder_moves_target_toward_context():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, ema_decay=0.5)

    with torch.no_grad():
        encoder.text_embed.weight.add_(1.0)

    before = jepa.target_encoder.text_embed.weight.clone()
    jepa.update_target_encoder()
    after = jepa.target_encoder.text_embed.weight

    assert not torch.equal(before, after)
    expected = before * 0.5 + (before + 1.0) * 0.5
    assert torch.allclose(after, expected, atol=1e-5)


def test_update_target_encoder_does_not_require_grad():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    jepa.update_target_encoder()
    for p in jepa.target_encoder.parameters():
        assert p.requires_grad is False


def test_trainable_parameters_returns_only_predictor_params():
    encoder = _small_encoder()
    jepa = JEPAModule(encoder, predictor_hidden_dim=32)
    trainable_ids = {id(p) for p in jepa.trainable_parameters()}
    predictor_ids = {id(p) for p in jepa.predictor.parameters()}
    assert trainable_ids == predictor_ids
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_jepa.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.jepa'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/jepa.py`:

```python
import copy
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.encoder import UniversalMultimodalEncoder
from dmoe.experts import ComplexExpert

MASK_TOKEN_ID = 31999


class JEPAModule(nn.Module):
    """
    Wraps an EXISTING encoder by reference (not a copy). The encoder passed
    in is assumed to only ever process text while wrapped here, since the
    EMA update in update_target_encoder() zips context/target parameters
    positionally and assumes they stay structurally identical -- that
    breaks if the wrapped encoder's lazily-created numeric_projectors
    ModuleDict grows differently on one side than the other.
    """

    def __init__(
        self,
        encoder: UniversalMultimodalEncoder,
        predictor_hidden_dim: int,
        mask_token_id: int = MASK_TOKEN_ID,
        mask_ratio: float = 0.3,
        ema_decay: float = 0.996,
    ):
        super().__init__()
        self.context_encoder = encoder
        self.target_encoder = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = ComplexExpert(encoder.d_model, predictor_hidden_dim)
        self.mask_token_id = mask_token_id
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay

    def _masked_span(self, seq_len: int) -> Tuple[int, int]:
        mask_len = max(1, int(seq_len * self.mask_ratio))
        mask_len = min(mask_len, seq_len)
        start = int(torch.randint(0, seq_len - mask_len + 1, (1,)).item())
        return start, start + mask_len

    def compute_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        start, end = self._masked_span(seq_len)

        masked_ids = token_ids.clone()
        masked_ids[:, start:end] = self.mask_token_id

        context_r, context_i = self.context_encoder(masked_ids)
        with torch.no_grad():
            target_r, target_i = self.target_encoder(token_ids)

        ctx_slice_r = context_r[:, start:end, :]
        ctx_slice_i = context_i[:, start:end, :]
        tgt_slice_r = target_r[:, start:end, :].detach()
        tgt_slice_i = target_i[:, start:end, :].detach()

        pred_r, pred_i = self.predictor(ctx_slice_r, ctx_slice_i)

        return F.mse_loss(pred_r, tgt_slice_r) + F.mse_loss(pred_i, tgt_slice_i)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for target_p, context_p in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
            target_p.mul_(self.ema_decay).add_(context_p.detach() * (1.0 - self.ema_decay))

    def trainable_parameters(self):
        return self.predictor.parameters()
```

- [ ] **Step 4: Run to verify all tests pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_jepa.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/jepa.py tests/test_jepa.py
git commit -m "feat: add JEPAModule for masked-latent self-supervised prediction

Wraps an existing encoder by reference; trains a ComplexExpert-based
predictor to predict an EMA target encoder's representation of a
masked contiguous span from the context encoder's representation of
the same (masked) span -- forces abstraction over surface token
statistics. Target encoder is excluded from backprop (EMA-only
update) to avoid the classic JEPA representational-collapse failure
mode. Fully isolated: no changes to any existing file."
```

---

### Task 2: Validation — loss decreases, no representational collapse

**Files:**
- Modify: `tests/test_jepa.py` (append only)

- [ ] **Step 1: Append the validation tests**

Append to `tests/test_jepa.py` (keep everything already in the file above this):

```python
def test_jepa_loss_decreases_with_training():
    torch.manual_seed(0)
    encoder = UniversalMultimodalEncoder(d_model=16)
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, mask_ratio=0.3, ema_decay=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(jepa.trainable_parameters()), lr=5e-3
    )

    losses = []
    for _ in range(60):
        token_ids = torch.randint(0, 50, (8, 10), dtype=torch.long)
        optimizer.zero_grad()
        loss = jepa.compute_loss(token_ids)
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()
        losses.append(loss.item())

    early_avg = sum(losses[:10]) / 10
    late_avg = sum(losses[-10:]) / 10
    assert late_avg < early_avg


def test_target_encoder_representations_do_not_collapse_after_training():
    torch.manual_seed(0)
    encoder = UniversalMultimodalEncoder(d_model=16)
    jepa = JEPAModule(encoder, predictor_hidden_dim=32, mask_ratio=0.3, ema_decay=0.9)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(jepa.trainable_parameters()), lr=5e-3
    )

    for _ in range(60):
        token_ids = torch.randint(0, 50, (8, 10), dtype=torch.long)
        optimizer.zero_grad()
        loss = jepa.compute_loss(token_ids)
        loss.backward()
        optimizer.step()
        jepa.update_target_encoder()

    with torch.no_grad():
        inputs = torch.randint(0, 50, (5, 10), dtype=torch.long)
        target_r, _ = jepa.target_encoder(inputs)
        pooled = target_r.mean(dim=1)  # [5, d_model]

    pairwise_dists = torch.cdist(pooled, pooled)
    off_diagonal = pairwise_dists[~torch.eye(5, dtype=torch.bool)]
    assert off_diagonal.mean().item() > 1e-3
```

Note on `ema_decay=0.9` here versus the class's `0.996` default: the spec
flagged this explicitly as an open question. `0.996` is tuned for
large-scale training runs with many thousands of steps; at this project's
60-step test scale, `0.996` would leave the target encoder essentially
frozen at its random initial copy the entire time. `0.9` lets it actually
track the context encoder within tens of steps, which is what these tests
need to mean anything. Production-scale training (outside the test suite)
would use the class default.

- [ ] **Step 2: Run to verify both pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_jepa.py -v`
Expected: `10 passed`. If `test_jepa_loss_decreases_with_training` fails on
the first run, that's genuine evidence to investigate — try a higher
learning rate or more steps before concluding the mechanism is broken, per
the spec's instruction not to weaken assertions to make tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jepa.py
git commit -m "test: verify JEPA loss decreases with training and target doesn't collapse"
```

---

### Task 3: Downstream probe comparison — does JEPA actually help

**Files:**
- Modify: `tests/test_jepa.py` (append only)

- [ ] **Step 1: Append the probe comparison test**

Append to `tests/test_jepa.py` (keep everything already in the file above this):

```python
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE


def _task_a_batch(batch_size=8, seq_len=6):
    tokens = torch.randint(0, 50, (batch_size, seq_len), dtype=torch.long)
    targets = (tokens + 1) % 50
    return tokens, targets


def _train_model(use_jepa: bool, steps: int, seed: int) -> SelfStabilizingDynamicMorphicMoE:
    torch.manual_seed(seed)
    model = SelfStabilizingDynamicMorphicMoE(
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
    params = list(model.parameters())
    jepa = None
    if use_jepa:
        jepa = JEPAModule(model.encoder, predictor_hidden_dim=32, mask_ratio=0.3, ema_decay=0.9)
        params += list(jepa.trainable_parameters())
    optimizer = torch.optim.AdamW(params, lr=5e-3)

    for _ in range(steps):
        tokens, targets = _task_a_batch()
        optimizer.zero_grad()
        out = model(tokens, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        if use_jepa:
            loss = loss + jepa.compute_loss(tokens)
        loss.backward()
        optimizer.step()
        if use_jepa:
            jepa.update_target_encoder()

    return model


def _train_and_eval_probe(model: SelfStabilizingDynamicMorphicMoE, probe_steps: int = 60, seed: int = 1) -> float:
    # The probe predicts Task A's rule output for the sequence's first
    # position only -- enough to test whether the frozen, pooled
    # representation linearly encodes the task's structure, without
    # needing a full sequence-level probe head.
    torch.manual_seed(seed)
    for p in model.encoder.parameters():
        p.requires_grad_(False)

    probe = torch.nn.Linear(model.d_model * 2, 50)
    probe_optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2)

    for _ in range(probe_steps):
        tokens, targets = _task_a_batch()
        with torch.no_grad():
            real, imag = model.encoder(tokens)
            pooled = torch.cat([real.mean(dim=1), imag.mean(dim=1)], dim=-1)
        probe_optimizer.zero_grad()
        logits = probe(pooled)
        probe_loss = F.cross_entropy(logits, targets[:, 0])
        probe_loss.backward()
        probe_optimizer.step()

    eval_tokens, eval_targets = _task_a_batch(batch_size=32)
    with torch.no_grad():
        real, imag = model.encoder(eval_tokens)
        pooled = torch.cat([real.mean(dim=1), imag.mean(dim=1)], dim=-1)
        logits = probe(pooled)
        eval_loss = F.cross_entropy(logits, eval_targets[:, 0]).item()
    return eval_loss


def test_downstream_probe_jepa_outperforms_task_only_baseline():
    model_jepa = _train_model(use_jepa=True, steps=80, seed=0)
    model_baseline = _train_model(use_jepa=False, steps=80, seed=0)

    jepa_probe_loss = _train_and_eval_probe(model_jepa, seed=1)
    baseline_probe_loss = _train_and_eval_probe(model_baseline, seed=1)

    assert jepa_probe_loss < baseline_probe_loss
```

Move the `import torch.nn.functional as F` and
`from dmoe.model import SelfStabilizingDynamicMorphicMoE` lines shown above
to the top of `tests/test_jepa.py` alongside the existing imports rather
than leaving them mid-file — they're written here adjacent to the code
that needs them purely so this step is self-contained to read.

- [ ] **Step 2: Run to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_jepa.py -v`
Expected: `11 passed`. As with Task 2's loss-decrease test: if this fails
on the first run, that's real signal the auxiliary loss isn't yet strong
enough at these settings to show up in a downstream probe — try increasing
`steps` in `_train_model`, tuning `lr`, or adjusting `mask_ratio`, rather
than loosening the assertion.

- [ ] **Step 3: Commit**

```bash
git add tests/test_jepa.py
git commit -m "test: add downstream linear-probe comparison for the JEPA objective

Trains two models (task loss only vs task loss + JEPA) on the same
synthetic rule task and seed, freezes each encoder, trains a small
probe on top of each, and compares probe loss on held-out data -- a
measured, narrow proxy for 'did this produce more useful structure in
the representation', not a claim about general understanding."
```

---

### Task 4: Export `JEPAModule` and final verification

**Files:**
- Modify: `dmoe/__init__.py` (targeted addition only — see the concurrency note at the top of this plan)

- [ ] **Step 1: Confirm Phase 2a's execution has landed**

Run: `git log --oneline -5`
Expected: the most recent commits include Phase 2a's "Phase 2a complete"
commit (or equivalent) and its `dmoe/__init__.py` update. If not, stop here
and wait — don't proceed with Step 2 against a mid-flight file.

- [ ] **Step 2: Read the current `dmoe/__init__.py` and add two lines**

Read the file first. It will already contain Phase 1's exports and (once
Phase 2a has landed) `ReplayBuffer`/`ContinualLearner` exports too. Add
exactly these two things without removing or reordering anything already
there:

1. An import line: `from dmoe.jepa import JEPAModule, MASK_TOKEN_ID`
2. Two new entries in the `__all__` list: `"JEPAModule"` and `"MASK_TOKEN_ID"`

- [ ] **Step 3: Sanity-check the package imports cleanly**

Run: `./.venv/Scripts/python.exe -c "import dmoe; print(dmoe.JEPAModule, dmoe.MASK_TOKEN_ID)"`
Expected: prints the class and `31999` with no errors.

- [ ] **Step 4: Run the full suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: every test passes (Phase 1's, Phase 2a's, and this phase's 11 in
`test_jepa.py`), 1 skipped for CUDA. No failures or collection errors.

- [ ] **Step 5: Final commit**

```bash
git add dmoe/__init__.py
git commit -m "$(cat <<'EOF'
Phase 2b-i complete: JEPA-style latent self-supervised objective

JEPAModule is implemented, tested (masking/EMA/gradient-isolation
unit tests, loss-decrease and anti-collapse sanity checks, and a
downstream linear-probe comparison showing a measurable benefit over
a task-loss-only baseline), and exported from the package.
Integration with Phase 2a's ContinualLearner and Phase 2b-ii (the
adaptive internal-simulation loop) are explicitly deferred follow-ups.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Plan self-review notes

- **Spec coverage:** `JEPAModule` with context/target encoder, `ComplexExpert`-reused predictor, contiguous-span masking, EMA update, `trainable_parameters()` (Task 1); loss-decrease and anti-collapse validation (Task 2); downstream probe comparison (Task 3); export (Task 4). Non-goals (image/numeric masking, `ContinualLearner` integration, Phase 2b-ii) are correctly excluded from every task.
- **Type/signature consistency:** `JEPAModule.__init__`'s parameters (`encoder, predictor_hidden_dim, mask_token_id, mask_ratio, ema_decay`) match every call site across Tasks 1–3. `compute_loss`/`update_target_encoder`/`trainable_parameters` signatures match their usage in the validation and probe tests.
- **Concurrency safety:** Tasks 1–3 create only new files and cannot conflict with Phase 2a's in-progress work. Task 4 is explicitly gated on Phase 2a having landed first, and is written as a targeted addition rather than a full-file replace specifically to avoid clobbering Phase 2a's own `__init__.py` change.
- **No placeholders:** every step has complete, runnable code.
