# SS-DM-MoE Phase 1 Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the single-file SS-DM-MoE draft (`main.py`) into a correct, tested `dmoe/` package — fixing the 8 bugs identified in the design spec, with a real pytest suite proving growth/shrink/routing mechanisms work.

**Architecture:** One module per subsystem (`complex_ops`, `encoder`, `manifold`, `attention`, `experts`, `layer`, `decoder`, `checkpoint`, `model`), built bottom-up in dependency order (no module imports a module built later in this plan), each with its own test file. `main.py` becomes a thin entry point calling `scripts/validate.py`.

**Tech Stack:** Python 3.14, PyTorch 2.15 (CPU in this environment, device-agnostic code), pytest. Project root: `D:\Dynamic_MOE_Transformer`. All commands below use the project's existing venv via `./.venv/Scripts/python.exe` (works from the Bash tool / git-bash; swap to `.\.venv\Scripts\python.exe` if running in PowerShell).

Spec: `docs/superpowers/specs/2026-11-04-dynamic-moe-phase1-hardening-design.md`

---

### Task 0: Project scaffolding and pytest install

**Files:**
- Create: `dmoe/__init__.py`
- Create: `tests/__init__.py`
- Create: `scripts/__init__.py`

- [ ] **Step 1: Create package directories and empty init files**

```bash
mkdir -p dmoe tests scripts
touch dmoe/__init__.py tests/__init__.py scripts/__init__.py
```

- [ ] **Step 2: Install pytest into the existing venv**

Run: `./.venv/Scripts/python.exe -m pip install pytest`
Expected: pip reports `Successfully installed pytest-...` (or "already satisfied" if re-run).

- [ ] **Step 3: Verify pytest runs**

Run: `./.venv/Scripts/python.exe -m pytest --version`
Expected: prints a pytest version string, e.g. `pytest 8.x.x`.

- [ ] **Step 4: Commit scaffolding**

```bash
git add dmoe/__init__.py tests/__init__.py scripts/__init__.py
git commit -m "chore: scaffold dmoe package, tests, and scripts directories"
```

---

### Task 1: `dmoe/complex_ops.py` — ComplexLinear, ModReLU, ComplexLayerNorm

**Files:**
- Create: `tests/test_complex_ops.py`
- Create: `dmoe/complex_ops.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_complex_ops.py`:

```python
import torch

from dmoe.complex_ops import ComplexLinear, ModReLU, ComplexLayerNorm


def test_complex_linear_matches_manual_computation():
    layer = ComplexLinear(2, 2, bias=True)
    with torch.no_grad():
        layer.weight_real.copy_(torch.tensor([[1.0, 2.0], [0.5, -1.0]]))
        layer.weight_imag.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        layer.bias_real.copy_(torch.tensor([0.1, 0.2]))
        layer.bias_imag.copy_(torch.tensor([-0.1, 0.0]))

    z_r = torch.tensor([[1.0, 2.0]])
    z_i = torch.tensor([[3.0, 4.0]])

    out_r, out_i = layer(z_r, z_i)

    w_r = layer.weight_real.detach()
    w_i = layer.weight_imag.detach()
    expected_r = z_r @ w_r.T - z_i @ w_i.T + layer.bias_real.detach()
    expected_i = z_r @ w_i.T + z_i @ w_r.T + layer.bias_imag.detach()

    assert torch.allclose(out_r, expected_r, atol=1e-6)
    assert torch.allclose(out_i, expected_i, atol=1e-6)


def test_complex_linear_gradients_flow():
    layer = ComplexLinear(4, 4)
    z_r = torch.randn(2, 4, requires_grad=True)
    z_i = torch.randn(2, 4, requires_grad=True)
    out_r, out_i = layer(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert layer.weight_real.grad is not None
    assert not torch.isnan(layer.weight_real.grad).any()
    assert z_r.grad is not None


def test_modrelu_zeroes_small_magnitude():
    act = ModReLU(1)
    with torch.no_grad():
        act.b.fill_(-5.0)
    z_r = torch.tensor([[3.0]])
    z_i = torch.tensor([[4.0]])
    out_r, out_i = act(z_r, z_i)
    assert torch.allclose(out_r, torch.zeros_like(out_r), atol=1e-5)
    assert torch.allclose(out_i, torch.zeros_like(out_i), atol=1e-5)


def test_modrelu_preserves_phase_direction_above_threshold():
    act = ModReLU(1)
    with torch.no_grad():
        act.b.fill_(0.0)
    z_r = torch.tensor([[3.0]])
    z_i = torch.tensor([[4.0]])
    out_r, out_i = act(z_r, z_i)
    mag_in = torch.sqrt(z_r ** 2 + z_i ** 2)
    mag_out = torch.sqrt(out_r ** 2 + out_i ** 2)
    assert torch.allclose(mag_out, mag_in, atol=1e-4)
    assert torch.allclose(out_r / z_r, out_i / z_i, atol=1e-4)


def test_complex_layer_norm_output_shape_and_finite():
    norm = ComplexLayerNorm(8)
    z_r = torch.randn(3, 8) * 10 + 5
    z_i = torch.randn(3, 8) * 10 - 5
    out_r, out_i = norm(z_r, z_i)
    assert out_r.shape == z_r.shape
    assert out_i.shape == z_i.shape
    assert not torch.isnan(out_r).any()
    assert not torch.isnan(out_i).any()
```

- [ ] **Step 2: Run test to verify it fails on missing module**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_complex_ops.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.complex_ops'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/complex_ops.py`:

```python
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.has_bias = bias

        scale = 1.0 / math.sqrt(2.0 * in_features)
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * scale)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * scale)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_real = F.linear(z_real, self.weight_real) - F.linear(z_imag, self.weight_imag)
        out_imag = F.linear(z_real, self.weight_imag) + F.linear(z_imag, self.weight_real)

        if self.has_bias:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag

        return out_real, out_imag


class ModReLU(nn.Module):
    def __init__(self, features: int, eps: float = 1e-7):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mag = torch.sqrt(z_real ** 2 + z_imag ** 2 + self.eps)
        activated_mag = F.relu(mag + self.b)
        scale = activated_mag / mag
        return z_real * scale, z_imag * scale


class ComplexLayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma_real = nn.Parameter(torch.ones(d_model))
        self.gamma_imag = nn.Parameter(torch.zeros(d_model))
        self.beta_real = nn.Parameter(torch.zeros(d_model))
        self.beta_imag = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, z_real: torch.Tensor, z_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_r = z_real.mean(dim=-1, keepdim=True)
        mean_i = z_imag.mean(dim=-1, keepdim=True)
        z_cent_r = z_real - mean_r
        z_cent_i = z_imag - mean_i

        var = (z_cent_r ** 2 + z_cent_i ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.eps)

        norm_r = z_cent_r / std
        norm_i = z_cent_i / std

        out_r = norm_r * self.gamma_real - norm_i * self.gamma_imag + self.beta_real
        out_i = norm_r * self.gamma_imag + norm_i * self.gamma_real + self.beta_imag
        return out_r, out_i
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_complex_ops.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/complex_ops.py tests/test_complex_ops.py
git commit -m "feat: add complex-valued linear, ModReLU, and layer norm ops"
```

---

### Task 2: `dmoe/encoder.py` — UniversalMultimodalEncoder (fixes bug #2 and #4)

**Files:**
- Create: `tests/test_encoder.py`
- Create: `dmoe/encoder.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_encoder.py`:

```python
import pytest
import torch

from dmoe.encoder import UniversalMultimodalEncoder


def test_text_input_shape():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randint(0, 1000, (2, 5), dtype=torch.long)
    r, i = enc(x)
    assert r.shape == (2, 5, 16)
    assert i.shape == (2, 5, 16)


def test_image_input_shape():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randn(2, 3, 8, 8)
    r, i = enc(x)
    assert r.shape[0] == 2
    assert r.shape[-1] == 16
    assert i.shape == r.shape


def test_numeric_input_shape_and_lazy_projector_reuse():
    enc = UniversalMultimodalEncoder(d_model=16)
    x1 = torch.randn(2, 5, 7)
    r1, i1 = enc(x1)
    assert r1.shape == (2, 5, 16)
    assert "7" in enc.numeric_projectors

    projector_before = enc.numeric_projectors["7"]
    x2 = torch.randn(3, 4, 7)
    r2, i2 = enc(x2)
    assert r2.shape == (3, 4, 16)
    assert enc.numeric_projectors["7"] is projector_before


def test_numeric_projection_distinguishes_feature_position():
    # Regression test for the original bug: a fixed all-ones projection made
    # output identical regardless of WHICH feature index held a signal.
    enc = UniversalMultimodalEncoder(d_model=16)
    torch.manual_seed(0)
    x_a = torch.zeros(1, 1, 5)
    x_a[0, 0, 0] = 10.0
    x_b = torch.zeros(1, 1, 5)
    x_b[0, 0, 4] = 10.0
    r_a, _ = enc(x_a)
    r_b, _ = enc(x_b)
    assert not torch.allclose(r_a, r_b)


def test_text_detection_works_for_int32_dtype():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randint(0, 1000, (2, 5), dtype=torch.int32)
    r, i = enc(x)
    assert r.shape == (2, 5, 16)


def test_rejects_unsupported_shape():
    enc = UniversalMultimodalEncoder(d_model=16)
    x = torch.randn(2, 2, 2, 2, 2)
    with pytest.raises(ValueError):
        enc(x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_encoder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.encoder'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/encoder.py`:

```python
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class UniversalMultimodalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.text_embed = nn.Embedding(32000, d_model)
        self.text_phase_proj = nn.Linear(d_model, d_model)

        self.image_patch_proj = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=4, stride=4)
        self.image_phase_proj = nn.Linear(d_model, d_model)

        self.numeric_projectors = nn.ModuleDict()
        self.num_phase_proj = nn.Linear(d_model, d_model)

        self.unify_real = nn.Linear(d_model, d_model)
        self.unify_imag = nn.Linear(d_model, d_model)

    def _get_numeric_projector(self, feature_dim: int, device: torch.device) -> nn.Linear:
        key = str(feature_dim)
        if key not in self.numeric_projectors:
            self.numeric_projectors[key] = nn.Linear(feature_dim, self.d_model).to(device)
        return self.numeric_projectors[key]

    def forward(self, x: torch.Tensor, modality_hint: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 2 and not x.is_floating_point():
            emb = self.text_embed(x)
            real_comp = self.unify_real(emb)
            positions = torch.arange(x.size(1), device=x.device, dtype=torch.float32).unsqueeze(-1)
            omega = torch.exp(
                torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
            phase_sin = torch.sin(positions * omega)
            phase_cos = torch.cos(positions * omega)
            phase_pos = torch.cat([phase_sin, phase_cos], dim=-1)[:, : self.d_model].unsqueeze(0)
            imag_comp = self.unify_imag(self.text_phase_proj(emb) + phase_pos)

        elif x.dim() == 4:
            B, C, H, W = x.shape
            if C == 1:
                x = x.repeat(1, 3, 1, 1)
            patches = self.image_patch_proj(x)
            patches = patches.flatten(2).transpose(1, 2)
            real_comp = self.unify_real(patches)
            imag_comp = self.unify_imag(self.image_phase_proj(patches))

        elif x.dim() == 3:
            B, T, feature_dim = x.shape
            projector = self._get_numeric_projector(feature_dim, x.device)
            proj = projector(x)
            real_comp = self.unify_real(proj)
            imag_comp = self.unify_imag(self.num_phase_proj(proj))

        else:
            raise ValueError(f"Unsupported multimodal tensor shape: {tuple(x.shape)}")

        return real_comp, imag_comp
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_encoder.py -v`
Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/encoder.py tests/test_encoder.py
git commit -m "feat: add multimodal encoder with learnable numeric projection

Fixes the draft's numeric-stream projection, which used a fixed
all-ones matrix that collapsed any input to the mean of its features
regardless of which feature carried the signal."
```

---

### Task 3: `dmoe/manifold.py` — GeometricCorrectiveManifold (fixes bug #8)

**Files:**
- Create: `tests/test_manifold.py`
- Create: `dmoe/manifold.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_manifold.py`:

```python
import torch

from dmoe.manifold import GeometricCorrectiveManifold


def test_output_shape_matches_input():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out_r, out_i, err = m(z_r, z_i)
    assert out_r.shape == z_r.shape
    assert out_i.shape == z_i.shape
    assert err.dim() == 0


def test_running_error_updates_only_in_training_mode():
    m = GeometricCorrectiveManifold(d_model=8)
    initial = m.running_error_magnitude.item()

    m.eval()
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    m(z_r, z_i)
    assert m.running_error_magnitude.item() == initial

    m.train()
    m(z_r, z_i)
    assert m.running_error_magnitude.item() != initial


def test_error_is_non_negative():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(4, 2, 8) * 100
    z_i = torch.randn(4, 2, 8) * 100
    _, _, err = m(z_r, z_i)
    assert err.item() >= 0.0


def test_gradients_flow_through_corrective_path():
    m = GeometricCorrectiveManifold(d_model=8)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    out_r, out_i, err = m(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert z_r.grad is not None
    assert not torch.isnan(z_r.grad).any()


def test_predictor_stages_are_named_clearly():
    m = GeometricCorrectiveManifold(d_model=8)
    assert hasattr(m, "predictor_stage1")
    assert hasattr(m, "predictor_stage2")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_manifold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.manifold'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/manifold.py`:

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

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p1_r, p1_i = self.predictor_stage1(z_r, z_i)
        pred_r, pred_i = self.predictor_stage2(p1_r, p1_i)

        delta_r = z_r - pred_r
        delta_i = z_i - pred_i

        error_norm = torch.sqrt(delta_r ** 2 + delta_i ** 2 + 1e-8)
        batch_mean_error = error_norm.mean()

        if self.training:
            self.running_error_magnitude.mul_(self.momentum).add_(
                batch_mean_error.detach() * (1.0 - self.momentum)
            )

        c1_r, c1_i = self.corrector_linear1(delta_r, delta_i)
        c1_act_r, c1_act_i = self.act(c1_r, c1_i)
        corr_r, corr_i = self.corrector_linear2(c1_act_r, c1_act_i)

        corrected_r = z_r + corr_r
        corrected_i = z_i + corr_i

        return corrected_r, corrected_i, batch_mean_error
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_manifold.py -v`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/manifold.py tests/test_manifold.py
git commit -m "feat: add geometric corrective manifold with clearer predictor naming"
```

---

### Task 4: `dmoe/attention.py` — ComplexMultiHeadAttention

**Files:**
- Create: `tests/test_attention.py`
- Create: `dmoe/attention.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_attention.py`:

```python
import pytest
import torch

from dmoe.attention import ComplexMultiHeadAttention


def test_output_shape():
    attn = ComplexMultiHeadAttention(d_model=16, num_heads=4)
    z_r = torch.randn(2, 5, 16)
    z_i = torch.randn(2, 5, 16)
    out_r, out_i = attn(z_r, z_i)
    assert out_r.shape == (2, 5, 16)
    assert out_i.shape == (2, 5, 16)


def test_rejects_non_divisible_heads():
    with pytest.raises(AssertionError):
        ComplexMultiHeadAttention(d_model=10, num_heads=3)


def test_gradients_flow():
    attn = ComplexMultiHeadAttention(d_model=8, num_heads=2)
    z_r = torch.randn(1, 3, 8, requires_grad=True)
    z_i = torch.randn(1, 3, 8, requires_grad=True)
    out_r, out_i = attn(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert z_r.grad is not None
    assert not torch.isnan(z_r.grad).any()


def test_multiple_head_configurations_preserve_shape():
    for d_model, num_heads in [(8, 1), (8, 2), (8, 4), (32, 8)]:
        attn = ComplexMultiHeadAttention(d_model=d_model, num_heads=num_heads)
        z_r = torch.randn(2, 4, d_model)
        z_i = torch.randn(2, 4, d_model)
        out_r, out_i = attn(z_r, z_i)
        assert out_r.shape == (2, 4, d_model)
        assert out_i.shape == (2, 4, d_model)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_attention.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.attention'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/attention.py`:

```python
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.complex_ops import ComplexLinear


class ComplexMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = ComplexLinear(d_model, d_model)
        self.k_proj = ComplexLinear(d_model, d_model)
        self.v_proj = ComplexLinear(d_model, d_model)
        self.out_proj = ComplexLinear(d_model, d_model)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = z_r.shape

        qr, qi = self.q_proj(z_r, z_i)
        kr, ki = self.k_proj(z_r, z_i)
        vr, vi = self.v_proj(z_r, z_i)

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        qr, qi = split_heads(qr), split_heads(qi)
        kr, ki = split_heads(kr), split_heads(ki)
        vr, vi = split_heads(vr), split_heads(vi)

        hermitian_real = (
            torch.matmul(qr, kr.transpose(-2, -1)) + torch.matmul(qi, ki.transpose(-2, -1))
        ) * self.scale
        attn_weights = F.softmax(hermitian_real, dim=-1)

        out_r = torch.matmul(attn_weights, vr)
        out_i = torch.matmul(attn_weights, vi)

        def combine_heads(t: torch.Tensor) -> torch.Tensor:
            return t.transpose(1, 2).contiguous().view(B, S, D)

        out_r, out_i = combine_heads(out_r), combine_heads(out_i)
        return self.out_proj(out_r, out_i)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_attention.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/attention.py tests/test_attention.py
git commit -m "feat: add complex multi-head attention with Hermitian inner product"
```

---

### Task 5: `dmoe/experts.py` — ComplexExpert, DynamicMorphicMoE (fixes bug #1 and #3)

**Files:**
- Create: `tests/test_experts.py`
- Create: `dmoe/experts.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_experts.py`:

```python
import torch

from dmoe.experts import ComplexExpert, DynamicMorphicMoE


def test_expert_forward_shape():
    expert = ComplexExpert(d_model=8, d_ff=16)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out_r, out_i = expert(z_r, z_i)
    assert out_r.shape == (2, 3, 8)
    assert out_i.shape == (2, 3, 8)


def test_clone_with_mutation_produces_perturbed_but_related_weights():
    expert = ComplexExpert(d_model=8, d_ff=16)
    clone = expert.clone_with_mutation(mutation_std=0.1)
    assert not torch.equal(expert.linear1.weight_real, clone.linear1.weight_real)
    assert torch.allclose(expert.linear1.weight_real, clone.linear1.weight_real, atol=0.6)


def test_entropy_is_tracked_automatically_after_forward():
    moe = DynamicMorphicMoE(d_model=8, d_ff=16, initial_experts=3)
    z_r = torch.randn(2, 4, 8)
    z_i = torch.randn(2, 4, 8)
    moe(z_r, z_i)
    assert moe._last_entropy.item() >= 0.0


def test_mutate_topology_spawns_expert_on_high_entropy():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=3, tau_split=1.0, tau_death=0.01, min_experts=2
    )
    moe.train()
    z_r = torch.randn(2, 4, 8)
    z_i = torch.randn(2, 4, 8)
    moe(z_r, z_i)
    moe._last_entropy = torch.tensor(5.0)

    spawned, pruned, _ = moe.mutate_topology()

    assert spawned is True
    assert pruned is False
    assert moe.num_experts == 4
    assert moe.router_linear.out_features == 4
    assert moe.expert_usage.shape == (4,)


def test_mutate_topology_prunes_expert_on_low_usage():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=3, tau_split=10.0, tau_death=0.1, min_experts=2
    )
    with torch.no_grad():
        moe.expert_usage.copy_(torch.tensor([0.01, 0.5, 0.49]))
    moe._last_entropy = torch.tensor(0.1)

    spawned, pruned, _ = moe.mutate_topology()

    assert spawned is False
    assert pruned is True
    assert moe.num_experts == 2
    assert moe.router_linear.out_features == 2
    assert moe.expert_usage.shape == (2,)


def test_mutate_topology_respects_min_experts_floor():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=2, tau_split=10.0, tau_death=0.9, min_experts=2
    )
    with torch.no_grad():
        moe.expert_usage.copy_(torch.tensor([0.01, 0.5]))
    moe._last_entropy = torch.tensor(0.1)

    spawned, pruned, _ = moe.mutate_topology()

    assert pruned is False
    assert moe.num_experts == 2


def test_gradients_flow_after_mutation():
    moe = DynamicMorphicMoE(d_model=8, d_ff=16, initial_experts=3, tau_split=1.0, min_experts=2)
    moe._last_entropy = torch.tensor(5.0)
    moe.mutate_topology()

    z_r = torch.randn(2, 4, 8, requires_grad=True)
    z_i = torch.randn(2, 4, 8, requires_grad=True)
    out_r, out_i, entropy = moe(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()

    assert z_r.grad is not None
    for expert in moe.experts:
        assert expert.linear1.weight_real.grad is not None
        assert not torch.isnan(expert.linear1.weight_real.grad).any()


def test_repeated_mutation_keeps_buffer_shape_consistent():
    moe = DynamicMorphicMoE(
        d_model=8, d_ff=16, initial_experts=3, tau_split=0.5, tau_death=0.9, min_experts=2
    )
    for _ in range(5):
        moe._last_entropy = torch.tensor(5.0)
        moe.mutate_topology()
    assert moe.expert_usage.shape == (moe.num_experts,)
    assert moe.router_linear.out_features == moe.num_experts
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_experts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.experts'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/experts.py`:

```python
from typing import Tuple

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
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.tau_split = tau_split
        self.tau_death = tau_death
        self.min_experts = min_experts

        self.experts = nn.ModuleList([ComplexExpert(d_model, d_ff) for _ in range(initial_experts)])
        self.router_linear = nn.Linear(2 * d_model, initial_experts)
        self.register_buffer("expert_usage", torch.ones(initial_experts) / initial_experts)
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

        acc_r = torch.zeros_like(z_r)
        acc_i = torch.zeros_like(z_i)

        for j, expert in enumerate(self.experts):
            weight_j = router_probs[:, :, j : j + 1]
            exp_out_r, exp_out_i = expert(z_r, z_i)
            acc_r = acc_r + weight_j * exp_out_r
            acc_i = acc_i + weight_j * exp_out_i

        return acc_r, acc_i, entropy

    def mutate_topology(self, last_entropy: float = None) -> Tuple[bool, bool, str]:
        if last_entropy is None:
            last_entropy = self._last_entropy.item()

        spawned = False
        pruned = False
        action_log = []
        device = self.router_linear.weight.device

        if last_entropy > self.tau_split:
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

                pruned = True
                action_log.append(f"Expunged dormant expert #{least_idx} -> New Total: {len(self.experts)}")

        log_str = "; ".join(action_log) if action_log else "Stable"
        return spawned, pruned, log_str
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_experts.py -v`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/experts.py tests/test_experts.py
git commit -m "feat: add dynamic MoE experts with automatic entropy tracking

Fixes the draft's horizontal-morphing bug where _last_entropy was read
by the mutation logic but never actually set by forward(), so splits
and deaths only ever happened when a caller manually overrode it."
```

---

### Task 6: `dmoe/layer.py` — DynamicMorphicTransformerLayer

**Files:**
- Create: `tests/test_layer.py`
- Create: `dmoe/layer.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_layer.py`:

```python
import torch

from dmoe.layer import DynamicMorphicTransformerLayer


def test_forward_shape_stable():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out_r, out_i, entropy = layer(z_r, z_i)
    assert out_r.shape == (2, 3, 8)
    assert out_i.shape == (2, 3, 8)


def test_salience_updates_in_training_mode():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    layer.train()
    initial = layer.layer_salience.item()
    z_r = torch.randn(2, 3, 8) * 5
    z_i = torch.randn(2, 3, 8) * 5
    layer(z_r, z_i)
    assert layer.layer_salience.item() != initial


def test_salience_does_not_update_in_eval_mode():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    layer.eval()
    initial = layer.layer_salience.item()
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    layer(z_r, z_i)
    assert layer.layer_salience.item() == initial


def test_gradients_flow():
    layer = DynamicMorphicTransformerLayer(d_model=8, num_heads=2, d_ff=16, initial_experts=2)
    z_r = torch.randn(2, 3, 8, requires_grad=True)
    z_i = torch.randn(2, 3, 8, requires_grad=True)
    out_r, out_i, entropy = layer(z_r, z_i)
    (out_r.sum() + out_i.sum()).backward()
    assert z_r.grad is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_layer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.layer'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/layer.py`:

```python
from typing import Tuple

import torch
import torch.nn as nn

from dmoe.complex_ops import ComplexLayerNorm
from dmoe.attention import ComplexMultiHeadAttention
from dmoe.experts import DynamicMorphicMoE


class DynamicMorphicTransformerLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, initial_experts: int = 4):
        super().__init__()
        self.d_model = d_model
        self.norm1 = ComplexLayerNorm(d_model)
        self.attn = ComplexMultiHeadAttention(d_model, num_heads)

        self.norm2 = ComplexLayerNorm(d_model)
        self.moe = DynamicMorphicMoE(d_model, d_ff, initial_experts=initial_experts)

        self.alpha_residual = nn.Parameter(torch.ones(1))
        self.register_buffer("layer_salience", torch.tensor(1.0, dtype=torch.float32))

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n1_r, n1_i = self.norm1(z_r, z_i)
        att_r, att_i = self.attn(n1_r, n1_i)
        mid_r = z_r + self.alpha_residual * att_r
        mid_i = z_i + self.alpha_residual * att_i

        n2_r, n2_i = self.norm2(mid_r, mid_i)
        moe_r, moe_i, entropy = self.moe(n2_r, n2_i)
        out_r = mid_r + self.alpha_residual * moe_r
        out_i = mid_i + self.alpha_residual * moe_i

        if self.training:
            with torch.no_grad():
                diff_norm = torch.sqrt((out_r - z_r) ** 2 + (out_i - z_i) ** 2 + 1e-8).mean()
                in_norm = torch.sqrt(z_r ** 2 + z_i ** 2 + 1e-8).mean()
                salience = diff_norm / (in_norm + 1e-5)
                self.layer_salience.mul_(0.95).add_(salience * 0.05)

        return out_r, out_i, entropy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_layer.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/layer.py tests/test_layer.py
git commit -m "feat: add dynamic morphic transformer layer with salience tracking"
```

---

### Task 7: `dmoe/decoder.py` — UniversalProjectiveDecoder

**Files:**
- Create: `tests/test_decoder.py`
- Create: `dmoe/decoder.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/test_decoder.py`:

```python
import pytest
import torch

from dmoe.decoder import UniversalProjectiveDecoder


def test_token_head_shape():
    dec = UniversalProjectiveDecoder(d_model=8, vocab_size=100, continuous_dim=4)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out = dec(z_r, z_i, head="token")
    assert out.shape == (2, 3, 100)


def test_continuous_head_shape():
    dec = UniversalProjectiveDecoder(d_model=8, vocab_size=100, continuous_dim=4)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out = dec(z_r, z_i, head="continuous")
    assert out.shape == (2, 3, 4)


def test_latent_head_shape():
    dec = UniversalProjectiveDecoder(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    out = dec(z_r, z_i, head="latent")
    assert out.shape == (2, 3, 8)


def test_unknown_head_raises():
    dec = UniversalProjectiveDecoder(d_model=8)
    z_r = torch.randn(2, 3, 8)
    z_i = torch.randn(2, 3, 8)
    with pytest.raises(ValueError):
        dec(z_r, z_i, head="nonsense")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_decoder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dmoe.decoder'`

- [ ] **Step 3: Write the implementation**

Create `dmoe/decoder.py`:

```python
import torch
import torch.nn as nn


class UniversalProjectiveDecoder(nn.Module):
    def __init__(self, d_model: int, vocab_size: int = 32000, continuous_dim: int = 8):
        super().__init__()
        self.d_model = d_model

        self.proj_real = nn.Linear(d_model, d_model)
        self.proj_imag = nn.Linear(d_model, d_model)
        self.final_norm = nn.LayerNorm(d_model)

        self.token_head = nn.Linear(d_model, vocab_size)
        self.continuous_head = nn.Linear(d_model, continuous_dim)

    def forward(self, z_r: torch.Tensor, z_i: torch.Tensor, head: str = "token") -> torch.Tensor:
        unified_hidden = self.final_norm(self.proj_real(z_r) + self.proj_imag(z_i))
        if head == "token":
            return self.token_head(unified_hidden)
        elif head == "continuous":
            return self.continuous_head(unified_hidden)
        elif head == "latent":
            return unified_hidden
        else:
            raise ValueError(f"Unknown output head specification: {head}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_decoder.py -v`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add dmoe/decoder.py tests/test_decoder.py
git commit -m "feat: add universal projective decoder with token/continuous/latent heads"
```

---

### Task 8: `dmoe/checkpoint.py` — topology-aware save/load (fixes bug #6)

**Files:**
- Create: `dmoe/checkpoint.py`

No standalone test file for this task — `checkpoint.py` only does anything useful in the context of a full model, so it's exercised by `test_model_mutation.py` in Task 10. This task just adds the implementation.

- [ ] **Step 1: Write the implementation**

Create `dmoe/checkpoint.py`:

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
        "min_layers": model.min_layers,
        "max_layers": model.max_layers,
    }
    experts_per_layer = [layer.moe.num_experts for layer in model.layers]
    torch.save(
        {
            "config": config,
            "experts_per_layer": experts_per_layer,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str, model_cls, device: Optional[torch.device] = None) -> nn.Module:
    payload = torch.load(path, map_location=device or "cpu", weights_only=True)
    config = payload["config"]
    experts_per_layer = payload["experts_per_layer"]

    model = model_cls(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        initial_layers=len(experts_per_layer),
        initial_experts=experts_per_layer[0] if experts_per_layer else 1,
        theta_up=config["theta_up"],
        theta_down=config["theta_down"],
        min_layers=config["min_layers"],
        max_layers=config["max_layers"],
    )

    rebuilt_layers = nn.ModuleList(
        [
            DynamicMorphicTransformerLayer(config["d_model"], config["num_heads"], config["d_ff"], initial_experts=n)
            for n in experts_per_layer
        ]
    )
    model.layers = rebuilt_layers
    model.load_state_dict(payload["state_dict"])

    if device is not None:
        model = model.to(device)

    return model
```

- [ ] **Step 2: Commit**

```bash
git add dmoe/checkpoint.py
git commit -m "feat: add topology-aware checkpoint save/load

A model whose layer/expert counts change at runtime can't round-trip
through a plain state_dict across different mutation states (shape
mismatch on load). This serializes topology alongside weights and
rebuilds the exact per-layer expert counts before loading."
```

---

### Task 9: `dmoe/model.py` — SelfStabilizingDynamicMorphicMoE + optimizer-state-preserving rebuild (fixes bug #5 and #7)

**Files:**
- Create: `dmoe/model.py`
- Modify: `dmoe/__init__.py`

- [ ] **Step 1: Write the implementation**

Create `dmoe/model.py`:

```python
from typing import Dict, List, Union

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
        min_layers: int = 1,
        max_layers: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta_up = theta_up
        self.theta_down = theta_down
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
        z_r, z_i, structural_error = self.manifold_corr(z_r, z_i)

        entropy_traces = []
        for layer in self.layers:
            z_r, z_i, entropy = layer(z_r, z_i)
            entropy_traces.append(entropy)

        output = self.decoder(z_r, z_i, head=output_head)

        return {
            "output": output,
            "structural_error": structural_error,
            "routing_entropies": entropy_traces,
            "z_real": z_r,
            "z_imag": z_i,
        }

    def execute_graph_mutation(self) -> Dict[str, Union[int, List[str]]]:
        mutation_logs = []
        current_error = self.manifold_corr.running_error_magnitude.item()
        device = next(self.parameters()).device

        if current_error > self.theta_up and len(self.layers) < self.max_layers:
            new_layer = DynamicMorphicTransformerLayer(
                self.d_model,
                self.num_heads,
                self.d_ff,
                initial_experts=self.layers[-1].moe.num_experts,
            ).to(device)
            self.layers.append(new_layer)
            mutation_logs.append(
                f"[Vertical Growth] Structural Error={current_error:.4f} > {self.theta_up:.4f}. "
                f"Appended Layer #{len(self.layers)}."
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
            spawned, pruned, log_msg = layer.moe.mutate_topology()
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

- [ ] **Step 2: Update the package `__init__.py`**

Replace the contents of `dmoe/__init__.py`:

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
]
```

- [ ] **Step 3: Sanity-check the package imports cleanly**

Run: `./.venv/Scripts/python.exe -c "import dmoe; print(dmoe.SelfStabilizingDynamicMorphicMoE)"`
Expected: prints `<class 'dmoe.model.SelfStabilizingDynamicMorphicMoE'>` with no errors.

- [ ] **Step 4: Commit**

```bash
git add dmoe/model.py dmoe/__init__.py
git commit -m "feat: assemble full SS-DM-MoE model with optimizer-state-preserving rebuild

Fixes the draft's optimizer-reset bug: rebuilding a fresh AdamW after
every structural mutation discarded Adam momentum/variance for every
parameter that didn't change. rebuild_optimizer_preserving_state keeps
state for surviving parameters (matched by identity) and only lets
newly-created parameters start fresh."
```

---

### Task 10: `tests/test_model_mutation.py` — vertical/horizontal mutation, checkpoint round-trip, optimizer preservation, device consistency

**Files:**
- Create: `tests/test_model_mutation.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_model_mutation.py`:

```python
import pytest
import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE, rebuild_optimizer_preserving_state


def _tiny_model(**overrides):
    kwargs = dict(
        d_model=8,
        num_heads=2,
        d_ff=16,
        initial_layers=2,
        initial_experts=2,
        theta_up=0.4,
        theta_down=0.05,
        min_layers=1,
        max_layers=4,
    )
    kwargs.update(overrides)
    return SelfStabilizingDynamicMorphicMoE(**kwargs)


def test_vertical_growth_triggers_above_theta_up():
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    report = model.execute_graph_mutation()
    assert model.num_layers == 3
    assert any("Vertical Growth" in log for log in report["mutation_logs"])


def test_vertical_growth_respects_max_layers():
    model = _tiny_model(initial_layers=4, max_layers=4)
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    report = model.execute_graph_mutation()
    assert model.num_layers == 4


def test_vertical_pruning_triggers_below_theta_down():
    model = _tiny_model(initial_layers=3, min_layers=1)
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.0)
        model.layers[1].layer_salience.fill_(0.01)
    report = model.execute_graph_mutation()
    assert model.num_layers == 2
    assert any("Vertical Pruning" in log for log in report["mutation_logs"])


def test_vertical_pruning_respects_min_layers():
    model = _tiny_model(initial_layers=1, min_layers=1)
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.0)
        model.layers[0].layer_salience.fill_(0.01)
    report = model.execute_graph_mutation()
    assert model.num_layers == 1


def test_checkpoint_round_trip_reproduces_output(tmp_path):
    model = _tiny_model()
    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out_before = model(x)["output"]

    path = tmp_path / "ckpt.pt"
    model.save_checkpoint(str(path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(path))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)["output"]

    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_checkpoint_round_trip_after_mutation_with_nonuniform_experts(tmp_path):
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()

    model.layers[0].moe._last_entropy = torch.tensor(5.0)
    model.execute_graph_mutation()

    experts_before = [l.moe.num_experts for l in model.layers]
    assert len(set(experts_before)) > 1  # proves per-layer expert counts diverged

    model.eval()
    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    with torch.no_grad():
        out_before = model(x)["output"]

    path = tmp_path / "ckpt_mutated.pt"
    model.save_checkpoint(str(path))
    loaded = SelfStabilizingDynamicMorphicMoE.load_checkpoint(str(path))
    loaded.eval()
    with torch.no_grad():
        out_after = loaded(x)["output"]

    assert [l.moe.num_experts for l in loaded.layers] == experts_before
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_rebuild_optimizer_preserves_state_for_surviving_params():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    targets = torch.randint(0, 100, (1, 4))
    out = model(x)["output"]
    loss = F.cross_entropy(out.view(-1, out.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()

    tracked_param = model.layers[0].moe.experts[0].linear1.weight_real
    state_before = optimizer.state[tracked_param]["exp_avg"].clone()

    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()

    new_optimizer = rebuild_optimizer_preserving_state(optimizer, model, lr=1e-3)
    state_after = new_optimizer.state[tracked_param]["exp_avg"]
    assert torch.equal(state_before, state_after)


def test_model_mutate_and_move_to_device_is_consistent():
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()
    model.layers[0].moe._last_entropy = torch.tensor(5.0)
    model.execute_graph_mutation()

    devices = {p.device for p in model.parameters()}
    assert len(devices) == 1

    x = torch.randint(0, 100, (1, 4), dtype=torch.long)
    out = model(x)["output"]
    assert out.device == next(model.parameters()).device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_runs_on_cuda_after_mutation():
    model = _tiny_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()
    model = model.to("cuda")
    x = torch.randint(0, 100, (1, 4), dtype=torch.long, device="cuda")
    out = model(x)["output"]
    assert out.device.type == "cuda"
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_model_mutation.py -v`
Expected: `8 passed, 1 skipped` (the CUDA test skips on this CPU-only environment).

- [ ] **Step 3: Commit**

```bash
git add tests/test_model_mutation.py
git commit -m "test: cover vertical/horizontal mutation, checkpoint round-trip, optimizer state"
```

---

### Task 11: `tests/test_end_to_end.py` — full multimodal + optimization smoke tests

**Files:**
- Create: `tests/test_end_to_end.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_end_to_end.py`:

```python
import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE


def _small_model():
    return SelfStabilizingDynamicMorphicMoE(
        d_model=16,
        num_heads=2,
        d_ff=32,
        initial_layers=2,
        initial_experts=2,
        theta_up=0.4,
        theta_down=0.05,
        min_layers=1,
        max_layers=4,
    )


def test_sequential_multimodal_forward_passes_succeed():
    model = _small_model()
    model.eval()

    text = torch.randint(0, 500, (2, 6), dtype=torch.long)
    out_text = model(text, output_head="token")
    assert out_text["output"].shape[:2] == (2, 6)

    image = torch.randn(2, 3, 16, 16)
    out_image = model(image, output_head="continuous")
    assert out_image["output"].shape[0] == 2

    numeric = torch.randn(2, 10, 5)
    out_numeric = model(numeric, output_head="continuous")
    assert out_numeric["output"].shape[0] == 2


def test_short_optimization_loop_runs_without_errors_and_loss_is_finite():
    torch.manual_seed(0)
    model = _small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    text = torch.randint(0, 500, (2, 6), dtype=torch.long)
    targets = torch.randint(0, 500, (2, 6))

    for _ in range(5):
        optimizer.zero_grad()
        out = model(text, output_head="token")
        loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
        loss = loss + 0.1 * out["structural_error"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        assert torch.isfinite(loss).all()


def test_full_mutation_cycle_then_optimization_step_does_not_error():
    model = _small_model()
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(0.9)
    model.execute_graph_mutation()

    model.layers[0].moe._last_entropy = torch.tensor(5.0)
    model.execute_graph_mutation()

    with torch.no_grad():
        model.layers[0].moe.expert_usage[0] = 0.0001
    model.layers[0].moe._last_entropy = torch.tensor(0.1)
    model.execute_graph_mutation()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    text = torch.randint(0, 500, (2, 6), dtype=torch.long)
    targets = torch.randint(0, 500, (2, 6))

    out = model(text, output_head="token")
    loss = F.cross_entropy(out["output"].view(-1, out["output"].size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss).all()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_end_to_end.py -v`
Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_end_to_end.py
git commit -m "test: add end-to-end multimodal and mutation-then-optimize smoke tests"
```

---

### Task 12: Rewrite `scripts/validate.py` and `main.py`, retire the monolithic draft

**Files:**
- Create: `scripts/validate.py`
- Modify: `main.py` (replace entire contents — the old draft moves into `dmoe/` across Tasks 1-9, so this removes the now-duplicated copy)

- [ ] **Step 1: Write the validation harness**

Create `scripts/validate.py`:

```python
"""Human-readable stress-test trace for the SS-DM-MoE package.

Demonstrates sequential multimodal ingestion, forced vertical growth,
forced horizontal expert split, forced horizontal expert death, and a
full optimization step. Thresholds are forced explicitly via direct
buffer/attribute manipulation below -- nothing is hidden in tuned
constructor defaults.
"""

import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE


def run_validation_suite():
    print("=" * 80)
    print("SS-DM-MoE VALIDATION HARNESS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target Device: {device}\n")

    d_model = 64
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=d_model,
        num_heads=4,
        d_ff=128,
        initial_layers=2,
        initial_experts=3,
        theta_up=0.45,
        theta_down=0.08,
        min_layers=1,
        max_layers=6,
    ).to(device)

    print(f"Initial depth: {model.num_layers} layers")
    print(f"Initial experts per layer: {[l.moe.num_experts for l in model.layers]}\n")

    print("-" * 80)
    print("PHASE 1: Sequential multimodal ingestion")
    text_input = torch.randint(0, 1000, (4, 16), dtype=torch.long, device=device)
    out_text = model(text_input, output_head="token")
    print(
        f"  Text:     in={list(text_input.shape)} out={list(out_text['output'].shape)} "
        f"structural_error={out_text['structural_error'].item():.5f}"
    )

    image_input = torch.randn(4, 3, 32, 32, device=device)
    out_img = model(image_input, output_head="continuous")
    print(
        f"  Image:    in={list(image_input.shape)} out={list(out_img['output'].shape)} "
        f"structural_error={out_img['structural_error'].item():.5f}"
    )

    numeric_input = torch.randn(4, 24, 8, device=device)
    out_num = model(numeric_input, output_head="continuous")
    print(
        f"  Numeric:  in={list(numeric_input.shape)} out={list(out_num['output'].shape)} "
        f"structural_error={out_num['structural_error'].item():.5f}"
    )

    print("\n" + "-" * 80)
    print("PHASE 2: Forcing structural error above theta_up to trigger vertical growth")
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(model.theta_up + 0.3)
    report = model.execute_graph_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Depth now: {model.num_layers} layers, experts: {report['experts_per_layer']}")

    print("\n" + "-" * 80)
    print("PHASE 3: Forcing routing entropy above tau_split to trigger expert mitosis")
    model.layers[0].moe._last_entropy = torch.tensor(model.layers[0].moe.tau_split + 0.2)
    report = model.execute_graph_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Layer 0 experts now: {model.layers[0].moe.num_experts}")

    print("\n" + "-" * 80)
    print("PHASE 4: Forcing an expert's usage below tau_death to trigger expulsion")
    with torch.no_grad():
        model.layers[0].moe.expert_usage[0] = model.layers[0].moe.tau_death * 0.1
    model.layers[0].moe._last_entropy = torch.tensor(model.layers[0].moe.tau_split * 0.1)
    report = model.execute_graph_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Layer 0 experts now: {model.layers[0].moe.num_experts}")

    print("\n" + "-" * 80)
    print("PHASE 5: Full optimization step on the mutated architecture")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    forward_res = model(text_input, output_head="token")
    dummy_targets = torch.randint(0, 1000, (4, 16), device=device)
    task_loss = F.cross_entropy(
        forward_res["output"].view(-1, forward_res["output"].size(-1)),
        dummy_targets.view(-1),
    )
    total_loss = task_loss + 0.1 * forward_res["structural_error"]
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"  Total loss: {total_loss.item():.4f}  Grad norm: {grad_norm.item():.4f}")
    print("\n" + "=" * 80)
    print("VALIDATION HARNESS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_validation_suite()
```

- [ ] **Step 2: Replace `main.py` entirely**

Replace the full contents of `main.py` (currently the 900+ line monolithic draft) with:

```python
from scripts.validate import run_validation_suite

if __name__ == "__main__":
    run_validation_suite()
```

- [ ] **Step 3: Run the harness end to end**

Run: `./.venv/Scripts/python.exe main.py`
Expected: prints the five-phase trace ending in `VALIDATION HARNESS COMPLETE`, with no tracebacks. Confirm the printed layer count goes from 2 to 3 in Phase 2, and the Phase 3 expert count for layer 0 increases, and the Phase 4 expert count for layer 0 decreases back down.

- [ ] **Step 4: Commit**

```bash
git add scripts/validate.py main.py
git commit -m "refactor: replace monolithic draft with thin entry point into dmoe package

main.py no longer contains the model implementation -- it now just
calls the validation harness in scripts/validate.py, which imports
the hardened dmoe package built across the preceding tasks."
```

---

### Task 13: Full suite run and final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the complete pytest suite**

Run: `./.venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all tests pass (one CUDA test skipped on this CPU-only environment). Total should be in the mid-40s passed + 1 skipped given the counts from each task above.

- [ ] **Step 2: Run the validation harness once more as a final sanity check**

Run: `./.venv/Scripts/python.exe main.py`
Expected: same five-phase trace as Task 12, Step 3, completing without errors.

- [ ] **Step 3: Confirm git log shows the full task history**

Run: `git log --oneline`
Expected: one commit per task (0 through 12), plus the initial commit from brainstorming.

- [ ] **Step 4: Final commit marking Phase 1 complete**

```bash
git commit --allow-empty -m "$(cat <<'EOF'
Phase 1 complete: SS-DM-MoE hardened into tested dmoe package

All 8 bugs from the design spec are fixed and covered by tests:
automatic entropy tracking, learnable numeric-stream projection,
safe buffer replacement on expert mutation, robust token-dtype
detection, device-consistent mutation, topology-aware checkpointing,
optimizer-state-preserving rebuild, and clearer manifold naming.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

## Plan self-review notes

- **Spec coverage:** all 8 bugs map to a task (encoder→#2/#4, experts→#1/#3, model→#5/#7, checkpoint→#6, manifold→#8); every file in the spec's package layout has a creating task; every file in the spec's testing-plan section has a corresponding test task, plus `test_decoder.py` added for completeness since the spec's module list included `decoder.py` even though the prose testing section didn't spell out a dedicated file for it.
- **Type/signature consistency checked:** `DynamicMorphicMoE.mutate_topology()` (Task 5) is called with no arguments by `execute_graph_mutation()` (Task 9) and directly in tests — signature `last_entropy: float = None` supports both. `checkpoint.save_checkpoint`/`load_checkpoint` (Task 8) signatures match their usage as `checkpoint_module.save_checkpoint(self, path)` / `checkpoint_module.load_checkpoint(path, cls, device=device)` in `model.py` (Task 9). `rebuild_optimizer_preserving_state` (Task 9) signature matches its test call in Task 10.
- **No placeholders:** every step has complete, runnable code; no "TODO"/"add validation" left anywhere.
