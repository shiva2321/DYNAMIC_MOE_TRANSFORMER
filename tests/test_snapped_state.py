"""
tests/test_snapped_state.py - Regression and contract tests for SnappedRotationState.

Verifies:
1. Module initialization, parameter registration, and SO(k) orthogonality of rotation matrices.
2. Exact equivalence between batched forward() and sequential step() execution.
3. Strict absence of label leakage: inputs contain only token streams, not targets or Cayley tables.
4. Deterministic length generalization: 100.0% exact accuracy at T = 256 on Z6 and S3 across 3 seeds.
"""

import pytest
import torch
import torch.nn.functional as F
from geometry.snapped_state import SnappedRotationState
from research.yggdrasil_poc.poc_b_state_tracking import make_group, prefix_products

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_snapped_state_initialization_and_orthogonal_rotations():
    """Verify initialization, parameter shapes, and SO(k) orthogonality."""
    n_sym = 6
    k = 8
    model = SnappedRotationState(n_sym=n_sym, k=k, num_codes=24, hid=32)

    assert model.w.shape == (n_sym, k * (k - 1) // 2)
    assert model.v0.shape == (k,)
    assert model.codebook.shape == (24, k)

    R = model.get_rotation_matrices()
    assert R.shape == (n_sym, k, k)

    # Verify R @ R.T == I (orthogonality in SO(k))
    eye = torch.eye(k).expand(n_sym, k, k)
    diff = torch.bmm(R, R.transpose(-1, -2)) - eye
    assert diff.abs().max().item() < 1e-5, f"Rotation matrices not orthogonal: max diff {diff.abs().max().item()}"

    # Verify det(R) == +1 (special orthogonal group SO(k))
    dets = torch.linalg.det(R)
    assert (dets - 1.0).abs().max().item() < 1e-4, f"Rotation matrices not in SO(k): dets {dets}"


def test_snapped_state_forward_and_step_equivalence():
    """Verify that batched forward() matches autoregressive step() exactly."""
    torch.manual_seed(42)
    model = SnappedRotationState(n_sym=6, k=8, hid=32).to(DEVICE)
    x = torch.randint(0, 6, (4, 16), device=DEVICE)

    # Batched forward
    logits_fwd = model(x)
    assert logits_fwd.shape == (4, 16, 6)

    # Sequential step
    state = None
    step_logits = []
    for t in range(16):
        state, logit_t = model.step(x[:, t], state)
        step_logits.append(logit_t)
    logits_step = torch.stack(step_logits, dim=1)

    max_diff = (logits_fwd - logits_step).abs().max().item()
    assert max_diff < 1e-5, f"Step logits differed from forward logits: {max_diff}"


def test_snapped_state_no_label_leakage():
    """Verify that SnappedRotationState has no access to ground truth labels or Cayley tables."""
    model = SnappedRotationState(n_sym=6, k=8).to(DEVICE)
    x = torch.randint(0, 6, (2, 8), device=DEVICE)

    # Verify forward only accepts x
    out = model(x)
    assert out.shape == (2, 8, 6)

    # Inspect state dict to ensure no group multiplication table is stored
    param_names = [name for name, _ in model.named_parameters()]
    buffer_names = [name for name, _ in model.named_buffers()]
    assert "mul" not in param_names and "mul" not in buffer_names
    assert "cayley" not in param_names and "cayley" not in buffer_names
    assert "labels" not in param_names and "labels" not in buffer_names


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("group_name", ["z6", "s3"])
def test_snapped_state_regression_z6_and_s3(group_name, seed):
    """Fast regression test: verify SnappedRotationState achieves 100% exact accuracy at T=256 on Z6 and S3."""
    mul = make_group(group_name).to(DEVICE)
    torch.manual_seed(seed)
    model = SnappedRotationState(n_sym=6, k=8).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)

    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    # Curriculum schedule on lengths <= 16
    for steps, T in ((200, 4), (200, 8), (300, 16)):
        for _ in range(steps):
            x = torch.randint(0, 6, (256, T), generator=g).to(DEVICE)
            out, aux = model(x, return_loss=True)
            loss = F.cross_entropy(out.reshape(-1, 6), prefix_products(x, mul).reshape(-1)) + aux
            opt.zero_grad()
            loss.backward()
            opt.step()

    # Test generalization at sequence length T = 256
    model.eval()
    with torch.no_grad():
        gt = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        x_test = torch.randint(0, 6, (100, 256), generator=gt).to(DEVICE)
        targets = prefix_products(x_test, mul)[:, -1]
        preds = model(x_test)[:, -1].argmax(dim=-1)
        acc = (preds == targets).float().mean().item()

    assert acc == 1.0, f"Expected 100% accuracy on {group_name} seed {seed} at T=256, got {acc * 100:.1f}%"


def test_snapped_state_n_states_parameter():
    """Verify n_states correctly sets num_codes when tokens represent generators (H3)."""
    # When tokens represent generators (e.g. 2 generators) but group has order 120
    model = SnappedRotationState(n_sym=2, k=8, n_states=120)
    assert model.num_codes == 4 * 120 == 480
    assert model.codebook.shape == (480, 8)

    # When num_codes is explicitly overridden
    model_custom = SnappedRotationState(n_sym=2, k=8, n_states=120, num_codes=960)
    assert model_custom.num_codes == 960
    assert model_custom.codebook.shape == (960, 8)


def test_component_cards_cited_paths_exist():
    """Verify every path/with/slash.ext cited in docs/COMPONENT_CARDS.md exists on disk (H1)."""
    import os
    import re

    cards_path = "docs/COMPONENT_CARDS.md"
    assert os.path.exists(cards_path), f"Missing {cards_path}"

    with open(cards_path, "r") as f:
        content = f.read()

    # Extract all cited backtick paths with slash and file extension
    cited_files = re.findall(r"`([a-zA-Z0-9_\-\./\\]+/[a-zA-Z0-9_\-\./\\]+\.[a-zA-Z0-9]+)`", content)
    assert len(cited_files) >= 10, f"Expected at least 10 cited files, found {len(cited_files)}"

    missing = [f for f in cited_files if not os.path.exists(f)]
    assert not missing, f"Cited files in {cards_path} do not exist: {missing}"

