"""
Unit Tests: Anisotropy-Driven Expert Mitosis on S^{D-1}.

The previous spawn rule compared an unnormalised resonance score against an absolute
constant. Measured against the live brain, the minimum max-resonance over all 5,295
crystallized concepts was 1.237 while spawn_threshold was 0.40 -- the criterion was not
merely rare, it was unreachable, and it could not distinguish a genuinely bimodal
receptive field from one that was simply diffuse.

These tests pin the replacement: a split fires only when an expert's tangent-space
second moment is large (lambda_1 > tau), elongated (lambda_1/lambda_2 > rho) AND
bimodal along that axis (excess kurtosis < kappa < 0). The kurtosis gate is what stops
a unimodal stretched field from being fragmented -- without it, splitting the live brain
worsened routing imbalance (Gini 0.509 -> 0.627).
"""

import math

import pytest
import torch
import torch.nn.functional as F

from ishmn.geometry import exp_map_sphere, geodesic_distance_sphere, project_to_sphere
from ishmn.spherical_dynamic_moe import SphericalDynamicMoE


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _moe(num_experts: int = 2, **kw):
    return SphericalDynamicMoE(
        d_model=32,
        d_ff=64,
        d_context=32,
        hyperspace_dim=8,
        top_k=1,
        initial_experts=num_experts,
        split_min_count=16,
        device=DEVICE,
        **kw,
    )


def _field(moe, expert_idx: int, spread_along: torch.Tensor, n: int = 256, shape: str = "bimodal"):
    """
    Generates inputs around an expert's anchor with a controlled shape:
      'isotropic' - round blob            (low anisotropy, kurtosis ~ 0)
      'elongated' - unimodal Gaussian ray (high anisotropy, kurtosis ~ 0)
      'bimodal'   - two separated lobes   (high anisotropy, kurtosis < 0)
    """
    mu = moe.expert_u[expert_idx].unsqueeze(0).expand(n, -1)
    g = torch.Generator(device="cpu").manual_seed(3)
    if shape == "isotropic":
        v = (torch.randn(n, moe.hyperspace_dim, generator=g) * 0.30).to(DEVICE)
    else:
        if shape == "elongated":
            coeff = torch.randn(n, 1, generator=g) * 0.55
        elif shape == "bimodal":
            sign = torch.where(torch.rand(n, 1, generator=g) < 0.5, -1.0, 1.0)
            coeff = sign * 0.55 + torch.randn(n, 1, generator=g) * 0.06
        else:
            raise ValueError(shape)
        jitter = (torch.randn(n, moe.hyperspace_dim, generator=g) * 0.02).to(DEVICE)
        v = coeff.to(DEVICE) * spread_along.unsqueeze(0) + jitter
    v = v - (v * mu).sum(-1, keepdim=True) * mu
    return exp_map_sphere(mu, v)


def _accumulate(moe, expert_idx: int, axis: torch.Tensor, shape: str, rounds: int = 40, n: int = 64):
    idx = torch.full((n,), expert_idx, dtype=torch.long, device=DEVICE)
    for _ in range(rounds):
        moe.observe_receptive_field(_field(moe, expert_idx, axis, n=n, shape=shape), idx)


def test_observe_receptive_field_accumulates_scatter():
    moe = _moe()
    assert moe.expert_scatter.shape == (2, 8, 8)
    assert torch.count_nonzero(moe.expert_scatter) == 0

    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    u = _field(moe, 0, axis, n=64)
    moe.observe_receptive_field(u, torch.zeros(64, dtype=torch.long, device=DEVICE))

    assert torch.count_nonzero(moe.expert_scatter[0]) > 0
    assert moe.expert_reservoir_fill[0].item() == 64
    assert torch.count_nonzero(moe.expert_scatter[1]) == 0
    assert moe.expert_route_count[0].item() == 64


def test_isotropic_field_does_not_split():
    """A round receptive field is diffuse, not bimodal -- it must not trigger mitosis."""
    moe = _moe()
    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    _accumulate(moe, 0, axis, "isotropic")

    crit = moe.evaluate_split_criterion(0)
    assert crit["route_count"] >= moe.split_min_count
    assert crit["anisotropy"] < moe.split_anisotropy_ratio, (
        f"isotropic field reported anisotropy {crit['anisotropy']:.3f}"
    )
    assert not crit["should_split"]
    assert moe.mitose_anisotropic(0) is None


def test_elongated_field_splits_into_separated_daughters():
    moe = _moe()
    before = moe.num_experts
    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    _accumulate(moe, 0, axis, "bimodal")

    crit = moe.evaluate_split_criterion(0)
    assert crit["should_split"], f"criterion did not fire: {crit}"
    expected_sep = 2.0 * math.sqrt(crit["lambda_1"])

    child = moe.mitose_anisotropic(0)
    assert child is not None and moe.num_experts == before + 1

    sep = geodesic_distance_sphere(
        moe.expert_u[0].unsqueeze(0), moe.expert_u[child].unsqueeze(0)
    ).item()
    assert sep == pytest.approx(expected_sep, rel=0.15), (
        f"daughters separated by {sep:.4f} rad, expected ~{expected_sep:.4f}"
    )
    assert sep > moe.merge_angle_threshold, "daughters would be merged back immediately"


def test_daughters_are_differentiated_and_mirrored():
    """Mirrored perturbation: daughters differ, and the parent sits midway between them."""
    moe = _moe()
    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    pre = {k: v.detach().clone() for k, v in moe.experts[0].state_dict().items()}
    _accumulate(moe, 0, axis, "bimodal")

    child = moe.mitose_anisotropic(0)
    assert child is not None

    sd_p = moe.experts[0].state_dict()
    sd_c = moe.experts[child].state_dict()
    float_keys = [k for k in pre if torch.is_floating_point(pre[k])]
    assert float_keys

    assert any(not torch.allclose(sd_p[k], sd_c[k]) for k in float_keys), "daughters are identical"
    for k in float_keys:
        midpoint = 0.5 * (sd_p[k] + sd_c[k])
        assert torch.allclose(midpoint, pre[k], atol=1e-6), f"{k} is not mirrored about the parent"


def test_split_grants_merge_immunity():
    moe = _moe()
    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    _accumulate(moe, 0, axis, "bimodal")

    child = moe.mitose_anisotropic(0)
    assert child is not None
    assert moe.expert_lockout[0] > 0 and moe.expert_lockout[child] > 0

    # Force the anchors together so the merge rule would otherwise fire immediately.
    moe.expert_u[child] = moe.expert_u[0].clone()
    n_before = moe.num_experts
    report = moe.sleep_consolidate_and_prune(prune_usage_threshold=0.0)
    assert report["merged"] == 0, "merge immunity did not protect the fresh split"
    assert moe.num_experts == n_before


def test_buffers_stay_consistent_across_split_and_prune():
    moe = _moe(num_experts=3)
    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    _accumulate(moe, 0, axis, "bimodal")
    moe.mitose_anisotropic(0)
    moe.sleep_consolidate_and_prune(prune_usage_threshold=0.0)

    e = moe.num_experts
    for name in ["expert_u", "expert_r", "expert_usage", "expert_bias",
                 "routed_load_ema", "expert_scatter", "expert_route_count", "expert_lockout",
                 "expert_reservoir", "expert_reservoir_fill"]:
        assert getattr(moe, name).size(0) == e, f"{name} has {getattr(moe, name).size(0)} rows, expected {e}"


def test_forward_still_works_and_collects_stats_in_training():
    moe = _moe(num_experts=3).train()
    x = torch.randn(48, 32, device=DEVICE)
    u = project_to_sphere(torch.randn(48, 8, device=DEVICE))
    r = torch.rand(48, 1, device=DEVICE) + 1.0

    out, meta = moe(x, u, r)
    assert out.shape == (48, 32)
    assert moe.expert_route_count.sum().item() == 48
    assert torch.count_nonzero(moe.expert_scatter) > 0


def test_unimodal_elongated_field_does_not_split():
    """High anisotropy is not enough: a stretched but single-lobed field must be left alone."""
    moe = _moe()
    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    _accumulate(moe, 0, axis, "elongated")

    crit = moe.evaluate_split_criterion(0)
    assert crit["anisotropy"] > moe.split_anisotropy_ratio, (
        "fixture failed to produce an elongated field"
    )
    assert crit["excess_kurtosis"] > moe.split_kurtosis_threshold, (
        f"unimodal ray reported excess kurtosis {crit['excess_kurtosis']:.3f}"
    )
    assert not crit["should_split"], "anisotropy alone triggered a split on a unimodal field"
    assert moe.mitose_anisotropic(0) is None


def test_bimodal_field_is_detected_as_platykurtic():
    moe = _moe()
    axis = project_to_sphere(torch.randn(8, device=DEVICE))
    _accumulate(moe, 0, axis, "bimodal")

    crit = moe.evaluate_split_criterion(0)
    assert crit["excess_kurtosis"] < moe.split_kurtosis_threshold
    assert crit["should_split"]


def test_legacy_checkpoint_without_mitosis_buffers_loads_strictly():
    """A checkpoint written before receptive-field tracking must still load with strict=True."""
    donor = _moe(num_experts=3)
    legacy = {
        k: v.clone() for k, v in donor.state_dict().items()
        if not k.startswith(("expert_scatter", "expert_route_count", "expert_lockout",
                             "expert_reservoir", "expert_reservoir_fill"))
    }
    assert not any("reservoir" in k for k in legacy)

    fresh = _moe(num_experts=3)
    fresh.load_state_dict(legacy, strict=True)

    assert fresh.expert_reservoir.size(0) == fresh.num_experts
    assert fresh.expert_reservoir_fill.sum().item() == 0.0
    assert fresh.expert_scatter.abs().sum().item() == 0.0
