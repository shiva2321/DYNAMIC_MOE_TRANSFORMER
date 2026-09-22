"""
Regression Tests: REM Consolidation Must Not Destroy Causal Reversibility.

Root cause guarded by these tests:
    sleep_consolidate()'s REM objective is geodesic alignment to crystallized anchors
    plus SIGReg. It contains no reversibility term, and it is evaluated on synthetic
    anchor latents (z[:, :8] = u * r, remaining 24 dims exactly zero, ||z|| ~ 1.0)
    rather than real Layer-14 encoded states (||z|| ~ 7.7, all 32 dims active).
    Stepping w_fwd / w_bwd under that objective destroys W_bwd o W_fwd ~ id.

    Measured on artifacts/unified_lifelong_brain.pt against 3000 real Layer-14
    latents: causal_residual 0.392 -> 1.157 rad over 20 REM steps. Freezing the
    transition pair reduced the same drift to +0.047 rad.
"""

import math
import types

import pytest
import torch
import torch.nn.functional as F

from ishmn import UniversalCognitiveEngine, GroundedHybridEngine
from ishmn.geometry import geodesic_distance_sphere
from ishmn.sigreg import SIGReg


@pytest.fixture
def uce():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        max_experts=8,
        consensus_threshold=0.70,
        max_thinking_steps=3,
        device=device,
    )
    for i in range(12):
        u = F.normalize(torch.randn(8, device=device), p=2, dim=-1)
        engine.memory.semantic_concepts.append({
            "concept_id": i,
            "u": u,
            "r": 1.5,
            "word": f"concept_{i}",
            "usage_count": 10,
        })
    return engine


def _in_distribution_probe(uce, n: int = 256):
    """Latents matched to the real Layer-14 encoded distribution (all dims active)."""
    gen = torch.Generator(device="cpu").manual_seed(17)
    z = (torch.randn(n, uce.d_model, generator=gen) * 1.20).to(uce.device)
    c = (torch.randn(n, uce.d_model, generator=gen) * 0.55).to(uce.device)
    return z, c


def _fit_transition_pair(uce, steps: int = 150):
    """Give w_fwd / w_bwd a genuine inverse relationship so there is something to destroy."""
    z, c = _in_distribution_probe(uce, n=512)
    z_next = z + 0.35 * torch.tanh(z @ torch.eye(uce.d_model, device=uce.device).roll(1, 0))
    params = list(uce.world_model.w_fwd.parameters()) + list(uce.world_model.w_bwd.parameters())
    opt = torch.optim.AdamW(params, lr=3e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = (
            F.mse_loss(uce.world_model.forward_transition(z, c), z_next)
            + F.mse_loss(uce.world_model.backward_transition(z_next, c), z)
            + uce.world_model.causal_residual(z, c).mean()
        )
        loss.backward()
        opt.step()
    return z, c


def _rem_loss(uce, sigreg, batch: int = 6):
    """Reproduces the REM objective from GroundedHybridEngine.sleep_consolidate."""
    concepts = uce.memory.semantic_concepts[:batch]
    u_targets = torch.stack([sc["u"] for sc in concepts], dim=0)
    r_targets = torch.ones(len(concepts), 1, device=uce.device) * 1.5

    v_raw = torch.randn_like(u_targets) * 0.05
    v_tan = v_raw - (v_raw * u_targets).sum(dim=-1, keepdim=True) * u_targets
    v_norm = torch.norm(v_tan, dim=-1, keepdim=True) + 1e-7
    u_pert = F.normalize(
        torch.cos(v_norm) * u_targets + torch.sin(v_norm) * (v_tan / v_norm), p=2, dim=-1
    )

    z_init = torch.zeros(len(concepts), uce.d_model, device=uce.device)
    z_init[:, : u_pert.shape[-1]] = u_pert * r_targets

    delib = uce.world_model.deliberate_and_halting_loop(
        z_init=z_init, context=z_init, moe_layer=uce.moe
    )
    u_relaxed, _ = uce.world_model.project_manifold(delib["final_latent"])
    geo_loss = geodesic_distance_sphere(u_relaxed, u_targets).mean()
    sig_loss, _ = sigreg(u_relaxed * math.sqrt(uce.hyperspace_dim))
    return geo_loss + 0.2 * sig_loss


def test_rem_step_does_not_update_causal_transition_pair(uce):
    """The REM optimizer step must leave w_fwd / w_bwd bit-identical."""
    sigreg = SIGReg(num_slices=16, lambda_reg=0.2).to(uce.device)
    stub = types.SimpleNamespace(uce=uce)

    before = {
        name: p.detach().clone()
        for name, p in uce.world_model.named_parameters()
        if name.startswith("w_fwd.") or name.startswith("w_bwd.")
    }
    assert before, "expected w_fwd / w_bwd parameters to exist"

    for _ in range(5):
        GroundedHybridEngine._rem_optimizer_step(stub, _rem_loss(uce, sigreg))

    for name, p in uce.world_model.named_parameters():
        if name in before:
            assert torch.equal(p.detach(), before[name]), (
                f"REM consolidation modified causal transition parameter {name}; "
                "this destroys W_bwd o W_fwd ~ id."
            )


def test_rem_consolidation_preserves_causal_reversibility(uce):
    """Causal residual on in-distribution latents must not regress across REM replay."""
    sigreg = SIGReg(num_slices=16, lambda_reg=0.2).to(uce.device)
    stub = types.SimpleNamespace(uce=uce)

    z, c = _fit_transition_pair(uce)
    with torch.no_grad():
        before = uce.world_model.causal_residual(z, c).mean().item()
    assert before < 0.35, f"fixture did not establish an inverse pair (residual {before:.4f})"

    for _ in range(20):
        GroundedHybridEngine._rem_optimizer_step(stub, _rem_loss(uce, sigreg))

    with torch.no_grad():
        after = uce.world_model.causal_residual(z, c).mean().item()

    assert after - before < 0.05, (
        f"REM consolidation degraded causal reversibility: "
        f"{before:.4f} -> {after:.4f} rad (+{after - before:.4f})"
    )


def test_rem_step_still_updates_consolidation_targets(uce):
    """The fix must not freeze everything -- MoE / perceptor must still consolidate."""
    sigreg = SIGReg(num_slices=16, lambda_reg=0.2).to(uce.device)
    stub = types.SimpleNamespace(uce=uce)

    before = {n: p.detach().clone() for n, p in uce.named_parameters()}
    for _ in range(5):
        GroundedHybridEngine._rem_optimizer_step(stub, _rem_loss(uce, sigreg))

    changed = [
        n for n, p in uce.named_parameters()
        if n in before and not torch.equal(p.detach(), before[n])
    ]
    assert changed, "REM consolidation updated no parameters at all"
    assert not any(
        n.startswith("world_model.w_fwd.") or n.startswith("world_model.w_bwd.")
        for n in changed
    )
