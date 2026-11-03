import math
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
    moe = DynamicMorphicMoE(d_model=8, d_ff=16, initial_experts=4)
    with torch.no_grad():
        moe.router_linear.weight.zero_()
        moe.router_linear.bias.zero_()
    z_r = torch.randn(2, 4, 8)
    z_i = torch.randn(2, 4, 8)
    _, _, entropy = moe(z_r, z_i)
    expected_entropy = math.log(4.0)
    assert torch.allclose(entropy, torch.tensor(expected_entropy), atol=1e-5)
    assert torch.allclose(moe._last_entropy, torch.tensor(expected_entropy), atol=1e-5)


def test_mutate_topology_spawns_expert_on_high_entropy_from_forward():
    # Uniform 3-expert routing has entropy ln(3) ~= 1.0986 > tau_split=1.0
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
    for step in range(12):
        if step % 3 == 2:
            moe._last_entropy = torch.tensor(0.1)
            with torch.no_grad():
                moe.expert_usage[0] = 0.001
        else:
            moe._last_entropy = torch.tensor(5.0)
        moe.mutate_topology()
        assert moe.expert_usage.shape == (moe.num_experts,)
        assert moe.expert_usage.dtype == torch.float32
        assert moe.expert_usage.device == moe.router_linear.weight.device
        assert moe.router_linear.out_features == moe.num_experts


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
        moe.expert_age.copy_(torch.tensor([5.0, 0.0]))
    moe.freeze_mature_experts()

    before_frozen = moe.experts[0].linear1.weight_real.clone()
    before_active = moe.experts[1].linear1.weight_real.clone()
    optimizer = torch.optim.AdamW(moe.parameters(), lr=1e-1)

    z_r = torch.randn(2, 4, 8)
    z_i = torch.randn(2, 4, 8)
    acc_r, acc_i, _ = moe(z_r, z_i)
    (acc_r.sum() + acc_i.sum()).backward()
    optimizer.step()

    assert torch.equal(before_frozen, moe.experts[0].linear1.weight_real)
    assert not torch.equal(before_active, moe.experts[1].linear1.weight_real)


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
        d_model=8, d_ff=16, initial_experts=3, tau_split=1.0, tau_death=0.9, min_experts=2, maturity_age=1
    )
    with torch.no_grad():
        moe.expert_age.fill_(5.0)
    moe.freeze_mature_experts()
    moe._last_entropy = torch.tensor(5.0)
    moe.mutate_topology()
    assert moe.expert_age.shape == (4,)
    assert moe.expert_frozen.shape == (4,)
    assert moe.expert_age[-1].item() == 0.0
    assert bool(moe.expert_frozen[-1]) is False
    assert moe.experts[-1].linear1.weight_real.requires_grad is True

    with torch.no_grad():
        moe.expert_usage[0] = 0.0001
    moe._last_entropy = torch.tensor(0.1)
    moe.mutate_topology()
    assert moe.expert_age.shape == (moe.num_experts,)
    assert moe.expert_frozen.shape == (moe.num_experts,)


