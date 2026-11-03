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
