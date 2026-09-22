import unittest
import torch
from ishmn.dynamic_optimizer import DynamicWarmupAdamW
from ishmn.dendritic_expert import TwoCompartmentDendriticExpert
from ishmn.spherical_dynamic_moe import SphericalDynamicMoE
from ishmn.continuous_self_loop import ContinuousSelfLoop, TriggeredEmission
from ishmn.geometry import project_to_sphere

class TestDynamicMoE(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.d_model = 16
        self.d_ff = 32
        self.hyperspace_dim = 6

    def test_dynamic_warmup_adamw(self):
        linear = torch.nn.Linear(10, 10)
        optimizer = DynamicWarmupAdamW(linear.parameters(), lr=1e-3, default_group_warmup_steps=10)
        self.assertEqual(len(optimizer.param_groups), 1)

        # Dynamically spawn a new parameter
        new_param = torch.nn.Parameter(torch.randn(10, 10))
        optimizer.add_dynamic_param_group([new_param], group_name="spawned_test")
        self.assertEqual(len(optimizer.param_groups), 2)

        # Check warmup progression
        optimizer.update_dynamic_schedules()
        self.assertLess(optimizer.param_groups[1]["lr"], optimizer.param_groups[0]["lr"])

    def test_dendritic_expert(self):
        expert = TwoCompartmentDendriticExpert(
            d_model=self.d_model,
            d_ff=self.d_ff,
            d_context=self.d_model,
            beta_nmda=0.5
        )
        x = torch.randn(2, self.d_model)
        ctx = torch.randn(2, self.d_model)

        # Forward with context
        y_ctx = expert(x, context=ctx)
        self.assertEqual(y_ctx.shape, (2, self.d_model))

        # Forward without context
        y_no_ctx = expert(x, context=None)
        self.assertEqual(y_no_ctx.shape, (2, self.d_model))

    def test_spherical_dynamic_moe_spawning_and_routing(self):
        moe = SphericalDynamicMoE(
            d_model=self.d_model,
            d_ff=self.d_ff,
            d_context=self.d_model,
            hyperspace_dim=self.hyperspace_dim,
            initial_experts=2,
            max_experts=6,
            spawn_threshold=0.80  # high threshold to trigger spawning
        )
        optimizer = DynamicWarmupAdamW(moe.parameters(), lr=1e-3)
        self.assertEqual(moe.num_experts, 2)

        x = torch.randn(3, self.d_model)
        u_x = project_to_sphere(torch.randn(3, self.hyperspace_dim))
        r_x = torch.tensor([[4.0], [4.5], [5.0]])

        # Forward should trigger neurogenesis
        moe.train()
        out, meta = moe(x, u_x, r_x, optimizer=optimizer)
        self.assertEqual(out.shape, (3, self.d_model))
        self.assertGreater(moe.num_experts, 2)
        self.assertEqual(len(optimizer.param_groups), 2)

    def test_continuous_self_loop_execution(self):
        loop = ContinuousSelfLoop(
            d_model=self.d_model,
            d_hidden=32,
            hyperspace_dim=self.hyperspace_dim,
            emission_threshold=0.01,  # low threshold to verify emission object creation
            device=self.device
        )

        # Inject input perturbation
        stim = torch.randn(1, self.d_model)
        loop.inject_input(stim, modality="continuous", tag="test_stimulus")

        # Run tick
        emission = loop.loop_tick()
        self.assertIsNotNone(emission)
        self.assertIsInstance(emission, TriggeredEmission)
        self.assertEqual(emission.active_experts, loop.moe.num_experts)

    def test_spherical_dynamic_moe_sleep_consolidation_and_apoptosis(self):
        moe = SphericalDynamicMoE(
            d_model=self.d_model,
            d_ff=self.d_ff,
            d_context=self.d_model,
            hyperspace_dim=self.hyperspace_dim,
            initial_experts=4,
            min_experts=1,
            merge_angle_threshold=0.30
        )
        optimizer = DynamicWarmupAdamW(moe.parameters(), lr=1e-3)
        self.assertEqual(moe.num_experts, 4)

        # Force expert 0 and 1 to be co-resonant (almost identical anchor angle)
        with torch.no_grad():
            moe.expert_u[1] = moe.expert_u[0].clone()
            # Expert 3 is completely dormant (low usage), while others have high usage
            moe.expert_usage = torch.tensor([50.0, 50.0, 50.0, 0.1])

        # Run sleep consolidation with apoptotic threshold 0.05
        report = moe.sleep_consolidate_and_prune(prune_usage_threshold=0.05, optimizer=optimizer)

        self.assertGreaterEqual(report["merged"], 1)
        self.assertGreaterEqual(report["pruned"], 1)
        self.assertLess(moe.num_experts, 4)
        self.assertGreaterEqual(moe.num_experts, moe.min_experts)

        # Forward pass post-sleep to verify stability
        x = torch.randn(2, self.d_model)
        u_x = project_to_sphere(torch.randn(2, self.hyperspace_dim))
        r_x = torch.tensor([[1.0], [1.5]])
        out, meta = moe(x, u_x, r_x, optimizer=optimizer)
        self.assertEqual(out.shape, (2, self.d_model))

        # Backward pass to ensure optimizer and parameter groups are intact
        loss = out.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == '__main__':
    unittest.main()

