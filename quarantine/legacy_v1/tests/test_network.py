import unittest
import torch
from ishmn.memory import InfiniteSphericalMemory
from ishmn.network import SphericalHyperspaceControllerNet
from ishmn.geometry import project_to_sphere

class TestNetworkAndMemory(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.batch_size = 4
        self.hyperspace_dim = 6
        self.payload_dim = 16
        self.num_slots = 12

    def test_memory_read_write(self):
        mem = InfiniteSphericalMemory(
            num_slots=self.num_slots,
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim,
            initial_radius_shells=3
        )
        state = mem.init_memory(self.batch_size)

        # Create query and write targets
        u_w = project_to_sphere(torch.randn(self.batch_size, self.hyperspace_dim))
        r_w = torch.ones(self.batch_size, 1) * 2.0
        add_vec = torch.randn(self.batch_size, self.payload_dim)
        erase_vec = torch.ones(self.batch_size, self.payload_dim) * 0.5
        write_gate = torch.ones(self.batch_size, 1)
        kappa = torch.ones(self.batch_size, 1) * 5.0
        beta = torch.ones(self.batch_size, 1) * 2.0

        # Perform write
        write_w, state = mem.write(state, u_w, r_w, add_vec, erase_vec, write_gate, kappa, beta)
        self.assertEqual(write_w.shape, (self.batch_size, self.num_slots))

        # Perform read at same location
        read_payload, read_w, state = mem.read(state, u_w, r_w, kappa, beta)
        self.assertEqual(read_payload.shape, (self.batch_size, self.payload_dim))
        self.assertEqual(read_w.shape, (self.batch_size, self.num_slots))

    def test_riemannian_steering(self):
        mem = InfiniteSphericalMemory(
            num_slots=self.num_slots,
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim
        )
        state = mem.init_memory(self.batch_size)
        
        tangent_vel = torch.randn(self.batch_size, self.hyperspace_dim) * 0.1
        radial_vel = torch.randn(self.batch_size, 1) * 0.1
        steered_state = mem.steer_memory_geodesics(state, tangent_vel, radial_vel)
        
        # Ensure steered directions remain strictly on S^{D-1}
        norms = torch.norm(steered_state["u"], p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        # Ensure radii remain positive
        self.assertTrue((steered_state["r"] >= 0).all())

    def test_dynamic_shell_spawning(self):
        mem = InfiniteSphericalMemory(
            num_slots=8,
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim,
            max_slots=16
        )
        state = mem.init_memory(self.batch_size)
        self.assertEqual(state["u"].size(1), 8)

        u_new = torch.randn(self.batch_size, self.hyperspace_dim)
        v_new = torch.randn(self.batch_size, self.payload_dim)
        state = mem.spawn_infinite_shell(state, u_new, v_new, shell_delta=1.5)
        self.assertEqual(state["u"].size(1), 9)

    def test_controller_net_forward_and_backward(self):
        input_dim = 10
        hidden_dim = 32
        output_dim = 10
        seq_len = 5

        model = SphericalHyperspaceControllerNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim,
            num_slots=self.num_slots,
            enable_steering=True
        )

        x_seq = torch.randn(self.batch_size, seq_len, input_dim, requires_grad=True)
        y_seq, final_state, traces = model(x_seq, record_traces=True)

        self.assertEqual(y_seq.shape, (self.batch_size, seq_len, output_dim))
        self.assertEqual(len(traces), seq_len)

        # Compute dummy loss and backward pass
        loss = y_seq.sum()
        loss.backward()

        # Check gradients exist and are finite
        self.assertIsNotNone(x_seq.grad)
        self.assertFalse(torch.isnan(x_seq.grad).any())
        self.assertIsNotNone(model.fc_steer_tangent.weight.grad)

    def test_tri_tier_memory(self):
        mem = InfiniteSphericalMemory(
            num_slots=8,
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim
        )
        state = mem.init_memory(1)

        # 1. Log multiple experiential episodes into Episodic Memory
        u_ep1 = project_to_sphere(torch.randn(1, self.hyperspace_dim))
        u_ep2 = project_to_sphere(u_ep1 + torch.randn(1, self.hyperspace_dim) * 0.05) # Closely resonant
        u_ep3 = project_to_sphere(torch.randn(1, self.hyperspace_dim))                 # Distant
        
        v_ep = torch.randn(1, self.payload_dim)
        mem.log_episode(u_ep1, torch.tensor([[1.5]]), v_ep, surprise=0.1, domain="physics")
        mem.log_episode(u_ep2, torch.tensor([[1.6]]), v_ep, surprise=0.2, domain="physics")
        mem.log_episode(u_ep3, torch.tensor([[3.0]]), v_ep, surprise=0.8, domain="neuroscience")
        
        self.assertEqual(len(mem.episodic_memory), 3)

        # 2. Resonant recall
        recalled = mem.recall_resonant_episodes(u_ep1, top_k=2)
        self.assertGreaterEqual(len(recalled), 1)
        self.assertEqual(recalled[0]["domain"], "physics")

        # 3. Semantic concept crystallization (sleep phase)
        cryst_report = mem.crystallize_semantic_concepts(cluster_angle_threshold=0.30, min_cluster_size=2)
        self.assertEqual(cryst_report["status"], "crystallized")
        self.assertGreaterEqual(len(mem.semantic_concepts), 1)

        # 4. Tri-tier recall
        kappa = torch.tensor([[4.0]])
        beta = torch.tensor([[1.0]])
        fused_payload, meta, state = mem.tri_tier_recall(state, u_ep1, torch.tensor([[1.5]]), kappa, beta)
        self.assertEqual(fused_payload.shape, (1, self.payload_dim))
        self.assertIn("recalled_episodes", meta)
        self.assertIn("recalled_semantics", meta)

if __name__ == '__main__':
    unittest.main()
