import unittest
import torch
from ishmn.perception import UniversalPerceptor
from ishmn.memory import InfiniteSphericalMemory
from ishmn.evolution_engine import (
    PredictiveWorldModel,
    IntrinsicCuriosityEngine,
    DreamingConsolidationEngine
)

class TestEvolution(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.B = 2
        self.hyperspace_dim = 6
        self.payload_dim = 16

    def test_universal_perceptor(self):
        perceptor = UniversalPerceptor(
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim,
            embedding_dim=32
        )

        # Test continuous vector input
        cont_in = torch.randn(self.B, 10)
        u, r, v = perceptor(cont_in, modality="continuous")
        self.assertEqual(u.shape, (self.B, self.hyperspace_dim))
        self.assertEqual(r.shape, (self.B, 1))
        self.assertEqual(v.shape, (self.B, self.payload_dim))
        # Ensure u is on unit sphere
        norms = torch.norm(u, p=2, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        # Ensure r is positive
        self.assertTrue((r > 0).all())

        # Test discrete tokens input
        tokens_in = torch.randint(0, 100, (self.B, 8))
        u_tok, r_tok, v_tok = perceptor(tokens_in, modality="tokens")
        self.assertEqual(u_tok.shape, (self.B, self.hyperspace_dim))

    def test_predictive_world_model_and_curiosity(self):
        world_model = PredictiveWorldModel(
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim,
            hidden_dim=32
        )
        curiosity_engine = IntrinsicCuriosityEngine()

        u = torch.randn(self.B, self.hyperspace_dim)
        u = u / torch.norm(u, dim=-1, keepdim=True)
        r = torch.ones(self.B, 1) * 2.0
        v = torch.randn(self.B, self.payload_dim)

        u_pred, r_pred, v_pred = world_model(u, r, v)
        self.assertEqual(u_pred.shape, (self.B, self.hyperspace_dim))

        surprise, metrics = curiosity_engine.compute_surprise(
            u_pred, r_pred, v_pred, u, r, v
        )
        self.assertEqual(surprise.shape, (self.B,))
        self.assertIn("surprise_mean", metrics)

    def test_dreaming_consolidation(self):
        mem = InfiniteSphericalMemory(
            num_slots=8,
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim,
            initial_radius_shells=3
        )
        state = mem.init_memory(self.B)
        world_model = PredictiveWorldModel(
            hyperspace_dim=self.hyperspace_dim,
            payload_dim=self.payload_dim
        )
        dreamer = DreamingConsolidationEngine(mem, world_model)

        new_state, metrics = dreamer.dream_and_consolidate(state, dream_steps=2)
        self.assertIn("active_slots", metrics)
        self.assertIn("max_shell_radius", metrics)
        self.assertEqual(new_state["u"].size(0), self.B)

if __name__ == '__main__':
    unittest.main()
