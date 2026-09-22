import pytest
import torch
from ishmn.geometry import geodesic_distance_sphere
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine


def test_native_language_learning_and_crystallization():
    device = torch.device("cpu")
    uce = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        device=device
    )

    mini_corpus = [
        "the king is a man who rules the kingdom",
        "the queen is a woman who rules the kingdom",
        "a king is to a man as a queen is to a woman",
        "the prince is a boy in the royal family",
        "the princess is a girl in the royal family"
    ] * 5

    res = uce.train_language_corpus(mini_corpus, epochs=5, lr=0.03, window_size=3)
    assert res["status"] == "trained_and_crystallized"
    assert res["vocabulary_size"] > 5
    assert len(uce.memory.semantic_concepts) > 5

    # Verify that concepts are stored in memory with valid coordinates on S^7
    concept = uce.memory.get_concept_by_word("king")
    assert concept is not None
    assert concept["u"].shape[-1] == 8
    assert torch.allclose(torch.norm(concept["u"], dim=-1), torch.tensor([1.0]), atol=1e-4)


def test_native_analogy_resolution_on_memory():
    device = torch.device("cpu")
    uce = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        device=device
    )

    # Pre-register learned test attractors directly in memory
    u_man = torch.zeros(1, 8)
    u_man[0, 0] = 1.0
    u_king = torch.zeros(1, 8)
    u_king[0, 0] = 0.8
    u_king[0, 1] = 0.6
    u_woman = torch.zeros(1, 8)
    u_woman[0, 2] = 1.0
    u_queen = torch.zeros(1, 8)
    u_queen[0, 2] = 0.8
    u_queen[0, 1] = 0.6

    uce.memory.add_semantic_concept("man", u_man)
    uce.memory.add_semantic_concept("king", u_king)
    uce.memory.add_semantic_concept("woman", u_woman)
    uce.memory.add_semantic_concept("queen", u_queen)

    # Solve analogy
    analogy = uce.solve_analogy("man", "king", "woman", modality="subwords")
    assert analogy["predicted_word"] == "queen"
    assert analogy["isometry_preserved"] is True
    assert analogy["confidence"] > 70.0


def test_honest_knowledge_boundary_detection():
    device = torch.device("cpu")
    uce = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        device=device
    )

    # Clear memory concepts
    uce.memory.semantic_concepts.clear()

    # Query unknown words not in memory
    res = uce.solve_analogy("marsupial", "kangaroo", "monotreme")
    assert "error" in res
    assert res["predicted_word"] is None
    assert res["confidence"] == 0.0
    assert "marsupial" in res["error"]
