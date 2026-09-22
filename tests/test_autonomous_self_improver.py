"""
Unit tests for Autonomous Self-Improving Reasoner (ASER).
Tests concept pair sampling, relational hypothesis synthesis,
unsupervised causal verification, and self-play step execution.
"""

import pytest
import torch
from unittest.mock import MagicMock

from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.grounded_hybrid_engine import GroundedHybridEngine
from ishmn.autonomous_self_improver import AutonomousSelfImprover
from ishmn.geometry import project_to_sphere


@pytest.fixture
def mock_engine():
    device = torch.device("cpu")
    uce = UniversalCognitiveEngine(
        hyperspace_dim=8,
        d_model=32,
        d_hidden=64,
        num_memory_slots=8,
        device=device
    )
    # Add mock semantic concepts
    uce.memory.semantic_concepts = [
        {
            "concept_id": 0,
            "word": "gravity",
            "u": project_to_sphere(torch.randn(1, 8)),
            "r": 1.5,
            "v": torch.randn(1, 32),
            "domain": "physics",
            "metadata": {"description": "Universal gravitational attraction proportional to mass."}
        },
        {
            "concept_id": 1,
            "word": "orbit",
            "u": project_to_sphere(torch.randn(1, 8)),
            "r": 1.8,
            "v": torch.randn(1, 32),
            "domain": "celestial",
            "metadata": {"description": "Curved path of a celestial object around a star or planet."}
        },
        {
            "concept_id": 2,
            "word": "energy",
            "u": project_to_sphere(torch.randn(1, 8)),
            "r": 1.2,
            "v": torch.randn(1, 32),
            "domain": "thermodynamics",
            "metadata": {"description": "Conserved quantitative property transferred to perform work."}
        }
    ]

    # Create mock engine
    engine = MagicMock()
    engine.uce = uce
    engine.device = device

    # Mock SLM and tokenizer
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "mock prompt"
    tokenizer.decode.return_value = "Gravity provides the centripetal force necessary to maintain an orbital trajectory."
    
    slm = MagicMock()
    slm.generate.return_value = torch.tensor([[10, 20, 30]])

    engine.slm = slm
    engine.tokenizer = tokenizer
    return engine


def test_sample_concept_pair(mock_engine):
    improver = AutonomousSelfImprover(mock_engine)
    c1, c2 = improver.sample_concept_pair()
    assert c1["word"] != c2["word"]
    assert "u" in c1 and "u" in c2


def test_synthesize_relational_hypothesis(mock_engine):
    improver = AutonomousSelfImprover(mock_engine)
    c1 = mock_engine.uce.memory.semantic_concepts[0]
    c2 = mock_engine.uce.memory.semantic_concepts[1]

    hypo = improver.synthesize_relational_hypothesis(c1, c2)
    assert hypo["concept_a"] == "gravity"
    assert hypo["concept_b"] == "orbit"
    assert "geodesic_distance" in hypo
    assert "u_midpoint" in hypo
    assert len(hypo["hypothesis_text"]) > 0


def test_verify_hypothesis_causality(mock_engine):
    improver = AutonomousSelfImprover(mock_engine)
    c1 = mock_engine.uce.memory.semantic_concepts[0]
    c2 = mock_engine.uce.memory.semantic_concepts[1]
    hypo = improver.synthesize_relational_hypothesis(c1, c2)

    verif = improver.verify_hypothesis_causality(hypo)
    assert "verdict" in verif
    assert "causal_error" in verif
    assert "consensus" in verif
    assert verif["verdict"] in ["verified_invariant", "contradiction_detected", "uncertain", "rejected_empty"]


def test_run_self_play_step(mock_engine):
    improver = AutonomousSelfImprover(mock_engine)
    initial_concepts = len(mock_engine.uce.memory.semantic_concepts)

    record = improver.run_self_play_step()
    assert record["cycle"] == 1
    assert "concept_a" in record
    assert "concept_b" in record
    assert "causal_error" in record
    assert len(improver.telemetry_history) == 1
