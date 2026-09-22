"""
Unit tests for the Deliberative Reasoner Engine in ISHMN.
"""

import pytest
import torch
from ishmn import UniversalCognitiveEngine, DeliberativeReasoner, ReasoningResult

@pytest.fixture
def uce_instance():
    device = torch.device("cpu")
    engine = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        max_experts=None,
        device=device
    )
    return engine

def test_deliberative_reasoner_init(uce_instance):
    assert hasattr(uce_instance, "reasoner")
    assert isinstance(uce_instance.reasoner, DeliberativeReasoner)

def test_deliberative_reasoner_imagine_branch(uce_instance):
    z_start = torch.randn(1, 32)
    z_ctx = torch.randn(1, 32)
    
    sim = uce_instance.reasoner.imagine_branch(
        z_start=z_start,
        z_context=z_ctx,
        lookahead_depth=6
    )
    assert "final_latent" in sim
    assert "causal_consistency" in sim
    assert sim["steps_survived"] <= 6
    assert len(sim["trajectory"]) > 0

def test_deliberative_reasoner_branch_pruning(uce_instance):
    z_start = torch.randn(1, 32)
    
    # Define two branches: one valid, one that terminates in dead end
    cand_1 = ("branch_viable", torch.randn(1, 32))
    cand_2 = ("branch_dead_end", torch.randn(1, 32))
    
    def mock_evaluator_factory(branch_id):
        def eval_step(step, z, u, r):
            if branch_id == "branch_dead_end" and step >= 2:
                return {"is_dead_end": True, "info": "wall collision"}
            return {"is_dead_end": False}
        return eval_step

    res = uce_instance.reason(
        premise=z_start,
        candidate_branches=[cand_1, cand_2],
        modality="continuous",
        lookahead_depth=5,
        external_branch_evaluator=mock_evaluator_factory
    )

    assert isinstance(res, ReasoningResult)
    assert res.selected_branch == "branch_viable"
    assert res.pruned_dead_ends == 1
    assert res.total_candidates == 2
    assert res.viable_candidates == 1

def test_deliberative_reasoner_subword_hypotheses(uce_instance):
    premise = "the sun radiates thermal heat and electromagnetic light"
    cand_good = ("congruent_warmth", "provides natural illumination and warms the surface")
    cand_bad = ("contradiction_freeze", "spontaneously creates freezing liquid nitrogen ice")
    
    res = uce_instance.reason(
        premise=premise,
        candidate_branches=[cand_good, cand_bad],
        modality="subwords",
        lookahead_depth=4
    )
    
    assert isinstance(res, ReasoningResult)
    assert len(res.ranked_branches) == 2
    assert res.selected_branch in ["congruent_warmth", "contradiction_freeze"]
    assert "explanation" in res.__dict__
    assert len(res.reasoning_trace) == 2
