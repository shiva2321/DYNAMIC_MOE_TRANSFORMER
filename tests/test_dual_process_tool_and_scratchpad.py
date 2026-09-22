"""
Unit Tests for Dual-Process Autonomous Agent & Domain-Agnostic Tool/Scratchpad Deliberation.

Verifies:
1. AST Safe Expression Evaluation (Arbitrary scale, zero hardcoded rules).
2. Action Dispatcher (Discrete cognitive-motor actions & episodic memory logging).
3. Dual-Process Solver (System 1 intuitive manifold + System 2 exact tool execution).
4. Continuous Living Self-Loop Tool Emission (Autonomous perception-to-action).
"""

import pytest
import torch
from ishmn import (
    UniversalCognitiveEngine,
    ContinuousSelfLoop,
    ACTION_DELIBERATE,
    ACTION_MOTOR_STEP,
    ACTION_EMIT_COMMUNICATION,
    ACTION_INVOKE_SYMBOLIC_TOOL,
    ACTION_NAMES,
    safe_eval_expression
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def uce():
    """Initializes a compact UniversalCognitiveEngine for dual-process testing."""
    engine = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        max_experts=4,
        device=DEVICE
    )
    return engine


def test_safe_eval_expression():
    """Verifies safe AST math evaluation on simple, complex, and astronomical numbers."""
    # Small numbers
    assert safe_eval_expression("12 + 7") == 19.0
    assert safe_eval_expression("18 - 6") == 12.0
    assert safe_eval_expression("24 * 2") == 48.0

    # User's test query: tens-of-thousands
    assert safe_eval_expression("23164 - 45") == 23119.0

    # User's test query: astronomical 23 trillion
    assert safe_eval_expression("23146484561521 + 45132030") == 23146529693551.0

    # Continuous decimals & parentheses
    assert safe_eval_expression("(9.5 - 4.5) * 2") == 10.0


def test_dispatch_action_symbolic_tool(uce):
    """Verifies dispatch_action executes the tool safely and writes verified truth into episodic memory."""
    initial_episodes = len(uce.memory.episodic_memory)
    res = uce.dispatch_action(
        action_id=ACTION_INVOKE_SYMBOLIC_TOOL,
        query="23164 - 45",
        log_to_episodic=True
    )

    assert res["status"] == "success"
    assert res["result"] == 23119.0
    assert res["action_name"] == "INVOKE_SYMBOLIC_TOOL"
    # Verify exact truth was logged into episodic memory
    assert len(uce.memory.episodic_memory) == initial_episodes + 1


def test_solve_with_dual_process_standard_and_astronomical(uce):
    """Verifies solve_with_dual_process produces both continuous intuition and exact discrete result."""
    # 1. Tens-of-thousands query
    res1 = uce.solve_with_dual_process("23164 - 45", enable_system2=True)
    assert res1["query"] == "23164 - 45"
    assert res1["system2_exact_result"] == 23119.0
    assert res1["action_selected"] == ACTION_INVOKE_SYMBOLIC_TOOL
    assert res1["action_name"] == "INVOKE_SYMBOLIC_TOOL"
    assert res1["causal_consistency"] > 0.0

    # 2. Astronomical 23 Trillion query
    res2 = uce.solve_with_dual_process("23146484561521 + 45132030", enable_system2=True)
    assert res2["system2_exact_result"] == 23146529693551.0
    assert res2["action_selected"] == ACTION_INVOKE_SYMBOLIC_TOOL
    assert "u" in res2["manifold_coords"]


def test_continuous_self_loop_tool_emission(uce):
    """Verifies the living ContinuousSelfLoop triggers action dispatch on mathematical input perturbations."""
    loop = ContinuousSelfLoop(uce=uce, emission_threshold=0.10)
    
    # Inject input perturbation tagged with the equation
    dummy_input = torch.randn(1, 4, device=DEVICE)
    loop.inject_input(dummy_input, modality="continuous", tag="23164 - 45")
    
    # Run loop ticks until an emission triggers or max ticks reached
    triggered_emissions = []
    for _ in range(8):
        em = loop.loop_tick()
        if em is not None:
            triggered_emissions.append(em)
            break

    assert len(triggered_emissions) > 0, "Self-loop failed to trigger emission on perturbation"
    em = triggered_emissions[0]
    assert em.action_type == ACTION_INVOKE_SYMBOLIC_TOOL
    assert em.action_result == 23119.0
