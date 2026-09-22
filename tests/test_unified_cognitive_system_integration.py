"""
Unit and Integration Tests: Unified Cognitive System Integration.
Verifies deep coupling of all ISHMN cognitive subsystems with NativeSphericalTransformer:
1. Active VSA Episodic Memory soft-attention reads & surprise-adaptive Hebbian writes.
2. Universal World Model forward causal trajectory loss computation.
3. Neuro-Symbolic Scratchpad arithmetic interception during autoregressive generation.
4. UniversalCognitiveEngine think_and_speak closed-loop cognitive cycle.
"""

import pytest
import torch
import torch.nn.functional as F

from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.data_loader import CharacterTokenizer, SovereignTokenizer
from ishmn.neurosymbolic_scratchpad import NeuroSymbolicScratchpadLogitsProcessor


@pytest.fixture
def char_tokenizer():
    return CharacterTokenizer()


@pytest.fixture
def uce():
    return UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        vocab_size=128
    )


@pytest.fixture
def native_transformer(char_tokenizer):
    return NativeSphericalTransformer(
        vocab_size=char_tokenizer.vocab_size,
        d_model=32,
        num_layers=2,
        num_heads=2,
        hyperspace_dim=8,
        d_ff=64,
        max_seq_len=64,
        enable_vsa_memory=True,
        enable_world_model=True
    )


def test_native_transformer_vsa_memory_coupling(native_transformer, char_tokenizer):
    """Verifies that tokens write to and read from VSA episodic memory."""
    x = torch.randint(0, char_tokenizer.vocab_size, (2, 16))

    # Initial forward: memory is empty
    assert not native_transformer.vsa_memory.occupied.any()
    out1 = native_transformer(x, targets=x)
    assert "loss" in out1
    assert "cone_norm" in out1

    # Manually store an episodic key-value pair to simulate high-surprise event
    u_test = torch.randn(1, 8)
    u_test = F.normalize(u_test, p=2, dim=-1)
    r_test = torch.tensor([[1.5]])
    v_test = torch.randn(1, 32)
    native_transformer.vsa_memory.write(u_test, r_test, v_test)
    assert native_transformer.vsa_memory.occupied.any()

    # Second forward: memory read is active
    out2 = native_transformer(x, targets=x)
    assert out2["loss"] is not None

    # Autoregressive generation with write_to_vsa=True
    gen = native_transformer.generate(x[:1, :4], max_new_tokens=6, write_to_vsa=True)
    assert gen.size(1) == 10
    assert native_transformer.vsa_memory.occupied.sum().item() >= 1


def test_native_transformer_world_model_causal_loss(native_transformer, char_tokenizer):
    """Verifies that causal world model computes transition loss across the token trajectory."""
    x = torch.randint(0, char_tokenizer.vocab_size, (2, 20))
    out = native_transformer(x, targets=x)

    assert "world_model_causal" in out["loss_components"]
    assert out["loss_components"]["world_model_causal"] > 0.0
    assert out["loss"] is not None


def test_uce_attachment_and_think_and_speak(uce, native_transformer, char_tokenizer):
    """Verifies UCE attaching the native transformer and executing the closed-loop think_and_speak cycle."""
    uce.attach_native_transformer(native_transformer, tokenizer=char_tokenizer)
    assert hasattr(uce, "native_transformer")
    assert uce.native_transformer is native_transformer

    # Execute closed-loop cognitive processing
    prompt = "What is 12 + 15?"
    result = uce.think_and_speak(
        prompt=prompt,
        max_new_tokens=8,
        enable_born_rule=True,
        enable_scratchpad=True,
        enable_deliberation=True,
        write_to_vsa=True
    )

    assert "response" in result
    assert "elapsed_sec" in result
    assert "hopfield_energy" in result
    assert "dual_process" in result
    assert result["tokens_generated"] == 8

    # System 2 should have intercepted arithmetic
    dp = result["dual_process"]
    assert dp is not None
    assert dp["system2_exact_result"] == 27.0
    assert dp["action_name"] == "INVOKE_SYMBOLIC_TOOL"


def test_scratchpad_arithmetic_interceptor(uce, char_tokenizer):
    """Verifies that the scratchpad processor boosts logits for exact math solutions."""
    scratchpad = NeuroSymbolicScratchpadLogitsProcessor(uce=uce, tokenizer=char_tokenizer, boost_weight=8.0)

    # Simulate sequence ending in "8 * 5 = "
    prompt = "8 * 5 = "
    input_ids = torch.tensor([[ord(c) for c in prompt]], dtype=torch.long)
    scores = torch.zeros(1, 128)

    modified_scores = scratchpad(input_ids, scores)
    # The token for '4' (first digit of 40) or ' ' should be boosted
    boosted = (modified_scores > 0).any()
    assert boosted.item() is True
