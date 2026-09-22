"""
Unit and Integration Tests for Hugging Face Curriculum Streaming Pipeline.
Verifies:
1. SovereignTokenizer (tiktoken GPT-2 and Character ASCII) encoding/decoding.
2. Schema extractors across FineWeb-Edu, GSM8K, MathInstruct, Python Codes, UltraChat, and Dolly.
3. Resilient offline fallback stream behavior.
4. HuggingFaceCurriculumStreamer multi-stage batch sampling and sequence packing.
"""

import pytest
import torch

from ishmn.data_loader import (
    CharacterTokenizer,
    SovereignTokenizer,
    HFDatasetStream,
    HuggingFaceCurriculumStreamer,
    CURATED_FALLBACK_CORPUS,
    extract_fineweb_edu,
    extract_gsm8k,
    extract_math_instruct,
    extract_python_codes,
    extract_python_alpaca,
    extract_ultrachat,
    extract_dolly
)


def test_character_tokenizer():
    tok = CharacterTokenizer()
    assert tok.vocab_size == 128
    text = "Hello, Spherical World!"
    encoded = tok.encode(text)
    assert len(encoded) == len(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_sovereign_tokenizer_gpt2():
    tok = SovereignTokenizer(encoding_name="gpt2")
    assert tok.vocab_size == 50257
    assert tok.is_subword is True

    text = "Universal Riemannian Attention on S^7 x R+."
    encoded = tok.encode(text)
    assert isinstance(encoded, list)
    assert len(encoded) > 0
    decoded = tok.decode(encoded)
    assert decoded == text


def test_schema_extractors():
    # 1. FineWeb-Edu
    s1 = {"text": "Educational article explaining mitochondrial ATP synthesis."}
    assert extract_fineweb_edu(s1) == s1["text"]

    # 2. GSM8K
    s2 = {"question": "What is 15 * 4?", "answer": "15 * 4 = 60"}
    res2 = extract_gsm8k(s2)
    assert "Question: What is 15 * 4?" in res2
    assert "Solution: 15 * 4 = 60" in res2

    # 3. MathInstruct
    s3 = {"instruction": "Calculate derivative of x^2.", "output": "2x"}
    res3 = extract_math_instruct(s3)
    assert "Instruction: Calculate derivative" in res3
    assert "Output: 2x" in res3

    # 4. Python Codes
    s4 = {"instruction": "Write binary search", "output": "def binary_search(): pass"}
    res4 = extract_python_codes(s4)
    assert "Instruction: Write binary search" in res4
    assert "Code:\ndef binary_search(): pass" in res4

    # 5. Python Alpaca with input
    s5 = {"instruction": "Sort list", "input": "[3, 1, 2]", "output": "[1, 2, 3]"}
    res5 = extract_python_alpaca(s5)
    assert "Input: [3, 1, 2]" in res5
    assert "Code:\n[1, 2, 3]" in res5

    # 6. UltraChat
    s6 = {
        "messages": [
            {"role": "user", "content": "Explain relativity."},
            {"role": "assistant", "content": "Spacetime curves around mass."}
        ]
    }
    res6 = extract_ultrachat(s6)
    assert "User: Explain relativity." in res6
    assert "Assistant: Spacetime curves around mass." in res6

    # 7. Dolly
    s7 = {
        "instruction": "Summarize photosynthesis",
        "context": "Light energy converts CO2 to sugar.",
        "response": "Photosynthesis produces sugar using light."
    }
    res7 = extract_dolly(s7)
    assert "Instruction: Summarize photosynthesis" in res7
    assert "Context: Light energy converts CO2 to sugar." in res7
    assert "Response: Photosynthesis produces sugar" in res7


def test_hf_dataset_stream_fallback():
    # Stream in offline fallback mode
    stream = HFDatasetStream(
        dataset_name="nonexistent/fake_dataset",
        fallback_samples=CURATED_FALLBACK_CORPUS["general_knowledge"]
    )
    sample = stream.get_next_sample()
    assert isinstance(sample, str)
    assert len(sample) > 20
    assert sample in CURATED_FALLBACK_CORPUS["general_knowledge"]


def test_curriculum_streamer_offline():
    tok = SovereignTokenizer("gpt2")
    streamer = HuggingFaceCurriculumStreamer(
        tokenizer=tok,
        batch_size=3,
        seq_len=24,
        weights=(0.50, 0.25, 0.25),
        prefetch_size=4,
        offline_mode=True
    )
    try:
        inputs, targets = streamer.sample_batch()
        assert inputs.shape == (3, 24)
        assert targets.shape == (3, 24)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long

        # Verify token IDs are within valid vocabulary
        assert (inputs >= 0).all() and (inputs < tok.vocab_size).all()
        assert (targets >= 0).all() and (targets < tok.vocab_size).all()
    finally:
        streamer.close()


def test_curriculum_streamer_live_sampling():
    tok = SovereignTokenizer("gpt2")
    streamer = HuggingFaceCurriculumStreamer(
        tokenizer=tok,
        batch_size=2,
        seq_len=32,
        weights=(0.50, 0.25, 0.25),
        prefetch_size=4,
        offline_mode=False
    )
    try:
        inputs, targets = streamer.sample_batch()
        assert inputs.shape == (2, 32)
        assert targets.shape == (2, 32)
        decoded_sample = tok.decode(inputs[0])
        assert isinstance(decoded_sample, str)
        assert len(decoded_sample) > 0
    finally:
        streamer.close()
