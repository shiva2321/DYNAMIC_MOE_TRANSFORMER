import pytest
import torch
from ishmn.evaluation_harness import (
    answers_match,
    evaluate_mathematical_reasoning,
    evaluate_causal_sensemaking,
    evaluate_algorithmic_code,
    evaluate_curriculum_validation,
    evaluate_mmoe_comprehensive
)
from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import CharacterTokenizer


def test_answers_match():
    assert answers_match(60.0, 60.0) is True
    assert answers_match("60", 60.0) is True
    assert answers_match(" 60.0001 ", 60.0) is True
    assert answers_match("125", 125.0) is True
    assert answers_match("invalid", 60.0) is False
    assert answers_match(None, 60.0) is False


def test_ngram_contamination_guard():
    from ishmn.honest_eval_harness import check_ngram_contamination, assert_zero_contamination

    corpus = "The quick brown fox jumps over the lazy dog in the middle of summer."
    clean_test = "A fast auburn canine leaps across an inactive hound."
    contaminated_test = "The quick brown fox jumps over the lazy dog and sleeps."

    assert check_ngram_contamination(corpus, clean_test, n=8) is False
    assert check_ngram_contamination(corpus, contaminated_test, n=8) is True

    # assert_zero_contamination should pass for clean test
    assert_zero_contamination(corpus, [clean_test], n=8)

    # assert_zero_contamination must raise AssertionError for contaminated test
    with pytest.raises(AssertionError):
        assert_zero_contamination(corpus, [contaminated_test], n=8)


def test_mmoe_evaluation_harness_integration():
    device = torch.device("cpu")
    tokenizer = CharacterTokenizer()

    # Small test model
    model = NativeSphericalTransformer(
        vocab_size=128,
        d_model=32,
        num_layers=2,
        num_heads=2,
        hyperspace_dim=8,
        d_ff=64,
        max_seq_len=128,
        num_experts=2,
        top_k=1,
        enable_vsa_memory=True,
        enable_world_model=True,
        device=device
    )

    uce = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        vocab_size=128,
        device=device
    )
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    # 1. Test Math Reasoning
    res_math = evaluate_mathematical_reasoning(uce, device=device)
    assert "accuracy" in res_math
    assert res_math["accuracy"] >= 80.0  # System 2 scratchpad delivers high precision
    assert res_math["total"] == 10

    # 2. Test Causal Sensemaking
    res_science = evaluate_causal_sensemaking(uce)
    assert "accuracy" in res_science
    assert res_science["total"] == 8

    # 3. Test Algorithmic Code
    res_code = evaluate_algorithmic_code(model, uce, tokenizer, device=device)
    assert "accuracy" in res_code
    assert res_code["total"] == 8

    # 4. Test Curriculum Validation with dummy batches
    dummy_in = torch.randint(0, 128, (2, 16))
    dummy_tgt = torch.randint(0, 128, (2, 16))
    val_batches = [(dummy_in, dummy_tgt)]
    res_curriculum = evaluate_curriculum_validation(model, uce, tokenizer, val_batches, device=device)
    assert "accuracy" in res_curriculum
    assert "perplexity" in res_curriculum

    # 5. Test Comprehensive Master Eval
    comp = evaluate_mmoe_comprehensive(
        model=model,
        uce=uce,
        tokenizer=tokenizer,
        validation_batches=val_batches,
        device=device
    )
    assert "composite_accuracy" in comp
    assert "domains" in comp
    assert 0.0 <= comp["composite_accuracy"] <= 100.0
