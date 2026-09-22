"""
Tests for the Honest Evaluation Harness metric validation suite.

These tests exist because a metric was read backwards. On 2026-09-21 the harness reported
Distinct-1 0.867 / Distinct-2 1.000 for the 500M checkpoint and that was recorded as
"zero repetition collapse, 100% unique 2-grams" -- while the same checkpoint, decoded
greedily, emitted " the the the the the the the the the the the the the world."

Distinct-n is a band, not a score to maximise. The tests below pin that: the healthy band
is derived from reference corpora at matched length, and BOTH edges are failures.

They also run the contamination auditor against this repository's own files, so the
difference between the contaminated and the honest evaluation path is asserted, not
asserted-about.
"""

import textwrap
from pathlib import Path

import pytest

from ishmn.metric_validation import (
    DEGENERATE_REFERENCES,
    HEALTHY_REFERENCES,
    REFERENCE_CORPORA,
    audit_contamination,
    calibrate_distinct_n,
    diagnose_generation,
    distinct_n,
    distinct_n_per_sample,
    extract_text_from_python,
    find_contaminating_ngrams,
    healthy_distinct2_band,
    token_concentration,
    tokenize_words,
)

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Core metric behaviour
# ---------------------------------------------------------------------------

def test_distinct_n_endpoints():
    assert distinct_n(tokenize_words("a b c d e"), 1) == 1.0
    assert distinct_n(tokenize_words("a a a a a"), 1) == pytest.approx(0.2)
    assert distinct_n(tokenize_words("a a a a a"), 2) == pytest.approx(0.25)
    assert distinct_n([], 2) == 0.0
    assert distinct_n(["only"], 2) == 0.0


def test_token_concentration_detects_runs():
    flat = token_concentration(tokenize_words("the the the the the the world"))
    assert flat["top_token"] == "the"
    assert flat["top_token_share"] == pytest.approx(6 / 7)
    assert flat["longest_run"] == 6

    varied = token_concentration(tokenize_words("alpha beta gamma delta epsilon"))
    assert varied["longest_run"] == 1
    assert varied["unique_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Calibration: the anchor that makes the metric interpretable
# ---------------------------------------------------------------------------

def test_random_tokens_score_perfect_distinctness():
    """The exact value the harness reported as a success is what noise scores."""
    table = calibrate_distinct_n()
    assert table["uniform_random"]["distinct_1"] == 1.0
    assert table["uniform_random"]["distinct_2"] == 1.0


def test_fluent_text_never_reaches_perfect_distinctness():
    table = calibrate_distinct_n()
    for name in HEALTHY_REFERENCES:
        d2 = table[name]["distinct_2"]
        assert 0.5 < d2 < 1.0, f"{name} scored Distinct-2 {d2:.3f}"


def test_healthy_band_excludes_both_failure_modes():
    low, high = healthy_distinct2_band()
    table = calibrate_distinct_n()

    for name in HEALTHY_REFERENCES:
        assert low <= table[name]["distinct_2"] <= high, f"{name} fell outside the band"

    assert table["uniform_random"]["distinct_2"] >= high, "random tokens must be excluded"
    assert table["repetition_collapse"]["distinct_2"] < low, "collapse must be excluded"


def test_band_tracks_sample_length():
    """Short samples score higher by construction, so the band must move with length."""
    short_low, short_high = healthy_distinct2_band(word_budget=12)
    long_low, long_high = healthy_distinct2_band(word_budget=50)
    assert short_low >= long_low
    assert short_high >= long_high


# ---------------------------------------------------------------------------
# Diagnosis verdicts
# ---------------------------------------------------------------------------

def test_perfect_distinctness_is_flagged_random_not_healthy():
    res = diagnose_generation([REFERENCE_CORPORA["uniform_random"]])
    assert res["verdict"] == "random_like"
    assert res["distinct_2"] == pytest.approx(1.0)
    assert "near-uniform" in res["reason"]


def test_greedy_collapse_is_flagged():
    """The literal output of native_moe_500m_peak.pt at step 235."""
    res = diagnose_generation([
        "the the the the the the the the the the the the the world.",
        "the the the the the the the the the the the the the the the the",
    ])
    assert res["verdict"] == "repetition_collapse"
    assert res["top_token"] == "the"
    assert res["longest_run"] >= 4


def test_fluent_generation_is_healthy():
    res = diagnose_generation([REFERENCE_CORPORA[name] for name in HEALTHY_REFERENCES])
    assert res["verdict"] == "healthy", res["reason"]


def test_phrase_cycling_is_caught_without_exact_runs():
    res = diagnose_generation([REFERENCE_CORPORA["phrase_cycling"]])
    assert res["verdict"] in ("low_diversity", "repetition_collapse")
    assert res["verdict"] != "healthy"


def test_degenerate_references_never_pass():
    for name in DEGENERATE_REFERENCES:
        res = diagnose_generation([REFERENCE_CORPORA[name]])
        assert res["verdict"] != "healthy", f"{name} was reported healthy"


def test_insufficient_data_is_not_a_pass():
    res = diagnose_generation(["short"])
    assert res["verdict"] == "insufficient_data"


# ---------------------------------------------------------------------------
# The cross-prompt pooling bug
# ---------------------------------------------------------------------------

def test_pooling_across_prompts_inflates_distinctness():
    """
    honest_eval_harness pools every completion before forming bigrams, which invents
    bigrams spanning unrelated prompts. Diversity belongs to one generation.
    """
    completions = ["alpha beta alpha beta", "gamma delta gamma delta", "epsilon zeta epsilon zeta"]
    per_sample = distinct_n_per_sample(completions, 2)
    pooled_tokens = [t for c in completions for t in tokenize_words(c)]
    pooled = distinct_n(pooled_tokens, 2)

    assert pooled > per_sample, "expected pooling to inflate the score"
    res = diagnose_generation(completions)
    assert res["pooling_inflation"] > 0
    assert res["distinct_2"] == pytest.approx(per_sample)


# ---------------------------------------------------------------------------
# Contamination auditing against this repository's real files
# ---------------------------------------------------------------------------

def test_find_contaminating_ngrams_reports_the_actual_phrase():
    train = "the electron is the fundamental subatomic particle that carries a negative electric charge"
    test = "which subatomic particle that carries a negative electric charge orbits the nucleus"
    grams = find_contaminating_ngrams(train, test, n=7)
    assert grams, "expected an overlap to be reported"
    assert "subatomic particle that carries a negative electric" in grams[0]

    clean = "what is the boiling point of water at sea level in degrees celsius"
    assert find_contaminating_ngrams(train, clean, n=7) == []


def test_audit_reports_structured_findings():
    train = ["the celestial body at the center of our solar system is the sun"]
    items = [
        "what celestial body at the center of our solar system provides light",
        "which gas do plants absorb during respiration at night",
    ]
    rep = audit_contamination(train, items, n=8)
    assert rep["clean"] is False
    assert rep["items_checked"] == 2
    assert rep["items_contaminated"] == 1
    assert rep["findings"][0]["overlapping_ngrams"]


def test_extract_text_from_python_drops_boilerplate():
    """
    Auditing raw source produces false positives: two unrelated modules that both contain
    `from typing import Dict, Any, List, Tuple, Optional` register an 8-gram overlap.
    Only string literals carry training data or questions.
    """
    src = textwrap.dedent(
        """
        from typing import Dict, Any, List, Tuple, Optional
        import torch
        CURRICULUM = ['the electron carries a negative electric charge in an atom']
        x = 'tiny'
        """
    )
    text = extract_text_from_python(src)
    assert "electron carries a negative electric charge" in text
    assert "typing" not in text and "torch" not in text
    assert "tiny" not in text, "short literals should be dropped"

    assert extract_text_from_python("def broken(:") == "def broken(:"


def test_raw_source_audit_would_false_positive_on_boilerplate():
    """Pins why extraction is required, not optional."""
    boilerplate = textwrap.dedent(
        """
        from typing import Dict, Any, List, Tuple, Optional
        import torch
        import torch.nn.functional as F
        """
    )
    a = boilerplate
    b = boilerplate
    assert find_contaminating_ngrams(a, b, n=8), "expected boilerplate to trip a raw-source audit"
    assert not find_contaminating_ngrams(
        extract_text_from_python(a), extract_text_from_python(b), n=8
    ), "extraction should remove the false positive"


def test_all_training_scripts_are_clean_against_held_out_bank():
    """
    The live firewall. Every training script in experiments/ is audited against the
    held-out question bank. This is the gate that must keep passing.
    """
    from ishmn.honest_eval_harness import HELD_OUT_OOD_QUESTIONS

    scripts = sorted((REPO / "experiments").glob("train_*.py"))
    if not scripts:
        pytest.skip("no training scripts present")

    corpus = [
        extract_text_from_python(p.read_text(encoding="utf-8", errors="ignore"))
        for p in scripts
    ]
    items = [q["prompt"] + " " + " ".join(q["choices"]) for q in HELD_OUT_OOD_QUESTIONS]
    rep = audit_contamination(corpus, items, n=8)
    assert rep["clean"], (
        f"{rep['items_contaminated']} held-out item(s) leak into training scripts: "
        f"{rep['findings']}"
    )


def test_auditor_detects_the_historical_push_to_100_contamination():
    """
    Regression guard with teeth.

    experiments/push_mmoe_to_100.py has since been deleted, but it fine-tuned on the
    verbatim answers to the questions in ishmn/evaluation_harness.py until
    composite_accuracy reached 100.0 -- on a model that could not form a sentence. The
    curriculum below is reproduced from that file. The auditor must catch it, otherwise
    the firewall is decorative.
    """
    historical_curriculum = (
        "In atomic physics, an atom consists of a nucleus containing positively charged "
        "protons and neutral neutrons, surrounded by orbiting electrons. An electron is the "
        "fundamental subatomic particle that carries a negative electric charge. "
        "Question: What subatomic particle carries a negative electric charge in an atom? "
        "Answer: Electron. "
        "The celestial body at the center of our solar system is The Sun. Living plants "
        "absorb carbon dioxide from the air during photosynthesis. "
        "In Python programming, the keyword to define a function is def. The statement used "
        "to iterate over a list is for. In binary search, input elements must be sorted."
    )

    battery_path = REPO / "ishmn" / "evaluation_harness.py"
    if not battery_path.exists():
        pytest.skip("legacy battery removed")

    battery_text = extract_text_from_python(
        battery_path.read_text(encoding="utf-8", errors="ignore")
    )
    rep = audit_contamination([historical_curriculum], [battery_text], n=8)
    assert not rep["clean"], "auditor failed to detect known answer-key contamination"
    grams = rep["findings"][0]["overlapping_ngrams"]
    assert any("subatomic particle" in g for g in grams), grams


def test_held_out_questions_are_mutually_distinct():
    """A battery that repeats itself inflates accuracy through correlated items."""
    from ishmn.honest_eval_harness import HELD_OUT_OOD_QUESTIONS

    prompts = [q["prompt"] for q in HELD_OUT_OOD_QUESTIONS]
    assert len(set(prompts)) == len(prompts), "duplicate prompts in held-out bank"

    targets = [q["target"] for q in HELD_OUT_OOD_QUESTIONS]
    for q in HELD_OUT_OOD_QUESTIONS:
        assert q["target"] in q["choices"], f"target missing from choices: {q['prompt'][:50]}"
    assert len(targets) >= 10, "held-out bank too small to resolve anything"


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------

def test_harness_exposes_contamination_gate():
    from ishmn.honest_eval_harness import audit_training_scripts_against_held_out_bank

    rep = audit_training_scripts_against_held_out_bank()
    assert "clean" in rep
    if "skipped" not in rep:
        assert rep["scripts_audited"], "no training scripts were audited"
        assert rep["clean"], f"held-out bank leaks into training: {rep['findings']}"


def test_harness_fluency_report_carries_a_verdict():
    """The report must state what the numbers mean, not just emit them."""
    import inspect
    from ishmn import honest_eval_harness as H

    src = inspect.getsource(H.evaluate_generative_fluency_sample)
    for key in ('"verdict"', '"distinct_2_per_sample"', '"greedy"', '"pooling_inflation"'):
        assert key in src, f"fluency report no longer exposes {key}"

    suite_src = inspect.getsource(H.run_honest_evaluation_suite)
    assert "contamination_audit" in suite_src, "suite no longer runs the contamination gate"


# ---------------------------------------------------------------------------
# Length dead-zone: Distinct-n cannot decide at the harness's default length
# ---------------------------------------------------------------------------

def test_distinct_n_is_not_discriminative_at_short_lengths():
    """
    evaluate_generative_fluency_sample generates 25 tokens by default. At that length
    fluent references also score Distinct-2 ~ 1.0, so the metric carries no signal.
    """
    from ishmn.metric_validation import distinct_n_is_discriminative

    assert not distinct_n_is_discriminative(25)
    assert distinct_n_is_discriminative(60)


def test_short_fluent_sample_is_indeterminate_not_healthy():
    fluent_short = "a list in python is mutable which means its contents can be changed after creation"
    res = diagnose_generation([fluent_short], word_budget=25)
    assert res["verdict"] == "indeterminate_short_sample"
    assert res["distinct_n_discriminative"] is False
    assert "60 words" in res["reason"]


def test_collapse_still_detected_inside_the_dead_zone():
    """Length-robust detectors must keep working where Distinct-n cannot."""
    res = diagnose_generation(["the the the the the the the the the the the the"], word_budget=25)
    assert res["verdict"] == "repetition_collapse"


def test_long_samples_recover_discrimination():
    res = diagnose_generation([REFERENCE_CORPORA["uniform_random"]], word_budget=60)
    assert res["verdict"] == "random_like"
