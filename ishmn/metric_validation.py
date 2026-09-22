"""
Metric Validation Suite for the Honest Evaluation Harness.

A metric that has never been calibrated against a known input is not an instrument,
it is a number generator. This module supplies the calibration.

Motivating failure (measured 2026-09-21 on artifacts/checkpoints/native_moe_500m_peak.pt,
step 235, val PPL 1189): the harness reported Distinct-1 0.867 / Distinct-2 1.000, which was
read as "zero repetition collapse, 100% unique bigrams". Greedy decoding from that same
checkpoint produced:

    "The mitochondria is the" -> " the the the the the the the the the the the the the world."

Both statements are true at once. The harness samples at temperature 0.7 from a near-uniform
distribution over a 50k vocabulary, so the tokens it scores are effectively random draws --
and uniform random tokens score Distinct-2 = 1.000 exactly. The metric was measuring the
sampler, not the model. Reference calibration makes that inversion impossible to miss:

    natural English prose   Distinct-2 ~ 0.93
    worked math solution    Distinct-2 ~ 0.89
    uniform random tokens   Distinct-2 = 1.00      <- the reported value
    "the the the the ..."   Distinct-2 ~ 0.00

Distinct-n is therefore NOT monotonically "higher is better". It is a band: too low means
repetition collapse, too high means the output is indistinguishable from noise. This module
exposes that band, plus two length-robust collapse detectors (token concentration and longest
run) that do not depend on sample length the way Distinct-n does.

It also provides contamination auditing that reports WHICH n-grams overlap, rather than a
bare boolean, so a failure is actionable.
"""

import ast
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "extract_text_from_python",
    "tokenize_words",
    "distinct_n",
    "distinct_n_per_sample",
    "token_concentration",
    "REFERENCE_CORPORA",
    "calibrate_distinct_n",
    "distinct_n_is_discriminative",
    "healthy_distinct2_band",
    "diagnose_generation",
    "find_contaminating_ngrams",
    "audit_contamination",
]


# ---------------------------------------------------------------------------
# Tokenisation and core metrics
# ---------------------------------------------------------------------------

def tokenize_words(text: str) -> List[str]:
    """Lowercased word tokens. Matches honest_eval_harness.check_ngram_contamination."""
    return re.findall(r"\b\w+\b", text.lower())


def distinct_n(tokens: Sequence[str], n: int = 2) -> float:
    """Ratio of unique n-grams to total n-grams. Returns 0.0 when undefined."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def distinct_n_per_sample(completions: Iterable[str], n: int = 2) -> float:
    """
    Mean Distinct-n computed WITHIN each completion, then averaged.

    The harness pools every completion into one token list before forming n-grams, which
    manufactures spurious n-grams spanning the boundary between unrelated prompts and
    inflates the score. Diversity is a property of a single generation, so it must be
    measured inside one and only then averaged.
    """
    scores = []
    for text in completions:
        toks = tokenize_words(text)
        if len(toks) >= n:
            scores.append(distinct_n(toks, n))
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def token_concentration(tokens: Sequence[str]) -> Dict[str, Any]:
    """
    Length-robust degeneracy signals.

    Distinct-n rises mechanically as samples get shorter, so it cannot be compared across
    differing lengths without care. These two do not have that problem:
      top_token_share - fraction of output taken by the single most frequent token
      longest_run     - longest consecutive repeat of one token
    """
    if not tokens:
        return {"top_token_share": 0.0, "longest_run": 0, "unique_ratio": 0.0, "top_token": None}

    counts: Dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    top_token = max(counts, key=counts.get)

    longest = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        longest = max(longest, run)

    return {
        "top_token_share": counts[top_token] / len(tokens),
        "longest_run": longest,
        "unique_ratio": len(counts) / len(tokens),
        "top_token": top_token,
    }


# ---------------------------------------------------------------------------
# Reference corpora: the calibration anchors
# ---------------------------------------------------------------------------

REFERENCE_CORPORA: Dict[str, str] = {
    "natural_prose": (
        "the cell is the basic structural and functional unit of all known organisms it is "
        "the smallest unit of life that can replicate independently and cells are often "
        "called the building blocks of life the study of cells is called cell biology or "
        "cellular biology and it examines how cells divide grow and respond to their "
        "environment over the course of a lifetime"
    ),
    "worked_solution": (
        "janet sells the remainder at the farmers market she has sixteen eggs she eats three "
        "for breakfast and uses four for muffins so she has sixteen minus three minus four "
        "which is nine eggs she sells nine eggs at two dollars each so she makes eighteen "
        "dollars every day at the market"
    ),
    "technical_code_prose": (
        "a list in python is mutable which means its contents can be changed after creation "
        "whereas a tuple is immutable and cannot be modified once it has been defined this "
        "makes tuples hashable so they can serve as dictionary keys while lists cannot be "
        "used that way at all"
    ),
    "repetition_collapse": " ".join(["the"] * 60),
    "phrase_cycling": " ".join(["the model is the model is the model is"] * 8),
    "uniform_random": " ".join(f"tok{i}" for i in range(60)),
}

#: Corpora that represent acceptable, fluent output.
HEALTHY_REFERENCES = ("natural_prose", "worked_solution", "technical_code_prose")

#: Corpora that represent failure modes a healthy band must exclude.
DEGENERATE_REFERENCES = ("repetition_collapse", "phrase_cycling", "uniform_random")


def calibrate_distinct_n(word_budget: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Scores every reference corpus, optionally truncated to `word_budget` words.

    Distinct-n depends on length, so a generation of 25 words must be compared against
    references truncated to 25 words. Passing the budget makes the comparison honest.
    """
    table: Dict[str, Dict[str, float]] = {}
    for name, text in REFERENCE_CORPORA.items():
        toks = tokenize_words(text)
        if word_budget is not None:
            toks = toks[:word_budget]
        conc = token_concentration(toks)
        table[name] = {
            "distinct_1": distinct_n(toks, 1),
            "distinct_2": distinct_n(toks, 2),
            "top_token_share": conc["top_token_share"],
            "longest_run": float(conc["longest_run"]),
            "words": float(len(toks)),
        }
    return table


def distinct_n_is_discriminative(word_budget: Optional[int], n: int = 2) -> bool:
    """
    Whether Distinct-n can separate fluent text from noise at this sample length.

    It usually cannot at short lengths. Twenty-five words of ordinary English rarely repeats
    a bigram, so fluent prose and uniform random tokens both score ~1.000 and the metric
    carries no signal. `evaluate_generative_fluency_sample` generates 25 tokens by default,
    which sits inside that dead zone.

    Discriminative power is defined as the healthy upper edge staying clear of the random
    reference: if fluent text already saturates at 1.0, there is nothing left to detect.
    """
    table = calibrate_distinct_n(word_budget)
    healthy_max = max(table[k][f"distinct_{n}"] for k in HEALTHY_REFERENCES)
    return healthy_max < 0.99


def healthy_distinct2_band(word_budget: Optional[int] = None) -> Tuple[float, float]:
    """
    Empirical [low, high] Distinct-2 band for fluent text at the given length.

    The band is derived from the healthy references rather than hand-set, so it tracks the
    length correction automatically. The upper edge sits strictly below the random-token
    reference, which is the whole point: 1.000 must fall OUTSIDE the healthy band.
    """
    table = calibrate_distinct_n(word_budget)
    healthy = [table[k]["distinct_2"] for k in HEALTHY_REFERENCES]
    low = min(healthy) * 0.75          # tolerance below the least diverse fluent reference
    high = min(1.0, max(healthy) + (1.0 - max(healthy)) * 0.5)  # midpoint toward random
    return low, high


def diagnose_generation(
    completions: Sequence[str],
    word_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Classifies a batch of generated completions against the calibrated references.

    Verdicts:
      insufficient_data          - too few words to judge
      repetition_collapse        - one token dominates, or a long consecutive run
      indeterminate_short_sample - no collapse, but samples are too short for Distinct-n
                                   to separate fluency from noise
      low_diversity        - Distinct-2 below the healthy band (phrase cycling)
      random_like          - Distinct-2 at or above the random-token reference
      healthy              - inside the calibrated band

    `random_like` is NOT a pass. It is the signature of sampling from a near-uniform
    distribution, which is what an untrained model does.
    """
    texts = [c for c in completions if c and c.strip()]
    pooled = [t for text in texts for t in tokenize_words(text)]

    if len(pooled) < 8:
        return {
            "verdict": "insufficient_data",
            "reason": f"only {len(pooled)} words generated",
            "words": len(pooled),
        }

    budget = word_budget
    if budget is None:
        per_sample_lengths = [len(tokenize_words(t)) for t in texts]
        budget = max(per_sample_lengths) if per_sample_lengths else None

    d1 = distinct_n_per_sample(texts, 1)
    d2 = distinct_n_per_sample(texts, 2)
    d2_pooled = distinct_n(pooled, 2)
    conc = token_concentration(pooled)
    low, high = healthy_distinct2_band(budget)

    discriminative = distinct_n_is_discriminative(budget)

    if conc["longest_run"] >= 4 or conc["top_token_share"] >= 0.30:
        verdict = "repetition_collapse"
        reason = (
            f"token {conc['top_token']!r} is {conc['top_token_share']:.1%} of output, "
            f"longest consecutive run {conc['longest_run']}"
        )
    elif not discriminative:
        # Collapse detectors above are length-robust and already cleared; Distinct-n is
        # saturated at this length, so no fluency claim can be supported either way.
        verdict = "indeterminate_short_sample"
        reason = (
            f"samples are ~{budget} words; fluent references also score Distinct-2 ~1.0 at "
            f"this length, so the metric cannot separate fluency from noise. No repetition "
            f"collapse detected, but fluency is unproven. Generate >= 60 words to decide."
        )
    elif d2 >= high:
        verdict = "random_like"
        reason = (
            f"Distinct-2 {d2:.3f} >= {high:.3f}; fluent text at this length scores "
            f"{low:.3f}-{high:.3f}. Values at the random-token reference indicate sampling "
            f"from a near-uniform distribution, not fluency."
        )
    elif d2 < low:
        verdict = "low_diversity"
        reason = f"Distinct-2 {d2:.3f} < {low:.3f}; repeated phrases without exact token runs"
    else:
        verdict = "healthy"
        reason = f"Distinct-2 {d2:.3f} inside calibrated band [{low:.3f}, {high:.3f}]"

    return {
        "verdict": verdict,
        "reason": reason,
        "distinct_1": d1,
        "distinct_2": d2,
        "distinct_2_pooled": d2_pooled,
        "pooling_inflation": d2_pooled - d2,
        "healthy_band": (low, high),
        "distinct_n_discriminative": discriminative,
        "words": len(pooled),
        **{k: v for k, v in conc.items() if k != "top_token"},
        "top_token": conc["top_token"],
    }


# ---------------------------------------------------------------------------
# Contamination auditing
# ---------------------------------------------------------------------------

def extract_text_from_python(source: str, min_words: int = 4) -> str:
    """
    Pulls only the natural-language string literals out of Python source.

    Auditing raw source files produces false positives from shared boilerplate: two
    unrelated files both containing `from typing import Dict, Any, List, Tuple, Optional`
    register an 8-gram overlap that has nothing to do with test leakage. Training data and
    question banks live in string literals, so that is what must be compared.

    Falls back to the raw source if it will not parse, and keeps only literals with at
    least `min_words` words so identifiers and format fragments are excluded.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    chunks: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value.split()) >= min_words:
                chunks.append(node.value)
    return "\n".join(chunks)


def find_contaminating_ngrams(
    training_text: str,
    test_text: str,
    n: int = 8,
    limit: int = 10,
) -> List[str]:
    """
    Returns the actual overlapping n-grams, not merely whether any exist.

    `check_ngram_contamination` answers yes/no, which tells you a suite is compromised but
    not which item or which phrase. A leak is only fixable once you can see it.
    """
    train_tokens = tokenize_words(training_text)
    test_tokens = tokenize_words(test_text)
    if len(test_tokens) < n or len(train_tokens) < n:
        return []

    train_ngrams = {
        tuple(train_tokens[i:i + n]) for i in range(len(train_tokens) - n + 1)
    }
    hits: List[str] = []
    seen = set()
    for j in range(len(test_tokens) - n + 1):
        gram = tuple(test_tokens[j:j + n])
        if gram in train_ngrams and gram not in seen:
            seen.add(gram)
            hits.append(" ".join(gram))
            if len(hits) >= limit:
                break
    return hits


def audit_contamination(
    training_texts: Sequence[str],
    eval_items: Sequence[str],
    n: int = 8,
) -> Dict[str, Any]:
    """
    Cross-checks an evaluation battery against every training corpus it may have seen.

    Returns a structured report rather than raising, so a caller can log the full picture
    before deciding to fail. `clean` is the single field a gate should read.
    """
    corpus = "\n".join(training_texts)
    findings = []
    for item in eval_items:
        grams = find_contaminating_ngrams(corpus, item, n=n)
        if grams:
            findings.append({
                "eval_item": item[:120],
                "overlapping_ngrams": grams,
                "num_overlaps": len(grams),
            })
    return {
        "clean": len(findings) == 0,
        "n": n,
        "items_checked": len(eval_items),
        "items_contaminated": len(findings),
        "findings": findings,
    }
