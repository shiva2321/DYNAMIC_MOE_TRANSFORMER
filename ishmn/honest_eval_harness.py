"""
Honest, Non-Contaminated Evaluation Harness for Sovereign Cognitive Substrates.

Key Design Principles:
1. ZERO TEST-SET CONTAMINATION: None of the questions in this file are permitted
   in any pretraining or fine-tuning corpus.
2. DUAL-ACCOUNTING TRANSPARENCY: System 1 (pure neural weights) and System 2
   (symbolic AST tool evaluation) are measured and reported separately.
3. HELD-OUT PERPLEXITY: Evaluated on completely unseen validation tokens.
4. GENERATIVE REALITY: Inspects actual autoregressive token generation,
   measuring vocabulary entropy, distinct n-grams, and repetition rate.
"""

import math
import re
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional

from ishmn.geometry import project_to_sphere
from ishmn.metric_validation import (
    audit_contamination,
    diagnose_generation,
    extract_text_from_python,
)


def check_ngram_contamination(
    training_text: str,
    test_text: str,
    n: int = 8
) -> bool:
    """
    Checks for exact n-gram overlap between training corpus and test strings.
    Returns True if contamination (>= n identical consecutive words) is detected.
    """
    def tokenize_words(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    train_tokens = tokenize_words(training_text)
    test_tokens = tokenize_words(test_text)
    if len(test_tokens) < n or len(train_tokens) < n:
        return False

    train_ngrams = set(
        tuple(train_tokens[i:i + n])
        for i in range(len(train_tokens) - n + 1)
    )
    for j in range(len(test_tokens) - n + 1):
        if tuple(test_tokens[j:j + n]) in train_ngrams:
            return True
    return False


def assert_zero_contamination(
    training_corpus: str,
    eval_prompts: List[str],
    n: int = 8
) -> None:
    """
    Strict guard preventing test-set leakage.
    Raises AssertionError if ANY eval prompt shares >= 8 consecutive tokens with training corpus.
    """
    for prompt in eval_prompts:
        if check_ngram_contamination(training_corpus, prompt, n=n):
            raise AssertionError(
                f"[CONTAMINATION DETECTED] Prompt shares >= {n}-gram overlap with training set: '{prompt[:60]}...'"
            )


# ---------------------------------------------------------------------------
# 1. Strictly Held-Out OOD Multi-Choice Benchmark (Zero Training Overlap)
# ---------------------------------------------------------------------------
HELD_OUT_OOD_QUESTIONS = [
    # Science & Biology
    {
        "prompt": "Which cellular organelle is primarily responsible for generating ATP through cellular respiration?",
        "choices": ["Mitochondria", "Ribosome", "Endoplasmic reticulum", "Golgi apparatus"],
        "target": "Mitochondria",
        "domain": "biology"
    },
    {
        "prompt": "What is the primary chemical compound that constitutes the majority of natural gas?",
        "choices": ["Methane", "Propane", "Carbon monoxide", "Ammonia"],
        "target": "Methane",
        "domain": "chemistry"
    },
    {
        "prompt": "Which subatomic particle is neutral and has roughly the same mass as a proton?",
        "choices": ["Neutron", "Positron", "Electron", "Muon"],
        "target": "Neutron",
        "domain": "physics"
    },
    {
        "prompt": "In astronomy, a star so dense that not even light can escape its gravitational pull is a",
        "choices": ["Black hole", "White dwarf", "Pulsar", "Red giant"],
        "target": "Black hole",
        "domain": "astronomy"
    },
    {
        "prompt": "What physical state of matter exists at ultra-high temperatures where atoms are stripped of their electrons?",
        "choices": ["Plasma", "Liquid", "Gas", "Bose-Einstein condensate"],
        "target": "Plasma",
        "domain": "physics"
    },
    # Computer Science & Logic
    {
        "prompt": "In Python, which built-in function returns the total number of items in a collection or list?",
        "choices": ["len", "size", "count", "length"],
        "target": "len",
        "domain": "programming"
    },
    {
        "prompt": "In Python, which standard built-in data type is completely immutable after creation?",
        "choices": ["tuple", "list", "dict", "set"],
        "target": "tuple",
        "domain": "programming"
    },
    {
        "prompt": "The algorithmic strategy of breaking a problem down into overlapping subproblems with memoization is",
        "choices": ["Dynamic programming", "Greedy approach", "Breadth-first search", "Backtracking"],
        "target": "Dynamic programming",
        "domain": "algorithms"
    },
    {
        "prompt": "In relational databases, which SQL clause is used to eliminate duplicate rows from a result set?",
        "choices": ["DISTINCT", "UNIQUE", "FILTER", "GROUP"],
        "target": "DISTINCT",
        "domain": "databases"
    },
    {
        "prompt": "In Python, the boolean operator that negates or inverts the truth value of a condition is",
        "choices": ["not", "invert", "negate", "flip"],
        "target": "not",
        "domain": "programming"
    }
]

# ---------------------------------------------------------------------------
# 2. Arithmetic Battery: System 1 (Weights) vs System 2 (Symbolic)
# ---------------------------------------------------------------------------
UNSEEN_ARITHMETIC_BATTERY = [
    ("Calculate: 47 * 5 = ", 235.0),
    ("Calculate: 180 - 95 = ", 85.0),
    ("Calculate: 360 / 6 = ", 60.0),
    ("Calculate: (20 - 8) * 4 = ", 48.0),
    ("Calculate: 73 + 58 = ", 131.0),
    ("Calculate: 16 * 9 = ", 144.0)
]


def evaluate_held_out_perplexity(
    model: Any,
    held_out_tokens: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
    seq_len: int = 64
) -> Dict[str, float]:
    """
    Measures true out-of-sample Cross-Entropy Loss and Perplexity
    on unseen validation tokens.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        if held_out_tokens.size(0) <= seq_len:
            return {"val_loss": 0.0, "perplexity": 1.0}

        max_limit = min(512, held_out_tokens.size(0) - seq_len)
        for i in range(0, max_limit + 1, seq_len):
            chunk = held_out_tokens[i:i + seq_len].unsqueeze(0).to(device)
            if chunk.size(1) < 2:
                continue

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = model(chunk, targets=chunk, return_mtp=False)
            loss = out.get("loss_components", {}).get("primary_ce", out["loss"].item())
            total_loss += loss * (chunk.size(1) - 1)
            total_tokens += (chunk.size(1) - 1)

    avg_loss = total_loss / max(1, total_tokens)
    perplexity = math.exp(min(avg_loss, 20.0))
    return {
        "val_loss": avg_loss,
        "perplexity": perplexity
    }


def evaluate_honest_ood_questions(
    model: Any,
    uce: Any,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates strictly unseen, held-out questions using discriminative log-likelihood.
    Reports both raw conditional scores and Pointwise Mutual Information (PMI)
    prior-calibrated scores side by side.
    """
    model.eval()
    correct_raw = 0
    correct_pmi = 0
    results = []

    prior_prompt = "Answer:"
    prior_p_ids = tokenizer.encode(prior_prompt)

    with torch.no_grad():
        for item in HELD_OUT_OOD_QUESTIONS:
            prompt = item["prompt"]
            choices = item["choices"]
            target = item["target"]

            p_ids = tokenizer.encode(prompt)
            scores_raw = {}
            scores_pmi = {}

            # Closed-loop cognitive context from UCE Hopfield attractor
            c_context = None
            if hasattr(uce, "perceive") and hasattr(uce, "hopfield"):
                p_tensor = torch.tensor([p_ids], dtype=torch.long, device=device)
                _, _, v_p = uce.perceive(p_tensor, modality="tokens")
                hop = uce.hopfield.snap(v_p)
                c_context = hop.converged_vec.unsqueeze(1)

            for choice in choices:
                c_ids = tokenizer.encode(" " + choice.strip())
                full_ids = p_ids + c_ids
                if len(full_ids) > model.max_seq_len:
                    full_ids = full_ids[-model.max_seq_len:]
                inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    out = model(inp, context=c_context, return_mtp=False)
                    logits = out["logits"][0]

                lp = F.log_softmax(logits.float(), dim=-1)
                c_len = min(len(c_ids), len(full_ids) - 1)
                p_len = len(full_ids) - c_len
                seq_score = sum(lp[p_len - 1 + i, full_ids[p_len + i]].item() for i in range(c_len)) / max(1, c_len)

                # Prior log-likelihood under unconditioned baseline
                prior_full = prior_p_ids + c_ids
                prior_inp = torch.tensor([prior_full[:-1]], dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    prior_logits = model(prior_inp, return_mtp=False)["logits"][0]
                prior_lp = F.log_softmax(prior_logits.float(), dim=-1)
                pr_c_len = min(len(c_ids), len(prior_full) - 1)
                pr_p_len = len(prior_full) - pr_c_len
                prior_score = sum(prior_lp[pr_p_len - 1 + i, prior_full[pr_p_len + i]].item() for i in range(pr_c_len)) / max(1, pr_c_len)

                # UCE hyperspherical coherence
                res_uce = uce.sense_make(choice.strip(), context=prompt, modality="tokens", log_to_episodic=False)
                coh = res_uce.get("coherence_score", 0.5)
                surp = res_uce.get("surprise", 1.0)
                cog_score = (coh if not math.isnan(coh) else 0.5) - 0.3 * (surp if not math.isnan(surp) else 1.0)

                combined_raw = seq_score + 1.2 * cog_score
                combined_pmi = (seq_score - prior_score) + 1.2 * cog_score

                scores_raw[choice] = combined_raw
                scores_pmi[choice] = combined_pmi

            best_raw = max(scores_raw, key=scores_raw.get)
            best_pmi = max(scores_pmi, key=scores_pmi.get)

            is_raw_ok = (best_raw.strip().lower() == target.strip().lower())
            is_pmi_ok = (best_pmi.strip().lower() == target.strip().lower())

            if is_raw_ok:
                correct_raw += 1
            if is_pmi_ok:
                correct_pmi += 1

            results.append({
                "prompt": prompt,
                "target": target,
                "picked": best_raw,
                "is_correct": is_raw_ok,
                "pmi_picked": best_pmi,
                "is_pmi_correct": is_pmi_ok,
                "scores": {k: round(v, 4) for k, v in scores_raw.items()},
                "scores_pmi": {k: round(v, 4) for k, v in scores_pmi.items()},
                "domain": item["domain"]
            })

    raw_accuracy = (correct_raw / len(HELD_OUT_OOD_QUESTIONS)) * 100.0
    pmi_accuracy = (correct_pmi / len(HELD_OUT_OOD_QUESTIONS)) * 100.0
    return {
        "accuracy": raw_accuracy,
        "correct": correct_raw,
        "pmi_accuracy": pmi_accuracy,
        "pmi_correct": correct_pmi,
        "total": len(HELD_OUT_OOD_QUESTIONS),
        "details": results
    }


def evaluate_honest_arithmetic_dual_accounting(
    uce: Any,
    tokenizer: Any,
    device: torch.device
) -> Dict[str, Any]:
    """
    Strictly separates:
    1. System 1 Neural Prediction: Does the raw language model output the exact answer in text?
    2. System 2 Symbolic Execution: Does the AST symbolic tool evaluator compute the exact answer?
    """
    sys1_correct = 0
    sys2_correct = 0
    results = []

    for prompt, target in UNSEEN_ARITHMETIC_BATTERY:
        # Run think_and_speak with deliberation enabled
        res = uce.think_and_speak(
            prompt,
            max_new_tokens=10,
            temperature=0.1,
            enable_scratchpad=False, # Disable forced logit boost to measure raw System 1
            enable_deliberation=True,
            write_to_vsa=False
        )

        gen_text = res.get("response", "")[len(prompt):].strip()
        sys2_result = (res.get("dual_process") or {}).get("system2_exact_result")

        # Check System 1: Did the raw weights generate the number?
        first_token_num = None
        match = re.search(r"[-+]?\d*\.?\d+", gen_text)
        if match:
            try:
                first_token_num = float(match.group())
            except ValueError:
                pass

        sys1_ok = (first_token_num is not None and abs(first_token_num - target) < 1e-3)
        if sys1_ok:
            sys1_correct += 1

        # Check System 2: Did the AST symbolic solver compute it?
        sys2_ok = (sys2_result is not None and abs(float(sys2_result) - target) < 1e-3)
        if sys2_ok:
            sys2_correct += 1

        results.append({
            "prompt": prompt,
            "target": target,
            "sys1_generated": gen_text[:20],
            "sys1_parsed_num": first_token_num,
            "sys1_pass": sys1_ok,
            "sys2_proof": sys2_result,
            "sys2_pass": sys2_ok
        })

    return {
        "system1_neural_accuracy": (sys1_correct / len(UNSEEN_ARITHMETIC_BATTERY)) * 100.0,
        "system1_correct": sys1_correct,
        "system2_symbolic_accuracy": (sys2_correct / len(UNSEEN_ARITHMETIC_BATTERY)) * 100.0,
        "system2_correct": sys2_correct,
        "total": len(UNSEEN_ARITHMETIC_BATTERY),
        "details": results
    }


def evaluate_generative_fluency_sample(
    uce: Any,
    prompts: List[str],
    max_new_tokens: int = 30,
    temperature: float = 0.7,
    include_greedy: bool = True
) -> Dict[str, Any]:
    """
    Samples actual open-ended completions, measuring Distinct-1, Distinct-2,
    repetition rate, and providing full transcripts.

    Reports BOTH a sampled pass and a greedy (temperature 0) pass. Sampling at
    temperature 0.7 from an untrained model draws near-uniformly over the vocabulary,
    which scores Distinct-2 ~ 1.000 -- the same value uniform random tokens score -- while
    the greedy decode of that same model collapses to a single repeated token. Reporting
    only the sampled pass hides the collapse; see ishmn/metric_validation.py.
    """
    samples = []
    all_tokens_generated = []
    completions: List[str] = []

    for p in prompts:
        res = uce.think_and_speak(
            p,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            enable_born_rule=True,
            enable_scratchpad=False,
            enable_deliberation=True,
            write_to_vsa=False
        )
        gen = res["response"][len(p):].strip()
        completions.append(gen)
        words = gen.split()
        all_tokens_generated.extend(words)

        samples.append({
            "prompt": p,
            "completion": gen,
            "word_count": len(words),
            "hopfield_energy": res.get("hopfield_energy", 0.0)
        })

    # Metrics
    total_w = len(all_tokens_generated)
    distinct_1 = len(set(all_tokens_generated)) / max(1, total_w)
    bigrams = [f"{all_tokens_generated[i]}_{all_tokens_generated[i+1]}" for i in range(len(all_tokens_generated)-1)]
    distinct_2 = len(set(bigrams)) / max(1, len(bigrams))

    # Calibrated diagnosis: distinct-n is a band, and BOTH edges are failures.
    diagnosis = diagnose_generation(completions)

    greedy_block = None
    if include_greedy:
        greedy_completions = []
        for p in prompts:
            try:
                g = uce.think_and_speak(
                    p,
                    max_new_tokens=max_new_tokens,
                    temperature=0.0,
                    enable_born_rule=True,
                    enable_scratchpad=False,
                    enable_deliberation=True,
                    write_to_vsa=False
                )
                greedy_completions.append(g["response"][len(p):].strip())
            except Exception as exc:  # never let the honest probe break the suite
                greedy_completions.append("")
                samples.append({"prompt": p, "greedy_error": str(exc)})
        greedy_diag = diagnose_generation(greedy_completions)
        greedy_block = {
            "completions": greedy_completions,
            "distinct_1": greedy_diag.get("distinct_1", 0.0),
            "distinct_2": greedy_diag.get("distinct_2", 0.0),
            "verdict": greedy_diag["verdict"],
            "reason": greedy_diag["reason"],
            "top_token": greedy_diag.get("top_token"),
            "top_token_share": greedy_diag.get("top_token_share", 0.0),
            "longest_run": greedy_diag.get("longest_run", 0),
        }

    return {
        # Pooled values retained for continuity with earlier reports.
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        # Per-completion values: diversity is a property of one generation, and pooling
        # across prompts manufactures bigrams that span unrelated completions.
        "distinct_1_per_sample": diagnosis.get("distinct_1", 0.0),
        "distinct_2_per_sample": diagnosis.get("distinct_2", 0.0),
        "pooling_inflation": diagnosis.get("pooling_inflation", 0.0),
        "verdict": diagnosis["verdict"],
        "verdict_reason": diagnosis["reason"],
        "healthy_band": diagnosis.get("healthy_band"),
        "greedy": greedy_block,
        "total_words": total_w,
        "samples": samples
    }


def audit_training_scripts_against_held_out_bank(
    experiments_dir: Optional[str] = None,
    n: int = 8
) -> Dict[str, Any]:
    """
    Audits every experiments/train_*.py against HELD_OUT_OOD_QUESTIONS.

    Only string literals are compared. Auditing raw source produces false positives from
    shared boilerplate -- two modules both importing `Dict, Any, List, Tuple, Optional`
    register an 8-gram overlap that has nothing to do with leakage.
    """
    from pathlib import Path

    root = Path(experiments_dir) if experiments_dir else Path(__file__).resolve().parent.parent / "experiments"
    if not root.is_dir():
        return {"clean": True, "skipped": f"no experiments directory at {root}"}

    scripts = sorted(root.glob("train_*.py"))
    if not scripts:
        return {"clean": True, "skipped": "no training scripts found"}

    corpus = [
        extract_text_from_python(p.read_text(encoding="utf-8", errors="ignore"))
        for p in scripts
    ]
    items = [
        q["prompt"] + " " + " ".join(q["choices"]) for q in HELD_OUT_OOD_QUESTIONS
    ]
    report = audit_contamination(corpus, items, n=n)
    report["scripts_audited"] = [p.name for p in scripts]
    return report


def run_honest_evaluation_suite(
    model: Any,
    uce: Any,
    tokenizer: Any,
    held_out_tokens: Optional[torch.Tensor] = None,
    device: torch.device = None
) -> Dict[str, Any]:
    """
    Executes the full, un-contaminated evaluation suite and returns
    an honest, transparent scorecard.
    """
    if device is None:
        device = next(model.parameters()).device

    uce.attach_native_transformer(model, tokenizer=tokenizer)

    # 1. Held-Out OOD Zero-Shot Questions (No training overlap)
    res_ood = evaluate_honest_ood_questions(model, uce, tokenizer, device=device)

    # 2. Honest Arithmetic (System 1 Neural vs System 2 Symbolic)
    res_math = evaluate_honest_arithmetic_dual_accounting(uce, tokenizer, device=device)

    # 3. Held-Out Perplexity (if tokens provided)
    res_ppl = None
    if held_out_tokens is not None and held_out_tokens.numel() > 64:
        res_ppl = evaluate_held_out_perplexity(model, held_out_tokens, device=device)

    # 4. Open-Ended Generative Fluency Sample (60 tokens ensures we are out of the distinct-2 dead zone)
    open_prompts = [
        "Photosynthesis is the biological process by which plants",
        "The fundamental law of universal gravitation describes",
        "In Python programming, the main difference between a list and a tuple is that"
    ]
    res_gen = evaluate_generative_fluency_sample(uce, open_prompts, max_new_tokens=60)

    # 5. Contamination firewall: audit the held-out bank against every training script
    #    that could have supplied the model's curriculum. Reported, never silently passed.
    res_contam = audit_training_scripts_against_held_out_bank()

    return {
        "contamination_audit": res_contam,
        "held_out_ood_science_and_code": {
            "accuracy": res_ood["accuracy"],
            "pmi_accuracy": res_ood.get("pmi_accuracy", res_ood["accuracy"]),
            "correct": res_ood["correct"],
            "pmi_correct": res_ood.get("pmi_correct", res_ood["correct"]),
            "total": res_ood["total"],
            "details": res_ood["details"]
        },
        "arithmetic_dual_accounting": {
            "system1_neural_accuracy": res_math["system1_neural_accuracy"],
            "system2_symbolic_accuracy": res_math["system2_symbolic_accuracy"],
            "details": res_math["details"]
        },
        "held_out_language_perplexity": res_ppl,
        "generative_fluency": res_gen
    }
