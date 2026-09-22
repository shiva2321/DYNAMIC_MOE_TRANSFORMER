"""
Official Reasoning Benchmark Harness.
Evaluates Language Models on:
1. ARC-Challenge (AI2 Reasoning Challenge): Grade-school scientific, physical, and causal reasoning.
2. GSM8K (Grade School Math 8K): Multi-step arithmetic and numerical problem-solving.

Supports:
--mode baseline : Pure ungrounded base SLM (Qwen2.5-1.5B)
--mode grounded : Grounded Hybrid Engine (Qwen2.5-1.5B + ISHMN S^7 Latent Sandwich)
"""

import os
import sys
import re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.grounded_hybrid_engine import GroundedHybridEngine

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def answers_match(pred: str, gold: str, tol: float = 1e-6) -> bool:
    """Robust numeric match with tolerance for floating point and integer equivalence."""
    if not pred or not gold:
        return False
    try:
        p_val = float(pred.replace(",", "").strip())
        g_val = float(gold.replace(",", "").strip())
        return abs(p_val - g_val) <= tol * max(1.0, abs(g_val))
    except (ValueError, TypeError):
        return pred.strip().lower() == gold.strip().lower()


def extract_gsm8k_answer(text: str) -> str:
    """Extracts numeric answer following #### or standard answer patterns."""
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        clean = match.group(1).replace(",", "").strip()
        try:
            return str(float(clean)) if "." in clean else str(int(clean))
        except ValueError:
            return clean

    # Standard answer phrasing patterns: "answer is ... <num>", "total is ... <num>"
    patterns = [
        r"(?:answer is|equals?|total is|total of|needs to pay|worth)\s*\$?(-?[\d,]+(?:\.\d+)?)",
        r"\$(-?[\d,]+(?:\.\d+)?)",
    ]
    for pat in patterns:
        m = re.findall(pat, text, re.IGNORECASE)
        if m:
            clean = m[-1].replace(",", "").strip()
            num_m = re.search(r"-?[\d]+(?:\.\d+)?", clean)
            if num_m:
                c = num_m.group(0)
                try:
                    return str(float(c)) if "." in c else str(int(c))
                except ValueError:
                    return c

    # Fallback: extract last number in text
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    if nums:
        clean = nums[-1].replace(",", "").strip()
        try:
            return str(float(clean)) if "." in clean else str(int(clean))
        except ValueError:
            return clean
    return ""


def score_choices_with_uce(uce, question: str, choices: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Evaluates each candidate choice against UCE's crystallized physical & causal invariants.
    Returns relative score for each choice label.
    """
    labels = choices["label"]
    texts = choices["text"]
    scores = {}
    for lbl, txt in zip(labels, texts):
        hypo = f"{question} {txt}"
        res = uce.sense_make(hypo, modality="subwords", log_to_episodic=False)
        coh = res.get("coherence_score", 0.5)
        surp = res.get("surprise", 1.0)
        # Higher coherence and lower geodesic surprise indicates alignment with physical invariants
        score = coh - 0.4 * surp
        scores[lbl] = score

    # Zero-center scores
    mean_s = sum(scores.values()) / max(len(scores), 1)
    return {lbl: s - mean_s for lbl, s in scores.items()}


class ChoiceLogitsProcessor(LogitsProcessor):
    """
    Directly aligns multiple-choice logits with UCE's physical plausibility score.
    """
    def __init__(self, choice_scores: Dict[str, float], tokenizer, boost_weight: float = 2.5):
        self.choice_scores = choice_scores
        self.tokenizer = tokenizer
        self.boost_weight = boost_weight

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for lbl, rel_score in self.choice_scores.items():
            for prefix in ["", " "]:
                tok_ids = self.tokenizer.encode(prefix + lbl, add_special_tokens=False)
                if tok_ids:
                    tid = tok_ids[0]
                    if tid < scores.shape[-1]:
                        scores[:, tid] += self.boost_weight * rel_score
        return scores


def evaluate_arc_challenge(
    model,
    tokenizer,
    engine: GroundedHybridEngine,
    num_samples: int = 50,
    is_grounded: bool = False
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" EVALUATING ARC-CHALLENGE (Physical & Scientific Reasoning) [{'GROUNDED' if is_grounded else 'BASELINE'}]")
    print("=" * 80)

    dataset = load_dataset("ai2_arc", "ARC-Challenge", split="test")
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(samples, 1):
        q_id = item["id"]
        question = item["question"]
        choices = item["choices"]
        ans_key = item["answerKey"].strip()

        # Format choices
        labels = choices["label"]
        texts = choices["text"]
        choice_str = "\n".join([f"{lbl}. {txt}" for lbl, txt in zip(labels, texts)])

        prompt = (
            f"Question: {question}\n\n"
            f"Choices:\n{choice_str}\n\n"
            f"Answer with the single letter corresponding to the correct choice (e.g. A, B, C, or D).\n"
            f"Answer:"
        )

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([formatted], return_tensors="pt").to(model.device)

        if is_grounded and engine is not None:
            # Score choices via UCE physical coherence
            choice_scores = score_choices_with_uce(engine.uce, question, choices)
            choice_proc = ChoiceLogitsProcessor(choice_scores, tokenizer, boost_weight=2.8)

            engine.sandwich.is_active = True
            engine.sandwich.reset_telemetry()
            lp = LogitsProcessorList([choice_proc])

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=15,
                    do_sample=False,
                    logits_processor=lp
                )
            gen_text = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            telemetry = engine.sandwich.deliberation_telemetry
            avg_ticks = float(np.mean([t["step_count"] for t in telemetry])) if telemetry else 1.0
            mean_err = float(np.mean([t["causal_error"] for t in telemetry])) if telemetry else 0.0
        else:
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=15, do_sample=False)
            gen_text = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            telemetry = []
            avg_ticks = 1.0
            mean_err = 0.0

        # Parse answer letter (first letter matching available labels)
        pred_label = ""
        first_token = gen_text.split()[0] if gen_text else ""
        first_clean = re.sub(r"[^A-Za-z0-9]", "", first_token).upper()
        if first_clean in labels:
            pred_label = first_clean
        else:
            found = [lbl for lbl in labels if re.search(r"\b" + lbl + r"\b", gen_text.upper())]
            pred_label = found[0] if found else (first_clean[:1] if first_clean else "")

        is_correct = (pred_label == ans_key)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": i,
            "id": q_id,
            "question": question,
            "choices": choice_str,
            "answerKey": ans_key,
            "predicted": pred_label,
            "raw_output": gen_text,
            "is_correct": is_correct,
            "avg_thinking_ticks": avg_ticks,
            "mean_causal_error": mean_err
        })

        acc = (correct / total) * 100
        print(f"  [ARC {i:2d}/{num_samples}] Acc: {acc:.1f}% ({correct}/{total}) | Q: {q_id} | Pred: {pred_label} | Gold: {ans_key}", flush=True)

    final_accuracy = (correct / total) * 100
    print(f"\n[+] ARC-Challenge Completed: {correct}/{total} Correct ({final_accuracy:.2f}% Accuracy)\n", flush=True)
    return {
        "benchmark": "ARC-Challenge",
        "is_grounded": is_grounded,
        "num_samples": total,
        "correct": correct,
        "accuracy_pct": final_accuracy,
        "details": results
    }


def evaluate_gsm8k(
    model,
    tokenizer,
    engine: GroundedHybridEngine,
    num_samples: int = 30,
    is_grounded: bool = False,
    max_new_tokens: int = 450,
    enable_scratchpad: bool = False
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print(f" EVALUATING GSM8K (Multi-Step Mathematical Reasoning) [{'GROUNDED' if is_grounded else 'BASELINE'}]")
    print(f" Manifold Deliberation: {'ENABLED' if is_grounded else 'DISABLED'} | Scratchpad Logits: {enable_scratchpad}")
    print("=" * 80, flush=True)

    dataset = load_dataset("gsm8k", "main", split="test")
    samples = [dataset[i] for i in range(min(num_samples, len(dataset)))]

    correct = 0
    total = 0
    results = []

    for i, item in enumerate(samples, 1):
        question = item["question"]
        gold_solution = item["answer"]
        gold_num = extract_gsm8k_answer(gold_solution)

        prompt = (
            f"Solve this math problem step by step. Conclude your response with '#### <number>'.\n\n"
            f"Question: {question}\n\n"
            f"Solution:"
        )

        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([formatted], return_tensors="pt").to(model.device)

        if is_grounded and engine is not None:
            res = engine.generate_grounded(
                prompt,
                max_new_tokens=max_new_tokens,
                enable_sandwich=True,
                enable_neurosymbolic_scratchpad=enable_scratchpad
            )
            gen_text = res["text"].strip()
            avg_ticks = res.get("avg_thinking_ticks", 1.0)
            mean_err = res.get("mean_causal_error", 0.0)
        else:
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            gen_text = tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()
            avg_ticks = 1.0
            mean_err = 0.0

        pred_num = extract_gsm8k_answer(gen_text)
        is_correct = answers_match(pred_num, gold_num) if (pred_num and gold_num) else False
        if is_correct:
            correct += 1
        total += 1

        acc = (correct / total) * 100
        print(f"  [GSM {i:2d}/{num_samples}] Acc: {acc:.1f}% ({correct}/{total}) | Pred: {pred_num} | Gold: {gold_num}", flush=True)

        results.append({
            "idx": i,
            "question": question,
            "gold_solution": gold_solution,
            "gold_num": gold_num,
            "pred_num": pred_num,
            "raw_output": gen_text,
            "is_correct": is_correct,
            "avg_thinking_ticks": avg_ticks,
            "mean_causal_error": mean_err
        })

        if i % 10 == 0 or i == num_samples:
            acc = (correct / total) * 100
            print(f"  Processed {i:2d}/{num_samples} | Accuracy: {correct}/{total} ({acc:.1f}%)")

    final_accuracy = (correct / total) * 100
    print(f"\n[+] GSM8K Completed: {correct}/{total} Correct ({final_accuracy:.2f}% Accuracy)")
    return {
        "benchmark": "GSM8K",
        "is_grounded": is_grounded,
        "num_samples": total,
        "correct": correct,
        "accuracy_pct": final_accuracy,
        "details": results
    }


def main():
    parser = argparse.ArgumentParser(description="Official Reasoning Benchmark Evaluation")
    parser.add_argument("--mode", choices=["baseline", "grounded"], default="baseline")
    parser.add_argument("--arc_samples", type=int, default=50)
    parser.add_argument("--gsm_samples", type=int, default=30)
    parser.add_argument("--gsm_tokens", type=int, default=450)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    is_grounded = (args.mode == "grounded")
    out_file = args.save_path or f"artifacts/benchmark_{args.mode}_results.json"

    print("=" * 85)
    print(f" OFFICIAL REASONING BENCHMARK EVALUATION [{args.mode.upper()}]")
    print(f" Device: {DEVICE} | ARC Samples: {args.arc_samples} | GSM8K Samples: {args.gsm_samples} | GSM Tokens: {args.gsm_tokens}")
    print("=" * 85)

    if is_grounded:
        print("[*] Initializing Grounded Hybrid Cognitive Engine...")
        engine = GroundedHybridEngine(
            model_name=DEFAULT_MODEL_NAME,
            brain_path=DEFAULT_BRAIN_PATH,
            device=DEVICE
        )
        model = engine.slm
        tokenizer = engine.tokenizer
    else:
        print(f"[*] Initializing Base SLM: {DEFAULT_MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_MODEL_NAME,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        model.eval()
        engine = None

    t0 = time.time()
    arc_results = evaluate_arc_challenge(
        model=model,
        tokenizer=tokenizer,
        engine=engine,
        num_samples=args.arc_samples,
        is_grounded=is_grounded
    )

    gsm_results = evaluate_gsm8k(
        model=model,
        tokenizer=tokenizer,
        engine=engine,
        num_samples=args.gsm_samples,
        is_grounded=is_grounded,
        max_new_tokens=args.gsm_tokens
    )

    total_time = time.time() - t0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": args.mode,
        "elapsed_sec": total_time,
        "arc_challenge": arc_results,
        "gsm8k": gsm_results
    }

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 85)
    print(" BENCHMARK SUMMARY REPORT")
    print("=" * 85)
    print(f" Mode                       : {args.mode.upper()}")
    print(f" ARC-Challenge Accuracy     : {arc_results['correct']}/{arc_results['num_samples']} ({arc_results['accuracy_pct']:.2f}%)")
    print(f" GSM8K Multi-Step Accuracy  : {gsm_results['correct']}/{gsm_results['num_samples']} ({gsm_results['accuracy_pct']:.2f}%)")
    print(f" Combined Total Time        : {total_time:.2f}s")
    print(f" Saved Detailed Report to   : {out_file}")
    print("=" * 85)


if __name__ == "__main__":
    main()
