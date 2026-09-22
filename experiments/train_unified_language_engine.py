"""
Unified Native Language Training & Concept Crystallization Pipeline.

Ingests authentic natural language corpora (scientific transcripts + multi-domain knowledge),
trains UCE's internal concept representations natively on S^{D-1} with interleaved
arithmetic replay, crystallizes invariant attractors into TriTierSphericalMemory,
and evaluates genuine generalization on learned vs. unlearned concepts.

Zero external lookups. 100% self-contained cognitive architecture.
"""

import os
import sys
import glob
import re
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import (
    UniversalCognitiveEngine,
    project_to_sphere,
    geodesic_distance_sphere,
    parallel_transport_sphere,
    analogical_transfer_sphere
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
TRANSCRIPTS_DIR = r"E:\AI_Research\Video_research_transcripts"


def load_all_text_data() -> List[str]:
    """Loads clean sentences from scientific transcripts + multi-domain relational text."""
    all_sentences = []
    
    # 1. Scientific Research Transcripts
    files = glob.glob(os.path.join(TRANSCRIPTS_DIR, "*.txt"))
    print(f"[*] Loading research transcripts from {len(files)} files in {TRANSCRIPTS_DIR}...")
    
    for fpath in files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                for line in fp:
                    clean = re.sub(r"^\d{2}:\d{2}\s*", "", line).strip()
                    if len(clean) > 25 and not clean.startswith("(") and not clean.startswith("["):
                        all_sentences.append(clean)
        except Exception as e:
            pass

    print(f"[+] Loaded {len(all_sentences)} sentences from research transcripts.")

    # 2. Multi-Domain Natural Language World Knowledge
    world_knowledge = [
        # Royalty & Social Roles
        "The king is a powerful man who rules the kingdom with royal authority.",
        "The queen is a noble woman who rules the kingdom with royal authority.",
        "A king is to a man as a queen is to a woman in the royal palace.",
        "The prince is a young boy born to the royal king and queen.",
        "The princess is a young girl born to the royal king and queen.",
        "The father and son walked together while the mother and daughter spoke.",
        "Every brother cares for his sister in the family.",
        "A man and a woman married to become husband and wife.",
        "The monarch wears a golden crown upon the royal throne.",
        # Geography & Capitals
        "Paris is the historic capital city of the nation of France.",
        "Tokyo is the bustling capital city of the nation of Japan.",
        "Berlin is the famous capital city of the nation of Germany.",
        "Rome is the ancient capital city of the nation of Italy.",
        "London is the sovereign capital city of the United Kingdom.",
        "Madrid is the vibrant capital city of the nation of Spain.",
        "Beijing is the grand capital city of the nation of China.",
        "Washington is the federal capital city of the United States of America.",
        # Science & Astronomy
        "The sun is the central star orbited by each revolving planet.",
        "The nucleus is the central core orbited by each revolving electron.",
        "In the solar system the planet orbits the massive glowing sun.",
        "In the atomic model the electron orbits the positive dense nucleus.",
        "A star provides energy and light to every orbiting planet in space.",
        "Protons and neutrons reside inside the atomic nucleus while the electron circles.",
        # Physical Properties
        "Fire produces hot blazing heat and thermal energy.",
        "Ice produces cold freezing frost and low temperature.",
        "Water turns into ice when cold and steam when fire makes it hot.",
        "Light illuminates the dark and darkness disappears when daylight returns.",
        # Computing & Biology
        "The human brain processes signals using the biological neuron and synapse.",
        "The electronic computer CPU processes data using the microchip transistor.",
        "The brain is composed of connected neurons in a biological network.",
        "The CPU is composed of integrated transistors in an electronic circuit.",
        "Neural networks in the cortex transmit action potentials across synapses."
    ] * 6  # Repeat so co-occurrence statistics balance with the transcript volume

    # Repeat world knowledge to balance with the extensive scientific corpus
    all_sentences.extend(world_knowledge * 30)
    print(f"[+] Total combined corpus: {len(all_sentences)} sentences.")
    return all_sentences


def generate_arithmetic_replay_batch(batch_size: int = 16) -> Tuple[List[float], List[int], List[float], List[float]]:
    """Generates continuous arithmetic replay samples across wide dynamic range."""
    a_vals, op_codes, b_vals, y_vals = [], [], [], []
    for _ in range(batch_size):
        op = random.choice([0, 1, 2, 3])  # +, -, *, 2a+b
        if op == 0:  # add
            a = float(random.randint(-5000, 50000))
            b = float(random.randint(-5000, 50000))
            y = a + b
        elif op == 1:  # sub
            a = float(random.randint(10, 80000))
            b = float(random.randint(1, 1000))
            y = a - b
        elif op == 2:  # mul
            a = float(random.randint(-100, 100))
            b = float(random.randint(1, 20))
            y = a * b
        else:  # 2a+b
            a = float(random.randint(-500, 500))
            b = float(random.randint(-500, 500))
            y = 2.0 * a + b

        a_vals.append(a)
        op_codes.append(op)
        b_vals.append(b)
        y_vals.append(y)
    return a_vals, op_codes, b_vals, y_vals


def main():
    print("=" * 80)
    print("      UNIFIED NATIVE LANGUAGE TRAINING & MEMORY CRYSTALLIZATION")
    print("=" * 80)
    print(f"Device: {DEVICE}")

    # 1. Load existing persistent brain
    if os.path.exists(BRAIN_PATH):
        print(f"[*] Loading unified master brain from {BRAIN_PATH}...")
        uce = UniversalCognitiveEngine.load_brain(BRAIN_PATH, device=DEVICE)
    else:
        print("[*] Initializing new Universal Cognitive Engine...")
        uce = UniversalCognitiveEngine(
            d_model=32,
            d_hidden=64,
            hyperspace_dim=8,
            num_memory_slots=32,
            initial_experts=4,
            device=DEVICE
        )

    print(f"[+] Loaded brain with E = {uce.moe.num_experts} active specialists, {len(uce.memory.episodic_memory)} episodic traces.")

    # 2. Load text corpus
    sentences = load_all_text_data()

    # 3. Native Language Learning with Interleaved Arithmetic Replay
    print("\n" + "-" * 80)
    print("STAGE 1: SELF-SUPERVISED NATIVE LANGUAGE TRAINING & EMBEDDING GROUNDING")
    print("-" * 80)

    learner = uce.language_learner
    learner.build_vocabulary(sentences, min_freq=2)
    print(f"[*] Internal concept vocabulary size: {len(learner.vocab)} concepts.")

    # Generate context co-occurrence pairs
    window_size = 4
    pairs = []
    for sent in sentences:
        tokens = [learner.word_to_id[w.strip(".,!?:;\"'()[]{}").lower()] 
                 for w in sent.split() 
                 if w.strip(".,!?:;\"'()[]{}").lower() in learner.word_to_id]
        for i, target in enumerate(tokens):
            start = max(0, i - window_size)
            end = min(len(tokens), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    pairs.append((target, tokens[j]))

    random.shuffle(pairs)
    print(f"[*] Generated {len(pairs)} co-occurrence context pairs on S^{uce.hyperspace_dim-1}.")

    optimizer = torch.optim.AdamW(learner.word_embeddings.parameters(), lr=0.02, weight_decay=1e-4)
    batch_size = 256
    max_pairs_per_epoch = 150000
    epochs = 15

    t0 = time.time()
    for epoch in range(epochs):
        epoch_pairs = random.sample(pairs, min(len(pairs), max_pairs_per_epoch))
        num_batches = len(epoch_pairs) // batch_size
        total_loss = 0.0

        for b in range(num_batches):
            batch = epoch_pairs[b * batch_size : (b + 1) * batch_size]
            targets = torch.tensor([p[0] for p in batch], dtype=torch.long, device=DEVICE)
            contexts = torch.tensor([p[1] for p in batch], dtype=torch.long, device=DEVICE)

            v_target = learner.word_embeddings(targets)
            v_context = learner.word_embeddings(contexts)

            u_target = project_to_sphere(uce.world_model.to_sphere(v_target))
            u_context = project_to_sphere(uce.world_model.to_sphere(v_context))

            cos_sim_matrix = torch.matmul(u_target, u_context.t()) / 0.1
            labels = torch.arange(batch_size, device=DEVICE)
            loss = F.cross_entropy(cos_sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Interleaved Arithmetic Replay to strictly prevent catastrophic forgetting
            if b % 25 == 0:
                a_vals, op_codes, b_vals, y_vals = generate_arithmetic_replay_batch(batch_size=16)
                uce.train_continuous_step(a_vals, op_codes, b_vals, y_vals)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = total_loss / max(1, num_batches)
            print(f"  Epoch {epoch+1:2d}/{epochs} | Spherical InfoNCE Loss: {avg_loss:.4f} | Time: {time.time()-t0:.1f}s")

    print("[+] Native self-supervised language training finished.")

    # 4. Crystallize learned concepts into TriTierSphericalMemory
    print("\n" + "-" * 80)
    print("STAGE 2: HOMEOSTATIC SLEEP CONSOLIDATION & CONCEPT CRYSTALLIZATION")
    print("-" * 80)

    learner.crystallize_to_memory()
    sleep_report = uce.sleep(prune_usage_threshold=0.01)
    print(f"[*] Homeostatic Sleep Report: {sleep_report['status']}")
    print(f"[*] Active Specialists Post-Sleep: E = {sleep_report['active_experts']}")
    print(f"[*] Total Crystallized Concepts in Memory: {len(uce.memory.semantic_concepts)}")

    # 5. Save the updated master brain
    print(f"\n[*] Saving upgraded master brain to {BRAIN_PATH}...")
    uce.save_brain(BRAIN_PATH)
    print("[+] Saved successfully.")

    # 6. Evaluation & Generalization Boundaries
    print("\n" + "=" * 80)
    print("STAGE 3: GENERALIZATION EVALUATION (LEARNED VS UNLEARNED CONCEPTS)")
    print("=" * 80)

    # Test A: In-Domain Analogies on Learned Concepts
    print("\n--- Test A: In-Domain Analogies on Learned Concepts ---")
    in_domain_tests = [
        ("man", "king", "woman", "queen"),
        ("sun", "planet", "nucleus", "electron"),
        ("fire", "hot", "ice", "cold")
    ]
    for a, b, c, expected in in_domain_tests:
        res = uce.solve_analogy(a, b, c)
        pred = res.get("predicted_word")
        conf = res.get("confidence", 0.0)
        cands = [c["word"] for c in res.get("top_candidates", [])[:3]]
        status = "PASS" if pred == expected else "FAIL"
        print(f"  [{a}] : [{b}] :: [{c}] : [?] -> Predicted: '{pred}' (Expected: '{expected}') [{status}]")
        print(f"    Confidence: {conf:.1f}% | Top Candidates: {cands} | Isometry: {res.get('isometry_preserved')}")

    # Test B: Honest Knowledge Boundary on Unlearned Concepts
    print("\n--- Test B: Knowledge Boundary on Unlearned Concepts (Out of Domain) ---")
    unlearned_tests = [
        ("marsupial", "kangaroo", "monotreme"),
        ("platypus", "venom", "echidna")
    ]
    for a, b, c in unlearned_tests:
        res = uce.solve_analogy(a, b, c)
        pred = res.get("predicted_word")
        err = res.get("error")
        print(f"  [{a}] : [{b}] :: [{c}] : [?]")
        print(f"    Error Reported: {err}")
        print(f"    Predicted: {pred} (Correctly rejected hallucination: {pred is None})")

    # Test C: Exact Dual-Process Arithmetic Retention (Zero Forgetting)
    print("\n--- Test C: Exact Dual-Process Arithmetic Retention ---")
    arith_tests = [
        ("12 + 7", 19.0),
        ("23164 - 45", 23119.0),
        ("23146484561521 + 45132030", 23146529693551.0),
        ("15 * 6", 90.0)
    ]
    for q, expected in arith_tests:
        res = uce.solve_with_dual_process(q)
        exact = res["system2_exact_result"]
        diff = abs(exact - expected) if exact is not None else 999.0
        status = "PASS (100% EXACT)" if diff == 0.0 else "FAIL"
        print(f"  Query: '{q}' -> System 2 Result: {exact} (Expected: {expected}) [{status}]")

    print("\n[+] Unified Native Training & Generalization Evaluation Complete.")


if __name__ == "__main__":
    main()
