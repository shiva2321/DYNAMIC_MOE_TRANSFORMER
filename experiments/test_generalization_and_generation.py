import os
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import SovereignTokenizer
from ishmn.training_engine import MODEL_PROFILES
from ishmn.geometry import geodesic_distance_sphere

def test_ood_generalization():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt_path = 'artifacts/checkpoints/native_moe_acc100.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'artifacts/checkpoints/native_moe_acc96_plus.pt'
    if not os.path.exists(ckpt_path):
        ckpt_path = 'artifacts/checkpoints/native_moe_acc80_plus.pt'

    
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = MODEL_PROFILES['nano']
    tokenizer = SovereignTokenizer('gpt2')

    model = NativeSphericalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg['d_model'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        hyperspace_dim=cfg['hyperspace_dim'],
        d_ff=cfg['d_ff'],
        max_seq_len=96,
        mtp_future_tokens=cfg['mtp_future_tokens'],
        vsa_capacity=cfg['vsa_capacity'],
        num_experts=cfg['num_experts'],
        top_k=cfg['top_k'],
        enable_shared_expert=cfg['enable_shared_expert'],
        enable_aux_loss_free_balancing=cfg['enable_aux_loss_free_balancing'],
        enable_vsa_memory=True,
        enable_world_model=True,
        device=device
    ).to(device)

    uce = UniversalCognitiveEngine(
        d_model=cfg['d_model'],
        d_hidden=cfg['d_ff'],
        hyperspace_dim=cfg['hyperspace_dim'],
        num_memory_slots=32,
        vocab_size=tokenizer.vocab_size,
        initial_experts=cfg['num_experts'],
        device=device
    ).to(device)
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    uce.load_state_dict(ckpt['uce_state_dict'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    uce.eval()

    print("\n" + "=" * 75)
    print(" 1. OUT-OF-DISTRIBUTION (OOD) UNSEEN GENERALIZATION BATTERY")
    print("=" * 75)

    # A. Unseen Arithmetic (Testing pure algorithmic computation, zero memorization)
    ood_math = [
        ("Calculate: 37 * 6 = ", "222.0"),
        ("Calculate: 150 - 68 = ", "82.0"),
        ("Calculate: 250 / 5 = ", "50.0"),
        ("Calculate: (12.5 - 2.5) * 8 = ", "80.0"),
        ("Calculate: 19 + 84 = ", "103.0"),
        ("Calculate: 45 * 3 = ", "135.0")
    ]
    math_correct = 0
    print("\n[A. Unseen Arithmetic (Generalization vs Lookup)]")
    for prompt, target in ood_math:
        res = uce.think_and_speak(prompt, max_new_tokens=10, temperature=0.1)
        gen = res["response"]
        matched = target in gen or abs(float(target) - float(gen.split("=")[-1].strip().split()[0] if "=" in gen and gen.split("=")[-1].strip().split()[0].replace(".", "").isdigit() else -999)) < 1e-4
        ok = "PASS" if matched else "FAIL"
        if matched:
            math_correct += 1
        print(f"  [{ok}] Prompt: '{prompt}' | Target: {target} | Output: '{gen}'")
    math_acc = (math_correct / len(ood_math)) * 100.0

    # B. Unseen Causal Sensemaking (Novel Physical Scenarios)
    ood_causal = [
        (
            "a helium balloon released into the Earth's lower atmosphere",
            "rises upward due to buoyancy forces lighter than air",
            "sinks down into molten lead like an anchor"
        ),
        (
            "a burning candle placed inside a sealed airtight glass container",
            "extinguishes when available oxygen gas is consumed",
            "burns forever without consuming any chemical fuel"
        ),
        (
            "liquid water heated to 200 degrees Celsius in an open pan",
            "rapidly boils away and transforms into steam vapor",
            "freezes solid into cryogenic nitrogen ice"
        ),
        (
            "a solid chunk of diamond placed in cold water",
            "remains intact and sinks to the bottom of the container",
            "dissolves immediately into boiling liquid magma"
        )
    ]
    causal_correct = 0
    print("\n[B. Unseen Causal Scenarios (Physical Sensemaking Generalization)]")
    for premise, fact, contra in ood_causal:
        _, _, z_p = uce.perceive(premise, modality="subwords")
        res_f = uce.sense_make(fact, prior_z=z_p, context=premise, modality="subwords", log_to_episodic=False)
        res_c = uce.sense_make(contra, prior_z=z_p, context=premise, modality="subwords", log_to_episodic=False)
        
        coh_f = res_f.get("coherence_score", 0.5)
        coh_c = res_c.get("coherence_score", 0.5)
        surp_f = res_f.get("surprise", 1.0)
        surp_c = res_c.get("surprise", 1.0)
        
        # Sequence score
        p_ids = tokenizer.encode(premise)
        f_ids = tokenizer.encode(" " + fact.strip())
        c_ids = tokenizer.encode(" " + contra.strip())
        
        with torch.no_grad():
            inp_f = torch.tensor([(p_ids + f_ids)[-model.max_seq_len:-1]], device=device)
            out_f = model(inp_f)
            lp_f = F.log_softmax(out_f["logits"][0], dim=-1)
            sc_f = sum(lp_f[-len(f_ids) + i, (p_ids + f_ids)[-len(f_ids) + i]].item() for i in range(len(f_ids))) / len(f_ids)
            
            inp_c = torch.tensor([(p_ids + c_ids)[-model.max_seq_len:-1]], device=device)
            out_c = model(inp_c)
            lp_c = F.log_softmax(out_c["logits"][0], dim=-1)
            sc_c = sum(lp_c[-len(c_ids) + i, (p_ids + c_ids)[-len(c_ids) + i]].item() for i in range(len(c_ids))) / len(c_ids)

        score_f = sc_f + 1.2 * (coh_f - 0.3 * surp_f)
        score_c = sc_c + 1.2 * (coh_c - 0.3 * surp_c)
        discriminated = score_f > score_c
        if discriminated:
            causal_correct += 1
        ok = "PASS" if discriminated else "FAIL"
        print(f"  [{ok}] Premise: '{premise[:40]}...' | Fact Score: {score_f:.3f} vs Contra: {score_c:.3f}")
    causal_acc = (causal_correct / len(ood_causal)) * 100.0

    # C. Unseen Algorithmic Syntax
    ood_algo = [
        (
            "In Python, the keyword to handle exceptions inside a try block is",
            ["except", "catch", "error", "rescue"],
            "except"
        ),
        (
            "In Python, the boolean operator that returns True only if both operands are True is",
            ["and", "or", "not", "both"],
            "and"
        ),
        (
            "A data structure operating on a Last-In-First-Out (LIFO) discipline is a",
            ["stack", "queue", "tree", "heap"],
            "stack"
        ),
        (
            "The time complexity to search an element in an unsorted list of size n is",
            ["O(n)", "O(1)", "O(log n)", "O(n^2)"],
            "O(n)"
        )
    ]
    algo_correct = 0
    print("\n[C. Unseen Algorithmic & Computer Science Generalization]")
    from ishmn.evaluation_harness import score_candidate_choices
    for prompt, choices, target in ood_algo:
        is_ok, best, scores = score_candidate_choices(model, uce, tokenizer, prompt, choices, target, device=device)
        if is_ok:
            algo_correct += 1
        ok = "PASS" if is_ok else "FAIL"
        print(f"  [{ok}] Prompt: '{prompt[:40]}...' | Target: '{target}' | Picked: '{best}'")
    algo_acc = (algo_correct / len(ood_algo)) * 100.0

    print("\n" + "=" * 75)
    print(f" OOD GENERALIZATION ACCURACY SUMMARY")
    print(f"   - Unseen Arithmetic Reasoning        : {math_acc:.1f}%")
    print(f"   - Unseen Causal Physical Sensemaking : {causal_acc:.1f}%")
    print(f"   - Unseen Algorithmic Logic & Syntax  : {algo_acc:.1f}%")
    print(f"   >>> COMPOSITE OOD GENERALIZATION     : {(math_acc + causal_acc + algo_acc) / 3.0:.2f}%")
    print("=" * 75)

    # 2. Open-Ended Natural Language Generation Showcase
    print("\n" + "=" * 75)
    print(" 2. OPEN-ENDED NATURAL LANGUAGE GENERATION & THINK-AND-SPEAK")
    print("=" * 75)

    gen_prompts = [
        "Explain why plants need carbon dioxide.\nAssistant:",
        "User: How does gravity work?\nAssistant:",
        "In Python, write a function to calculate the square of a number:\ndef square(x):",
        "Question: Why does ice melt when heated?\nAnswer:",
        "Calculate: (18 * 5) + 10 = "
    ]

    for p in gen_prompts:
        print(f"\n[Prompt]:\n{p}")
        res = uce.think_and_speak(
            prompt=p,
            max_new_tokens=30,
            temperature=0.6,
            enable_born_rule=True,
            enable_scratchpad=True,
            enable_deliberation=True
        )
        print(f"[Generated Response]:\n{res['response']}")
        print(f"[Hopfield Energy]: {res.get('hopfield_energy', 0.0):.4f} | [Deliberation]: {res.get('dual_process', {}).get('status', 'complete')}")

if __name__ == '__main__':
    test_ood_generalization()
