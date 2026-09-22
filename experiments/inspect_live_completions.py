"""
Inspection script to evaluate live qualitative completions from the trained Dual-Cortex Brain.
Checkpoint: artifacts/checkpoints/native_moe_micro_latest.pt (Step 6958)
"""
import torch
from ishmn import UniversalCognitiveEngine
from ishmn.native_cognitive_transformer import NativeSphericalTransformer
from ishmn.data_loader import SovereignTokenizer
from ishmn.training_engine import MODEL_PROFILES

def inspect_completions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing model on device: {device}")
    
    cfg = MODEL_PROFILES["micro"]
    tokenizer = SovereignTokenizer("gpt2")
    
    model = NativeSphericalTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        hyperspace_dim=cfg["hyperspace_dim"],
        d_ff=cfg["d_ff"],
        max_seq_len=96,
        mtp_future_tokens=cfg["mtp_future_tokens"],
        vsa_capacity=cfg["vsa_capacity"],
        num_experts=cfg["num_experts"],
        top_k=cfg["top_k"],
        max_experts=cfg["num_experts"],
        enable_shared_expert=cfg["enable_shared_expert"],
        enable_aux_loss_free_balancing=cfg["enable_aux_loss_free_balancing"],
        enable_vsa_memory=True,
        enable_world_model=False,
        tie_weights=True,
        device=device
    ).to(device)

    uce = UniversalCognitiveEngine(
        d_model=cfg["d_model"],
        d_hidden=cfg["d_ff"],
        hyperspace_dim=cfg["hyperspace_dim"],
        num_memory_slots=32,
        vocab_size=tokenizer.vocab_size,
        initial_experts=cfg["num_experts"],
        device=device
    ).to(device)
    uce.attach_native_transformer(model, tokenizer=tokenizer)

    ckpt_path = "artifacts/checkpoints/native_moe_micro_latest.pt"
    print(f"[*] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    uce.load_state_dict(ckpt["uce_state_dict"], strict=False)
    
    # Ensure scientific invariants are seeded
    num_seeded = uce.seed_scientific_invariants()
    print(f"[+] Active Hopfield Attractor Count: {len(uce.hopfield)} (Seeded: {num_seeded})")

    model.eval()
    uce.eval()

    test_prompts = [
        # Domain 1: Physics & Physical Invariants
        {"domain": "Physics", "prompt": "The fundamental law of conservation of energy states that", "max_toks": 35},
        {"domain": "Physics", "prompt": "Newton's third law of motion dictates that for every action", "max_toks": 35},
        {"domain": "Physics", "prompt": "In thermodynamics, entropy is a measure of", "max_toks": 35},
        
        # Domain 2: Computer Science & Data Structures
        {"domain": "Computer Science", "prompt": "In Python programming, the main difference between a list and a tuple is that", "max_toks": 35},
        {"domain": "Computer Science", "prompt": "In relational databases, the SQL clause used to eliminate duplicate rows is", "max_toks": 25},
        {"domain": "Computer Science", "prompt": "A binary search algorithm has a time complexity of", "max_toks": 30},
        
        # Domain 3: Biology & Natural Science
        {"domain": "Biology", "prompt": "The cellular organelle known as the powerhouse of the cell is the", "max_toks": 25},
        {"domain": "Chemistry", "prompt": "The primary chemical compound found in natural gas is", "max_toks": 25},
        
        # Domain 4: Mathematical & Symbolic Reasoning
        {"domain": "Mathematics", "prompt": "Calculate: (15 * 4) + 20 =", "max_toks": 15},
        {"domain": "Mathematics", "prompt": "Calculate: 125 / 5 =", "max_toks": 15},
    ]

    print("\n" + "=" * 90)
    print(" LIVE DUAL-CORTEX QUALITATIVE INSPECTION (Step 6,958 Checkpoint)")
    print("=" * 90)

    for i, item in enumerate(test_prompts, 1):
        domain = item["domain"]
        prompt = item["prompt"]
        max_toks = item["max_toks"]

        # Run think_and_speak with deliberation and System 2 AST enabled
        with torch.no_grad():
            res = uce.think_and_speak(
                prompt,
                max_new_tokens=max_toks,
                temperature=0.3,
                enable_scratchpad=True,
                enable_deliberation=True,
                write_to_vsa=False
            )

        full_text = res.get("response", "")
        gen_text = full_text[len(prompt):].strip()
        dual = res.get("dual_process") or {}
        sys2_res = dual.get("system2_exact_result")
        chosen_route = dual.get("chosen_route", "system1_neural")

        # Hopfield Attractor Information
        p_tensor = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        _, _, v_p = uce.perceive(p_tensor, modality="tokens")
        hop = uce.hopfield.snap(v_p)

        print(f"\n--- [Test {i}/10] Domain: {domain.upper()} ---")
        print(f"  Prompt     : \"{prompt}\"")
        print(f"  Completion : \"{gen_text}\"")
        print(f"  Attractor  : {hop.best_concept} (Confidence: {hop.confidence:.3f}, Energy: {hop.energy:.3f})")
        if sys2_res is not None:
            print(f"  System 2   : Exact Symbolic Result = {sys2_res} (Route: {chosen_route})")

    print("\n" + "=" * 90)
    print(" INSPECTION COMPLETE")
    print("=" * 90)

if __name__ == "__main__":
    inspect_completions()
