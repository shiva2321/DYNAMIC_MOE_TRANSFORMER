"""
Crystallize Scientific Invariants into UCE Neocortical Lattice.

Embeds 20+ foundational physical, astronomical, thermodynamic, and biological invariants
directly into UCE's permanent Semantic Concept Lattice (S^7 x R+), audits Gate 3 zero-forgetting,
and updates the SLM-Manifold Cross-Space Adapter Bridge.
"""

import os
import sys
import time
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.universal_cognitive_engine import UniversalCognitiveEngine
from ishmn.geometry import project_to_sphere, geodesic_distance_sphere
from ishmn.data import SCIENTIFIC_INVARIANTS
from experiments.train_slm_manifold_adapter import train_adapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"


def crystallize_invariants():
    print("=" * 80)
    print(" CRYSTALLIZING NEOCORTICAL SCIENTIFIC INVARIANTS")
    print("=" * 80)
    print(f"[*] Device: {DEVICE}")

    # 1. Load UCE Brain Checkpoint
    print(f"[*] Loading Brain from: {BRAIN_PATH}...")
    uce = UniversalCognitiveEngine.load_brain(BRAIN_PATH, device=DEVICE)
    uce.eval()
    initial_count = len(uce.memory.semantic_concepts)
    print(f"[+] Loaded brain with {initial_count} semantic invariants.")

    # 2. Crystallize each scientific invariant
    print(f"[*] Embedding {len(SCIENTIFIC_INVARIANTS)} scientific invariant principles...")
    existing_words = {sc.get("word"): i for i, sc in enumerate(uce.memory.semantic_concepts)}
    added = 0
    updated = 0

    with torch.no_grad():
        for inv in SCIENTIFIC_INVARIANTS:
            word = inv["word"]
            desc = inv["description"]
            domain = inv["domain"]

            # Perceive description text into manifold representation
            u_perc, r_perc, v_perc = uce.perceive(desc, modality="subwords")
            u_clean = project_to_sphere(u_perc.cpu())
            r_clean = float(r_perc.mean().item())
            v_clean = v_perc.cpu()

            concept_entry = {
                "concept_id": len(uce.memory.semantic_concepts),
                "word": word,
                "u": u_clean,
                "r": r_clean,
                "v": v_clean,
                "domain": domain,
                "exemplar_count": 50,
                "metadata": {
                    "description": desc,
                    "keywords": inv["keywords"],
                    "is_scientific_invariant": True
                }
            }

            if word in existing_words:
                idx = existing_words[word]
                uce.memory.semantic_concepts[idx] = concept_entry
                updated += 1
            else:
                uce.memory.semantic_concepts.append(concept_entry)
                existing_words[word] = len(uce.memory.semantic_concepts) - 1
                added += 1

    # Invalidate cache so queries immediately use updated concepts
    uce.memory._sem_cache_u = None
    uce.memory._sem_cache_v = None

    print(f"[+] Crystallization complete: {added} added, {updated} updated.")
    print(f"    - Total Semantic Lattice Invariants: {len(uce.memory.semantic_concepts)}")

    # 3. Audit Zero Catastrophic Forgetting (Gate 3)
    print(f"\n[*] Auditing Gate 3 Zero-Forgetting...")
    math_checks = [
        ("12 + 7", 19.0),
        ("23164 - 45", 23119.0),
        ("23146484561521 + 45132030", 23146529693551.0)
    ]
    for expr, exp_val in math_checks:
        res = uce.solve_with_dual_process(expr, enable_system2=True)
        assert res["system2_exact_result"] == exp_val, f"Failed on {expr}"
        print(f"  [+] Gate 3 Check Passed: {expr} = {exp_val}")

    # 4. Save Updated Brain
    uce.save_brain(BRAIN_PATH)
    print(f"[+] Saved updated brain to: {BRAIN_PATH}")

    # 5. Re-train Adapter Bridge to align Qwen2.5 with new invariants
    print(f"\n[*] Fine-aligning SLM-Manifold Adapter Bridge...")
    train_adapter()

    print("\n" + "=" * 80)
    print(" NEOCORTICAL INVARIANT CRYSTALLIZATION COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    crystallize_invariants()
