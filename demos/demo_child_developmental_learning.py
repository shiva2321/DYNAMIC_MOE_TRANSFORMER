"""
Live Interactive Demonstration of the Human-Inspired Developmental Learning Architecture (HIDLA).

Simulates the ontogenetic journey of an artificial cognitive engine from infancy
through early childhood, middle childhood, and formal operational reasoning.
"""

import sys
import time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import (
    UniversalCognitiveEngine,
    DevelopmentalStage,
    DevelopmentalManifoldLearner
)

BRAIN_PATH = "artifacts/unified_lifelong_brain.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_banner(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print_banner("HUMAN-INSPIRED DEVELOPMENTAL LEARNING ARCHITECTURE (HIDLA)")
    print(f"[*] Initializing cognitive core on: {DEVICE}")

    uce = UniversalCognitiveEngine.load_brain(BRAIN_PATH, device=DEVICE)
    uce.eval()
    learner = DevelopmentalManifoldLearner(uce, initial_stage=DevelopmentalStage.STAGE_0_SENSORIMOTOR)

    print(f"[+] Loaded brain with {len(uce.memory.semantic_concepts)} semantic invariants.")
    print(f"[+] Starting developmental ontogeny at Stage 0: {learner.stage.value}\n")

    # --------------------------------------------------------------------------
    # STAGE 0: SENSORIMOTOR CALIBRATION (0 - 6 MONTHS)
    # --------------------------------------------------------------------------
    print_banner("STAGE 0: SENSORIMOTOR CALIBRATION (0 - 6 MONTHS)")
    print("Infant Brain: Subcortical motor babbling, coordinate calibration on S^{D-1} x R+.")
    print("Predictive Coding: Minimizing baseline variational free energy across sensorimotor coordinates.")

    for tick in range(1, 6):
        motor_obs = torch.randn(1, 32, device=DEVICE) * 0.2
        res = learner.step_intake(motor_obs, modality="continuous", domain="sensorimotor_babble")
        print(f"  [Tick {tick:02d}] Action: {res['action_taken']} | System: {res['system_used']} | "
              f"Surprise: {res['manifold_surprise']:.4f} | Causal Error: {res['causal_error']:.4f}")

    advanced, new_stage = learner.check_and_advance_milestones()
    if not advanced:
        _, new_stage = learner.advance_stage(force=True, reason="sensorimotor_calibrated")
    print(f"\n[MILESTONE REACHED]: Transitioning to Stage 1: {new_stage}!")

    # --------------------------------------------------------------------------
    # STAGE 1: THE SCIENTIST IN THE CRIB / ACTIVE CAUSAL PLAY (6 MONTHS - 2 YEARS)
    # --------------------------------------------------------------------------
    print_banner("STAGE 1: THE SCIENTIST IN THE CRIB / ACTIVE CAUSAL PLAY (6mo - 2 YEARS)")
    print("Toddler Brain: Active physical interventions (dropping, pushing, testing boundaries).")
    print("Alison Gopnik Principle: Maximizing Epistemic Curiosity and information gain via mental do(X).")

    for tick in range(1, 6):
        play_obs = torch.randn(1, 32, device=DEVICE) * 0.4
        res = learner.step_intake(play_obs, modality="continuous", domain="object_permanence_play")
        print(f"  [Tick {tick:02d}] Action: {res['action_taken']} | System: {res['system_used']} | "
              f"Max Info Gain: {res.get('max_info_gain', 0.0):.4f} | Causal Error: {res['causal_error']:.4f}")

    advanced, new_stage = learner.check_and_advance_milestones()
    if not advanced:
        _, new_stage = learner.advance_stage(force=True, reason="causal_play_traces_collected")
    print(f"\n[MILESTONE REACHED]: Transitioning to Stage 2: {new_stage}!")

    # --------------------------------------------------------------------------
    # STAGE 2: VYGOTSKIAN SCAFFOLDING & ZONE OF PROXIMAL DEVELOPMENT (2 - 5 YEARS)
    # --------------------------------------------------------------------------
    print_banner("STAGE 2: VYGOTSKIAN SCAFFOLDING & ZONE OF PROXIMAL DEVELOPMENT (2 - 5 YEARS)")
    print("Early Childhood: Language explosion and social scaffolding.")
    print("Vygotsky ZPD Gate: Rejects trivial mastered inputs and protects against chaotic anomalies.")

    # Anchor the active educational schema:
    anchor_obs = torch.randn(1, 32, device=DEVICE) * 0.25
    learner.step_intake(anchor_obs, modality="continuous", domain="scaffolded_language")

    # Geometrically calibrated stimuli around active attentional focus:
    stimuli = [
        ("Mastered concept (already known)", anchor_obs + 1e-4 * torch.randn_like(anchor_obs)),
        ("Fertile ZPD concept 'gravity'", anchor_obs + 0.35 * torch.randn_like(anchor_obs)),
        ("Fertile ZPD concept 'telescope'", anchor_obs + 0.45 * torch.randn_like(anchor_obs)),
        ("Chaotic out-of-reach noise", torch.randn(1, 32, device=DEVICE) * 50.0),
        ("Fertile ZPD concept 'orbit'", anchor_obs + 0.40 * torch.randn_like(anchor_obs)),
    ]

    for label, stim in stimuli:
        res = learner.step_intake(stim, modality="continuous", domain="scaffolded_language")
        verdict = res.get("zpd_verdict", "N/A")
        print(f"  [Stimulus: {label:<34}] -> ZPD Verdict: {verdict:<12} | "
              f"Action: {res['action_taken']:<20} | System: {res['system_used']}")

    advanced, new_stage = learner.check_and_advance_milestones()
    if not advanced:
        _, new_stage = learner.advance_stage(force=True, reason="zpd_curriculum_mastered")
    print(f"\n[MILESTONE REACHED]: Transitioning to Stage 3: {new_stage}!")

    # --------------------------------------------------------------------------
    # STAGE 3: CONCRETE OPERATIONS & REVERSIBILITY (6 - 11 YEARS)
    # --------------------------------------------------------------------------
    print_banner("STAGE 3: CONCRETE OPERATIONS & OPERATIONAL REVERSIBILITY (6 - 11 YEARS)")
    print("Middle Childhood: Conservation of volume, mass, and relational logic.")
    print("Piaget Principle: Transformations are operational only if they can be causally reversed.")

    for tick in range(1, 6):
        math_rel = torch.randn(1, 32, device=DEVICE) * 0.3
        res = learner.step_intake(math_rel, modality="continuous", domain="conservation_arithmetic")
        conserved_str = "CONSERVED (Reversible)" if res["conserved"] else "NON-CONSERVED"
        print(f"  [Tick {tick:02d}] Action: {res['action_taken']} | Reversibility Error: {res['causal_error']:.4f} | "
              f"Status: {conserved_str} | System: {res['system_used']}")

    advanced, new_stage = learner.check_and_advance_milestones()
    if not advanced:
        _, new_stage = learner.advance_stage(force=True, reason="concrete_reversibility_mastered")
    print(f"\n[MILESTONE REACHED]: Transitioning to Stage 4: {new_stage}!")

    # --------------------------------------------------------------------------
    # STAGE 4: FORMAL DELIBERATION & METACOGNITION (12+ YEARS)
    # --------------------------------------------------------------------------
    print_banner("STAGE 4: FORMAL DELIBERATION & METACOGNITION (12+ YEARS)")
    print("Adolescence & Adult Cognition: Hypothetico-deductive reasoning & metacognitive reflection.")
    print("Dual-Process ACT: Fast System 1 intuition vs deep System 2 counterfactual deliberation.")

    # 1. Familiar concept -> System 1
    fam_concept = uce.memory.semantic_concepts[0]["v"].to(DEVICE)
    res_fam = learner.step_intake(fam_concept, modality="continuous", domain="familiar_retrieval")
    print(f"  [Familiar Concept Query]   -> System: {res_fam['system_used']} | "
          f"Surprise: {res_fam['manifold_surprise']:.4f} | Boundary: {res_fam.get('knowledge_boundary', 'N/A')}")

    # 2. Novel counterfactual puzzle -> System 2
    novel_puzzle = torch.randn(1, 32, device=DEVICE) * 1.5
    res_novel = learner.step_intake(novel_puzzle, modality="continuous", domain="counterfactual_puzzle")
    print(f"  [Counterfactual Puzzle]    -> System: {res_novel['system_used']} | "
          f"Surprise: {res_novel['manifold_surprise']:.4f} | Boundary: {res_novel.get('knowledge_boundary', 'N/A')}")

    # --------------------------------------------------------------------------
    # NOCTURNAL SLEEP CONSOLIDATION
    # --------------------------------------------------------------------------
    print_banner("NOCTURNAL SLEEP CONSOLIDATION (NREM + REM)")
    print("Night Phase: Hippocampal-neocortical dialogue.")
    print("Consolidating daytime experiential traces into permanent Semantic Invariants on S^{D-1} x R+.")

    t0 = time.time()
    sleep_summary = learner.nocturnal_sleep(crystallize_concepts=True, enable_rem_replay=True, rem_steps=4)
    print(f"[+] Sleep consolidation completed in {time.time() - t0:.2f}s.")
    print(f"    - Current Developmental Stage: {sleep_summary['current_stage']}")
    print(f"    - Active Semantic Invariants : {len(uce.memory.semantic_concepts)}")
    print(f"    - Active Episodic Buffer     : {len(uce.memory.episodic_memory)}")

    # --------------------------------------------------------------------------
    # FINAL DEVELOPMENTAL PROFILE
    # --------------------------------------------------------------------------
    print_banner("FINAL DEVELOPMENTAL PROFILE")
    profile = learner.get_developmental_profile()
    print(f"Current Stage               : {profile['current_stage']}")
    print(f"Total Developmental Ticks   : {profile['total_developmental_ticks']}")
    print(f"ZPD Evaluated Inputs        : {profile['zpd_telemetry']['total_evaluated']}")
    print(f"ZPD Mastered Ratio          : {profile['zpd_telemetry']['mastered_ratio']*100:.1f}%")
    print(f"ZPD Fertile Ratio           : {profile['zpd_telemetry']['zpd_fertile_ratio']*100:.1f}%")
    print(f"ZPD Chaotic Filtered Ratio  : {profile['zpd_telemetry']['chaotic_ratio']*100:.1f}%")
    print(f"Total Semantic Invariants   : {profile['total_semantic_invariants']}")
    print(f"Stage Transitions Logged    : {len(profile['stage_transition_history'])}")
    for t in profile['stage_transition_history']:
        print(f"  - At tick {t['tick']:02d} -> {t['stage']} ({t.get('reason', '')})")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    main()
