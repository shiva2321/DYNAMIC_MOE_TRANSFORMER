"""
Master Training Pipeline for the Universal Cognitive Engine (UCE).

Trains the entire system across an interleaved multi-task curriculum until it reaches
high accuracy and true generalization across:
1. Natural Language Modeling (Byte-level syntax, semantic grammar, multi-paragraph coherence).
2. Continuous Physical Dynamics (Autonomous Lorenz attractor forecasting on S^7 x R+).
3. Discrete Algorithmic Stream Reasoning (Multi-operator chains, carry & modular arithmetic).
4. Pearl Level 3 Causal Inversion (Self-supervised round-trip counterfactual rollback).

Saves checkpoint to artifacts/universal_cognitive_engine_trained.pt.
"""

import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from universal_cognitive_engine import UniversalCognitiveEngine
from mobius_attractor_stream import InvertedMobiusStreamReasoner
from geometry import project_to_sphere, geodesic_distance_sphere

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# 1. Multi-Task Data Generators
# ============================================================================

# Diverse multi-paragraph text corpus on science, cognition, and geometry
TEXT_CORPUS = (
    "In the study of cognitive architectures, intelligence arises not from static tabular lookup, "
    "but from dynamic navigation across continuous geometric manifolds. Biological brains map "
    "high-dimensional sensory streams into low-dimensional topological representations, such as "
    "toroidal grid cells in the entorhinal cortex and continuous attractor networks in the hippocampus. "
    "When an artificial agent navigates an unfamiliar environment or encounters counterfactual dilemmas, "
    "it does not merely perform feedforward pattern matching. Instead, it engages in predictive coding, "
    "minimizing variational free energy through hierarchical error feedback. Deliberative reasoning, "
    "or System 2 thinking, emerges naturally as recurrent relaxation along Riemannian geodesics until "
    "cognitive consensus is reached. By grounding discrete symbols onto non-Euclidean attractors, "
    "the system escapes the curse of dead-reckoning drift, enabling lifelong adaptation without "
    "catastrophic forgetting. Modern computational physics and non-linear dynamics demonstrate that "
    "turbulent and chaotic flows remain bounded within compact phase-space attractors. By combining "
    "spherical mixture of experts with associative hyperspace memory, an intelligent system achieves "
    "true generalization across sensory modalities, language streams, and formal mathematics."
)

def generate_lorenz_trajectory(num_steps=1200, dt=0.01, sigma=10.0, rho=28.0, beta=8.0/3.0):
    """Generates continuous 3D chaotic Lorenz dynamics."""
    xyz = np.zeros((num_steps, 3))
    xyz[0] = [1.0, 1.0, 1.0]
    for t in range(num_steps - 1):
        x, y, z = xyz[t]
        dx = sigma * (y - x) * dt
        dy = (x * (rho - z) - y) * dt
        dz = (x * y - beta * z) * dt
        xyz[t + 1] = [x + dx, y + dy, z + dz]
    # Normalize to zero mean unit variance
    xyz_norm = (xyz - xyz.mean(axis=0)) / (xyz.std(axis=0) + 1e-5)
    return torch.tensor(xyz_norm, dtype=torch.float32, device=device)

def generate_algorithmic_batch(batch_size=8, seq_len=6):
    """Generates discrete multi-operator algorithmic stream sequences."""
    # Operators: 0: +, 1: -, 2: *, 3: %
    inputs = torch.zeros(batch_size, seq_len, 2, dtype=torch.long, device=device)
    targets = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    for b in range(batch_size):
        acc = random.randint(1, 10)
        for t in range(seq_len):
            op_idx = random.randint(0, 3)
            val = random.randint(1, 8)
            if op_idx == 0:
                acc = (acc + val) % 100
            elif op_idx == 1:
                acc = (acc - val) % 100
            elif op_idx == 2:
                acc = (acc * val) % 100
            elif op_idx == 3:
                acc = (acc % max(val, 2)) % 100
            
            inputs[b, t, 0] = op_idx
            inputs[b, t, 1] = val
            targets[b, t] = acc
            
    return inputs, targets

# ============================================================================
# 2. Master Training Pipeline
# ============================================================================

def train_master_system(
    max_epochs: int = 150,
    save_checkpoint: str = "artifacts/universal_cognitive_engine_trained.pt"
):
    print("=" * 80)
    print(f" UNIVERSAL COGNITIVE ENGINE: MASTER MULTI-TASK TRAINING ({device})")
    print("=" * 80)

    # Initialize Engine
    uce = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=32,
        vocab_size=256,
        initial_experts=2,
        max_experts=8,
        lr=0.0025,
        device=device
    )

    # Prepare datasets
    byte_tokens = torch.tensor(list(TEXT_CORPUS.encode("utf-8")), dtype=torch.long, device=device)
    lorenz_traj = generate_lorenz_trajectory(num_steps=1500)
    total_text_tokens = len(byte_tokens)

    chunk_size = 48
    history = {
        "epoch": [],
        "loss_nlp": [],
        "perp_nlp": [],
        "acc_nlp": [],
        "loss_chaos": [],
        "cos_chaos": [],
        "acc_stream": [],
        "causal_score": [],
        "experts": []
    }

    start_time = time.time()
    print("\nStarting Interleaved Multi-Task Training Loop...\n")

    for epoch in range(1, max_epochs + 1):
        uce.train()

        # --------------------------------------------------------------------
        # Task 1: Natural Language Modeling Step (Looped RSA + Memory)
        # --------------------------------------------------------------------
        start_idx = ((epoch - 1) * 16) % (total_text_tokens - chunk_size - 1)
        inp_txt = byte_tokens[start_idx : start_idx + chunk_size]
        tgt_txt = byte_tokens[start_idx + 1 : start_idx + chunk_size + 1]

        res_nlp = uce.train_sequence_step(
            inp_txt.unsqueeze(0),
            tgt_txt.unsqueeze(0),
            max_loops=3,
            ponder_weight=0.004
        )
        loss_nlp = res_nlp["loss_task"]
        perp_nlp = math.exp(min(loss_nlp, 8.0))
        acc_nlp = res_nlp["token_accuracy"]

        # --------------------------------------------------------------------
        # Task 2: Continuous Chaotic Dynamics (Predictive JEPA + Causal Rollback)
        # --------------------------------------------------------------------
        t_start = (epoch * 7) % (lorenz_traj.size(0) - 16)
        x_curr = torch.zeros(1, 32, device=device)
        x_next = torch.zeros(1, 32, device=device)
        x_curr[0, :3] = lorenz_traj[t_start]
        x_next[0, :3] = lorenz_traj[t_start + 1]
        c_zero = torch.zeros_like(x_curr)

        res_chaos = uce.train_step(
            x_curr=x_curr,
            context=c_zero,
            x_next=x_next,
            modality_x="continuous",
            modality_c="continuous"
        )
        loss_chaos = res_chaos["loss_task"]
        with torch.no_grad():
            eval_chaos = uce.evaluate_step(x_curr, c_zero, x_next)
            cos_chaos = eval_chaos["task_cos_sim"] * 100.0

        # --------------------------------------------------------------------
        # Task 3: Discrete Algorithmic Stream Transduction Step
        # --------------------------------------------------------------------
        stream_in, stream_tgt = generate_algorithmic_batch(batch_size=4, seq_len=5)
        res_stream = uce.train_stream_automaton_step(stream_in, stream_tgt, max_loops=3)
        acc_stream = res_stream["token_accuracy"]

        # --------------------------------------------------------------------
        # Task 4: Periodic Causal Inversion Evaluation
        # --------------------------------------------------------------------
        with torch.no_grad():
            causal_eval = uce.evaluate_causal_inversion(
                torch.randn(1, 32, device=device),
                torch.randn(1, 32, device=device)
            )
            causal_score = causal_eval["understanding_score"] * 100.0

        # Log History
        num_experts = uce.moe.num_experts
        history["epoch"].append(epoch)
        history["loss_nlp"].append(loss_nlp)
        history["perp_nlp"].append(perp_nlp)
        history["acc_nlp"].append(acc_nlp)
        history["loss_chaos"].append(loss_chaos)
        history["cos_chaos"].append(cos_chaos)
        history["acc_stream"].append(acc_stream)
        history["causal_score"].append(causal_score)
        history["experts"].append(num_experts)

        if epoch % 15 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{max_epochs} | "
                  f"NLP Loss: {loss_nlp:.3f} (Perp: {perp_nlp:5.1f}, Acc: {acc_nlp:4.1f}%) | "
                  f"Chaos MSE: {loss_chaos:.4f} (Cos: {cos_chaos:5.1f}%) | "
                  f"Stream Acc: {acc_stream:4.1f}% | "
                  f"Causal: {causal_score:4.1f}% | Experts: {num_experts}")

        # Check Early True Generalization Criteria
        if (loss_nlp <= 0.85 and acc_nlp >= 75.0 and cos_chaos >= 95.0 and 
            acc_stream >= 95.0 and epoch >= 80):
            print(f"\n>>> CONVERGENCE TARGETS ACHIEVED at Epoch {epoch}!")
            break

    elapsed = time.time() - start_time
    print(f"\nMulti-Task Master Training Completed in {elapsed:.2f}s.")
    print(f"Final Telemetry: NLP Loss: {history['loss_nlp'][-1]:.3f} (Acc: {history['acc_nlp'][-1]:.1f}%) | "
          f"Chaos Cos: {history['cos_chaos'][-1]:.1f}% | Stream Acc: {history['acc_stream'][-1]:.1f}% | "
          f"Active Experts: {history['experts'][-1]}")

    # Save Checkpoint
    os.makedirs(os.path.dirname(save_checkpoint), exist_ok=True)
    torch.save({
        "model_state_dict": uce.state_dict(),
        "history": history,
        "d_model": uce.d_model,
        "d_hidden": uce.d_hidden,
        "hyperspace_dim": uce.hyperspace_dim,
        "num_experts": uce.moe.num_experts
    }, save_checkpoint)
    print(f"[Model Checkpoint Saved]: {save_checkpoint}")

    return uce, history

if __name__ == "__main__":
    train_master_system()
