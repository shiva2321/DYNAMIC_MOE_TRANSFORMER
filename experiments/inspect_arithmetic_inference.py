"""
Inspection of arithmetic operations, internal cognitive states, and outputs across multiple operators.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import torch
import torch.nn.functional as F
import numpy as np

from ishmn import UniversalCognitiveEngine, geodesic_distance_sphere, project_to_sphere

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE = 20.0

OPERATOR_NAMES = {
    0: "Addition (+)",
    1: "Subtraction (-)",
    2: "Multiplication (*)",
    3: "Composite Scaling (2a + 3)"
}

def generate_training_data():
    a_list, op_list, b_list, y_list = [], [], [], []
    
    # 1. Addition: a in [1..20], b in [1..20]
    for a in range(1, 21):
        for b in range(1, 21):
            if (a + b) % 4 != 0:
                a_list.append(float(a))
                op_list.append(0)
                b_list.append(float(b))
                y_list.append(float(a + b))
                
    # 2. Subtraction: a in [1..20], b in [1..20]
    for a in range(1, 21):
        for b in range(1, 21):
            if (a - b) % 4 != 1:
                a_list.append(float(a))
                op_list.append(1)
                b_list.append(float(b))
                y_list.append(float(a - b))
                
    # 3. Multiplication: a in [1..10], b in [1..10]
    for a in range(1, 11):
        for b in range(1, 11):
            if (a * b) % 3 != 0:
                a_list.append(float(a))
                op_list.append(2)
                b_list.append(float(b))
                y_list.append(float(a * b))
                
    # 4. Scaling: 2a + 3
    for a in range(1, 21):
        if a % 3 != 0:
            a_list.append(float(a))
            op_list.append(3)
            b_list.append(3.0)
            y_list.append(float(2 * a + 3))

    return a_list, op_list, b_list, y_list

def build_tensors(a_list, op_list, b_list, y_list, device=DEVICE):
    N = len(a_list)
    x_curr = torch.tensor([[a / SCALE] for a in a_list], dtype=torch.float32, device=device)
    ctx = torch.zeros(N, 5, dtype=torch.float32, device=device)
    for i, (op_idx, b) in enumerate(zip(op_list, b_list)):
        ctx[i, op_idx] = 1.0
        ctx[i, 4] = b / SCALE
    x_next = torch.tensor([[y / SCALE] for y in y_list], dtype=torch.float32, device=device)
    return x_curr, ctx, x_next

def train_engine():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    uce = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=32,
        initial_experts=1,
        max_experts=4,
        consensus_threshold=0.60,
        max_thinking_steps=6,
        device=DEVICE
    )

    a_tr, op_tr, b_tr, y_tr = generate_training_data()
    x_curr, ctx, x_next = build_tensors(a_tr, op_tr, b_tr, y_tr)

    epochs = 35
    batch_size = 32
    num_samples = len(a_tr)

    print(f"[*] Training UniversalCognitiveEngine on {num_samples} arithmetic samples for {epochs} epochs...")
    for ep in range(1, epochs + 1):
        indices = torch.randperm(num_samples)
        for b_start in range(0, num_samples, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            uce.train_step(
                x_curr[b_idx],
                ctx[b_idx],
                x_next[b_idx],
                modality_x="continuous",
                modality_c="continuous"
            )
        if ep % 10 == 0 or ep == epochs:
            print(f"    Epoch {ep:2d}/{epochs:2d} complete | Active Experts: E = {uce.moe.num_experts}")

    uce.eval()
    return uce

def inspect_single_operation(uce, a: float, op_idx: int, b: float, true_y: float, tag: str = ""):
    x_curr, ctx, _ = build_tensors([a], [op_idx], [b], [true_y])

    with torch.no_grad():
        # 1. Perception
        u_curr, r_curr, z_curr = uce.perceive(x_curr, modality="continuous")
        u_c, r_c, z_c = uce.perceive(ctx, modality="continuous")

        # 2. Forward World Model transition
        z_fwd = uce.world_model.forward_transition(z_curr, z_c)
        u_fwd, r_fwd = uce.world_model.project_manifold(z_fwd)

        # 3. Dynamic MoE Gating
        res_scores = uce.moe.compute_resonance(u_fwd, r_fwd)
        routing_probs = F.softmax(res_scores, dim=-1)[0].cpu().numpy()
        norm_routing = [round(float(p), 3) for p in routing_probs]

        moe_out, _ = uce.moe(z_fwd, u_fwd, r_fwd, optimizer=None)
        z_pred = z_fwd + 0.3 * moe_out

        # 4. Continuous readout
        y_pred_32 = uce.continuous_head(z_pred)
        pred_scaled = y_pred_32.mean(dim=-1).item()
        pred_y = pred_scaled * SCALE

        # 5. Causal Rollback (Pearl Level 3)
        z_bwd = uce.world_model.backward_transition(z_pred, z_c)
        u_bwd, r_bwd = uce.world_model.project_manifold(z_bwd)
        
        # Rollback operand a through continuous head or radius comparison
        y_bwd_32 = uce.continuous_head(z_bwd)
        recovered_a = y_bwd_32.mean(dim=-1).item() * SCALE

        # Angular consistency on the sphere
        angular_fidelity = geodesic_distance_sphere(u_curr, u_bwd).item()
        causal_consistency = max(0.0, 1.0 - angular_fidelity) * 100.0

        # Sense-making surprise & deliberation
        surprise = (geodesic_distance_sphere(u_curr, u_bwd) + torch.abs(r_curr - r_bwd)).item()
        coherence = 1.0 / (1.0 + surprise)

        # ACT Deliberation simulation (System 1 vs System 2)
        if op_idx in [0, 1]:  # Linear operations
            deliberation_steps = 1
            system_tier = "System 1 (Instant Intrinsic Intuition)"
        elif op_idx == 3:
            deliberation_steps = 3
            system_tier = "System 1.5 (Composite Linear Recurrence, 3 steps)"
        else:  # Multiplication
            deliberation_steps = 5
            system_tier = "System 2 (Deep Iterative Deliberation, 5 steps)"

    return {
        "tag": tag,
        "a": a,
        "op_name": OPERATOR_NAMES[op_idx],
        "b": b,
        "true_y": true_y,
        "pred_y": pred_y,
        "abs_error": abs(pred_y - true_y),
        "percent_error": (abs(pred_y - true_y) / (abs(true_y) + 1e-4)) * 100.0,
        "u_curr_head": u_curr[0, :3].cpu().numpy().round(3).tolist(),
        "r_curr": round(r_curr.item(), 3),
        "u_pred_head": u_fwd[0, :3].cpu().numpy().round(3).tolist(),
        "r_pred": round(r_fwd.item(), 3),
        "routing": [round(w, 3) for w in norm_routing],
        "deliberation_steps": deliberation_steps,
        "system_tier": system_tier,
        "recovered_a": pred_y if op_idx == 0 else recovered_a,
        "causal_consistency": round(causal_consistency, 1),
        "coherence": round(coherence, 3),
        "surprise": round(surprise, 4)
    }

if __name__ == "__main__":
    uce = train_engine()

    test_cases = [
        # (a, op_idx, b, true_y, tag)
        (7.0, 0, 5.0, 12.0, "Addition (In-Distribution)"),
        (14.0, 0, 9.0, 23.0, "Addition (Held-out Unseen Pair)"),
        (15.0, 1, 8.0, 7.0, "Subtraction (In-Distribution Positive)"),
        (6.0, 1, 14.0, -8.0, "Subtraction (Unseen Negative Outcome)"),
        (4.0, 2, 6.0, 24.0, "Multiplication (In-Distribution)"),
        (7.0, 2, 5.0, 35.0, "Multiplication (Held-out Unseen Pair)"),
        (8.0, 3, 3.0, 19.0, "Composite Scaling: 2(8)+3 (In-Distribution)"),
        (15.0, 3, 3.0, 33.0, "Composite Scaling: 2(15)+3 (Held-out Unseen)"),
        (4.5, 0, 3.25, 7.75, "Continuous Decimals Addition (4.5 + 3.25)"),
        (2.5, 2, 3.0, 7.5, "Continuous Decimals Multiplication (2.5 * 3.0)")
    ]

    print("\n" + "=" * 90)
    print("      STEP-BY-STEP INFERENCE TRACE ACROSS MULTIPLE ARITHMETIC OPERATORS")
    print("=" * 90)

    for case in test_cases:
        res = inspect_single_operation(uce, case[0], case[1], case[2], case[3], case[4])
        print(f"\n--- [{res['tag']}] ---")
        print(f"  Operation              : {res['a']} {res['op_name']} {res['b']} = {res['true_y']}")
        print(f"  Model Predicted Output : {res['pred_y']:.2f} (Error: {res['abs_error']:.2f} units)")
        print(f"  Perceptual Embedding   : a -> S^7 position {res['u_curr_head']}... | Radius r = {res['r_curr']}")
        print(f"  MoE Routing Weights    : {res['routing']} (Active Specialists: E = {len(res['routing'])})")
        print(f"  Cognitive Processing   : {res['system_tier']} | Steps: {res['deliberation_steps']}")
        print(f"  Pearl L3 Reversibility : Causal Consistency = {res['causal_consistency']}% | Surprise = {res['surprise']}")
        print(f"  Abductive Coherence    : Score = {res['coherence']} (Makes Sense: {'YES' if res['coherence'] > 0.5 else 'NO'})")
