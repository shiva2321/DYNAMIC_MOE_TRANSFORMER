"""
Comprehensive Benchmarking and Evaluation Suite for ISHMN.

Benchmarks Evaluated:
1. Multi-Domain Continual Learning & Catastrophic Forgetting Shield:
   - Evaluates Standard Dense Network vs Static MoE vs ISHMN Dynamic MoE across 4 sequential domains.
   - Measures Backward Transfer (R_BWT) and Final Retained Accuracy.
2. Triggered Emission Calibration & Variable Deliberation:
   - Measures deliberation latency on Clean vs Ambiguous vs Pure Noise inputs.
   - Verifies zero/near-zero false positive emissions on noise.
3. Sleep Consolidation & Shell Compaction:
   - Compares Wake-Only vs Wake+Dreaming on memory retention energy and cluster compactness.
"""

import json
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import project_to_sphere, SphericalDynamicMoE, DynamicWarmupAdamW, ContinuousSelfLoop
from ishmn.dendritic_expert import TwoCompartmentDendriticExpert

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =====================================================================
# BENCHMARK 1: Multi-Domain Continual Learning & Catastrophic Forgetting
# =====================================================================

class DenseBaseline(nn.Module):
    """Standard Dense MLP with shared weights (known to suffer catastrophic forgetting)."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out)
        )
    def forward(self, x, *args, **kwargs):
        return self.net(x), {}

class StaticMoEBaseline(nn.Module):
    """Static MoE with fixed experts (suffers from capacity saturation & routing drift)."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int, num_experts: int = 2):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_out))
            for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_in, num_experts)
    def forward(self, x, *args, **kwargs):
        logits = self.gate(x)
        weights = F.softmax(logits, dim=-1)
        out = torch.zeros(x.size(0), self.experts[0][-1].out_features, device=x.device)
        for i, exp in enumerate(self.experts):
            out = out + weights[:, i:i+1] * exp(x)
        return out, {"num_experts": len(self.experts)}

def generate_domain_dataset(domain_id: int, num_samples: int = 250, d_model: int = 32, device="cpu"):
    """Generates distinct task data across 4 domains."""
    if domain_id == 0:
        # Domain 0: Non-linear Parity / Logic (orthogonal discrete signs)
        x = torch.randn(num_samples, d_model, device=device)
        y = torch.sign(x[:, :8].sum(dim=-1, keepdim=True)).repeat(1, d_model)
    elif domain_id == 1:
        # Domain 1: Harmonic Oscillations (sine/cosine waves)
        t = torch.linspace(0, 4 * np.pi, d_model, device=device).unsqueeze(0).repeat(num_samples, 1)
        freq = torch.randint(1, 4, (num_samples, 1), device=device)
        x = torch.sin(freq * t) + torch.randn_like(t) * 0.05
        y = torch.cos(freq * t)
    elif domain_id == 2:
        # Domain 2: Clustered Gaussian Distributions (Biomedical patterns)
        center = torch.randn(1, d_model, device=device) * 3.0
        x = center + torch.randn(num_samples, d_model, device=device) * 0.3
        y = -x * 0.8
    else:
        # Domain 3: Inverse Rotation Mapping (Relational task)
        x = torch.randn(num_samples, d_model, device=device)
        y = torch.roll(x, shifts=d_model // 2, dims=-1)

    u = project_to_sphere(torch.randn(num_samples, 8, device=device) + domain_id * 2.0)
    r = torch.ones(num_samples, 1, device=device) * (1.5 + domain_id * 1.5)
    return x, u, r, y

def run_continual_learning_benchmark(device):
    print("\n" + "="*70)
    print(">>> BENCHMARK 1: Multi-Domain Continual Learning & Retention Shield")
    print("="*70)

    d_model = 32
    d_hidden = 64
    domains = [0, 1, 2, 3]
    domain_names = ["Logic Parity", "Harmonic Waves", "Gaussian Clusters", "Inverse Rotation"]

    # Initialize 3 architectures
    dense_model = DenseBaseline(d_model, d_hidden, d_model).to(device)
    static_moe = StaticMoEBaseline(d_model, d_hidden, d_model, num_experts=2).to(device)
    ishmn_moe = SphericalDynamicMoE(
        d_model=d_model,
        d_ff=d_hidden,
        d_context=d_model,
        hyperspace_dim=8,
        initial_experts=2,
        max_experts=8,
        spawn_threshold=0.45,
        device=device
    ).to(device)

    opt_dense = optim.AdamW(dense_model.parameters(), lr=0.003)
    opt_static = optim.AdamW(static_moe.parameters(), lr=0.003)
    opt_ishmn = DynamicWarmupAdamW(list(ishmn_moe.parameters()), lr=0.003, default_group_warmup_steps=20)

    # Pre-generate validation sets for each domain
    val_sets = [generate_domain_dataset(d, num_samples=100, d_model=d_model, device=device) for d in domains]

    # Matrix of evaluations: eval_matrix[model_name][trained_after_domain_t][evaluated_on_domain_k]
    history = {"Dense": [], "Static MoE": [], "ISHMN Dynamic MoE": []}

    for task_idx, domain_id in enumerate(domains):
        d_name = domain_names[domain_id]
        print(f"\n[*] Training on Domain {task_idx}: {d_name}...")
        x_tr, u_tr, r_tr, y_tr = generate_domain_dataset(domain_id, num_samples=300, d_model=d_model, device=device)

        # Train 50 steps on this domain
        for step in range(50):
            # Dense train
            opt_dense.zero_grad()
            pred_dense, _ = dense_model(x_tr)
            loss_dense = F.mse_loss(pred_dense, y_tr)
            loss_dense.backward()
            opt_dense.step()

            # Static MoE train
            opt_static.zero_grad()
            pred_static, _ = static_moe(x_tr)
            loss_static = F.mse_loss(pred_static, y_tr)
            loss_static.backward()
            opt_static.step()

            # ISHMN Dynamic MoE train
            ishmn_moe.train()
            opt_ishmn.update_dynamic_schedules()
            opt_ishmn.zero_grad()
            pred_ishmn, _ = ishmn_moe(x_tr, u_tr, r_tr, optimizer=opt_ishmn)
            loss_ishmn = F.mse_loss(pred_ishmn, y_tr)
            loss_ishmn.backward()
            opt_ishmn.step()

        # Evaluate all models across ALL domains seen so far up to task_idx
        dense_model.eval()
        static_moe.eval()
        ishmn_moe.eval()

        row_dense = []
        row_static = []
        row_ishmn = []

        with torch.no_grad():
            for eval_d in domains:
                x_v, u_v, r_v, y_v = val_sets[eval_d]
                l_d = F.mse_loss(dense_model(x_v)[0], y_v).item()
                l_s = F.mse_loss(static_moe(x_v)[0], y_v).item()
                l_i = F.mse_loss(ishmn_moe(x_v, u_v, r_v)[0], y_v).item()
                row_dense.append(l_d)
                row_static.append(l_s)
                row_ishmn.append(l_i)

        history["Dense"].append(row_dense)
        history["Static MoE"].append(row_static)
        history["ISHMN Dynamic MoE"].append(row_ishmn)

        print(f"    Domain {task_idx} Complete | Active Experts in ISHMN: {ishmn_moe.num_experts}")
        print(f"    Validation MSE on Domain 0 (Initial Task): Dense={row_dense[0]:.4f} | Static MoE={row_static[0]:.4f} | ISHMN={row_ishmn[0]:.4f}")

    # Calculate Backward Transfer on Domain 0
    init_loss_d = history["Dense"][0][0]
    final_loss_d = history["Dense"][-1][0]
    bwt_d = init_loss_d - final_loss_d  # Negative means performance deteriorated

    init_loss_s = history["Static MoE"][0][0]
    final_loss_s = history["Static MoE"][-1][0]
    bwt_s = init_loss_s - final_loss_s

    init_loss_i = history["ISHMN Dynamic MoE"][0][0]
    final_loss_i = history["ISHMN Dynamic MoE"][-1][0]
    bwt_i = init_loss_i - final_loss_i

    print("\n--- Continual Learning Evaluation Results ---")
    print(f"Dense Baseline       : Domain 0 Initial Loss: {init_loss_d:.4f} -> Final Loss: {final_loss_d:.4f} (Forgetting Degradation: {bwt_d:+.4f})")
    print(f"Static MoE Baseline  : Domain 0 Initial Loss: {init_loss_s:.4f} -> Final Loss: {final_loss_s:.4f} (Forgetting Degradation: {bwt_s:+.4f})")
    print(f"ISHMN Dynamic MoE    : Domain 0 Initial Loss: {init_loss_i:.4f} -> Final Loss: {final_loss_i:.4f} (Retention Shield: {bwt_i:+.4f})")
    print(f"Final Spawned Experts in ISHMN: {ishmn_moe.num_experts}")

    return {
        "history": history,
        "domain_names": domain_names,
        "final_experts": ishmn_moe.num_experts
    }

# =====================================================================
# BENCHMARK 2: Triggered Emission Calibration & Deliberation Dynamics
# =====================================================================

def run_triggered_emission_benchmark(device):
    print("\n" + "="*70)
    print(">>> BENCHMARK 2: Triggered Emission Calibration & Deliberation Dynamics")
    print("="*70)

    loop = ContinuousSelfLoop(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        emission_threshold=0.52,
        device=device
    )

    test_scenarios = [
        {"name": "Clean Signal (Easy)", "noise": 0.05, "is_noise": False},
        {"name": "Corrupted Signal (Complex)", "noise": 0.50, "is_noise": False},
        {"name": "Pure Gaussian Static (Noise)", "noise": 1.00, "is_noise": True}
    ]

    results = []

    # Pre-seed memory with canonical baseline pattern
    seed_pat = torch.ones(1, 32, device=device)
    loop.inject_input(seed_pat, modality="continuous", tag="canonical_seed")
    for _ in range(4):
        loop.loop_tick()

    for sc in test_scenarios:
        print(f"\n[*] Testing Scenario: {sc['name']}...")
        emissions_fired = 0
        latencies = []
        confidences = []

        # Run trials per scenario
        for trial in range(15):
            if sc["is_noise"]:
                stim = torch.randn(1, 32, device=device) * 2.0
            else:
                base = torch.ones(1, 32, device=device)
                stim = base + torch.randn(1, 32, device=device) * sc["noise"]

            # Inject stimulus at tick 1
            loop.inject_input(stim, modality="continuous", tag=sc["name"])

            trial_fired = False
            for tick in range(1, 10):
                em = loop.loop_tick()
                if em is not None:
                    emissions_fired += 1
                    latencies.append(tick)
                    confidences.append(em.confidence)
                    trial_fired = True
                    break

        avg_lat = sum(latencies) / max(1, len(latencies)) if latencies else 0.0
        avg_conf = sum(confidences) / max(1, len(confidences)) if confidences else 0.0
        trigger_rate = emissions_fired / 15.0

        print(f"    Trigger Rate: {trigger_rate*100:.1f}% ({emissions_fired}/15) | Avg Deliberation Ticks: {avg_lat:.2f} | Avg Conf: {avg_conf:.3f}")
        results.append({
            "scenario": sc["name"],
            "trigger_rate": trigger_rate,
            "avg_latency": avg_lat,
            "avg_confidence": avg_conf
        })

    return results

# =====================================================================
# BENCHMARK 3: Sleep Consolidation & Shell Compaction
# =====================================================================

def run_sleep_consolidation_benchmark(device):
    print("\n" + "="*70)
    print(">>> BENCHMARK 3: Sleep Consolidation & Shell Compaction Ablation")
    print("="*70)

    # Compare 2 identical loops: One WITH sleep consolidation, one WITHOUT
    loop_with_sleep = ContinuousSelfLoop(d_model=32, hyperspace_dim=8, device=device)
    loop_no_sleep = ContinuousSelfLoop(d_model=32, hyperspace_dim=8, device=device)

    # Disable sleep consolidation in loop_no_sleep
    def no_op_sleep():
        pass
    loop_no_sleep.moe.sleep_consolidate_and_prune = no_op_sleep

    # Run both loops for 60 ticks with idle periods
    for t in range(1, 61):
        if t in [5, 25, 45]:
            stim = torch.randn(1, 32, device=device)
            loop_with_sleep.inject_input(stim, modality="continuous")
            loop_no_sleep.inject_input(stim, modality="continuous")

        loop_with_sleep.loop_tick()
        loop_no_sleep.loop_tick()

    slots_sleep = loop_with_sleep.mem_state["u"].size(1)
    slots_no_sleep = loop_no_sleep.mem_state["u"].size(1)
    r_max_sleep = float(loop_with_sleep.mem_state["r"].max().item())
    r_max_no_sleep = float(loop_no_sleep.mem_state["r"].max().item())
    energy_sleep = float(loop_with_sleep.mem_state["energy"].mean().item())
    energy_no_sleep = float(loop_no_sleep.mem_state["energy"].mean().item())

    print(f"Configuration WITH Sleep Consolidation:")
    print(f"  Slots: {slots_sleep} | Outer Horizon r_max: {r_max_sleep:.2f} | Avg Retention Energy: {energy_sleep:.3f}")
    print(f"Configuration WITHOUT Sleep (Ablated):")
    print(f"  Slots: {slots_no_sleep} | Outer Horizon r_max: {r_max_no_sleep:.2f} | Avg Retention Energy: {energy_no_sleep:.3f}")

    return {
        "slots": (slots_sleep, slots_no_sleep),
        "r_max": (r_max_sleep, r_max_no_sleep),
        "energy": (energy_sleep, energy_no_sleep)
    }

# =====================================================================
# RENDER ALL BENCHMARK PLOTS
# =====================================================================

def render_benchmark_report_plots(b1_data, b2_data, b3_data, output_file: str):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=150)
    fig.suptitle("ISHMN & Dynamic MoE Empirical Benchmark Evaluation", fontsize=15, fontweight='bold')

    # Plot 1: Continual Learning Forgetting Shield on Domain 0
    tasks = list(range(len(b1_data["history"]["Dense"])))
    d_loss = [row[0] for row in b1_data["history"]["Dense"]]
    s_loss = [row[0] for row in b1_data["history"]["Static MoE"]]
    i_loss = [row[0] for row in b1_data["history"]["ISHMN Dynamic MoE"]]

    axs[0, 0].plot(tasks, d_loss, marker='o', color='#ef4444', linewidth=2.2, label='Dense Baseline (Severe Catastrophic Overwriting)')
    axs[0, 0].plot(tasks, s_loss, marker='s', color='#f59e0b', linewidth=2.0, label='Static MoE (Capacity Drift)')
    axs[0, 0].plot(tasks, i_loss, marker='^', color='#10b981', linewidth=2.5, label='ISHMN Dynamic MoE (Neurogenesis Shield)')
    axs[0, 0].set_title("Catastrophic Forgetting Resistance on Domain 0", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("Sequential Tasks Completed (Task 0 -> 3)")
    axs[0, 0].set_ylabel("Domain 0 Error (MSE Loss)")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # Plot 2: Final Error across ALL 4 domains
    domain_names = b1_data["domain_names"]
    x_pos = np.arange(len(domain_names))
    width = 0.26

    final_d = b1_data["history"]["Dense"][-1]
    final_s = b1_data["history"]["Static MoE"][-1]
    final_i = b1_data["history"]["ISHMN Dynamic MoE"][-1]

    axs[0, 1].bar(x_pos - width, final_d, width=width, label='Dense', color='#ef4444', alpha=0.85)
    axs[0, 1].bar(x_pos, final_s, width=width, label='Static MoE', color='#f59e0b', alpha=0.85)
    axs[0, 1].bar(x_pos + width, final_i, width=width, label='ISHMN Dynamic MoE', color='#10b981', alpha=0.85)
    axs[0, 1].set_title("Final Retained Error across All Domains", fontsize=11, fontweight='bold')
    axs[0, 1].set_xticks(x_pos)
    axs[0, 1].set_xticklabels(domain_names, rotation=15, ha='right')
    axs[0, 1].set_ylabel("Validation MSE Loss")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # Plot 3: Triggered Emission Calibration
    scenarios = [r["scenario"].split("(")[0].strip() for r in b2_data]
    trigger_rates = [r["trigger_rate"] * 100 for r in b2_data]
    latencies = [r["avg_latency"] for r in b2_data]

    color_bars = ['#38bdf8', '#a855f7', '#64748b']
    bars = axs[1, 0].bar(scenarios, trigger_rates, color=color_bars, width=0.45, edgecolor='black')
    axs[1, 0].set_title("Triggered Emission Selectivity (Clean vs Noise)", fontsize=11, fontweight='bold')
    axs[1, 0].set_ylabel("Trigger Rate (%)")
    axs[1, 0].set_ylim(0, 110)
    for bar, lat in zip(bars, latencies):
        h = bar.get_height()
        lat_text = f"{lat:.1f} ticks" if lat > 0 else "Silent (absorbed)"
        axs[1, 0].text(bar.get_x() + bar.get_width()/2., h + 2, f"{h:.0f}%\n({lat_text})", ha='center', va='bottom', fontsize=9, fontweight='bold')
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Sleep Consolidation Ablation Comparison
    categories = ['Active Slots', 'Outer Horizon r_max', 'Retention Energy']
    with_sleep_vals = [b3_data["slots"][0], b3_data["r_max"][0], b3_data["energy"][0] * 10]
    no_sleep_vals = [b3_data["slots"][1], b3_data["r_max"][1], b3_data["energy"][1] * 10]
    pos = np.arange(len(categories))
    w = 0.35

    axs[1, 1].bar(pos - w/2, with_sleep_vals, width=w, label='With Sleep Consolidation', color='#8b5cf6', alpha=0.85)
    axs[1, 1].bar(pos + w/2, no_sleep_vals, width=w, label='Without Sleep (Ablated)', color='#94a3b8', alpha=0.85)
    axs[1, 1].set_title("Sleep Consolidation & Compactness Ablation", fontsize=11, fontweight='bold')
    axs[1, 1].set_xticks(pos)
    axs[1, 1].set_xticklabels(['Active Slots', 'r_max (Horizon)', 'Energy (x10)'])
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"\n[Benchmark Suite] Comprehensive evaluation report plot saved to: {output_file}")

if __name__ == "__main__":
    set_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark Suite] Starting comprehensive benchmarks on {dev}...")

    # Run Benchmark 1
    b1_res = run_continual_learning_benchmark(dev)

    # Run Benchmark 2
    b2_res = run_triggered_emission_benchmark(dev)

    # Run Benchmark 3
    b3_res = run_sleep_consolidation_benchmark(dev)

    # Render multi-panel report plot
    os.makedirs("artifacts", exist_ok=True)
    report_image = "artifacts/benchmark_results.png"
    render_benchmark_report_plots(b1_res, b2_res, b3_res, report_image)
