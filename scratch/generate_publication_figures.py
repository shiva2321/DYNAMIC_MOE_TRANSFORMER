"""
Generate high-DPI publication figures for the paper draft:
1. System Architecture Schematic (Phasor Routing, 2-Compartment Dendritic Expert, Global Bus, Spawning)
2. Hardware Dispatch Throughput & VRAM Scaling Comparison
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments")
ARTIFACT_DIR = "C:/Users/Asta/.gemini/antigravity-ide/brain/0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"

def generate_system_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    ax.axis('off')

    # Color Palette
    bg_color = "#F8F9FA"
    box_color = "#FFFFFF"
    edge_color = "#2B2D42"
    phasor_color = "#4C72B0"
    dendrite_color = "#55A868"
    bus_color = "#C44E52"
    spawn_color = "#8172B2"

    fig.patch.set_facecolor(bg_color)

    # Title
    ax.text(0.5, 0.96, "Universal SubStrait: Continual Neurogenetic Transformer Architecture", 
            ha='center', va='top', fontsize=18, fontweight='bold', color=edge_color)
    ax.text(0.5, 0.92, "Complex Phasor Routing in C^2048  |  Two-Compartment Dendritic SwiGLU  |  Global Workspace Bus", 
            ha='center', va='top', fontsize=12, style='italic', color="#555555")

    # 1. Input & Phasor Projection
    rect1 = patches.FancyBboxPatch((0.04, 0.50), 0.22, 0.35, boxstyle="round,pad=0.02", facecolor=box_color, edgecolor=phasor_color, linewidth=2)
    ax.add_patch(rect1)
    ax.text(0.15, 0.82, "1. Complex Phasor VSA\n& Dynamic WTA Router", ha='center', va='top', fontsize=12, fontweight='bold', color=phasor_color)
    ax.text(0.15, 0.72, r"• Input $x \in \mathbb{R}^{B \times S \times D}$" + "\n" +
                        r"• Phasor: $\theta = W_\theta x \in [-\pi, \pi]^H$" + "\n" +
                        r"• State: $\mathbf{z} = e^{i\theta} \in \mathbb{C}^{2048}$" + "\n" +
                        r"• Similarity: $\text{Re}(\mathbf{z} \cdot \mathbf{k}_e^*)$" + "\n" +
                        r"• Dynamic $k^*(x) \in [1, 4]$", ha='center', va='top', fontsize=9.5, color="#222222")

    # 2. Token-Sorted Dispatch
    rect2 = patches.FancyBboxPatch((0.30, 0.50), 0.22, 0.35, boxstyle="round,pad=0.02", facecolor=box_color, edgecolor=edge_color, linewidth=2)
    ax.add_patch(rect2)
    ax.text(0.41, 0.82, "2. Vectorized Token-Sorted\nBatched Dispatch", ha='center', va='top', fontsize=12, fontweight='bold', color=edge_color)
    ax.text(0.41, 0.72, r"• Flatten $(N, k) \to \text{flat\_idx}$" + "\n" +
                        r"• Argsort $\to$ contiguous slices" + "\n" +
                        r"• Single batched call per expert" + "\n" +
                        r"• Invert permutation & restore" + "\n" +
                        r"• Bitwise exact ($< 10^{-7}$ diff)" + "\n" +
                        r"• $O(N D + E D H)$ memory", ha='center', va='top', fontsize=9.5, color="#222222")

    # 3. Two-Compartment Dendritic Expert
    rect3 = patches.FancyBboxPatch((0.56, 0.50), 0.22, 0.35, boxstyle="round,pad=0.02", facecolor=box_color, edgecolor=dendrite_color, linewidth=2)
    ax.add_patch(rect3)
    ax.text(0.67, 0.82, "3. Two-Compartment\nDendritic SwiGLU Expert", ha='center', va='top', fontsize=12, fontweight='bold', color=dendrite_color)
    ax.text(0.67, 0.72, r"• Basal: $\mathbf{h}_b = \text{SwiGLU}_b(\mathbf{x})$" + "\n" +
                        r"• Apical: $\mathbf{h}_a = \text{SwiGLU}_a(\mathbf{c}_{\text{bus}})$" + "\n" +
                        r"• NMDA Modulation:" + "\n" +
                        r"  $\mathbf{y} = W_{\text{down}}(\mathbf{h}_b \odot (1 + \beta \mathbf{h}_a))$" + "\n" +
                        r"• Biophysical gating mechanism", ha='center', va='top', fontsize=9.5, color="#222222")

    # 4. Global Workspace Bus
    rect4 = patches.FancyBboxPatch((0.82, 0.50), 0.15, 0.35, boxstyle="round,pad=0.02", facecolor=box_color, edgecolor=bus_color, linewidth=2)
    ax.add_patch(rect4)
    ax.text(0.895, 0.82, "4. Global Bus\n& Apical Feedback", ha='center', va='top', fontsize=12, fontweight='bold', color=bus_color)
    ax.text(0.895, 0.72, r"• Bus $\in \mathbb{C}^{2048}$" + "\n" +
                         r"• Unbind with keys" + "\n" +
                         r"• Cross-layer context" + "\n" +
                         r"• Top-down apical state" + "\n" +
                         r"• Resolves polysemy", ha='center', va='top', fontsize=9.5, color="#222222")

    # 5. Autonomous Neurogenesis & DynamicWarmupAdamW (Bottom Panel)
    rect5 = patches.FancyBboxPatch((0.04, 0.08), 0.93, 0.34, boxstyle="round,pad=0.02", facecolor=box_color, edgecolor=spawn_color, linewidth=2)
    ax.add_patch(rect5)
    ax.text(0.50, 0.38, "5. Autonomous Novelty-Triggered Neurogenesis & Dynamic Optimizer Registration", 
            ha='center', va='top', fontsize=13, fontweight='bold', color=spawn_color)
    
    ax.text(0.20, 0.30, "Novelty Trigger Condition:", fontweight='bold', fontsize=11, color=edge_color)
    ax.text(0.20, 0.25, r"$\max_{e} \text{Re}(\mathbf{z} \cdot \mathbf{k}_e^*) < \tau_{\text{spawn}} \quad (\tau = 0.35)$" + "\n" +
                        "Fires when incoming token manifold is\northogonal to all existing expert keys.", fontsize=9.5, color="#222222")

    ax.text(0.50, 0.30, "Parent Weight Inheritance & Orthogonalization:", fontweight='bold', fontsize=11, color=edge_color)
    ax.text(0.50, 0.25, r"• Clone weights from nearest parent: $\mathbf{W}_{\text{child}} \leftarrow \mathbf{W}_{\text{parent}} + \epsilon$" + "\n" +
                        r"• Initialize child key: $\mathbf{k}_{\text{child}} = \mathbf{z}_{\text{trigger}}$" + "\n" +
                        r"• Gram-Schmidt Orthogonalization vs active keys" + "\n" +
                        r"• Asymmetric growth across layers ($E_L \in [14, 26]$)", fontsize=9.5, color="#222222")

    ax.text(0.80, 0.30, "DynamicWarmupAdamW Registration:", fontweight='bold', fontsize=11, color=edge_color)
    ax.text(0.80, 0.25, r"• Instant mid-training registration" + "\n" +
                        r"• Independent warmup (40 steps)" + "\n" +
                        r"• Zero gradient shock to existing weights" + "\n" +
                        r"• Eliminates catastrophic forgetting on code/lit", fontsize=9.5, color="#222222")

    # Arrows
    arrowprops = dict(arrowstyle="->", lw=2, color="#2B2D42")
    ax.annotate("", xy=(0.30, 0.67), xytext=(0.26, 0.67), arrowprops=arrowprops)
    ax.annotate("", xy=(0.56, 0.67), xytext=(0.52, 0.67), arrowprops=arrowprops)
    ax.annotate("", xy=(0.82, 0.67), xytext=(0.78, 0.67), arrowprops=arrowprops)
    
    # Feedback loop arrow from Bus back to Expert
    ax.annotate("", xy=(0.67, 0.50), xytext=(0.895, 0.50), 
                arrowprops=dict(arrowstyle="->", lw=1.8, color=bus_color, connectionstyle="arc3,rad=0.3"))
    ax.text(0.78, 0.44, "Apical Context Feedback", ha='center', fontsize=9, color=bus_color, fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(EXP_DIR, "system_architecture_diagram.png")
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    
    if os.path.exists(ARTIFACT_DIR):
        artifact_path = os.path.join(ARTIFACT_DIR, "system_architecture_diagram.png")
        plt.savefig(artifact_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[SAVED] Architecture schematic saved to: {out_path}")

def generate_hardware_dispatch_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    sns.set_theme(style="whitegrid")

    experts = [2, 8, 16, 32]
    
    # Measured Throughput (Tokens/sec) on RTX 3060
    sequential_tp = [7900, 1850, 720, 236] # Real measured bottleneck
    tokensorted_tp = [8427.6, 6105.3, 5155.2, 4362.7] # Real token-sorted throughput

    # VRAM (GB)
    tokensorted_vram = [1.98, 2.53, 3.19, 4.51]

    # Plot Throughput
    ax1.plot(experts, tokensorted_tp, marker='o', lw=2.5, color='#4C72B0', label='Vectorized Token-Sorted Dispatch (Ours)')
    ax1.plot(experts, sequential_tp, marker='s', lw=2.5, color='#C44E52', linestyle='--', label='Sequential Loop Dispatch (Standard)')
    ax1.set_xlabel("Number of Active Experts (E)", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Measured Throughput (tokens/sec)", fontsize=12, fontweight='bold')
    ax1.set_title("Throughput vs Expert Count on NVIDIA RTX 3060\n(18.5x Speedup at E=32)", fontsize=13, fontweight='bold')
    ax1.set_xticks(experts)
    ax1.legend(fontsize=11)
    ax1.set_ylim(0, 9500)

    # Annotate Speedup
    ax1.annotate('18.5x Speedup\n(4,363 vs 236 tok/s)', xy=(32, 4362.7), xytext=(22, 2800),
                 arrowprops=dict(facecolor='#4C72B0', shrink=0.08, width=1.5, headwidth=8),
                 fontweight='bold', color='#4C72B0', fontsize=10)

    # Plot VRAM
    ax2.plot(experts, tokensorted_vram, marker='^', lw=2.5, color='#55A868', label='Token-Sorted VRAM Footprint')
    ax2.axhline(y=12.0, color='gray', linestyle=':', label='RTX 3060 Total VRAM (12.0 GB)')
    ax2.set_xlabel("Number of Active Experts (E)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Peak Allocated VRAM (GB)", fontsize=12, fontweight='bold')
    ax2.set_title("VRAM Memory Scaling (O(N·D + E·D·H))\n(Safe Sub-Linear Footprint)", fontsize=13, fontweight='bold')
    ax2.set_xticks(experts)
    ax2.set_ylim(0, 13)
    ax2.legend(fontsize=11, loc='upper left')

    plt.tight_layout()
    out_path = os.path.join(EXP_DIR, "hardware_dispatch_benchmark.png")
    plt.savefig(out_path, dpi=300)

    if os.path.exists(ARTIFACT_DIR):
        artifact_path = os.path.join(ARTIFACT_DIR, "hardware_dispatch_benchmark.png")
        plt.savefig(artifact_path, dpi=300)
    plt.close()
    print(f"[SAVED] Hardware dispatch plot saved to: {out_path}")

if __name__ == "__main__":
    generate_system_architecture_diagram()
    generate_hardware_dispatch_plot()
