"""
Generate High-Resolution Full System Architecture Flow Diagram.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_system_architecture():
    fig, ax = plt.subplots(figsize=(20, 14), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    c_bg = "#0f172a"
    c_card_bg = "#1e293b"
    c_border = "#334155"
    c_data = "#38bdf8"
    c_attn = "#818cf8"
    c_hyper = "#c084fc"
    c_spawn = "#34d399"
    c_expert = "#fb923c"
    c_out = "#f43f5e"
    c_text = "#f8fafc"
    c_subtext = "#94a3b8"

    fig.patch.set_facecolor(c_bg)

    # Title
    ax.text(50, 96.5, "UNIVERSAL SUBSTRAIT: FULL SYSTEM ARCHITECTURE & DATA FLOW",
            fontsize=20, fontweight="bold", color=c_text, ha="center", va="center")
    ax.text(50, 94.0, "End-to-End Flow: Interleaved Ingestion → Pure RoPE Attention → Hyperspace C^2048 Routing → Clonal Neurogenesis → Weight-Tied Output",
            fontsize=11, color=c_subtext, ha="center", va="center")

    # Helper function for styled boxes
    def draw_box(x, y, w, h, title, subtitle, color, border_color=None, alpha=0.9):
        if border_color is None:
            border_color = color
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
                                     facecolor=c_card_bg, edgecolor=border_color, linewidth=2, alpha=alpha)
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 2.2, title, fontsize=11, fontweight="bold", color=color, ha="center", va="center")
        if subtitle:
            ax.text(x + w/2, y + h/2 - 0.8, subtitle, fontsize=8.5, color=c_subtext, ha="center", va="center", wrap=True)

    def draw_arrow(x1, y1, x2, y2, color, label=""):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5, mutation_scale=15))
        if label:
            ax.text((x1 + x2)/2, (y1 + y2)/2 + 1.2, label, fontsize=8, color=color, fontweight="bold", ha="center")

    # 1. INPUT STREAMING & EMBEDDING (Left Column: x=3 to 22)
    # Header
    ax.text(12.5, 89, "1. INGESTION & EMBEDDING", fontsize=12, fontweight="bold", color=c_data, ha="center")
    draw_box(3, 76, 19, 10, "Interleaved Data Streams", "100% Unique Diverse Shards:\n• Stories (TinyStories)\n• Code (Python Repos)\n• WikiText-103 Facts\n• FineWeb-Edu Reasoning", c_data)
    draw_box(3, 61, 19, 11, "TikToken BPE & Weight-Tied Emb", "Vocab: 50,304 | d_model = 384\nWeight-Tied Matrix: W_embed\nInput Tokens: X in R^{B x S x D}\n(Shared with Output Head)", c_data)
    draw_box(3, 46, 19, 11, "Pure RoPE Encoding", "Rotary Position Embeddings\nZero absolute position tables\nEliminates phase interference\nRelative angle rotation R_theta(x)", c_data)

    draw_arrow(12.5, 76, 12.5, 72, c_data)
    draw_arrow(12.5, 61, 12.5, 57, c_data)

    # 2. TRANSFORMER ATTENTION (Column 2: x=27 to 46)
    ax.text(36.5, 89, "2. DYNAMIC ATTENTION BLOCK", fontsize=12, fontweight="bold", color=c_attn, ha="center")
    draw_box(27, 68, 19, 18, "Dynamic Sparse Attention", "Head Dim: 64 | 6 Heads\n• 4 Attention Sinks (Initial Tokens)\n• 4 Landmark Anchors (Key Cores)\n• 128 Foveal Local Sliding Window\nO(N) Complexity Scaling", c_attn)
    draw_box(27, 46, 19, 18, "RMSNorm & Residual Bridge", "LayerNorm: Pre-LN RMSNorm\nSkip-Connection:\nx_norm = RMSNorm(x)\nx = x + Attention(x_norm)", c_attn)

    draw_arrow(22, 51.5, 27, 51.5, c_data, "x_tokens")
    draw_arrow(36.5, 68, 36.5, 64, c_attn)

    # 3. HYPERSPACE ROUTING ENGINE (Column 3: x=51 to 72)
    ax.text(61.5, 89, "3. HYPERSPACE VSA ROUTER (C^2048)", fontsize=12, fontweight="bold", color=c_hyper, ha="center")
    draw_box(51, 74, 21, 12, "Complex Phasor Projection", "Linear: W_hyper: R^384 -> C^2048\nq = exp(i * W_hyper(x))\nUnit-Phasor Query Embedding", c_hyper)
    draw_box(51, 55, 21, 15, "Learnable Hyperspace Keys", "Trainable Parameters in C^2048:\nK_e = K_r + i * K_i (for e=1..N)\nHermitian Resonance:\nS(q, K_e) = Real(q * K_e^†) / sqrt(D)\nDirect Backpropagation Gradients", c_hyper)
    draw_box(51, 36, 21, 15, "Dynamic Gating & Dispatch", "Top-k Selection (Adaptive k=2..4)\nSoftmax Routing Weights: g_e\nDispatches tokens to experts:\nToken x_i -> Expert E_e", c_hyper)

    draw_arrow(46, 77, 51, 77, c_attn, "x_norm")
    draw_arrow(61.5, 74, 61.5, 70, c_hyper, "q in C^2048")
    draw_arrow(61.5, 55, 61.5, 51, c_hyper, "Resonance Scores")

    # 4. LIFELONG NEUROGENESIS & CLONAL MITOSIS (Bottom Box: x=51 to 72, y=14)
    draw_box(51, 15, 21, 17, "Autonomous Clonal Mitosis", "Novelty Trigger: max(S) < 0.30\n1. Find Closest Parent Expert\n2. Clone Weights: W_new = W_parent + ε\n3. Register in DynamicWarmupAdamW\nZero Cold Noise | Seamless Spawning", c_spawn, border_color=c_spawn)
    draw_arrow(61.5, 36, 61.5, 32, c_spawn, "Novelty Trigger?")

    # 5. DYNAMIC EXPERT COMPARTMENT (Column 4: x=77 to 96)
    ax.text(86.5, 89, "4. TWO-COMPARTMENT EXPERTS", fontsize=12, fontweight="bold", color=c_expert, ha="center")
    draw_box(77, 55, 20, 31, "SwiGLU Expert Pool", "Dynamic Pool: E_1, E_2, ... E_N\n(Grows dynamically from 2 to 60+)\nPer Expert Architecture:\n• Gate & Up: Linear(384, 768)\n• Activation: SiLU(Gate) * Up\n• Down: Linear(768, 384)\nOutput = Sum(g_e * Expert_e(x))", c_expert)
    
    draw_arrow(72, 43.5, 77, 60, c_hyper, "Gated Tokens")
    draw_arrow(72, 23.5, 77, 57, c_spawn, "Spawn New Circuit")

    # 6. OUTPUT STAGE (Top Right / Global Output: x=77 to 96, y=15 to 33)
    ax.text(86.5, 41, "5. WEIGHT-TIED OUTPUT", fontsize=12, fontweight="bold", color=c_out, ha="center")
    draw_box(77, 15, 20, 22, "Weight-Tied Projection Head", "Final RMSNorm(x_final)\nLinear Head: W_lm_head = W_embed^T\nLogits in R^{50,304}\nNext Token Probability:\nP(token) = Softmax(Logits / T)", c_out)

    draw_arrow(87, 55, 87, 37, c_expert, "x + MoE(x)")

    # Global Frame Box
    outer_box = patches.FancyBboxPatch((1, 4), 98, 88, boxstyle="round,pad=0.8,rounding_size=1.5",
                                       facecolor="none", edgecolor=c_border, linewidth=1.5, linestyle="--")
    ax.add_patch(outer_box)

    plt.tight_layout()
    out_path = "experiments/plots/full_system_architecture_flow.png"
    plt.savefig(out_path, facecolor=c_bg)
    plt.close()

    # Copy to brain artifacts folder
    import shutil
    brain_dir = r"C:\Users\Asta\.gemini\antigravity-ide\brain\0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"
    shutil.copy(out_path, os.path.join(brain_dir, "full_system_architecture_flow.png"))
    print(f"[SAVED] High-resolution architecture flow diagram saved to: {out_path}")

if __name__ == "__main__":
    draw_system_architecture()
