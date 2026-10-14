import sys
import os
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from geometry.layers import YggdrasilSandwichModel

def main():
    print("================================================================")
    print("  MAPPING ACTUAL TRAINED NEURAL NETWORK INTERNAL REPRESENTATIONS")
    print("================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train, device=device)
    y_train_t = torch.tensor(y_train, device=device)
    X_test_t = torch.tensor(X_test, device=device)
    y_test_t = torch.tensor(y_test, device=device)

    dim_h = 16
    dim_t = 16
    num_prototypes = 20
    hidden_dim = 64

    # 1. Train Model
    print("Training Yggdrasil Geometric Sandwich Model...")
    model = YggdrasilSandwichModel(
        in_features=64,
        hidden_dim=hidden_dim,
        num_prototypes=num_prototypes,
        num_classes=10,
        dim_h=dim_h,
        dim_t=dim_t,
        c=1.0,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(40):
        model.train()
        perm = torch.randperm(len(X_train), device=device)
        for b in range((len(X_train) + 127) // 128):
            idx = perm[b * 128 : (b + 1) * 128]
            bx, by = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            opt.step()
            model.cortex.clamp_prototypes()

    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=-1)
        acc = (preds == y_test_t).float().mean().item() * 100.0
    print(f"Model Trained! Test Accuracy: {acc:.2f}%\n")

    # 2. Extract Actual Internal Representations
    print("Extracting actual representations from all network layers...")
    with torch.no_grad():
        # A. Sensory Perception Layer (Flat Euclidean features)
        feat_enc = model.encoder(X_test_t).cpu().numpy()  # (450, 64)

        # B. Projector Output (Yggdrasil Manifold coordinates)
        state_manifold = model.projector(model.encoder(X_test_t)).cpu().numpy()  # (450, 33)
        h_coords = state_manifold[:, :dim_h]  # Hyperbolic (450, 16)
        t_coords = state_manifold[:, dim_h:dim_h + dim_t]  # Torus (450, 16)
        r_coords = state_manifold[:, -1]  # Radial (450,)

        # C. Central Cognitive Prototypes
        proto_data = model.cortex.prototypes.data.cpu().numpy()  # (20, 33)
        proto_h = proto_data[:, :dim_h]
        proto_t = proto_data[:, dim_h:dim_h + dim_t]
        proto_r = proto_data[:, -1]

        # D. Attention Weights across Prototypes
        logits_cortex, _ = model.cortex(torch.tensor(state_manifold, device=device))
        attn_weights = F.softmax(logits_cortex, dim=-1).cpu().numpy()  # (450, 20)

        # E. Receptive Fields of first encoder layer
        first_layer_weights = model.encoder[0].weight.data.cpu().numpy()  # (64, 64)

    # 3. Project Hyperbolic Space to 2D Poincare Disk
    # PCA to find the 2 dominant hyperbolic principal axes
    pca_h = PCA(n_components=2)
    h_combined = np.vstack([h_coords, proto_h])
    h_proj_all = pca_h.fit_transform(h_combined)
    # Ensure they stay strictly within the Poincare unit disk: norm < 0.98
    scale_h = 0.95 / (np.max(np.linalg.norm(h_proj_all, axis=-1)) + 1e-6)
    h_proj = h_proj_all[:len(h_coords)] * scale_h
    proto_h_proj = h_proj_all[len(h_coords):] * scale_h

    # 4. Project Torus Space to 2D Torus Coordinates
    # First 2 principal torus dimensions
    pca_t = PCA(n_components=2)
    t_combined = np.vstack([t_coords, proto_t])
    t_proj_all = pca_t.fit_transform(t_combined)
    # Wrap onto [-pi, pi)
    t_proj = (t_proj_all[:len(t_coords)] + np.pi) % (2.0 * np.pi) - np.pi
    proto_t_proj = (t_proj_all[len(t_coords):] + np.pi) % (2.0 * np.pi) - np.pi

    # Flat sensory PCA
    pca_flat = PCA(n_components=2)
    feat_proj = pca_flat.fit_transform(feat_enc)

    # 5. Build Scientific Multi-Panel Figure
    print("Rendering publication-quality Brain Map figure...")
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 11), dpi=200)

    cmap = plt.cm.get_cmap("tab10", 10)
    digit_colors = [cmap(i) for i in y_test]

    # Panel 1: Top "Bread" - Flat Sensory Perception Map (Encoder R^64)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.set_facecolor("#0e1117")
    scatter1 = ax1.scatter(
        feat_proj[:, 0], feat_proj[:, 1],
        c=y_test, cmap="tab10", alpha=0.75, s=25, edgecolors="none"
    )
    ax1.set_title("PART 1: SENSORY PERCEPTION (Flat Euclidean $\\mathbb{R}^{64}$)\nEncoder Latent Space ($N=450$ Test Digits)", fontsize=13, fontweight="bold", color="#58a6ff")
    ax1.set_xlabel("PCA Axis 1 (Feature Variation)", fontsize=10, color="#8b949e")
    ax1.set_ylabel("PCA Axis 2 (Feature Variation)", fontsize=10, color="#8b949e")
    ax1.grid(True, linestyle="--", alpha=0.15, color="#ffffff")
    cb1 = plt.colorbar(scatter1, ax=ax1, ticks=range(10), fraction=0.046, pad=0.04)
    cb1.set_label("Digit Class (0 - 9)", color="#c9d1d9")
    cb1.ax.tick_params(colors="#c9d1d9")

    # Inset for Receptive Fields (V1 visual columns analogy)
    # Show 4 sample receptive fields (8x8 pixel weights)
    inset_ax = ax1.inset_axes([0.68, 0.65, 0.3, 0.3])
    inset_ax.set_facecolor("#161b22")
    rf_grid = np.zeros((18, 18))
    for r in range(2):
        for c in range(2):
            w = first_layer_weights[r * 2 + c].reshape(8, 8)
            rf_grid[r*9:r*9+8, c*9:c*9+8] = w
    inset_ax.imshow(rf_grid, cmap="coolwarm")
    inset_ax.set_title("Sensory Filters (V1-like)", fontsize=8, color="#58a6ff")
    inset_ax.axis("off")

    # Panel 2: Central "Filling" (Hyperbolic Base - Poincaré Ball Tree Indexing)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_facecolor("#0e1117")
    # Draw Poincare boundary circle
    boundary = Circle((0, 0), 1.0, color="#f85149", fill=False, linewidth=2.5, linestyle="--", alpha=0.85, label="Boundary (||u|| = 1)")
    interior = Circle((0, 0), 1.0, color="#1f1f2e", fill=True, alpha=0.35)
    ax2.add_patch(interior)
    ax2.add_patch(boundary)

    # Plot test data points
    ax2.scatter(
        h_proj[:, 0], h_proj[:, 1],
        c=y_test, cmap="tab10", alpha=0.65, s=20, edgecolors="none"
    )

    # Plot actual 20 learned memory prototypes as golden stars
    ax2.scatter(
        proto_h_proj[:, 0], proto_h_proj[:, 1],
        c="#f1e05a", marker="*", s=160, edgecolors="#ffffff", linewidth=0.8,
        label=f"20 Memory Prototypes (Cortex Anchors)", zorder=10
    )

    # Draw geodesic rays from origin to prototypes showing taxonomic branching
    for p in proto_h_proj:
        ax2.plot([0, p[0]], [0, p[1]], color="#f1e05a", alpha=0.2, linewidth=1.0, linestyle=":")

    ax2.set_xlim(-1.15, 1.15)
    ax2.set_ylim(-1.15, 1.15)
    ax2.set_aspect("equal")
    ax2.set_title("PART 2A: COGNITIVE CORTEX - HYPERBOLIC BASE ($\\mathbb{H}^{16}$)\nPoincaré Ball Metric Tree (Taxonomic Depth & Episodic Anchors)", fontsize=13, fontweight="bold", color="#bc8cff")
    ax2.text(0, -0.05, "ROOT\nORIGIN", color="#c9d1d9", fontsize=8, ha="center", weight="bold")
    ax2.legend(loc="lower right", fontsize=8, facecolor="#161b22", edgecolor="#30363d")
    ax2.axis("off")

    # Panel 3: Central "Filling" (Toroidal Fiber - Clifford Torus Phase Space)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_facecolor("#0e1117")
    ax3.scatter(
        t_proj[:, 0], t_proj[:, 1],
        c=y_test, cmap="tab10", alpha=0.65, s=20, edgecolors="none"
    )
    ax3.scatter(
        proto_t_proj[:, 0], proto_t_proj[:, 1],
        c="#39d353", marker="D", s=70, edgecolors="#ffffff", linewidth=0.8,
        label="Prototypes (Grid Phase)", zorder=10
    )
    ax3.set_xlim(-np.pi, np.pi)
    ax3.set_ylim(-np.pi, np.pi)
    ax3.set_title("PART 2B: COGNITIVE CORTEX - TOROIDAL FIBER ($\\mathbb{T}^{16}$)\nClifford Torus Phase Space (Grid Cells & Invariant Relations)", fontsize=13, fontweight="bold", color="#3fb950")
    ax3.set_xlabel("Torus Angle $\\theta_1$ ($-\\pi$ to $\\pi$)", fontsize=10, color="#8b949e")
    ax3.set_ylabel("Torus Angle $\\theta_2$ ($-\\pi$ to $\\pi$)", fontsize=10, color="#8b949e")
    ax3.grid(True, linestyle="--", alpha=0.2, color="#3fb950")
    ax3.legend(loc="lower right", fontsize=8, facecolor="#161b22", edgecolor="#30363d")

    # Panel 4: Bottom "Bread" - Executive Memory Attention Heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_facecolor("#0e1117")
    # Average prototype attention per digit class (10 classes x 20 prototypes)
    class_proto_attn = np.zeros((10, num_prototypes))
    for c_idx in range(10):
        mask = (y_test == c_idx)
        if np.any(mask):
            class_proto_attn[c_idx] = np.mean(attn_weights[mask], axis=0)

    im = ax4.imshow(class_proto_attn, cmap="magma", aspect="auto")
    ax4.set_title("PART 3: EXECUTIVE DECISION & COGNITIVE ROUTING\nActual Prototype Attention Heatmap (Class vs 20 Memory Anchors)", fontsize=13, fontweight="bold", color="#ffa657")
    ax4.set_xlabel("Trained Memory Prototype Index (0 - 19)", fontsize=10, color="#8b949e")
    ax4.set_ylabel("True Digit Class (0 - 9)", fontsize=10, color="#8b949e")
    ax4.set_yticks(range(10))
    ax4.set_xticks(range(0, num_prototypes, 2))
    ax4.tick_params(colors="#c9d1d9")
    cb4 = plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    cb4.set_label("Attention Weight $\\alpha_{ij}$", color="#c9d1d9")
    cb4.ax.tick_params(colors="#c9d1d9")

    plt.tight_layout()

    out_image_path = "C:/Users/Asta/.gemini/antigravity-ide/brain/14208f8b-7fd3-49fe-acd8-545f9fbd8f96/actual_trained_brain_map.png"
    plt.savefig(out_image_path, dpi=200, facecolor="#090d13", bbox_inches="tight")
    plt.close()

    print(f"ACTUAL BRAIN MAP SUCCESSFULLY SAVED TO:\n{out_image_path}")

if __name__ == "__main__":
    main()
