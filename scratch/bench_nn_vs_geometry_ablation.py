import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from geometry.layers import YggdrasilSandwichModel

# ---------------------------------------------------------------------------
# Identically Sized Baseline Models for Fair Ablation
# ---------------------------------------------------------------------------

class FlatEuclideanMemoryCortex(nn.Module):
    """Memory cortex operating in standard Flat Euclidean space (R^D)."""
    def __init__(self, num_prototypes: int, dim: int, out_features: int):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim) * 0.1)
        self.value_matrix = nn.Parameter(torch.randn(num_prototypes, out_features) / (num_prototypes ** 0.5))

    def clamp_prototypes(self):
        pass  # Flat space has no boundary

    def forward(self, queries: torch.Tensor):
        # Pairwise Euclidean distance
        diff = queries.unsqueeze(1) - self.prototypes.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)
        logits = -dist_sq
        attn = F.softmax(logits, dim=-1)
        context = torch.matmul(attn, self.value_matrix)
        return logits, context

class FlatSandwichModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_prototypes: int, num_classes: int, total_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.projector = nn.Linear(hidden_dim, total_dim)
        self.cortex = FlatEuclideanMemoryCortex(num_prototypes, total_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        state = self.projector(feat)
        _, context = self.cortex(state)
        return self.head(context + feat)


class SphericalMemoryCortex(nn.Module):
    """Memory cortex operating on the Unit Hypersphere (S^(D-1))."""
    def __init__(self, num_prototypes: int, dim: int, out_features: int):
        super().__init__()
        raw_p = torch.randn(num_prototypes, dim)
        self.prototypes = nn.Parameter(F.normalize(raw_p, p=2, dim=-1))
        self.value_matrix = nn.Parameter(torch.randn(num_prototypes, out_features) / (num_prototypes ** 0.5))

    def clamp_prototypes(self):
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, p=2, dim=-1)

    def forward(self, queries: torch.Tensor):
        q_norm = F.normalize(queries, p=2, dim=-1)
        p_norm = F.normalize(self.prototypes, p=2, dim=-1)
        # Cosine distance: 1 - cos(theta)
        cos_sim = torch.matmul(q_norm, p_norm.t()).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        dist_sq = 2.0 * (1.0 - cos_sim)
        logits = -dist_sq
        attn = F.softmax(logits, dim=-1)
        context = torch.matmul(attn, self.value_matrix)
        return logits, context

class SphericalSandwichModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, num_prototypes: int, num_classes: int, total_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.projector = nn.Linear(hidden_dim, total_dim)
        self.cortex = SphericalMemoryCortex(num_prototypes, total_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        feat = self.encoder(x)
        state = F.normalize(self.projector(feat), p=2, dim=-1)
        _, context = self.cortex(state)
        return self.head(context + feat)


# ---------------------------------------------------------------------------
# 10-Seed Comparative Experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("================================================================")
    print("  ABLATION EXPERIMENT: NEURAL NETWORK vs GEOMETRIC SHAPE")
    print("================================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    digits = load_digits()
    X = digits.data.astype("float32") / 16.0
    y = digits.target.astype("int64")

    dim_h = 16
    dim_t = 16
    total_dim = dim_h + dim_t + 1  # 33
    hidden_dim = 64
    num_prototypes = 20
    epochs = 25
    batch_size = 128

    seeds = list(range(100, 110))  # 10 dev seeds >= 100

    results_flat = []
    results_sphere = []
    results_yggdrasil = []

    # Few-shot representation test (evaluating prototype quality with only 1 prototype per class)
    fewshot_flat = []
    fewshot_sphere = []
    fewshot_yggdrasil = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=seed, stratify=y
        )
        X_tr = torch.tensor(X_train, device=device)
        y_tr = torch.tensor(y_train, device=device)
        X_te = torch.tensor(X_test, device=device)
        y_te = torch.tensor(y_test, device=device)

        models = {
            "flat": FlatSandwichModel(64, hidden_dim, num_prototypes, 10, total_dim).to(device),
            "sphere": SphericalSandwichModel(64, hidden_dim, num_prototypes, 10, total_dim).to(device),
            "yggdrasil": YggdrasilSandwichModel(64, hidden_dim, num_prototypes, 10, dim_h, dim_t).to(device),
        }

        # Train each model under identical conditions
        criterion = nn.CrossEntropyLoss()
        for name, model in models.items():
            opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
            num_batches = (len(X_train) + batch_size - 1) // batch_size

            for epoch in range(epochs):
                model.train()
                perm = torch.randperm(len(X_train), device=device)
                for b in range(num_batches):
                    idx = perm[b * batch_size : (b + 1) * batch_size]
                    bx, by = X_tr[idx], y_tr[idx]

                    opt.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    opt.step()
                    model.cortex.clamp_prototypes()

            model.eval()
            with torch.no_grad():
                preds = model(X_te).argmax(dim=-1)
                acc = (preds == y_te).float().mean().item() * 100.0

            if name == "flat":
                results_flat.append(acc)
            elif name == "sphere":
                results_sphere.append(acc)
            elif name == "yggdrasil":
                results_yggdrasil.append(acc)

    # Statistical Summary: Mean +- 95% CI
    def stats(arr):
        a = np.array(arr)
        mean = np.mean(a)
        ci = 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))
        return mean, ci

    m_flat, ci_flat = stats(results_flat)
    m_sphere, ci_sphere = stats(results_sphere)
    m_ygg, ci_ygg = stats(results_yggdrasil)

    print("\n--- RESULTS OVER 10 INDEPENDENT SEEDS (I.I.D. Pattern Recognition) ---")
    print(f"Flat Euclidean Cortex:   {m_flat:.2f}% +- {ci_flat:.2f}%")
    print(f"Spherical Cortex:        {m_sphere:.2f}% +- {ci_sphere:.2f}%")
    print(f"Yggdrasil Manifold:      {m_ygg:.2f}% +- {ci_ygg:.2f}%")

    print("\n================================================================")
    print("  STRUCTURAL CAPABILITY COMPARISON: NOT MEASURED BY THIS SCRIPT")
    print("================================================================")
    print("Earlier versions printed hard-coded numbers here (tree distortion, holonomy, forgetting).")
    print("No code computed them; the tree-distortion value belonged to the pure hyperbolic geometry, not the")
    print("unified manifold. Computed structural metrics: scratch/bench_geometric_manifolds.py.")
    print("A continual-learning comparison has NOT been measured yet.")

if __name__ == "__main__":
    run_experiment()
