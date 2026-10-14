"""
Phase 1: Real Latent Geometry Health and Superposition Capacity.
Implements the tasks, models, and capacity evaluation specified in
research/geomlab_results/phase1/PREREG.md:
- Real text dataset loader (Banking77 intent queries)
- Models: EuclideanModel, SphericalModel, TorusModel
- Geometry health diagnostics (Effective rank, Uniformity, Mutual coherence)
- Bundle superposition capacity benchmark (M in 1, 2, 4, 8, 16, 32)
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from growmoe.frontend import HashFrontend
from growmoe.streams_real import _load_banking77
from geomlab.diagnostics import effective_rank, isotropy


# ---------------------------------------------------------------------------
# Data Loading and Feature Extraction
# ---------------------------------------------------------------------------

def load_banking_dataset(seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[str], np.ndarray]:
    """
    Loads Banking77 dataset, applies deterministic HashFrontend (1024 dims),
    and returns (train_feats, train_labels, val_feats, val_labels, codebook_texts, codebook_labels).
    """
    df_train, df_test = _load_banking77()
    
    # Map label names to IDs
    unique_labels = sorted(df_train["label"].unique())
    label_to_id = {lbl: i for i, lbl in enumerate(unique_labels)}
    
    train_texts = df_train["text"].tolist()
    train_labels = np.array([label_to_id[lbl] for lbl in df_train["label"]], dtype=np.int64)
    
    val_texts = df_test["text"].tolist()
    val_labels = np.array([label_to_id[lbl] for lbl in df_test["label"]], dtype=np.int64)

    fe = HashFrontend(d=1024)
    # Batch feature extraction
    train_feats = fe(train_texts).float()
    val_feats = fe(val_texts).float()

    # Create balanced evaluation codebook of N = 1000 items
    rng = np.random.RandomState(seed)
    # Sample 1000 items from test set (or combined)
    all_texts = val_texts
    all_labels = val_labels
    if len(all_texts) < 1000:
        all_texts = train_texts + val_texts
        all_labels = np.concatenate([train_labels, val_labels])
    
    perm = rng.permutation(len(all_texts))[:1000]
    codebook_texts = [all_texts[i] for i in perm]
    codebook_labels = all_labels[perm]

    return (
        train_feats,
        torch.from_numpy(train_labels),
        val_feats,
        torch.from_numpy(val_labels),
        codebook_texts,
        codebook_labels,
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class EuclideanModel(nn.Module):
    """
    Standard unconstrained Euclidean representation baseline.
    """
    def __init__(self, in_dim: int = 1024, d_model: int = 256, num_classes: int = 77):
        super().__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(in_dim, 2 * d_model)
        self.norm = nn.LayerNorm(2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.norm(h)
        z = self.fc2(h)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        logits = self.classifier(z)
        return logits, z


class SphericalModel(nn.Module):
    """
    Unit hypersphere S^{D-1} normalized representation.
    """
    def __init__(self, in_dim: int = 1024, d_model: int = 256, num_classes: int = 77):
        super().__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(in_dim, 2 * d_model)
        self.norm = nn.LayerNorm(2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, d_model)
        self.prototypes = nn.Parameter(torch.randn(num_classes, d_model))
        self.beta = nn.Parameter(torch.tensor(10.0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(x))
        h = self.norm(h)
        z = self.fc2(h)
        z_unit = F.normalize(z, p=2, dim=-1)
        return z_unit

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_unit = self.encode(x)
        proto_unit = F.normalize(self.prototypes, p=2, dim=-1)
        logits = self.beta * torch.matmul(z_unit, proto_unit.T)
        return logits, z_unit


class TorusModel(nn.Module):
    """
    Flat Clifford Torus T^D representation (complex unit phasors).
    """
    def __init__(self, in_dim: int = 1024, d_model: int = 256, num_classes: int = 77):
        super().__init__()
        self.d_model = d_model
        self.fc1 = nn.Linear(in_dim, 2 * d_model)
        self.norm = nn.LayerNorm(2 * d_model)
        self.fc2 = nn.Linear(2 * d_model, 2 * d_model)
        self.proto_phases = nn.Parameter(torch.empty(num_classes, d_model).uniform_(-math.pi, math.pi))
        self.beta = nn.Parameter(torch.tensor(10.0))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Output pairs (u, v)
        h = F.gelu(self.fc1(x))
        h = self.norm(h)
        uv = self.fc2(h).reshape(-1, self.d_model, 2)
        # Normalize to unit phasors (cos theta, sin theta)
        uv_norm = F.normalize(uv, p=2, dim=-1)
        # Return phase angles in [-pi, pi)
        phases = torch.atan2(uv_norm[:, :, 1], uv_norm[:, :, 0])
        return phases

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        phases = self.encode(x)  # (B, D)
        # Cosine difference with prototype phases
        # phases: (B, 1, D), proto_phases: (1, num_classes, D)
        diff = phases.unsqueeze(1) - self.proto_phases.unsqueeze(0)
        cos_sim = torch.cos(diff).mean(dim=-1)
        logits = self.beta * cos_sim
        return logits, phases


# ---------------------------------------------------------------------------
# Superposition Capacity Benchmark
# ---------------------------------------------------------------------------

def evaluate_bundle_capacity(
    representations: torch.Tensor,
    is_torus: bool = False,
    bundle_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32),
    num_trials: int = 100,
    seed: int = 0,
) -> Dict[int, float]:
    """
    Evaluates superposition capacity:
    Samples M distinct items from representations (N x D), forms bundle S = sum x_m,
    and checks if each item x_m is successfully retrieved among the top-M nearest neighbors.
    Returns: {M: retrieval_accuracy_percent}
    """
    rng = np.random.RandomState(seed)
    N, D = representations.shape
    device = representations.device
    results = {}

    if is_torus:
        # Complex unit phasors
        phasors = torch.exp(1j * representations)  # (N, D)

    for M in bundle_sizes:
        if M > N:
            continue
        hits = 0
        total_queries = 0

        for _ in range(num_trials):
            # Sample M distinct indices
            chosen_indices = rng.choice(N, size=M, replace=False)
            idx_set = set(chosen_indices.tolist())

            if is_torus:
                # Torus superposition: bundle is sum of phasors
                bundle = phasors[chosen_indices].sum(dim=0, keepdim=True)  # (1, D)
                # Compute inner product Re(phasors @ bundle^H)
                sim = (phasors @ bundle.conj().T).real.squeeze(-1)  # (N,)
            else:
                # Euclidean / Spherical: normalize to unit length for query
                items = representations[chosen_indices]
                bundle = items.sum(dim=0, keepdim=True)
                bundle_unit = F.normalize(bundle, p=2, dim=-1)
                rep_unit = F.normalize(representations, p=2, dim=-1)
                sim = torch.matmul(rep_unit, bundle_unit.T).squeeze(-1)  # (N,)

            # Top-M nearest neighbors to bundle
            top_m_indices = torch.topk(sim, k=M).indices.cpu().numpy()
            top_m_set = set(top_m_indices.tolist())

            # Count how many of the true bundled items are in the top-M
            recovered = len(idx_set.intersection(top_m_set))
            hits += recovered
            total_queries += M

        acc = (hits / total_queries) * 100.0 if total_queries > 0 else 0.0
        results[M] = float(acc)

    return results


def compute_mutual_coherence(vectors: torch.Tensor) -> float:
    """
    Computes max absolute pairwise cosine similarity max_{i != j} |cos(x_i, x_j)|.
    """
    v_norm = F.normalize(vectors, p=2, dim=-1)
    cos_matrix = torch.matmul(v_norm, v_norm.T).abs()
    # Mask diagonal
    N = cos_matrix.shape[0]
    cos_matrix.fill_diagonal_(0.0)
    return float(cos_matrix.max().item())
