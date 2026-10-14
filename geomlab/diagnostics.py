"""
Geometry health diagnostics for the general-intelligence system.
Implements the diagnostic suite defined in GEOMETRY_ATLAS_HANDOFF.md Section 4:
- effective_rank (entropy of normalized singular values)
- isotropy (mean pairwise cosine and Wang-Isola uniformity loss on unit sphere)
- intrinsic_dim_twonn (TwoNN intrinsic dimension estimator)
- hubness_skew (Fisher-Pearson skewness of k-nearest neighbor occurrence frequency)
- norm_stats (norm distribution metrics and drift)
- operating_point_adapter (interface to yggformer.report.operating_point)
"""

import math
from typing import Dict, Optional, Union
import numpy as np
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as F


def effective_rank(X: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Computes the effective rank (Roy & Vetterli 2007) of representation matrix X.
    Defined as exp(H) where H is the Shannon entropy of normalized singular values:
      p_i = sigma_i / sum(sigma_j)
      H = - sum(p_i * ln(p_i))
      erank = exp(H)
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X_flat = X.detach()
    if not X_flat.is_complex():
        X_flat = X_flat.float()
    if X_flat.dim() > 2:
        X_flat = X_flat.reshape(-1, X_flat.shape[-1])
    elif X_flat.dim() == 1:
        return 1.0

    if X_flat.numel() == 0:
        return 0.0

    try:
        s = torch.linalg.svdvals(X_flat)
    except Exception:
        # Fallback to CPU svdvals if CUDA error occurs
        s = torch.linalg.svdvals(X_flat.cpu())

    s = s[s > eps]
    if len(s) == 0:
        return 0.0

    s_sum = s.sum()
    if s_sum <= 0:
        return 0.0

    p = s / s_sum
    # Clamp p to avoid log(0)
    p_clamped = torch.clamp(p, min=eps)
    entropy = -torch.sum(p * torch.log(p_clamped)).item()
    return float(math.exp(entropy))


def isotropy(X: torch.Tensor, eps: float = 1e-8, t: float = 2.0, max_samples: int = 5000) -> Dict[str, float]:
    """
    Computes spherical isotropy and uniformity metrics:
    - mean pairwise cosine: 1 / (N * (N-1)) * sum_{i != j} cos(x_i, x_j)
    - uniformity loss (Wang & Isola 2020): log E_{x,y} exp(-t * ||x - y||^2) on the unit sphere
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X_flat = X.detach()
    if not X_flat.is_complex():
        X_flat = X_flat.float()
    if X_flat.dim() > 2:
        X_flat = X_flat.reshape(-1, X_flat.shape[-1])

    N = X_flat.shape[0]
    if N < 2:
        return {"mean_pairwise_cosine": 1.0, "uniformity_loss": 0.0}

    # Subsample if N exceeds max_samples to conserve memory and stay within charter limits
    if N > max_samples:
        indices = torch.randperm(N)[:max_samples]
        X_flat = X_flat[indices]
        N = max_samples

    u = F.normalize(X_flat, p=2, dim=-1, eps=eps)
    if u.is_complex():
        G = torch.matmul(u, u.conj().t()).real  # Pairwise cosine Gram matrix (N, N)
    else:
        G = torch.matmul(u, u.t())  # Pairwise cosine Gram matrix (N, N)

    # Exclude self-similarities on the diagonal
    num_pairs = N * (N - 1)
    sum_cos = (G.sum() - G.diagonal().sum()).item()
    mean_cos = float(sum_cos / num_pairs)

    # Uniformity loss: ||u_i - u_j||^2 = 2 - 2 * G_{i,j}
    # exp(-t * ||u_i - u_j||^2) = exp(-2t * (1 - G_{i,j}))
    dist_sq = 2.0 - 2.0 * G
    dist_sq.fill_diagonal_(float("inf"))  # Exclude diagonal

    kernel = torch.exp(-t * dist_sq)
    # Sum over off-diagonal entries
    kernel_sum = kernel[dist_sq != float("inf")].sum().item()
    mean_kernel = max(kernel_sum / num_pairs, 1e-12)
    uniformity_loss = float(math.log(mean_kernel))

    return {
        "mean_pairwise_cosine": mean_cos,
        "uniformity_loss": uniformity_loss,
    }


def intrinsic_dim_twonn(X: torch.Tensor, discard_fraction: float = 0.1, eps: float = 1e-12) -> float:
    """
    TwoNN intrinsic dimension estimator (Facco et al., Scientific Reports 2017).
    Estimates intrinsic dimensionality using the ratio of second-to-first nearest neighbor distances:
      mu_i = r_{i,2} / r_{i,1} >= 1
    Under Poisson process assumptions, -ln(1 - F(mu)) = d * ln(mu).
    Linear regression through the origin on the empirical CDF yields intrinsic dimension d.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X_flat = X.detach().float()
    if X_flat.dim() > 2:
        X_flat = X_flat.reshape(-1, X_flat.shape[-1])

    N = X_flat.shape[0]
    if N < 4:
        return 1.0

    # Pairwise Euclidean distances
    # Chunked or direct depending on N
    if N > 4000:
        # Subsample to stay memory-lean
        indices = torch.randperm(N)[:4000]
        X_flat = X_flat[indices]
        N = 4000

    # Compute exact Euclidean distances in float64 without matmul shortcut
    # to avoid catastrophic floating-point cancellation on closely-spaced points.
    X_d = X_flat.double()
    dists = torch.cdist(X_d, X_d, compute_mode="donot_use_mm_for_euclid_dist")
    dists.fill_diagonal_(float("inf"))

    # Top 2 nearest neighbors
    top2 = torch.topk(dists, k=2, dim=-1, largest=False).values
    r1 = top2[:, 0]
    r2 = top2[:, 1]

    # Filter invalid points where r1 == 0 or r2 <= r1
    valid_mask = (r1 > eps) & (r2 > r1)
    if valid_mask.sum() < 4:
        return 1.0

    r1_val = r1[valid_mask]
    r2_val = r2[valid_mask]
    mu = (r2_val / r1_val).cpu().numpy()

    mu_sorted = np.sort(mu)
    N_valid = len(mu_sorted)

    # Discard upper tail (e.g. top discard_fraction) as prescribed by Facco et al.
    n_fit = int(N_valid * (1.0 - discard_fraction))
    n_fit = max(n_fit, 3)

    mu_fit = mu_sorted[:n_fit]
    # Empirical CDF F(mu) = i / N_valid
    F_emp = np.arange(1, n_fit + 1) / float(N_valid)

    x = np.log(mu_fit)
    y = -np.log(1.0 - F_emp)

    # Linear regression through origin: y = d * x => d = sum(x * y) / sum(x^2)
    denom = np.sum(x * x)
    if denom <= eps:
        return 1.0

    slope = float(np.sum(x * y) / denom)
    return max(slope, 0.0)


def hubness_skew(X: torch.Tensor, k: int = 5, max_samples: int = 3000) -> float:
    """
    Computes the hubness skewness (Fisher-Pearson 3rd standardized moment)
    of the k-occurrence distribution N_k(i).
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X_flat = X.detach().float()
    if X_flat.dim() > 2:
        X_flat = X_flat.reshape(-1, X_flat.shape[-1])

    N = X_flat.shape[0]
    if N <= k:
        return 0.0

    if N > max_samples:
        indices = torch.randperm(N)[:max_samples]
        X_flat = X_flat[indices]
        N = max_samples

    X_d = X_flat.double()
    dists = torch.cdist(X_d, X_d, compute_mode="donot_use_mm_for_euclid_dist")
    dists.fill_diagonal_(float("inf"))

    knn_indices = torch.topk(dists, k=k, dim=-1, largest=False).indices
    counts = torch.bincount(knn_indices.reshape(-1), minlength=N).float().cpu().numpy()

    # Fisher-Pearson skewness
    skew_val = float(stats.skew(counts, bias=False))
    return 0.0 if math.isnan(skew_val) else skew_val


def norm_stats(X: torch.Tensor) -> Dict[str, float]:
    """
    Computes distribution statistics and drift of vector L2 norms:
    mean, std, median, min, max.
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    X_flat = X.detach().float()
    if X_flat.dim() > 2:
        X_flat = X_flat.reshape(-1, X_flat.shape[-1])

    norms = torch.linalg.norm(X_flat, dim=-1).cpu().numpy()
    if len(norms) == 0:
        return {
            "mean_norm": 0.0,
            "std_norm": 0.0,
            "median_norm": 0.0,
            "min_norm": 0.0,
            "max_norm": 0.0,
        }

    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms, ddof=1)) if len(norms) > 1 else 0.0,
        "median_norm": float(np.median(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
    }


def operating_point_adapter(
    model: nn.Module,
    batch: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict:
    """
    Adapter to yggformer.report.operating_point.
    If model is a YggFormer instance, delegates directly to yggformer's operating_point.
    Otherwise, inspects model layers, computing effective rank and norm health.
    """
    # Check if model is YggFormer
    if model.__class__.__name__ == "YggFormer" and hasattr(model, "blocks"):
        try:
            from yggformer.report import operating_point
            if batch is not None:
                return operating_point(model, batch, **kwargs)
        except Exception:
            pass

    # Generic module inspection
    layers_report = []
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            erank = effective_rank(param.data)
            nstats = norm_stats(param.data)
            layers_report.append({
                "name": name,
                "shape": list(param.shape),
                "effective_rank": erank,
                "norm_stats": nstats,
            })

    return {
        "model_type": model.__class__.__name__,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "layers": layers_report,
        "frac_heads_euclidean": None,
    }
