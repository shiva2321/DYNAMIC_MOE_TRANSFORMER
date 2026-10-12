"""Conformal p-values for novelty (low score = novel) and Benjamini-Hochberg selection.

Validity: if a test sample is exchangeable with the calibration scores, P(p <= alpha) <= alpha for any
alpha (finite-sample, distribution-free). Conformal p-values for several test points computed against
the same calibration set are positively dependent, so BH controls the FDR (Bates et al., Ann. Stat. 2023).
"""

import torch
from torch import Tensor


def conformal_pvalues(cal: Tensor, test: Tensor) -> Tensor:
    cal = torch.as_tensor(cal).reshape(-1)
    test = torch.as_tensor(test).reshape(-1)
    n = cal.numel()
    if n == 0:
        raise ValueError("conformal_pvalues needs at least one calibration score")
    cal_sorted, _ = torch.sort(cal)
    counts = torch.searchsorted(cal_sorted.contiguous(), test.to(cal_sorted.dtype).contiguous(), right=True)
    return ((1.0 + counts.to(torch.float64)) / (n + 1.0)).to(torch.float32)


def benjamini_hochberg(p: Tensor, q: float) -> Tensor:
    """Boolean mask of rejections (= flagged novel) at FDR level ``q``."""
    p = torch.as_tensor(p).reshape(-1)
    m = p.numel()
    if m == 0:
        return torch.zeros(0, dtype=torch.bool)
    sorted_p, order = torch.sort(p)
    thresholds = q * torch.arange(1, m + 1, dtype=sorted_p.dtype) / m
    passed = (sorted_p <= thresholds).nonzero()
    mask = torch.zeros(m, dtype=torch.bool)
    if passed.numel():
        k = int(passed.max().item())
        mask[order[: k + 1]] = True
    return mask
