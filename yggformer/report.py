"""
operating_point report: Audit metric extraction for YggFormer architectures.
Implements the operating-point diagnostic per section 5 of YGGDRASIL_ARCH_HANDOFF.md.
"""

import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .model import YggFormer


def compute_quantiles(arr: torch.Tensor) -> Dict[str, float]:
    vals = arr.detach().flatten().float()
    if vals.numel() == 0:
        return {"median": 0.0, "p95": 0.0, "max": 0.0}
    vals_np = vals.numpy()
    return {
        "median": float(np.median(vals_np)),
        "p95": float(np.percentile(vals_np, 95)),
        "max": float(np.max(vals_np)),
    }


def operating_point(
    model: YggFormer,
    batch: torch.Tensor,
    qk_layers: Optional[List[int]] = None,
) -> Dict:
    """
    Computes a JSON-serializable operating-point diagnostic for a trained YggFormer model.
    """
    model.eval()
    with torch.no_grad():
        _ = model(batch, is_causal=True, record_stats=True)

    layers_report = []
    all_median_u_k = []

    layers_to_inspect = qk_layers if qk_layers is not None else list(range(len(model.blocks)))

    for l_idx in layers_to_inspect:
        block = model.blocks[l_idx]
        attn = block.attn
        stats = attn.last_stats

        geom = stats.get("geometry") if stats else attn.geometry
        if stats is None or geom in {"euclid", "euclid_gate", "euclid_temp"}:
            layers_report.append({
                "layer_idx": l_idx,
                "geometry": geom,
                "c": None,
                "u_q_median": None,
                "u_q_p95": None,
                "u_q_max": None,
                "u_k_median": None,
                "u_k_p95": None,
                "u_k_max": None,
                "top1_mass": stats.get("top1_mass") if stats else None,
                "entropy": stats.get("entropy") if stats else None,
                "var_share_hyp": 0.0,
                "var_share_phase": 0.0,
                "var_share_bias": 0.0,
            })
            continue

        c_vals = stats["c"]
        u_q = stats["u_q"]  # (B, H, T)
        u_k = stats["u_k"]  # (B, H, T)

        q_q = compute_quantiles(u_q)
        q_k = compute_quantiles(u_k)

        all_median_u_k.append(q_k["median"])

        # Compute logit components and variance shares
        q = stats["q"]  # (B, H, T, DH)
        k = stats["k"]
        c_t = torch.tensor(c_vals).view(1, attn.n_heads, 1, 1)
        tau_t = torch.tensor(stats["tau"]).view(1, attn.n_heads, 1, 1)

        # Lorentz logit part: (1/tau) * (-q0 k0 + q.k)
        q0 = torch.sqrt(1.0 / c_t + (q ** 2).sum(dim=-1, keepdim=True))
        k0 = torch.sqrt(1.0 / c_t + (k ** 2).sum(dim=-1, keepdim=True))
        lor_logits = (-q0 @ k0.transpose(-2, -1) + q @ k.transpose(-2, -1)) / tau_t

        var_hyp = float(torch.var(lor_logits).item())
        var_phase = 0.0
        var_bias = 0.0

        if stats["th_q"] is not None and stats["th_k"] is not None and stats["kappa"] is not None:
            th_q = stats["th_q"]
            th_k = stats["th_k"]
            kappa_t = torch.tensor(stats["kappa"]).view(1, attn.n_heads, 1, 1)
            # phase cos similarity
            phase_logits = kappa_t * torch.cos(th_q.unsqueeze(-2) - th_k.unsqueeze(-3)).sum(dim=-1)
            var_phase = float(torch.var(phase_logits).item())
        else:
            phase_logits = 0.0

        if stats["key_bias"] is not None:
            # key_bias is (B, H, T), expands to (B, H, 1, T)
            bias_logits = stats["key_bias"].unsqueeze(-2)
            var_bias = float(torch.var(bias_logits).item())
        else:
            bias_logits = 0.0

        total_logits = lor_logits + phase_logits + bias_logits
        total_var = var_hyp + var_phase + var_bias + 1e-12

        # Attention metrics (softmax over causal mask)
        B, H, T_q, T_k = total_logits.shape
        if T_q == T_k:
            causal_mask = torch.triu(torch.full((T_q, T_k), float("-inf"), device=total_logits.device), diagonal=1)
        else:
            causal_mask = torch.zeros((T_q, T_k), device=total_logits.device)
            causal_mask[:, 1:] = torch.triu(torch.full((T_q, T_q), float("-inf"), device=total_logits.device), diagonal=1)
        masked_logits = total_logits + causal_mask
        attn_weights = F.softmax(masked_logits, dim=-1)


        top1_mass = float(attn_weights.max(dim=-1).values.mean().item())
        # Entropy: - sum p log p
        log_attn = F.log_softmax(masked_logits, dim=-1)
        entropy = float(-(attn_weights * log_attn).nan_to_num(0.0).sum(dim=-1).mean().item())

        layers_report.append({
            "layer_idx": l_idx,
            "geometry": attn.geometry,
            "c": c_vals,
            "u_q_median": q_q["median"],
            "u_q_p95": q_q["p95"],
            "u_q_max": q_q["max"],
            "u_k_median": q_k["median"],
            "u_k_p95": q_k["p95"],
            "u_k_max": q_k["max"],
            "top1_mass": top1_mass,
            "entropy": entropy,
            "var_share_hyp": var_hyp / total_var,
            "var_share_phase": var_phase / total_var,
            "var_share_bias": var_bias / total_var,
        })

    frac_euclidean = float(np.mean([m < 0.37 for m in all_median_u_k])) if all_median_u_k else 1.0

    return {
        "frac_heads_euclidean": frac_euclidean,
        "layers": layers_report,
    }
