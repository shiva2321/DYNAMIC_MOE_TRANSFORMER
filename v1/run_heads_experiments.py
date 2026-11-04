"""v1/run_heads_experiments.py: Run Part A experiments across removal sets and dev seeds.

Saves structured JSON results under results/v1/heads_r{rid}_s{seed}.json.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch

from costlab.env import fixed_threads
from growmoe.memory_baselines import ClassMean, KnnMemory, StreamingLDA
from growmoe.pools.base import CommonCfg
from m1.baselines_extra import MahalanobisProto, RandomFeatureRidge
import m1.beds
from v1.downdate import ExactLda, remove_classes

REMOVAL_SETS = {
    "R1": [0],
    "R2": [0, 1, 2, 3, 4],
    "R3": [101, 102, 103, 104, 105, 106],
    "R4": [77, 78],
}


def run_heads_experiment(rid: str, seed: int, out_dir: Path) -> Dict[str, Any]:
    """Execute Part A evaluation for removal set rid on dev seed."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"heads_{rid.lower()}_s{seed}.json"

    # Idempotence check
    if out_file.exists():
        try:
            cached = json.loads(out_file.read_text(encoding="utf-8"))
            if cached.get("status") == "complete":
                print(f"[SKIP] {out_file.name} already complete.")
                return cached
        except (OSError, ValueError):
            pass

    target_classes = REMOVAL_SETS[rid]
    target_set = set(target_classes)

    bed = m1.beds.build_bed("B3seq", 22, seed)

    # Collect stream data and separate removed samples
    all_batches: List[Tuple[torch.Tensor, torch.Tensor]] = []
    rem_x_list: List[torch.Tensor] = []
    rem_y_list: List[torch.Tensor] = []

    for x_b, y_b, r in bed.batches():
        all_batches.append((x_b, y_b))
        mask_rem = torch.isin(y_b, torch.tensor(target_classes, dtype=torch.long))
        if bool(mask_rem.any()):
            rem_x_list.append(x_b[mask_rem])
            rem_y_list.append(y_b[mask_rem])

    x_rem_all = torch.cat(rem_x_list, dim=0) if rem_x_list else torch.empty((0, bed.d))
    y_rem_all = torch.cat(rem_y_list, dim=0) if rem_y_list else torch.empty((0,), dtype=torch.long)
    rem_data = (x_rem_all, y_rem_all)

    # Collect retained test data across all regimes
    ret_test_x_list: List[torch.Tensor] = []
    ret_test_y_list: List[torch.Tensor] = []
    for r, (tx, ty) in bed.test_sets.items():
        ret_mask = ~torch.isin(ty, torch.tensor(target_classes, dtype=torch.long))
        if bool(ret_mask.any()):
            ret_test_x_list.append(tx[ret_mask])
            ret_test_y_list.append(ty[ret_mask])

    x_test_ret = torch.cat(ret_test_x_list, dim=0)
    y_test_ret = torch.cat(ret_test_y_list, dim=0)

    # Model initializers
    models_to_test = {
        "ClassMean": lambda: ClassMean(bed),
        "KnnMemory": lambda: KnnMemory(bed, k=5, capacity=None),
        "MahalanobisProto": lambda: MahalanobisProto(bed, shrinkage=0.1, scale=1.0),
        "RandomFeatureRidge": lambda: RandomFeatureRidge(bed, D=1024, ridge=0.1, scale=1.0),
        "ExactLda": lambda: ExactLda(bed, shrinkage=1e-2, scale=1.0),
        "StreamingLDA": lambda: StreamingLDA(bed, shrinkage=1e-2),
    }

    results: Dict[str, Any] = {
        "status": "complete",
        "removal_set": rid,
        "target_classes": target_classes,
        "seed": seed,
        "models": {},
    }

    for name, init_fn in models_to_test.items():
        # 1. Train full model
        m_full = init_fn()
        for xb, yb in all_batches:
            m_full.observe(xb, yb)

        # 2. Downdate model
        m_down = copy.deepcopy(m_full)
        downdate_meta = remove_classes(m_down, target_classes, data=rem_data if name == "RandomFeatureRidge" else None)

        # 3. Train fresh never-trained model
        m_never = init_fn()
        for xb, yb in all_batches:
            keep_m = ~torch.isin(yb, torch.tensor(target_classes, dtype=torch.long))
            if bool(keep_m.any()):
                m_never.observe(xb[keep_m], yb[keep_m])

        # 4. Measure difference of stored statistics
        stat_diffs: Dict[str, float] = {}
        if name == "ClassMean":
            stat_diffs["_sum"] = float((m_down._sum - m_never._sum).abs().max().item())
            stat_diffs["_cnt"] = float((m_down._cnt - m_never._cnt).abs().max().item())
        elif name == "KnnMemory":
            stat_diffs["_n"] = float(abs(m_down._n - m_never._n))
            if m_down._n == m_never._n:
                stat_diffs["_X"] = float((m_down.X - m_never.X).abs().max().item())
                stat_diffs["_Y"] = float((m_down.Y != m_never.Y).sum().item())
            else:
                stat_diffs["_X"] = float("nan")
        elif name == "MahalanobisProto":
            stat_diffs["_cnt"] = float((m_down._cnt - m_never._cnt).abs().max().item())
            stat_diffs["_sum"] = float((m_down._sum - m_never._sum).abs().max().item())
            stat_diffs["_outer"] = float((m_down._outer - m_never._outer).abs().max().item())
        elif name == "RandomFeatureRidge":
            stat_diffs["_G"] = float((m_down._G - m_never._G).abs().max().item())
            stat_diffs["_B"] = float((m_down._B - m_never._B).abs().max().item())
            stat_diffs["_cnt"] = float((m_down._cnt - m_never._cnt).abs().max().item())
        elif name == "ExactLda":
            stat_diffs["_cnt"] = float((m_down._cnt - m_never._cnt).abs().max().item())
            stat_diffs["_sum"] = float((m_down._sum - m_never._sum).abs().max().item())
            stat_diffs["_outer"] = float((m_down._outer - m_never._outer).abs().max().item())
        elif name == "StreamingLDA":
            stat_diffs["_cnt"] = float((m_down._cnt - m_never._cnt).abs().max().item())
            stat_diffs["_mu"] = float((m_down._mu - m_never._mu).abs().max().item())
            stat_diffs["_cov"] = float((m_down._cov - m_never._cov).abs().max().item())
            stat_diffs["_n"] = float(abs(m_down._n - m_never._n))

        max_stat_diff = max(stat_diffs.values()) if stat_diffs else 0.0

        # 5. Measure difference of solved weights
        weight_diffs: Dict[str, float] = {}
        if name == "MahalanobisProto":
            m_down._update_parameters()
            m_never._update_parameters()
            if m_down._inv_covs is not None and m_never._inv_covs is not None:
                weight_diffs["_inv_covs"] = float((m_down._inv_covs - m_never._inv_covs).abs().max().item())
            if m_down._means is not None and m_never._means is not None:
                weight_diffs["_means"] = float((m_down._means - m_never._means).abs().max().item())
        elif name == "RandomFeatureRidge":
            w_down = m_down._solve_weights()
            w_never = m_never._solve_weights()
            weight_diffs["_W"] = float((w_down - w_never).abs().max().item())
        elif name in ("ExactLda", "StreamingLDA"):
            wb_down = m_down._weights()
            wb_never = m_never._weights()
            if wb_down is not None and wb_never is not None:
                weight_diffs["w"] = float((wb_down[0] - wb_never[0]).abs().max().item())
                weight_diffs["b"] = float((wb_down[1] - wb_never[1]).abs().max().item())

        max_weight_diff = max(weight_diffs.values()) if weight_diffs else 0.0

        # 6. Evaluate predictions on retained test set
        pred_down = m_down.predict(x_test_ret)
        pred_nev = m_never.predict(x_test_ret)

        retained_indices = [c for c in range(bed.n_classes) if c not in target_set]

        probs_down = pred_down.probs[:, retained_indices]
        probs_nev = pred_nev.probs[:, retained_indices]
        max_prob_diff = float((probs_down - probs_nev).abs().max().item())

        argmax_down = pred_down.probs.argmax(dim=-1).cpu()
        argmax_nev = pred_nev.probs.argmax(dim=-1).cpu()
        argmax_agreement = float((argmax_down == argmax_nev).float().mean().item())

        acc_down = float((argmax_down == y_test_ret.cpu()).float().mean().item())
        acc_nev = float((argmax_nev == y_test_ret.cpu()).float().mean().item())

        is_exact = bool(max_prob_diff <= 1e-6 and max_stat_diff <= 1e-6 and argmax_agreement == 1.0)

        results["models"][name] = {
            "classification": "exact" if is_exact else "approximate",
            "max_stat_diff": max_stat_diff,
            "stat_diffs": stat_diffs,
            "max_weight_diff": max_weight_diff,
            "weight_diffs": weight_diffs,
            "max_prob_diff": max_prob_diff,
            "argmax_agreement": argmax_agreement,
            "test_accuracy_downdated": acc_down,
            "test_accuracy_never": acc_nev,
            "downdate_meta": downdate_meta,
        }

    # 7. Comparison between StreamingLDA and ExactLda on the exact same data
    el_model = ExactLda(bed, shrinkage=1e-2, scale=1.0)
    for xb, yb in all_batches:
        el_model.observe(xb, yb)

    slda_model = StreamingLDA(bed, shrinkage=1e-2)
    for xb, yb in all_batches:
        slda_model.observe(xb, yb)

    # Compute ExactLda's pooled covariance directly
    total_n_el = float(el_model._cnt.sum().item())
    pooled_scatter = torch.zeros(bed.d, bed.d, dtype=torch.float64)
    for c in range(bed.n_classes):
        nc = float(el_model._cnt[c].item())
        if nc > 0:
            muc = el_model._sum[c] / nc
            sc = el_model._outer[c] - nc * torch.outer(muc, muc)
            pooled_scatter += sc
    cov_exact_lda = pooled_scatter / total_n_el
    cov_streaming_lda = slda_model._cov

    cov_diff = (cov_exact_lda - cov_streaming_lda).abs()
    cov_max_diff = float(cov_diff.max().item())
    cov_frobenius = float(torch.linalg.norm(cov_exact_lda - cov_streaming_lda).item())

    # Retained test accuracy difference
    pred_el = el_model.predict(x_test_ret).probs.argmax(dim=-1).cpu()
    pred_slda = slda_model.predict(x_test_ret).probs.argmax(dim=-1).cpu()
    acc_el = float((pred_el == y_test_ret.cpu()).float().mean().item())
    acc_slda = float((pred_slda == y_test_ret.cpu()).float().mean().item())

    results["streaming_lda_vs_exact_lda"] = {
        "covariance_max_diff": cov_max_diff,
        "covariance_frobenius_diff": cov_frobenius,
        "exact_lda_retained_accuracy": acc_el,
        "streaming_lda_retained_accuracy": acc_slda,
        "accuracy_diff": acc_el - acc_slda,
        "agreement": float((pred_el == pred_slda).float().mean().item()),
    }

    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[DONE] Wrote {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Run Part A closed-form heads removal experiments")
    parser.add_argument("--rid", choices=["R1", "R2", "R3", "R4"], default="R1")
    parser.add_argument("--seed", type=int, default=800)
    parser.add_argument("--out-dir", type=str, default="results/v1")
    args = parser.parse_args()

    with fixed_threads(n=2, cpus=[8, 10]):
        run_heads_experiment(args.rid, args.seed, Path(args.out_dir))


if __name__ == "__main__":
    main()
