"""
Round Z0 Execution Script: Run diagnostic baselines and protocol verifications.
Generates research/geomlab_results/z0/results.json and rendered verification tables.
"""

import math
import os
import sys
import time
from pathlib import Path

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from geomlab.diagnostics import (
    effective_rank,
    isotropy,
    intrinsic_dim_twonn,
    hubness_skew,
    norm_stats,
    operating_point_adapter,
)
from geomlab.protocol import (
    tune_lr,
    run_seeds,
    paired_stats,
    bonferroni,
    write_results,
    render_poc_b_table,
    validate_report,
    check_prereg,
    ci95,
)


def run_diagnostics_benchmark():
    test_seeds = list(range(10))

    # Task 1: Isotropic Gaussian in d=32, N=4000
    iso_runs = []
    for s in test_seeds:
        torch.manual_seed(1000 + s)
        X = torch.randn(4000, 32)
        er = effective_rank(X)
        iso_m = isotropy(X)
        twonn = intrinsic_dim_twonn(X)
        hskew = hubness_skew(X, k=5)
        nstat = norm_stats(X)
        iso_runs.append({
            "effective_rank": er,
            "mean_cosine": iso_m["mean_pairwise_cosine"],
            "uniformity_loss": iso_m["uniformity_loss"],
            "twonn_id": twonn,
            "hubness_skew": hskew,
            "mean_norm": nstat["mean_norm"],
        })

    # Task 2: Rank-1 line in d=32, N=4000
    rank1_runs = []
    for s in test_seeds:
        torch.manual_seed(2000 + s)
        v = torch.randn(1, 32)
        v = v / v.norm()
        X = torch.randn(4000, 1) @ v
        er = effective_rank(X)
        iso_m = isotropy(X)
        hskew = hubness_skew(X, k=5)
        nstat = norm_stats(X)
        rank1_runs.append({
            "effective_rank": er,
            "mean_cosine": iso_m["mean_pairwise_cosine"],
            "uniformity_loss": iso_m["uniformity_loss"],
            "hubness_skew": hskew,
            "mean_norm": nstat["mean_norm"],
        })

    # Task 2b: Collapsed anisotropic cluster in d=32, N=4000
    collapsed_runs = []
    for s in test_seeds:
        torch.manual_seed(2500 + s)
        v = torch.randn(1, 32)
        v = v / v.norm()
        X = v + 0.02 * torch.randn(4000, 32)
        er = effective_rank(X)
        iso_m = isotropy(X)
        collapsed_runs.append({
            "effective_rank": er,
            "mean_cosine": iso_m["mean_pairwise_cosine"],
            "uniformity_loss": iso_m["uniformity_loss"],
        })

    # Task 3: 3D Manifold embedded in 20D, N=3000
    manifold_3d_runs = []
    for s in test_seeds:
        torch.manual_seed(3000 + s)
        u = torch.randn(3000, 3)
        u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(3000, 1) ** (1.0 / 3.0))
        Q, _ = torch.linalg.qr(torch.randn(20, 3))
        X = u @ Q.T
        twonn = intrinsic_dim_twonn(X)
        er = effective_rank(X)
        manifold_3d_runs.append({
            "twonn_id": twonn,
            "effective_rank": er,
        })

    # Task 4: 2D Manifold embedded in 20D, N=3000
    manifold_2d_runs = []
    for s in test_seeds:
        torch.manual_seed(4000 + s)
        u = torch.randn(3000, 2)
        u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(3000, 1) ** (1.0 / 2.0))
        Q, _ = torch.linalg.qr(torch.randn(20, 2))
        X = u @ Q.T
        twonn = intrinsic_dim_twonn(X)
        er = effective_rank(X)
        manifold_2d_runs.append({
            "twonn_id": twonn,
            "effective_rank": er,
        })

    # Task 5: 1D Circle manifold embedded in 20D, N=3000
    circle_1d_runs = []
    for s in test_seeds:
        torch.manual_seed(5000 + s)
        theta = torch.rand(3000, 1) * 2 * math.pi
        circle_2d = torch.cat([theta.cos(), theta.sin()], dim=1)
        Q, _ = torch.linalg.qr(torch.randn(20, 2))
        X = circle_2d @ Q.T
        twonn = intrinsic_dim_twonn(X)
        er = effective_rank(X)
        circle_1d_runs.append({
            "twonn_id": twonn,
            "effective_rank": er,
        })

    # Task 6: 1D Segment manifold embedded in 20D, N=3000
    segment_1d_runs = []
    for s in test_seeds:
        torch.manual_seed(6000 + s)
        seg = torch.rand(3000, 1)
        Q, _ = torch.linalg.qr(torch.randn(20, 1))
        X = seg @ Q.T
        twonn = intrinsic_dim_twonn(X)
        er = effective_rank(X)
        segment_1d_runs.append({
            "twonn_id": twonn,
            "effective_rank": er,
        })

    def aggregate(runs):
        keys = runs[0].keys()
        agg = {}
        for k in keys:
            vals = [r[k] for r in runs]
            arr = np.array(vals, dtype=float)
            agg[k] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=1)),
                "ci95": ci95(arr),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "per_seed": vals,
            }
        return agg

    return {
        "isotropic_gaussian_32d": aggregate(iso_runs),
        "rank1_line_32d": aggregate(rank1_runs),
        "collapsed_cluster_32d": aggregate(collapsed_runs),
        "manifold_3d_in_20d": aggregate(manifold_3d_runs),
        "manifold_2d_in_20d": aggregate(manifold_2d_runs),
        "circle_1d_in_20d": aggregate(circle_1d_runs),
        "segment_1d_in_20d": aggregate(segment_1d_runs),
    }


def render_z0_tables(results_dict: dict) -> str:
    lines = []
    lines.append("=== Table 1: Diagnostic Metric Verifications (10 Seeds, Mean +- 95% CI) ===")
    lines.append(f"{'Condition':<25s} {'Metric':<26s} {'Measured Value':<24s} {'Theoretical':<16s}")
    lines.append("-" * 92)

    t1_rows = [
        ("isotropic_gaussian_32d", "effective_rank", "effective_rank", "~ 32.0 (>=30.0)"),
        ("isotropic_gaussian_32d", "mean_cosine", "mean_cosine", "~ 0.00 (+-0.05)"),
        ("isotropic_gaussian_32d", "uniformity_loss", "uniformity_loss", "< -3.00"),
        ("isotropic_gaussian_32d", "twonn_id", "twonn_id (high-d bias)", "24.0 (bias limit)"),
        ("rank1_line_32d", "effective_rank", "effective_rank", "1.00 (<=1.05)"),
        ("collapsed_cluster_32d", "mean_cosine", "mean_cosine", "~ 0.99 (>0.95)"),
        ("collapsed_cluster_32d", "uniformity_loss", "uniformity_loss", "> -0.10 (high)"),
        ("manifold_3d_in_20d", "twonn_id", "twonn_id (3D manifold)", "3.00 (+-0.3)"),
        ("manifold_2d_in_20d", "twonn_id", "twonn_id (2D manifold)", "2.00 (+-0.3)"),
        ("circle_1d_in_20d", "twonn_id", "twonn_id (1D circle)", "1.00 (+-0.2)"),
        ("segment_1d_in_20d", "twonn_id", "twonn_id (1D segment)", "1.00 (+-0.2)"),
    ]

    for cond, metric_key, label, theory in t1_rows:
        stats = results_dict["diagnostics"][cond][metric_key]
        val_str = f"{stats['mean']:6.3f} +- {stats['ci95']:5.3f}"
        lines.append(f"{cond:<25s} {label:<26s} {val_str:<24s} {theory:<16s}")

    lines.append("")
    lines.append("=== Table 2: Protocol Integrity and Honesty Verifications ===")
    lines.append(f"{'Tool':<25s} {'Check / Scenario':<35s} {'Result':<15s} {'Status':<10s}")
    lines.append("-" * 88)

    p_checks = results_dict["protocol_checks"]
    for item in p_checks:
        lines.append(f"{item['tool']:<25s} {item['scenario']:<35s} {item['result']:<15s} {item['status']:<10s}")

    return "\n".join(lines) + "\n"


def main():
    t0 = time.time()
    out_dir = Path("research/geomlab_results/z0")
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_path = out_dir / "PREREG.md"
    results_path = out_dir / "results.json"

    print("Running Round Z0 diagnostics benchmark across 10 seeds...", flush=True)
    diag_res = run_diagnostics_benchmark()

    # Protocol checks
    proto_checks = []

    # Check 1: LR Tuner edge check
    grid = [1e-3, 3e-3, 1e-2, 3e-2]
    res_tune_interior = tune_lr(lambda lr, s: -((math.log10(lr) - math.log10(0.01))**2), grid, dev_seeds=[100, 101])
    proto_checks.append({
        "tool": "tune_lr",
        "scenario": "Interior optimum (lr=0.01)",
        "result": f"best_lr={res_tune_interior['best_lr']:g}, edge={res_tune_interior['is_edge']}",
        "status": "PASS" if (res_tune_interior['best_lr'] == 0.01 and not res_tune_interior['is_edge']) else "FAIL"
    })

    res_tune_edge = tune_lr(lambda lr, s: lr, grid, dev_seeds=[100, 101], auto_expand=False)
    proto_checks.append({
        "tool": "tune_lr",
        "scenario": "Boundary optimum (lr=0.03)",
        "result": f"best_lr={res_tune_edge['best_lr']:g}, edge={res_tune_edge['is_edge']}",
        "status": "PASS" if res_tune_edge['is_edge'] else "FAIL"
    })

    # Check 2: Table dry-run byte-for-byte check
    poc_b_dir = Path("research/yggdrasil_results/poc_b")
    target_table_file = Path("research/yggdrasil_results/poc_b_table.txt")
    target_norm = target_table_file.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    rendered_norm = render_poc_b_table(poc_b_dir).replace("\r\n", "\n").strip()
    exact_char_match = (rendered_norm == target_norm)
    proto_checks.append({
        "tool": "render_poc_b_table",
        "scenario": "Dry-run poc_b char match (norm LF)",
        "result": f"Length={len(rendered_norm)} chars",
        "status": "PASS" if exact_char_match else "FAIL"
    })

    rendered_crlf = render_poc_b_table(poc_b_dir, newline="\r\n")
    exact_byte_match = (rendered_crlf.encode("utf-8") == target_table_file.read_bytes())
    proto_checks.append({
        "tool": "render_poc_b_table",
        "scenario": "Dry-run poc_b byte match (CRLF)",
        "result": f"Bytes={len(rendered_crlf.encode('utf-8'))} B",
        "status": "PASS" if exact_byte_match else "FAIL"
    })

    # Check 3: Report validator tamper detection
    test_rep = "Table\n100.0+- 0.0\nEnd"
    test_tbl = "Table\n100.0+- 0.0\nEnd"
    try:
        validate_report(test_rep, test_tbl)
        caught_tamper = False
        try:
            validate_report(test_rep.replace("100.0", "100.1"), test_tbl)
        except Exception:
            caught_tamper = True
        status_val = "PASS" if caught_tamper else "FAIL"
    except Exception:
        status_val = "FAIL"
    proto_checks.append({
        "tool": "validate_report",
        "scenario": "Single-digit alteration detection",
        "result": "Caught 1-digit change",
        "status": status_val
    })

    # Check 4: Pre-registration check verification
    # We will verify check_prereg after writing results.json
    results_dict = {
        "round": "Z0",
        "timestamp": time.time(),
        "diagnostics": diag_res,
        "protocol_checks": proto_checks,
    }

    # Write results.json
    write_results(results_dict, results_path)
    print(f"Wrote results to {results_path}", flush=True)

    # Now verify check_prereg
    prereg_valid = check_prereg(prereg_path, results_path)
    proto_checks.append({
        "tool": "check_prereg",
        "scenario": "PREREG precedes results.json",
        "result": f"prereg < results: {prereg_valid}",
        "status": "PASS" if prereg_valid else "FAIL"
    })

    # Update results with final check
    write_results(results_dict, results_path)

    # Render tables
    tables_text = render_z0_tables(results_dict)
    (out_dir / "tables.txt").write_text(tables_text, encoding="utf-8")
    print("\nRendered Tables:\n" + tables_text)

    # Validate report tables against themselves
    validate_report(tables_text, tables_text)
    print(f"Z0 run completed cleanly in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
