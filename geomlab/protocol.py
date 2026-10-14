"""
Protocol library for Track Z: Automated experimental protocol, integrity checks, and reporting.
Implements the protocol requirements from GEOMETRY_ATLAS_HANDOFF.md Section 4:
- check_prereg: verifies PREREG file exists and precedes results file timestamps
- tune_lr: automated per-arm hyperparameter tuning on dev seeds >= 100 with grid-boundary checks
- run_seeds: deterministic multi-seed evaluation and aggregation
- paired_stats: paired t-test, Wilcoxon signed-rank test, Cohen's d_z, and 95% CIs
- bonferroni: family-wise error rate control
- write_results: JSON serialization of raw experiment data
- render_tables / render_poc_b_table: ASCII table rendering from stored JSON
- validate_report: strict verbatim validation of rendered tables against delivered reports
"""

import difflib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import stats


class ValidationError(Exception):
    """Raised when report text does not match rendered results table verbatim."""
    pass


class PreregError(Exception):
    """Raised when pre-registration integrity check fails."""
    pass


def ci95(arr: Sequence[float]) -> float:
    """Computes two-sided 95% Student's t confidence interval half-width."""
    a = np.asarray(arr, dtype=float)
    if len(a) < 2:
        return 0.0
    s = float(a.std(ddof=1))
    if s == 0.0 or math.isnan(s):
        return 0.0
    return float(stats.t.ppf(0.975, len(a) - 1) * s / math.sqrt(len(a)))


def check_prereg(
    prereg_path: Union[str, Path],
    results_path: Union[str, Path],
) -> bool:
    """
    Checks that the pre-registration document exists and its modification timestamp
    strictly precedes the results file.
    """
    p_prereg = Path(prereg_path)
    p_results = Path(results_path)

    if not p_prereg.exists():
        raise FileNotFoundError(f"PREREG file does not exist: {p_prereg}")
    if not p_results.exists():
        raise FileNotFoundError(f"Results file does not exist: {p_results}")

    mtime_prereg = p_prereg.stat().st_mtime
    mtime_results = p_results.stat().st_mtime

    if mtime_prereg >= mtime_results:
        raise PreregError(
            f"PREREG file is newer than results ({mtime_prereg:.3f} >= {mtime_results:.3f})! "
            f"Pre-registration must precede experimental results."
        )

    return True


def tune_lr(
    eval_fn: Callable[[float, int], float],
    lr_grid: Sequence[float],
    dev_seeds: Sequence[int] = (100, 101),
    metric_name: str = "score",
    higher_is_better: bool = True,
    auto_expand: bool = True,
    max_expansions: int = 3,
    expansion_factor: float = 3.0,
) -> Dict[str, Any]:
    """
    Tuning of learning rate on dev seeds >= 100 with grid boundary / edge check.
    If the best value is at the grid edge, extends the grid and reruns until an interior
    optimum is reached or max expansions are exhausted.
    """
    for s in dev_seeds:
        if s < 100:
            raise ValueError(f"Dev seeds must be >= 100 per protocol rules, got seed {s}")

    current_grid = sorted(list(lr_grid))
    scores: Dict[float, float] = {}
    expansion_count = 0

    while True:
        for lr in current_grid:
            if lr not in scores:
                seed_vals = [eval_fn(lr, s) for s in dev_seeds]
                scores[lr] = float(np.mean(seed_vals))

        if higher_is_better:
            best_lr = max(current_grid, key=lambda lr: scores[lr])
        else:
            best_lr = min(current_grid, key=lambda lr: scores[lr])

        is_min_edge = (best_lr == current_grid[0])
        is_max_edge = (best_lr == current_grid[-1])
        is_edge = is_min_edge or is_max_edge

        if is_edge and auto_expand and (expansion_count < max_expansions):
            expansion_count += 1
            if is_min_edge:
                new_lr = best_lr / expansion_factor
                current_grid.insert(0, new_lr)
            else:
                new_lr = best_lr * expansion_factor
                current_grid.append(new_lr)
            continue
        break

    return {
        "best_lr": best_lr,
        "grid": current_grid,
        "scores": {f"{lr:g}": scores[lr] for lr in current_grid},
        "is_edge": is_edge,
        "expansions": expansion_count,
        "metric_name": metric_name,
        "best_score": scores[best_lr],
    }


def run_seeds(
    runner_fn: Callable[[int], Dict[str, float]],
    test_seeds: Sequence[int] = tuple(range(10)),
) -> Dict[str, Any]:
    """
    Runs evaluation across test seeds and computes summary statistics:
    mean, std, 95% Student-t CI, min, max, and raw per-seed values.
    """
    per_seed_results: List[Dict[str, float]] = []
    for seed in test_seeds:
        res = runner_fn(seed)
        per_seed_results.append(res)

    metric_keys = list(per_seed_results[0].keys())
    summary: Dict[str, Any] = {
        "seeds": list(test_seeds),
        "metrics": {},
    }

    for key in metric_keys:
        vals = [r[key] for r in per_seed_results]
        arr = np.asarray(vals, dtype=float)
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        ci_val = ci95(arr)
        summary["metrics"][key] = {
            "mean": mean_val,
            "std": std_val,
            "ci95": ci_val,
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "values": vals,
        }

    return summary


def paired_stats(
    a: Sequence[float],
    b: Sequence[float],
) -> Dict[str, float]:
    """
    Computes paired difference statistics between arm A and baseline B:
    - mean difference
    - std of differences
    - 95% CI of difference
    - paired Student's t-test (t-statistic, p-value)
    - Wilcoxon signed-rank test (statistic, p-value)
    - Cohen's d_z effect size (mean_diff / std_diff)
    """
    arr_a = np.asarray(a, dtype=float)
    arr_b = np.asarray(b, dtype=float)
    if len(arr_a) != len(arr_b):
        raise ValueError(f"Arrays must have equal length: {len(arr_a)} vs {len(arr_b)}")
    if len(arr_a) < 2:
        raise ValueError("Must have at least 2 pairs for paired statistics")

    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    ci_diff = ci95(diff)

    # Paired t-test
    if std_diff == 0.0 or math.isnan(std_diff):
        t_stat = 0.0
        p_val_t = 1.0
        cohen_dz = 0.0
    else:
        t_res = stats.ttest_rel(arr_a, arr_b)
        t_stat = float(t_res.statistic)
        p_val_t = float(t_res.pvalue)
        cohen_dz = float(mean_diff / std_diff)

    # Wilcoxon signed-rank test
    if np.all(diff == 0):
        w_stat = 0.0
        p_val_w = 1.0
    else:
        try:
            w_res = stats.wilcoxon(arr_a, arr_b, zero_method="wilcox")
            w_stat = float(w_res.statistic)
            p_val_w = float(w_res.pvalue)
        except Exception:
            w_stat = 0.0
            p_val_w = 1.0

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95_diff": ci_diff,
        "t_statistic": t_stat,
        "p_value_ttest": p_val_t,
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": p_val_w,
        "cohen_dz": cohen_dz,
    }


def bonferroni(
    p_values: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Applies Bonferroni correction across a family of p-values:
      p_adjusted = min(1.0, p * m)
    """
    m = len(p_values)
    adj_p = [float(min(1.0, p * m)) for p in p_values]
    sig = [p_adj < alpha for p_adj in adj_p]
    return {
        "alpha": alpha,
        "n_comparisons": m,
        "raw_p_values": list(p_values),
        "adjusted_p_values": adj_p,
        "significant": sig,
    }


def write_results(
    results_data: Dict[str, Any],
    out_path: Union[str, Path],
) -> None:
    """Writes results dictionary to JSON with indentation and parent directory creation."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(results_data, indent=2, sort_keys=True), encoding="utf-8")


def render_poc_b_table(
    out_dir: Union[str, Path] = "research/yggdrasil_results/poc_b",
    newline: str = "\n",
) -> str:
    """
    Reproduces research/yggdrasil_results/poc_b_table.txt from stored JSON files.
    Allows specifying newline='\r\n' for exact Windows CRLF byte matching or '\n' for LF.
    """
    base_dir = Path(out_dir)
    models = ("torus", "so3", "gru", "torus_lat", "so8", "hh2", "vq_rot", "grp_lookup")
    lengths = (16, 32, 64, 128, 256)

    def load_json(p: Path) -> dict:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    lines = [
        "accuracy (%) of the product after the LAST symbol; mean +- 95% CI over 10 seeds; trained on lengths <= 16 (<=24 for S5)",
        f"{'task':4s} {'model':10s} {'lr':>6s} | " + " ".join(f"{'T=' + str(T):>12s}" for T in lengths) + " | seeds >=95% at T=256"
    ]

    for group in ("z6", "s3", "s5"):
        for m in models:
            res_file = base_dir / f"res_{m}.json"
            r = load_json(res_file)
            if group not in r:
                continue
            cells = " ".join(f"{100 * np.mean(r[group][str(T)]):6.1f}+-{100 * ci95(r[group][str(T)]):4.1f}" for T in lengths)
            solved = sum(v >= 0.95 for v in r[group]["256"])
            lines.append(f"{group:4s} {m:10s} {r['lr']:6g} | {cells} | {solved}/10")

    return newline.join(lines) + newline


def render_tables(
    results_dict_or_path: Union[Dict[str, Any], str, Path],
    title: str = "Results Summary",
) -> str:
    """Renders formatted ASCII table from results dictionary or results.json."""
    if isinstance(results_dict_or_path, (str, Path)):
        data = json.loads(Path(results_dict_or_path).read_text(encoding="utf-8"))
    else:
        data = results_dict_or_path

    lines = [f"=== {title} ==="]
    if "arms" in data:
        lines.append(f"{'Arm':<20s} {'Metric':<20s} {'Mean +- 95% CI':<24s}")
        lines.append("-" * 66)
        for arm_name, arm_data in data["arms"].items():
            if "metrics" in arm_data:
                for metric_name, m_stats in arm_data["metrics"].items():
                    val_str = f"{m_stats['mean']:.3f} +- {m_stats['ci95']:.3f}"
                    lines.append(f"{arm_name:<20s} {metric_name:<20s} {val_str:<24s}")
    elif "metrics" in data:
        lines.append(f"{'Metric':<24s} {'Mean +- 95% CI':<24s}")
        lines.append("-" * 50)
        for metric_name, m_stats in data["metrics"].items():
            val_str = f"{m_stats['mean']:.3f} +- {m_stats['ci95']:.3f}"
            lines.append(f"{metric_name:<24s} {val_str:<24s}")
    else:
        # Generic key-value dump
        for k, v in data.items():
            lines.append(f"{k}: {v}")

    return "\n".join(lines) + "\n"


def validate_report(
    report_text_or_path: Union[str, Path],
    rendered_table_or_path: Union[str, Path],
) -> None:
    """
    Validates that the rendered table appears verbatim inside the report text.
    Fails on any single character difference with a detailed unified diff.
    """
    if isinstance(report_text_or_path, Path) or (isinstance(report_text_or_path, str) and os.path.exists(report_text_or_path)):
        report_text = Path(report_text_or_path).read_text(encoding="utf-8")
    else:
        report_text = str(report_text_or_path)

    if isinstance(rendered_table_or_path, Path) or (isinstance(rendered_table_or_path, str) and os.path.exists(rendered_table_or_path)):
        rendered_table = Path(rendered_table_or_path).read_text(encoding="utf-8")
    else:
        rendered_table = str(rendered_table_or_path)

    # Normalize line endings
    norm_report = report_text.replace("\r\n", "\n")
    norm_table = rendered_table.replace("\r\n", "\n").strip()

    if norm_table not in norm_report:
        # Generate diff for debugging / reporting
        diff_lines = list(difflib.unified_diff(
            norm_table.splitlines(keepends=True),
            norm_report.splitlines(keepends=True),
            fromfile="rendered_table",
            tofile="report_text",
            n=3,
        ))
        diff_snippet = "".join(diff_lines[:30])
        raise ValidationError(
            f"Table diff mismatch: Rendered table does not match report text verbatim!\n{diff_snippet}"
        )
