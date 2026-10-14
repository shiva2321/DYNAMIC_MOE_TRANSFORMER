"""
Tests for Track Z, Round Z0: Protocol library and geometry-health diagnostics.

Requirements:
- effective rank ~ d for isotropic Gaussians, small for low-rank data
- TwoNN near 3 for a 3-D manifold embedded in 20-D
- the validator fails when one digit of a table is changed
- check_prereg fails without a PREREG file or with a PREREG newer than the results
- a dry run reproduces research/yggdrasil_results/poc_b_table.txt from stored JSON byte for byte
"""

import math
import os
import time
from pathlib import Path
import numpy as np
import pytest
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
    ValidationError,
    PreregError,
)


def test_effective_rank_properties():
    # 1. Isotropic Gaussian in d=32: effective rank should be close to d (~30-32)
    torch.manual_seed(42)
    N, D = 4000, 32
    X_iso = torch.randn(N, D)
    erank_iso = effective_rank(X_iso)
    assert erank_iso >= 30.0, f"Expected erank >= 30.0 for isotropic Gaussian, got {erank_iso}"
    assert erank_iso <= 32.0, f"Expected erank <= 32.0, got {erank_iso}"

    # 2. Rank-1 data: all points along a single 1D direction
    v = torch.randn(1, D)
    v = v / v.norm()
    scalars = torch.randn(N, 1)
    X_rank1 = scalars @ v
    erank_rank1 = effective_rank(X_rank1)
    assert erank_rank1 == pytest.approx(1.0, abs=0.05), f"Expected erank ~ 1.0 for rank-1, got {erank_rank1}"

    # 3. Low-rank rank-2 plane
    u1 = torch.randn(1, D)
    u2 = torch.randn(1, D)
    X_rank2 = torch.randn(N, 1) @ u1 + torch.randn(N, 1) @ u2
    erank_rank2 = effective_rank(X_rank2)
    assert 1.8 <= erank_rank2 <= 2.1, f"Expected erank ~ 2.0 for rank-2, got {erank_rank2}"


def test_intrinsic_dim_twonn():
    torch.manual_seed(123)
    # A 3-D manifold (uniform points inside 3-D ball/cube) embedded linearly/isometrically in 20-D
    N = 3000
    # Sample uniformly in 3D
    u = torch.randn(N, 3)
    u = u / u.norm(dim=-1, keepdim=True) * (torch.rand(N, 1) ** (1.0 / 3.0))

    # Random orthonormal basis in 20D (isometric embedding preserves all Euclidean distances)
    Q, _ = torch.linalg.qr(torch.randn(20, 3))  # (20, 3)
    X_20d = u @ Q.T  # (N, 20)

    id_est = intrinsic_dim_twonn(X_20d)
    # Expect estimated intrinsic dimension near 3 (tolerance +/- 0.3)
    assert 2.7 <= id_est <= 3.3, f"Expected TwoNN intrinsic dim near 3.0 for 3D manifold in 20D, got {id_est}"

    # Also test 2-D manifold (uniform points in 2-D ball) in 20-D
    u_2d = torch.randn(N, 2)
    u_2d = u_2d / u_2d.norm(dim=-1, keepdim=True) * (torch.rand(N, 1) ** (1.0 / 2.0))
    Q2, _ = torch.linalg.qr(torch.randn(20, 2))
    X_2d_in_20d = u_2d @ Q2.T
    id_2d = intrinsic_dim_twonn(X_2d_in_20d)
    assert 1.8 <= id_2d <= 2.2, f"Expected TwoNN intrinsic dim near 2.0 for 2D manifold in 20D, got {id_2d}"

    # Pre-registered 1-D manifold test: random points on circle embedded in 20-D
    theta = torch.rand(N, 1) * 2 * math.pi
    circle_2d = torch.cat([theta.cos(), theta.sin()], dim=1)
    Q1_circ, _ = torch.linalg.qr(torch.randn(20, 2))
    X_circle_in_20d = circle_2d @ Q1_circ.T
    id_circle = intrinsic_dim_twonn(X_circle_in_20d)
    assert 0.8 <= id_circle <= 1.2, f"Expected TwoNN intrinsic dim in [0.8, 1.2] for 1D circle in 20D, got {id_circle}"

    # Pre-registered 1-D manifold test: random points on line segment embedded in 20-D
    seg_1d = torch.rand(N, 1)
    Q1_seg, _ = torch.linalg.qr(torch.randn(20, 1))
    X_seg_in_20d = seg_1d @ Q1_seg.T
    id_seg = intrinsic_dim_twonn(X_seg_in_20d)
    assert 0.8 <= id_seg <= 1.2, f"Expected TwoNN intrinsic dim in [0.8, 1.2] for 1D segment in 20D, got {id_seg}"


def test_twonn_close_points_cancellation_regression():
    """
    Regression test for J1: ensures torch.cdist float64 exact mode does not suffer
    catastrophic cancellation on closely spaced points (dx ~ 1e-4) along a 1D curve.
    """
    torch.manual_seed(42)
    N = 3000
    # Closely spaced points along a 1D line in 20D
    x_1d = torch.rand(N, 1) * 1e-2  # compact range where neighbours have dx ~ 1e-5 to 1e-6
    Q, _ = torch.linalg.qr(torch.randn(20, 1))
    X_compact = x_1d @ Q.T
    id_est = intrinsic_dim_twonn(X_compact)
    assert 0.8 <= id_est <= 1.2, f"Expected robust TwoNN ~ 1.0 on closely spaced points, got {id_est}"


def test_isotropy_and_uniformity():
    torch.manual_seed(99)
    N, D = 2000, 16

    # Uniform isotropic points on sphere
    X_iso = torch.randn(N, D)
    X_iso = X_iso / X_iso.norm(dim=-1, keepdim=True)
    stats_iso = isotropy(X_iso)

    assert "mean_pairwise_cosine" in stats_iso
    assert "uniformity_loss" in stats_iso
    # Mean pairwise cosine on isotropic sphere is ~ 0
    assert abs(stats_iso["mean_pairwise_cosine"]) < 0.05

    # Clustered / collapsed points
    cluster_center = torch.randn(1, D)
    cluster_center = cluster_center / cluster_center.norm()
    X_clustered = cluster_center + 0.05 * torch.randn(N, D)
    X_clustered = X_clustered / X_clustered.norm(dim=-1, keepdim=True)
    stats_clustered = isotropy(X_clustered)

    assert stats_clustered["mean_pairwise_cosine"] > 0.90
    # Uniformity loss should be significantly higher (less uniform) for clustered points
    assert stats_clustered["uniformity_loss"] > stats_iso["uniformity_loss"]


def test_hubness_skew():
    torch.manual_seed(101)
    N, D = 1000, 16
    # Uniform points in R^16
    X_unif = torch.randn(N, D)
    skew_unif = hubness_skew(X_unif, k=5)
    # Skewness is modest for uniform points
    assert isinstance(skew_unif, float)

    # Clustered points with strong hubs
    hub = torch.randn(1, D) * 5.0
    cluster = torch.randn(N - 1, D)
    X_hub = torch.cat([hub, cluster], dim=0)
    skew_hub = hubness_skew(X_hub, k=5)
    assert isinstance(skew_hub, float)


def test_norm_stats():
    # Known norms: vectors with norms 1, 2, 3, 4, 5
    X = torch.eye(5) * torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])[:, None]
    stats = norm_stats(X)
    assert stats["mean_norm"] == pytest.approx(3.0)
    assert stats["min_norm"] == pytest.approx(1.0)
    assert stats["max_norm"] == pytest.approx(5.0)
    assert stats["median_norm"] == pytest.approx(3.0)


def test_validator_fails_when_one_digit_changed(tmp_path):
    report_text = (
        "Heading\n"
        "--- Table 1 ---\n"
        "task model lr | T=16 T=32\n"
        "z6 torus 0.1 | 100.0+- 0.0 79.7+-23.1\n"
        "--- End Table ---\n"
        "Conclusion."
    )
    rendered_table = (
        "--- Table 1 ---\n"
        "task model lr | T=16 T=32\n"
        "z6 torus 0.1 | 100.0+- 0.0 79.7+-23.1\n"
        "--- End Table ---"
    )

    # Valid: rendered table exists verbatim in report_text
    validate_report(report_text, rendered_table)

    # Corrupt: change one single digit: '79.7' -> '79.8'
    tampered_report = report_text.replace("79.7", "79.8")
    with pytest.raises(ValidationError) as exc_info:
        validate_report(tampered_report, rendered_table)
    assert "Table diff mismatch" in str(exc_info.value) or "mismatch" in str(exc_info.value).lower()

    # Corrupt: change one character: 'torus' -> 'torux'
    tampered_report2 = report_text.replace("torus", "torux")
    with pytest.raises(ValidationError):
        validate_report(tampered_report2, rendered_table)


def test_check_prereg_integrity(tmp_path):
    prereg = tmp_path / "PREREG.md"
    results = tmp_path / "results.json"

    # Case 1: Missing PREREG
    with pytest.raises((FileNotFoundError, PreregError)):
        check_prereg(prereg, results)

    # Create PREREG
    prereg.write_text("pre-registered predictions", encoding="utf-8")
    time.sleep(0.05)

    # Create results after PREREG
    results.write_text('{"accuracy": 1.0}', encoding="utf-8")
    assert check_prereg(prereg, results) is True

    # Case 2: PREREG newer than results (tampering)
    time.sleep(0.05)
    prereg.write_text("updated predictions after seeing results", encoding="utf-8")
    with pytest.raises(PreregError) as exc_info:
        check_prereg(prereg, results)
    assert "PREREG file is newer than results" in str(exc_info.value)


def test_dry_run_reproduces_poc_b_table_byte_for_byte():
    poc_b_dir = Path("research/yggdrasil_results/poc_b")
    target_table_file = Path("research/yggdrasil_results/poc_b_table.txt")

    assert poc_b_dir.exists(), f"{poc_b_dir} does not exist"
    assert target_table_file.exists(), f"{target_table_file} does not exist"

    target_content = target_table_file.read_text(encoding="utf-8")
    rendered = render_poc_b_table(poc_b_dir)

    # 1. Normalized character comparison (CRLF to LF)
    target_norm = target_content.replace("\r\n", "\n").strip()
    rendered_norm = rendered.replace("\r\n", "\n").strip()
    assert rendered_norm == target_norm, "Rendered table does not match poc_b_table.txt after newline normalization"

    # 2. Exact byte-for-byte reproduction when rendered with Windows CRLF
    rendered_crlf = render_poc_b_table(poc_b_dir, newline="\r\n")
    target_bytes = target_table_file.read_bytes()
    assert rendered_crlf.encode("utf-8") == target_bytes, "Rendered CRLF table does not match poc_b_table.txt byte-for-byte"


def test_tune_lr_interior_and_edge_detection():
    # Synthetic objective: quadratic loss (lr - 0.01)^2
    def eval_fn(lr, seed):
        return -((math.log10(lr) - math.log10(0.01)) ** 2)

    grid = [1e-3, 3e-3, 1e-2, 3e-2]
    res = tune_lr(eval_fn, grid, dev_seeds=[100, 101], metric_name="score", higher_is_better=True)
    assert res["best_lr"] == pytest.approx(0.01)
    assert res["is_edge"] is False

    # Edge objective: monotonic increasing with lr (best on max edge)
    def eval_edge(lr, seed):
        return lr

    res_edge = tune_lr(eval_edge, grid, dev_seeds=[100, 101], metric_name="score", higher_is_better=True, auto_expand=False)
    assert res_edge["best_lr"] == pytest.approx(3e-2)
    assert res_edge["is_edge"] is True


def test_paired_stats_and_bonferroni():
    a = np.array([10.0, 11.0, 12.0, 10.5, 11.5, 10.8, 12.2, 11.1, 10.9, 11.4])
    b = np.array([8.0, 8.5, 9.0, 8.2, 8.8, 8.1, 9.3, 8.6, 8.4, 8.7])

    stats = paired_stats(a, b)
    assert stats["mean_diff"] > 2.0
    assert stats["p_value_ttest"] < 0.001
    assert stats["cohen_dz"] > 2.0

    # Bonferroni adjustment
    p_vals = [0.01, 0.04, 0.20]
    adj = bonferroni(p_vals, alpha=0.05)
    assert adj["adjusted_p_values"][0] == pytest.approx(0.03)
    assert adj["adjusted_p_values"][1] == pytest.approx(0.12)
    assert adj["significant"][0] is True
    assert adj["significant"][1] is False


def test_operating_point_adapter():
    # Test on standard nn.Linear module
    linear_mod = torch.nn.Linear(32, 64)
    rep_linear = operating_point_adapter(linear_mod)
    assert rep_linear["model_type"] == "Linear"
    assert len(rep_linear["layers"]) >= 1
    assert "effective_rank" in rep_linear["layers"][0]

    # Test on YggFormer model
    from yggformer.config import YggCfg
    from yggformer.model import YggFormer
    cfg = YggCfg(vocab=32, d_model=64, n_layers=1, n_heads=2, head_dim=32, max_len=16, head_geometry="euclid", seed=100)
    yf = YggFormer(cfg).eval()
    batch = torch.randint(0, 32, (2, 8))
    rep_yf = operating_point_adapter(yf, batch=batch)
    assert "layers" in rep_yf
    assert "frac_heads_euclidean" in rep_yf


def test_run_seeds_and_write_results(tmp_path):
    def dummy_eval(seed):
        return {"acc": 0.90 + 0.01 * (seed % 3), "loss": 0.20 - 0.005 * (seed % 3)}

    res = run_seeds(dummy_eval, test_seeds=range(10))
    assert len(res["seeds"]) == 10
    assert "acc" in res["metrics"]
    assert "loss" in res["metrics"]
    assert 0.89 <= res["metrics"]["acc"]["mean"] <= 0.93

    out_file = tmp_path / "results.json"
    write_results(res, out_file)
    assert out_file.exists()
    import json
    loaded = json.loads(out_file.read_text(encoding="utf-8"))
    assert loaded["metrics"]["acc"]["mean"] == pytest.approx(res["metrics"]["acc"]["mean"])

