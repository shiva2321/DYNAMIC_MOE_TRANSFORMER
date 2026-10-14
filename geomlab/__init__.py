"""
geomlab: Geometry Core experimental and diagnostic component library.
Track Z verification toolkit for the general-intelligence system.
"""

from .diagnostics import (
    effective_rank,
    isotropy,
    intrinsic_dim_twonn,
    hubness_skew,
    norm_stats,
    operating_point_adapter,
)

from .protocol import (
    tune_lr,
    run_seeds,
    paired_stats,
    bonferroni,
    write_results,
    render_tables,
    render_poc_b_table,
    validate_report,
    check_prereg,
    ValidationError,
    PreregError,
    ci95,
)

from .spherical_stream import (
    SphericalStreamLM,
    BaselineLM,
    SphericalStreamWithSalienceLM,
    normalize_weights_,
    compute_speedup,
    l2_norm,
)

__all__ = [
    "effective_rank",
    "isotropy",
    "intrinsic_dim_twonn",
    "hubness_skew",
    "norm_stats",
    "operating_point_adapter",
    "tune_lr",
    "run_seeds",
    "paired_stats",
    "bonferroni",
    "write_results",
    "render_tables",
    "render_poc_b_table",
    "validate_report",
    "check_prereg",
    "ValidationError",
    "PreregError",
    "ci95",
    "SphericalStreamLM",
    "BaselineLM",
    "SphericalStreamWithSalienceLM",
    "normalize_weights_",
    "compute_speedup",
    "l2_norm",
]
