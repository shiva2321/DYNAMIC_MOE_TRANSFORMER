"""
Configuration for YggFormer architecture.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class YggCfg:
    vocab: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    head_dim: int = 32
    ffn_mult: float = 4.0
    max_len: int = 256
    head_geometry: str = "hyp_phase_bias"  # "euclid" | "hyp" | "hyp_phase" | "hyp_phase_bias"
    n_phase: Optional[int] = None           # Default: head_dim // 4
    c_init: float = 1.0
    learn_c: bool = True
    tau_init: float = 1.0
    qk_dtype: str = "float32"
    r_max: Optional[float] = None          # None -> safe_radius(qk_dtype)
    aggregate: str = "lorentz"             # "lorentz" | "tangent" | "euclid"
    null_key: bool = False
    seed: int = 0

    def __post_init__(self):
        valid_geoms = {"euclid", "euclid_gate", "euclid_temp", "hyp", "hyp_phase", "hyp_phase_bias"}
        if self.head_geometry not in valid_geoms:
            raise ValueError(f"Invalid head_geometry: {self.head_geometry}. Must be one of {valid_geoms}")

        if self.n_phase is None:
            self.n_phase = max(1, self.head_dim // 4)

        if self.r_max is None:
            from geometry.attention import safe_radius
            import torch
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dt = dtype_map.get(self.qk_dtype, torch.float32)
            self.r_max = safe_radius(dt)
