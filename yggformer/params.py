"""
Parameter matching utility: adjust ffn_mult to equalize parameter counts between architectures.
"""

from copy import deepcopy
from .config import YggCfg
from .model import YggFormer


def count_parameters(model: YggFormer) -> int:
    return sum(p.numel() for p in model.parameters())


def match_params(cfg_reference: YggCfg, cfg_target: YggCfg) -> YggCfg:
    """
    Adjusts ffn_mult in cfg_target so that its parameter count matches cfg_reference within 1%.
    """
    ref_model = YggFormer(cfg_reference)
    n_ref = count_parameters(ref_model)

    new_target_cfg = deepcopy(cfg_target)
    # Temporary model to find base parameters without FFN
    new_target_cfg.ffn_mult = 0.0
    zero_ffn_model = YggFormer(new_target_cfg)
    p_base = count_parameters(zero_ffn_model)

    # Each FFN layer has 2 * (d_model * ffn_dim) = 2 * d_model^2 * ffn_mult
    d_model = cfg_target.d_model
    n_layers = cfg_target.n_layers
    params_per_ffn_mult_unit = 2 * n_layers * (d_model ** 2)

    needed_ffn_params = n_ref - p_base
    if needed_ffn_params <= 0:
        optimal_ffn_mult = 1.0
    else:
        optimal_ffn_mult = needed_ffn_params / params_per_ffn_mult_unit

    new_target_cfg.ffn_mult = optimal_ffn_mult

    # Verify
    test_model = YggFormer(new_target_cfg)
    n_test = count_parameters(test_model)
    rel_err = abs(n_ref - n_test) / n_ref
    if rel_err > 0.01:
        # Fine-tune slightly if needed due to integer rounding
        new_target_cfg.ffn_mult = optimal_ffn_mult * (n_ref / n_test)

    return new_target_cfg
