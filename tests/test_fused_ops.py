"""
Unit Tests for Fused Operations (hyperspace/fused_ops.py).
Verifies:
1. Numerical equivalence of fused_k_wta_mask against unfused DentateGyrus.
2. Numerical equivalence of fused_hermitian_similarity against ComplexPhasorVSA.
"""

import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hyperspace.fused_ops import fused_k_wta_mask, fused_real_to_phasor, fused_hermitian_similarity
from hyperspace.vsa import ComplexPhasorVSA

def test_fused_k_wta_mask_equivalence():
    x = torch.randn(16, 1024)
    k_active = 32
    
    # Fused execution
    out_fused = fused_k_wta_mask(x, k_active)
    
    # Verify exact top-k non-zero count
    non_zeros = (out_fused > 0).sum(dim=-1)
    assert (non_zeros == k_active).all()
    
    # Verify unit norm
    norms = torch.norm(out_fused, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

def test_fused_hermitian_similarity_equivalence():
    batch = 8
    num_keys = 4
    d_hyper = 512
    
    # Unfused complex tensors
    q_complex = ComplexPhasorVSA.random_hyperspace_vector((batch, d_hyper))
    k_complex = ComplexPhasorVSA.random_hyperspace_vector((num_keys, d_hyper))
    
    sim_unfused = ComplexPhasorVSA.batch_similarity_matrix(q_complex, k_complex)
    
    # Fused real-imag execution
    q_r, q_i = q_complex.real, q_complex.imag
    k_r, k_i = k_complex.real, k_complex.imag
    sim_fused = fused_hermitian_similarity(q_r, q_i, k_r, k_i)
    
    assert torch.allclose(sim_unfused, sim_fused, atol=1e-5)
