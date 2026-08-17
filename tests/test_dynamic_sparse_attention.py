"""
Unit and Regression Tests for Dynamic & Sparse Attention Mechanism.
Verifies:
1. Forward pass on short sequences (dense fast-path).
2. Forward pass on extended sequences (dynamic sparse path with foveal window, sinks, and phasor landmarks).
3. RoPE rotary positional embedding extrapolation.
4. Sparsity ratio scaling.
5. Full backward gradient flow.
"""

import sys
import os
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.dynamic_sparse_attention import DynamicSparseAttention, RotaryEmbedding

def test_dynamic_sparse_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Testing Dynamic & Sparse Attention on {device}]")

    d_model = 256
    n_heads = 4
    foveal_window = 64
    num_sinks = 4
    num_landmarks = 2
    chunk_size = 32

    attn = DynamicSparseAttention(
        d_model=d_model,
        n_heads=n_heads,
        foveal_window=foveal_window,
        num_sinks=num_sinks,
        chunk_size=chunk_size,
        num_landmarks=num_landmarks,
        dynamic_span=True,
    ).to(device)

    # 1. Test Short Sequence (S = 32 <= foveal_window)
    x_short = torch.randn(2, 32, d_model, device=device, requires_grad=True)
    out_short, telem_short = attn(x_short)
    assert out_short.shape == x_short.shape
    assert not torch.isnan(out_short).any()
    print(f"1. Short Sequence (S=32): Shape {out_short.shape} | Sparsity: {telem_short['sparsity_ratio']*100:.1f}%")

    # 2. Test Extended Sequence (S = 256 > foveal_window)
    x_long = torch.randn(2, 256, d_model, device=device, requires_grad=True)
    out_long, telem_long = attn(x_long)
    assert out_long.shape == x_long.shape
    assert not torch.isnan(out_long).any()
    print(f"2. Extended Sequence (S=256): Shape {out_long.shape} | Active Window: {telem_long['active_window']} | Sparsity: {telem_long['sparsity_ratio']*100:.1f}%")

    # Verify Sparsity Savings
    assert telem_long['sparsity_ratio'] > 0.50, f"Expected >50% sparsity on long sequence, got {telem_long['sparsity_ratio']}"

    # 3. Test Backward Gradient Flow
    loss = out_long.sum()
    loss.backward()
    assert x_long.grad is not None
    assert not torch.isnan(x_long.grad).any()
    print("3. Backward Gradient Pass: Passed with Zero NaNs!")

    # 4. Test RoPE Sequence Extrapolation (S = 2048)
    rope = RotaryEmbedding(dim=64, max_seq_len=512).to(device)
    cos, sin = rope(x_short, seq_len=2048)
    assert cos.shape[2] == 2048
    print("4. RoPE Extrapolation to S=2048: Passed Dynamic Re-caching!")

    print("[ALL DYNAMIC SPARSE ATTENTION TESTS PASSED SUCCESSFULLY!]\n")

if __name__ == "__main__":
    test_dynamic_sparse_attention()
