"""Tests for costlab.count: MACs counting, Counter, and helpers."""

import pytest
import torch
import torch.nn as nn

from costlab.count import Counter, MacCount, count_macs, macs_matmul


def test_macs_matmul_analytic():
    """Verify macs_matmul helper calculation."""
    assert macs_matmul(10, 20, 30) == 6000
    assert macs_matmul(1, 1, 1) == 1
    assert macs_matmul(0, 5, 5) == 0


def test_counter_manual_accounting():
    """Verify Counter records and aggregates non-torch MACs correctly."""
    counter = Counter()
    assert counter.total() == 0
    assert counter.by_name() == {}

    counter.add("gemm_1", 1000)
    counter.add("gemm_2", 2500)
    counter.add("gemm_1", 500)

    assert counter.total() == 4000
    assert counter.by_name() == {"gemm_1": 1500, "gemm_2": 2500}


def test_count_macs_linear():
    """Verify count_macs for single Linear layer matches analytic count."""
    in_features, out_features = 64, 128
    batch_size = 16
    lin = nn.Linear(in_features, out_features, bias=False)
    x = torch.randn(batch_size, in_features)

    res = count_macs(lin, x)
    expected_macs = batch_size * in_features * out_features
    assert isinstance(res, MacCount)
    assert res.macs == expected_macs
    assert res.flops == 2 * expected_macs
    assert len(res.by_op) > 0


def test_count_macs_conv2d():
    """Verify count_macs on 2D convolution."""
    batch, in_ch, out_ch = 2, 3, 16
    h, w = 32, 32
    kernel_size = 3
    conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=1, bias=False)
    x = torch.randn(batch, in_ch, h, w)

    res = count_macs(conv, x)
    # Output is (2, 16, 32, 32), each output element computes 3 * 3 * 3 = 27 MACs
    expected_macs = batch * out_ch * h * w * (in_ch * kernel_size * kernel_size)
    assert res.macs == expected_macs
    assert res.flops == 2 * expected_macs


def test_count_macs_elementwise_not_counted():
    """Verify element-wise operations (ReLU, add) do not add to MACs count."""
    x = torch.randn(10, 10)
    y = torch.randn(10, 10)

    def elementwise_fn():
        return torch.relu(x + y)

    res = count_macs(elementwise_fn)
    assert res.macs == 0
    assert res.flops == 0
