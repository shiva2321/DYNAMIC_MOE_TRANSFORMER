"""Tests for costlab.memory: peak_memory tracking on CPU and GPU."""

import time
import numpy as np
import pytest
import torch

from costlab.memory import peak_memory


def test_peak_memory_cpu():
    """Verify peak_memory runs on CPU and returns all required keys."""
    def cpu_alloc():
        arr = np.ones((1000, 1000), dtype=np.float32)
        time.sleep(0.02)
        del arr

    res = peak_memory(cpu_alloc, device="cpu")
    assert "rss_start_bytes" in res
    assert "rss_peak_bytes" in res
    assert "rss_diff_bytes" in res
    assert res["rss_peak_bytes"] >= res["rss_start_bytes"]
    assert res["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_peak_memory_cuda():
    """Verify peak_memory tracks CUDA allocation peak."""
    def cuda_alloc():
        # Allocate 10 MB tensor
        t = torch.empty((2500000,), dtype=torch.float32, device="cuda")
        time.sleep(0.02)
        del t

    res = peak_memory(cuda_alloc, device="cuda")
    assert "cuda_peak_bytes" in res
    assert res["cuda_peak_bytes"] is not None
    assert res["cuda_peak_bytes"] >= 10 * 1024 * 1024
