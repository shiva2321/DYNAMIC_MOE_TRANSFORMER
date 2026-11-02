"""Tests added in the review of the first build: behaviour it got wrong or left silent."""

import pytest
import torch
import torch.nn as nn

from costlab.bytes import active_param_bytes
from costlab.env import fixed_threads
from costlab.memory import peak_memory
from costlab.timing import time_callable


class _OwnsParameterAndChild(nn.Module):
    """A module that owns a parameter itself and also has a child that owns another."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(8))
        self.child = nn.Linear(8, 8, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.child(x) * self.scale


def test_active_bytes_counts_parameters_owned_by_a_parent_module():
    model = _OwnsParameterAndChild()
    x = torch.randn(2, 8)
    res = active_param_bytes(model, model, x)
    assert res.by_module["(root)"] == 8 * 4
    assert res.by_module["child"] == 8 * 8 * 4
    assert res.total == (8 + 8 * 8) * 4


def test_time_callable_rejects_zero_repeats():
    with pytest.raises(ValueError):
        time_callable(lambda: None, repeats=0)


def test_fixed_threads_restores_threads_and_reports_the_blas_limit():
    before = torch.get_num_threads()
    with fixed_threads(1) as settings:
        assert torch.get_num_threads() == 1
        assert settings["threads"] == 1
        assert isinstance(settings["blas_limited"], bool)
    assert torch.get_num_threads() == before


def test_peak_memory_reports_how_many_samples_were_taken_and_failed():
    res = peak_memory(lambda: None)
    assert res["rss_samples_count"] >= 1
    assert res["rss_sampler_errors"] == 0
