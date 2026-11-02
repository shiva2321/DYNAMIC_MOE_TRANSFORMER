"""Tests for costlab.report: report schema, JSON serialization, and Markdown export."""

import json
from pathlib import Path

import torch
import torch.nn as nn

from costlab.report import measure, to_json, to_markdown


def test_measure_and_json_roundtrip(tmp_path: Path):
    """Verify measure produces expected schema and JSON serialization round-trips."""
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

    def make_fn(batch_size):
        x = torch.randn(batch_size, 16)
        return lambda: model(x)

    batch_sizes = (1, 4)
    rep = measure(make_fn, model=model, batch_sizes=batch_sizes, device="cpu")

    # Verify schema keys
    assert rep["schema_version"] == "1.0.0"
    assert "environment" in rep
    assert "measurements" in rep
    assert "bytes" in rep
    assert "macs" in rep
    assert "timing" in rep
    assert "memory" in rep
    assert "energy" in rep

    for b in batch_sizes:
        b_str = str(b)
        assert b_str in rep["measurements"]
        summary = rep["measurements"][b_str]
        assert "latency_median_ms" in summary
        assert "throughput_items_s" in summary
        assert "macs" in summary

    # JSON round trip
    json_path = tmp_path / "test_report.json"
    to_json(rep, json_path)
    assert json_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["schema_version"] == rep["schema_version"]
    assert str(batch_sizes[0]) in loaded["measurements"]

    # Markdown export
    md = to_markdown(rep)
    assert "# Costlab Measurement Report" in md
    assert "Environment" in md
    assert "Measurements Summary" in md
