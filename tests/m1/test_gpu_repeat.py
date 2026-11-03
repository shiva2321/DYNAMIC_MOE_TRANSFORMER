"""Fast, pinned tests for M1 GPU repeat machinery (Job G-1).

All tests run on tiny synthetic fixtures pinned to logical processors 8 and 10 with 2 threads.
Execution of the entire test module is under 30 seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from costlab.env import fixed_threads
from growmoe.experts import MLPExpert
from m1.arms import build_arm
from m1.beds import build_bed
from m1.cell import (
    cell_filename,
    execute_cell,
    is_valid_cell_record,
    load_cell_record,
    save_cell_record,
)
from m1.gpu_repeat import (
    adapt_arm_for_device,
    execute_gpu_cell,
    run_gpu_repeat,
)
from m1.gpu_report import format_verdict_sentence, run_gpu_report

TOL_OPTIMIZER_STEP_LOSS = 1e-4
TOL_GRADIENT_REL = 1e-3
TOL_GRADIENT_ABS = 1e-5


@pytest.fixture(scope="module", autouse=True)
def pin_logical_cores():
    """Pin test execution strictly to logical processors 8 and 10 with 2 threads."""
    with fixed_threads(n=2, cpus=[8, 10]):
        yield


def test_gpu_available():
    """Verify that NVIDIA CUDA GPU is available on this host (skipping is not allowed)."""
    assert torch.cuda.is_available(), "GPU is required on this machine; skipping is not permitted."


def test_device_move_preserves_parameters_and_buffers():
    """Device move of every arm type keeps parameters and buffers equal to CPU build."""
    bed = build_bed("B1", 2, 800)
    for arm_name in ("A3", "A4", "A6", "A8"):
        kwargs = {"D": 64} if arm_name == "A6" else {"hidden": 64, "inner_steps": 2}
        arm_cpu = build_arm(
            arm_name,
            bed,
            pool_memory_bytes=200000,
            pool_macs_budget=1000000,
            **kwargs,
        )

        cpu_tensors: Dict[str, torch.Tensor] = {}
        if hasattr(arm_cpu.model, "net"):
            for name, param in arm_cpu.model.net.named_parameters():
                cpu_tensors[f"param_{name}"] = param.detach().clone()
            for name, buf in arm_cpu.model.net.named_buffers():
                cpu_tensors[f"buf_{name}"] = buf.detach().clone()
        if hasattr(arm_cpu.model, "buffer"):
            cpu_tensors["buffer_x"] = arm_cpu.model.buffer.x.detach().clone()
            cpu_tensors["buffer_y"] = arm_cpu.model.buffer.y.detach().clone()
        if hasattr(arm_cpu.model, "_x"):
            cpu_tensors["_x"] = arm_cpu.model._x.detach().clone()
            cpu_tensors["_y"] = arm_cpu.model._y.detach().clone()
        if hasattr(arm_cpu.model, "W_proj"):
            cpu_tensors["W_proj"] = arm_cpu.model.W_proj.detach().clone()
            cpu_tensors["_G"] = arm_cpu.model._G.detach().clone()
            cpu_tensors["_B"] = arm_cpu.model._B.detach().clone()
            cpu_tensors["_cnt"] = arm_cpu.model._cnt.detach().clone()

        adapt_arm_for_device(arm_cpu, device="cuda")

        if hasattr(arm_cpu.model, "net"):
            for name, param in arm_cpu.model.net.named_parameters():
                assert param.is_cuda
                assert torch.equal(param.cpu(), cpu_tensors[f"param_{name}"])
            for name, buf in arm_cpu.model.net.named_buffers():
                assert buf.is_cuda
                assert torch.equal(buf.cpu(), cpu_tensors[f"buf_{name}"])
        if hasattr(arm_cpu.model, "buffer"):
            assert arm_cpu.model.buffer.x.is_cuda
            assert torch.equal(arm_cpu.model.buffer.x.cpu(), cpu_tensors["buffer_x"])
        if hasattr(arm_cpu.model, "_x"):
            assert arm_cpu.model._x.is_cuda
            assert torch.equal(arm_cpu.model._x.cpu(), cpu_tensors["_x"])
        if hasattr(arm_cpu.model, "W_proj"):
            assert arm_cpu.model.W_proj.is_cuda
            assert torch.equal(arm_cpu.model.W_proj.cpu(), cpu_tensors["W_proj"])
            assert torch.equal(arm_cpu.model._G.cpu(), cpu_tensors["_G"])


def test_device_cpu_record_equals_execute_cell():
    """execute_gpu_cell with device='cpu' returns identical algorithmic metrics to execute_cell."""
    bed = build_bed("B1", 2, 800)
    tuning = {"hidden": 32, "inner_steps": 1}
    rec_cpu_native = execute_cell(
        bed=bed,
        arm_name="A3",
        pool_memory_bytes=100000,
        split="dev",
        profile="full",
        tuning=tuning,
    )
    rec_gpu_module = execute_gpu_cell(
        bed=bed,
        arm_name="A3",
        pool_memory_bytes=100000,
        split="dev",
        profile="full",
        tuning=tuning,
        device="cpu",
    )

    assert rec_cpu_native["accuracy"] == rec_gpu_module["accuracy"]
    assert rec_cpu_native["per_regime_accuracy"] == rec_gpu_module["per_regime_accuracy"]
    assert rec_cpu_native["forgetting"] == rec_gpu_module["forgetting"]
    assert rec_cpu_native["memory_bytes"] == rec_gpu_module["memory_bytes"]
    assert rec_cpu_native["config"] == rec_gpu_module["config"]
    assert rec_cpu_native["update_cost"]["counted_macs"] == rec_gpu_module["update_cost"]["counted_macs"]
    for b in ("1", "32"):
        assert rec_cpu_native["answer_cost"][b]["counted_macs"] == rec_gpu_module["answer_cost"][b]["counted_macs"]


def test_cuda_optimizer_step_matches_cpu():
    """One CUDA optimizer step matches CPU step within TOL_OPTIMIZER_STEP_LOSS."""
    bed = build_bed("B1", 2, 800)
    net_cpu = MLPExpert(bed.d, 64, 1, bed.n_classes, seed=123)
    net_cuda = MLPExpert(bed.d, 64, 1, bed.n_classes, seed=123).to("cuda")

    for p_c, p_g in zip(net_cpu.parameters(), net_cuda.parameters()):
        p_g.data.copy_(p_c.data)

    opt_cpu = torch.optim.Adam(net_cpu.parameters(), lr=1e-3)
    opt_cuda = torch.optim.Adam(net_cuda.parameters(), lr=1e-3)

    x_cpu = torch.randn(32, bed.d)
    y_cpu = torch.randint(0, bed.n_classes, (32,))
    x_cuda = x_cpu.clone().to("cuda")
    y_cuda = y_cpu.clone().to("cuda")

    loss_cpu = net_cpu.train_step(x_cpu, y_cpu, opt_cpu)
    loss_cuda = net_cuda.train_step(x_cuda, y_cuda, opt_cuda)

    assert abs(loss_cuda - loss_cpu) < TOL_OPTIMIZER_STEP_LOSS


def test_cuda_gradients_match_cpu():
    """The backward pass on CUDA gives the CPU gradients (the loss alone is read before the update and says nothing about it)."""
    import torch.nn.functional as F

    bed = build_bed("B1", 2, 800)
    net_cpu = MLPExpert(bed.d, 64, 1, bed.n_classes, seed=321)
    net_cuda = MLPExpert(bed.d, 64, 1, bed.n_classes, seed=321).to("cuda")
    x = torch.randn(32, bed.d)
    y = torch.randint(0, bed.n_classes, (32,))
    for net, xx, yy in ((net_cpu, x, y), (net_cuda, x.to("cuda"), y.to("cuda"))):
        net.mark_seen(yy)
        F.cross_entropy(net(xx), yy).backward()
    for (name, p_c), (_, p_g) in zip(net_cpu.named_parameters(), net_cuda.named_parameters()):
        assert torch.allclose(p_c.grad, p_g.grad.cpu(), rtol=TOL_GRADIENT_REL, atol=TOL_GRADIENT_ABS), name


def test_identical_cuda_runs_reproducibility():
    """Two identical CUDA runs of a non-trivial tiny cell give bit-identical accuracy (a cell at 100 % would prove nothing)."""
    tuning = {"hidden": 64, "inner_steps": 3}
    r1 = execute_gpu_cell(build_bed("B1", 6, 800), "A3", pool_memory_bytes=100000, profile="det", tuning=tuning, device="cuda")
    r2 = execute_gpu_cell(build_bed("B1", 6, 800), "A3", pool_memory_bytes=100000, profile="det", tuning=tuning, device="cuda")
    assert 0.0 < r1["accuracy"] < 1.0, f"a degenerate cell cannot show reproducibility: {r1['accuracy']}"
    assert r1["accuracy"] == r2["accuracy"], f"Spread: {r1['accuracy']} vs {r2['accuracy']}"
    assert r1["per_regime_accuracy"] == r2["per_regime_accuracy"]
    assert r1["forgetting"] == r2["forgetting"]


def test_update_macs_come_from_the_cpu_cell_or_are_counted_and_checked():
    """With a matching CPU cell the update multiply-adds are reused (no second CPU pass); counted, they are checked against it."""
    bed = build_bed("B1", 2, 800)
    tuning = {"hidden": 32, "inner_steps": 1}
    cpu_rec = execute_cell(bed=bed, arm_name="A3", pool_memory_bytes=100000, split="dev", profile="full", tuning=tuning)
    reused = execute_gpu_cell(bed, "A3", pool_memory_bytes=100000, profile="det", tuning=tuning, device="cuda",
                              cpu_record=cpu_rec, reuse_cpu_update_macs=True)
    counted = execute_gpu_cell(bed, "A3", pool_memory_bytes=100000, profile="det", tuning=tuning, device="cuda", cpu_record=cpu_rec)
    alone = execute_gpu_cell(bed, "A3", pool_memory_bytes=100000, profile="det", tuning=tuning, device="cuda")
    assert reused["macs_check"] == {"update": "reused_from_cpu_cell", "answer_1": "equal", "answer_32": "equal"}
    assert counted["macs_check"] == {"update": "equal", "answer_1": "equal", "answer_32": "equal"}
    assert alone["macs_check"]["update"] == "no_cpu_cell"
    assert reused["update_cost"]["counted_macs"] == counted["update_cost"]["counted_macs"] == cpu_rec["update_cost"]["counted_macs"]


def test_missing_budgets_stops_with_clear_message(tmp_path: Path):
    """Missing reference pool budgets raises RuntimeError naming the missing cell."""
    with pytest.raises(RuntimeError) as exc_info:
        run_gpu_repeat(
            split="dev",
            beds=["B1"],
            ns=[2],
            arms=["A3"],
            seeds=[800],
            out=tmp_path / "out",
            budgets_from=tmp_path / "nonexistent_budgets",
            profile="det",
            device="cuda",
        )
    msg = str(exc_info.value)
    assert "needs a current A1 record in" in msg
    assert "missing cell:" in msg


def test_stale_or_corrupt_record_is_rerun(tmp_path: Path):
    """Corrupt or stale record is detected, reported, and rerun."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    bad_file = out_dir / cell_filename("B1", 2, "A6", 800)
    bad_file.write_text("{corrupt json", encoding="utf-8")

    run_gpu_repeat(
        split="dev",
        beds=["B1"],
        ns=[2],
        arms=["A6"],
        seeds=[800],
        out=out_dir,
        budgets_from=out_dir,
        tuning_dir=None,
        profile="det",
        device="cuda",
    )
    rec = load_cell_record(bad_file)
    assert is_valid_cell_record(rec)
    assert rec["arm"] == "A6"
    assert rec["device"] == "cuda"


def test_split_report_freeze_and_seeds_guard(tmp_path: Path):
    """--split report refuses seeds outside 810-819 and missing freeze."""
    from m1.freeze import check_frozen

    problems = check_frozen(seeds=[800])
    assert any("seed 800 is not a report seed" in p for p in problems)

    empty_root = tmp_path / "empty_repo"
    empty_root.mkdir()
    problems_empty = check_frozen(seeds=[810], root=empty_root)
    assert any("preregistration is not frozen" in p for p in problems_empty)

    with pytest.raises(ValueError) as exc:
        run_gpu_repeat(
            split="report",
            beds=["B1"],
            ns=[2],
            arms=["A6"],
            seeds=[800],
            out=tmp_path / "out",
            profile="det",
        )
    assert "not a report seed" in str(exc.value)


def test_report_refuses_mixed_splits(tmp_path: Path):
    """m1.gpu_report refuses to mix DEV and REPORT cells."""
    cpu_dir = tmp_path / "cpu"
    gpu_dir = tmp_path / "gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()

    bed_dev = build_bed("B1", 2, 800)
    bed_rep = build_bed("B1", 2, 810)
    rec_dev = execute_gpu_cell(bed_dev, "A6", split="dev", profile="det", device="cpu")
    rec_rep = execute_gpu_cell(bed_rep, "A6", split="report", profile="det", device="cuda")
    save_cell_record(cpu_dir / cell_filename("B1", 2, "A6", 800), rec_dev)
    save_cell_record(gpu_dir / cell_filename("B1", 2, "A6", 810), rec_rep)

    with pytest.raises(ValueError) as exc:
        run_gpu_report(cpu_dir=cpu_dir, gpu_dir=gpu_dir, out_dir=tmp_path / "out")
    assert "Split mismatch" in str(exc.value)


def _synthetic_gpu(cell: Dict[str, Any], up_wall: float) -> Dict[str, Any]:
    """The GPU cell of a hand-made CPU cell: same accuracy, other update wall-clock, device and VRAM fields."""
    gpu = json.loads(json.dumps(cell))
    gpu["update_cost"]["wall_clock_total"] = up_wall
    gpu.update(device="cuda", vram_peak_bytes=100_000_000)
    return gpu


def test_report_on_synthetic_cells_changed_unchanged_and_not_evaluated(tmp_path: Path):
    """Whole report on hand-made SYNTHETIC cells: a verdict that flips, one that does not, and a bed without any GPU cell."""
    from tests.m1.test_verdict import make_cell

    cpu_cells, gpu_cells = [], []
    for n_regimes, gpu_up_wall in ((10, 2.0), (30, 25.0), (100, 2.0)):   # the GPU baseline gets cheaper than the pool (10 s) on the first and the third
        pool = make_cell(bed="B1", n_regimes=n_regimes, arm="A1", seed=800, acc=0.40, up_wall=10.0, lat_b1=0.005, up_macs=1000, ans_macs=500, split="SYNTHETIC")
        base = make_cell(bed="B1", n_regimes=n_regimes, arm="A3", seed=800, acc=0.40, up_wall=20.0, lat_b1=0.004, up_macs=2000, ans_macs=500, split="SYNTHETIC")
        cpu_cells += [pool, base]
        gpu = _synthetic_gpu(base, gpu_up_wall)
        if n_regimes == 100:
            gpu["config"]["hidden"] = 4096                       # another width than the CPU cell: not a hardware difference, must not be paired
        gpu_cells.append(gpu)
    cpu_cells += [make_cell(bed="B2", n_regimes=6, arm="A1", seed=800, split="SYNTHETIC"),
                  make_cell(bed="B2", n_regimes=6, arm="A3", seed=800, up_wall=20.0, split="SYNTHETIC")]
    cpu_dir, gpu_dir = tmp_path / "cpu", tmp_path / "gpu"
    cpu_dir.mkdir()
    gpu_dir.mkdir()
    for rec in cpu_cells:
        save_cell_record(cpu_dir / cell_filename(rec["bed"], rec["n_regimes"], rec["arm"], rec["seed"]), rec)
    for rec in gpu_cells:
        save_cell_record(gpu_dir / cell_filename(rec["bed"], rec["n_regimes"], rec["arm"], rec["seed"]), rec)

    run_gpu_report(cpu_dir=cpu_dir, gpu_dir=gpu_dir, out_dir=tmp_path / "out")
    payload = json.loads((tmp_path / "out" / "gpu_verdicts.json").read_text(encoding="utf-8"))
    sentences = payload["sentences"]
    assert "SYNTHETIC" in (tmp_path / "out" / "REPORT_M1_GPU.md").read_text(encoding="utf-8").splitlines()[0]
    assert "CHANGED from not firing on CPU to firing on GPU" in sentences["B1_n10"]["R1_primary"]
    assert "UNCHANGED" in sentences["B1_n30"]["R1_primary"]
    assert "NOT EVALUATED" in sentences["B2"]["R1_primary"]
    assert "NOT EVALUATED" in sentences["B1_n100"]["R1_primary"]      # its only GPU cell has another configuration
    assert payload["gpu_substitutions"]["B1_n10"] == {"replaced": 1, "repeated_cpu_cells": 1}
    assert payload["gpu_substitutions"]["B2"] == {"replaced": 0, "repeated_cpu_cells": 1}
    assert payload["gpu_substitutions"]["B1_n100"] == {"replaced": 0, "repeated_cpu_cells": 1}
    assert payload["config_mismatches"] == ["B1_n100 A3 seed 800"]
    assert "B1_n100" not in payload["comparison"]


def test_synthetic_verdict_sentences_changed_and_unchanged():
    """Verdict sentence generator produces correct changed and unchanged statements."""
    c_v_unchanged = {"R1_primary": {"fired": False, "status": "FAIL", "triggering_baselines": []}}
    g_v_unchanged = {"R1_primary": {"fired": False, "status": "FAIL", "triggering_baselines": []}}
    s_unchanged = format_verdict_sentence("B1_n10", c_v_unchanged, g_v_unchanged, "R1_primary")
    assert "UNCHANGED" in s_unchanged

    c_v_changed = {"R1_primary": {"fired": False, "status": "FAIL", "triggering_baselines": []}}
    g_v_changed = {"R1_primary": {"fired": True, "status": "PASS", "triggering_baselines": ["A3"]}}
    s_changed = format_verdict_sentence("B1_n10", c_v_changed, g_v_changed, "R1_primary")
    assert "CHANGED" in s_changed
    assert "triggered by A3" in s_changed
