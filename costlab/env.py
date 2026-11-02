"""Environment discovery and thread/affinity controls."""

from __future__ import annotations

import contextlib
import datetime
import os
import platform
import subprocess
import warnings
from typing import Any, Generator

import psutil
import torch

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # ships with scikit-learn; without it numpy's BLAS threads cannot be limited, and the settings say so
    threadpool_limits = None


def _windows_cpu_name() -> str | None:
    """Return the processor name from the Windows registry, or None when it cannot be read."""
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        ) as key:
            val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
    except (ImportError, OSError):
        return None
    return val.strip() if isinstance(val, str) and val.strip() else None


def _get_friendly_cpu_name() -> str:
    """Return friendly CPU model name if available, else platform processor."""
    name = _windows_cpu_name() if platform.system() == "Windows" else None
    return name or platform.processor() or "unknown"


def _get_git_commit() -> str | None:
    """Return current git commit hash, or None when git or the repository is not available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = res.stdout.strip()
    return commit if commit else None


def _get_gpu_driver() -> str | None:
    """Return GPU driver version from NVML, or None when NVML is not available."""
    try:
        import pynvml
    except ImportError:
        return None
    try:
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
    except pynvml.NVMLError:
        return None
    return driver.decode("utf-8") if isinstance(driver, bytes) else str(driver)


def environment() -> dict[str, Any]:
    """Capture system, PyTorch, CUDA, CPU, RAM, and runtime environment details.

    Returns:
        dict: environment details including python version, torch version,
            CUDA availability and version, GPU name, GPU driver, CPU name,
            logical core count, total RAM in bytes, torch thread counts,
            relevant threading environment variables, git commit hash,
            and UTC ISO timestamp.
    """
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else None
    cuda_ver = torch.version.cuda if has_cuda else None
    gpu_driver = _get_gpu_driver() if has_cuda else None

    env_vars = {
        var: os.environ.get(var, "unset")
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": cuda_ver,
        "cuda_available": has_cuda,
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "cpu_name": _get_friendly_cpu_name(),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "env_vars": env_vars,
        "git_commit": _get_git_commit(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


@contextlib.contextmanager
def fixed_threads(
    n: int,
    cpus: list[int] | None = None,
) -> Generator[dict[str, Any], None, None]:
    """Context manager setting torch intra-op threads and optional CPU affinity.

    torch.set_num_threads does not touch the BLAS and OpenMP pools that numpy, scipy and
    scikit-learn use, so the same limit is also set through threadpoolctl when it is
    installed; the yielded settings say whether that happened ("blas_limited").
    Restores threads and CPU affinity on exit; a failed affinity restore raises a warning.
    On an SMT machine pin one logical processor per physical core: on this machine two threads
    on both siblings of one core (and its neighbour) gave 10-20 % jitter in 25 ms timings,
    one logical processor per core (for example [0, 2]) gave 1-3 %.

    Args:
        n: Number of threads.
        cpus: Optional list of logical CPU indices to pin the process to.

    Yields:
        dict: The active settings: "threads", "cpus" and "blas_limited".
    """
    old_threads = torch.get_num_threads()
    proc = psutil.Process()
    old_affinity = proc.cpu_affinity() if hasattr(proc, "cpu_affinity") else None

    with contextlib.ExitStack() as stack:
        if threadpool_limits is not None:
            stack.enter_context(threadpool_limits(limits=n))
        try:
            torch.set_num_threads(n)
            if cpus is not None and hasattr(proc, "cpu_affinity"):
                proc.cpu_affinity(list(cpus))
            yield {"threads": n, "cpus": cpus, "blas_limited": threadpool_limits is not None}
        finally:
            torch.set_num_threads(old_threads)
            if old_affinity is not None and hasattr(proc, "cpu_affinity"):
                try:
                    proc.cpu_affinity(old_affinity)
                except (psutil.Error, OSError) as exc:
                    warnings.warn(f"could not restore the CPU affinity {old_affinity}: {exc}", RuntimeWarning)
