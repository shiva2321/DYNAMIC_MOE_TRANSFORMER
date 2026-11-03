"""Short GPU-vs-CPU speed probe for the wide baselines of M1 (Q-83), real shapes, one process, under a minute.

CPU side: 2 threads pinned to logical processors 8 and 10 (physical cores 4 and 5), far from the timed pass on 0 and 2.
GPU side: the same two logical processors issue the kernels. Nothing is written to the repo except the JSON this script is told to write.
Shapes: d = 128, batch 64 new + 64 replay (replay_frac 0.5), 64 classes, Adam lr 3e-3 (growmoe.config.CommonCfg and ReplayNet.observe).
"""

import json
import sys
import time
from pathlib import Path

REPO = Path(r"D:\spherical_hyperspace_memory")
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from costlab.env import fixed_threads
from growmoe.experts import MLPExpert

D_IN, N_CLASSES, N_NEW, N_REPLAY = 128, 64, 64, 64
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "results" / "m1" / "diag" / "gpu_probe_rerun.json"


def make(hidden: int, device: str):
    net = MLPExpert(D_IN, hidden, 1, N_CLASSES, seed=0).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(1)
    x = F.normalize(torch.randn(N_NEW, D_IN, generator=g), dim=-1).to(device)
    y = torch.randint(0, N_CLASSES, (N_NEW,), generator=g).to(device)
    xr = F.normalize(torch.randn(N_REPLAY, D_IN, generator=g), dim=-1).to(device)
    yr = torch.randint(0, N_CLASSES, (N_REPLAY,), generator=g).to(device)
    return net, opt, x, y, (xr, yr)


def time_steps(hidden: int, device: str, warm: int, n: int) -> float:
    net, opt, x, y, rep = make(hidden, device)
    for _ in range(warm):
        net.train_step(x, y, opt, replay=rep)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        net.train_step(x, y, opt, replay=rep)
    if device == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000.0
    del net, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    return ms


def time_a6(D: int, device: str, reps: int) -> dict:
    g = torch.Generator().manual_seed(2)
    W = (torch.randn(D_IN, D, generator=g) / D_IN ** 0.5).to(device)
    x = F.normalize(torch.randn(N_NEW, D_IN, generator=g), dim=-1).to(device)
    y = torch.randint(0, N_CLASSES, (N_NEW,), generator=g).to(device)
    G = torch.zeros(D, D, dtype=torch.float64, device=device)
    B = torch.zeros(D, N_CLASSES, dtype=torch.float64, device=device)

    def acc():
        phi = F.relu(x @ W).double()
        G.add_(phi.T @ phi)
        B.add_(phi.T @ F.one_hot(y, N_CLASSES).double())

    for _ in range(2):
        acc()
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        acc()
    if device == "cuda":
        torch.cuda.synchronize()
    acc_ms = (time.perf_counter() - t0) / reps * 1000.0
    A = G + 0.1 * torch.eye(D, dtype=torch.float64, device=device)
    A = 0.5 * (A + A.T)
    t0 = time.perf_counter()
    L = torch.linalg.cholesky(A)
    Wt = torch.cholesky_solve(B, L)
    if device == "cuda":
        torch.cuda.synchronize()
    solve_ms = (time.perf_counter() - t0) * 1000.0
    del G, B, A, L, Wt
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"accumulate_ms": acc_ms, "solve_ms": solve_ms}


def main() -> None:
    res = {"start": time.strftime("%Y-%m-%d %H:%M:%S"), "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0), "train_step": {}, "a6": {}}
    free0, total = torch.cuda.mem_get_info()
    res["vram_free_before_mb"], res["vram_total_mb"] = free0 // 2 ** 20, total // 2 ** 20
    with fixed_threads(n=2, cpus=[8, 10]):
        for hidden, n_cpu, n_gpu in ((256, 200, 400), (1024, 100, 400), (4096, 40, 300), (16384, 12, 200)):
            cpu = time_steps(hidden, "cpu", 2, n_cpu)
            gpu = time_steps(hidden, "cuda", 20, n_gpu)
            res["train_step"][str(hidden)] = {"cpu_ms": cpu, "gpu_ms": gpu, "speedup": cpu / gpu}
            print(f"hidden {hidden:>6}: cpu {cpu:8.2f} ms  gpu {gpu:6.2f} ms  x{cpu / gpu:5.1f}", flush=True)
        for D in (1024, 4096):
            c = time_a6(D, "cpu", 6)
            g = time_a6(D, "cuda", 20)
            res["a6"][str(D)] = {"cpu": c, "gpu": g, "speedup_accumulate": c["accumulate_ms"] / g["accumulate_ms"], "speedup_solve": c["solve_ms"] / g["solve_ms"]}
            print(f"A6 D={D}: accumulate cpu {c['accumulate_ms']:.1f} gpu {g['accumulate_ms']:.1f} ms | solve cpu {c['solve_ms']:.0f} gpu {g['solve_ms']:.0f} ms", flush=True)
    res["vram_peak_mb"] = torch.cuda.max_memory_allocated() // 2 ** 20
    res["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("window", res["start"], "->", res["end"], "| peak VRAM allocated", res["vram_peak_mb"], "MB")


if __name__ == "__main__":
    main()
