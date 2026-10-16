"""The function-regime (s3) dev cells with the twin fix: a signature never holds a point twice, and the young expert is no longer given the x4 relaxation (infant_relax 1).

Finding (E-036): when an expert is founded its cluster enters its signature twice (the spawn, then the training loop), which wrecks the calibration of the new expert; the x4 infant relaxation hides that for
a few batches and blinds the pool to a new regime during them. This driver runs the SAME cells as run_s3_unaliased.py (same streams, seeds, arms, constants, pool seed = stream seed + 7, so the two are
paired) with two changes made in the process, no product file touched:

  * `VmfSignature.update` skips points the signature already holds (exactly what `PhyloCfg.sig_dedupe=True` does on branch stage4-1-cap-and-telemetry);
  * `infant_relax` = 1.0 (variant "x1") or left at 4.0 (variant "x4").

The results go to `cells_s3d_<variant>/` and carry the code stamp that was current when the cell started.

    python run_s3_dedupe.py list --variant x1
    python run_s3_dedupe.py run --variant x1 --workers 2        # run what is missing (safe to repeat after a power cut)
"""

import argparse
import dataclasses
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rv2d_lib as R  # noqa: E402

OFFSET = 7
SEEDS = (100, 101, 102)
CONST = json.loads((HERE / "dev_constants.json").read_text(encoding="utf-8")) if (HERE / "dev_constants.json").exists() else {"grow_gap": 0.17, "transfer_margin": 0.25}
STREAMS = ("F-15", "OV-0.6", "OV-0.85", "S1", "S2-5")
ARMS = ("S3", "S3+T+G")
VARIANTS = {"x1": 1.0, "x4": 4.0}
TOL = 1.0 - 1e-5


def out_dir(variant: str) -> Path:
    return HERE / f"cells_s3d_{variant}"


def cells():
    g, t = CONST["grow_gap"], CONST["transfer_margin"]
    return [R.Cell("s3", st, arm, s, gap=g if "G" in arm else 0.0, tmargin=t if "T" in arm else 0.0) for st in STREAMS for s in SEEDS for arm in ARMS]


def patch_signature() -> None:
    from growmoe.signature import VmfSignature
    original = VmfSignature.update

    def update_without_twins(self, x) -> None:
        u = self._unit(x)
        if self._n > 0:
            u = u[(u @ self.protos.T).max(dim=1).values < TOL]
        if u.shape[0]:
            original(self, u)
    VmfSignature.update = update_without_twins


def run_cell_here(cell: R.Cell, variant: str) -> None:
    import torch
    torch.set_num_threads(2)
    out = out_dir(variant)
    path = out / f"{cell.cid}.json"
    stamp = R.code_stamp()
    if R.cell_done(path, stamp):
        return
    import run_rv2c as R3
    patch_signature()
    orig = R3.arm_cfg
    relax = VARIANTS[variant]
    R3.arm_cfg = lambda arm, st, seed, steps: dataclasses.replace(orig(arm, st, seed, steps), seed=seed + OFFSET, infant_relax=relax)
    R.CELLS = out                                                                       # checkpoints and results of this driver live apart from every other cell folder
    out.mkdir(exist_ok=True)
    cpu0 = time.process_time()
    rec = R.run_cell(cell)
    rec["cpu_seconds"] = time.process_time() - cpu0
    rec["code_stamp"] = stamp
    rec["pool_seed_offset"] = OFFSET
    rec["twin_fix"] = {"variant": variant, "infant_relax": relax, "dedupe": True}
    R.atomic_write_text(path, json.dumps(rec))


def cmd_run(args) -> None:
    todo = [c for c in cells() if not R.cell_done(out_dir(args.variant) / f"{c.cid}.json", R.code_stamp())]
    print(f"{len(todo)} cells to run of {len(cells())} (variant {args.variant})", flush=True)
    running = []
    while todo or running:
        running = [p for p in running if p.poll() is None]
        while todo and len(running) < args.workers:
            c = todo.pop(0)
            running.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "one", "--cid", c.cid, "--variant", args.variant]))
            time.sleep(1)
        time.sleep(5)
    print("all cells finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "run", "one"))
    ap.add_argument("--variant", default="x1", choices=tuple(VARIANTS))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cid")
    a = ap.parse_args()
    if a.cmd == "list":
        for c in cells():
            print(c.cid)
    elif a.cmd == "run":
        cmd_run(a)
    else:
        run_cell_here({c.cid: c for c in cells()}[a.cid], a.variant)
