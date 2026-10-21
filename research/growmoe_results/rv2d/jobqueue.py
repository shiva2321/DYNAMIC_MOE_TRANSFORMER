"""One idempotent driver for the RV2-D runs (routing v2 stage 4). Built for a machine that loses power now and then (memory note plan-for-daily-power-outages).

    python jobqueue.py status --phase dev1       what is done, running, pending (a result file counts as done only if it parses)
    python jobqueue.py run    --phase dev1       launch what is missing, up to the CPU-thread budget, then exit
    python jobqueue.py run    --phase dev1 --wait     stay and keep launching as slots free up, until everything is done (then rebuild the report)
    python jobqueue.py run    --phase dev1 --detach   start the --wait scheduler as a hidden background process and return at once  (this is what RESUME_RV2D.cmd does)

Phases: dev1 (baselines and the calibration sweeps), dev2 (every arm at the chosen constants), confirm (seeds 85-99, single use; run on 2026-10-21 and CLOSED since: see CONFIRM_CLOSED.json).
Safe to run again and again: finished cells are skipped, a running cell is not started twice, an interrupted cell resumes from its own checkpoint (arithmetic cells save one after
every block or every 150 batches). After a power cut: run ``RESUME_RV2D.cmd <phase>``.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rv2d_lib as R  # noqa: E402

LOGS = HERE / "logs"
LOCK = HERE / "queue.lock"
CLOSED = HERE / "CONFIRM_CLOSED.json"                         # present once the confirmatory run is finished and reported: seeds 85-99 are single use, so that phase can never be started again
PY = sys.executable
REPORT_OF = {"dev1": "dev", "dev2": "dev", "confirm": "confirm"}               # which report a finished phase rebuilds (the smoke phase has none)
BUDGET = int(os.environ.get("RV2D_BUDGET", "9"))              # CPU threads this machine may give (default 3 cells x 3 threads, so that it stays usable); `run --budget N` raises it
THREADS_OF = {"arith-seq": 1, "arith-mixed16": 1, "arith-mixed128": 1, "banking": 2, "s3": 1}     # CORES a cell needs, for the budget. The arithmetic cells still set 3 torch threads themselves (the numbers do not change), but measured on this machine 3 threads make an expert step only 11 % faster than 1 (3.12 against 2.79 ms; the GPU 2.59 ms): the cell is Python-bound, so it needs about one core. A function-regime cell needs about 0.8.


class Job:
    def __init__(self, name, args, markers, threads=3):
        self.name, self.args, self.markers, self.threads = name, list(args), list(markers), threads
        self.match = f"--cid {name}"

    def done(self, stamp: str) -> bool:
        return all(R.cell_done(HERE / m, stamp) for m in self.markers)

    def checkpoint(self) -> bool:
        return any((HERE / m).with_name(Path(m).stem + ".ckpt").exists() for m in self.markers)


def build_jobs(phase):
    cells = json.loads((HERE / f"cells_{phase}.json").read_text(encoding="utf-8"))
    order = {"arith-seq": 0, "arith-mixed128": 1, "arith-mixed16": 1, "banking": 2, "s3": 3}               # the long checkpointed cells first
    jobs = []
    for c in sorted(cells, key=lambda c: (order[c["family"]], c["seed"])):
        cid = R.Cell(**c).cid
        jobs.append(Job(cid, ["run_cell.py", "--phase", phase, "--cid", cid], [f"{R.CELLS.name}/{cid}.json"], THREADS_OF[c["family"]]))
    return jobs


def running_command_lines():
    ps = "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { $_.CommandLine }"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    return [ln.replace('"', "") for ln in out.splitlines() if ln.strip()]


def is_running(job, lines):
    return any(job.match in ln and "run_cell.py" in ln for ln in lines)


def closed_note(phase):
    """The record of a closed phase (only ``confirm`` can be closed), or None."""
    return json.loads(CLOSED.read_text(encoding="utf-8")) if phase == "confirm" and CLOSED.exists() else None


def states(jobs, stamp=None):
    lines, stamp = running_command_lines(), stamp or R.code_stamp()               # a result of other code than the present counts as missing
    return {j.name: "DONE" if j.done(stamp) else ("RUNNING" if is_running(j, lines) else "PENDING") for j in jobs}


def cmd_status(phase, verbose=False):
    jobs = build_jobs(phase)
    closed = closed_note(phase)
    st = states(jobs, closed["stamp"] if closed else None)                        # a closed phase is read at the stamp it was run at, not at the (adopted) code of today
    for j in jobs:
        if st[j.name] != "DONE" and (verbose or st[j.name] == "RUNNING"):
            print(f"{j.name:<90s} {st[j.name]:<8s}{'  [checkpoint present: will resume]' if (st[j.name] != 'DONE' and j.checkpoint()) else ''}")
    counts = {k: sum(1 for v in st.values() if v == k) for k in ("DONE", "RUNNING", "PENDING")}
    print(f"{phase}: {counts['DONE']} done, {counts['RUNNING']} running, {counts['PENDING']} pending (of {len(jobs)} cells)")
    if closed:
        print(f"{phase} is CLOSED since {closed['date']}: {closed['note']}")
    return st


def launch(job, phase):
    LOGS.mkdir(exist_ok=True)
    out = open(LOGS / f"q_{job.name}.log", "ab")
    err = open(LOGS / f"q_{job.name}.err", "ab")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)         # hidden console, own process group: survives the shell that started it
    if BUDGET > 9:
        flags |= getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0x00004000)                               # a wide budget runs below normal priority so that the PC stays usable
    env = {**os.environ, "OMP_NUM_THREADS": str(job.threads), "MKL_NUM_THREADS": str(job.threads)}           # without this a cell that sets no thread count takes all 12 cores and three of them fight
    if BUDGET > 9:
        env.update({"OMP_WAIT_POLICY": "PASSIVE", "KMP_BLOCKTIME": "0"})                                    # more threads than cores: idle worker threads sleep instead of spinning (same partition of the work, same numbers)
    p = subprocess.Popen([PY, str(HERE / job.args[0])] + [str(a) for a in job.args[1:]], cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags, close_fds=True, env=env)
    print(f"{time.strftime('%H:%M:%S')} launched {job.name} (pid {p.pid})", flush=True)


def command_line_of(pid: int) -> str:
    ps = f"(Get-CimInstance Win32_Process -Filter 'ProcessId={int(pid)}').CommandLine"
    try:
        return subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"{time.strftime('%H:%M:%S')} could not read the command line of pid {pid}: {exc}", flush=True)
        return ""


def scheduler_alive() -> bool:
    """True when the scheduler named in queue.lock is still running. A lock left behind by a power cut names a dead process (or a new one that is not a scheduler), and the scheduler that is being
    started now carries --wait itself, so the command lines of all python processes cannot be used for this."""
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return False
    if pid == os.getpid():
        return False
    line = command_line_of(pid)
    return "jobqueue.py" in line and "--wait" in line


def refuse_if_closed(phase) -> bool:
    closed = closed_note(phase)
    if closed:
        print(f"the {phase} phase is CLOSED ({closed['note']}); its seeds are single use, so nothing was started")
    return bool(closed)


def cmd_run(phase, wait):
    if refuse_if_closed(phase):
        return
    if wait and scheduler_alive():
        print("another scheduler is already running (queue.lock); nothing to do")
        return
    if wait:
        LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    jobs = build_jobs(phase)
    try:
        while True:
            st = states(jobs)
            used = sum(j.threads for j in jobs if st[j.name] == "RUNNING")
            for j in jobs:
                if st[j.name] == "PENDING" and used + j.threads <= BUDGET:
                    launch(j, phase)
                    used += j.threads
                    st[j.name] = "RUNNING"
            if not wait:
                break
            if all(v == "DONE" for v in st.values()):
                print(f"{time.strftime('%H:%M:%S')} everything is done; rebuilding the report", flush=True)
                if (HERE / "report_rv2d.py").exists() and phase in REPORT_OF:
                    subprocess.run([PY, str(HERE / "report_rv2d.py"), REPORT_OF[phase]], cwd=str(HERE))
                break
            time.sleep(30)
    finally:
        if wait and LOCK.exists():
            try:
                LOCK.unlink()
            except Exception:
                pass


def cmd_detach(phase):
    if refuse_if_closed(phase):
        return
    LOGS.mkdir(exist_ok=True)
    out = open(LOGS / "queue_scheduler.log", "ab")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    p = subprocess.Popen([PY, str(HERE / "jobqueue.py"), "run", "--phase", phase, "--wait", "--budget", str(BUDGET)], cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=out, stderr=out, creationflags=flags, close_fds=True)
    print(f"scheduler for {phase} started in the background (pid {p.pid}); log: logs/queue_scheduler.log; check with: python jobqueue.py status --phase {phase}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("status", "run"))
    ap.add_argument("--phase", default="dev1", choices=("dev1", "dev2", "confirm", "smoke"))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--budget", type=int, default=None, help="CPU threads the queue may use (default 9, or RV2D_BUDGET)")
    a = ap.parse_args()
    if a.budget is not None:
        BUDGET = a.budget
    if a.cmd == "status":
        cmd_status(a.phase, a.v)
    elif a.detach:
        cmd_detach(a.phase)
    else:
        cmd_run(a.phase, a.wait)
