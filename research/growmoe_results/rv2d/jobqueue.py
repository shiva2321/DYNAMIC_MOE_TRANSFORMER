"""One idempotent driver for the RV2-D runs (routing v2 stage 4). Built for a machine that loses power now and then (memory note plan-for-daily-power-outages).

    python jobqueue.py status --phase dev1       what is done, running, pending (a result file counts as done only if it parses)
    python jobqueue.py run    --phase dev1       launch what is missing, up to the CPU-thread budget, then exit
    python jobqueue.py run    --phase dev1 --wait     stay and keep launching as slots free up, until everything is done (then rebuild the report)
    python jobqueue.py run    --phase dev1 --detach   start the --wait scheduler as a hidden background process and return at once  (this is what RESUME_RV2D.cmd does)

Phases: dev1 (baselines and the calibration sweeps), dev2 (every arm at the chosen constants), confirm (seeds 85-99, single use, only after the user's go).
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
PY = sys.executable
REPORT_OF = {"dev1": "dev", "dev2": "dev", "confirm": "confirm"}               # which report a finished phase rebuilds (the smoke phase has none)
BUDGET = 9                                                     # CPU threads this machine may give (3 cells x 3 threads), so that it stays usable for other work


class Job:
    def __init__(self, name, args, markers, threads=3):
        self.name, self.args, self.markers, self.threads = name, list(args), list(markers), threads
        self.match = f"--cid {name}"

    def done(self) -> bool:
        return all(R.valid_json(HERE / m) for m in self.markers)

    def checkpoint(self) -> bool:
        return any((HERE / m).with_name(Path(m).stem + ".ckpt").exists() for m in self.markers)


def build_jobs(phase):
    cells = json.loads((HERE / f"cells_{phase}.json").read_text(encoding="utf-8"))
    order = {"arith-seq": 0, "arith-mixed128": 1, "arith-mixed16": 1, "banking": 2, "s3": 3}               # the long checkpointed cells first
    jobs = []
    for c in sorted(cells, key=lambda c: (order[c["family"]], c["seed"])):
        cid = R.Cell(**c).cid
        jobs.append(Job(cid, ["run_cell.py", "--phase", phase, "--cid", cid], [f"{R.CELLS.name}/{cid}.json"], 3))
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


def states(jobs):
    lines = running_command_lines()
    return {j.name: "DONE" if j.done() else ("RUNNING" if is_running(j, lines) else "PENDING") for j in jobs}


def cmd_status(phase, verbose=False):
    jobs = build_jobs(phase)
    st = states(jobs)
    for j in jobs:
        if st[j.name] != "DONE" and (verbose or st[j.name] == "RUNNING"):
            print(f"{j.name:<90s} {st[j.name]:<8s}{'  [checkpoint present: will resume]' if (st[j.name] != 'DONE' and j.checkpoint()) else ''}")
    counts = {k: sum(1 for v in st.values() if v == k) for k in ("DONE", "RUNNING", "PENDING")}
    print(f"{phase}: {counts['DONE']} done, {counts['RUNNING']} running, {counts['PENDING']} pending (of {len(jobs)} cells)")
    return st


def launch(job, phase):
    LOGS.mkdir(exist_ok=True)
    out = open(LOGS / f"q_{job.name}.log", "ab")
    err = open(LOGS / f"q_{job.name}.err", "ab")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)         # hidden console, own process group: survives the shell that started it
    p = subprocess.Popen([PY, str(HERE / job.args[0])] + [str(a) for a in job.args[1:]], cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags, close_fds=True)
    print(f"{time.strftime('%H:%M:%S')} launched {job.name} (pid {p.pid})", flush=True)


def scheduler_alive() -> bool:
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text().split()[0])
    except Exception:
        return False
    return pid != os.getpid() and any("jobqueue.py" in ln and "run" in ln and "--wait" in ln for ln in running_command_lines())


def cmd_run(phase, wait):
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
    LOGS.mkdir(exist_ok=True)
    out = open(LOGS / "queue_scheduler.log", "ab")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    p = subprocess.Popen([PY, str(HERE / "jobqueue.py"), "run", "--phase", phase, "--wait"], cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=out, stderr=out, creationflags=flags, close_fds=True)
    print(f"scheduler for {phase} started in the background (pid {p.pid}); log: logs/queue_scheduler.log; check with: python jobqueue.py status --phase {phase}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("status", "run"))
    ap.add_argument("--phase", default="dev1", choices=("dev1", "dev2", "confirm", "smoke"))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        cmd_status(a.phase, a.v)
    elif a.detach:
        cmd_detach(a.phase)
    else:
        cmd_run(a.phase, a.wait)
