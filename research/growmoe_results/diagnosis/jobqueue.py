"""One idempotent driver for the whole diagnosis study. Built for a machine that loses power now and then (memory note plan-for-daily-power-outages).

    python jobqueue.py status            what is done, running, pending (a result file counts as done only if it parses)
    python jobqueue.py run               launch what is missing, up to the CPU-thread budget, then exit
    python jobqueue.py run --wait        stay and keep launching as slots free up, until everything is done (then rebuild the report)
    python jobqueue.py run --detach      start the --wait scheduler as a hidden background process and return at once  (this is what RESUME_DIAGNOSIS.cmd does)

Safe to run again and again: finished work is skipped, work that is running is not started twice, interrupted work resumes from its own result files / checkpoints
(``run_seqprobe.py`` saves a checkpoint after every block). After a power cut: run ``RESUME_DIAGNOSIS.cmd`` (or ``python jobqueue.py run --detach``).
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

from ladder_cells import shard_files  # noqa: E402

LOGS = HERE / "logs"
LOCK = HERE / "queue.lock"
PY = sys.executable
BUDGET = 12                                                    # CPU threads this machine can give (12 logical cores); jobs declare theirs


def ok_json(rel: str) -> bool:
    try:
        with open(HERE / rel, "r", encoding="utf-8") as fh:
            json.load(fh)
        return True
    except Exception:
        return False


class Job:
    def __init__(self, name, args, markers, threads=3, deps=(), note="", check=None):
        self.name, self.args, self.markers, self.threads, self.deps, self.note, self.check = name, list(args), list(markers), threads, tuple(deps), note, check
        self.match = " ".join([args[0]] + [str(a) for a in args[1:]])

    def done(self) -> bool:
        if self.check is not None:
            try:
                return bool(self.check())
            except Exception:
                return False
        return all(ok_json(m) if m.endswith(".json") else (HERE / m).exists() for m in self.markers)

    def checkpoint(self) -> bool:
        return any((HERE / m).with_suffix(".ckpt").exists() for m in self.markers if m.endswith(".json"))


def build_jobs():
    J = []
    # the long, checkpointed sequential runs first (they decide how long everything takes)
    for v in ("default", "forced_spawn"):
        for s in (0, 1):
            J.append(Job(f"seq3-{v}-s{s}", ["run_seqprobe.py", "--variant", v, "--seed", s, "--out", "seqprobe3"], [f"seqprobe3/{v}_s{s}.json"], 3,
                         note="checkpoint after every block; also saves the final pool and the routing-rule comparison"))
    J.append(Job("router", ["analyse_router.py"], ["router.json"], 2, deps=[f"seq3-{v}-s{s}" for v in ("default", "forced_spawn") for s in (0, 1)], note="learned router on the saved pools",
                 check=lambda: ok_json("router.json") and len(json.load(open(HERE / "router.json", encoding="utf-8"))) >= 4))
    # the rest of the study (all finished on 2026-10-14; kept here so that the study can be re-created and re-checked)
    for k in range(3):
        J.append(Job(f"ladder-{k}", ["run_ladder.py", "--seeds", "0,1", "--shard", f"{k}/3", "--all"], ["ladder/" + f for f in shard_files(k, 3, (0, 1), "all")], 3))
    J.append(Job("ladder-subset", ["run_ladder.py", "--seeds", "0,1", "--shard", "0/1", "--subset"], ["ladder/" + f for f in shard_files(0, 1, (0, 1), "subset")], 1))
    J.append(Job("separability", ["probe_separability.py"], ["separability.json"], 2))
    J.append(Job("routing-rule", ["probe_routing_rule.py"], ["routing_rule.json"], 2))
    for s in (0, 1):
        J.append(Job(f"seq-default-s{s}-instrumented", ["run_seqprobe.py", "--variant", "default", "--seed", s, "--out", "seqprobe2"], [f"seqprobe2/default_s{s}.json"], 3))
        J.append(Job(f"seq-forced-s{s}-instrumented", ["run_seqprobe.py", "--variant", "forced_spawn", "--seed", s, "--out", "seqprobe2"], [f"seqprobe2/forced_spawn_s{s}.json"], 3))
    J.append(Job("seq-probe600-s0", ["run_seqprobe.py", "--variant", "probe600", "--seed", 0, "--out", "seqprobe"], ["seqprobe/probe600_s0.json"], 3))
    for cfg, s in (("shipped", 0), ("shipped", 1), ("tuned", 0)):
        J.append(Job(f"growth-{cfg}-s{s}", ["probe_growth.py", "--config", cfg, "--seed", s], [f"growth/{cfg}_s{s}.json"], 3))
    for s in (0, 1):
        J.append(Job(f"grow-from-small-s{s}", ["probe_grow_from_small.py", "--seed", s], [f"grow_from_small/s{s}.json"], 3))
        J.append(Job(f"widen-mid-128-s{s}", ["probe_widen_midstream.py", "--seed", s], [f"widen_midstream/s{s}.json"], 3))
        J.append(Job(f"widen-mid-16-s{s}", ["probe_widen_midstream.py", "--seed", s, "--start", 16], [f"widen_midstream/h16_s{s}.json"], 3))
        J.append(Job(f"horizon-s{s}", ["probe_horizon.py", "--seed", s], [f"horizon/s{s}.json"], 3))
        for g, tag in ((0.005, "0p005"), (0.0, "0p0")):
            J.append(Job(f"grow-threshold-{tag}-s{s}", ["probe_grow_threshold.py", "--seed", s, "--min-gain", g], [f"grow_threshold/h16_g{tag}_s{s}.json"], 3))
    return J


def running_command_lines():
    ps = "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | ForEach-Object { $_.CommandLine }"
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return []
    return [ln.replace('"', "") for ln in out.splitlines() if ln.strip()]


def is_running(job, lines):
    m = job.match.replace('"', "")
    return any(m in ln for ln in lines)


def states(jobs):
    lines = running_command_lines()
    st = {}
    for j in jobs:
        st[j.name] = "DONE" if j.done() else ("RUNNING" if is_running(j, lines) else "PENDING")
    return st


def cmd_status():
    jobs = build_jobs()
    st = states(jobs)
    print(f"{'job':<34s} {'state':<8s} note")
    for j in jobs:
        if st[j.name] != "DONE" or "-v" in sys.argv:
            extra = ("  [checkpoint present: will resume]" if (st[j.name] != "DONE" and j.checkpoint()) else "")
            print(f"{j.name:<34s} {st[j.name]:<8s} {j.note}{extra}")
    counts = {k: sum(1 for v in st.values() if v == k) for k in ("DONE", "RUNNING", "PENDING")}
    print(f"\n{counts['DONE']} done, {counts['RUNNING']} running, {counts['PENDING']} pending (of {len(jobs)} jobs)")
    return st


def launch(job):
    LOGS.mkdir(exist_ok=True)
    tag = job.name
    out = open(LOGS / f"q_{tag}.log", "ab")
    err = open(LOGS / f"q_{tag}.err", "ab")
    # hidden console, own process group, no handle inheritance: survives the shell that started it (a Start-Process -WindowStyle Hidden equivalent)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    p = subprocess.Popen([PY, str(HERE / job.args[0])] + [str(a) for a in job.args[1:]], cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=out, stderr=err, creationflags=flags, close_fds=True)
    print(f"{time.strftime('%H:%M:%S')} launched {job.name} (pid {p.pid})", flush=True)


def scheduler_alive() -> bool:
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text().split()[0])
    except Exception:
        return False
    lines = running_command_lines()
    return any("jobqueue.py" in ln and "run" in ln and str(pid) != str(os.getpid()) for ln in lines) and pid != os.getpid()


def cmd_run(wait: bool):
    if wait and scheduler_alive():
        print("another scheduler is already running (queue.lock); nothing to do")
        return
    if wait:
        LOCK.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    jobs = build_jobs()
    by_name = {j.name: j for j in jobs}
    try:
        while True:
            st = states(jobs)
            used = sum(j.threads for j in jobs if st[j.name] == "RUNNING")
            for j in jobs:
                if st[j.name] != "PENDING":
                    continue
                if any(st.get(d) != "DONE" for d in j.deps):
                    continue
                if used + j.threads > BUDGET:
                    continue
                launch(j)
                used += j.threads
                st[j.name] = "RUNNING"
            if not wait:
                break
            if all(v == "DONE" for v in st.values()):
                print(f"{time.strftime('%H:%M:%S')} everything is done; rebuilding the report", flush=True)
                subprocess.run([PY, str(HERE / "report_diagnosis.py")], cwd=str(HERE))
                break
            if not any(v == "RUNNING" for v in st.values()) and not any(v == "PENDING" and all(st.get(d) == "DONE" for d in by_name[n].deps) for n, v in st.items() for _ in [0]):
                print("nothing running and nothing launchable (a dependency never finished?)", flush=True)
                break
            time.sleep(30)
    finally:
        if wait and LOCK.exists():
            try:
                LOCK.unlink()
            except Exception:
                pass


def cmd_detach():
    LOGS.mkdir(exist_ok=True)
    out = open(LOGS / "queue_scheduler.log", "ab")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    p = subprocess.Popen([PY, str(HERE / "jobqueue.py"), "run", "--wait"], cwd=str(HERE), stdin=subprocess.DEVNULL, stdout=out, stderr=out, creationflags=flags, close_fds=True)
    print(f"scheduler started in the background (pid {p.pid}); log: logs/queue_scheduler.log; check with: python jobqueue.py status")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("status", "run"))
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--detach", action="store_true")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    if a.cmd == "status":
        cmd_status()
    elif a.detach:
        cmd_detach()
    else:
        cmd_run(a.wait)
