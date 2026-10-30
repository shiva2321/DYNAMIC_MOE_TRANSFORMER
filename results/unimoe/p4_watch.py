"""Prints one line per P4 milestone (a tuning result, a finished chain step, a crash); exits when the capacity evaluation has finished."""
import re
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
LOGS = ["p4_capacity_widths.log", "p4_capacity_tune.log", "p4_capacity_eval.log", "p4_oracle_lda.log", "p4_main_tune_seq.log", "p4_main_tune_mix.log", "p4_main_tune_drift.log", "p4_main_run.log", "p4_refusal.log"]
seen = set()


def text(name):
    p = D / name
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def once(key, message):
    if key not in seen:
        seen.add(key)
        print(message, flush=True)


while True:
    for name in LOGS:
        t = text(name)
        if "Traceback" in t:
            tail = t[t.index("Traceback"):].splitlines()[:10]
            once(("crash", name), f"CRASH in {name}: " + " | ".join(tail))
        for m in re.finditer(r"^(seq|mix|drift) chosen .*$|^chosen threshold .*$", t, re.M):
            once(("chosen", name, m.group(0)), f"{name}: {m.group(0)}")
    n_w = len(re.findall(r"^seed", text("p4_capacity_widths.log"), re.M))
    if n_w == 24:
        once("widths", "capacity-widths finished (24 cells)")
    n_e = len(re.findall(r"^seed", text("p4_capacity_eval.log"), re.M))
    if n_e == 24:
        once("eval", "capacity-eval finished (24 cells)")
        sys.exit(0)
    time.sleep(20)
