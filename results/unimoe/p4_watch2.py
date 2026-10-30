"""Milestones of the second round: prints a line when a log reaches its expected number of result lines, and on any crash. Exits when everything is done."""
import re
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
EXPECT = {"p4_main_confirm.log": (r"^seed", 36), "p4_refusal_density.log": (r"^(UG0|UPLUS) seed", 6), "p4_small_start.log": (r"^seed", 18),
          "p4_refusal_confirm.log": (r"^(UG0|UPLUS) seed", 4)}
EXTRA = ["run_v2_confirm.log"]
seen = set()


def text(name):
    p = D / name
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def once(key, message):
    if key not in seen:
        seen.add(key)
        print(message, flush=True)


while True:
    for name in list(EXPECT) + EXTRA:
        t = text(name)
        if "Traceback" in t:
            once(("crash", name), f"CRASH in {name}: " + " | ".join(t[t.index("Traceback"):].splitlines()[:10]))
    for name, (pat, n) in EXPECT.items():
        k = len(re.findall(pat, text(name), re.M))
        if k >= n:
            once(("done", name), f"{name}: finished ({k} result lines)")
    if text("run_v2_confirm.log") and len(re.findall(r"^seed|final", text("run_v2_confirm.log"), re.M)) and ("done", "p4_refusal_confirm.log") in seen:
        once("all", "second round finished")
        sys.exit(0)
    time.sleep(30)
