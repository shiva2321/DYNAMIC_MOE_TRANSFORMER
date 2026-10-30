"""Milestones of the third mini-round and the confirmation chain: prints a line when a log reaches its expected number of result lines, and on any crash. Exits when all are done."""
import re
import sys
import time
from pathlib import Path

D = Path(__file__).resolve().parent
EXPECT = {
    "p4_small_tune_mix.log": (r"chosen small-start", 1),
    "p4_small_tune_drift.log": (r"chosen small-start", 1),
    "p4_small_tune_seq.log": (r"chosen small-start", 1),
    "p4_small_eval.log": (r"^seed", 9),
    "p4_refusal_confirm.log": (r"^(UG0|UPLUS) seed", 4),
}
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
    if all(("done", n) in seen for n in EXPECT):
        once("all", "third mini-round and confirmation chain finished")
        sys.exit(0)
    time.sleep(30)
