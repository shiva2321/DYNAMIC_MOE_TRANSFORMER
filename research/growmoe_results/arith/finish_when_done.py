"""Wait until every arithmetic queue has finished, then regenerate RESULTS_ARITH.md (and the curve report) and leave a marker file. Run in the background."""
import subprocess
import sys
import time
from pathlib import Path

import psutil

HERE = Path(__file__).resolve().parent


def busy() -> bool:
    for p in psutil.process_iter(["cmdline"]):
        cl = " ".join(p.info["cmdline"] or [])
        if "run_arith.py" in cl or "launch_main" in cl or "launch_curve" in cl:
            return True
    return False


while busy():
    time.sleep(60)
out = subprocess.run([sys.executable, str(HERE / "report_arith.py"), "--dir", "results"], capture_output=True, text=True, cwd=str(HERE))
(HERE / "logs" / "final_report.txt").write_text(out.stdout + out.stderr[-2000:], encoding="utf-8")
(HERE / "logs" / "ALL_DONE.txt").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8")
print("done", flush=True)
