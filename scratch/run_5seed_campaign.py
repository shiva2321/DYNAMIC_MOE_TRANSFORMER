"""
Automated 5-Seed Continual Learning Campaign Runner.
Executes the remaining seeds in sequence by directly invoking the exact canonical
`exp_10domain_continual_learning.py` script via subprocess.
Guarantees 100% config parity, crash resume-ability, and exact logging.
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments")
RUNNER_SCRIPT = os.path.join(PROJECT_ROOT, "exp_10domain_continual_learning.py")

SEEDS = [1337, 42, 7, 123, 999]
MODELS = [
    "static_moe",                  # Matched-compute 16-expert baseline
    "hyperspace_budgeted_spawn"    # Dynamic neurogenesis
]

def check_completed(model: str, seed: int) -> bool:
    suffix = f"_seed_{seed}" if seed != 1337 else ""
    res_path = os.path.join(EXP_DIR, f"continual_10domain_{model}{suffix}_results.json")
    if os.path.exists(res_path):
        try:
            with open(res_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "acc_matrix" in d and len(d["acc_matrix"]) == 10:
                    return True
        except Exception:
            return False
    return False

def main():
    print("=" * 100)
    print("    UNIVERSAL SUBSTRAIT: 5-SEED CONTINUAL LEARNING CAMPAIGN QUEUE (N=5 per model)")
    print(f"    Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"    Runner Target: {RUNNER_SCRIPT}")
    print("=" * 100)

    queue = []
    for model in MODELS:
        for seed in SEEDS:
            queue.append((model, seed))

    total_tasks = len(queue)
    print(f"Total Campaign Queue: {total_tasks} tasks ({len(MODELS)} models x {len(SEEDS)} seeds)\n")

    t_campaign_start = time.perf_counter()

    for idx, (model, seed) in enumerate(queue, 1):
        suffix = f"_seed_{seed}" if seed != 1337 else ""
        print("-" * 100)
        print(f"[{idx}/{total_tasks}] Checking Task: Model = {model.upper()} | Seed = {seed}")
        
        if check_completed(model, seed):
            print(f"  --> [ALREADY COMPLETED] Found valid result file: continual_10domain_{model}{suffix}_results.json. Skipping.\n")
            continue

        print(f"  --> [LAUNCHING] Starting clean run via canonical exp_10domain_continual_learning.py")
        cmd = [
            sys.executable, "-u", RUNNER_SCRIPT,
            "--model", model,
            "--seed", str(seed),
            "--steps", "360",
            "--ratio", "0.20"
        ]

        t_start = time.perf_counter()
        proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

        for line in iter(proc.stdout.readline, ''):
            print(line, end='', flush=True)

        proc.stdout.close()
        return_code = proc.wait()
        t_elapsed = time.perf_counter() - t_start

        if return_code != 0:
            print(f"\n[ERROR] Run failed with return code {return_code}! Aborting queue.")
            sys.exit(return_code)

        print(f"\n  --> [COMPLETED] Model {model} (Seed {seed}) finished in {t_elapsed/60:.2f} minutes ({t_elapsed:.1f}s)\n")

    t_campaign_total = time.perf_counter() - t_campaign_start
    print("=" * 100)
    print(f"ALL 5-SEED CAMPAIGN RUNS COMPLETED in {t_campaign_total/3600:.2f} hours ({t_campaign_total/60:.2f} mins).")
    print("Compiling Master 5-Seed Statistical Report...")
    print("=" * 100)

    # Automatically run statistical aggregator
    aggregator = os.path.join(PROJECT_ROOT, "scratch", "compile_10domain_final_report.py")
    subprocess.run([sys.executable, "-u", aggregator], cwd=PROJECT_ROOT)

if __name__ == "__main__":
    main()
