"""Edge probe for A3 on B1 (N = 30): one more doubling of the hidden width (8192 -> 16384) at the chosen 20 steps, development seeds 800-804 (E-072).

Reads and writes the shared trial cache of results/m1/tuning (a finished seed is read back, not rerun) and does not touch B1.json.
The first run wrote results/m1/diag/a3_edge_probe_B1.json (kept as it came out); this script writes a3_edge_probe_B1_rerun.json from the cache so the two can be compared.
"""

import json
from pathlib import Path

from costlab.env import fixed_threads
from m1.tune import TuningDriver

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "m1" / "diag" / "a3_edge_probe_B1_rerun.json"


def main() -> None:
    with fixed_threads(n=2, cpus=[0, 2]):
        driver = TuningDriver(bed_name="B1", n_regimes=30, out_dir=ROOT / "results" / "m1" / "tuning", seeds=[800, 801, 802, 803, 804])
        res = driver.evaluate_candidate("A3", {"hidden": 16384, "inner_steps": 20}, ref_pool="A1")
    chosen_8192 = {"hidden": 8192, "inner_steps": 20}
    ref = driver.evaluate_candidate("A3", chosen_8192, ref_pool="A1")                # read back from the cache
    summary = {"probe_16384": res, "reference_8192": ref, "gain_points": (res["mean_accuracy"] - ref["mean_accuracy"]) * 100.0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"16384: {res['mean_accuracy'] * 100:.2f} %  8192: {ref['mean_accuracy'] * 100:.2f} %  gain {summary['gain_points']:.2f} points")


if __name__ == "__main__":
    main()
