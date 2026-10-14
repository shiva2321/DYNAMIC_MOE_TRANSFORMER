"""CVA-1B: Spawn rules evaluation and calibration.

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Compares six candidate spawn rules:
  R0 today: evidence > log(1/0.01) (approx 4.605)
  R1 size-aware: evidence / sqrt(m) > z1
  R2 per-point: evidence / m > z2
  R3 exact permutation test: perm_reject(alpha=0.001, stat="meandist")
  R4 both: R0 AND R3
  R5 exact test at alpha=0.01 (stat="meandist")

Seeds:
  Calibration: 0-9 (thresholds z1 and z2 fixed on them)
  Evaluation: 10-19 (all reported numbers come from them)
"""

import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from growmoe.settest import generator, perm_reject
from growmoe.signature import VmfSignature
from growmoe.vmf import log_vmf_norm
from research.growmoe_results.cva.cva1_gate_evidence import centres, draw

torch.set_num_threads(3)
D = 128
CAL_SEEDS = list(range(10))        # 0-9
EVAL_SEEDS = list(range(10, 20))    # 10-19
OUT_DIR = Path("research/growmoe_results/cva")


def compute_evidence(
    cluster: torch.Tensor,
    live_sigs: List[VmfSignature],
    seed: int,
    rep: int,
) -> float:
    """Evidence formula reproduced line for line from growmoe/pools/phylo.py lines 682-701."""
    sig_c = VmfSignature(D, max_protos=256, seed=seed * 1000 + rep)
    cx_sub = cluster[:256]
    sig_c.update(cx_sub)
    sig_c.fit()

    if hasattr(sig_c, "kappa") and getattr(sig_c, "kappa") != 1.0:
        logf_c_loo = sig_c.cal_scores + log_vmf_norm(D, sig_c.kappa)
    else:
        logf_c_loo = sig_c.cal_scores
    protos_c = sig_c.protos

    if len(live_sigs) == 1:
        max_live_logf = live_sigs[0].logpdf(protos_c)
    else:
        live_logfs = torch.stack([s.logpdf(protos_c) for s in live_sigs], dim=1)
        max_live_logf = live_logfs.max(dim=1).values

    return float((logf_c_loo - max_live_logf).sum().item())


def calibrate_thresholds() -> Tuple[float, float, List[float], List[float]]:
    """Calibrate z1 and z2 on seeds 0-9 over all (n_fit >= 256, m) cells under negative control."""
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting CVA-1B Calibration (seeds 0-9)")
    n_fit_vals = [256, 1024]
    m_vals = [32, 64, 128, 256]

    cal_sqrt = []
    cal_m = []

    for n_fit in n_fit_vals:
        for m in m_vals:
            for seed in CAL_SEEDS:
                rng = np.random.default_rng(seed)
                a, _, _ = centres(0.0, seed)
                sig_A = VmfSignature(D, max_protos=256, seed=seed)
                sig_A.update(draw(a, n_fit, rng))
                sig_A.fit()

                for rep in range(10):
                    cluster = draw(a, m, rng)
                    ev = compute_evidence(cluster, [sig_A], seed, rep)
                    cal_sqrt.append(ev / math.sqrt(m))
                    cal_m.append(ev / m)

    z1 = float(np.percentile(cal_sqrt, 99))
    z2 = float(np.percentile(cal_m, 99))
    print(
        f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): "
        f"Calibration complete: z1 = {z1:.4f}, z2 = {z2:.4f} (from {len(cal_sqrt)} draws)"
    )
    return z1, z2, cal_sqrt, cal_m


def evaluate_rules(
    evidence: float,
    m: int,
    live_sigs: List[VmfSignature],
    cluster: torch.Tensor,
    seed: int,
    rep: int,
    z1: float,
    z2: float,
    timings: Dict[str, List[float]],
) -> Dict[str, bool]:
    """Evaluate candidate spawn rules R0-R5 and profile runtime."""
    threshold_r0 = math.log(1.0 / 0.01)  # 4.60517

    # R0
    t0 = time.perf_counter()
    spawn_r0 = evidence > threshold_r0
    timings["R0"].append((time.perf_counter() - t0) * 1000.0)

    # R1
    t0 = time.perf_counter()
    spawn_r1 = (evidence / math.sqrt(m)) > z1
    timings["R1"].append((time.perf_counter() - t0) * 1000.0)

    # R2
    t0 = time.perf_counter()
    spawn_r2 = (evidence / m) > z2
    timings["R2"].append((time.perf_counter() - t0) * 1000.0)

    # R3 (exact permutation test at alpha=0.001)
    t0 = time.perf_counter()
    gen = generator(seed, rep)
    # Reject for EVERY live expert
    rej_r3 = True
    for s in live_sigs:
        rejected, _, _ = perm_reject(
            ref=s.protos, batch=cluster, alpha=0.001, n_perm=999, gen=gen, stat="meandist"
        )
        if not rejected:
            rej_r3 = False
            break
    spawn_r3 = rej_r3
    timings["R3"].append((time.perf_counter() - t0) * 1000.0)

    # R4 (R0 AND R3)
    t0 = time.perf_counter()
    spawn_r4 = spawn_r0 and spawn_r3
    timings["R4"].append((time.perf_counter() - t0) * 1000.0)

    # R5 (exact permutation test at alpha=0.01)
    t0 = time.perf_counter()
    gen_r5 = generator(seed, rep)
    rej_r5 = True
    for s in live_sigs:
        rejected, _, _ = perm_reject(
            ref=s.protos, batch=cluster, alpha=0.01, n_perm=999, gen=gen_r5, stat="meandist"
        )
        if not rejected:
            rej_r5 = False
            break
    spawn_r5 = rej_r5
    timings["R5"].append((time.perf_counter() - t0) * 1000.0)

    return {
        "R0": spawn_r0,
        "R1": spawn_r1,
        "R2": spawn_r2,
        "R3": spawn_r3,
        "R4": spawn_r4,
        "R5": spawn_r5,
    }


def run_evaluation(z1: float, z2: float):
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting Evaluation (seeds 10-19)")
    rule_names = ["R0", "R1", "R2", "R3", "R4", "R5"]
    timings: Dict[str, List[float]] = {r: [] for r in rule_names}
    evidence_times: List[float] = []

    # 1. Negative Control (1 Live Expert)
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Running Negative Control (1 live expert)")
    neg_1exp_results = []
    n_fit_vals = [64, 256, 1024]
    m_vals = [32, 64, 128, 256]

    for n_fit in n_fit_vals:
        for m in m_vals:
            counts = {r: 0 for r in rule_names}
            for seed in EVAL_SEEDS:
                rng = np.random.default_rng(seed)
                a, _, _ = centres(0.0, seed)
                sig_A = VmfSignature(D, max_protos=256, seed=seed)
                sig_A.update(draw(a, n_fit, rng))
                sig_A.fit()

                for rep in range(10):
                    cluster = draw(a, m, rng)
                    t0 = time.perf_counter()
                    ev = compute_evidence(cluster, [sig_A], seed, rep)
                    evidence_times.append((time.perf_counter() - t0) * 1000.0)

                    decisions = evaluate_rules(ev, m, [sig_A], cluster, seed, rep, z1, z2, timings)
                    for r, val in decisions.items():
                        if val:
                            counts[r] += 1

            rates = {r: counts[r] / 100.0 for r in rule_names}
            entry = {
                "n_fit": n_fit,
                "m": m,
                **{f"spawn_{r}": rates[r] for r in rule_names},
            }
            neg_1exp_results.append(entry)
            print(
                f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): NEG 1-EXP n_fit={n_fit:4d} m={m:3d} -> "
                + " ".join(f"{r}={rates[r]:.2f}" for r in rule_names)
            )

    # 2. Positive Control (1 Live Expert, n_fit = 256)
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Running Positive Control (1 live expert, n_fit=256)")
    pos_results = []
    rho_vals = [0.0, 0.3, 0.6, 0.85, 0.95]
    pos_m_vals = [32, 64, 128]

    for rho in rho_vals:
        for m in pos_m_vals:
            counts = {r: 0 for r in rule_names}
            for seed in EVAL_SEEDS:
                rng = np.random.default_rng(seed)
                a, b_centre, _ = centres(rho, seed)
                sig_A = VmfSignature(D, max_protos=256, seed=seed)
                sig_A.update(draw(a, 256, rng))
                sig_A.fit()

                for rep in range(10):
                    cluster = draw(b_centre, m, rng)
                    t0 = time.perf_counter()
                    ev = compute_evidence(cluster, [sig_A], seed, rep)
                    evidence_times.append((time.perf_counter() - t0) * 1000.0)

                    decisions = evaluate_rules(ev, m, [sig_A], cluster, seed, rep, z1, z2, timings)
                    for r, val in decisions.items():
                        if val:
                            counts[r] += 1

            rates = {r: counts[r] / 100.0 for r in rule_names}
            entry = {
                "rho": rho,
                "m": m,
                "n_fit": 256,
                **{f"spawn_{r}": rates[r] for r in rule_names},
            }
            pos_results.append(entry)
            print(
                f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): POS rho={rho:.2f} m={m:3d} -> "
                + " ".join(f"{r}={rates[r]:.2f}" for r in rule_names)
            )

    # 3. Negative Control with Two Live Experts (A and C at rho = 0)
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Running Negative Control (2 live experts: A and C)")
    neg_2exp_results = []
    for n_fit in n_fit_vals:
        for m in m_vals:
            counts = {r: 0 for r in rule_names}
            for seed in EVAL_SEEDS:
                rng = np.random.default_rng(seed)
                a, _, u = centres(0.0, seed)
                c_centre = u  # rho = 0 with A

                sig_A = VmfSignature(D, max_protos=256, seed=seed)
                sig_A.update(draw(a, n_fit, rng))
                sig_A.fit()

                sig_C = VmfSignature(D, max_protos=256, seed=seed + 100)
                sig_C.update(draw(c_centre, n_fit, rng))
                sig_C.fit()

                for rep in range(10):
                    cluster = draw(a, m, rng)
                    t0 = time.perf_counter()
                    ev = compute_evidence(cluster, [sig_A, sig_C], seed, rep)
                    evidence_times.append((time.perf_counter() - t0) * 1000.0)

                    decisions = evaluate_rules(ev, m, [sig_A, sig_C], cluster, seed, rep, z1, z2, timings)
                    for r, val in decisions.items():
                        if val:
                            counts[r] += 1

            rates = {r: counts[r] / 100.0 for r in rule_names}
            entry = {
                "n_fit": n_fit,
                "m": m,
                **{f"spawn_{r}": rates[r] for r in rule_names},
            }
            neg_2exp_results.append(entry)
            print(
                f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): NEG 2-EXP n_fit={n_fit:4d} m={m:3d} -> "
                + " ".join(f"{r}={rates[r]:.2f}" for r in rule_names)
            )

    # Compute runtime stats (mean ms per decision)
    mean_ev_time = float(np.mean(evidence_times))
    runtime_per_decision = {}
    for r in rule_names:
        # Full decision time including prerequisite computations
        if r in ("R0", "R1", "R2"):
            total_ms = mean_ev_time + float(np.mean(timings[r]))
        elif r == "R4":
            total_ms = mean_ev_time + float(np.mean(timings["R0"])) + float(np.mean(timings["R3"]))
        else:  # R3, R5
            total_ms = float(np.mean(timings[r]))
        runtime_per_decision[r] = {
            "rule_step_ms": float(np.mean(timings[r])),
            "total_decision_ms": total_ms,
        }

    return neg_1exp_results, pos_results, neg_2exp_results, runtime_per_decision


def determine_verdicts(
    neg_results: List[dict],
    pos_results: List[dict],
    rule_names: List[str],
) -> Dict[str, dict]:
    """Check predeclared criteria for each rule:
    1. Negative control: <= 0.01 in every cell with n_fit >= 256.
    2. Positive control: >= 0.99 for rho <= 0.6 and m >= 64; >= 0.9 for rho = 0.85 and m = 128.
    PASS only if both criteria hold in all their cells; otherwise list failing cells.
    """
    verdicts = {}
    for r in rule_names:
        failing_cells = []

        # Check negative control cells with n_fit >= 256
        for entry in neg_results:
            if entry["n_fit"] >= 256:
                val = entry[f"spawn_{r}"]
                if val > 0.01:
                    failing_cells.append(f"NEG(n_fit={entry['n_fit']},m={entry['m']},rate={val:.2f}>0.01)")

        # Check positive control
        for entry in pos_results:
            rho = entry["rho"]
            m = entry["m"]
            val = entry[f"spawn_{r}"]
            if rho <= 0.6 and m >= 64:
                if val < 0.99:
                    failing_cells.append(f"POS(rho={rho},m={m},rate={val:.2f}<0.99)")
            elif abs(rho - 0.85) < 1e-4 and m == 128:
                if val < 0.90:
                    failing_cells.append(f"POS(rho=0.85,m=128,rate={val:.2f}<0.90)")

        verdict = "PASS" if len(failing_cells) == 0 else f"FAIL ({', '.join(failing_cells)})"
        verdicts[r] = {
            "verdict": "PASS" if len(failing_cells) == 0 else "FAIL",
            "failing_cells": failing_cells,
        }

    return verdicts


def write_outputs(
    z1: float,
    z2: float,
    neg_1exp: List[dict],
    pos: List[dict],
    neg_2exp: List[dict],
    runtimes: Dict[str, dict],
    verdicts: Dict[str, dict],
):
    rules = ["R0", "R1", "R2", "R3", "R4", "R5"]
    rule_descriptions = {
        "R0": "today: evidence > log(1/0.01)",
        "R1": f"size-aware: evidence / sqrt(m) > {z1:.4f}",
        "R2": f"per-point: evidence / m > {z2:.4f}",
        "R3": "exact permutation test: perm_reject(alpha=0.001)",
        "R4": "both: R0 AND R3",
        "R5": "exact permutation test: perm_reject(alpha=0.01)",
    }

    out_data = {
        "metadata": {
            "title": "CVA-1B: Spawn Rules Verification",
            "d": D,
            "cal_seeds": CAL_SEEDS,
            "eval_seeds": EVAL_SEEDS,
            "z1": z1,
            "z2": z2,
            "rules": rule_descriptions,
        },
        "negative_control_1exp": neg_1exp,
        "positive_control_1exp": pos,
        "negative_control_2exp": neg_2exp,
        "runtimes": runtimes,
        "verdicts": verdicts,
    }

    with open(OUT_DIR / "cva1b.json", "w") as f:
        json.dump(out_data, f, indent=2)

    md_lines = [
        "# CVA-1B Results: Evaluation of Spawn Rules",
        "",
        "VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dimension: d = {D} | Calibration seeds: 0-9 | Evaluation seeds: 10-19",
        f"Calibrated thresholds (seeds 0-9, pooled n_fit >= 256, m): z1 = {z1:.4f}, z2 = {z2:.4f}",
        "",
        "---",
        "",
        "## Summary & Verdicts",
        "",
        "| Rule | Description | Decision Time (ms) | Verdict | Failing Cells |",
        "|:---:|:---|:---:|:---:|:---|",
    ]

    for r in rules:
        v_info = verdicts[r]
        t_ms = runtimes[r]["total_decision_ms"]
        fails = "; ".join(v_info["failing_cells"]) if v_info["failing_cells"] else "None"
        md_lines.append(
            f"| **{r}** | {rule_descriptions[r]} | {t_ms:.2f} ms | **{v_info['verdict']}** | {fails} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Negative Control (1 Live Expert)",
        "",
        "**Criterion**: Share spawned <= 0.01 in every cell with n_fit >= 256 (seeds 10-19, 100 draws/cell).",
        "",
        "| n_fit | m | R0 (today) | R1 (sqrt) | R2 (/m) | R3 (perm 0.001) | R4 (both) | R5 (perm 0.01) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for row in neg_1exp:
        md_lines.append(
            f"| {row['n_fit']} | {row['m']} | {row['spawn_R0']:.2f} | {row['spawn_R1']:.2f} | "
            f"{row['spawn_R2']:.2f} | {row['spawn_R3']:.2f} | {row['spawn_R4']:.2f} | {row['spawn_R5']:.2f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Positive Control (1 Live Expert, n_fit = 256)",
        "",
        "**Criteria**: >= 0.99 for rho <= 0.6 and m >= 64; >= 0.90 for rho = 0.85 and m = 128; rho = 0.95 reported only.",
        "",
        "| rho | m | R0 (today) | R1 (sqrt) | R2 (/m) | R3 (perm 0.001) | R4 (both) | R5 (perm 0.01) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for row in pos:
        md_lines.append(
            f"| {row['rho']:.2f} | {row['m']} | {row['spawn_R0']:.2f} | {row['spawn_R1']:.2f} | "
            f"{row['spawn_R2']:.2f} | {row['spawn_R3']:.2f} | {row['spawn_R4']:.2f} | {row['spawn_R5']:.2f} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Negative Control (2 Live Experts: A and C at rho = 0)",
        "",
        "**Criterion**: Rule must hold against every live expert (seeds 10-19, 100 draws/cell).",
        "",
        "| n_fit | m | R0 (today) | R1 (sqrt) | R2 (/m) | R3 (perm 0.001) | R4 (both) | R5 (perm 0.01) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for row in neg_2exp:
        md_lines.append(
            f"| {row['n_fit']} | {row['m']} | {row['spawn_R0']:.2f} | {row['spawn_R1']:.2f} | "
            f"{row['spawn_R2']:.2f} | {row['spawn_R3']:.2f} | {row['spawn_R4']:.2f} | {row['spawn_R5']:.2f} |"
        )

    with open(OUT_DIR / "CVA1B_RESULTS.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(
        f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Saved cva1b.json and CVA1B_RESULTS.md to {OUT_DIR}"
    )


def main():
    t_start = time.time()
    z1, z2, cal_sqrt, cal_m = calibrate_thresholds()
    neg_1exp, pos, neg_2exp, runtimes = run_evaluation(z1, z2)
    verdicts = determine_verdicts(neg_1exp, pos, ["R0", "R1", "R2", "R3", "R4", "R5"])
    write_outputs(z1, z2, neg_1exp, pos, neg_2exp, runtimes, verdicts)
    print(
        f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Total runtime = {time.time() - t_start:.2f}s"
    )


if __name__ == "__main__":
    main()
