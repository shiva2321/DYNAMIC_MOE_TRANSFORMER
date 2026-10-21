"""CVA-1: Gate and evidence verification for PhyloMoE.

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Lines copied from growmoe/pools/phylo.py (lines 682-701):
-----------------------------------------------------------------------------
sig_c = self.sig_cls(
    d=self.cfg.d,
    max_protos=256,
    seed=self.cfg.seed * 1000 + len(self.experts),
)
sig_c.update(cx_sub)
sig_c.fit()

if hasattr(sig_c, "kappa") and getattr(sig_c, "kappa") != 1.0:
    logf_c_loo = sig_c.cal_scores + log_vmf_norm(self.cfg.d, sig_c.kappa)
else:
    logf_c_loo = sig_c.cal_scores
protos_c = sig_c.protos
live_logfs = torch.stack([
    rec.meta["sig"].logpdf(protos_c) for rec in self.experts.values()
], dim=1)
max_live_logf = live_logfs.max(dim=1).values

evidence = float((logf_c_loo - max_live_logf).sum().item())
threshold = math.log(1.0 / self.cfg.alpha_spawn)
-----------------------------------------------------------------------------
"""

import json
import math
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from sklearn.metrics import roc_auc_score
import torch

from growmoe.config import PhyloCfg
from growmoe.experts import MLPExpert
from growmoe.pools.phylo import PhyloMoE
from growmoe.signature import VmfSignature
from growmoe.vmf import log_vmf_norm

torch.set_num_threads(3)
D = 128
SEEDS = list(range(20))
OUT_DIR = Path("research/growmoe_results/cva")


def centres(rho: float, seed: int):
    """Unit vectors a and b with cosine rho (b = rho*a + sqrt(1-rho^2)*u, u a unit vector orthogonal to a)."""
    rng = np.random.default_rng([seed, 0xC])
    z1 = rng.standard_normal(D)
    a = z1 / np.linalg.norm(z1)
    z2 = rng.standard_normal(D)
    u_raw = z2 - np.dot(z2, a) * a
    u = u_raw / np.linalg.norm(u_raw)
    b = rho * a + math.sqrt(max(0.0, 1.0 - rho**2)) * u
    b = b / np.linalg.norm(b)
    return a, b, u


def draw(centre: np.ndarray, n: int, rng: np.random.Generator) -> torch.Tensor:
    """unit(8.0 * centre + standard_normal(n, d)) (how growmoe.streams_m1.function_regimes builds a regime)."""
    z = rng.standard_normal((n, D))
    raw = 8.0 * centre + z
    norm = np.linalg.norm(raw, axis=1, keepdims=True)
    unit = raw / norm
    return torch.from_numpy(unit.astype(np.float32))


def run_experiment_1_1():
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting Experiment 1.1")
    n_fit_vals = [32, 64, 128, 256, 512, 1024]
    q_vals = [0.02, 0.05]
    results = []

    for n_fit in n_fit_vals:
        for q in q_vals:
            flag_inputs_seeds = []
            flag_batches_seeds = []
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                a, _, _ = centres(0.0, seed)
                x_fit = draw(a, n_fit, rng)
                sig = VmfSignature(D, max_protos=256, seed=seed)
                sig.update(x_fit)
                sig.fit()

                pool = PhyloMoE(PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", d=D, hidden=16, n_classes=8, seed=seed, q=q, route_rule="accept_then_logf"))
                pool.add_expert(MLPExpert(D, 16, 1, 8, seed=seed), parent=None, state="active", meta={"sig": sig})

                x_fresh = draw(a, 5000, rng)
                tot_in, flg_in, tot_b, flg_b = 0, 0, 0, 0
                for start in range(0, 5000, 64):
                    batch = x_fresh[start : start + 64]
                    pred = pool.predict(batch)
                    unk = pred.unknown
                    flg_in += int(unk.sum().item())
                    tot_in += batch.shape[0]
                    tot_b += 1
                    if unk.any():
                        flg_b += 1

                flag_inputs_seeds.append(flg_in / tot_in)
                flag_batches_seeds.append(flg_b / tot_b)

            mean_in = float(np.mean(flag_inputs_seeds))
            std_in = float(np.std(flag_inputs_seeds))
            mean_b = float(np.mean(flag_batches_seeds))
            std_b = float(np.std(flag_batches_seeds))

            # Criteria: for n_fit >= 128, mean (a) <= 1.5*q and mean (b) <= 1.5*q.
            # (For n_fit = 32 the smallest attainable p-value 1/(n+1) is above q = 0.02, so "never flags" is expected)
            crit_val = 1.5 * q
            if n_fit < 128:
                verdict = "NOTE"
            elif mean_in <= crit_val and mean_b <= crit_val:
                verdict = "PASS"
            else:
                verdict = "DEFECT"

            entry = {
                "n_fit": n_fit,
                "q": q,
                "mean_inputs_flagged": mean_in,
                "std_inputs_flagged": std_in,
                "mean_batches_flagged": mean_b,
                "std_batches_flagged": std_b,
                "threshold_1_5_q": crit_val,
                "verdict": verdict,
            }
            results.append(entry)
            print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): 1.1 n_fit={n_fit:4d} q={q:.2f} -> inputs={mean_in:.5f}+-{std_in:.5f}, batches={mean_b:.5f}+-{std_b:.5f} [{verdict}]")

    return results


def run_experiment_1_2():
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting Experiment 1.2")
    n_fit = 256
    q = 0.02
    rho_vals = [0.0, 0.3, 0.6, 0.85, 0.95]
    results = []

    for rho in rho_vals:
        flagged_seeds = []
        auroc_seeds = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            a, b_centre, _ = centres(rho, seed)
            x_fit = draw(a, n_fit, rng)
            sig = VmfSignature(D, max_protos=256, seed=seed)
            sig.update(x_fit)
            sig.fit()

            pool = PhyloMoE(PhyloCfg(router=False, spawn_rule="probe", growth_rule="probe", d=D, hidden=16, n_classes=8, seed=seed, q=q, route_rule="accept_then_logf"))
            pool.add_expert(MLPExpert(D, 16, 1, 8, seed=seed), parent=None, state="active", meta={"sig": sig})

            x_A = draw(a, 5000, rng)
            x_B = draw(b_centre, 5000, rng)

            flg_b = 0
            for start in range(0, 5000, 64):
                batch = x_B[start : start + 64]
                pred = pool.predict(batch)
                flg_b += int(pred.unknown.sum().item())
            flagged_seeds.append(flg_b / 5000)

            p_A = pool.predict(x_A).p_nov.numpy()
            p_B = pool.predict(x_B).p_nov.numpy()
            labels = np.concatenate([np.zeros(5000), np.ones(5000)])  # 1 = foreign regime B
            scores = np.concatenate([1.0 - p_A, 1.0 - p_B])  # higher score = novel
            auroc_seeds.append(float(roc_auc_score(labels, scores)))

        mean_flg = float(np.mean(flagged_seeds))
        std_flg = float(np.std(flagged_seeds))
        mean_auc = float(np.mean(auroc_seeds))
        std_auc = float(np.std(auroc_seeds))

        entry = {
            "rho": rho,
            "mean_flagged_B": mean_flg,
            "std_flagged_B": std_flg,
            "mean_auroc": mean_auc,
            "std_auroc": std_auc,
            "verdict": "REPORT",
        }
        results.append(entry)
        print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): 1.2 rho={rho:4.2f} -> flagged={mean_flg:.4f}+-{std_flg:.4f}, auroc={mean_auc:.4f}+-{std_auc:.4f} [REPORT]")

    return results


def run_experiment_1_3():
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting Experiment 1.3")
    threshold = math.log(1.0 / 0.01)  # 4.60517
    n_fit_vals = [64, 256, 1024]
    variants = ["direct", "incremental"]
    m_vals = [32, 64, 128, 256]

    neg_results = []
    # Negative control: cluster is drawn from A itself
    for n_fit in n_fit_vals:
        for var in variants:
            for m in m_vals:
                ev_all = []
                above_count = 0
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    a, _, _ = centres(0.0, seed)
                    sig_A = VmfSignature(D, max_protos=256, seed=seed)
                    if var == "direct":
                        sig_A.update(draw(a, n_fit, rng))
                    else:
                        for _ in range(n_fit // 64):
                            sig_A.update(draw(a, 64, rng))
                    sig_A.fit()

                    for rep in range(10):
                        cluster = draw(a, m, rng)
                        # Reproduce formula line for line from phylo.py lines 682-701:
                        sig_c = VmfSignature(D, max_protos=256, seed=seed * 1000 + rep)
                        cx_sub = cluster[:256]
                        sig_c.update(cx_sub)
                        sig_c.fit()

                        if hasattr(sig_c, "kappa") and getattr(sig_c, "kappa") != 1.0:
                            logf_c_loo = sig_c.cal_scores + log_vmf_norm(D, sig_c.kappa)
                        else:
                            logf_c_loo = sig_c.cal_scores
                        protos_c = sig_c.protos
                        max_live_logf = sig_A.logpdf(protos_c)
                        evidence = float((logf_c_loo - max_live_logf).sum().item())
                        ev_all.append(evidence)
                        if evidence > threshold:
                            above_count += 1

                share_above = above_count / 200
                mean_ev = float(np.mean(ev_all))
                std_ev = float(np.std(ev_all))

                if share_above <= 0.03:
                    verdict = "PASS"
                else:
                    verdict = "DEFECT-EVIDENCE-UNCALIBRATED"

                entry = {
                    "n_fit": n_fit,
                    "variant": var,
                    "m": m,
                    "share_above_threshold": share_above,
                    "mean_evidence": mean_ev,
                    "std_evidence": std_ev,
                    "threshold": threshold,
                    "verdict": verdict,
                }
                neg_results.append(entry)
                print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): 1.3 NEG n_fit={n_fit:4d} {var:11s} m={m:3d} -> share={share_above:.3f}, ev={mean_ev:8.2f}+-{std_ev:6.2f} [{verdict}]")

    # Positive control: cluster drawn from B with rho in {0, 0.3, 0.6, 0.85}
    pos_results = []
    rho_vals = [0.0, 0.3, 0.6, 0.85]
    for rho in rho_vals:
        for m in m_vals:
            ev_all = []
            above_count = 0
            for seed in SEEDS:
                rng = np.random.default_rng(seed)
                a, b_centre, _ = centres(rho, seed)
                sig_A = VmfSignature(D, max_protos=256, seed=seed)
                sig_A.update(draw(a, 256, rng))
                sig_A.fit()

                for rep in range(10):
                    cluster = draw(b_centre, m, rng)
                    sig_c = VmfSignature(D, max_protos=256, seed=seed * 1000 + rep)
                    cx_sub = cluster[:256]
                    sig_c.update(cx_sub)
                    sig_c.fit()

                    if hasattr(sig_c, "kappa") and getattr(sig_c, "kappa") != 1.0:
                        logf_c_loo = sig_c.cal_scores + log_vmf_norm(D, sig_c.kappa)
                    else:
                        logf_c_loo = sig_c.cal_scores
                    protos_c = sig_c.protos
                    max_live_logf = sig_A.logpdf(protos_c)
                    evidence = float((logf_c_loo - max_live_logf).sum().item())
                    ev_all.append(evidence)
                    if evidence > threshold:
                        above_count += 1

            share_above = above_count / 200
            mean_ev = float(np.mean(ev_all))
            std_ev = float(np.std(ev_all))

            # CRITERION: >= 0.99 at rho 0 and 0.3 for m >= 64; if this control fails your formula is wrong: STOP and report.
            if rho in (0.0, 0.3) and m >= 64:
                if share_above >= 0.99:
                    verdict = "PASS"
                else:
                    verdict = "FAIL_STOP"
            else:
                verdict = "REPORT"

            entry = {
                "rho": rho,
                "n_fit": 256,
                "variant": "direct",
                "m": m,
                "share_above_threshold": share_above,
                "mean_evidence": mean_ev,
                "std_evidence": std_ev,
                "threshold": threshold,
                "verdict": verdict,
            }
            pos_results.append(entry)
            print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): 1.3 POS rho={rho:4.2f} m={m:3d} -> share={share_above:.3f}, ev={mean_ev:8.2f}+-{std_ev:6.2f} [{verdict}]")
            if verdict == "FAIL_STOP":
                raise RuntimeError(f"STOP: positive control failed at rho={rho}, m={m} with share={share_above} < 0.99")

    return neg_results, pos_results


def run_experiment_1_4():
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting Experiment 1.4")
    threshold = math.log(1.0 / 0.01)  # 4.60517
    n_fit_vals = [64, 256, 1024]
    variants = ["direct", "incremental"]
    m_vals = [32, 64, 128, 256]
    results = []

    for n_fit in n_fit_vals:
        for var in variants:
            for m in m_vals:
                ev_all = []
                above_count = 0
                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    a, _, u = centres(0.0, seed)
                    c_centre = u  # rho = 0 with A

                    sig_A = VmfSignature(D, max_protos=256, seed=seed)
                    sig_C = VmfSignature(D, max_protos=256, seed=seed + 100)
                    if var == "direct":
                        sig_A.update(draw(a, n_fit, rng))
                        sig_C.update(draw(c_centre, n_fit, rng))
                    else:
                        for _ in range(n_fit // 64):
                            sig_A.update(draw(a, 64, rng))
                            sig_C.update(draw(c_centre, 64, rng))
                    sig_A.fit()
                    sig_C.fit()

                    for rep in range(10):
                        cluster = draw(a, m, rng)
                        sig_c = VmfSignature(D, max_protos=256, seed=seed * 1000 + rep)
                        cx_sub = cluster[:256]
                        sig_c.update(cx_sub)
                        sig_c.fit()

                        if hasattr(sig_c, "kappa") and getattr(sig_c, "kappa") != 1.0:
                            logf_c_loo = sig_c.cal_scores + log_vmf_norm(D, sig_c.kappa)
                        else:
                            logf_c_loo = sig_c.cal_scores
                        protos_c = sig_c.protos
                        live_logfs = torch.stack([sig_A.logpdf(protos_c), sig_C.logpdf(protos_c)], dim=1)
                        max_live_logf = live_logfs.max(dim=1).values
                        evidence = float((logf_c_loo - max_live_logf).sum().item())
                        ev_all.append(evidence)
                        if evidence > threshold:
                            above_count += 1

                share_above = above_count / 200
                mean_ev = float(np.mean(ev_all))
                std_ev = float(np.std(ev_all))

                if share_above <= 0.03:
                    verdict = "PASS"
                else:
                    verdict = "DEFECT-EVIDENCE-UNCALIBRATED"

                entry = {
                    "n_fit": n_fit,
                    "variant": var,
                    "m": m,
                    "share_above_threshold": share_above,
                    "mean_evidence": mean_ev,
                    "std_evidence": std_ev,
                    "threshold": threshold,
                    "verdict": verdict,
                }
                results.append(entry)
                print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): 1.4 NEG 2-EXP n_fit={n_fit:4d} {var:11s} m={m:3d} -> share={share_above:.3f}, ev={mean_ev:8.2f}+-{std_ev:6.2f} [{verdict}]")

    return results


def write_outputs(exp1_1, exp1_2, exp1_3_neg, exp1_3_pos, exp1_4):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_data = {
        "exp1_1_gate_false_novelty": exp1_1,
        "exp1_2_gate_power_overlap": exp1_2,
        "exp1_3_negative_control": exp1_3_neg,
        "exp1_3_positive_control": exp1_3_pos,
        "exp1_4_two_experts_negative_control": exp1_4,
    }
    with open(OUT_DIR / "cva1.json", "w") as f:
        json.dump(all_data, f, indent=2)

    # Format CVA1_RESULTS.md
    md_lines = [
        "# CVA-1 Results: Gate and Evidence Verification",
        "",
        "VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Dimension: d = 128 everywhere | Simulation seeds: 0-19 (20 seeds)",
        "",
        "---",
        "",
        "## Experiment 1.1: Gate False Novelty",
        "",
        "**Criterion**: For n_fit >= 128, mean share of inputs flagged <= 1.5*q and mean share of batches flagged <= 1.5*q. (For n_fit = 32 smallest attainable p-value 1/(n+1) > q=0.02, so 'never flags' is expected note).",
        "",
        "| n_fit | q | Mean Inputs Flagged | Std Inputs | Mean Batches Flagged | Std Batches | Threshold (1.5*q) | Criterion | Verdict |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for r in exp1_1:
        md_lines.append(
            f"| {r['n_fit']} | {r['q']:.2f} | {r['mean_inputs_flagged']:.5f} | {r['std_inputs_flagged']:.5f} | {r['mean_batches_flagged']:.5f} | {r['std_batches_flagged']:.5f} | {r['threshold_1_5_q']:.3f} | mean <= {r['threshold_1_5_q']:.3f} | {r['verdict']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Experiment 1.2: Gate Power Against Overlap",
        "",
        "**Criterion**: No pass/fail; report share of inputs flagged and AUROC of p_nov (A vs B).",
        "",
        "| rho | Mean Flagged (B) | Std Flagged | Mean AUROC | Std AUROC | Criterion | Verdict |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for r in exp1_2:
        md_lines.append(
            f"| {r['rho']:.2f} | {r['mean_flagged_B']:.4f} | {r['std_flagged_B']:.4f} | {r['mean_auroc']:.4f} | {r['std_auroc']:.4f} | Report | {r['verdict']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Experiment 1.3: Evidence-Test False Spawns (Negative Control)",
        "",
        "**Criterion**: Share of clusters with evidence > log(1/0.01) must be <= 0.03 in every cell. A cell above 0.03 is registered as DEFECT-EVIDENCE-UNCALIBRATED with its measured share.",
        "",
        "| n_fit | Variant | Cluster Size m | Share Above Threshold | Mean Evidence | Std Evidence | Criterion | Verdict |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for r in exp1_3_neg:
        md_lines.append(
            f"| {r['n_fit']} | {r['variant']} | {r['m']} | {r['share_above_threshold']:.3f} | {r['mean_evidence']:.2f} | {r['std_evidence']:.2f} | share <= 0.03 | {r['verdict']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Experiment 1.3: Evidence-Test True Spawns (Positive Control)",
        "",
        "**Criterion**: Share above threshold >= 0.99 at rho 0.0 and 0.3 for m >= 64. If this control fails, formula is wrong: STOP and report.",
        "",
        "| rho | n_fit | Variant | Cluster Size m | Share Above Threshold | Mean Evidence | Std Evidence | Criterion | Verdict |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for r in exp1_3_pos:
        crit_str = ">= 0.99" if (r['rho'] in (0.0, 0.3) and r['m'] >= 64) else "Report"
        md_lines.append(
            f"| {r['rho']:.2f} | {r['n_fit']} | {r['variant']} | {r['m']} | {r['share_above_threshold']:.3f} | {r['mean_evidence']:.2f} | {r['std_evidence']:.2f} | {crit_str} | {r['verdict']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Experiment 1.4: Evidence-Test False Spawns with Second Live Expert (Regime C at rho = 0)",
        "",
        "**Criterion**: Share above threshold <= 0.03 in every cell.",
        "",
        "| n_fit | Variant | Cluster Size m | Share Above Threshold | Mean Evidence | Std Evidence | Criterion | Verdict |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for r in exp1_4:
        md_lines.append(
            f"| {r['n_fit']} | {r['variant']} | {r['m']} | {r['share_above_threshold']:.3f} | {r['mean_evidence']:.2f} | {r['std_evidence']:.2f} | share <= 0.03 | {r['verdict']} |"
        )

    with open(OUT_DIR / "CVA1_RESULTS.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Saved cva1.json and CVA1_RESULTS.md to {OUT_DIR}")


if __name__ == "__main__":
    t0 = time.time()
    exp1_1 = run_experiment_1_1()
    exp1_2 = run_experiment_1_2()
    exp1_3_neg, exp1_3_pos = run_experiment_1_3()
    exp1_4 = run_experiment_1_4()
    write_outputs(exp1_1, exp1_2, exp1_3_neg, exp1_3_pos, exp1_4)
    print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Total runtime = {time.time() - t0:.2f}s")
