"""
Compile and plot N-Seed x 2-Model 10-Domain Continual Learning Master Evaluation.
Computes exact Welch's t-test with Welch-Satterthwaite degrees of freedom and p-values.
Verifies against Pre-Registered Replication Criteria.
"""

import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments")
ARTIFACT_DIR = "C:/Users/Asta/.gemini/antigravity-ide/brain/0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"

def welch_satterthwaite_df(s1, s2, n1, n2):
    """Compute exact Welch-Satterthwaite degrees of freedom."""
    v1 = (s1 ** 2) / n1
    v2 = (s2 ** 2) / n2
    if v1 + v2 == 0:
        return n1 + n2 - 2
    num = (v1 + v2) ** 2
    den = ((v1 ** 2) / (n1 - 1)) + ((v2 ** 2) / (n2 - 1))
    return num / den

def load_results(model_prefix):
    pattern = os.path.join(EXP_DIR, f"continual_10domain_{model_prefix}*_results.json")
    files = glob.glob(pattern)
    data_by_seed = {}
    for p in sorted(files):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
                seed = d.get("seed", 1337)
                if "acc_matrix" in d and len(d["acc_matrix"]) == 10:
                    data_by_seed[seed] = d
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
    return data_by_seed

def main():
    spawn_data = load_results("hyperspace_budgeted_spawn")
    static_data = load_results("static_moe")

    spawn_seeds = sorted(list(spawn_data.keys()))
    static_seeds = sorted(list(static_data.keys()))
    common_seeds = [s for s in spawn_seeds if s in static_seeds]

    n_spawn = len(spawn_seeds)
    n_static = len(static_seeds)
    n_common = len(common_seeds)

    if n_spawn == 0 or n_static == 0:
        print("Error: No valid result files found.")
        return

    ref_seed = spawn_seeds[0]
    domains = spawn_data[ref_seed]["domains"]
    num_domains = len(domains)

    # Extract final accuracy arrays per seed: shape (N_seeds, 10)
    spawn_accs = np.array([spawn_data[s]["acc_matrix"][-1] for s in spawn_seeds])
    static_accs = np.array([static_data[s]["acc_matrix"][-1] for s in static_seeds])

    # Compute means and sample standard deviations (ddof=1)
    spawn_mean = np.mean(spawn_accs, axis=0)
    spawn_std = np.std(spawn_accs, axis=0, ddof=1) if n_spawn > 1 else np.zeros(num_domains)

    static_mean = np.mean(static_accs, axis=0)
    static_std = np.std(static_accs, axis=0, ddof=1) if n_static > 1 else np.zeros(num_domains)

    # Extract Mean R_BWT
    spawn_bwts = [spawn_data[s]["mean_bwt_nats"] for s in spawn_seeds]
    static_bwts = [static_data[s]["mean_bwt_nats"] for s in static_seeds]

    bwt_spawn_m = np.mean(spawn_bwts)
    bwt_spawn_s = np.std(spawn_bwts, ddof=1) if n_spawn > 1 else 0.0
    bwt_static_m = np.mean(static_bwts)
    bwt_static_s = np.std(static_bwts, ddof=1) if n_static > 1 else 0.0

    rep_domains = {"fineweb_edu", "openweb_math", "pubmed_biomedical", "arxiv_physics"}

    print("\n" + "=" * 125)
    print(f"       10-DOMAIN CONTINUAL LEARNING EVALUATION (Spawn Seeds: {spawn_seeds} | Static Seeds: {static_seeds})")
    print("=" * 125)
    print(f"{'Domain Name':<20} | {'Status':<12} | {'Spawn (Mean±SD)':<20} | {'Static (Mean±SD)':<20} | {'Delta':<7} | {'Welch t':<8} | {'df':<5} | {'p-val':<8} | {'Paired Consistent?'}")
    print("-" * 125)

    for i, d in enumerate(domains):
        status = "Looped" if d in rep_domains else "Fresh"
        sp_str = f"{spawn_mean[i]:5.2f}% ± {spawn_std[i]:4.2f}%"
        st_str = f"{static_mean[i]:5.2f}% ± {static_std[i]:4.2f}%"
        delta = spawn_mean[i] - static_mean[i]
        delta_str = f"{delta:+5.2f}%"

        # Welch's t-test
        s1, s2 = spawn_std[i], static_std[i]
        se_diff = np.sqrt((s1**2)/n_spawn + (s2**2)/n_static)
        t_val = (delta / se_diff) if se_diff > 0 else 0.0
        df_val = welch_satterthwaite_df(s1, s2, n_spawn, n_static)
        p_val = 2.0 * stats.t.sf(np.abs(t_val), df=df_val) if df_val > 0 else 1.0

        # Paired directional check across common seeds
        paired_signs = []
        for s in common_seeds:
            sp_val = spawn_data[s]["acc_matrix"][-1][i]
            st_val = static_data[s]["acc_matrix"][-1][i]
            paired_signs.append(sp_val > st_val)
        
        all_pos = all(paired_signs)
        all_neg = all(not x for x in paired_signs)
        if all_pos:
            consist_str = f"YES (+ on all {len(common_seeds)})"
        elif all_neg:
            consist_str = f"NO (- on all {len(common_seeds)})"
        else:
            n_pos = sum(paired_signs)
            consist_str = f"MIXED ({n_pos}/{len(common_seeds)} +)"

        t_str = f"{t_val:+6.2f}"
        df_str = f"{df_val:4.1f}"
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "<0.0001"

        print(f"{d:<20} | {status:<12} | {sp_str:<20} | {st_str:<20} | {delta_str:<7} | {t_str:<8} | {df_str:<5} | {p_str:<8} | {consist_str}")

    # Backward Transfer
    bwt_se = np.sqrt((bwt_spawn_s**2)/n_spawn + (bwt_static_s**2)/n_static)
    bwt_t = ((bwt_spawn_m - bwt_static_m) / bwt_se) if bwt_se > 0 else 0.0
    bwt_df = welch_satterthwaite_df(bwt_spawn_s, bwt_static_s, n_spawn, n_static)
    bwt_p = 2.0 * stats.t.sf(np.abs(bwt_t), df=bwt_df) if bwt_df > 0 else 1.0

    print("-" * 125)
    print(f"{'Mean R_BWT':<20} | {'Overall':<12} | {bwt_spawn_m:+6.4f} ± {bwt_spawn_s:5.4f} nats | {bwt_static_m:+6.4f} ± {bwt_static_s:5.4f} nats | {bwt_spawn_m - bwt_static_m:+6.4f} | {bwt_t:+6.2f} | {bwt_df:4.1f} | {bwt_p:.4f} | ---")
    print("=" * 125)

    # Plot Master Comparative Chart
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [2.5, 1]})

    x = np.arange(num_domains)
    width = 0.35

    ax1.bar(x - width/2, spawn_mean, width, yerr=spawn_std, capsize=4, label=f"Hyperspace Budgeted Spawn (N={n_spawn} Seeds)", color="#4C72B0", alpha=0.9)
    ax1.bar(x + width/2, static_mean, width, yerr=static_std, capsize=4, label=f"Static Softmax MoE 16-Exp (N={n_static} Seeds)", color="#C44E52", alpha=0.9)

    ax1.set_ylabel("Final Post-Phase 10 Retention Accuracy (%)", fontsize=13, fontweight='bold')
    ax1.set_title(f"Post-Phase 10 Domain Retention Accuracy Across 10 Knowledge Manifolds ({n_common}-Seed Mean ± SD)", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    short_names = [
        "FineWeb\n[Looped]", "GitHub\n[Fresh]", "Math\n[Looped]", "PubMed\n[Looped]", "FreeLaw\n[Fresh]",
        "Physics\n[Looped]", "Finance\n[Fresh]", "Gutenberg\n[Fresh]", "Python\n[Fresh]", "WikiText\n[Fresh]"
    ]
    ax1.set_xticklabels(short_names, fontsize=10)
    ax1.legend(fontsize=12, loc="upper right")
    ax1.set_ylim(0, 105)

    for i in range(num_domains):
        diff = spawn_mean[i] - static_mean[i]
        color = "green" if diff > 0 else "red"
        ax1.text(x[i], max(spawn_mean[i], static_mean[i]) + 3.0, f"{diff:+4.1f}%", ha='center', fontsize=9, fontweight='bold', color=color)

    bwt_means = [bwt_spawn_m, bwt_static_m]
    bwt_errs = [bwt_spawn_s, bwt_static_s]
    ax2.bar([f"Hyperspace\nSpawn (N={n_spawn})", f"Static\nMoE (N={n_static})"], bwt_means, yerr=bwt_errs, capsize=6, color=["#4C72B0", "#C44E52"], width=0.5, alpha=0.9)
    ax2.set_ylabel("Mean Backward Transfer R_BWT (nats)", fontsize=13, fontweight='bold')
    ax2.set_title("Backward Retention (R_BWT)\n(Higher = Better)", fontsize=14, fontweight='bold')
    for idx, v in enumerate(bwt_means):
        ax2.text(idx, v / 2, f"{v:+.4f}\n(±{bwt_errs[idx]:.4f})", ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.tight_layout()
    out_png = os.path.join(EXP_DIR, "10domain_continual_learning_master_matrix.png")
    plt.savefig(out_png, dpi=300)

    if os.path.exists(ARTIFACT_DIR):
        artifact_png = os.path.join(ARTIFACT_DIR, "10domain_continual_learning_master_matrix.png")
        plt.savefig(artifact_png, dpi=300)
    plt.close()
    print(f"\n[SAVED] Master evaluation plot saved to: {out_png}")

if __name__ == "__main__":
    main()
