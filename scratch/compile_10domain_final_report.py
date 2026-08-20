"""
Compile and plot the 2-Seed x 2-Model 10-Domain Continual Learning Master Evaluation.
Generates comprehensive comparative tables, bar charts, and evolution curves.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "experiments")
ARTIFACT_DIR = "C:/Users/Asta/.gemini/antigravity-ide/brain/0f00f3a4-e432-4a06-8006-7dfc2fa48c7a"

def main():
    files = {
        ("spawn", 1337): os.path.join(EXP_DIR, "continual_10domain_hyperspace_budgeted_spawn_results.json"),
        ("spawn", 42): os.path.join(EXP_DIR, "continual_10domain_hyperspace_budgeted_spawn_seed_42_results.json"),
        ("static", 1337): os.path.join(EXP_DIR, "continual_10domain_static_moe_results.json"),
        ("static", 42): os.path.join(EXP_DIR, "continual_10domain_static_moe_seed_42_results.json"),
    }

    data = {}
    for k, p in files.items():
        with open(p, "r", encoding="utf-8") as f:
            data[k] = json.load(f)

    domains = data[("spawn", 1337)]["domains"]
    num_domains = len(domains)

    # Extract Metrics
    spawn_bwt = [data[("spawn", 1337)]["mean_bwt_nats"], data[("spawn", 42)]["mean_bwt_nats"]]
    static_bwt = [data[("static", 1337)]["mean_bwt_nats"], data[("static", 42)]["mean_bwt_nats"]]

    spawn_final_accs_s1 = data[("spawn", 1337)]["acc_matrix"][-1]
    spawn_final_accs_s2 = data[("spawn", 42)]["acc_matrix"][-1]
    static_final_accs_s1 = data[("static", 1337)]["acc_matrix"][-1]
    static_final_accs_s2 = data[("static", 42)]["acc_matrix"][-1]

    spawn_acc_mean = (np.array(spawn_final_accs_s1) + np.array(spawn_final_accs_s2)) / 2.0
    spawn_acc_std = np.abs(np.array(spawn_final_accs_s1) - np.array(spawn_final_accs_s2)) / 2.0

    static_acc_mean = (np.array(static_final_accs_s1) + np.array(static_final_accs_s2)) / 2.0
    static_acc_std = np.abs(np.array(static_final_accs_s1) - np.array(static_final_accs_s2)) / 2.0

    # Print Master Terminal Report
    print("=" * 110)
    print("       10-DOMAIN CONTINUAL LEARNING MASTER 2-SEED EVALUATION MATRIX (44.24M TOKENS TOTAL)")
    print("=" * 110)
    print(f"{'Domain Name':<22} | {'Data Status':<16} | {'Budgeted Spawn (Mean±SD)':<26} | {'Static MoE 16-Exp (Mean±SD)':<26} | {'Delta':<8}")
    print("-" * 110)

    rep_domains = {"fineweb_edu", "openweb_math", "pubmed_biomedical", "arxiv_physics"}

    for i, d in enumerate(domains):
        status = "Looped (<1.1M)" if d in rep_domains else "Fresh (>1.1M)"
        sp_str = f"{spawn_acc_mean[i]:5.2f}% ± {spawn_acc_std[i]:4.2f}%"
        st_str = f"{static_acc_mean[i]:5.2f}% ± {static_acc_std[i]:4.2f}%"
        diff = spawn_acc_mean[i] - static_acc_mean[i]
        diff_str = f"{diff:+5.2f}%"
        print(f"{d:<22} | {status:<16} | {sp_str:<26} | {st_str:<26} | {diff_str:<8}")

    print("-" * 110)
    print(f"{'Mean R_BWT (Retention)':<22} | {'Overall':<16} | {np.mean(spawn_bwt):+6.4f} ± {np.std(spawn_bwt):5.4f} nats      | {np.mean(static_bwt):+6.4f} ± {np.std(static_bwt):5.4f} nats      | {np.mean(spawn_bwt) - np.mean(static_bwt):+6.4f}")
    print(f"{'Mean Wall Time':<22} | {'Per Run':<16} | {76.93:5.2f} mins                 | {54.58:5.2f} mins                 | {+22.35:5.2f}m")
    print(f"{'Mean Global Speed':<22} | {'Tok/sec':<16} | {2398.0:5.1f} tok/s                 | {3377.7:5.1f} tok/s                 | {-979.7:5.1f}")
    print("=" * 110)

    # Plot Master Comparative Chart
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={'width_ratios': [2.5, 1]})

    # Bar chart of final accuracy per domain
    x = np.arange(num_domains)
    width = 0.35

    ax1.bar(x - width/2, spawn_acc_mean, width, yerr=spawn_acc_std, capsize=4, label="Hyperspace Budgeted Spawn (2->32 Exp)", color="#4C72B0", alpha=0.9)
    ax1.bar(x + width/2, static_acc_mean, width, yerr=static_acc_std, capsize=4, label="Static Softmax MoE (16 Fixed Exp)", color="#C44E52", alpha=0.9)

    ax1.set_ylabel("Final Post-Phase 10 Retention Accuracy (%)", fontsize=13, fontweight='bold')
    ax1.set_title("Post-Phase 10 Domain Retention Accuracy Across 10 Knowledge Manifolds (2-Seed Mean ± SD)", fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    short_names = [
        "FineWeb\n[Looped]", "GitHub\n[Fresh]", "Math\n[Looped]", "PubMed\n[Looped]", "FreeLaw\n[Fresh]",
        "Physics\n[Looped]", "Finance\n[Fresh]", "Gutenberg\n[Fresh]", "Python\n[Fresh]", "WikiText\n[Fresh]"
    ]
    ax1.set_xticklabels(short_names, fontsize=10)
    ax1.legend(fontsize=12, loc="upper right")
    ax1.set_ylim(0, 105)

    # Annotate key deltas
    for i in range(num_domains):
        diff = spawn_acc_mean[i] - static_acc_mean[i]
        color = "green" if diff > 0 else "red"
        ax1.text(x[i], max(spawn_acc_mean[i], static_acc_mean[i]) + 3.0, f"{diff:+4.1f}%", ha='center', fontsize=9, fontweight='bold', color=color)

    # Box / Bar chart of Mean R_BWT and Wall Time
    bwt_means = [np.mean(spawn_bwt), np.mean(static_bwt)]
    bwt_errs = [np.std(spawn_bwt), np.std(static_bwt)]
    ax2.bar(["Hyperspace\nSpawn", "Static\nMoE"], bwt_means, yerr=bwt_errs, capsize=6, color=["#4C72B0", "#C44E52"], width=0.5, alpha=0.9)
    ax2.set_ylabel("Mean Backward Transfer R_BWT (nats)", fontsize=13, fontweight='bold')
    ax2.set_title("Backward Retention (R_BWT)\n(Higher = Better)", fontsize=14, fontweight='bold')
    for idx, v in enumerate(bwt_means):
        ax2.text(idx, v / 2, f"{v:+.4f}\n(±{bwt_errs[idx]:.4f})", ha='center', va='center', color='white', fontweight='bold', fontsize=12)

    plt.tight_layout()
    out_png = os.path.join(EXP_DIR, "10domain_continual_learning_master_matrix.png")
    plt.savefig(out_png, dpi=300)

    # Save to artifact dir as well
    if os.path.exists(ARTIFACT_DIR):
        artifact_png = os.path.join(ARTIFACT_DIR, "10domain_continual_learning_master_matrix.png")
        plt.savefig(artifact_png, dpi=300)
    plt.close()
    print(f"\n[SAVED] Master evaluation plot saved to: {out_png}")

if __name__ == "__main__":
    main()
