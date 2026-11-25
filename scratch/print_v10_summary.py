import json
from pathlib import Path

t1 = json.loads(Path("results/real_pilot_v7/test_track1/summary.json").read_text())
t2 = json.loads(Path("results/real_pilot_v7/test_track2/summary.json").read_text())

print("=== TRACK 1 (5 seeds: 0..4) ===")
for arm, d in t1["arms"].items():
    tk = d["track1"]
    print(f"\n[{arm}] (n={d['n_seeds']}, time={tk['train_seconds']['mean']:.1f}s)")
    print(
        f"  seen={tk['seen_acc']['mean']:.3f}+/-{tk['seen_acc']['ci95']:.3f} | "
        f"held={tk['heldout_unseen_acc']['mean']:.3f}+/-{tk['heldout_unseen_acc']['ci95']:.3f} "
        f"vals={[round(v, 3) for v in tk['heldout_unseen_acc']['values']]}"
    )
    print(
        f"  para={tk['paraphrase_acc']['mean']:.3f}+/-{tk['paraphrase_acc']['ci95']:.3f} | "
        f"iit_seen={tk['iit_causal_acc']['mean']:.3f} | "
        f"iit_held={tk['iit_heldout_acc']['mean']:.3f}+/-{tk['iit_heldout_acc']['ci95']:.3f} | "
        f"rand_slice={tk['iit_random_slice_null']['mean']:.3f} | "
        f"burst={tk['burst_immunity_acc']['mean']:.3f} | "
        f"near_ood={tk['near_ood_abstain_auroc']['mean']:.3f}"
    )
    for sh in ("shots_4", "shots_8", "shots_16"):
        ep = tk["episodic_v10"][sh]
        print(
            f"  [{sh:8s}] lookup={ep['lookup_acc']['mean']:.3f}+/-{ep['lookup_acc']['ci95']:.3f} | "
            f"shuff={ep['shuffled_null_acc']['mean']:.3f} | "
            f"imp={ep['impossible_perm_null_acc']['mean']:.3f} | "
            f"comp={ep['comp_equiv_induction_acc']['mean']:.3f}+/-{ep['comp_equiv_induction_acc']['ci95']:.3f} "
            f"vals={[round(v, 3) for v in ep['comp_equiv_induction_acc']['values']]} | "
            f"shift={ep['unseen_meaning_shift_acc']['mean']:.3f}+/-{ep['unseen_meaning_shift_acc']['ci95']:.3f} "
            f"vals={[round(v, 3) for v in ep['unseen_meaning_shift_acc']['values']]} | "
            f"s_shuff={ep['shift_shuffled_null_acc']['mean']:.3f} | "
            f"blind={ep['shift_no_mem_blind_acc']['mean']:.3f}"
        )

print("\n=== TRACK 2 (3 seeds: 0..2) ===")
for arm, d in t2["arms"].items():
    print(f"\n[{arm}] (n={d['n_seeds']}, time={d['train_seconds']['mean']:.1f}s)")
    print(
        f"  b77_acc={d['b77_intent_77_acc']['mean']:.3f}+/-{d['b77_intent_77_acc']['ci95']:.3f} "
        f"vals={[round(v, 3) for v in d['b77_intent_77_acc']['values']]} | "
        f"b77_auc={d['b77_same_vs_overlap_diff_auc']['mean']:.3f}+/-{d['b77_same_vs_overlap_diff_auc']['ci95']:.3f} | "
        f"r5={d['mean_final_r5']['mean']:.3f} | hf={d['heldout_families_auc']['mean']:.3f} | "
        f"nf={d['mean_nuance_flip_auc']['mean']:.3f}"
    )
    print(
        f"  mean_k={d['mean_k']['mean']:.3f}+/-{d['mean_k']['ci95']:.3f} | E={d['num_experts']['mean']:.1f} | "
        f"ami={d['func_routing_ami']['mean']:.3f} | "
        f"lm_ami={d['length_matched_role_ami']['mean']:.3f}+/-{d['length_matched_role_ami']['ci95']:.3f} | "
        f"len_ami={d['routing_length_ami']['mean']:.3f}+/-{d['routing_length_ami']['ci95']:.3f} | "
        f"pert={d['routing_perturb_agree']['mean']:.3f}+/-{d['routing_perturb_agree']['ci95']:.3f}"
    )
    print(
        f"  les_acc_drop={d['lesion_max_acc_drop']['mean']:.3f} | "
        f"les_role_drop={d['lesion_max_role_drop']['mean']:.3f} | "
        f"les_rel_delta={d['lesion_mean_rel_delta']['mean']:.3f} | "
        f"role_spec={d['lesion_role_specificity']['mean']:.2f}"
    )
