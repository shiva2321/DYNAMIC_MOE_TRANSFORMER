"""Extract verbatim statistics from results/real_pilot_v8/test_track1 and test_track2."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "results" / "real_pilot_v8"

t1 = json.loads((V8 / "test_track1" / "summary.json").read_text())
print("=== TRACK 1 (5 seeds: 0..4) ===")
for arm, d in t1["arms"].items():
    m = d["track1"]
    print(f"\n[{arm}] (n={d['n_seeds']}, time={m['train_seconds']['mean']:.1f}s)")
    print(
        f"  seen={m['seen_acc']['mean']:.3f}+/-{m['seen_acc']['ci95']:.3f} | "
        f"held={m['heldout_unseen_acc']['mean']:.3f}+/-{m['heldout_unseen_acc']['ci95']:.3f} vals={[round(x,3) for x in m['heldout_unseen_acc']['values']]}"
    )
    print(
        f"  para={m['paraphrase_acc']['mean']:.3f}+/-{m['paraphrase_acc']['ci95']:.3f} | "
        f"iit_seen={m['iit_causal_acc']['mean']:.3f} | "
        f"iit_held={m['iit_heldout_acc']['mean']:.3f}+/-{m['iit_heldout_acc']['ci95']:.3f} | "
        f"rand_slice={m['iit_random_slice_null']['mean']:.3f} | "
        f"burst={m['burst_immunity_acc']['mean']:.3f}+/-{m['burst_immunity_acc']['ci95']:.3f} vals={[round(x,3) for x in m['burst_immunity_acc']['values']]} | "
        f"near_ood={m['near_ood_abstain_auroc']['mean']:.3f}"
    )
    for sh in ("shots_4", "shots_8", "shots_16"):
        e = m["episodic_v10"][sh]
        print(
            f"  [{sh:8s}] lookup={e['lookup_acc']['mean']:.3f}+/-{e['lookup_acc']['ci95']:.3f} | "
            f"shuff={e['shuffled_null_acc']['mean']:.3f} | imp={e['impossible_perm_null_acc']['mean']:.3f} | "
            f"comp={e['comp_equiv_induction_acc']['mean']:.3f}+/-{e['comp_equiv_induction_acc']['ci95']:.3f} vals={[round(x,3) for x in e['comp_equiv_induction_acc']['values']]} | "
            f"shift={e['unseen_meaning_shift_acc']['mean']:.3f}+/-{e['unseen_meaning_shift_acc']['ci95']:.3f} vals={[round(x,3) for x in e['unseen_meaning_shift_acc']['values']]} | "
            f"s_shuff={e['shift_shuffled_null_acc']['mean']:.3f} | blind={e['shift_no_mem_blind_acc']['mean']:.3f}"
        )

t2 = json.loads((V8 / "test_track2" / "summary.json").read_text())
print("\n=== TRACK 2 (3 seeds: 0..2) ===")
for arm, d in t2["arms"].items():
    print(f"\n[{arm}] (n={d['n_seeds']}, time={d['train_seconds']['mean']:.1f}s)")
    print(
        f"  b77_acc={d['b77_intent_77_acc']['mean']:.3f}+/-{d['b77_intent_77_acc']['ci95']:.3f} vals={[round(x,3) for x in d['b77_intent_77_acc']['values']]} | "
        f"b77_auc={d['b77_same_vs_overlap_diff_auc']['mean']:.3f}+/-{d['b77_same_vs_overlap_diff_auc']['ci95']:.3f} | "
        f"r5={d['mean_final_r5']['mean']:.3f}+/-{d['mean_final_r5']['ci95']:.3f} | "
        f"hf={d['heldout_families_auc']['mean']:.3f} | nf={d['mean_nuance_flip_auc']['mean']:.3f}"
    )
    print(
        f"  mean_k={d['mean_k']['mean']:.3f}+/-{d['mean_k']['ci95']:.3f} | E={d['num_experts']['mean']:.1f} | "
        f"ami={d['func_routing_ami']['mean']:.3f} | "
        f"lm_ami={d['length_matched_role_ami']['mean']:.3f}+/-{d['length_matched_role_ami']['ci95']:.3f} vals={[round(x,3) for x in d['length_matched_role_ami']['values']]} | "
        f"len_ami={d['routing_length_ami']['mean']:.3f}+/-{d['routing_length_ami']['ci95']:.3f} | "
        f"pert={d['routing_perturb_agree']['mean']:.3f}+/-{d['routing_perturb_agree']['ci95']:.3f} vals={[round(x,3) for x in d['routing_perturb_agree']['values']]}"
    )
    print(
        f"  les_acc_drop={d['lesion_max_acc_drop']['mean']:.3f} | les_role_drop={d['lesion_max_role_drop']['mean']:.3f} | "
        f"les_rel_delta={d['lesion_mean_rel_delta']['mean']:.3f} | role_spec={d['lesion_role_specificity']['mean']:.2f}"
    )
    if arm == "CognitiveDMoE":
        decs = [json.loads((V8 / "test_track2" / f"seed{s}_CognitiveDMoE.json").read_text())["detached_decoder_diag"] for s in (0, 1, 2)]
        print(f"  detached_decoder_diags={decs}")
