"""Programmatically build results/real_pilot_v8/tuning_log.json from dev trial artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V8_ROOT = ROOT / "results" / "real_pilot_v8"

TRIAL_NOTES = [
    ("track1", 1, "dev_t1_trial1", "Initial v11 Track 1 config (lr_muon=0.012, wd=0.22, gf_alpha=0.95, gf_lamb=1.5, op_rank_low=12, coupled clip)."),
    ("track1", 2, "dev_t1_trial2", "Decoupled op_opt (zero trunk clip interference), lr_muon=0.009, lr_adam=0.0035, wd=0.14, gf_alpha=0.98, gf_lamb=2.0, op_rank_low=12. SELECTED & FROZEN FOR TRACK 1 TEST."),
    ("track1", 3, "dev_t1_trial3", "Low Grokfast (gf_lamb=0.5, wd=0.12) + op_rank_low=4 + LOOCV gate (failed to grok on seed 100)."),
    ("track1", 4, "dev_t1_trial4", "Grokfast cooldown (gf_lamb=2.5->0 at 75%) + op_rank_low=4 + LOOCV gate (LOOCV suppressed low-rank head during cold start)."),
    ("track2", 1, "dev_t2_trial1", "Initial v11 Track 2 config (shift_invariant_pool=True, grad_scale=0.15, cris_weight=0.45, frozen_exempt_balance=True, GPM readout-exempt)."),
    ("track2", 2, "dev_t2_trial2", "Sharp chunker (shift_invariant_pool=False), grad_scale=0.0 + no_grad lower trunk in extract_router_queries, zero-mean centered plastic expert_bias, cris_weight=0.35."),
    ("track2", 3, "dev_t2_trial3", "Froze surf_proj_basis after warmup (>=160 steps), exempted q_proj/ctx_proj from GPM, exp_orth every 4 steps, cris_weight=0.25, pert_every=4, shared_dropout=0.15. SELECTED & FROZEN FOR TRACK 2 TEST."),
    ("track2", 4, "dev_t2_trial4", "Reduced pert_every=8 and shared_dropout=0.10 (degraded b77, len_ami, and pert vs Trial 3)."),
]

log_entries = []
for track, num, folder, notes in TRIAL_NOTES:
    p = V8_ROOT / folder / "summary.json"
    data = json.loads(p.read_text())
    res_summary = {}
    for arm, stats in data["arms"].items():
        if track == "track1":
            t1 = stats["track1"]
            ep4 = t1["episodic_v10"]["shots_4"]
            ep8 = t1["episodic_v10"]["shots_8"]
            ep16 = t1["episodic_v10"]["shots_16"]
            res_summary[arm] = {
                "train_seconds": round(t1["train_seconds"]["mean"], 2),
                "seen_acc": round(t1["seen_acc"]["mean"], 4),
                "heldout_unseen_acc": round(t1["heldout_unseen_acc"]["mean"], 4),
                "paraphrase_acc": round(t1["paraphrase_acc"]["mean"], 4),
                "iit_heldout_acc": round(t1["iit_heldout_acc"]["mean"], 4),
                "burst_immunity_acc": round(t1["burst_immunity_acc"]["mean"], 4),
                "shots_4_comp_equiv": round(ep4["comp_equiv_induction_acc"]["mean"], 4),
                "shots_8_lookup": round(ep8["lookup_acc"]["mean"], 4),
                "shots_8_comp_equiv": round(ep8["comp_equiv_induction_acc"]["mean"], 4),
                "shots_8_shift": round(ep8["unseen_meaning_shift_acc"]["mean"], 4),
                "shots_16_shift": round(ep16["unseen_meaning_shift_acc"]["mean"], 4),
                "shots_8_shift_shuff": round(ep8["shift_shuffled_null_acc"]["mean"], 4),
            }
        else:
            res_summary[arm] = {
                "train_seconds": round(stats["train_seconds"]["mean"], 2),
                "b77_acc": round(stats["b77_intent_77_acc"]["mean"], 4),
                "b77_auc": round(stats["b77_same_vs_overlap_diff_auc"]["mean"], 4),
                "mean_final_r5": round(stats["mean_final_r5"]["mean"], 4),
                "mean_k": round(stats["mean_k"]["mean"], 4),
                "num_experts": round(stats["num_experts"]["mean"], 2),
                "lm_ami": round(stats["length_matched_role_ami"]["mean"], 4),
                "len_ami": round(stats["routing_length_ami"]["mean"], 4),
                "pert": round(stats["routing_perturb_agree"]["mean"], 4),
                "les_d": round(stats["lesion_mean_rel_delta"]["mean"], 4),
                "role_spec": round(stats["lesion_role_specificity"]["mean"], 3),
            }
    log_entries.append({
        "trial": num,
        "track": track,
        "seed": 100,
        "out_dir": f"results/real_pilot_v8/{folder}",
        "changes": notes,
        "results": res_summary,
    })

out_path = V8_ROOT / "tuning_log.json"
out_path.write_text(json.dumps(log_entries, indent=2))
print(f"Wrote {out_path} with {len(log_entries)} dev trials.")
