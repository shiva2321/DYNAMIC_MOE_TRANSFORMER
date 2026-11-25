"""Print every detailed probe table and sample generations from test_v1."""
from __future__ import annotations

import json
import pathlib

s = json.loads(pathlib.Path("results/real_pilot_v1/test_v1_summary.json").read_text())
DOMS = ["wiki", "story", "code", "math", "forum"]
ARMS = list(s["arms"].keys())

print("=== 1. SEQUENTIAL PEAK VS FINAL BPB & FORGETTING (%) ===")
for arm in ARMS:
    a = s["arms"][arm]
    if "peak_after_own_bpb" not in a:
        continue
    pk = {d: round(a["peak_after_own_bpb"][d]["mean"], 3) for d in DOMS}
    fn = {d: round(a["final_bpb"][d]["mean"], 3) for d in DOMS}
    fg = {d: round(a["forgetting_bpb"][d]["mean"], 3) for d in DOMS[:4]}
    fgp = {d: round(a["forgetting_pct"][d]["mean"], 1) for d in DOMS[:4]}
    print(f"{arm:17s} | peak={pk} | final={fn} | dBPB={fg} ({a['mean_forgetting_4prior_bpb']['mean']:+.3f}+/-{a['mean_forgetting_4prior_bpb']['std']:.3f}, {a['mean_forgetting_4prior_pct']['mean']:+.1f}%)")

print("\n=== 2. COMPUTE, MEMORY, ROUTING, LIFECYCLE, GPM SATURATION ===")
for arm in ARMS:
    a = s["arms"][arm]
    print(f"{arm:17s} | sec={a['train_seconds']['mean']:.1f} fwd={a['forward_passes']['mean']:.0f} bwd={a['backward_passes']['mean']:.0f} skip={a['skipped_steps']['mean']:.1f} params={a['params_final']['mean']:.0f} vram={a['peak_vram_mb']['mean']:.1f}MB",
          f"exp={a.get('experts_final',{}).get('mean')} nmi={a.get('nmi',{}).get('mean')}" if "experts_final" in a or "nmi" in a else "")
    if "gpm_at_cap_trajectory" in a:
        print(f"    experts_traj={a['experts_trajectory']}")
        print(f"    gpm_at_cap_traj={a['gpm_at_cap_trajectory']} mean_cap_traj={a['gpm_mean_cap_trajectory']}")

print("\n=== 3. BANKING77 MEANING PROBE ===")
mr = s["meaning_references"]
print(f"REFS (n_tr={mr['n_train']}, n_te={mr['n_test']}, chance={mr['chance']:.4f}, maj={mr['majority']:.4f}):")
print(f"  bag_of_bytes   : acc={mr['bag_of_bytes']['intent_77_acc']:.4f} auc={mr['bag_of_bytes']['same_vs_overlap_diff_auc']:.4f} sim_pos={mr['bag_of_bytes']['sim_same_intent']:.4f} sim_neg={mr['bag_of_bytes']['sim_diff_high_overlap']:.4f}")
print(f"  Cog_untrained  : acc={mr['Cog_untrained_acc']['mean']:.4f}+/-{mr['Cog_untrained_acc']['std']:.4f} auc={mr['Cog_untrained_auc']['mean']:.4f}")
print(f"  Dense_untrained: acc={mr['Dense_untrained_acc']['mean']:.4f}+/-{mr['Dense_untrained_acc']['std']:.4f} auc={mr['Dense_untrained_auc']['mean']:.4f}")
for arm in ARMS:
    mb = s["arms"][arm]["meaning_banking77"]
    print(f"  {arm:15s}: acc={mb['intent_77_acc']['mean']:.4f}+/-{mb['intent_77_acc']['std']:.4f} auc={mb['same_vs_overlap_diff_auc']['mean']:.4f}+/-{mb['same_vs_overlap_diff_auc']['std']:.4f} sim_pos={mb['sim_same_intent']['mean']:.4f} sim_neg={mb['sim_diff_high_overlap']['mean']:.4f}")

print("\n=== 4. CONTEXT SHUFFLE & NOISE10 ROBUSTNESS ===")
for arm in ARMS:
    cn = s["arms"][arm]["ctx_noise"]
    line = " | ".join(f"{d}:cln={cn[d]['suffix_clean_bpb']['mean']:.3f},+shuf={cn[d]['delta_shuffle']['mean']:+.3f},+n10={cn[d]['delta_noise10']['mean']:+.3f}" for d in ["wiki", "story", "code", "math"])
    print(f"  {arm:17s} | {line}")

print("\n=== 5. COMPOSITIONAL GENERALIZATION & ARITHMETIC ===")
ab = s["arms"]["Cog_Full"]["compositional"]["arith_baselines"]
print(f"arith_n={ab['arith_n']} n_small={ab['arith_n_small']} copy_first={ab['copy_first']:.4f} majority={ab['majority']:.4f}")
for arm in ARMS:
    c = s["arms"][arm]["compositional"]
    print(f"  {arm:17s} | arith={c['arith_exact_match']['mean']:.4f} small={c['arith_exact_small']['mean']:.4f} gsm_le2={c['gsm_simple_le2_bpb']['mean']:.3f} gsm_ge4={c['gsm_multi_ge4_bpb']['mean']:.3f} bind2_ex={c['var_binding_exact2']['mean']:.4f} bind_bpb={c['var_binding_2byte_bpb']['mean']:.3f} vs_shuf={c['var_binding_shuffled_1st_byte_bpb']['mean']:.3f} code_sh={c['code_shallow_bpb']['mean']:.3f} deep={c['code_deep_bpb']['mean']:.3f}")

print("\n=== 6. COGNITIVE INTERNALS (Thinking, World Model, Surprise, LTM, PC, Revisit) ===")
for arm in ["Cog_Full", "Cog_NoGPM", "Cog_SingleExpert", "Cog_SPGC", "Cog_Mixed"]:
    a = s["arms"][arm]
    print(f"--- {arm} ---")
    if "thinking_ablation" in a:
        ta = {d: (round(a["thinking_ablation"][d]["delta_adp_minus_0"]["mean"], 4), round(a["thinking_ablation"][d]["mean_steps_adaptive"]["mean"], 3), round(a["thinking_ablation"][d]["surprise_nll_corr"]["mean"], 3)) for d in DOMS}
        print("  think (dBPB_adp-0, steps, surp_corr):", ta)
    if "eval_internals" in a:
        ei = {d: (round(a["eval_internals"][d]["wm_cos_1step"]["mean"], 3), round(a["eval_internals"][d]["pc_trigger_frac"]["mean"], 5), round(a["eval_internals"][d]["surprise_p10"]["mean"], 3), round(a["eval_internals"][d]["surprise_p50"]["mean"], 3)) for d in DOMS}
        print("  eval_internals (wm_cos, pc_trig, p10, p50):", ei)
    if "ltm_and_pc" in a:
        lp = a["ltm_and_pc"]
        print(f"  ltm: no={lp['no_ltm']['mean']:.4f} match={lp['matched_ltm']['mean']:.4f} dist={lp['distractor_ltm']['mean']:.4f} slots={lp['slots_added']['mean']:.2f} | pc_clean={lp['pc_clean_trig']['mean']:.5f} pc_contra={lp['pc_contra_trig']['mean']:.5f}")
    if "revisit" in a:
        rv = a["revisit"]
        print(f"  revisit: spawns={rv['spawns_on_revisit']['mean']:.2f} dBPB={{d: round(rv['delta_bpb'][d]['mean'], 3) for d in DOMS}}".replace("{d: round(rv['delta_bpb'][d]['mean'], 3) for d in DOMS}", str({d: round(rv["delta_bpb"][d]["mean"], 3) for d in DOMS})))

print("\n=== 7. GENERATION STATS (Greedy & T=0.8) ===")
for arm in ARMS:
    a = s["arms"][arm]
    if "gen_stats_greedy" not in a:
        continue
    gs = {d: (round(a["gen_stats_greedy"][d]["distinct_4gram_frac"]["mean"], 3), round(a["gen_stats_greedy"][d]["max_4gram_repeat"]["mean"], 1), round(a["gen_stats_greedy"][d]["det_identical_rate"], 2), round(a["gen_stats_greedy"][d]["utf8_valid_rate"], 2)) for d in DOMS}
    ts = {d: (round(a["gen_stats_temp08"][d]["distinct_4gram_frac"]["mean"], 3), round(a["gen_stats_temp08"][d]["max_4gram_repeat"]["mean"], 1), round(a["gen_stats_temp08"][d]["utf8_valid_rate"], 2)) for d in DOMS}
    print(f"  {arm:15s} greedy(d4,rep,det,utf8):", gs)
    print(f"  {arm:15s} temp08(d4,rep,utf8)    :", ts)
