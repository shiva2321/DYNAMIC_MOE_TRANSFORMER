import json
import pathlib
import sys

p = pathlib.Path(sys.argv[1])
seed = sys.argv[2] if len(sys.argv) > 2 else "0"
for fp in sorted(p.glob(f"seed{seed}_*.json")):
    d = json.loads(fp.read_text())
    arm = d["arm"]
    print(f"\n==================== {arm} (sec={d['train_seconds']:.1f}, fwd={d.get('forward_passes')}, bwd={d.get('backward_passes')}, skip={d.get('skipped_steps')}) ====================")
    if "matrix" in d:
        for r in d["matrix"]:
            gpm = r.get("_gpm_summary", {})
            doms = ["wiki", "story", "code", "math", "forum"]
            b = " ".join(f"{k}={r[k]['bpb']:.3f}" for k in doms)
            extra = ""
            if "_num_experts" in r:
                extra = f" | exp={r['_num_experts']} states={r.get('_expert_states')} gpm={gpm.get('mean_rank_ratio_of_cap',0):.3f} cap={gpm.get('at_cap_count',0)}"
            print(f"  after {r['_after_stream']:5s}: {b}{extra}")
        r_last = d["matrix"][-1]
        if "surprise_p10_p50_p90" in r_last["wiki"]:
            print("  surprise_q [p10,p50,p90] & corr & pc_trig:")
            for k in ["wiki", "story", "code", "math", "forum"]:
                print(f"    {k:5s}: q={[round(v,3) for v in r_last[k].get('surprise_p10_p50_p90',[])]} corr={r_last[k].get('surprise_nll_corr'):.3f} pc_trig={r_last[k].get('pc_trigger_frac'):.4f} wm_cos={r_last[k].get('wm_cos_1step'):.3f}")
    if "events" in d:
        sp = [e for e in d["events"] if not e["event"].startswith("extended_")]
        ext_cnt = sum(1 for e in d["events"] if e["event"].startswith("extended_"))
        print(f"  events (non-ext, plus {ext_cnt} extended):", sp[:30])
    if "routing" in d:
        print("  routing:", d["routing"])
    if "thinking_ablation" in d:
        print("  thinking_ablation:", {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in d["thinking_ablation"].items()})
    if "ltm_and_pc_probe" in d:
        print("  ltm_pc:", d["ltm_and_pc_probe"])
    if "revisit_probe" in d:
        rv = d["revisit_probe"]
        print("  revisit:", {"spawns": rv["spawns_on_revisit"], "routes": rv["revisit_train_routes"], "delta_bpb": {k: round(v, 4) for k, v in rv["delta_bpb"].items()}})
    if "context_and_noise_probe" in d:
        print("  ctx_noise:", {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in d["context_and_noise_probe"].items()})
    if "compositional_probe" in d:
        print("  comp:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d["compositional_probe"].items()})
    if "meaning_probe_banking77" in d:
        print("  meaning:", d["meaning_probe_banking77"])
    if "generation" in d:
        print("  --- GENERATIONS (greedy sample 0 per domain) ---")
        for k in ["wiki", "story", "code", "math", "forum"]:
            g = d["generation"][k][0]
            print(f"    [{k}] det={g['deterministic_repeat_identical']} route={g['routed_experts_sampled']} stats={g['stats_greedy']}")
            print(f"      P: {g['prompt']!r}")
            print(f"      G: {g['generated_greedy']!r}")
