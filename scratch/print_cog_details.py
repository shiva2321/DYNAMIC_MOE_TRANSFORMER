import json
import pathlib

res = pathlib.Path("results/real_pilot_v1/test_v1")
DOMS = ["wiki", "story", "code", "math", "forum"]

print("=== Cog_Full per-seed trajectories ===")
for sd in [0, 1, 2]:
    d = json.loads((res / f"seed{sd}_Cog_Full.json").read_text())
    print(f"-- Seed {sd} (nmi={d['routing']['nmi_domain_vs_expert']:.4f}, final_mean={sum(d['matrix'][-1][k]['bpb'] for k in DOMS)/5:.3f}) --")
    for row in d["matrix"]:
        gpm = row.get("_gpm_summary", {})
        bpbs = {k: round(row[k]["bpb"], 3) for k in DOMS}
        print(f"  after {row['_after_stream']:5s} exp={row['_num_experts']} cap={gpm.get('at_cap_count',0)}/{gpm.get('num_projected',0)} ({gpm.get('mean_rank_ratio_of_cap',0):.3f}): {bpbs}")
    rv = d["revisit_probe"]
    print("  revisit spawns:", rv["spawns_on_revisit"],
          "before:", {k: round(rv["bpb_before"][k], 3) for k in DOMS},
          "after:", {k: round(rv["bpb_after"][k], 3) for k in DOMS})

print("\n=== Cog_Full s0 T=0.8 & greedy samples (story, code, math, wiki, forum) ===")
d0 = json.loads((res / "seed0_Cog_Full.json").read_text())
for dom in DOMS:
    for k in range(2):
        g = d0["generation"][dom][k]
        print(f"  [{dom} #{k}] route={g['routed_experts_sampled']}")
        print(f"    P : {g['prompt']!r}")
        print(f"    G : {g['generated_greedy']!r}")
        print(f"    T8: {g['generated_temp08']!r}")
