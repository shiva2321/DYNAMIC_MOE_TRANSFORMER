"""Print verbatim sample generations, routing confusion matrices, and lifecycle events from test_v1."""
from __future__ import annotations

import json
import pathlib

RES = pathlib.Path("results/real_pilot_v1/test_v1")
DOMS = ["wiki", "story", "code", "math", "forum"]

for arm in ["Cog_Full", "Cog_NoGPM", "Cog_SPGC"]:
    print(f"\n=== LIFECYCLE & ROUTING ({arm}) ===")
    for s in [0, 1, 2]:
        d = json.loads((RES / f"seed{s}_{arm}.json").read_text())
        ev = [e for e in d["events"] if not e["event"].startswith("extended_") and e["event"] != "local_nuance_detected"]
        loc_cnt = sum(1 for e in d["events"] if e["event"] == "local_nuance_detected")
        ext_cnt = sum(1 for e in d["events"] if e["event"].startswith("extended_"))
        print(f"  s{s} (ext={ext_cnt}, local_nuance={loc_cnt}):", ev)
        if s == 0:
            print("  s0 routing_conf:", d["routing"]["confusion_domain_x_expert"])
            print("  s0 nmi:", d["routing"]["nmi_domain_vs_expert"])
            if "revisit_probe" in d:
                print("  s0 revisit_routes:", d["revisit_probe"]["revisit_train_routes"])
                print("  s0 revisit_events:", d["revisit_probe"]["revisit_events"][:10])
            print("  s0 banking77_routes:", d["meaning_probe_banking77"]["trained"].get("test_expert_routes"))

print("\n=== VERBATIM GENERATIONS (seed 0, sample 0 per domain) ===")
for arm in ["Cog_Full", "Dense_NoReplay", "Dense_Replay", "Cog_Mixed", "Dense_Joint"]:
    d = json.loads((RES / f"seed0_{arm}.json").read_text())
    print(f"\n--- {arm} (seed 0) ---")
    for dom in DOMS:
        g = d["generation"][dom][0]
        print(f"  [{dom}] route={g['routed_experts_sampled']}")
        print(f"    PROMPT : {g['prompt']!r}")
        print(f"    GREEDY : {g['generated_greedy']!r}")
        print(f"    TEMP0.8: {g['generated_temp08']!r}")
