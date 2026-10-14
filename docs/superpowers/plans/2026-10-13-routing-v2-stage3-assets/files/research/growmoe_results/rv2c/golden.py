"""Flag-off fingerprints of PhyloMoE (routing v2 stage 3, spec statement S3-9: with both flags off the pool is bit-identical to what it was before stage 3).

    python research/growmoe_results/rv2c/golden.py record     # RV3-0 only: run on the code BEFORE any stage-3 edit; writes golden_flags_off.json
    python research/growmoe_results/rv2c/golden.py check [--quick]   # recompute (all 20 fingerprints, or the 3 of --quick) and compare; exit status 1 on any difference

A fingerprint is built from integers and booleans only (routed ids, novelty flags, expert counts, event kinds and steps, predicted classes), run single-threaded, so
it does not depend on float formatting. The recorded file holds a sha256 per (configuration, stream) and a few readable counts to help diagnose a mismatch.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                                            # noqa: E402

from growmoe.config import PhyloCfg                                                     # noqa: E402
from growmoe.pools.phylo import PhyloMoE                                                # noqa: E402
from growmoe.streams import s1_types, s2_permuted                                       # noqa: E402
from growmoe.streams_m1 import function_regimes                                         # noqa: E402

GOLDEN = HERE / "golden_flags_off.json"
BASE = dict(q=0.02, alpha_spawn=0.01, m_min=32, inner_steps=5)
if "set_gate" in PhyloCfg.__dataclass_fields__:                                          # after stage 3 exists: pin both flags OFF so this file keeps testing the flags-off pool
    BASE.update(set_gate=False, group_vote=False)                                       # even after a default is flipped (the recorded file is the same either way)
CONFIGS = {
    "default": dict(route_rule="accept_then_logf"),
    "default_rule": dict(),
    "no_vote": dict(route_rule="accept_then_logf", batch_vote=False),
    "exact_scan": dict(route_rule="accept_then_logf", gate_shortlist=0),
    "cascade_from_8": dict(route_rule="accept_then_logf", gate_shortlist=2, gate_fingerprint_k=1, gate_shortlist_min_experts=8),
}
STREAMS = {
    "OV-0.6-smoke": lambda: function_regimes(100, overlap=0.6, n_regimes=4, n_train=128, n_test=64),
    "F-smoke": lambda: function_regimes(101, n_regimes=4, n_train=128, n_test=64),
    "S2-smoke": lambda: s2_permuted(100, n_regimes=3, schedule=[0, 1, 2, 0], n_train=128, n_test=50),
    "S1-smoke": lambda: s1_types(100, n_train=100, n_test=50),
}


def _fingerprint(config: str, stream: str) -> dict:
    st = STREAMS[stream]()
    pool = PhyloMoE(PhyloCfg(seed=7, d=st.d, n_classes=st.n_classes, **BASE, **CONFIGS[config]))
    trace = []
    for x, y, ctx in st.algo_batches(64, 7):
        rec = pool.observe(x, y, ctx)
        trace.append([rec.assigned.tolist(), rec.novel.tolist(), int(rec.n_experts), list(rec.spawned)])
    answers = []
    for r, (x, _) in sorted(st.test_sets().items()):
        pred = pool.predict(x)
        answers.append([pred.probs.argmax(1).tolist(), pred.expert.tolist(), pred.unknown.tolist()])
    events = [[e.kind, int(e.step)] for e in pool.events]
    blob = json.dumps([trace, answers, events], sort_keys=True).encode("utf-8")
    return {"sha256": hashlib.sha256(blob).hexdigest(), "batches": len(trace), "experts": len(pool.experts), "events": len(events),
            "kinds": sorted({k for k, _ in events})}


QUICK = ["default|OV-0.6-smoke", "default|S1-smoke", "no_vote|S1-smoke"]


def compute(keys=None) -> dict:
    """Fingerprints of the given "config|stream" keys (all of them by default)."""
    keys = keys or [f"{c}|{s}" for c in CONFIGS for s in STREAMS]
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        return {k: _fingerprint(*k.split("|")) for k in keys}
    finally:
        torch.set_num_threads(threads)


def differences(keys=None) -> list:
    """Keys whose fingerprint differs from the recorded one (an empty list = bit-identical)."""
    recorded = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = compute(keys)
    return [k for k in sorted(now) if recorded.get(k, {}).get("sha256") != now[k]["sha256"]]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "record":
        rec = compute()
        GOLDEN.write_text(json.dumps(rec, indent=1, sort_keys=True), encoding="utf-8")
        for k, v in rec.items():
            print(f"{k:32s} batches {v['batches']:3d} experts {v['experts']:2d} events {v['events']:3d} {v['kinds']}")
        print(f"wrote {GOLDEN}")
    else:
        bad = differences(QUICK if "--quick" in sys.argv else None)
        print("bit-identical" if not bad else f"DIFFERENT: {bad}")
        sys.exit(1 if bad else 0)
