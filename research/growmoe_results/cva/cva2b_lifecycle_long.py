"""CVA-2B: Long-stream lifecycle verification (dormant, archive, revive).

VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)

Stream: growmoe.streams.s2_permuted (30 regimes, 150 segments, ~470 batches of 64).
Arms:
  LIFE_ON : dormant_after=60, archive_after=120, merge_every=50
  LIFE_OFF: dormant_after=10**9, archive_after=10**9, merge_every=10**9

Seeds: 10-14.
"""

import json
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from growmoe.accounting import pool_report
from growmoe.config import PhyloCfg
from growmoe.fairfight import run_fast
from growmoe.pools.phylo import PhyloMoE
from growmoe.streams import s2_permuted

torch.set_num_threads(3)
SEEDS = [10, 11, 12, 13, 14]
OUT_DIR = Path("research/growmoe_results/cva")


class MonitoredPhyloMoE(PhyloMoE):
    """Subclass of PhyloMoE that tracks expert states every 25 observe calls."""

    def __init__(self, cfg: PhyloCfg) -> None:
        super().__init__(cfg)
        self.timeline: List[dict] = []
        self.dormant_transitions: int = 0
        self.max_non_active: int = 0
        self._prev_states: Dict[int, str] = {}

    def observe(self, x, y=None, ctx=None):
        res = super().observe(x, y=y, ctx=ctx)

        # Track dormant transitions and non-active counts
        non_active = 0
        for eid, rec in self.experts.items():
            prev = self._prev_states.get(eid)
            if prev == "active" and rec.state == "dormant":
                self.dormant_transitions += 1
            self._prev_states[eid] = rec.state
            if rec.state != "active":
                non_active += 1

        if non_active > self.max_non_active:
            self.max_non_active = non_active

        # Snapshot every 25 observe calls
        if self.step % 25 == 0:
            st_counts = {"infant": 0, "active": 0, "dormant": 0, "archived": 0}
            for exp in self.experts.values():
                if exp.state in st_counts:
                    st_counts[exp.state] += 1
                else:
                    st_counts[exp.state] = 1
            rep = pool_report(self, self.cfg.d)
            self.timeline.append({
                "call": self.step,
                "n_experts": len(self.experts),
                "states": st_counts,
                "stored_vectors": rep.stored_vectors,
            })
        return res


def build_schedule(seed: int) -> List[int]:
    """list(range(30)) followed by four more passes over all 30 regimes."""
    sched = list(range(30))
    for p in range(4):
        rng = np.random.default_rng([seed, 0x5343, p])
        sched.extend(rng.permutation(30).tolist())
    return sched


def run_arm(seed: int, arm_name: str) -> Tuple[object, dict]:
    sched = build_schedule(seed)
    stream = s2_permuted(seed, n_regimes=30, n_train=200, n_test=100, schedule=sched)

    if arm_name == "LIFE_ON":
        cfg = PhyloCfg(
            seed=seed,
            d=stream.d,
            n_classes=stream.n_classes,
            inner_steps=20,
            q=0.02,
            alpha_spawn=0.01,
            m_min=32,
            route_rule="accept_then_logf",
            dormant_after=60,
            archive_after=120,
            merge_every=50,
        )
    else:  # LIFE_OFF
        cfg = PhyloCfg(
            seed=seed,
            d=stream.d,
            n_classes=stream.n_classes,
            inner_steps=20,
            q=0.02,
            alpha_spawn=0.01,
            m_min=32,
            route_rule="accept_then_logf",
            dormant_after=10**9,
            archive_after=10**9,
            merge_every=10**9,
        )

    holder = []

    def make_pool():
        p = MonitoredPhyloMoE(cfg)
        holder.append(p)
        return p

    fast_res = run_fast(make_pool, stream, seed=seed, eval_every="final")
    pool = holder[0]

    # Gather event counts
    events_by_kind = {}
    for ev in pool.events:
        events_by_kind[ev.kind] = events_by_kind.get(ev.kind, 0) + 1

    # Include tracked dormant transitions
    events_by_kind["dormant_transitions"] = pool.dormant_transitions

    rep = pool_report(pool, stream.d)

    rec = {
        "seed": seed,
        "arm": arm_name,
        "acc": float(fast_res.acc),
        "n_experts": len(pool.experts),
        "events": events_by_kind,
        "params": rep.params,
        "stored_vectors": rep.stored_vectors,
        "memory_bytes": rep.memory_bytes,
        "gate_macs": rep.gate_macs,
        "max_non_active": pool.max_non_active,
        "timeline": pool.timeline,
        "wall_clock": round(fast_res.wall_clock, 2),
    }

    return pool, rec


def check_invariants(pool: MonitoredPhyloMoE) -> Tuple[bool, List[str]]:
    """Criterion (c):
    Every archived expert has replay None and optimiser None, still has its signature,
    and an archived expert that was revived kept its expert id.
    """
    issues = []
    for eid, rec in pool.experts.items():
        if rec.state == "archived":
            if rec.replay is not None:
                issues.append(f"Expert {eid} is archived but replay is not None")
            if rec.opt is not None:
                issues.append(f"Expert {eid} is archived but opt is not None")
            if rec.meta.get("sig") is None:
                issues.append(f"Expert {eid} is archived but signature is None")

    # Check that revived experts kept their expert id
    revive_events = pool.events.of_kind("revive")
    for ev in revive_events:
        rev_eid = ev.fields.get("expert")
        if rev_eid is not None and rev_eid not in pool.experts:
            issues.append(f"Revived expert {rev_eid} not found in pool.experts")

    return len(issues) == 0, issues


def main():
    t_start = time.time()
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Starting CVA-2B Lifecycle Evaluation")

    life_on_runs = []
    life_off_runs = []
    pools_on = {}

    for s in SEEDS:
        print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Running Seed {s} LIFE_ON")
        pool_on, rec_on = run_arm(s, "LIFE_ON")
        life_on_runs.append(rec_on)
        pools_on[s] = pool_on
        print(
            f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): "
            f"Seed {s} LIFE_ON: acc={rec_on['acc']*100:.2f}%, experts={rec_on['n_experts']}, "
            f"archive={rec_on['events'].get('archive', 0)}, revive={rec_on['events'].get('revive', 0)}, "
            f"dormant_trans={rec_on['events'].get('dormant_transitions', 0)} ({rec_on['wall_clock']}s)"
        )

        print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Running Seed {s} LIFE_OFF")
        _, rec_off = run_arm(s, "LIFE_OFF")
        life_off_runs.append(rec_off)
        print(
            f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): "
            f"Seed {s} LIFE_OFF: acc={rec_off['acc']*100:.2f}%, experts={rec_off['n_experts']} ({rec_off['wall_clock']}s)"
        )

    # Determinism run: repeat LIFE_ON on seed 10
    print("VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Running Determinism Repeat (Seed 10 LIFE_ON)")
    _, rec_on_repeat = run_arm(10, "LIFE_ON")
    det_acc_match = rec_on_repeat["acc"] == life_on_runs[0]["acc"]
    det_events_match = rec_on_repeat["events"] == life_on_runs[0]["events"]
    print(
        f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): "
        f"Seed 10 Determinism: acc_match={det_acc_match}, events_match={det_events_match}"
    )

    # Analyze Predeclared Criteria:
    # (a) dormant-state changes, archive and revive each happen at least once in LIFE_ON
    dormant_happened = all(r["events"].get("dormant_transitions", 0) >= 1 for r in life_on_runs)
    archive_happened = all(r["events"].get("archive", 0) >= 1 for r in life_on_runs)
    revive_happened = all(r["events"].get("revive", 0) >= 1 for r in life_on_runs)
    crit_a_pass = dormant_happened and archive_happened and revive_happened

    # (b) Mean paired accuracy LIFE_ON - LIFE_OFF >= -1.0 point (-0.01 in float)
    acc_diffs_pts = [(r_on["acc"] - r_off["acc"]) * 100.0 for r_on, r_off in zip(life_on_runs, life_off_runs)]
    mean_diff_pts = float(np.mean(acc_diffs_pts))
    crit_b_pass = mean_diff_pts >= -1.0

    # (c) Invariants hold in LIFE_ON at the end
    inv_results = {s: check_invariants(pools_on[s]) for s in SEEDS}
    crit_c_pass = all(ok for ok, _ in inv_results.values())

    # (d) Memory saved: stored_vectors and memory_bytes as share of LIFE_OFF
    stored_ratios = [r_on["stored_vectors"] / r_off["stored_vectors"] for r_on, r_off in zip(life_on_runs, life_off_runs)]
    mem_ratios = [r_on["memory_bytes"] / r_off["memory_bytes"] for r_on, r_off in zip(life_on_runs, life_off_runs)]
    mean_stored_share = float(np.mean(stored_ratios))
    mean_mem_share = float(np.mean(mem_ratios))

    # Compile outputs
    out_data = {
        "metadata": {
            "title": "CVA-2B: Long-stream Lifecycle Verification",
            "stream": "s2_permuted(30 regimes, 150 segments, n_train=200, n_test=100)",
            "seeds": SEEDS,
            "arms": {
                "LIFE_ON": "dormant_after=60, archive_after=120, merge_every=50",
                "LIFE_OFF": "dormant_after=10**9, archive_after=10**9, merge_every=10**9",
            },
        },
        "criteria": {
            "crit_a_lifecycle_events": {
                "pass": crit_a_pass,
                "dormant_happened": dormant_happened,
                "archive_happened": archive_happened,
                "revive_happened": revive_happened,
            },
            "crit_b_paired_accuracy": {
                "pass": crit_b_pass,
                "mean_paired_diff_points": mean_diff_pts,
                "paired_diffs_by_seed": dict(zip(SEEDS, acc_diffs_pts)),
                "threshold": -1.0,
            },
            "crit_c_invariants": {
                "pass": crit_c_pass,
                "details_by_seed": {s: {"ok": ok, "issues": issues} for s, (ok, issues) in inv_results.items()},
            },
            "crit_d_memory_report": {
                "mean_stored_vectors_share": mean_stored_share,
                "mean_memory_bytes_share": mean_mem_share,
            },
            "determinism": {
                "pass": det_acc_match and det_events_match,
                "acc_match": det_acc_match,
                "events_match": det_events_match,
            },
        },
        "life_on_runs": life_on_runs,
        "life_off_runs": life_off_runs,
    }

    with open(OUT_DIR / "cva2b.json", "w") as f:
        json.dump(out_data, f, indent=2)

    # Build Markdown
    md_lines = [
        "# CVA-2B Results: Long-stream Lifecycle Verification",
        "",
        "VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "Stream: s2_permuted (30 regimes, 150 segments of 200 samples = 30,000 samples, ~470 batches of 64)",
        f"Seeds: {SEEDS}",
        "",
        "---",
        "",
        "## Summary of Predeclared Criteria",
        "",
        "| Criterion | Description | Value / Measurement | Threshold | Verdict |",
        "|:---:|:---|:---:|:---:|:---:|",
        f"| **(a)** | Lifecycle events occur (dormant, archive, revive) | Dormant: {dormant_happened}, Archive: {archive_happened}, Revive: {revive_happened} | >= 1 each | **{'PASS' if crit_a_pass else 'FAIL'}** |",
        f"| **(b)** | Mean paired accuracy (LIFE_ON - LIFE_OFF) | {mean_diff_pts:+.2f} percentage points | >= -1.0 pt | **{'PASS' if crit_b_pass else 'FAIL'}** |",
        f"| **(c)** | End-of-run invariants (archived opt/replay=None, sig kept, revived keeps eid) | All invariants hold on all seeds | Strictly hold | **{'PASS' if crit_c_pass else 'FAIL'}** |",
        f"| **(d)** | Memory saved (share of LIFE_OFF) | Stored vectors: {mean_stored_share*100:.1f}%, Memory bytes: {mean_mem_share*100:.1f}% | Report only | **REPORT** |",
        f"| **Det** | Determinism repeat on Seed 10 | Acc match: {det_acc_match}, Events match: {det_events_match} | Identical | **{'PASS' if (det_acc_match and det_events_match) else 'FAIL'}** |",
        "",
        "---",
        "",
        "## Paired Performance Comparison",
        "",
        "| Seed | Acc (LIFE_ON) | Acc (LIFE_OFF) | Paired Diff (pts) | Experts (ON / OFF) | Stored Vecs (ON / OFF) | Memory MB (ON / OFF) | Gate MACs (ON / OFF) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r_on, r_off in zip(life_on_runs, life_off_runs):
        diff = (r_on["acc"] - r_off["acc"]) * 100.0
        mb_on = r_on["memory_bytes"] / (1024 * 1024)
        mb_off = r_off["memory_bytes"] / (1024 * 1024)
        md_lines.append(
            f"| {r_on['seed']} | {r_on['acc']*100:.2f}% | {r_off['acc']*100:.2f}% | {diff:+.2f} pt | "
            f"{r_on['n_experts']} / {r_off['n_experts']} | {r_on['stored_vectors']} / {r_off['stored_vectors']} | "
            f"{mb_on:.2f} MB / {mb_off:.2f} MB | {r_on['gate_macs']} / {r_off['gate_macs']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Lifecycle Event Counts (LIFE_ON)",
        "",
        "| Seed | Spawns | Extends | Alarms | Dormant Transitions | Archives | Revives | Merges | Max Non-Active |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r_on in life_on_runs:
        ev = r_on["events"]
        md_lines.append(
            f"| {r_on['seed']} | {ev.get('spawn', 0)} | {ev.get('extend', 0)} | {ev.get('alarm', 0)} | "
            f"{ev.get('dormant_transitions', 0)} | {ev.get('archive', 0)} | {ev.get('revive', 0)} | "
            f"{ev.get('merge', 0)} | {r_on['max_non_active']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## Timeline Snapshots (Every 25 observe calls, Seed 10 LIFE_ON)",
        "",
        "| Call | Experts | Infant | Active | Dormant | Archived | Stored Vectors |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for snap in life_on_runs[0]["timeline"]:
        st = snap["states"]
        md_lines.append(
            f"| {snap['call']} | {snap['n_experts']} | {st.get('infant', 0)} | {st.get('active', 0)} | "
            f"{st.get('dormant', 0)} | {st.get('archived', 0)} | {snap['stored_vectors']} |"
        )

    with open(OUT_DIR / "CVA2B_RESULTS.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Saved cva2b.json and CVA2B_RESULTS.md to {OUT_DIR}")
    print(f"VERIFICATION (criteria fixed in advance by Claude, 2026-10-14): Total runtime = {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()
