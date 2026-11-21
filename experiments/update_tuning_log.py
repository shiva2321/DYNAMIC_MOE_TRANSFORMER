"""Build results/<results_root>/tuning_log.json directly from dev run summary artifacts."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

from summarize_real_pilot import DOMAINS, summarize_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="real_pilot_v2")
    cli_args = ap.parse_args()
    pilot_dir = ROOT / "results" / cli_args.results_root
    trials: List[Dict[str, Any]] = []
    for run_dir in sorted(pilot_dir.glob("dev_*")):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "_meta.json"
        if not meta_path.exists():
            continue
        meta_raw = json.loads(meta_path.read_text()).get("meta", {})
        if meta_raw.get("eval_split") != "dev":
            continue
        summary = summarize_dir(run_dir)
        arm_entries: Dict[str, Any] = {}
        for arm, a in summary["arms"].items():
            mean_diag = (
                float(sum(a["diag_bpb_by_domain"][d]["mean"] for d in DOMAINS) / len(DOMAINS))
                if "diag_bpb_by_domain" in a
                else None
            )
            entry: Dict[str, Any] = {
                "train_seconds": a["train_seconds"]["mean"],
                "forward_passes": a["forward_passes"]["mean"],
                "ep_ltm_passes": a["ep_ltm_passes"]["mean"],
                "skipped_steps": a["skipped_steps"]["mean"],
                "mean_final_val_bpb": a["mean_final_bpb"]["mean"],
                "mean_diag_val_bpb": mean_diag,
                "final_val_bpb": {d: a["final_bpb_by_domain"][d]["mean"] for d in DOMAINS},
                "mean_forgetting_bpb": a.get("mean_forgetting_bpb", {}).get("mean"),
                "forgetting_by_domain": {
                    d: a["forgetting_by_domain"][d]["mean"] for d in DOMAINS[:-1]
                } if "forgetting_by_domain" in a else None,
                "experts_final": a.get("routing", {}).get("num_experts", a.get("num_experts", {})).get("mean"),
                "spawns": a.get("spawns_per_seed"),
                "nmi_domain_vs_expert": a.get("routing", {}).get("nmi", {}).get("mean"),
                "confusion_domain_x_expert": a.get("routing", {}).get("sample_confusion"),
                "gpm_mean_cap_ratio": a.get("gpm", {}).get("mean_ratio", {}).get("mean"),
                "gpm_at_cap_count": a.get("gpm", {}).get("at_cap_count", {}).get("mean"),
                "mean_token_steps": a.get("mean_token_steps_overall", {}).get("mean"),
                "thinking_delta_adp_minus_0": a.get("thinking_delta_mean", {}).get("mean"),
                "ltm_gain_vs_dist": a.get("ltm_and_pc", {}).get("gain_vs_dist", {}).get("mean"),
                "ltm_gain_vs_none": a.get("ltm_and_pc", {}).get("gain_vs_none", {}).get("mean"),
                "pc_clean_trig": a.get("ltm_and_pc", {}).get("pc_clean_trig", {}).get("mean"),
                "pc_contra_trig": a.get("ltm_and_pc", {}).get("pc_contra_trig", {}).get("mean"),
                "revisit_spawns": a.get("revisit", {}).get("spawns_on_revisit", {}).get("mean"),
                "banking77_acc": a.get("meaning_banking77", {}).get("intent_77_acc", {}).get("mean"),
                "banking77_auc": a.get("meaning_banking77", {}).get("same_vs_overlap_diff_auc", {}).get("mean"),
            }
            arm_entries[arm] = entry
        trials.append({
            "dev_run": run_dir.name,
            "seeds": meta_raw.get("seeds"),
            "eval_split": meta_raw.get("eval_split"),
            "steps_per_stream": meta_raw.get("steps_per_stream"),
            "cog_v6_cfg": meta_raw.get("cog_v6_cfg"),
            "v6_proto": meta_raw.get("v6_proto"),
            "arms": arm_entries,
        })

    out = {
        "experiment": cli_args.results_root,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": "All tuning/calibration runs strictly used disjoint dev validation splits; test_windows was never accessed.",
        "trials": trials,
    }
    out_path = pilot_dir / "tuning_log.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"Wrote {out_path} ({len(trials)} dev trials)")


if __name__ == "__main__":
    main()

