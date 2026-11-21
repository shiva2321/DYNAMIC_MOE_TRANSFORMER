"""Aggregate real_pilot JSON outputs cleanly across seeds without manual calculation.

Run: python experiments/summarize_real_pilot.py --dir results/real_pilot_v2/dev_base
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DOMAINS = ["wiki", "story", "code", "math", "forum"]


def _mean_std(vals: List[float]) -> Dict[str, Any]:
    clean = [float(v) for v in vals if v is not None and not np.isnan(v)]
    if not clean:
        return {"mean": None, "std": None, "vals": []}
    return {
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean)),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "vals": clean,
    }


def _aggregate_arm(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"n_seeds": len(records), "seeds": [r["seed"] for r in records]}
    out["train_seconds"] = _mean_std([r["train_seconds"] for r in records])
    out["forward_passes"] = _mean_std([r["forward_passes"] for r in records])
    out["backward_passes"] = _mean_std([r["backward_passes"] for r in records])
    out["ep_ltm_passes"] = _mean_std([r.get("ep_ltm_passes", 0) for r in records])
    out["skipped_steps"] = _mean_std([r.get("skipped_steps", 0) for r in records])
    out["peak_vram_mb"] = _mean_std([r.get("peak_vram_mb") for r in records])
    out["params_final"] = _mean_std([r.get("params_final") for r in records])

    is_mixed = "final" in records[0]
    out["is_mixed"] = is_mixed

    final_bpb_by_dom: Dict[str, List[float]] = {d: [] for d in DOMAINS}
    diag_bpb_by_dom: Dict[str, List[float]] = {d: [] for d in DOMAINS}
    forget_by_dom: Dict[str, List[float]] = {d: [] for d in DOMAINS[:-1]}
    mean_final_all, mean_forget_all = [], []
    mean_steps_dom: Dict[str, List[float]] = {d: [] for d in DOMAINS}
    pc_trig_dom: Dict[str, List[float]] = {d: [] for d in DOMAINS}
    spawns_per_seed: List[List[str]] = []

    for r in records:
        row_final = r["final"] if is_mixed else r["matrix"][-1]
        for di, d in enumerate(DOMAINS):
            fb = float(row_final[d]["bpb"])
            final_bpb_by_dom[d].append(fb)
            if row_final[d].get("mean_token_steps") is not None:
                mean_steps_dom[d].append(float(row_final[d]["mean_token_steps"]))
            if row_final[d].get("pc_trigger_frac") is not None:
                pc_trig_dom[d].append(float(row_final[d]["pc_trigger_frac"]))
            if not is_mixed:
                db = float(r["matrix"][di][d]["bpb"])
                diag_bpb_by_dom[d].append(db)
                if di < len(DOMAINS) - 1:
                    forget_by_dom[d].append(fb - db)
        mean_final_all.append(float(np.mean([row_final[d]["bpb"] for d in DOMAINS])))
        if not is_mixed:
            mean_forget_all.append(
                float(np.mean([row_final[d]["bpb"] - r["matrix"][di][d]["bpb"] for di, d in enumerate(DOMAINS[:-1])]))
            )
        spawns_per_seed.append(
            [f"{e['stream']}@{e['step']}:{e['event']}" for e in r.get("events", []) if "spawned_" in e["event"] or "mixed_split_" in e["event"]]
        )

    out["mean_final_bpb"] = _mean_std(mean_final_all)
    out["final_bpb_by_domain"] = {d: _mean_std(v) for d, v in final_bpb_by_dom.items()}
    if not is_mixed:
        out["diag_bpb_by_domain"] = {d: _mean_std(v) for d, v in diag_bpb_by_dom.items()}
        out["forgetting_by_domain"] = {d: _mean_std(v) for d, v in forget_by_dom.items()}
        out["mean_forgetting_bpb"] = _mean_std(mean_forget_all)
    if any(mean_steps_dom.values()):
        out["mean_token_steps_by_domain"] = {d: _mean_std(v) for d, v in mean_steps_dom.items()}
        out["mean_token_steps_overall"] = _mean_std([x for v in mean_steps_dom.values() for x in v])
    if any(pc_trig_dom.values()):
        out["pc_trigger_frac_by_domain"] = {d: _mean_std(v) for d, v in pc_trig_dom.items()}
    out["spawns_per_seed"] = spawns_per_seed

    if any("routing" in r for r in records):
        r_recs = [r["routing"] for r in records if "routing" in r]
        out["routing"] = {
            "nmi": _mean_std([x.get("nmi_domain_vs_expert") for x in r_recs]),
            "num_experts": _mean_std([
                (r["matrix"][-1]["_num_experts"] if "matrix" in r else r.get("num_experts"))
                for r in records
            ]),
            "sample_confusion": [x.get("confusion_domain_x_expert") for x in r_recs],
        }
    if any("num_experts" in r for r in records):
        out["num_experts"] = _mean_std([r.get("num_experts") for r in records if "num_experts" in r])
    if any("gpm_ranks" in r and r["gpm_ranks"] for r in records):
        g_recs = [r["gpm_ranks"] for r in records if r.get("gpm_ranks")]
        out["gpm"] = {
            "at_cap_count": _mean_std([g.get("at_cap_count") for g in g_recs]),
            "num_projected": _mean_std([g.get("num_projected") for g in g_recs]),
            "mean_ratio": _mean_std([g.get("mean_rank_ratio_of_cap") for g in g_recs]),
            "max_ratio": _mean_std([g.get("max_rank_ratio_of_cap") for g in g_recs]),
            "per_stage_means": [
                [m_row.get("_gpm_summary", {}).get("mean_rank_ratio_of_cap") for m_row in r.get("matrix", [])]
                for r in records
            ],
        }

    if any("thinking_ablation" in r for r in records):
        ta_recs = [r["thinking_ablation"] for r in records if "thinking_ablation" in r]
        ta_dom: Dict[str, Any] = {}
        for d in DOMAINS:
            ta_dom[d] = {
                "bpb_adaptive": _mean_std([x[d]["bpb_adaptive"] for x in ta_recs]),
                "bpb_think0": _mean_std([x[d]["bpb_think0"] for x in ta_recs]),
                "bpb_think2": _mean_std([x[d]["bpb_think2"] for x in ta_recs]),
                "delta_adp_minus_0": _mean_std([x[d]["delta_adp_minus_0"] for x in ta_recs]),
                "mean_steps_adaptive": _mean_std([x[d]["mean_steps_adaptive"] for x in ta_recs]),
                "surprise_nll_corr": _mean_std([x[d].get("surprise_nll_corr") for x in ta_recs]),
            }
        out["thinking_ablation"] = ta_dom
        out["thinking_delta_mean"] = _mean_std(
            [float(np.mean([x[d]["delta_adp_minus_0"] for d in DOMAINS])) for x in ta_recs]
        )

    if any("ltm_and_pc_probe" in r for r in records):
        lp = [r["ltm_and_pc_probe"] for r in records if "ltm_and_pc_probe" in r]
        out["ltm_and_pc"] = {
            "no_ltm": _mean_std([x["ltm_bpb_no_ltm"] for x in lp]),
            "matched_ltm": _mean_std([x["ltm_bpb_matched_ltm"] for x in lp]),
            "distractor_ltm": _mean_std([x["ltm_bpb_distractor_ltm"] for x in lp]),
            "gain_vs_dist": _mean_std([x["ltm_bpb_distractor_ltm"] - x["ltm_bpb_matched_ltm"] for x in lp]),
            "gain_vs_none": _mean_std([x["ltm_bpb_no_ltm"] - x["ltm_bpb_matched_ltm"] for x in lp]),
            "pc_clean_trig": _mean_std([x["pc_clean_pairs"]["pc_trigger_frac"] for x in lp]),
            "pc_contra_trig": _mean_std([x["pc_contradiction_pairs"]["pc_trigger_frac"] for x in lp]),
            "pc_clean_bpb": _mean_std([x["pc_clean_pairs"]["mean_bpb"] for x in lp]),
            "pc_contra_bpb": _mean_std([x["pc_contradiction_pairs"]["mean_bpb"] for x in lp]),
        }

    if any("revisit_probe" in r for r in records):
        rv = [r["revisit_probe"] for r in records if "revisit_probe" in r]
        out["revisit"] = {
            "spawns_on_revisit": _mean_std([x["spawns_on_revisit"] for x in rv]),
            "delta_bpb_by_dom": {d: _mean_std([x["delta_bpb"][d] for x in rv]) for d in DOMAINS},
            "routes_sample": rv[0].get("revisit_train_routes"),
        }

    if any("meaning_probe_banking77" in r for r in records):
        mp = [r["meaning_probe_banking77"]["trained"] for r in records if "meaning_probe_banking77" in r]
        out["meaning_banking77"] = {
            "intent_77_acc": _mean_std([x["intent_77_acc"] for x in mp]),
            "same_vs_overlap_diff_auc": _mean_std([x["same_vs_overlap_diff_auc"] for x in mp]),
        }

    if any("context_and_noise_probe" in r for r in records):
        cn = [r["context_and_noise_probe"] for r in records if "context_and_noise_probe" in r]
        out["context_and_noise"] = {
            d: {
                "clean_bpb": _mean_std([x[d]["suffix_clean_bpb"] for x in cn]),
                "shuffle_delta": _mean_std([x[d]["delta_shuffle"] for x in cn]),
                "noise10_delta": _mean_std([x[d]["delta_noise10"] for x in cn]),
            }
            for d in ("wiki", "story", "code", "math")
        }

    if any("compositional_probe" in r for r in records):
        cp = [r["compositional_probe"] for r in records if "compositional_probe" in r]
        out["compositional"] = {
            "var_bind_2byte_bpb": _mean_std([x["var_binding_2byte_bpb"] for x in cp]),
            "var_bind_shuffled_1st_bpb": _mean_std([x["var_binding_shuffled_1st_byte_bpb"] for x in cp]),
            "var_bind_em": _mean_std([x["var_binding_exact2"] for x in cp]),
            "gsm_simple_bpb": _mean_std([x["gsm_simple_le2_bpb"] for x in cp]),
            "gsm_multi_bpb": _mean_std([x["gsm_multi_ge4_bpb"] for x in cp]),
            "arith_em": _mean_std([x["arith_exact_match"] for x in cp]),
            "code_shallow_bpb": _mean_std([x["code_shallow_bpb"] for x in cp]),
            "code_deep_bpb": _mean_std([x["code_deep_bpb"] for x in cp]),
        }
    return out


def summarize_dir(run_dir: Path) -> Dict[str, Any]:
    meta_path = run_dir / "_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    by_arm: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted(run_dir.glob("seed*_*.json")):
        data = json.loads(p.read_text())
        by_arm.setdefault(data["arm"], []).append(data)
    arms_out = {arm: _aggregate_arm(recs) for arm, recs in by_arm.items()}
    return {"meta": meta.get("meta"), "null_and_trivial": meta.get("null_and_trivial"), "arms": arms_out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    summary = summarize_dir(args.dir)
    out_path = args.out if args.out is not None else args.dir / "summary.json"
    out_path.write_text(json.dumps(summary, indent=1))
    print(f"Wrote {out_path}")
    for arm, a in summary["arms"].items():
        bpb = a["mean_final_bpb"]["mean"]
        fg = a.get("mean_forgetting_bpb", {}).get("mean")
        nmi = a.get("routing", {}).get("nmi", {}).get("mean")
        sec = a["train_seconds"]["mean"]
        exp = a.get("routing", {}).get("num_experts", a.get("num_experts", {})).get("mean")
        fg_s = f"{fg:+.3f}" if fg is not None else "n/a"
        nmi_s = f"{nmi:.3f}" if nmi is not None else "n/a"
        exp_s = f"{exp:.1f}" if exp is not None else "n/a"
        print(f"  {arm:16s} | bpb={bpb:.3f} | forget={fg_s} | nmi={nmi_s} | exp={exp_s} | sec={sec:.1f}s")
        if "ltm_and_pc" in a:
            lp = a["ltm_and_pc"]
            print(
                f"    ltm_gain_dist={lp['gain_vs_dist']['mean']:+.4f} "
                f"ltm_gain_none={lp['gain_vs_none']['mean']:+.4f} "
                f"pc_clean={lp['pc_clean_trig']['mean']:.4f} pc_contra={lp['pc_contra_trig']['mean']:.4f}"
            )
        if "thinking_delta_mean" in a:
            st = a.get("mean_token_steps_overall", {}).get("mean")
            print(f"    think_delta(adp-0)={a['thinking_delta_mean']['mean']:+.4f} mean_steps={st}")


if __name__ == "__main__":
    main()
