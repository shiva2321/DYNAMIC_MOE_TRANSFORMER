"""Generate results/real_pilot_v1/tuning_log.json from dev runs on disk and write frozen PREREG.md."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from real_pilot_data import DOMAINS, MAX_TRAIN_BYTES, SEQ_LEN, TEST_SEQS
from run_real_pilot import COG_CFG, DENSE_D, DENSE_HEADS, DENSE_LAYERS, PROTO, REPLAY_FRAC, REPLAY_WINDOWS

RES_ROOT = ROOT / "results" / "real_pilot_v1"

DEV_DIRS = [
    ("smoke_v1", 0, 50, {"gpm_energy": 0.97, "spgc_lr": 0.002}),
    ("dev_steps200_g97", 100, 200, {"gpm_energy": 0.97, "spgc_lr": 0.0005}),
    ("dev_steps200_g90", 100, 200, {"gpm_energy": 0.90}),
    ("dev_steps400_g97", 100, 400, {"gpm_energy": 0.97}),
]

tuning_entries = []
for name, seed, steps, overrides in DEV_DIRS:
    d = RES_ROOT / name
    meta = json.loads((d / "_meta.json").read_text()) if (d / "_meta.json").exists() else {}
    arms_summary = {}
    for fp in sorted(d.glob(f"seed{seed}_*.json")):
        rec = json.loads(fp.read_text())
        arm = rec["arm"]
        if "matrix" in rec:
            final_bpbs = {dom: rec["matrix"][-1][dom]["bpb"] for dom in DOMAINS}
            peak_bpbs = {dom: min(row[dom]["bpb"] for row in rec["matrix"]) for dom in DOMAINS}
            mean_final = sum(final_bpbs.values()) / len(DOMAINS)
        else:
            final_bpbs = {dom: rec["final"][dom]["bpb"] for dom in DOMAINS}
            peak_bpbs = final_bpbs
            mean_final = sum(final_bpbs.values()) / len(DOMAINS)
        arms_summary[arm] = {
            "train_seconds": rec["train_seconds"],
            "mean_final_val_bpb": mean_final,
            "final_val_bpb": final_bpbs,
            "peak_val_bpb": peak_bpbs,
            "experts_final": rec["matrix"][-1].get("_num_experts") if "matrix" in rec else rec.get("num_experts"),
            "gpm_mean_cap_ratio": rec.get("gpm_ranks", {}).get("mean_rank_ratio_of_cap"),
            "gpm_at_cap_count": rec.get("gpm_ranks", {}).get("at_cap_count"),
            "nmi_domain_vs_expert": rec.get("routing", {}).get("nmi_domain_vs_expert"),
            "banking77_acc": rec.get("meaning_probe_banking77", {}).get("trained", {}).get("intent_77_acc"),
            "banking77_auc": rec.get("meaning_probe_banking77", {}).get("trained", {}).get("same_vs_overlap_diff_auc"),
        }
    tuning_entries.append({
        "dev_run": name,
        "seed": seed,
        "eval_split": meta.get("meta", {}).get("eval_split", "dev"),
        "steps_per_stream": steps,
        "overrides": overrides,
        "arms": arms_summary,
    })

tuning_log = {
    "experiment": "real_pilot_v1",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "note": "All tuning/calibration runs strictly used disjoint dev validation splits; test_windows was never accessed.",
    "trials": tuning_entries,
    "selection_rationale": {
        "gpm_energy": "Selected 0.97 (v5 default) over 0.90 because 0.97 achieved lower mean validation BPB on dev_steps200 (2.479 vs 2.513), half the math forgetting (+0.319 vs +0.627), and higher banking77 semantic discrimination AUC (0.369 vs 0.296).",
        "steps_per_stream": "Selected 400 steps/stream over 200 because at 400 steps Cog_Full separates wiki vs story into 5 specialists (1->5 experts instead of 4), keeps wiki/story ahead of Dense_Replay, Dense_Replay reaches multi-line Python & 7% gsm8k arithmetic, and dev_steps400 cleanly exposes real GPM rank-cap saturation (~255s Cog_Full train time, ~30 min/seed across all 9 arms).",
        "spgc_lr": "Adjusted from 0.002 (smoke_v1 overshoot) to 0.0005 with acquire->consolidate transition at step=400 to match transformer scale without instability.",
        "defaults_preserved_uncalibrated": "Preserved v5 COG_CFG defaults (theta_halt=0.10, skip_threshold=0.08, pc_thresh=0.65, auto_write_ltm=False, extend_loss_margin=0.25) intentionally rather than tuning them on dev so real failure modes and calibration gaps are measured honestly in test_v1.",
    },
}
(RES_ROOT / "tuning_log.json").write_text(json.dumps(tuning_log, indent=2))

prereg_md = f"""# Pre-Registration: `real_pilot_v1` (End-to-End Real Multi-Domain Pilot)

**Frozen Timestamp (UTC):** {datetime.now(timezone.utc).isoformat()}  
**Status:** FROZEN BEFORE TEST RUN (`--mode test`)  
**Script:** `experiments/run_real_pilot.py` + `experiments/real_pilot_data.py`

## 1. Experimental Matrix (9 Arms, Matched Protocol)
- **Sequential 5-Domain Stream Order:** `wiki` -> `story` -> `code` -> `math` -> `forum` (byte-level UTF-8, `SEQ_LEN={SEQ_LEN}`, `TEST_SEQS={TEST_SEQS}` fixed non-overlapping windows per domain, `MAX_TRAIN_BYTES={MAX_TRAIN_BYTES:,}`).
- **Training Budget:** `400` optimizer steps per domain (`2,000` steps total per run; `batch={PROTO['batch']}`, `4,096,000` training bytes seen per run).
- **Arms:**
  1. `Cog_Full` (`RealSelfStabilizingDMoE` + `NullSpaceProjector` (`gpm_energy=0.97`, `max_rank_ratio=0.85`, `freeze_shared_norms_on_update=True`, `freeze_unprojected_shared=True`) + `align_wm_target_with_encoder=True` + `compute_total_loss` thinking calibration)
  2. `Dense_NoReplay` (`CausalDenseRoPEBaseline`, `d_model={DENSE_D}`, `layers={DENSE_LAYERS}`, `heads={DENSE_HEADS}`, matched to initial Cog param count `5.29M`)
  3. `Dense_Replay` (`CausalDenseRoPEBaseline` with equal-memory reservoir buffer: `{REPLAY_WINDOWS}` windows (~66 KB), `replay_frac={REPLAY_FRAC}`)
  4. `Dense_Large` (`CausalDenseRoPEBaseline` parameter-matched to grown 5-expert `Cog_Full`, `10.23M`)
  5. `Cog_NoGPM` (Ablation: identical to `Cog_Full` with `gpm=None`)
  6. `Cog_SingleExpert` (Ablation: identical to `Cog_Full` with `auto_lifecycle=False`, no GPM)
  7. `Dense_Joint` (Upper-bound mixed stream baseline: `2,000` steps uniform mixture across all 5 domains)
  8. `Cog_Mixed` (Stream-regime stress test: `Cog_Full` on uniform mixed stream across all 5 domains)
  9. `Cog_SPGC` (`ContinualLearner(enable_spgc=True, spgc_lr=5e-4, spgc_auto_consolidate_steps=400)` + attached GPM on sequential stream)
- **Disjoint Test Seeds:** `[0, 1, 2]` (3 seeds; cost-justified per `research-protocol.md` §6 because each full 9-arm seed requires ~30 GPU minutes on RTX 3060 12GB -> ~90 minutes total. Per `research-protocol.md` §9, claims from this 3-seed pilot are capped at `OBSERVED`, never `GENERALIZES`).

## 2. Frozen Hyperparameters
- `PROTO`: `{json.dumps(PROTO)}` + `gpm_energy=0.97`, `spgc_lr=0.0005`
- `COG_CFG`: `{json.dumps(COG_CFG)}`

## 3. Null Controls & Baselines
- Per-domain `unigram` and `bigram` byte Markov models trained on train pools.
- `Untrained Cog_Full` and `Untrained Dense` baselines evaluated identically on test windows and probes.
- `bag_of_bytes_domain_accuracy_with_true_labels` (256-d byte histogram nearest-centroid classifier fit using ground-truth domain labels) as trivial router lower-bound reference against `Cog_Full` blind `nmi_domain_vs_expert`.
- `gsm8k` calculator `copy_first_operand` and `majority_string` baselines.
- Zero-shot variable binding `shuffled_binding` null control.
- LTM `distractor_ltm` null control vs `no_ltm` and `matched_ltm`.
- `banking77` `chance (1/77)`, `majority`, `bag_of_bytes`, `Cog_untrained`, and `Dense_untrained` baselines for linear intent probe and semantic-vs-overlap AUC.

## 4. Pre-Registered Primary Hypotheses & Explicit Refutation Criteria
- **H1 (Zero-Replay Continual Retention):** `Cog_Full` achieves substantially lower final mean test BPB and catastrophic forgetting than `Dense_NoReplay`, `Dense_Large`, and `Cog_NoGPM` (and outperforms `Cog_NoGPM` on prior domains, isolating GPM contribution).
  - *Refutation:* If `Cog_Full` forgetting matches/exceeds `Dense_NoReplay` or `Cog_NoGPM`, zero-replay protection fails on real data.
- **H2 (Blind Domain Specialization):** Under sequential shifts (`wiki -> story -> code -> math -> forum`), `Cog_Full` autonomously spawns specialists (`4–6` experts) with test-time `NMI > 0.35` without receiving domain labels, and `0` spurious spawns on post-horizon domain revisit (`revisit_probe`).
  - *Refutation:* If NMI <= `0.2` or revisit causes uncontrolled expert cloning (`spawns_on_revisit > 0`).
- **H3 (Known Limitations Explicitly Monitored Rather Than Hidden):** We pre-register checking (a) whether `Cog_Full` hits the `85%` GPM rank-cap ceiling by Domains 3–4 (degrading Domain 4–5 protection vs `Dense_Replay`), (b) whether `Cog_Mixed` fails to split experts (`exp=1`) under uniform domain mixing, (c) whether uncalibrated defaults (`theta_halt=0.10`, `pc_thresh=0.65`, `auto_write_ltm=False`) fail to trigger early halting, NL PC rectification, or LTM recall lift over distractor control, and (d) exact wall-clock/FLOP pass overhead (`~4.5x–5x` slower than Dense).
- **H4 (Cognitive Representations & Probes — Exploratory/`OBSERVED`):** Measure `thinking_ablation` delta & `surprise_nll_corr`, `context_shuffle` delta, `noise10` delta, `gsm8k` simple-vs-multistep BPB & exact match, `var_binding`, `banking77` 77-class accuracy & same-intent-vs-overlap AUC, and greedy/`T=0.8` sample generations across all 5 domains.
"""
(RES_ROOT / "PREREG.md").write_text(prereg_md)
print("Wrote tuning_log.json and PREREG.md successfully.")
