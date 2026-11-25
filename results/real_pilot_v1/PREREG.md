# Pre-Registration: `real_pilot_v1` (End-to-End Real Multi-Domain Pilot)

**Frozen Timestamp (UTC):** 2026-11-19T19:22:45.558261+00:00  
**Status:** FROZEN BEFORE TEST RUN (`--mode test`)  
**Script:** `experiments/run_real_pilot.py` + `experiments/real_pilot_data.py`

## 1. Experimental Matrix (9 Arms, Matched Protocol)
- **Sequential 5-Domain Stream Order:** `wiki` -> `story` -> `code` -> `math` -> `forum` (byte-level UTF-8, `SEQ_LEN=128`, `TEST_SEQS=256` fixed non-overlapping windows per domain, `MAX_TRAIN_BYTES=3,000,000`).
- **Training Budget:** `400` optimizer steps per domain (`2,000` steps total per run; `batch=16`, `4,096,000` training bytes seen per run).
- **Arms:**
  1. `Cog_Full` (`RealSelfStabilizingDMoE` + `NullSpaceProjector` (`gpm_energy=0.97`, `max_rank_ratio=0.85`, `freeze_shared_norms_on_update=True`, `freeze_unprojected_shared=True`) + `align_wm_target_with_encoder=True` + `compute_total_loss` thinking calibration)
  2. `Dense_NoReplay` (`CausalDenseRoPEBaseline`, `d_model=192`, `layers=6`, `heads=6`, matched to initial Cog param count `5.29M`)
  3. `Dense_Replay` (`CausalDenseRoPEBaseline` with equal-memory reservoir buffer: `512` windows (~66 KB), `replay_frac=0.25`)
  4. `Dense_Large` (`CausalDenseRoPEBaseline` parameter-matched to grown 5-expert `Cog_Full`, `10.23M`)
  5. `Cog_NoGPM` (Ablation: identical to `Cog_Full` with `gpm=None`)
  6. `Cog_SingleExpert` (Ablation: identical to `Cog_Full` with `auto_lifecycle=False`, no GPM)
  7. `Dense_Joint` (Upper-bound mixed stream baseline: `2,000` steps uniform mixture across all 5 domains)
  8. `Cog_Mixed` (Stream-regime stress test: `Cog_Full` on uniform mixed stream across all 5 domains)
  9. `Cog_SPGC` (`ContinualLearner(enable_spgc=True, spgc_lr=5e-4, spgc_auto_consolidate_steps=400)` + attached GPM on sequential stream)
- **Disjoint Test Seeds:** `[0, 1, 2]` (3 seeds; cost-justified per `research-protocol.md` §6 because each full 9-arm seed requires ~30 GPU minutes on RTX 3060 12GB -> ~90 minutes total. Per `research-protocol.md` §9, claims from this 3-seed pilot are capped at `OBSERVED`, never `GENERALIZES`).

## 2. Frozen Hyperparameters
- `PROTO`: `{"batch": 16, "lr": 0.001, "wd": 0.01, "clip": 1.0, "warmup": 30, "wm_weight": 0.15}` + `gpm_energy=0.97`, `spgc_lr=0.0005`
- `COG_CFG`: `{"d_model": 192, "num_heads": 6, "d_ff": 512, "initial_layers": 6, "initial_experts": 1, "chunk_size": 1, "enable_memory": true, "num_ltm_slots": 32, "enable_three_level_attn": true, "local_window": 16, "chunk_stride": 8, "enable_layer_stm": true, "signed_beta": true, "use_rope": true, "causal_attn": true, "auto_write_ltm": false, "enable_thinking": true, "max_thinking_steps": 2, "skip_threshold": 0.08, "theta_halt": 0.1, "enable_world_model": true, "wm_rollout_horizon": 1, "enable_pc_rectification": true, "pc_thresh": 0.65, "hybrid_mode": true, "router_mode": "causal_proto", "num_router_protos": 64, "router_sim_threshold": 0.55, "auto_lifecycle": true, "max_experts": 6, "min_mature_steps": 40, "novel_confirm_steps": 3, "novel_surprise_threshold": 0.45, "dormant_age": 1000000000, "prune_age": 1000000000, "route_during_train": true, "use_trunk_routing": true, "router_context_blend": 0.2, "extend_loss_margin": 0.25, "align_wm_target_with_encoder": true}`

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
