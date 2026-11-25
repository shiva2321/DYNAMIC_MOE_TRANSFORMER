# Pre-Registration: `real_pilot_v2` (`v6` End-to-End Fixes & Efficiency Evaluation)

**Frozen Timestamp (UTC):** 2026-11-20T05:00:00Z  
**Status:** FROZEN BEFORE TEST RUN (`--mode test --seeds 0 1 2 --out_name test_v2`)  
**Script:** `experiments/run_real_pilot.py` + `experiments/real_pilot_data.py` + `experiments/summarize_real_pilot.py`

## 1. Headline Claim & Primary Metrics
- **Claim:** On real multi-domain byte streams (`wiki -> story -> code -> math -> forum`) and uniformly mixed streams, `Cog_v6` (`v6` GPM step-budgeted null-space projection with full-stack private adapters, zero-contamination quarantine deferral, IDF causal-prototype routing, marginal-energy halting, adaptive relative PC gating, and intra-window episodic LTM training) substantially lowers mean catastrophic forgetting BPB and final mean BPB compared to both `Cog_Full` (`v5`) and `Dense_NoReplay`/`Dense_Large`, eliminates shared-trunk GPM cap saturation (`0` layers at cap vs `83–106` in `v5`), activates selective halting/PC/LTM mechanisms, and enables co-active specialist emergence (`>1` experts, `NMI > 0.25`) under uniform domain mixing (`Cog_v6_Mixed`), where `Cog_Mixed` (`v5`) previously collapsed to `1` expert.
- **Headline Metrics:** Mean final held-out BPB (lower is better), mean forgetting BPB across prior domains (`wiki, story, code, math`, lower is better), test-time blind routing `NMI`, GPM layers at cap count, adaptive token steps & thinking delta vs `steps=0`, LTM matched vs distractor/none BPB, PC contradiction vs clean trigger fraction, wall-clock seconds, and forward/backward counts.
- **Explicit Falsifiers:**
  1. If `Cog_v6` fails to beat `Cog_Full (v5)`, `Dense_NoReplay`, and `Dense_Large` on mean final test BPB and mean forgetting BPB -> `v6` continual-fix hypothesis is refuted.
  2. If `Cog_v6` hits the GPM rank cap across >25% of monitored layers or `code` forgetting still exceeds `+0.50 BPB` -> GPM saturation fix is refuted.
  3. If `Cog_v6_Mixed` stays at `1` expert or `NMI <= 0.15`, or degrades final BPB vs `Cog_Mixed` (`v5`) -> mixed-stream co-active mitosis fix is refuted.
  4. If `thinking_delta (adaptive - think0) >= 0`, or `pc_contradiction_trig <= pc_clean_trig`, or `matched_ltm >= distractor_ltm`, report that specific mechanism as `NON-FUNCTIONAL / UNVERIFIED` honestly alongside successes.

## 2. Experimental Matrix (8 Arms, Matched Protocol Inside Same Script)
- **Sequential 5-Domain Stream Order:** `wiki -> story -> code -> math -> forum`, byte-level UTF-8 (`SEQ_LEN=128`, `TEST_SEQS=256` non-overlapping windows per domain, `MAX_TRAIN_BYTES=3,000,000`, disjoint held-out `test` split).
- **Training Budget:** `400` optimizer steps per stream (`2,000` total steps; `batch=16`, `4,096,000` training bytes seen).
- **Arms (`test_v2`):**
  1. `Cog_v6`: All `v6` fixes active (`cog_v6_params_init=5,354,021`, `cog_v6_params_5exp=10,529,637`).
  2. `Cog_v6_Mixed`: `Cog_v6` with `enable_mixed_split=True, router_center_multi=True, defer_on_novel_streak=False, preserve_input_novel_on_extend=False, max_coactive_experts=5` on uniformly mixed batches.
  3. `Cog_Full`: Re-run `v5` architecture baseline (`cog_params_init=5,293,397`, `cog_params_5exp=10,226,517`).
  4. `Cog_Mixed`: Re-run `v5` mixed baseline (`Cog_Full` on uniform mixed stream).
  5. `Dense_NoReplay`: `CausalDenseRoPEBaseline` (`d_ff=1856`, `5,281,792` params, parameter-matched to initial Cog).
  6. `Dense_Replay`: `CausalDenseRoPEBaseline` with equal-memory `512`-window reservoir replay (`replay_frac=0.25`).
  7. `Dense_Large`: `CausalDenseRoPEBaseline` parameter-matched to 5-expert `Cog_v6` (`d_ff=4080`, `10,419,232` params).
  8. `Dense_Joint`: Upper-bound uniform mixed-stream baseline (`d_ff=1856`, `2,000` steps).
- **Disjoint Seeds & Tuning Budget:**
  - Validation/tuning ran strictly on `eval_split="dev"` and `seed=100` across **4 logged dev trials** (`dev_t1_base`, `dev_t2_opt`, `dev_t3_final`, `dev_t4_chk` stored in `results/real_pilot_v2/tuning_log.json`). Held-out `test` windows were NEVER accessed during tuning.
  - Test run uses disjoint seeds **`[0, 1, 2]`** (3 seeds; claims strictly capped at `OBSERVED` per `research-protocol.md` §6 and §9 given `~70` GPU minutes on RTX 3060 12GB for 8 arms x 3 seeds).

## 3. Frozen Configuration (`COG_V6_CFG` + `V6_PROTO` + `PROTO`)
- `PROTO`: `{"batch": 16, "lr": 0.001, "wd": 0.01, "clip": 1.0, "warmup": 30, "wm_weight": 0.15, "spgc_lr": 0.0005, "gpm_energy": 0.97}`
- `V6_PROTO`: `{"step_rank_ratio": 0.26, "step_rank_decay": 0.84, "ep_ltm_every": 6, "ep_ltm_weight": 0.45, "quarantine_cap": 4}`
- `COG_V6_CFG`: `{"d_model": 192, "num_heads": 6, "d_ff": 512, "initial_layers": 6, "initial_experts": 1, "chunk_size": 1, "enable_memory": true, "num_ltm_slots": 32, "enable_three_level_attn": true, "local_window": 16, "chunk_stride": 8, "enable_layer_stm": true, "signed_beta": true, "use_rope": true, "causal_attn": true, "auto_write_ltm": false, "enable_thinking": true, "max_thinking_steps": 2, "skip_threshold": 0.15, "theta_halt": 0.22, "min_surprise_drop": 0.008, "enable_world_model": true, "wm_rollout_horizon": 1, "enable_pc_rectification": true, "pc_thresh": 0.24, "hybrid_mode": true, "router_mode": "causal_proto", "num_router_protos": 64, "router_sim_threshold": 0.55, "auto_lifecycle": true, "max_experts": 6, "min_mature_steps": 40, "novel_confirm_steps": 3, "novel_surprise_threshold": 0.45, "dormant_age": 1000000000, "prune_age": 1000000000, "route_during_train": true, "use_trunk_routing": true, "router_context_blend": 0.20, "extend_loss_margin": 0.25, "align_wm_target_with_encoder": true, "enable_full_stack_adapters": true, "adapter_rank": 16, "router_use_idf": true, "router_center_multi": false, "train_revisit_conf_margin": 0.02, "preserve_input_novel_on_extend": true, "defer_on_novel_streak": true, "enable_mixed_split": false, "mixed_split_gap": 0.18, "max_coactive_experts": 4, "use_marginal_drop_halt": true, "halt_use_energy_only": true, "skip_final_lookahead": true, "pc_mode": "adaptive_rel", "ltm_null_logit": 1.5}`

## 4. Oracles & Null Controls
- **Zero Oracles:** No domain labels, domain boundary notifications, task IDs, or held-out ground-truth targets are exposed at training or inference routing time.
- **Null Controls Executed in Same Script:** `unigram` & `bigram` byte Markov models, untrained `Cog` & `Dense` baselines, `bag_of_bytes_domain_accuracy_with_true_labels`, `gsm8k` `copy_first` / `majority`, `shuffled_binding` null control, `distractor_ltm` & `no_ltm` controls, and `banking77` `chance (1/77)` / `majority` / `bag_of_bytes` / `untrained` baselines.

