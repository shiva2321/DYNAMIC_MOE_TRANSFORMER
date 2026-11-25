# Pre-Registration: `real_pilot_v3` (Tier 1 & Tier 2 Efficiency + Fidelity Evaluation)

**Frozen Before Execution (UTC):** 2026-11-20T07:55:00Z  
**Status:** FROZEN BEFORE DEV/TEST RUNS  
**Script:** `experiments/run_real_pilot.py` + `experiments/summarize_real_pilot.py` + `experiments/update_tuning_log.py`

## 1. Headline Claim, Metrics & Explicit Falsifiers
- **Claim (Tier 1 — Engineering Equivalence):** Batched shape-bucketed GPM projection (`dmoe/null_space.py`), single-sync routing dispatch (`dmoe/real_model.py`, `dmoe/adapters.py`, `dmoe/thinking.py`, `dmoe/decoder.py`), unused-state scan bypass (`dmoe/memory.py`), and AMP-safe scalar calibration (`dmoe/thinking.py`, `dmoe/world_model.py`) materially lower wall-clock runtime of `Cog_v6` without degrading held-out `mean_final_bpb`, `mean_forgetting_bpb`, blind `routing_nmi`, or `gpm_at_cap_count`.
- **Claim (Tier 2 — Architectural Cadence):** Adding hybrid alternating Layer-STM cadence (`tier2_stm_every in {2, 4}` selected on `dev`), amortized intra-window episodic passes (`tier2_ep_ltm_every = 12`), and strided GPM covariance accumulation (`tier2_gpm_cov_every = 2`) yields substantial additional wall-clock & VRAM savings while preserving continual retention and routing quality over `Dense_NoReplay`.
- **Headline Metrics:** `train_seconds` (wall-clock), `tokens_per_sec` (`bytes/sec`), `mean_final_bpb` (lower is better), `mean_forgetting_bpb` (lower is better), `routing_nmi`, `gpm_at_cap_count`, `peak_vram_mb`, and cognitive probes (`thinking_delta_mean`, `pc_contra_trig / pc_clean_trig`, `matched_ltm - distractor_ltm`).
- **Explicit Falsifiers (Pre-Registered):**
  1. **Tier 1 Speed Falsifier:** If `Cog_v6` (`Tier 1`) does not achieve at least a **15% wall-clock speedup** over pre-Tier-1 `test_v2` `Cog_v6` (`250.6s`, i.e., must run `< 213s` on identical `steps=400`, `seeds 0 1 2`) OR degrades `mean_final_bpb` / `mean_forgetting_bpb` beyond seed variance (`> +0.04 BPB`) -> Tier 1 efficiency-equivalence claim is refuted.
  2. **Tier 2 Trade-off Falsifier:** If `Cog_v6_Tier2` fails to save **>= 25% total training time** vs `Cog_v6` (`Tier 1`), OR increases `mean_final_bpb` / `mean_forgetting_bpb` by **`> +0.05 BPB`** vs `Cog_v6`, OR drops blind `routing_nmi < 0.50`, OR fails to outperform `Dense_NoReplay` run inside the same script -> Tier 2 no-compromise claim is refuted (reported honestly as an accuracy-for-speed trade-off).
  3. **Mechanism Falsifiers:** Same as `real_pilot_v2` (`thinking_delta >= 0`, `pc_contra_trig <= pc_clean_trig`, `matched_ltm >= distractor_ltm`, `gpm_at_cap > 25%`) — any mechanism that fails its falsifier is reported as non-functional/negative.

## 2. Frozen Experimental Protocol
- **Dataset & Schedule:** Real multi-domain UTF-8 streams `wiki -> story -> code -> math -> forum`, `SEQ_LEN = 128`, `batch = 16`, `steps = 400` per domain (`2,000` optimizer steps; `4,096,000` training bytes seen).
- **Arms (`--mode test --seeds 0 1 2`):**
  1. `Dense_NoReplay`: `CausalDenseRoPEBaseline` (`5.28M` params, parameter-matched to initial `Cog_v6`, re-run in the same script).
  2. `Cog_v6`: `COG_V6_CFG` + `V6_PROTO` with all Tier 1 exact optimizations (`layer_stm_every=1, ep_ltm_every=6, gpm_cov_every=1`).
  3. `Cog_v6_Tier2`: Tier 1 + Tier 2 architectural cadence (`tier2_stm_every` selected strictly on `dev` seed `100` from `{2, 4}`, `tier2_ep_ltm_every=12`, `tier2_gpm_cov_every=2`).
- **Validation Budget & Disjoint Seeds:** At most **2 validation trials** on `eval_split="dev"` (`seed=100`) logged mechanically in `results/real_pilot_v3/tuning_log.json`. Held-out `test` windows (`seeds=[0, 1, 2]`) are touched ONLY ONCE after `dev` selection. Because `n = 3` seeds on local RTX 3060, headline test claims remain capped at `OBSERVED` per `research-protocol.md` §§6, 9.
- **Oracles & Null Controls:** Zero domain labels/boundaries given during training or routing. Same `ngram_baselines`, `untrained_eval`, `bag_of_bytes`, `distractor_ltm`, and `var_bind_shuffled` null controls computed inside the script.
