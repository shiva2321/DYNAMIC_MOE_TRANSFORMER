# Pre-Registration: `real_pilot_v5` (`Cog_v8` Consolidated Non-Token Architecture)

**Pre-Registered Before Dev/Test Runs:** 2026-11-21 (UTC 2026-11-20T18:35Z)  
**Branch:** `v5-arch-efficiency` · **Runner:** `experiments/run_v8_cognitive_pilot.py`  
**Claim Tier Ceiling:** `OBSERVED (3 seeds)` (`seeds = [0, 1, 2]` on RTX 3060 GPU per `research-protocol.md` §§6, 9; 20-seed CI `REPRODUCED` not claimed without 20 seeds).

---

## 1. Consolidated Engineering Hypothesis & Non-Token Commitment

`Cog_v8` keeps **0% next-byte cross-entropy loss in the cognitive core** (`ByteChunker`, `Layers 0..5`, `Memory`, `WorldModel`, `ContextRouter`, `Thinker`, `expr_head`, `ctx_head`, `rel_pool_proj`, `fsp_head`, `wm_ret_proj` never receive gradients from `decoder`). An isolated LoRA `decoder` verbalizer updates strictly behind `.detach()` (`detached_renderer_diag_bpb` reported as secondary readout check).

`Cog_v8` integrates:
1. **`v3` SPGC + `I10`**: Meaning-consistency-gated diagonal BONG Fisher precision scaling (`fisher_scale` downweighted on uncertain/contradictory batches, gentle `prec_scale=0.85`).
2. **`v6` GPM + Adapters + Tier-2 Efficiency**: Decaying rank-budgeted NullSpace projection (`0/cap` saturation), full-stack private rank-16 adapters, `layer_stm_every=4`, and **single-pass** quarantine (eliminating the redundant `no_grad` screen forward pass).
3. **`v7` Dynamic `k in {0,1,2}` Routing + Thinker RLP + Deep JEPA + FSP**: Loss-free balancing, Expert-Router Coupling, multi-horizon deep EMA World Model, Future Summary Prediction, RLP progress loss.
4. **New `v8` Fixes (`I4/I5/I6/I7` + Masking/Relational Pool)**:
   - Explicit `BYTE_PAD_ID=257` & whitespace content masking (`chunk_content_mask`) across pooling, `FSP`, and `L_wm_ret` (which runs strictly on unpadded 128B continuous sequences, including real contiguous domain windows `wiki/story/code/math/forum`).
   - Order-sensitive Relational Pool (`rel_pool_proj` over `[u_mean, u_dir, u_last]`) preserving clause/argument direction without breaking strict chunk causality, plus unnormalized discriminative features `z_expr_raw_pool` alongside unit-normalized `z_expr_pool`.
   - `I7` Procedural structural pre-training (`150` lightweight steps on synthetic Dyck brackets, symmetric vs ordered relations, count sensitivity — zero lexical overlap with held-out families `E–F`).
   - `I5` Causal Interchange-Intervention Training (`forward_interchange`) on `BeliefMeaningTask` swapping intermediate causal subspaces `(m)` vs `(ctx)` across inputs + `I6` chunk-level `extract_episode_kv` episodic induction in `LongTermSlotMemory`.

---

## 2. Frozen Arms & Controls (Run in Same Script on Identical Splits)

1. `BagOfBytes_Null`: Byte histogram lower-bound control.
2. `Untrained_Cog_v8`: Random-init `Cog_v8` control.
3. `Upkeeper_M09`: Budgeted `<= 16 KB` surface hash table control (Track 1).
4. `Dense_v8_Matched`: Parameter-matched Causal Dense RoPE Transformer trained under the **same `v8` losses + `I7`**.
5. `Cog_v6_Ref`: Consolidated `v6` next-byte predictor baseline (`BPB` + continual probes) re-run cleanly in the same script at matched `steps_per_dom`.
6. `Cog_v8_SequenceTop1`: Sequence-level static Top-1 ablation (isolating token-level `k in {0,1,2}`).
7. `Cog_v8_FrozenRandomRouter`: Frozen random router control.
8. `Cog_v8_NoI5_NoRel`: Ablation disabling `I5` interchange training, `I7` procedural pre-training, and relational directional pooling.
9. **`Cog_v8` (Primary Pure From-Scratch)**: Zero external teacher anywhere.
10. **`Cog_v8_I9` (Teacher-Informed Track 2)**: Adds offline `all-MiniLM-L6-v2` Gram geometry distillation strictly during unlabeled Track-2 training (`teacher_informed=true`).
11. **`Cog_v8_Mixed_I9` (Teacher-Informed Mixed-Stream)**: Uniformly mixed 5-domain stream without domain labels (`teacher_informed=true`).

---

## 3. Disjoint Data, Calibration Budget & Protocol

- **Track 2 (`DOMAINS = wiki, story, code, math, forum`)**: `SEQ_LEN=128`, `batch=16`, `steps_per_dom=250` (`1,250` backward steps total in `test`). Every batch mixes `4` raw unedited 128B domain windows + `6` verified `A–D` edit sequences + `6` Banking77 train utterances (`pad_id=257`).
- **Held-out anti-shortcut edit test (`HELDOUT_FAMILIES = clause_quant, code_logic_ctrl`)**: 100% untouched/disjoint from `TRAIN_FAMILIES` (`A–D`) and `I7` procedural pre-training (`test_v8_families_disjoint`).
- **Track 1 (`BeliefMeaningTask`, strictly from-scratch)**: `1,000` `p1` steps (`24`-byte chunks, `bfloat16` CUDA ~`7s`) including `I5` interchange interventions + `I6` episodic induction probes (`novel phi_ep`, zero gradient updates), `20` `pb` burst steps, `160` `p2`, `160` `p3` steps.
- **Validation Budget:** At most **4** `--mode dev --seed 100` trials logged to `results/real_pilot_v5/tuning_log.json`, followed by **ONE** frozen `--mode test --seeds 0 1 2 --out_name test_final` run.

---

## 4. Pre-Registered Headline Metrics & Explicit Falsifiers

| Falsifier | Pre-Registered Bar (`test_final`, 3-seed means) | Kills Claim If Failed |
|---|---|---|
| **`V8-F1` Semantic Separation & Linear Restored** | On Banking77 test split: `Cog_v8` `b77_acc >= 0.45` (vs `v7=0.197`, recovering near `v6=0.495`) **and** `b77_auc >= 0.52` (Pure `Cog_v8`) / `>= 0.62` (`Cog_v8_I9`, `_Mixed_I9`), beating `BagOfBytes` & `Dense_v8_Matched`. | Non-token core cannot recover intent classification alongside overlap-trap AUC. |
| **`V8-F2` Concept Retrieval & Zero-Replay Retention** | Track 2 `Concept R@5 >= 0.28` beating `BagOfBytes_Null` (`~0.24`), and `forgetting_nuance_auc <= Dense_v8_Matched`, `forgetting_r5 <= Dense_v8_Matched`. | Continuous chunk world-model cannot retrieve future concepts without token CE. |
| **`V8-F3` Dynamic `k in {0,1,2}` & Functional Routing** | `MidSeq Switch >= 0.35`, no single `k > 90%`, `num_experts >= 3`, and `func_routing_nmi > Untrained_Cog_v8` / `Cog_v8_FrozenRandomRouter`. | Routing collapses to sequence/single-expert or ignores functional roles. |
| **`V8-F4` In-Family & Held-Out Anti-Shortcut (`F4b`)** | In-family `NuanceFlip >= 0.85` (`Selectivity >= 1.5x`), **and `HeldOutFam AUC >= 0.65` on unseen `E–F` families (`>= +0.12` over `Cog_v8_NoI5_NoRel` / `Dense_v8_Matched`)**, **and `EscMargin > Dense`**. | Order-sensitive relational pooling + `I7` fail to break shortcut ceiling on unseen rules. |
| **`V8-F5` Track 1 Composition, `I5` IIT & `I6` Episodic** | `Cog_v8` Track 1: `seen >= 0.90`, `heldout_unseen >= 0.78`, `paraphrase >= 0.75`, `burst >= 0.85`, `abstain >= 0.82`, **`iit_causal_acc >= 0.75`**, **`episodic_fewshot_acc >= 0.35` (vs `Dense` / chance `0.0625`)**, `rollout_delta > 0`. | `I5`/`I6` do not establish causal abstraction or beat `v7` Track 1 substantially. |
| **`V8-F6` RTX 3060 Efficiency & GPM Health** | Track 2 `train_seconds <= 220s` (faster than `v7` `304.9s` and competitive with `v6` `240s`), Peak VRAM `< 2.5 GB`, `gpm_at_cap_count == 0`. | Consolidated stack stays too slow or saturates GPM capacity. |
| **`V8-F7` Isolated Non-Token Core Integrity** | `test_v8_gradient_coverage_and_detached_decoder` proves `0` decoder CE gradient enters trunk/Thinker/WM/router and all enabled cognitive modules receive gradient. | Claimed non-token core leaks token-CE shortcuts or leaves modules untrained. |
