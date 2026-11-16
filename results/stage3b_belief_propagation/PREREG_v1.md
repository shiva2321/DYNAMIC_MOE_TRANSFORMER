# Pre-Registration (DRAFT): Belief-Revision Propagation on Stage 3B (`stage3b_belief_propagation_v1`)

- **Experiment ID**: `stage3b_belief_propagation_v1`
- **Status**: DRAFT, 2026-11-16. Nothing has been run. Sections marked **[DEV]** are filled from DEV seeds and then frozen; everything else, including every falsifier and margin in section 7, is fixed now and must not change after the first DEV run.
- **Amendment A (2026-11-16, before any run)**: added the efficiency claim E, compute accounting (section 4.1), the eval cadence (section 3), arm `A3_ComputeMatched`, the DEV budget for `A9`, gates G3 and G4, falsifiers F11 to F13, and their readings. No run, tuning log or result existed when this was added (checked: `results/stage3b_belief_propagation/` held only this file; the runner did not exist).
- **Depends on (exists, tested)**: `dmoe/data/stage3_tasks.py` (`build_compositional_belief_tables`, `generate_3b_seen_batch`, `generate_3b_pair_eval_batch`), `dmoe/upkeeper.py` (`CalibratedBayesianUpkeeper`), `dmoe/stage3_baselines.py` (`DenseRoPEContinualBaseline`, `BudgetedDictionaryBaseline`), `torch.utils.flop_counter.FlopCounterMode` (present in the installed torch 2.6.0).
- **Depends on (does not exist yet)**: the propagation sweep (section 4), the phase-2 table and mask builder (section 2), the compute ledger (section 4.1), and the runner `experiments/run_stage3b_belief_propagation.py` -> `results/stage3b_belief_propagation/results_v1.json`.
- **Not a predecessor lock**: no Stage 3 or X1 results exist in `results/`. This experiment contains X1's four arms as a by-product on this bed; it does not replace X1 on Track K.

## 1. Claim and the one new mechanism

**Mechanism P (selective demotion).** When the exact store executes a genuine revision, entries that the neural lane had previously *explained* and now confidently *contradicts* lose their fusion weight, so the neural lane answers for them until direct evidence arrives. Entries the neural lane never explained (exceptions) are never touched by P.

**Claim.** After a partial law revision shown on only some keys, a hybrid (neural lane + counting store + P) is correct on revised keys it was *not* shown, without damaging unrevised keys or exceptions, and its accuracy over the whole table exceeds the dictionary, the neural lane alone, and the same hybrid without P.

**Claim E (efficiency, secondary; judged only through F11 to F13).** Revising by update (`A4`: continue from the phase-1 model, store + P) reaches the revision target (section 4.1) with less training compute than retraining a Dense lane from scratch on the revised table (`A9`), and `A4`'s gain over `A3` is not bought by `A4`'s extra compute (`A3_ComputeMatched`). Claim E is not judged if F1 fails; it is then recorded as "not reached".

**What this is not.** Dependencies here are single-hop (many keys share one law). Multi-hop chains of derived beliefs are out of scope. Claim E is about the cost of a *revision* given an already trained model; it says nothing about the cost of learning the law in the first place (phase 1), which every neural arm pays in full.

## 2. Bed: partial revision of the Stage 3B law

Per seed `s`, from `build_compositional_belief_tables(seed=s, num_exceptions=24, split_ratio=0.75)`: 192 seen pairs, 64 held-out pairs, 24 exception pairs inside the seen set.

- **Revised rows `R`**: 8 of the 16 `prev` symbols, drawn with `torch.Generator().manual_seed(s * 131 + 7)`.
- **Phase-2 truth `T2[a, b]`**: `rev_table[a, b]` if `a in R`, else `init_table[a, b]`. For non-exception pairs in `R` the revised value differs from the old value on every pair (shift of 5 in phi-space).
- **Phase-2 visible mask `M2`** (same generator, continued):
  - *Evidence*: per revised row, 6 of its non-exception seen pairs, plus every exception pair in that row.
  - *Confirmations*: per unrevised row, 6 of its non-exception seen pairs. Exception pairs in unrevised rows are never in `M2`.
  - If a row has fewer than 6 eligible pairs, take all of them.

Evaluation subsets (all scored against `T2` with `generate_3b_pair_eval_batch`, batch 64, 4 batches, accuracy and NLL):

| Subset | Definition | Tests |
|---|---|---|
| `evid` | pairs in `M2`, rows in `R` | direct revision |
| `stale` | seen, non-exception, rows in `R`, not in `M2` | **propagation (primary)** |
| `held_rev` | held-out pairs, rows in `R` | propagation to never-seen keys |
| `conf` | pairs in `M2`, rows not in `R` | retention |
| `silent` | seen, non-exception, rows not in `R`, not in `M2` | locality |
| `exc_silent` | exception pairs, rows not in `R` | over-propagation guard |
| `held_unrev` | held-out pairs, rows not in `R` | locality on never-seen keys |
| `W` | all 256 pairs, uniform | whole-table score |

Chance is 1/16 = 0.0625. The bed has no label noise, so the ">0.97 means leakage" guard does not apply; leakage is checked by gates G1 and G2 instead.

## 3. Stream

1. **Phase 1, establish**: `steps_p1` **[DEV]** batches from `generate_3b_seen_batch(init_table, seen_mask, batch_size=32, seq_len=16)`, `source_id` rotating over 3 sources.
2. **Phase B, burst**: 5 batches under `burst_table` on `seen_mask`, single `source_id="noisy_single"`.
3. **Phase 2, revision**: `steps_p2` **[DEV]** batches from `generate_3b_seen_batch(T2, M2, ...)`, `source_id` rotating over 3 sources.

Evaluate at end of phase 1, end of phase B, and at 25%, 50% and 100% of phase 2. Falsifiers use the 100% checkpoint unless stated.

**Cost curve (for F11 only).** In addition, evaluate `stale` and `W` every `eval_every = max(1, steps_p2 // 20)` optimiser steps of phase 2 (and every `max(1, steps_A9 // 20)` steps of `A9`'s run). These evaluations draw from a separate generator seeded `s * 977 + 3`, run under `torch.no_grad()`, and are excluded from all compute counts; they must not change the training stream (gate G3).

## 4. Mechanism P, exactly

- **Flag.** Each primary entry gains one bit, `explained`. At every confirmation it is set to `1` if the neural lane's argmax for that position equals the stored value, else `0`. It uses logits already computed in that forward pass.
- **Trigger.** A `gate_targets` call that executes at least one flip opens a *revision event*. No flip, no sweep.
- **Sweep.** `n_adapt` **[DEV]** optimiser steps after a flip, and at most once per `n_adapt` steps while the event is open: for each primary entry with `explained == 1` that has not been confirmed since the event opened, query the neural lane on that pair. If its argmax differs from the stored value and its top probability is at least `tau_derive` **[DEV]**, set the entry's count to `1` (below `min_fuse_count`, so fusion weight is 0). The stored value is not changed.
- **Close.** The event closes after two consecutive sweeps that demote nothing.
- **Accounting.** Store entries are charged 17 bytes (16 + the flag). All store arms and the dictionary get `max_total_bytes = 8192`; the existing 4096 default holds only 140 primary entries, fewer than the 192 seen pairs. Sweep queries are counted and clocked.

### 4.1 Compute accounting (applies to every arm)

- **Counter.** Training compute is measured, not estimated from op counts on paper: every training forward pass, backward pass, optimiser step, `gate_targets` / `fuse_logits` call and P sweep query runs inside `torch.utils.flop_counter.FlopCounterMode`. Evaluation passes run outside it.
- **Known blind spot (declared).** `FlopCounterMode` counts only the aten ops it has formulas for (matmul, attention, convolution and similar). Elementwise ops and Python-side store operations (dict reads and writes, count updates) are not counted as FLOPs. To cover them, each arm also records (a) wall-clock per phase on the same device and process, with the first 5 steps of each phase excluded as warm-up and CUDA synchronised before each timestamp, and (b) the raw number of store reads, store writes and sweep queries.
- **Ledger.** Per arm, per seed, per phase (1, B, 2) the results file records: counted FLOPs, split into training forward+backward, upkeeper calls and P sweep queries; wall-clock seconds; store ops; trainable parameter count; store bytes.
- **Revision target.** A seed reaches the target at the first cost-curve checkpoint where both `stale >= 0.50` (the F1 floor) and `W >= 0.70` hold. `W >= 0.70` is a new frozen choice made in Amendment A.
- **Cost-to-target.** `C_A4` = counted FLOPs from the start of phase 2 to the first checkpoint where `A4` reaches the target (the *marginal* cost of revising an existing model). `C_A9` = counted FLOPs from `A9`'s initialisation to the first checkpoint where `A9` reaches it (the cost of brute-force retraining). `A9` sees revised labels on every seen pair, more than `A4` sees, so it is an oracle-favoured comparator; that bias is against `A4`. The *total* cost of `A4` (phases 1 + B + 2) is also reported next to `C_A9`, as a descriptive figure only.

## 5. Arms (equal data, equal steps, equal store budget)

| Arm | Neural lane | Store + fusion | Demotion rule |
|---|---|---|---|
| `A1_Dictionary` | none | `BudgetedDictionaryBaseline` | none |
| `A2_Dense` | `DenseRoPEContinualBaseline` | none | none |
| `A3_Dense_Hybrid` | Dense | `CalibratedBayesianUpkeeper` (`gate_targets` + `fuse_logits`) | none |
| `A3_ComputeMatched` | Dense | same as `A3` | none; phase 2 extended by `k` extra optimiser steps on further `generate_3b_seen_batch(T2, M2, ...)` batches (same generator, continued), `k` = smallest integer making its phase-2 counted FLOPs >= `A4`'s on the same seed |
| `A4_Dense_Hybrid_P` **(proposed)** | Dense | same | P |
| `A4b_NoFlag` | Dense | same | P ignoring `explained` |
| `A7_GlobalDistrust` | Dense | same | on first flip, demote every entry not since confirmed |
| `A8_OracleScope` | Dense | same | demote exactly the true `stale` entries (ceiling) |
| `A5_dMoE_Hybrid`, `A6_dMoE_Hybrid_P` | `RealSelfStabilizingDMoE` | same | none / P (secondary, reported, not in falsifiers) |
| `A9_JointOracle` | Dense trained from scratch on `T2` over all seen pairs | none | ceiling for `held_rev` |
| `Null_Untrained`, `Null_ShuffledTargets` | Dense | as `A4` | P |

The primary neural lane is Dense because dMoE did not beat Dense+Upkeeper on Track K (card M09). The isolating control for P is `A3`.

## 6. Splits, tuning, gates

- **DEV seeds** `[100..104]`, **TEST seeds** `[0..19]`. TEST is untouched until this file is frozen; log every DEV run in `tuning_log_v1.json`.
- **Tuned on DEV [DEV]**: learning rate per neural arm from `{3e-3, 5e-3, 7e-3}`; `steps_p1`, `steps_p2` (raise until `A2` and `A3` stop improving on `evid` and `conf`, so no baseline is under-trained); `n_adapt` from `{5, 10, 20}`; `tau_derive` from `{0.5, 0.7, 0.9}`. `A3` receives every setting `A4` receives. `A3_ComputeMatched` uses `A3`'s settings unchanged (only `k` differs, and `k` is computed, not tuned). `A9` gets its own learning rate from the same grid and its own budget `steps_A9` **[DEV]**: start at `steps_p1 + steps_p2` and double until G4 passes and `W` stops improving.
- **G0, bed is usable (DEV, before anything else)**: `A2_Dense` phase-1 held-out accuracy >= 0.60 on DEV mean. If it fails, the neural lane has not learned the law, propagation cannot be tested, and the bed or lane is fixed first. No TEST run.
- **G1, dictionary bound**: `A1` `stale` <= 0.02 and `held_rev` <= 0.10.
- **G2, causality**: future-token perturbation changes logits by <= 1e-5 for `A2`, `A3`, `A4`.
- **G3, evaluation does not perturb training**: for `A4` on DEV seed 100, the phase-2 training-loss trajectory with the cost-curve evaluations switched on equals the trajectory with them switched off (bitwise on CPU; within 1e-6 on CUDA). If it fails, the cost curve is invalid and is fixed before any further run.
- **G4, retraining comparator is not a straw man**: `A9` reaches the revision target (section 4.1) on every DEV seed within `steps_A9`. If it cannot, F11 is not run and this is reported.

## 7. Falsifiers (frozen now; all must hold on TEST means, n = 20)

"CI" is a paired 95% bootstrap interval over seeds. F1 to F10 judge the main claim; F11 to F13 judge Claim E only, so a failure there does not change the verdict on the main claim, and the reverse also holds.

1. **F1 propagation.** `A4.stale >= A3.stale + 0.25`, `A4.stale >= 0.50`, and CI lower bound of `A4.stale - A3.stale` > 0.
2. **F2 never-seen keys not harmed.** `A4.held_rev >= A3.held_rev - 0.02`.
3. **F3 hybrid beats every single lane.** `A4.W > max(A1.W, A2.W, A3.W) + 0.03`, CI lower bound of each difference > 0, and on every subset `A4 >= max(A1, A2) - 0.05`.
4. **F4 locality.** `A4.silent >= A3.silent - 0.03` and `A4.held_unrev >= A3.held_unrev - 0.03`.
5. **F5 exceptions survive.** `A4.exc_silent >= A3.exc_silent - 0.05`.
6. **F6 selectivity is doing the work.** `A4.W > A7.W + 0.03` and `A4.exc_silent > A4b.exc_silent + 0.10`.
7. **F7 burst immunity.** In phase B, `A4` demotes 0 entries in at least 19 of 20 seeds, and `A4` seen accuracy after phase B is within 0.02 of end of phase 1.
8. **F8 calibration.** `A4` NLL on `W` <= `A3` NLL on `W` and <= `A2` NLL on `W`.
9. **F9 cost.** Store bytes <= 8192 at every checkpoint; sweep queries per revision event <= 4 x the number of primary entries; both counted queries and wall-clock for phase 2 reported for `A3` and `A4`.
10. **F10 nulls fail.** Both nulls have `stale` < 0.15 and `W` < 0.20.
11. **F11 update is cheaper than retraining (Claim E).** In at least 18 of 20 TEST seeds, `A4` reaches the revision target and `C_A4 <= 0.5 * C_A9`. A seed where `A4` never reaches the target counts as a failure. Also reported (descriptive, not a pass condition): per-seed ratios `C_A4 / C_A9`, their geometric mean with a 95% bootstrap CI, the Wilson interval on the seed count, and the same comparison in wall-clock.
12. **F12 the gain is not bought by compute.** `A4.stale >= A3_ComputeMatched.stale + 0.25` and CI lower bound of `A4.stale - A3_ComputeMatched.stale` > 0; and `A4.W > A3_ComputeMatched.W + 0.03` with CI lower bound > 0.
13. **F13 overhead of P is bounded.** Mean over TEST seeds of `A4` phase-2 counted FLOPs <= 1.25 x `A3` phase-2 counted FLOPs, and the same ratio in wall-clock is reported next to it.

## 8. How each outcome will be read (written before any run)

| Outcome | Reading |
|---|---|
| All pass | P works on single-hop shared-law dependencies; next is multi-hop and a latent-keyed store. |
| F1 fails and `A2.stale` is low | The neural lane does not generalise a revision from partial evidence. P cannot help; the bottleneck is the neural lane. Compare with `A9` and `A8`. |
| F1 passes and `A2.stale` is about equal to `A4.stale` | P only removes the store's stale override. Report it as a staleness fix, not as reasoning. |
| F3 fails | The hybrid adds nothing over its best single lane on this bed. |
| F5 or F6 fails | Demotion works but is not selective; the `explained` flag is not a valid dependency signal. |
| F4 fails | Phase-2 gradient steps drift the neural lane on unrevised rows; the problem is interference, not propagation. |
| `A8` far above `A4` | Scope detection, not demotion, is the limit. |
| F11 fails, F1 passes | P revises correctly but not cheaply: retraining from scratch is competitive on this bed. Report "no efficiency advantage". |
| F12 fails, F1 passes | `A4`'s gain over `A3` is reproduced by spending the same compute on more gradient steps; P is not the cause. |
| F13 fails | P's bookkeeping costs more than 25% of a phase-2 run; any efficiency claim must be stated net of this. |
| F11 passes, wall-clock ratio does not agree | The FLOP saving is eaten by uncounted (Python, store, elementwise) overhead; report wall-clock as the binding number. |

A failed falsifier stays in the ledger. Any change to sections 2 to 7 after the first DEV run needs a new `PREREG_v2.md` and a note of what changed and why.
