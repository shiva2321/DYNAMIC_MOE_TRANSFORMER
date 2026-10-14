# Kickoff message to Gemini, 2026-10-14: CVA-2B (what do dormant, archive, revive, merge and growth save and cost on a LONG stream?) and CVA-5A (are the resource numbers right?)

Why: in every benchmark so far the lifecycle states never fire, because the thresholds (dormant after 400 calls, archive after 800) are longer than the streams (60 to 150 calls). So nobody knows what they buy (memory) or cost (accuracy after an expert wakes up). CVA-2B builds a long cycling stream and measures it with the lifecycle on and off. CVA-5A recounts the pool's parameters and stored vectors by hand and compares them with the accounting module that the memory and efficiency claims rest on. Simulation only: no product code is touched. The dev job `sens_dev.py` may still be running in the background; keep to 3 threads and run one thing at a time.

```
STOP RULE (read first): this message authorises ONLY the steps below (CVA-2B and CVA-5A), in this order. After the REPORT you STOP. You do not edit anything under growmoe/ or any existing test, bar, seed or pre-registration, you do not touch research/growmoe_results/rv2c, m0, m2a, rv2a, rv2b, rv3, m5 or the existing files of research/growmoe_results/cva, you write NO documentation beyond the generated files named below, and you do not commit. If a check shows a DEFECT you register it in the report and you do NOT fix it. "STOP IF" means: stop at once and paste what you have; do not try to repair it.

Use torch.set_num_threads(3). Simulation seeds 10-14 only (never 60-79 or 100-109). Label every printout "VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)".

CVA-2B. Create research/growmoe_results/cva/cva2b_lifecycle_long.py writing cva2b.json and CVA2B_RESULTS.md (generated).
 Stream: growmoe.streams.s2_permuted(seed, n_regimes=30, n_train=200, n_test=100, schedule=SCHED) where SCHED = list(range(30)) followed by four more passes over all 30 regimes, each pass in its own random order (numpy default_rng([seed, 0x5343, pass_index]).permutation(30)). That is 150 segments of 200 samples, about 470 observe calls of 64.
 Arms (same stream, same seed, PhyloCfg(seed=seed, d=stream.d, n_classes=stream.n_classes, inner_steps=20, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf") plus):
  LIFE_ON : dormant_after=60, archive_after=120, merge_every=50 (these three values are fixed here, in advance).
  LIFE_OFF: dormant_after=10**9, archive_after=10**9, merge_every=10**9.
 Run each arm with growmoe.fairfight.run_fast(make_pool, stream, seed, eval_every="final") where make_pool builds a subclass of PhyloMoE that every 25 observe calls appends {call, n_experts, states: {infant, active, dormant, archived counts}, stored_vectors (growmoe.accounting.pool_report(self, d).stored_vectors)} to a list kept on the pool (the subclass lives in your script; observe calls super().observe unchanged). Keep the pool object (a holder list) to read the list and `pool.events` afterwards.
 Record per run: final accuracy; final experts; event counts by kind (spawn, extend, alarm, dormant/archive/revive if the event log has them, merge, grow); pool_report fields (params, stored_vectors, memory_bytes, gate_macs); the timeline; the maximum number of non-active experts.
 Seeds 10-14. PREDECLARED CRITERIA: (a) the events dormant-state changes, archive and revive each happen at least once in LIFE_ON (if not, report "lifecycle unreachable even on this stream" and the idle gaps you observed); (b) mean paired accuracy LIFE_ON - LIFE_OFF >= -1.0 point; (c) invariants hold in LIFE_ON at the end: every archived expert has replay None and optimiser None, still has its signature, and an archived expert that was revived kept its expert id (check from the event log); (d) report (no pass/fail) memory saved: stored_vectors and memory_bytes of LIFE_ON as a share of LIFE_OFF, and gate_macs of both (dormant experts still gate, so no gate saving is expected).
 Also run LIFE_ON a second time on seed 10 with the same arguments and confirm the final accuracy and event counts are identical (determinism).

CVA-5A. Create research/growmoe_results/cva/cva5a_accounting_recount.py writing cva5a.json and CVA5A_RESULTS.md. Run the real pool on growmoe.streams.s2_permuted(seed=10, n_regimes=6, n_train=200, n_test=50, schedule=[0,1,2,3,4,5,0,1]) with the LIFE_ON settings above (dormant_after=10, archive_after=20, merge_every=50) so that at least one expert is archived; then recount BY HAND, without calling growmoe.accounting:
  parameters = sum of p.numel() over every parameter of every live expert's module (count each distinct tensor once);
  stored vectors = sum over live experts of signature.n + replay.n (an archived expert's replay is None: count 0);
  gate multiply-adds per input = sum over live experts of signature.n * d (every stored signature vector is scanned once);
  memory bytes = 4 * parameters + 4 * d * stored vectors.
 Compare each with growmoe.accounting.pool_report(pool, d). CRITERION: all four equal exactly. Repeat for the same pool with `set_gate=True` after a few observe calls (stage 3 windows must count as stored vectors: add sum of rec.meta["windows"].n_samples() where present). If any number differs, register DEFECT-ACCOUNTING with both values.
 STOP IF: either script runs more than 60 minutes; RAM goes above 85 %; a stream cannot be built.

REPORT (paste, do not summarise): (1) the full CVA2B_RESULTS.md; (2) the full CVA5A_RESULTS.md; (3) the last line of `python -m pytest tests/growmoe -q` (unchanged: 642 passed); (4) the files you created (git status --short research/growmoe_results/cva); (5) anything odd.
Then STOP.
```
