# Kickoff message to Gemini, 2026-10-14: CVA-1B (which rule decides "this cluster is a new regime" without false spawns?) and CVA-3A (how often does the drift alarm go off on a stable stream?)

Why: CVA-1 found that the probation evidence test is uncalibrated: a cluster drawn from an existing expert's OWN regime passes it 0 to 2 % of the time at 32 to 64 points, 5 to 12 % at 128 and 33 to 52 % at 256 (Claude re-ran it with other seeds and code: same picture). CVA-1B compares candidate replacement rules on the same simulation, with thresholds fixed on separate seeds. CVA-3A measures false alarms of the competence monitor (a false alarm spawns a duplicate expert). Simulation only: no product code is touched. The dev-seed job `sens_dev.py` is running in the background (about 6 of 12 cores): keep to 3 threads, one thing at a time.

```
STOP RULE (read first): this message authorises ONLY the steps below (CVA-1B and CVA-3A), in this order. After the REPORT you STOP. You do not edit anything under growmoe/ or any existing test, bar, seed or pre-registration, you do not touch research/growmoe_results/rv2c, m0, m2a, rv2a, rv2b, rv3, m5 or the existing files of research/growmoe_results/cva, you write NO documentation beyond the generated files named below, and you do not commit. If a check shows a DEFECT you register it in the report and you do NOT fix it. "STOP IF" means: stop at once and paste what you have; do not try to repair it.

Use torch.set_num_threads(3). Simulation seeds 0-19 only (never 60-79 or 100-109): seeds 0-9 are CALIBRATION seeds (thresholds are fixed on them), seeds 10-19 are EVALUATION seeds (every reported number comes from them). Label every printout "VERIFICATION (criteria fixed in advance by Claude, 2026-10-14)".

CVA-1B. Create research/growmoe_results/cva/cva1b_spawn_rules.py writing cva1b.json and CVA1B_RESULTS.md (generated). Import the helpers (centres, draw, the evidence formula) from cva1_gate_evidence.py if it has an `if __name__ == "__main__"` guard; otherwise copy them verbatim into the new file (do not edit the old one). d = 128, data as in CVA-1.
 Setting: one live expert A built from n_fit points (n_fit in {64, 256, 1024}); a candidate cluster of m points (m in {32, 64, 128, 256}); the rule answers "spawn?".
 Rules:
  R0 today: evidence > log(1/0.01), evidence as in CVA-1.
  R1 size-aware: evidence / sqrt(m) > z1.
  R2 per-point: evidence / m > z2.
  R3 exact permutation test: growmoe.settest.perm_reject(ref=live_sig.protos, batch=cluster, alpha=0.001, n_perm=999, gen=growmoe.settest.generator(seed, rep), stat="meandist") returns rejected -> spawn.
  R4 both: R0 AND R3.
  R5 exact test at alpha 0.01 (n_perm 999).
 With two live experts present (A and an unrelated C at rho = 0) the rule must hold against every live expert: for R0-R2 use the maximum of the experts' logpdf as the evidence code does; for R3 and R5 spawn only if the test rejects for EVERY live expert.
 Calibration (seeds 0-9, negative control = clusters drawn from A): z1 and z2 = the 99th percentile of the pooled negative-control values over all (n_fit >= 256, m) cells. Then evaluation (seeds 10-19, 10 draws per seed per cell = 100 draws per cell):
  Negative control: share of clusters from A that each rule would spawn, per (n_fit, m). PREDECLARED CRITERION: <= 0.01 (one percent) in every cell with n_fit >= 256.
  Positive control: clusters from regime B with rho in {0, 0.3, 0.6, 0.85, 0.95}, m in {32, 64, 128}, n_fit = 256: share spawned. PREDECLARED CRITERION: >= 0.99 for rho <= 0.6 and m >= 64; >= 0.9 for rho = 0.85 and m = 128; rho = 0.95 reported only.
 Verdict column per rule: PASS only if both criteria hold in all their cells; otherwise list the failing cells. Report also the runtime per decision of each rule (milliseconds).

CVA-3A. Create research/growmoe_results/cva/cva3a_monitor_false_alarms.py writing cva3a.json and CVA3A_RESULTS.md. Use the real pool: PhyloMoE(PhyloCfg(seed=s, d=128, n_classes=8, inner_steps=5, q=0.02, alpha_spawn=0.01, m_min=32, route_rule="accept_then_logf", arm_monitor_after_infancy=True)) (all other fields default) on a STATIONARY single-regime stream: growmoe.streams_m1.function_regimes(s, n_regimes=2, n_train=20000, n_test=64) and feed ONLY the first segment (regime 0) in batches of 64 with the labels (use stream.algo_batches(64, s) and stop after the first segment; the first segment holds 20,000 samples = 313 batches). Seeds s = 10-19. Record per run: number of `alarm` events, number of spawn and extend events, final number of experts, and the call index of every alarm. Also repeat with competence=False to confirm no alarms. No pass/fail: report alarms per 10,000 labelled samples (mean and range over seeds) and say whether any seed ended with more than one expert.
 STOP IF: CVA-1B's calibration cannot be run because a helper cannot be imported or copied; any script runs more than 60 minutes; RAM goes above 85 %.

REPORT (paste, do not summarise): (1) the full CVA1B_RESULTS.md; (2) the full CVA3A_RESULTS.md; (3) the last line of `python -m pytest tests/growmoe -q` (it must be unchanged: 642 passed); (4) the files you created (git status --short research/growmoe_results/cva); (5) anything odd.
Then STOP.
```
