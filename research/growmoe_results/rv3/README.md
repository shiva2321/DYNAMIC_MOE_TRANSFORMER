# rv3 - gatekeeper research spikes (2026-10-12, notebook entry E-023)

Throwaway scripts on DEV seeds 100-102, exploratory, NOT pre-registered. They are the evidence behind `research/GATEKEEPER_RESEARCH_2026-10.md`. None of them touches a product file: the set-level gate is a patched copy of `PhyloMoE.observe` built at import time (`inspect.getsource` + text insertion). Numbers in the report come from `RESULTS_RV3.md`, which `report_rv3.py` generates from the JSON files.

## Order of the evidence

| Step | Script | Output | What it answers |
|---|---|---|---|
| A1 | `spike_set_tests.py a1` | `spike_set_tests_a1*.{json,txt}` | Which set statistic is calibrated and powerful in a clean setting (a signature per regime, in-regime vs foreign batches)? |
| A2 | `spike_set_tests.py a2` | `spike_set_tests_a2*` | The same statistics logged inside the running pool (shadow mode), reference = the expert's stored prototypes |
| A2-recent | `spike_set_tests.py a2recent` | `spike_set_tests_a2recent*` | Shadow mode, reference = the owner's last five trained batches |
| (first run) | `spike_set_tests.py` | `spike_set_tests_log.txt` | First attempt; permutation p-values were computed in float32, so detection at alpha 0.001 reads 0 (artifact, kept as a record) |
| power | `spike_power_vs_batch.py` | `spike_power_vs_batch*` | Does the batch test keep its power for small batches? |
| trace | `diag_set_gate.py` | (stdout) | Every set-test decision on one stream, used to find why the live gate over-fired |
| integration | `spike_set_gate.py` | `spike_set_gate_*.json`, `*_log.txt` | The set-level gate inside the live pool: accuracy, expert counts, false novelty on known batches |
| routing | `spike_arbitration.py` | `spike_arbitration_*` | Test-time routing: per-sample arbitration upgrades (negative), group routing by vote, by set test, with and without a per-sample veto |
| ordered batches | `spike_sorted_batches.py` | `spike_sorted_batches_*` | What happens when the exchangeability assumption is violated (batches sorted by label), with and without the circuit breaker |
| report | `report_rv3.py` | `RESULTS_RV3.md` | All tables, generated |

Small table generators (numbers are read from the JSON files, never typed): `final_table.py [variant]` (vote against a set-gate variant, per stream), `compare_variants.py` (the design decisions side by side), `routing_table.py` (group routing, both pool types), `summ.py` (compact summary of the integration logs).

## Code generations of the integration spike (which file produced which log)

The variants a log names did not change meaning between generations; new variants were added. Each snapshot is kept so the log can be tied to the code that wrote it.

| Snapshot | Candidate rule | Logs |
|---|---|---|
| `spike_set_gate_gen1.py` | candidates = the per-sample gate's owners first, then by summed log-density; untestable candidate blocks a novelty call | `gen1_ownerfirst/*` (stopped early: it spawned an expert per batch at centre cosine 0.85, 104 experts for 10 regimes) |
| `spike_set_gate_gen2.py` | candidates = the three experts whose reference mean is nearest to the batch mean; novel iff all reject; reference stored / recent / both | `spike_set_gate_g2A*`, `g2B*`; `g3keep*` ran with the next snapshot (same variant meaning) |
| `spike_set_gate_gen2b_windows.py` | adds per-expert source windows (`md_windows*`) | (used in interactive checks only) |
| `spike_set_gate_gen3_veto.py` | one pair-based decision routine; ownership policies own / keep / veto | `g3keep*` |
| `spike_set_gate_gen4_loss.py` | adds the label-aware ownership policy (among the experts the test admits, the lowest loss on the batch's labels) | `spike_set_gate_g4A*`, `g4B*` |
| `spike_set_gate.py` | adds the circuit breaker (`*_cb` variants; first version crashed on a name the patched function could not see, fixed) and the MMD statistic (`mmd_windows_loss`) | `spike_set_gate_g5mmd*`, `spike_sorted_batches_cb*`; the final file reproduces the generation 2 to 4 results exactly (checked on four cells) |

Policies: `own` = the accepted samples go to the nearest admitted expert; `keep` = the vote's assignment stands, the test only decides novelty; `veto` = the vote's owner is tried first; `loss` = among the admitted experts, the one with the lowest cross-entropy on the batch's labels.

## Reproduce

```
python research/growmoe_results/rv3/spike_set_tests.py a1
python research/growmoe_results/rv3/spike_set_tests.py a2
python research/growmoe_results/rv3/spike_set_tests.py a2recent
python research/growmoe_results/rv3/spike_power_vs_batch.py
python research/growmoe_results/rv3/spike_set_gate.py --streams OV-0.6,OV-0.85 --variants vote,md_windows_loss --tag t
python research/growmoe_results/rv3/spike_arbitration.py --variant md_windows_loss --streams OV-0.85 --tag t
python research/growmoe_results/rv3/spike_sorted_batches.py --streams F-15 --variants vote,md_windows_loss,md_windows_loss_cb --tag t
python research/growmoe_results/rv3/report_rv3.py
```

The spikes import `research/growmoe_results/rv2a/run_rv2a.py` (streams, budgets) and need the product package of commit 090ece8 or later with the stage-1 default (`batch_vote=True`).
