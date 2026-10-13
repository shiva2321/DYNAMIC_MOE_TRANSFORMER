# Kickoff message to Gemini, 2026-10-13: M5 (a real-text class-incremental stream, and today's pool measured on it)

To be sent AFTER Gemini has reported RV2-8 and Claude has answered (the two rounds touch different files, but the suite and the machine are shared). Why: every live-pool result so far is on synthetic streams (Q-04). E-024 showed that real text categories are nearly invisible to the per-sample gate (AUROC about 0.65) and visible to a batch test; we need a real stream on which merged experts can actually hurt, to measure today's pool now and the stage-3 gate later. This task builds the stream and measures today's pool. It changes nothing in the pool.

```
STOP RULE (read first): this message authorises ONLY the steps below (M5), in this order. After you paste the REPORT you STOP. You do not change growmoe/config.py or any pool code, you do not edit research/growmoe_results/m0, rv2a, rv2b or rv3, any existing test, bar, seed or pre-registration, you write NO documentation, and you do not commit. Every "STOP IF" means: stop at once and report, do not try to repair it.

WHAT TO BUILD
1. A new module growmoe/streams_real.py (do not edit growmoe/streams.py) with
   banking_incremental(seed=0, frontend=None, n_regimes=6, n_train=400, n_test=150, revisit=(2, 0, 4), revisit_size=200) -> Stream.
   - Data: banking77 from the LOCAL Hugging Face cache, no download: ~/.cache/huggingface/hub/datasets--banking77/snapshots/<hash>/data/train-00000-of-00001.parquet and test-00000-of-00001.parquet (columns text, label). If the cache is missing raise a clear error (tests skip).
   - frontend: a callable taking a list of strings and returning an (n, d) unit-length float tensor. Default growmoe.streams.HashFrontend() (offline, fast). The real runs inject research/growmoe_results/m0/run_m0.py::HybridFrontend(d=128, seed=0) (frozen all-MiniLM-L6-v2 from the local cache).
   - Regimes: embed every intent's training texts with the frontend, take the unit mean per intent, and group the 77 intents into n_regimes groups by clustering those means (growmoe.fingerprint.spherical_kmeans is available), deterministic given the seed. Each regime is the set of intents of one group (group sizes differ; that is fine).
   - Labels: class-incremental with disjoint blocks. Label = the position of the intent in a fixed order, n_classes = 77. A regime's inputs carry only its own intents' labels.
   - Segments: first visits in regime order, n_train documents each sampled from that regime's TRAINING texts, then the revisits (schedule = list(range(n_regimes)) + list(revisit)), each revisit revisit_size documents not used before when possible. Build them with growmoe.streams._segment, Regime and Stream exactly as s1_types does.
   - Test sets: from the OFFICIAL TEST split, restricted to the regime's intents, up to n_test documents per regime.
   - Stream.meta: {"schedule": ..., "groups": [intents per regime]}.
2. Tests FIRST in tests/growmoe/test_gm_m5_banking.py (default HashFrontend, each under 5 s, skip if the cache is missing): shapes and unit norm; deterministic given a seed; different seeds differ; label blocks disjoint across regimes and covering 77 intents once; every segment of a regime contains only that regime's labels; no test document appears in any training segment; the schedule is as specified; test sets contain only the regime's intents. Keep the RED output.
3. A smoke script research/growmoe_results/m5/run_m5_smoke.py (new folder; do NOT edit run_m0.py): dev seeds 100, 101, 102 only. Build the stream with HybridFrontend(d=128, seed=0). Arms, all with inner_steps = 200 and route_rule "accept_then_logf", q 0.02, alpha_spawn 0.01, m_min 32:
   (a) today's pool: PhyloMoE(PhyloCfg(...defaults...));
   (b) a single network: the same with max_experts = 1 (the forgetting baseline);
   (c) kNN_all, the lookup baseline: reuse research/growmoe_results/m0/run_m0.py::factory("kNN_all", ...).
   For each arm and seed: final accuracy over all regimes' test sets (growmoe.fairfight.run_fast with eval_every="final"), number of experts, seconds, peak memory; for (a) also the accuracy per regime. Print a table (mean over the seeds) and write smoke_m5.json. Generated numbers only. NO tuning. Label the output "DEV, exploratory, not pre-registered".

REPORT (paste, do not summarise): (1) RED then GREEN output of the new tests; (2) the groups of intents per regime for seed 100 and the number of training and test documents per regime; (3) the smoke table and the per-regime accuracy of the pool; (4) the last line of `python -m pytest tests/growmoe -q` (all existing tests must pass unchanged); (5) the files you created; (6) anything odd (regimes that are tiny or nearly empty, arms that failed, long runtimes).
STOP IF: the cache is missing; any existing test needs editing; the smoke run takes more than 30 minutes; RAM goes above 85 %.
Then STOP.
```
