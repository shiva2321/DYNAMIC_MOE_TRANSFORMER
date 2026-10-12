# Amendments to the M2-A pre-registration

None to any statement, threshold, stream or seed: every statement was evaluated exactly as written in `PREREG.md`.

## Note 1 (2026-10-10, after the result): the code default changed

The PREREG decision rule for "P1 PASS and every guard holds" is "adopt arm_monitor_after_infancy=True as the PhyloMoE default (a separate, reviewed change)". The user approved that on 2026-10-10 ("if it really helps then keep it on by default and make note of it and mention wherever it matters"). Done test-first: `test_flag_is_on_by_default_since_the_m2a_repair_was_confirmed_on_fresh_seeds`, `test_the_default_pool_behaves_exactly_like_the_explicit_flag_on`, `test_the_phase_one_entry_point_pins_the_old_behaviour_so_recorded_results_reproduce`; full GrowMoE suite 432 passed. Pinned to the old behaviour so recorded results reproduce: `growmoe.bench.run_phase1`, `research/growmoe_results/m0/run_m0.py` (`A_KW`). `run_m2a.py` sets the flag explicitly for both arms. Documented in `docs/COMPONENT_CARDS.md` (card 13 and the Phase 1 card), `docs/superpowers/handoff/GROWMOE_HANDOFF.md` (config defaults, 13f, 13g), `growmoe/config.py`, `growmoe/competence.py`.

## Note 2 (2026-10-12): `batch_vote` defaulted to ON afterwards; M2-A stays reproducible

Not an amendment of any statement. M2-A's arms are built from `run_m0.A_KW`, which now pins `batch_vote=False`, so `A_old` and `A_new` keep the per-sample assignment they were run with. The RV2-A test used `A_new` as its reference arm on seeds 20-39: on F-30 it scored 43.3 % (M2-A, seeds 10-19: 42.6 %) and on F-15 45.9 % (M2-A: 46.7 %), i.e. the reference reproduces.
