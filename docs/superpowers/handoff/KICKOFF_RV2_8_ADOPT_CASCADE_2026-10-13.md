# Kickoff message to Gemini, 2026-10-13: RV2-8 (adopt the stage-2 cascade as the default, test-first, historical entry points pinned)

Sent by the mediator as below, after Claude's audit of RV2-7 (timestamps, hashes of every file in the provenance line, 120 run files, byte-identical regeneration of `RESULTS_RV2B.md` and `rv2b_runs.json`; C1 to G all PASS as expected).

```
STOP RULE (read first): this message authorises ONLY the steps below (RV2-8), in this order. After you paste the REPORT you STOP. You do not commit, you do not edit any file under research/growmoe_results/rv2b/, rv2a/ or rv3/, any pre-registration, bar, seed, statement or decision rule, and you write NO documentation (component cards, handoff, reports, notebook, memory: Claude writes those). Every "STOP IF" means: stop at once and report, do not try to repair it.

VERDICT ON RV2-7 (Claude audited it against the files): APPROVED. PREREG.md (10:49:56) is older than the first run file (10:55:41); all hashes in the RESULTS_RV2B.md provenance line match the files on disk; 120 files in runs/; `report` regenerates RESULTS_RV2B.md and rv2b_runs.json byte for byte. C1, C2, C3, C4 and G are all PASS, as expected. The decision line is: "C1, C3 and C4 PASS: adopt the cascade as the PhyloMoE default (gate_shortlist=2, gate_fingerprint_k=1, gate_shortlist_min_experts=64), test-first, with the historical entry points pinned." This message authorises exactly that adoption.

Two corrections to your report, so that you do not build on them: in growmoe/config.py the current defaults are gate_shortlist = 0, gate_fingerprint_k = 4, gate_fingerprint_refresh = 128, gate_shortlist_min_experts = 16 (your report said 32 and left the refresh out); and `measured_gate_macs_per_input` is a new function in accounting.py, not a change to an old one. Read the files, not your report. Also: the +0.04 point difference on OV-0.3 is statistically detectable only because the paired runs are deterministic; it is not an effect and must not appear anywhere as a gain.

STEP 1 - tests first. Create tests/growmoe/test_gm_rv2b_pins.py (mirror tests/growmoe/test_gm_rv2a_pins.py, which pinned batch_vote the same way) with these tests, run them and keep the RED output:
a. defaults: PhyloCfg().gate_shortlist == 2, .gate_fingerprint_k == 1, .gate_shortlist_min_experts == 64; gate_fingerprint_refresh stays 128.
b. growmoe.bench._phase1_cfg(seed=0).gate_shortlist == 0, and a caller can override it (_phase1_cfg(seed=0, gate_shortlist=2).gate_shortlist == 2).
c. research/growmoe_results/m0/run_m0.py: A_KW["gate_shortlist"] == 0 and PhyloCfg(**A_KW).gate_shortlist == 0 (load it with importlib as the rv2a pins test does). The M2-A and RV2-A arms are built from A_KW, so this pins them too.
d. a fake pool (the helper style of test_gm_rv2b_cascade.py) with 20 experts and the DEFAULT config gives results bit-identical to the same pool with gate_shortlist=0 (torch.equal on the P and LogF matrices of _gate_matrices, and on predict().unknown), because 20 < 64.
e. a fake pool with 70 experts and the default config does use the cascade: gate_stats["predict"]["stage1"] > 0.
f. the recorded RV2-B arms still pin themselves: run_rv2b.factory("A_exact", ...) builds a pool whose cfg.gate_shortlist == 0 and factory("A_cascade", ...) one whose cfg has gate_shortlist == 2, gate_fingerprint_k == 1 and gate_shortlist_min_experts == 8 (use a tiny stream object or only the cfg; do not run a stream).

STEP 2 - make them green with the smallest change.
- growmoe/config.py: gate_shortlist = 2, gate_fingerprint_k = 1, gate_shortlist_min_experts = 64, and replace the "OFF until the pre-registered test decides" comment by: "ON since 2026-10-13 (RV2-B, fresh seeds 40-59: C1, C2, C3, C4 and G all PASS, research/growmoe_results/rv2b/RESULTS_RV2B.md). Acts only with at least gate_shortlist_min_experts experts (below that the exact scan runs, bit for bit); Phase 1, M0, M2-A and RV2-A ran with the exact scan; pass gate_shortlist=0 to force it."
- growmoe/bench.py, in _phase1_cfg, next to the existing batch_vote line: kw.setdefault("gate_shortlist", 0).
- research/growmoe_results/m0/run_m0.py, in A_KW, next to batch_vote=False: gate_shortlist=0 (with a short comment in the same style).
- Search the repository for any other place that constructs PhyloCfg(...) to REPRODUCE a recorded result (grep -rn "PhyloCfg(" growmoe research tests), list what you found in the report and pin only those that reproduce recorded results. run_rv2b.py already sets gate_shortlist explicitly in both arms: leave it.
STOP IF any existing test needs to be edited (charter rule 10): paste the failing output and ask.

STEP 3 - suite. `python -m pytest tests/growmoe -q`. Expected: 520 + the number of tests you added, all passing.

REPORT (paste, do not summarise):
1. The RED output of test_gm_rv2b_pins.py before step 2 (the failing test names and messages), and its GREEN output after.
2. `git diff` of growmoe/config.py, growmoe/bench.py and research/growmoe_results/m0/run_m0.py (only these three should be modified by you; say so if not).
3. The list of PhyloCfg( constructions you found and which you pinned.
4. The last line of the suite and the sha256 (first 12 digits) of config.py, bench.py and run_m0.py after your change.
5. Anything that surprised you.

Then STOP. Do not start anything else.
```

## What Claude checks when the report comes back
1. The three modified files only; `git diff` shows exactly the intended lines; no existing test edited (the 520 pass unchanged).
2. RED output really fails for the stated reasons (defaults, pins), GREEN after.
3. Re-run `python -m pytest tests/growmoe -q` and `python research/growmoe_results/rv2b/run_rv2b.py report` (must still regenerate `RESULTS_RV2B.md` byte for byte: the report script hashes `config.py` only through the provenance line of stage 2, so a changed `config.py` hash is expected and is recorded in the adoption notes).
4. Then Claude writes the documentation: component cards (the stage-2 card), `GROWMOE_HANDOFF.md`, `SCOPE_CHARTER.md`, `MOE_CONTINUAL_LEARNING_REPORT_2026-10.md` (Appendix F), `ROUTING_RESEARCH_2026-10.md`, the m0 and m2a AMENDMENT notes, the lab notebook (E-021 completed, claims C-25 and C-26) and memory.
