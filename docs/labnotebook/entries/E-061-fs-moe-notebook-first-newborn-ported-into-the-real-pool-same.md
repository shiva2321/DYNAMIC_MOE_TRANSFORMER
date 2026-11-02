# E-061 - FS-MoE notebook-first newborn ported into the real pool: same accuracy with 1% of the network parameters, the size test idle, a weak-parent flaw fixed

- **ID:** E-061
- **Date:** 2026-11-01
- **Track:** G
- **Type:** experiment
- **Status:** completed on report seeds 710-714 (5 seeds, four beds); follows E-060; the ablation (arms U8x, U128x, NBn) was added after the first results and is exploratory
- **Evidence grade:** B (script-generated, 5 seeds, predictions declared before any report-seed run and scored by script; knobs and two design repairs found on dev seeds 700-702; NOT a pre-registration; no independent reviewer)
- **Code / commit:** not committed (standing rule); `fsmoe/nbpool.py` (`NotebookExpert`, `NbCfg`, `nb_cfg`, `NbPool`), `fsmoe/nbpool_lab.py`, `fsmoe/report_n.py`, `fsmoe/scorecard_n.py`, `fsmoe/figures_n.py`; tests `tests/fsmoe/test_fs_nbpool.py` (18 tests; 138 pass in tests/fsmoe; 16 mutants of the new code killed, 5 of them only after tests were added; 91 mutants killed over E-055 to E-061); design `docs/superpowers/specs/2026-11-01-fsmoe-nbpool-design.md`; growmoe/ and unimoe/ imported (`PhyloMoE` through `UniPool`, `MLPExpert`, `Reservoir`, `LdaHead`, the transfer verdict), never edited
- **People:** Claude designed, built and ran it on the user's answer "Port into real pool" (2026-11-01) to the next-step question after E-060 (Q-63); no reviewer yet

## Question
Does the notebook-first newborn of the growth lab (E-060: a newborn is a closed-form notebook; a network is given only if it wins a held-out test against the notebook) work inside the real pool (`UniPool`: `PhyloMoE` with a closed-form head in every expert) at equal accuracy and less memory? (Q-63)

## Why we asked it
E-060: right-sizing at birth matched 64-unit fixed experts with 61 % of the parameters in a lab with simplified routing; the open step was the production pool with its conformal gate, vMF signatures, probation and transfer spawn test.

## Prediction (written BEFORE the report-seed runs)
`docs/superpowers/specs/2026-11-01-fsmoe-nbpool-design.md`: P1-P12, fixed after dev runs on seeds 700-702 (g and t on 700). Scored by `fsmoe/scorecard_n.py`: **9 of 12 held** (P1 to P8, P11); missed P9 (at most 30 % of NB's experts end with a network: f10 2 %, f30 4 %, g 34 %, t 7 %), P10 (wall time at most 1.6 times on f30: 1.79), P12 (g: U128 not better than U8 by more than 0.02: +0.039).

## Method (how)
- `NbPool(UniPool)`, switch `nb_first`: every module handed to `add_expert` becomes a `NotebookExpert` (an `LdaHead` with the interface the pool uses on a network: `proba`, `__call__`, `clone`, `inherit`, `shrink_perturb`, `state_dict` / `load_state_dict`, `seen`, `train_step`; one inert scalar so the pool can build an optimiser). A notebook expert's batch is split 85 / 15: the training part enters its head once and its replay, the held-out part a reservoir of 150 (counted in the pool's memory; released when the notebook is kept and its replay is full). A child is primed once with its founding samples. Judging every 10 observe calls for non-infant notebooks with 200+ replay samples and 60+ held-out: candidates 8x1, 8x3, 32x1, 64x1 trained 200 steps on the replay, compared as blended answers with the notebook alone; margin 2 points, cheapest within 1.5 points of the best; a promotion replaces `rec.module`, rebuilds the optimiser, hands the notebook head over as `meta["lda"]`. Notebook experts sit out merges and the width-growth step. Requires `spawn_rule="transfer"` and `growth_rule="gap"`.
- The worse-than-chance rule (`nb_chance` 1.0): a parent whose zero-shot loss on a new cluster is at least the log of the classes it has seen cannot vouch for it; the cluster founds a child. Found on dev seeds: the shipped verdict compares the cluster loss with the parent's own reference losses, which are high for notebooks and small networks, so new regimes were absorbed (2-3 regimes lost per run on two dev seeds, accuracy 0.42 / 0.45).
- Beds: `f10`, `f30` growmoe's function-regime stream (viability shootout recipe: 5 inner steps, q 0.02, alpha_spawn 0.01, m_min 32; d = 128, 8 classes, 800 samples per visit, two visits; read after the stream with `run_fast`); `g` the growth lab's mixed world (16 regimes, 8 need width by the oracle; d = 32, 6 classes; 2 inner steps) through the real pool; `t` the FS-MoE three-domain teaching bed (real embeddings, 101 skills, the tuned recipe: 1 inner step, replay 4,000, lda_blend 0.5).
- Arms: U8, U32, U128 (the shipped ULDA pool with networks of 8 / 32 / 128 units at birth), NB, NBc (candidate sizes 8 and 32). Exploratory ablation: U8x and U128x (the shipped pool plus only the worse-than-chance rule), NBn (never judged: notebooks only; g has U8x only). Seeds: dev 700-702, report 710-714.

## Results (copied from the generated files)
`results/fsmoe/N/REPORT_N.md`, `results/fsmoe/SCORECARD_N.md` (means over 5 seeds):
- Accuracy, U128 / NB / NBc: f10 0.493 / 0.497 / 0.493; f30 0.485 / 0.482 / 0.479; g 0.607 / 0.627 / 0.634; text bed teaching / new wording 0.965 / 0.895, 0.961 / 0.890, 0.962 / 0.890. U32: 0.488, 0.479, 0.600, 0.963 / 0.896. U8: 0.355, 0.293, 0.568, 0.963 / 0.891 with 0.64 / 0.45 / 0.82 experts per regime on f10 / f30 / g.
- Network parameters, U128 / NB: f10 345,680 / 3,634; f30 1,037,040 / 11,493; g 133,895 / 11,983; text bed 278,703 / 6,271. Memory (MB), U128 / NB: f10 6.31 / 4.94; f30 18.90 / 14.80 (U32 15.94); g 3.83 / 3.81; text bed 12.85 / 11.01.
- Experts with a network (NB): f10 0.2, f30 1.2, g 5.0 of about 15, text bed 0.2 of 3; sizes given over all beds and seeds: 8x1 12, 8x3 3, 32x1 10, 64x1 8. Judging cost (NB / NBc, G-MACs): f10 31.0 / 10.5; f30 93.0 / 31.4; g 11.3 / 3.8; text bed 86.6 / 36.9. Wall time f30: U128 82 s, NB 147 s (1.79 times), NBc 96 s (1.17 times).
- Worst regime (g): U128 0.408, U8 0.366, NB 0.337, NBc 0.359.
- Ablation (accuracy, experts per regime): f30 U8x 0.471 (1.00) against U8 0.293 (0.45); f10 U8x 0.480 (1.02) against 0.355 (0.64); g U8x 0.596 (0.94) against 0.568 (0.82); U128x against U128: f10 0.491 against 0.493, f30 0.478 against 0.485; text bed unchanged. NBn against NB: f10 0.493 / 0.497, f30 0.481 / 0.482, g 0.620 / 0.627, text bed mean of the two readings 0.926 / 0.925; NBn memory is higher (f30 16.61 against 14.80 MB; it keeps held-out buffers it never uses).

## Interpretation (what we understood, and on what basis)
Plain words: the notebook-first pool does what the shipped pool does with almost no network parameters and 14-22 % less memory, but that is because the closed-form notebooks carry the pool on these beds; the held-out size test hands out networks rarely and a pool of notebooks only scores the same. A weak expert breaks the shipped spawn test, and a one-line rule (a parent worse than chance cannot vouch for a cluster) repairs it. Technical: measured: all of the above on these four beds; the shipped pool trains a network on one batch plus at most 256 replay samples (4,000 in the text recipe), so capacity can hardly be used (g: U128 over U8 by 0.039), which is why the size test finds little to do; the mechanism of the weak-parent flaw is read from the events (extends with p about 0.016 and a cluster loss of about 4 against a reference of about 3, chance being 2.1) and confirmed by the ablation (U8x), not by a separate probe.

## Caveats and what went wrong
- Data-limited beds, 3 experts in the text bed, no deployment stream and no 7-domain stream; accuracy gaps are inside the intervals, memory and parameter gaps are not.
- 9 of 12 predictions held (F-83). The notebooks-only arm and the ablation came after the first results.
- Two design faults caught on dev seeds before any report seed: new regimes absorbed by weak parents (F-84); held-out buffers kept forever by notebooks that can never be judged again.
- A first version of the tests let 5 of 16 mutants survive (a head not handed over at promotion, merge order, the extend priming, the blended judging, the growth-skip); fixed (F-85).

## Consequence
Report: `research/FSMOE_NBPOOL_REPORT_2026-11-01.md`; figure `results/fsmoe/FIGURE_NBPOOL.png`. Q-63 answered; new Q-65, Q-66. Decisions D-56, D-57. Claims C-164..C-170. Nothing becomes a default: the switch is offered for memory-limited use; the worse-than-chance rule is the candidate to put into growmoe's spawn test behind a flag (awaits the user's go).

## Evidence
- `research/FSMOE_NBPOOL_REPORT_2026-11-01.md` - the plain-English report
- `results/fsmoe/N/REPORT_N.md` - generated tables, including the exploratory ablation
- `results/fsmoe/SCORECARD_N.md` - the scorecard
- `results/fsmoe/FIGURE_NBPOOL.png` - the figure
- `docs/superpowers/specs/2026-11-01-fsmoe-nbpool-design.md` - design, dev findings and predictions

## Reproduce
```
python -m fsmoe.nbpool_lab run --bed f30 --seeds 710 --arms U8 U32 U128 NB NBc      (also f10, t, g; ablation arms U8x U128x NBn; one process per seed and arm, OMP_NUM_THREADS=2)
python -m fsmoe.report_n --seeds 710 711 712 713 714
python -m fsmoe.scorecard_n --seeds 710 711 712 713 714
python -m fsmoe.figures_n --seeds 710 711 712 713 714
python -m pytest tests/fsmoe
```
