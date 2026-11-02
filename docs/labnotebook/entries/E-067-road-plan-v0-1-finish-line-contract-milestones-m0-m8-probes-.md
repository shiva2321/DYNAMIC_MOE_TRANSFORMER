# E-067 - Road plan v0.1: finish-line contract, milestones M0-M8, probes, stuck protocol

- **ID:** E-067
- **Date:** 2026-11-02
- **Track:** W
- **Type:** design
- **Status:** in progress (DRAFT v0.1; waiting for the user's review and the decisions in ROADMAP section 10)
- **Evidence grade:** H (a plan: its statements are proposals and hypotheses; the inventory in it is copied from claims of grade A and B; the literature digest is grade L, abstract or search-summary level, some of it from secondary pages)
- **Code / commit:** not committed (standing rule D-15); documents under `docs/roadmap/` and the figure script `docs/roadmap/make_roadmap_figure.py`; no code of the pool was touched
- **People:** Claude designed and wrote it on the user's request of 2026-11-02 ("brainstorm, deep research and analysis of all ... a definitive road plan which we can follow and modify"), after the user's answers on the finish line, pooled routing, a knowledge lifecycle with removal, custom parts, proof to the field, one machine, and "a failed milestone is a reason to explore, not to stop"; the builder check was done by Claude reading the document as the builder (no independent reviewer yet)

## Question
What is the shortest honest path from the lab prototype (a router and a pool of small experts on frozen sentence vectors) to the user's finish line (a complete working mixture-of-experts model that learns continually without unwanted forgetting, routes and notices novelty and nuance, and is efficient against a dense model), and what has to exist at each point on it?

## Why we asked it
The user asked, after the status answer of 2026-11-02 (lab prototype, ladder step 1 done and step 2 partial), for a living guidepost. It follows E-053 to E-066 and the open-question register.

## Prediction (written BEFORE the numbers, if there was one)
None: a design step. The plan itself carries proposed bars for each milestone, to be frozen in each milestone's pre-registration.

## Method (how)
- Re-read the notebook registers (`CLAIMS.md`, `DECISIONS.md`, `FAILURES.md`, `OPEN_QUESTIONS.md`, `PAPER_NOTES.md`, `REFERENCES.md`, `GLOSSARY.md`), `research/README.md`, the entries E-025, E-060, E-061, E-066, `results/unimoe/ANALYSIS_P4.md`.
- Checked the machine on 2026-11-02: GPU, RAM, CPU, disk, installed libraries; repository state (157 commits ahead of the remote, 131 uncommitted entries).
- About 30 web searches and 5 page fetches on 2026-11-02 on: closed-form and analytic continual learning, compute-budgeted evaluation, language-model continual pre-training, sparse memory finetuning, nested learning, mixture-of-experts continual learning and routing drift, unlearning in MoE, retrieval against fine-tuning, MoE inference efficiency, GPU energy measurement, small models and embedding models, benchmarks, plasticity loss, scaling laws of forgetting, open-world novelty, predictive coding. Results are in `docs/roadmap/RESEARCH_DIGEST.md` with a tag on each entry (L, R).
- The deep-research skill available here calls a paid external service (Gemini, 2-5 USD per task); it was not used.
- Drafted the files; fixed the figure (two rounds); builder check; corrected the gaps it found (equal compute defined, devices, confirmatory runs only on the user's go, a download fallback for M2, label quality for the nuance suite).

## Results (what came out)
Files (living documents, not hashed as evidence): `docs/roadmap/ROADMAP.md` (goal, finish-line contract F1-F8, where we are, thesis and bets B1-B8, the road, branch points BP1-BP7, stuck protocol, change protocol, rules, risks R1-R11, decisions owed), `MILESTONES.md` (M0-M8 and probes P1-P3 with proposed bars and costs), `STATE_BOARD.md` (levels L1-L5 by ability; 33 components with status, evidence, gap, milestone, next action), `IDEA_BOARD.md` (30 ideas in seven obstacle groups plus 10 ideas already tried and failed), `RESEARCH_DIGEST.md`, `CHANGELOG.md`, `ROADMAP_FIGURE.png`.
Structure: six phases, nine milestones, three probes, about 59-101 sessions in all (most in M5, the trained-representation small language model). Three early probes: a pool of small language models (P1), exact removal by downdating (P2), a cheap-first novelty cascade (P3).
Corrections made on the way (also told to the user): the training-time efficiency summary was incomplete (E-025 whole-run time 0.22-0.35x of the same-memory network against fixed-size experts 1.71x) and C-25 said "7-14x" where its table gives 6.4-11x (F-107).

## Interpretation (what we understood, and on what basis)
Plain words: the biggest gaps between us and the finish line are a trained representation, a measured and honest efficiency story against the baselines people really use (compute-matched replay and closed-form heads on frozen features), a cheap internal novelty detector, a pool that stays accurate and cheap at thousands of experts, and independent replication. The plan climbs the five evidence levels column by column and keeps a protocol so that a failed milestone becomes the next experiment, never a dead end. Technical: measured: the inventory (cited claims); assumed: every bar, every effort estimate, the idea chances, the hypothesis that exact removal is possible (grade H until P2), the hypothesis that the combination of calibrated task-free novelty, pooled routing, closed-form fast memory, lifecycle with exact removal and a single-GPU compute-matched evaluation is not yet published (abstract-level reading only; the check is part of M8).

## Caveats and what went wrong
- Literature at abstract or search-summary level, some of it from secondary pages; 2026 identifiers not independently confirmed; nothing read in full.
- All bars are proposals; the efforts are wide estimates; the idea chances are my judgement.
- No independent reviewer read the plan; the builder check was done by the author.
- The plan was drafted in one session; the figure needed two rounds of layout fixes.

## Consequence
Waiting for the user's review. If approved: record the decisions as D-rows, start M0 (backup first, after the user's answer on commit and push), write the M1 pre-registration. New open questions Q-74 to Q-79 (the milestones and probes as questions). Next entries: M0, then P2 and P3, then M1.

## Evidence
- `docs/roadmap/ROADMAP.md` - the road plan (living; not in the evidence manifest because it changes)
- `docs/roadmap/MILESTONES.md`, `STATE_BOARD.md`, `IDEA_BOARD.md`, `RESEARCH_DIGEST.md`, `CHANGELOG.md` - the parts
- `docs/roadmap/ROADMAP_FIGURE.png` - summary figure

## Reproduce
```
python docs/roadmap/make_roadmap_figure.py
```
