# Knowing what it knows: knowledge status and reuse for language streams, first proven base on arithmetic in everyday language

Date 2026-10-26. Status: **written design for review** (a research design with one pre-registered first phase). Nothing is built. Authors: the user (intent, examples, the criterion for "understanding") and Claude (structure, tests, bars, marked as Claude's calls where they are).
Companion note with sources and the idea menu: `research/NOVELTY_ROUTING_RESEARCH_2026-10-26.md`. Vertical and horizontal growth are pinned and are not part of this design.

## 1. Intent and why

The user's goal is a system that learns meaning and core understanding, not only next-state or next-token prediction over seen data. Concretely: given a stream of natural-language messages, the system recognises for every part of a message whether it is something new, something it knows only a little, something it can build from things it knows, or something it knows well, and acts accordingly (answer, answer with caution and collect more, learn from relatives, learn from scratch, or hold).

Why this and why now. In everything we have measured on the dynamic expert pool (notebook E-013 to E-043), growing and merging experts was the easier part. The hard part is telling what is truly new from a variation, and which expert should answer. Between different kinds of data (code, maths, text, images) novelty is easy to see; inside one kind (text against text) it is not. Our per-sample check misses 64 to 77 % of inputs from an unseen real-text category (AUROC 0.65 to 0.68), while a test on a group of 16 or more inputs sees almost all of them (E-024). The field's best 2026 continual intent-discovery paper reaches novel precision 0.74 to 0.88 but recalls only 0.39 to 0.61 of the novel inputs and recovers almost none of the true structure (arXiv 2609.17866).

The user's own examples, which define the target:
1. Arithmetic. Addition, subtraction, division and modulo are different skills although their inputs look alike (the same kinds of numbers). A system taught addition and subtraction, then shown worked examples that expose the intermediate additions of multiplication, should see that multiplication is "addition with extra steps" and learn it from what it knows.
2. Tangled language. One message carries several things at once (for example a bank-card problem plus a sum to compute).
3. Cross-domain reuse. A system that learned mazes should reuse that logic in a snake game (the far end of the same ladder).

The user's criterion for "it understood": cross-domain transfer, shown by learning a related new thing from what is already known instead of treating everything unfamiliar as brand new.

## 2. The idea in one picture

Every part of a message gets a **knowledge status** and a **relation**:

| status | meaning | action |
|---|---|---|
| Solid | a known skill covers it and has enough evidence and competence | answer |
| Thin | a known skill covers it but with little evidence or uncertain competence | answer with caution, collect more, ask |
| Related | no skill covers it, but it is close in FUNCTION to known skills (its parents) | build it from the parents |
| Unseen | nothing is close | learn from scratch, or hold, or abstain |
| Conflict | the input looks covered, but the known answer has changed or contradicts | update or branch, never silently merge |

A message is cut into parts; the parts that known skills explain are taken out ("explaining away", as in sparse coding); whatever is left over is the candidate novel piece. "Close in function" is not "looks alike": addition and multiplication have inputs that look the same and are related; maze and snake have inputs that look different and are related.

## 3. Goals, non-goals, anti-goals

Goals (Phase 1, this design):
- G1. A test bed with exact ground truth for skill, surface form, part boundaries, parent skills and the taught state (what the system has and has not been taught, and how much).
- G2. An information-ceiling measurement: for each candidate representation, can ANY signal in it separate the cases we care about (same skill in new wording, near-twins with a different skill, taught against untaught skills)? This tells us whether the detector or the representation is the bottleneck.
- G3. A status machine (Solid, Thin, Related, Unseen, Conflict) per part, calibrated, with a measured false-novelty rate, evaluated on single-part and tangled messages.
- G4. Relation from worked examples: when worked traces of a new skill show steps that are instances of known skills, the system names those skills as parents.
- G5. One pre-registered confirmatory run on fresh seeds, an independent review, notebook entries, and an explicit go or no-go on a paper (section 10).

Non-goals:
- NG1. No growth work (vertical or horizontal), no change to `growmoe/` product code.
- NG2. No composition engine (skills that call skills) and no learning of multiplication from traces: that is Phase 2, and Phase 1 only has to show that the status and relation parts are sound.
- NG3. No maze or snake, no images, no code domain in Phase 1.
- NG4. No claim of "understanding". Claims are limited to the measured behaviours: wording invariance, function sensitivity, calibrated self-knowledge, relation to parents.
- NG5. No large models (nothing above about 2 B parameters), no new downloads, no use of the user's private conversation memory (`E:\universal-ai-memory`) without a separate go.

Anti-goals (what would make this a failure even if every number looks good):
- A1. A result that rests on surface cues (the operator symbol, a template quirk, the message length). The test bed must contain controls a word-matcher fails, and the hashed n-gram representation (R0) must fail at least one of them; if it passes everything, the test bed is broken and no result counts.
- A2. Any threshold chosen on the confirmatory seeds.
- A3. Generator leakage: the label recoverable from a feature that is not meant to carry it (template id, filler source, seed).
- A4. An audit whose lists (units, streams, cases) are copied from the code under test instead of from this document.
- A5. A representation or detector that is right on the synthetic templates and wrong on real phrasing; hence the real-text realism check in section 5.5.

## 4. Constraints (given, not to be re-decided)

- Windows 11 PC, 12 logical cores, 31.7 GB RAM, one RTX 3060 (12 GB). The PC loses power most days: every job is small, checkpointed, writes results atomically, and is restarted by one idempotent command (see the memory note and `research/growmoe_results/rv2d/jobqueue.py` for the pattern). Keep experiments memory-lean: one GPU job at a time, no large dense matrices.
- Offline: use only what is in the local Hugging Face cache. Models present and allowed: `sentence-transformers/all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, `bert-base-uncased`, `distilbert-base-uncased`, `gpt2`, `distilgpt2`, `EleutherAI/pythia-70m`, `google/flan-t5-small`, `Qwen/Qwen2.5-1.5B-Instruct`. Present but excluded by NG5: Qwen2.5-7B-Instruct, Qwen3-30B-A3B, Qwen3-Embedding-8B. Datasets present and usable for filler and realism: banking77, 20 Newsgroups (SetFit), AG News, GSM8K (`gsm8k`, `openai/gsm8k`), `microsoft/orca-math-word-problems-200k`, `TIGER-Lab/MathInstruct`, `glue`, `ultrachat_200k`, `oasst2`.
- Method discipline of this repository (README of the lab notebook): numbers copied by script from generated files; the prediction recorded before the numbers; dev seeds for tuning and bars, test seeds single-use; negative results are entries; pre-registration hash-frozen before the confirmatory run; an independent reviewer who did not write the code; every default change pinned.
- Reuse, read-only: `growmoe/signature.py` (vMF signature), `conformal.py`, `settest.py` (permutation tests), `router.py`, `transfer.py`, `frontend.py` (hashed n-gram frontend), `arith.py` (operand and answer format). New code lives in a new folder `research/knowledge_status/`. Do not edit `growmoe/**` (its code stamp is tied to the closed RV2-D records; see `research/growmoe_results/PRE_STAGE4_PINS.md`).
- Seeds (Claude's call, to be written into DECISIONS D-05 on approval): dev 100 to 199; confirmatory single-use 400 to 429 (30 seeds). Seeds 0 to 99 are used (D-05).
- Notebook: use the next free ids (`python docs/labnotebook/tools/labnb.py new ...`, exclusive creation); never overwrite another track's files.

## 5. The test bed ("arithmetic in everyday clothes")

### 5.1 Skills and numbers
Six skills as in `growmoe/arith.py`: add, sub, mul, div (integer quotient), mod, max. Operands 1 to 2 digits by default (Claude's call). In E-026 even 8,000 examples per skill gave solvers of only add 83 %, sub 68 %, mul 17 % on 1 to 5 digits; with 1 to 2 digits a Solid skill can really be solid, so that a status means something. The builder may widen the range only if the competence table of 5.6 supports it. The relation graph that serves as ground truth for G4 is fixed here, by definition of the operations: mul has parent {add} (repeated addition); div has parent {sub} (repeated subtraction); mod has parents {div, sub}; max has parent {sub} (compare by subtracting); add and sub have no parents (primitives). (Claude's call; the user's example fixes only mul from add and div from sub.)

### 5.2 Surface families (changes of wording that must NOT change the skill)
S1 symbols ("17 + 5", "17 plus 5"), S2 number words ("seventeen plus five"), S3 imperative or question phrasing ("add 5 to 17", "what is 17 and 5 together"), S4 short stories ("I had 17 apples and got 5 more, how many now?"), S5 operand order where the skill is commutative ("5 and 17, added"), S6 chatter around the request (a sentence before and after, drawn from the cached banking77, AG News or ultrachat text, with no arithmetic content), S7 typos, casing and spacing noise, S8 register (formal, casual, terse). Each family has several templates; template ids are recorded and must not predict the skill (anti-goal A3, to be checked by a classifier on template id alone, which must stay at chance). Template pools (Claude's call, from the builder check): there are two DISJOINT pools, one for dev (design, tuning, bars, the learned map R5) and one for the confirmatory run. The confirmatory pool is written by Claude, hash-frozen together with the pre-registration, and is not shown to any model or tuned against until the confirmatory run. A seed varies the numbers, the chatter drawn, the teaching assignment (5.6) and the sampling, not the templates.

### 5.3 Controls that a word-matcher fails
- Near-twins: the same sentence with only the operator word or symbol changed ("17 + 5" and "17 times 5"; the same story with "more" changed to "each").
- Look-alike traps: arithmetic words used for something else ("the 17th plus a note about 5 apples") whose skill is "none".
- Same wording, different numbers of the same skill (must be the same skill).
These come with exact labels and are part of every evaluation, not an afterthought.

### 5.4 Tangled messages and parts
Two-part messages: arithmetic request A plus arithmetic request B (different skills), and arithmetic request plus an unrelated everyday sentence (a banking77 or AG News sentence). Ground truth: character spans for every part, the skill of every part (or "none"), and the taught state of that skill. Evaluate first with gold spans (this isolates representation and status from segmentation), then with automatic segmentation (sentence and clause splitting; sliding windows). Parts must be meaning-sized (a request with its values), never single tokens (see the research note on why). Non-arithmetic parts (chatter, the look-alike traps) belong to a taught class "Other content": known content, always taught, status Solid-other. They are not skills and must never be reported as Unseen arithmetic (Claude's call, from the builder check). An easy control is also included: a part of a content type that was never taught (for example a recipe sentence), which is Unseen across types.

### 5.5 Realism check
A small realism set (about 300 messages) built from GSM8K and orca-math word problems restricted to one-step problems whose operation is clear from the solution text, labelled by the operation in the reference solution (script-extracted, then a human-readable spot check of 50 by the user or the reviewer). Used only to check that the findings are not artefacts of our templates; never for tuning.

### 5.6 Taught states (what the system has been taught)
- Teaching assignments are ROTATED (Claude's call, from the builder check): the taught state must not be tied to a skill's identity. Each run draws an assignment of the six skills to the states Solid, Thin and Unseen; a draw is valid when it respects the relation graph of 5.1 for the trace runs (for the mul trace, add must be Solid). A representative draw is add and sub Solid (about 2,000 examples each), mod Thin (about 20), mul, div, max Unseen. One skill per run is also swept through the counts 5, 20, 100, 500, 2,000 so that Thin and Solid are measured on the same skill.
- Teaching examples carry the skill name and the answer (a supervised open-set setting: a teacher says "this is addition"). Discovering skills from unlabelled streams, as the expert pool's spawn machinery does, is Phase 2 and returns to the design after this base is proven. (Claude's call; the one open point for the user.)
- The status truth is derived from MEASURED evidence and competence, not from the count alone: before any status is scored, a competence table is produced (held-out accuracy of each taught skill's solver at its taught count). Draft rule, to be fixed on dev: Solid = at least 500 examples and held-out accuracy >= 0.90; Thin = at most 100 examples, or accuracy between chance + 0.10 and 0.60; anything in between is a grey zone, reported and excluded from gated scores.
- Teaching uses surface families S1 to S3 only. S4 to S8 are never used in teaching. Wording invariance is measured only there.
- Trace condition (for G4): worked traces of mul, for example "3 times 4: 4 + 4 = 8, 8 + 4 = 12, so 12", and of div ("12 divided by 4: 12 - 4 = 8, 8 - 4 = 4, 4 - 4 = 0, three times, so 3"), in several wordings. What makes a trace step an instance of a known skill (Claude's call, from the builder check): the representation assigns the step to skill s AND the taught solver of s reproduces the step's result. A trace is valid when its steps chain (the running values reach the stated answer). The parent set of the trace is the set of skills with at least one verified step in a valid trace. A fixed parser may extract operands (declared simplification), but a parser that keys on the operator symbol to decide the skill is not allowed (anti-goal A1). Decoys: (i) wrong steps, so the chain does not reach the stated answer; (ii) valid additions that are irrelevant to the header's answer (for example a max request with unrelated additions); (iii) a trace of a skill outside the parent relation written with steps that do not compute it. Counts are the builder's call. In Phase 1 a relation is attributed only from traces: a plain message of an unseen skill, such as "17 times 5", is Unseen, because relation by function without traces is Phase 2.
- Conflict condition: a rule change on a taught skill (for example "plus" now means something else in a declared new context), used only to check that the Conflict status can fire; the number of cases is small and the Conflict results are reported but not gated in Phase 1 (Claude's call).

## 6. What is compared

### 6.1 Representations (the "eyes")
R0 hashed 1-3-gram frontend (`growmoe/frontend.py`), the word-matcher control. R1 MiniLM-L6 mean pooled (the pool's current input; both the 384-d vector and the 128-d random projection the pool uses). R2 all-mpnet-base-v2. R3 bert-base mean pooled. R4 decoder hidden states: `gpt2` and `Qwen/Qwen2.5-1.5B-Instruct`, the last-token state of the message with a fixed instruction suffix, at several middle layers (the "what is being asked" hypothesis). R5 (conditional, only if R1 to R4 fail the ceiling) a small learned map on top of the best of R1 to R4, trained with contrastive pairs: rewordings pulled together, near-twins and different skills pushed apart, using ONLY the K-solid and K-thin teaching data and the S1 to S3 surfaces.

### 6.2 Detectors and signals
Baselines: the pool's per-sample vMF conformal gate; k-nearest-neighbour distance; Mahalanobis and Relative Mahalanobis (background = general text from the cached corpora); classifier confidence and energy; an autoencoder descriptor as in SEMA; the group test of `growmoe/settest.py` at group sizes 1, 4, 8, 16. Candidate: the status machine of section 2, whose inputs per part are (i) local evidence (how many taught examples lie near the part), (ii) competence (held-out accuracy of the skill's solver near the part), (iii) novelty (a calibrated conformal p-value), (iv) relation (which known skills' steps explain the part). The solver of a skill (what gives "competence") is a small learner on parsed operands as in E-026; a fixed parser from message to operands is allowed in Phase 1 and must be declared as a simplification in the write-up (builder's call on the form). A part is assigned to a taught skill by the nearest taught prototype in the representation (labels come from teaching, 5.6); misassignment between skills is measured and reported as its own failure mode (a confusion matrix).
Calibration (Claude's call, from the builder check): the conformal calibration set is held-out data of the taught skills in S1 to S3 only. The headline level is q = 0.05, and the curve over q in {0.01, 0.02, 0.05, 0.10} is reported. This makes the false-Unseen bar a real test of wording invariance, and it may fail without R5, which is an acceptable and informative outcome. A second, separately reported "adapted" condition lets the calibration use a few examples of S4 to S8; it is not gated, and if only the adapted condition passes, the claim is downgraded to "needs adaptation to new wording".

## 7. Measurements and draft bars

All bars below are Claude's first guesses. They are to be re-set from dev-seed results (seeds 100 to 199) and then frozen, with their hash, before any confirmatory seed runs. Chance levels are given so a bar is never read without its floor.

E1, information ceiling (per representation; no detector involved):
- B1a wording invariance: a linear probe for skill identity (6 skills plus "none"), trained on S1 to S3, tested on the held-out surfaces S4 to S8; draft bar >= 0.95 (chance 1/7).
- B1b near-twin separation: AUROC of "same skill" against "near-twin" pairs by distance alone, no training; draft bar >= 0.95 (chance 0.5).
- B1c taught against untaught: across the teaching assignments of 5.6, the distance of a part to the taught set (nearest taught prototype; no training on untaught skills) separates parts of taught skills from parts of untaught skills; AUROC draft bar >= 0.90 (chance 0.5). A probe on a frozen embedding cannot learn "taught", because that is not in the message.
- B2 validity of the test bed: R0 must fail at least one of B1a and B1b by 0.15 or more; otherwise the test bed is invalid (anti-goal A1).
- Decision: a representation that fails B1 is not used for status; if all of R1 to R4 fail, R5 is built; if R5 also fails, Phase 1 ends with the negative result and a new representation plan (still a publishable finding, see section 10).

E2, status on single-part messages (gold spans):
- B3a statuses of arithmetic parts: macro F1 over Solid, Thin, Unseen >= 0.85; false "Unseen" on held-out surface variants (S4 to S8) of Solid skills <= 0.05 at q = 0.05 with calibration on S1 to S3 only (this is the calibration claim, 6.2); Thin separated from Solid with AUROC >= 0.85 using evidence and competence; parts of Other content reported as Unseen arithmetic <= 0.05.
- B3b the baselines of section 6.2 are reported on the same cases with the same calibration, so every claim has its control (repository rule D-06).
E3, tangled messages, with gold spans: B4a a known arithmetic part plus an unseen arithmetic part: the uncovered (residual) part is the truly unseen part in >= 0.85 of messages and the known part is not flagged in >= 0.95. B4b an arithmetic part plus Other content: nothing is flagged as novel arithmetic and the Other part is labelled Solid-other in >= 0.95 (the harder case is B4a; B4b alone would reduce to telling arithmetic from everyday text, which any detector does). B6 with automatic segmentation: B4a within 0.10 of the gold-span number.
E4, relation from traces: B5 the parent set of a mul trace is exactly {add} and of a div trace exactly {sub} in >= 0.90 of traces; decoy traces receive a false relation in <= 0.10.
Realism: B7 the status of the realism set agrees with the template result within 0.10 on Solid against Unseen.
Consequence (Claude's call, from the builder check): the cost of a false Unseen (a skill learned twice, about 4 points in the pool, E-034) and of a missed Unseen (a new skill absorbed by an old one, about 7 points, E-030) is reported in Phase 1 as an OFFLINE ESTIMATE, the measured error counts weighted by those recorded costs, clearly labelled an estimate and not gated. An online test, where statuses drive what is learned and routed in a stream, belongs to Phase 2.

## 8. Phases and order

- P0 (before any code): read the 10 to 15 closest papers in full (the research note lists them; only abstracts were read so far) and write down what is and is not new (section 10 needs it).
- P1a generator, labels, leakage checks, unit tests (test first). P1b E1 on dev seeds, table, representation decision. P1c (conditional) R5. P1d status machine and baselines on dev; freeze the pre-registration (bars, constants, seeds, code stamp) with its hash. P1e confirmatory run once on seeds 400 to 429. P1f independent review by a fresh reviewer, notebook entries, claims, the paper decision (section 10).
- A stop after P1b is expected if the ceiling is the problem: the user is shown the table before more is built.
- Phase 2 (outside this design, listed so Phase 1 does not block it): skills that call skills, learning mul and div from worked traces with learning curves against scratch, then relatedness by function for different-looking inputs, then other domains. Growth returns only after Phase 1.

## 9. Compute and size

Everything is small: about 6 skills x 8 surface families x 500 messages plus a few thousand tangled messages; embeddings of tens of thousands of short texts; the largest model is Qwen2.5-1.5B in fp16 (about 3 GB) for hidden states, one job at a time. Most steps are CPU. Expected effort: a few days of build and a few hours of compute per full run.

## 10. Contribution and paper plan

What may be worth reporting (to be decided by a stated rule, not by hope):
- C-a. The test bed with exact ground truth for wording, function, parts, taught state and relation: a benchmark for "knowledge status", which the field's open-intent and out-of-distribution benchmarks do not provide (they do not separate wording from function, and none that I found grades "thin" or "related").
- C-b. The information-ceiling analysis: which representations can separate near-twins and function changes at all. A clean negative here (for example "sentence encoders cannot see an operator change") is itself useful.
- C-c. The status machine with a calibrated false-novelty rate, if it beats the baselines of 6.2.
- C-d. Relation from worked traces, if B5 holds.
Honest limits to state: synthetic arithmetic with templates; one domain; small models; AI agents wrote and reviewed the work with one human, so an outside human review is wanted before any submission; the related-work reading is not done yet.
Decision gate at P1f: write a paper (a benchmark and analysis paper, or a workshop paper on continual and open-world learning, or a technical report on arXiv) if at least C-a and C-b are done on fresh seeds and the P0 reading finds nothing that already does them. A method claim (C-c, C-d) is added only if its bars passed. Paper B of `docs/labnotebook/PAPER_NOTES.md` (methodology and negative results) can absorb the negative findings.

## 11. What is decided, what is Claude's call, what is left

Decided by the user: the goal and the criterion (transfer, not just novelty); arithmetic as the base because the steps of the ladder are the same idea at different heights and this one is exactly checkable; worked examples with the intermediate additions as the teaching signal; the five-way status per part; growth pinned.
Claude's calls (to be checked): the five status names and actions; operand sizes (1 to 2 digits); the relation graph (mod from div and sub, max from sub); the surface families and controls; the representation ladder; rotated teaching assignments, taught counts and the competence-derived status truth; the "Other content" class; disjoint dev and confirmatory template pools; the calibration protocol (S1 to S3 only, q = 0.05, an adapted variant reported separately); what counts as a verified trace step, the decoys, and relation only from traces in Phase 1; offline consequence estimate; seeds 100 to 199 and 400 to 429; all draft bars and the non-gating of Conflict; the P0 reading before code; the paper rule.
The one open point for the user: Phase 1 teaches with labelled skills (a teacher names the skill), so it tests a supervised open-set status machine; finding skills by itself in an unlabelled stream is Phase 2.
Left to the builder: template and message counts, the form of the parser and the skill solvers, the segmentation methods, the probe and calibration details, scheduling of runs within the constraints, code structure under `research/knowledge_status/`.
