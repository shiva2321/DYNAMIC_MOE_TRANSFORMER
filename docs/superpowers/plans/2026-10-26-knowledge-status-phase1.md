# Knowledge Status, Phase 1: the test bed, the ceiling test and the status machine - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** build and pre-register the first proven base of "a system that knows what it knows": a test bed with exact ground truth (arithmetic in everyday language), an information-ceiling test of representations, a five-way knowledge status per part of a message, and relation-from-worked-traces.

**Architecture:** a new folder `research/knowledge_status/` of small flat modules (generator, representations, taught states, ceiling metrics, status machine and baselines, segmentation, traces, runners, freeze, report), tested in `tests/knowledge_status/`. It reads `growmoe/` (signature, conformal, set tests, arithmetic format, hashed frontend) and never edits it. Every runner is idempotent, writes atomically and resumes after a power cut. Four stop points (after the ceiling table, after dev results, before the freeze, before the single-use confirmatory run) are where the user is told the news in plain words and decides.

**Tech Stack:** Python 3.12, numpy, torch (CPU; GPU only for decoder hidden states), scikit-learn (probes), sentence-transformers and transformers from the local Hugging Face cache (offline), pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-knowledge-status-and-routing-design.md` (approved by the user 2026-10-26). Sources and idea menu: `research/NOVELTY_ROUTING_RESEARCH_2026-10-26.md`.

## Global Constraints

- Offline only: set `HF_HUB_OFFLINE=1`; use only the local cache. Allowed models: `sentence-transformers/all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-multilingual-MiniLM-L12-v2`, `bert-base-uncased`, `distilbert-base-uncased`, `gpt2`, `distilgpt2`, `EleutherAI/pythia-70m`, `google/flan-t5-small`, `Qwen/Qwen2.5-1.5B-Instruct`. Excluded: Qwen2.5-7B-Instruct, Qwen3-30B-A3B, Qwen3-Embedding-8B (nothing above about 2 B parameters, no new downloads). Do not open `E:\universal-ai-memory`.
- Datasets allowed: banking77, 20 Newsgroups (SetFit), AG News, `gsm8k` and `openai/gsm8k`, `microsoft/orca-math-word-problems-200k`, `TIGER-Lab/MathInstruct`, `glue`, `ultrachat_200k`, `oasst2`, `ai2_arc`.
- Hardware and robustness: 12 logical cores, 31.7 GB RAM, one RTX 3060 (12 GB); one GPU job at a time, small tensors; the PC loses power most days, so every job is small, checkpointed, writes results atomically (write a `.tmp`, fsync, `os.replace`; pattern: `research/growmoe_results/rv2d/rv2d_lib.py::atomic_write_text`), and restarts with one idempotent command. Never swallow an exception silently (the commit hook blocks it); catch the specific type.
- Seeds: dev 100 to 199; confirmatory single-use 400 to 429. Seeds 0 to 99 are used (DECISIONS D-05).
- Method discipline: numbers are copied by script from generated files; the prediction is written before the numbers; thresholds are chosen on dev seeds only; bars are hash-frozen before the confirmatory run; negative results are notebook entries; the independent reviewer is a fresh session; every claim keeps its control (D-06).
- Do not edit `growmoe/**` (its code stamp is tied to closed records). New code only under `research/knowledge_status/` and `tests/knowledge_status/`.
- Notebook: next free ids via `python docs/labnotebook/tools/labnb.py new ...` (exclusive creation); never overwrite another track's files; `labnb.py verify` must pass; ledger files are ASCII with CRLF, entries ASCII with LF; backticked paths in entries must exist.
- Plain-English rule (the user asked for it): after every task append a dated block to `research/knowledge_status/PROGRESS.md` with three short parts in everyday words: what we did, what we learned, what comes next; and say the same in chat. Define any technical term the first time it appears; never drop the term, explain it.
- Test first, always: no code without a failing test that was run and seen to fail.

## Review Focus (input classes the spec implies but does not test directly)

1. Chatter that accidentally carries arithmetic cues (digits, "plus", "mod" inside "modern"): pinned in Task 4.
2. Arithmetic edge cases: zero divisors, negative subtraction results, commutative skills reordering the operands, zero and one: pinned in Task 3.
3. Awkward text for the encoders: empty, very long, non-ASCII, typo-heavy strings: pinned in Task 6.
4. Leakage between splits and pools: the same or near-same text in teaching and test sets, or a template shared by the dev and confirmatory pools: pinned in Tasks 3, 7 and 14.
5. Interrupted runs and corrupt cache files: a truncated result or cache file must be ignored and recomputed, never read: pinned in Task 8.
6. Metrics with an absent class (macro F1 when a status never occurs in a run): pinned in Task 10.

## File Structure

`research/knowledge_status/`: `skills.py` (skills, relation graph, truth), `pools.py` and `pool_dev.py` (template pools), `generator.py` (messages, parts, spans), `corpora.py` (offline chatter and content loaders), `reps.py` (representations and cache), `assignments.py` (teaching assignments, solvers, competence, status truth), `ceiling.py` (E1 metrics), `learned_map.py` (R5, conditional), `baselines.py`, `status.py` (the status machine), `segment.py`, `traces.py`, `realism.py`, `evaluate.py` (E2 to E4 metrics), `runlib.py` (atomic writes, code stamp, idempotent cells), `run_e1.py`, `run_e2.py`, `run_e3.py`, `freeze_prereg.py`, `report.py`, `audit.py`, `PROGRESS.md`, `RELATED_WORK.md`. The confirmatory pool lives in `pool_confirm.py`, created in Task 14. Tests mirror the modules in `tests/knowledge_status/` with a `conftest.py` that puts the folder on `sys.path`.

---

### Task 1: Related-work reading (P0, before any code)

**Files:** Create `research/knowledge_status/RELATED_WORK.md`.

**Plain words:** we have only read abstracts of the nearest papers. Before building, read the ten to fifteen closest in full so we know what is already done and what we may honestly call new.

**Interfaces:** Produces `RELATED_WORK.md` with, per paper: citation and link, what it does, data and metrics, what it does NOT test, how it differs from the design (sections 5 to 7), and a final paragraph "what is and is not new in this design". Reading list (from the research note): TOP-ID (arXiv 1904.08524), uncertainty-aware continual intent discovery (2609.17866), ADB and DA-ADB (2012.10209, 2203.05823), Relative Mahalanobis (2106.09022), Mahalanobis for transformers (2101.03778), Failing Loudly (1810.11953), MINAS and OLINDDA stream novelty, SEMA (2403.18886), Remembering Transformer (2404.07518), FLEX (2608.01437), component-based OOD (2604.21546), Kim et al. open-world continual learning (2304.10038), the factual-sensitivity embedding paper (neurips.cc/virtual/2026/164747), compositional generalization benchmarks such as SCAN, and a library-learning paper such as DreamCoder (recalled, find and verify).

- [ ] **Step 1: Read each paper in full** (WebFetch or a downloaded PDF; if a paper cannot be read in full say so and tag it abstract-only).
- [ ] **Step 2: Write the file** with the structure above and the tag per paper `[full]` or `[abstract]`.
- [ ] **Step 3: Verify** each paper has all five fields and the closing paragraph names at least the claims C-a to C-d of spec section 10 as "new", "partly new" or "already done", each with the paper that decides it.
- [ ] **Step 4: Plain-English update** to `PROGRESS.md` (create it) and in chat: what we found already exists.
- [ ] **Step 5: Commit** `docs(ks): related-work reading for knowledge-status phase 1`.

---

### Task 2: Scaffolding, skills and the relation graph

**Files:** Create `research/knowledge_status/skills.py`, `runlib.py`, `PROGRESS.md` (if absent), `tests/knowledge_status/conftest.py`, `tests/knowledge_status/test_ks_skills.py`; modify `.gitignore` (add `research/knowledge_status/_cache/`); add DECISIONS row D-39 (seeds 100 to 199 dev, 400 to 429 confirmatory single use, reserved from 2026-10-26) with `labnb.py` conventions.

**Interfaces:**
- Produces in `skills.py`: `SKILLS = ("add","sub","mul","div","mod","max")`; `ARITH_OP_ID = {"add":0,"sub":1,"mul":2,"mod":3,"div":4,"max":5}` (as `growmoe/arith.py`); `PARENTS: dict[str, frozenset[str]]` = add and sub empty, mul {add}, div {sub}, mod {div, sub}, max {sub}; `COMMUTATIVE = frozenset({"add","mul","max"})`; `truth(skill: str, a: int, b: int) -> int` (integer quotient for div, remainder for mod, never called with b = 0 for div or mod).
- Produces in `runlib.py`: `atomic_write_text(path, text)`, `code_stamp(files: Sequence[Path]) -> str` (sha256 of the files with CRLF folded to LF, first 16 hex), `cell_done(path, stamp) -> bool` (False for a missing, unparsable or other-stamp file), `set_threads(n: int = 3)`.

- [ ] **Step 1: Write failing tests** in `test_ks_skills.py`: `test_truth_matches_growmoe_arith` (for 200 random pairs `truth` equals `growmoe.arith.answers` with the mapped op id); `test_parent_graph_is_acyclic_and_rooted_at_add_and_sub`; `test_mul_parent_is_add_and_div_parent_is_sub`; `test_commutative_skills_give_the_same_answer_when_swapped` and `test_non_commutative_skills_do_not` (sub, div, mod on a case where they differ); `test_cell_done_ignores_truncated_and_foreign_stamp_files`; `test_atomic_write_leaves_no_partial_file_when_the_write_fails` (monkeypatch `os.replace` to raise, assert the target is untouched).
- [ ] **Step 2: Run** `python -m pytest tests/knowledge_status/test_ks_skills.py -q` - expect FAIL (modules missing).
- [ ] **Step 3: Implement** the two modules and the conftest (`sys.path.insert(0, research/knowledge_status)` and `os.environ.setdefault("HF_HUB_OFFLINE","1")`).
- [ ] **Step 4: Run** the same tests - expect all PASS.
- [ ] **Step 5: Register the seeds** (D-39) and commit `feat(ks): skills, relation graph, run library`.

---

### Task 3: Surface families, the dev template pool and single-part messages

**Files:** Create `research/knowledge_status/pools.py`, `pool_dev.py`, `generator.py`, `tests/knowledge_status/test_ks_generator.py`.

**Plain words:** this makes the messages we will test on: the same arithmetic request written eight different ways (symbols, number words, imperative, little stories, reordered, noisy, casual or formal), each with an exact label of which skill it needs.

**Interfaces:**
- Produces in `pools.py`: `load_pool(name: str) -> dict[str, list[Template]]` for `"dev"`; `"confirm"` raises `PermissionError` unless `os.environ["KS_ALLOW_CONFIRM"] == "1"` (so dev code cannot touch it by accident). `Template` has `template_id: str`, `family: str` in `S1..S8`, and a format string with slots `{a}`, `{b}`, `{aw}`, `{bw}` (number words), and an operator cue field.
- Produces in `generator.py`: `Part(start: int, end: int, kind: str, skill: str | None, a: int | None, b: int | None, answer: int | None)` with `kind` in `"skill" | "other" | "unseen_content"`; `Message(text: str, parts: tuple[Part, ...], family: str, template_ids: tuple[str, ...], pool: str, seed: int)`; `number_to_words(n: int) -> str` (0 to 9999, negatives as "minus ..."); `make_single(rng: np.random.Generator, pool: str, skill: str, family: str, digits: tuple[int, int] = (1, 2)) -> Message`.

- [ ] **Step 1: Write failing tests**: `test_number_to_words_known_values` (0, 7, 13, 21, 100, 1234 and -8); `test_every_family_renders_every_skill`; `test_labels_are_exact` (a test-side parser for S1 and S2 recovers `a`, `b` and the skill from the text, and `answer` equals `truth`); `test_commutative_reorder_keeps_the_label` (S5); `test_no_zero_divisor_for_div_and_mod` (500 draws); `test_negative_subtraction_renders_and_labels` ("seventeen minus twenty-five" gives -8); `test_same_seed_same_messages`; `test_template_id_alone_does_not_predict_the_skill` (logistic regression on one-hot template id, accuracy within chance + 0.05 over 3,000 messages); `test_dev_pool_templates_are_unique_and_nonempty_per_family`.
- [ ] **Step 2: Run** - expect FAIL.
- [ ] **Step 3: Implement** the pool (at least 6 templates per family S1 to S8, authored by hand, covering all six skills for S1 to S3, S5 only for commutative skills for the swapped form) and the generator. Typo noise (S7) must keep the numbers intact; register S8 changes tone only.
- [ ] **Step 4: Run** - expect PASS.
- [ ] **Step 5: Commit** `feat(ks): dev template pool and single-part messages`.

---

### Task 4: Controls and corpora (near-twins, traps, Other content)

**Files:** Create `research/knowledge_status/corpora.py`; extend `generator.py`; tests `tests/knowledge_status/test_ks_controls.py`.

**Interfaces:**
- Produces in `corpora.py`: `load_other_sentences(n: int, seed: int, source: str = "banking77|agnews") -> list[str]` (cached local data, sentences without digits, without any arithmetic cue word from `ARITH_CUES`, length 5 to 40 words), `load_unseen_content(n: int, seed: int) -> list[str]` (ai2_arc science questions, same filters), `ARITH_CUES: frozenset[str]` (plus, minus, times, divided, mod, remainder, add, subtract, multiply, sum, total, product, quotient, max, larger, and the symbols + - x * / %).
- Produces in `generator.py`: `make_near_twin(rng, msg: Message, to_skill: str) -> Message` (same sentence, only the operator cue changed, keeping the operands), `make_trap(rng, pool: str) -> Message` (arithmetic words used for something else; one `other` part, skill None), `make_other(rng, pool, seed) -> Message`.

- [ ] **Step 1: Write failing tests**: `test_chatter_has_no_digits_or_operator_cues` (1,000 sentences from each source; also catches "modern" versus the cue "mod" by whole-word matching); `test_near_twin_differs_only_in_the_operator_cue` (token diff is exactly the cue; operands equal; skills differ); `test_trap_has_no_skill_label`; `test_unseen_content_is_disjoint_from_other_content`; `test_loaders_work_offline_and_fail_loudly_when_the_cache_is_missing` (point the cache path at an empty dir, expect a clear `FileNotFoundError` message naming the dataset).
- [ ] **Step 2: Run** - expect FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Commit** `feat(ks): near-twin, trap and Other-content controls`.

---

### Task 5: Tangled messages and gold spans

**Files:** extend `generator.py`; tests `tests/knowledge_status/test_ks_tangled.py`.

**Interfaces:** `make_tangled(rng, pool: str, kinds: tuple[str, str], skills: tuple[str | None, str | None] = ...) -> Message` where `kinds` are `"skill"` or `"other"`; two parts in order or swapped at random; `Message.parts` hold exact character spans.

- [ ] **Step 1: Write failing tests**: `test_spans_cut_the_text_exactly` (for every part, `text[start:end]` re-renders that part and the spans do not overlap); `test_skill_part_labels_survive_the_join`; `test_both_orders_occur`; `test_known_plus_unseen_pairs_use_two_different_skills`; `test_two_requests_in_one_message_are_both_recoverable` (the test-side parser finds both operand pairs).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement** (join with a sentence boundary or a connective drawn from a small list). **Step 4: Run** - PASS. **Step 5: Commit** `feat(ks): tangled messages with gold spans`.

---

### Task 6: Representations and the cache

**Files:** Create `research/knowledge_status/reps.py`; tests `tests/knowledge_status/test_ks_reps.py`.

**Plain words:** a "representation" is how a sentence is turned into numbers so it can be compared. We try several, from a plain word-counter (the control) to language-model internals, and keep every result on disk so a power cut costs nothing.

**Interfaces:**
- Produces: `REP_IDS`; `class Representation` with `id: str`, `dim: int`, `encode(texts: Sequence[str]) -> np.ndarray` (float32, shape (n, dim), unit length, finite, deterministic); `get_rep(rep_id: str) -> Representation` for `R0_hash` (`growmoe.frontend.HashFrontend`), `R1_minilm384`, `R1_minilm128` (the pool's fixed random projection, as `research/growmoe_results/m0/run_m0.py::HybridFrontend`), `R2_mpnet`, `R3_bert` (mean pool), `R4_gpt2_L{layer}` and `R4_qwen_L{layer}` (last-token hidden state at a middle layer after the fixed suffix `"\nThe operation requested is"`); `cached_encode(rep: Representation, texts: Sequence[str], cache_dir: Path) -> np.ndarray` (one `.npz` per rep id and 1,000-text block keyed by a sha256 of the texts, written atomically).

- [ ] **Step 1: Write failing tests** with the small models only (`R0_hash`, `R1_minilm128`, `R4_gpt2_L6` on `distilgpt2` stand-in via a `model_name` override): `test_unit_length_finite_and_deterministic`; `test_empty_very_long_and_non_ascii_texts_encode` ("", 5,000 characters, "caf\u00e9 \u4e09 plus \u4e94", a typo-heavy string); `test_cache_hit_returns_identical_vectors_without_recomputing` (count calls); `test_truncated_cache_file_is_ignored_and_recomputed`; `test_offline_mode_never_touches_the_network` (set `HF_HUB_OFFLINE=1`; a missing model raises a clear error naming the model).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** Qwen runs in fp16 on the GPU, batch size 16, one job at a time; layers chosen as `{L//3, L//2, 2L//3}` of the model depth. **Step 4: Run** - PASS.
- [ ] **Step 5: Commit** `feat(ks): representations R0 to R4 with an atomic cache`.

---

### Task 7: Teaching assignments, solvers, the competence table, status truth

**Files:** Create `research/knowledge_status/assignments.py`; tests `tests/knowledge_status/test_ks_assignments.py`.

**Plain words:** this decides who "knows" what in each run (which skill is well known, barely known, or never seen) and measures how well each taught skill can really answer, so that "well known" is a measured fact and not just a label.

**Interfaces:**
- Produces: `draw_assignment(rng, need_solid: frozenset[str] = frozenset(), n_solid: int = 2, n_thin: int = 1) -> dict[str, str]` (skill to `"solid" | "thin" | "unseen"`, with the skills in `need_solid` forced Solid and the rest drawn); `COUNTS = {"solid": 2000, "thin": 20}` and `COUNT_SWEEP = (5, 20, 100, 500, 2000)`; `class Solver` with `fit(a: np.ndarray, b: np.ndarray, y: np.ndarray)` and `predict(a, b) -> np.ndarray` (a small MLP on digit one-hot features of the parsed operands, as `growmoe.arith.encode_problems`); `competence_table(assignment, counts, rng, pool) -> dict[str, dict]` (per taught skill: examples, held-out accuracy, chance level); `status_truth(count: int, accuracy: float, chance: float) -> str` returning `"solid" | "thin" | "grey"` by the draft rule (Solid: count >= 500 and accuracy >= 0.90; Thin: count <= 100 or chance + 0.10 <= accuracy <= 0.60; otherwise grey), with the thresholds in one `THRESHOLDS` dict that dev may change and the freeze records.
- Teaching examples use families S1 to S3 only and carry `(text, skill, answer)`; test sets use S4 to S8.

- [ ] **Step 1: Write failing tests**: `test_assignment_is_not_tied_to_the_skill_identity` (over 200 draws every skill is Solid, Thin and Unseen at least once); `test_need_solid_is_respected` (mul run forces add Solid; div run forces sub Solid); `test_status_truth_boundaries` (count 500 with accuracy 0.90 is solid, 499 is grey, count 20 is thin, accuracy 0.95 at count 50 is thin because count <= 100 wins); `test_no_identical_text_between_teaching_and_test_sets` (normalised lowercase text hash; also no teaching template id in the test set); `test_competence_table_reports_chance_and_accuracy` (add reaches >= 0.9 at 2,000 examples with 1 to 2 digit operands on one dev seed; if this fails the operand range must shrink, not the bar); `test_solver_is_deterministic_for_a_seed`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Commit** `feat(ks): teaching assignments, solvers and measured status truth`.

---

### Task 8: E1, the information-ceiling test (stop point 1)

**Files:** Create `research/knowledge_status/ceiling.py`, `run_e1.py`, `tests/knowledge_status/test_ks_ceiling.py`.

**Plain words:** before building any detector, ask "can ANY signal in this representation tell the cases apart at all?" If even the best trained probe cannot, no detector built on that representation can, and we must change the representation instead.

**Interfaces:**
- Produces in `ceiling.py`: `probe_accuracy(Xtr, ytr, Xte, yte) -> float` (standardised logistic regression); `pair_auroc(X: np.ndarray, same: Sequence[tuple[int, int]], other: Sequence[tuple[int, int]]) -> float` (cosine distance, AUROC that "same" pairs are closer); `taught_auroc(protos: np.ndarray, X: np.ndarray, is_taught: np.ndarray) -> float` (distance to the nearest taught prototype; AUROC of taught against untaught parts); `run_ceiling(rep_id: str, seed: int, pool: str = "dev") -> dict` returning B1a (probe trained on S1 to S3, tested on S4 to S8, 7 classes: six skills plus "none"), B1b (same skill against near-twin pairs), B1c (across teaching assignments) and the chance levels.
- `run_e1.py --pool dev --seeds 100-104 --reps all`: one cell per `(rep, seed)`, results to `research/knowledge_status/results/e1/{rep}__s{seed}.json` with the code stamp, resumable (`cell_done`), then `report_e1()` writes `results/E1_TABLE.md` (mean and range over seeds, with the B2 validity line for R0).

- [ ] **Step 1: Write failing tests** on constructed embeddings: `test_pair_auroc_is_one_for_separated_and_half_for_random`; `test_probe_accuracy_matches_chance_on_noise` (within 0.05 of 1/7 over 5 repeats); `test_taught_auroc_is_one_when_untaught_parts_are_far`; `test_run_ceiling_on_a_fake_oracle_representation` (a stub representation that encodes skill identity perfectly gives B1a 1.0, and one that encodes only the template gives B1a near chance); `test_resume_skips_complete_cells_and_ignores_truncated_files` (create one good and one truncated result file, rerun, assert only the truncated cell is recomputed).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Run E1 on dev** (`python research/knowledge_status/run_e1.py --pool dev --seeds 100-104 --reps all`); expect hours on CPU, one GPU job at a time. Before the numbers appear write the prediction into `PROGRESS.md` (spec section 7: R0 fails the near-twin test; off-the-shelf sentence encoders blur "17 + 5" and "17 x 5").
- [ ] **Step 6: STOP POINT 1.** Verify `E1_TABLE.md` exists and passes the B2 validity check (R0 fails B1a or B1b by >= 0.15; otherwise the test bed is invalid and the plan stops to fix it). Tell the user in plain words which representations can and cannot see the differences, and wait for the decision on Task 9 (build the learned map or not).
- [ ] **Step 7: Commit** `feat(ks): information-ceiling test and dev table`.

---

### Task 9 (conditional): R5, a learned map that ignores wording

**Files:** Create `research/knowledge_status/learned_map.py`, `tests/knowledge_status/test_ks_learned_map.py`. Run only if Task 8 says R1 to R4 fail B1.

**Interfaces:** `class LearnedMap` with `fit(X: np.ndarray, skill: np.ndarray, family: np.ndarray, template: np.ndarray) -> None` (a small linear or one-hidden-layer map trained contrastively: same skill and different surface pulled together, near-twins and different skills pushed apart; uses ONLY K-solid and K-thin teaching data in S1 to S3) and `transform(X) -> np.ndarray` (unit length).

- [ ] **Step 1: Write failing tests**: `test_fit_refuses_data_from_held_out_families` (raises `ValueError` if any row has a family in S4 to S8); `test_fit_refuses_untaught_skills` (raises if a skill outside the taught set is present); `test_map_separates_a_constructed_toy_with_nuisance_directions` (a toy where one direction is wording and one is skill: after the map the skill direction dominates).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Re-run E1 for R5** on the same dev seeds and add it to `E1_TABLE.md`. If R5 also fails B1, STOP and report the negative result (spec section 7 decision).
- [ ] **Step 6: Commit** `feat(ks): learned wording-invariant map (R5)`.

---

### Task 10: Detectors, baselines and the status machine

**Files:** Create `research/knowledge_status/baselines.py`, `status.py`, `tests/knowledge_status/test_ks_status.py`.

**Plain words:** this is the part that says, for a piece of a message, "solid, thin, unseen, other-content, or in conflict", and says how sure it is. We also build the standard methods from the literature so every claim has its control.

**Interfaces:**
- Produces in `status.py`: `SkillGroup(name: str, protos: np.ndarray, n_examples: int, solver: Solver | None, competence: float)` (the only way the machine sees a skill, so a label-free mode can replace the labelled one in Phase 2; no label may be used at test time); `Assessment(status: str, skill: str | None, p_value: float, evidence: int, competence: float, group_size: int)` with `status` in `"solid" | "thin" | "unseen" | "other" | "related" | "conflict"`; `class StatusMachine(groups: dict[str, SkillGroup], other: SkillGroup, cal_vectors: np.ndarray, cal_groups: np.ndarray, q: float = 0.05, thresholds: dict = THRESHOLDS)` with `assess(part_vec: np.ndarray) -> Assessment` and `assess_group(part_vecs: np.ndarray) -> Assessment` (the group test of `growmoe.settest.perm_pvalue` against the nearest group's source vectors). The conformal p-values use `growmoe.signature.VmfSignature.pvalue`; calibration data are held-out S1 to S3 vectors of taught skills only. `related` is never returned by `assess` (relation comes from traces, Task 12). `conflict` is returned only through `flag_conflict(part_vec, known_answer_disagreement: bool)` and is not gated.
- Produces in `baselines.py`: `vmf_conformal`, `knn_distance`, `mahalanobis`, `relative_mahalanobis` (background Gaussian from the Other-content corpora), `classifier_confidence`, `energy`, `autoencoder_descriptor` (reconstruction-error z-score), each with `fit(X, y)` and `novelty_score(X) -> np.ndarray` (higher is newer), plus `group_test(ref, batch, size) -> float` for sizes 1, 4, 8, 16.

- [ ] **Step 1: Write failing tests**: `test_false_novelty_is_at_most_q_on_exchangeable_data` (synthetic von Mises-Fisher groups; at q = 0.05 the false-unseen rate over 5,000 held-out points of a taught group is <= 0.07); `test_unseen_group_is_flagged_when_far` (rate >= 0.9); `test_thin_is_reported_for_a_group_with_few_examples` (n = 20 gives thin, n = 2,000 gives solid, same data otherwise); `test_other_content_is_never_reported_as_unseen_arithmetic_on_toy`; `test_machine_needs_only_skill_groups_and_a_vector` (construct from `SkillGroup` objects with no label arrays; pass a vector; no label argument exists in any signature of `assess`); `test_macro_f1_ignores_an_absent_class_and_reports_it` (when "thin" never occurs the score is computed over the present classes and `absent` lists it); `test_every_baseline_scores_higher_for_far_points` (same toy for all seven).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Commit** `feat(ks): status machine and baselines`.

---

### Task 11: Segmentation, E2 and E3 evaluation, dev results (stop point 2)

**Files:** Create `research/knowledge_status/segment.py`, `evaluate.py`, `run_e2.py`, `tests/knowledge_status/test_ks_evaluate.py`.

**Interfaces:**
- Produces in `segment.py`: `gold_parts(msg: Message) -> list[tuple[int, int]]`; `sentence_clause_parts(text: str) -> list[tuple[int, int]]` (split on sentence ends and on clause connectives); `window_parts(text: str, width: int = 8, stride: int = 4) -> list[tuple[int, int]]` (word windows).
- Produces in `evaluate.py`: `status_scores(assessments, truth) -> dict` (macro F1 over present statuses, per-class F1, confusion matrix, false-unseen rate on held-out surfaces S4 to S8, other-as-unseen rate); `skill_confusion(assessments, truth) -> np.ndarray` (misassignment between skills, reported as its own failure mode); `tangled_scores(msgs, assessments_by_part, mode: str) -> dict` for B4a, B4b and B6; `offline_consequence(errors: dict, costs: dict = {"false_unseen": 4.0, "missed_unseen": 7.0}) -> float` (labelled an estimate).
- `run_e2.py --pool dev --seeds 100-129`: per seed draw an assignment, build the competence table, teach, assess single-part, near-twin, trap, Other and tangled messages under each representation that passed B1 (and R5 if built), for the status machine and every baseline at the same calibration (q curve over 0.01, 0.02, 0.05, 0.10 and the separately reported "adapted" condition); results to `results/e2/`; `report_e2()` writes `results/E2_TABLE.md`.

- [ ] **Step 1: Write failing tests**: `test_gold_parts_equal_the_message_spans`; `test_sentence_clause_parts_cover_the_text_without_overlap`; `test_status_scores_on_a_hand_made_confusion` (known F1 values); `test_other_content_in_a_tangled_message_is_not_flagged_as_novel_arithmetic` (B4b logic on a stub machine); `test_adapted_condition_is_reported_separately_and_never_gates`; `test_offline_consequence_is_labelled_estimate` (the output string contains "estimate").
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Run E2 on dev** (resumable); write the prediction first. Then fix the draft bars and the `THRESHOLDS` from dev results only, by a declared rule written into `PROGRESS.md` before looking (for example: bar = the dev mean minus two standard deviations, never below the draft bar of spec section 7).
- [ ] **Step 6: STOP POINT 2.** Plain-English report to the user: which statuses the machine gets right, where baselines do better, the false-unseen rate with calibration on S1 to S3 only, and whether the wording-invariance claim holds or is downgraded to "needs adaptation". Wait for the go to continue.
- [ ] **Step 7: Commit** `feat(ks): segmentation, status evaluation and dev results`.

---

### Task 12: Worked traces and relation (E4)

**Files:** Create `research/knowledge_status/traces.py`, `run_e3.py`, `tests/knowledge_status/test_ks_traces.py`.

**Plain words:** we show the system a worked example such as "3 times 4: 4 + 4 = 8, 8 + 4 = 12, so 12" and check that it notices each step is addition it already knows, that the steps chain to the answer, and so names addition as the parent of multiplication. Decoys check it is not fooled.

**Interfaces:**
- Produces: `Step(text: str, skill: str | None, a: int, b: int, result: int)`; `Trace(text: str, header_skill: str, a: int, b: int, answer: int, steps: tuple[Step, ...], decoy: str | None)` with `decoy` in `None | "wrong_steps" | "irrelevant_valid_steps" | "unrelated_skill"`; `make_trace(rng, skill: str, variant: int, decoy: str | None = None) -> Trace` for `mul` (repeated addition) and `div` (repeated subtraction) in several wordings; `verify_trace(trace: Trace, rep: Representation, machine: StatusMachine, solvers: dict[str, Solver]) -> TraceVerdict(parents: frozenset[str], chain_valid: bool, verified_steps: int)`: a step counts for skill s only if the machine assigns the step to s AND the solver of s reproduces `result`; the chain is valid when the running values reach `answer`. The operand parser may read numbers but must not receive or use the operator symbol to pick the skill.

- [ ] **Step 1: Write failing tests**: `test_mul_trace_steps_chain_to_the_answer` and the same for div; `test_parent_set_of_a_mul_trace_is_exactly_add` and `{sub}` for div (with add and sub Solid); `test_wrong_steps_decoy_is_not_chain_valid`; `test_irrelevant_valid_steps_decoy_gets_no_relation` (a max request with unrelated additions); `test_unrelated_skill_decoy_gets_no_relation`; `test_relation_needs_the_parent_to_be_solid` (add Unseen gives an empty parent set); `test_verify_does_not_read_the_operator_symbol` (replace the symbol in every step text with "?" and the verdict is unchanged for S2 to S4 style traces); `test_plain_unseen_message_is_not_given_a_relation` ("17 times 5" is Unseen, parents empty).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Run E3 on dev** and add the B5 numbers to `E2_TABLE.md` (or `E3_TABLE.md`). **Step 6: Commit** `feat(ks): worked traces and relation from traces`.

---

### Task 13: The realism set

**Files:** Create `research/knowledge_status/realism.py`, `tests/knowledge_status/test_ks_realism.py`; output `results/realism_set.jsonl` and `results/realism_spotcheck.md`.

**Interfaces:** `build_realism_set(n: int = 300, seed: int = 100) -> list[dict]` (one-step problems from the cached GSM8K and orca-math whose reference solution contains exactly one arithmetic operation, labelled by that operation via the calculator annotation or a strict regex; fields: text, skill, operands, source, index); `spotcheck(rows, k: int = 50, seed: int = 100) -> str` (markdown a human can read).

- [ ] **Step 1: Write failing tests**: `test_only_single_operation_solutions_are_kept`; `test_label_matches_the_reference_answer` (recompute with `truth`); `test_no_row_appears_twice`; `test_realism_rows_never_enter_teaching_or_tuning` (a guard function `assert_not_realism(texts)` that the teaching code calls).
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS. **Step 5:** write the 50-row spot check and ask the user or the reviewer to read it; record their verdict in `PROGRESS.md`. **Step 6: Commit** `feat(ks): realism set and spot check`.

---

### Task 14: Confirmatory pool, pre-registration and the reviewer gate (stop point 3)

**Files:** Create `research/knowledge_status/pool_confirm.py`, `freeze_prereg.py`, `tests/knowledge_status/test_ks_freeze.py`; outputs `PREREG.md`, `bars.json`, `bars.sha256`.

**Interfaces:** `freeze_prereg.py` (refuses to run a second time): writes `bars.json` (the dev-derived bars and `THRESHOLDS`, q = 0.05 with the curve levels, group sizes, representation list, seeds 400 to 429, the rule that produced each number), `bars.sha256`, the sha256 of `pool_confirm.py`, the code stamp of `research/knowledge_status/*.py` plus the `growmoe` files it imports, and `PREREG.md` (statements, predictions with expected outcomes, decision rules, kill rule).

- [ ] **Step 1: Author the confirmatory pool** (`pool_confirm.py`): templates for S1 to S8 disjoint from `pool_dev.py` (different sentence frames and vocabulary), written by hand; do not run any model on it.
- [ ] **Step 2: Write failing tests**: `test_dev_and_confirm_pools_share_no_template_or_frame` (normalised template strings and 5-gram overlap below 0.2); `test_confirm_pool_is_blocked_without_the_flag`; `test_freeze_refuses_a_second_run`; `test_changing_a_bar_changes_the_hash`; `test_prereg_lists_every_statement_of_spec_section_7` (B1a to B1c, B2, B3a, B3b, B4a, B4b, B5, B6, B7); `test_verdict_rules_catch_mutants` (flip each comparison in the verdict function in turn; at least one test fails per mutant).
- [ ] **Step 3: Run** - FAIL. **Step 4: Implement** and run - PASS.
- [ ] **Step 5: Reviewer gate.** Dispatch a fresh reviewer (not the author of the code) to read `PREREG.md`, the generator, `evaluate.py` and the verdict code against the spec, and to check for leakage (A3), surface-cue success (A1) and lists copied from the code (A4). Fix findings; re-run the tests.
- [ ] **Step 6: STOP POINT 3.** Tell the user in plain words what is frozen and what the single-use run will decide, and wait for the explicit go (seeds 400 to 429 can be used once).
- [ ] **Step 7: Run the freeze and commit** `feat(ks): confirmatory pool and frozen pre-registration`.

---

### Task 15: The confirmatory run (single use, only on the user's go)

**Files:** Create `research/knowledge_status/CONFIRM_CLOSED.json` after the run; use `run_e1.py`, `run_e2.py`, `run_e3.py` with `--pool confirm --seeds 400-429` and `KS_ALLOW_CONFIRM=1`.

- [ ] **Step 1: Preconditions:** clean repository; code stamp equals the frozen one; `bars.sha256` matches; no record exists for seeds 400 to 429.
- [ ] **Step 2: Launch** with the idempotent command; after a power cut relaunch the same command. Write nothing about results until all cells exist.
- [ ] **Step 3: Verify** the cell count equals the list made at the freeze; then write `CONFIRM_CLOSED.json` (date, stamp, cell count) and make the runners refuse to start the confirmatory phase again (as `research/growmoe_results/rv2d/jobqueue.py` does).
- [ ] **Step 4: Commit** `results(ks): confirmatory run seeds 400-429`.

---

### Task 16: Report, independent audit, notebook, paper decision

**Files:** Create `research/knowledge_status/report.py`, `audit.py`, `tests/knowledge_status/test_ks_report.py`; notebook entries and registers; `docs/superpowers/handoff/REVIEW_BRIEF_KS_PHASE1.md`.

**Interfaces:** `report.py` computes every verdict from the records by script (PASS, FAIL or NOT MEASURABLE per statement, with the interval where one applies) and writes `RESULTS_KS_PHASE1.md`; `audit.py` recomputes them independently with its lists taken from the spec text (the eleven units of each statement, the seven classes, the 30 seeds), compares the record set with the list made at the freeze, and reports unreadable records instead of skipping them.

- [ ] **Step 1: Write failing tests**: `test_report_verdict_matches_a_hand_computed_toy`; `test_audit_unit_lists_come_from_the_spec_constants_not_from_report_code` (import the spec constants module only); `test_audit_flags_a_missing_and_an_extra_record`.
- [ ] **Step 2: Run** - FAIL. **Step 3: Implement.** **Step 4: Run** - PASS.
- [ ] **Step 5: Run the report and the audit**; both must agree; any disagreement is investigated before anything is written.
- [ ] **Step 6: Independent review** by a fresh reviewer using `REVIEW_BRIEF_KS_PHASE1.md` (steps: freeze integrity, record set, recomputation, code read, re-run of a few cells, claims against records).
- [ ] **Step 7: Notebook:** entries for the related-work reading, the test bed and E1 table, the dev review, the confirmatory run and the review (grades honest: A only for the pre-registered fresh-seed results); claims, failures, decisions, open questions, glossary terms (Solid, Thin, Related, Unseen, Conflict; information ceiling); `labnb.py index`, `manifest`, `verify`.
- [ ] **Step 8: Paper decision gate** (spec section 10): a one-page memo `research/knowledge_status/PAPER_DECISION.md` applying the stated rule (C-a and C-b done on fresh seeds and the related-work reading finds nothing that already does them): write, do not write, or write only the negative-results part; add the outcome to `docs/labnotebook/PAPER_NOTES.md`.
- [ ] **Step 9: Plain-English wrap-up** to the user: what we proved, what we did not, what changes in the plan for Phase 2 (self-discovery of skills first). **Step 10: Commit.**

---

## Self-review notes (kept for the reviewer)

Spec coverage: G1 (Tasks 3 to 5, 7), G2 (Tasks 6, 8, 9), G3 (Tasks 10, 11), G4 (Task 12), G5 (Tasks 14 to 16); non-goals and anti-goals are enforced by the constraints, the `KS_ALLOW_CONFIRM` guard, the template-id and leakage tests, the B2 validity check, and the audit built from the spec. The interface rule (no labels at test time; skills read as groups) is Task 10. Phase 2 self-discovery is left out on purpose and is the first item after Task 16.
Names used across tasks: `Message`, `Part`, `SkillGroup`, `Assessment`, `StatusMachine`, `Solver`, `Representation`, `Trace`, `THRESHOLDS`, `COUNTS`, `COUNT_SWEEP` are each defined once (Tasks 3, 10, 7, 6, 12).
