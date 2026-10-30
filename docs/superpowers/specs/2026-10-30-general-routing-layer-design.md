# A domain-neutral routing layer, and how it plugs into the expert pool

Date 2026-10-30. Status: design by Claude's rulings, written after the user said (2026-10-30): "start with the pool wiring ... this seems to be just specific to arithmetic, whereas we need a general algorithm and working system". Dev-seed research; nothing here is pre-registered and the single-use seeds 400-429 stay unspent. Companion: `research/knowledge_status/ROUTING_ALGORITHM_2026-10-29.md` (v1, arithmetic).

## 1. What is arithmetic-shaped in v1 (the honest list)

1. The question lists six operations (`OPS`, letters A-G) and the status rule reads "the model named an operation nobody taught". The never-taught skills are found partly because the question OFFERS them as options. A system that does not know the skill's name cannot offer it.
2. The boundary question is "does this ask for a calculation?".
3. The check of a stated answer is `skills.truth` / `PerfectSolver` on parsed operands.
4. "Other content" is a class defined by the test bed (everyday sentences next to arithmetic).
5. The experts of the lab are arithmetic solvers on parsed numbers; the real pool (`growmoe/pools/phylo.py`) never sees language: it takes feature vectors, has its own density gate, and ignores `ctx`.

## 2. The domain-neutral contract

The layer talks to the pool through four small things; only the readers and the verifiers know a domain.

- **Roster.** One `Card` per expert: the name the teacher gave (any string) and a few of the expert's own teaching messages as exemplars. Cards are rebuilt from the pool, so the roster grows with the pool. Nothing in a card is a domain word we wrote.
- **Reader** (`inputs -> probabilities over the cards and "none of these"`). For text: the language model is shown the message and a multiple-choice question whose options are the roster (best candidates first when the roster is large) and one last option "none of these". The prompt contains no domain word. Another modality plugs in another reader with the same output (a density gate on vectors, a function check).
- **Decision** (domain-free). Claimed = the most probable option is a card and its probability is at least tau; otherwise unclaimed. Claimed gets that card's state (Solid or Thin, from the expert's measured competence). Team = usable cards, most probable first, added until they hold `mass` of the probability on cards (cap 3). Unclaimed means "not mine", nothing more.
- **Verifier** (`expert, input, stated answer -> agrees?`). The expert's own prediction against the stated answer; the group permutation test of v1 is the same statistic with the shuffle over a window.

Ruling R1: **unclaimed is not "new skill"**. Everyday text, noise and a real new skill are all unclaimed. A new expert is created only when unclaimed inputs recur coherently and a teacher or the stream gives feedback (the pool's existing probation and set-test machinery). This removes the arithmetic-only boundary question from the general layer; it stays available as an optional `scope` gate.

Ruling R2: **closed list**. The question offers only the taught cards plus "none of these". That is harder than v1 (never-taught skills are no longer offered by name) and it is the only version that can be general.

## 3. Pool wiring (no change to `growmoe/`, NG1 holds)

`PhyloMoE` is vector-in and cannot be told who owns a sample. So the wiring is a thin pool of its own, `RoutedPool`, that implements the same `ContinualPool` protocol (observe / predict / summary) over text with features, reuses `growmoe` experts, replay and the set test read-only, and takes its gate from the layer above. It is compared with `PhyloMoE` on the same stream (accuracy, number of experts, duplicate experts for one skill, merged skills, compute). A hybrid that consults both gates is the last step, only if each catches something the other misses.

## 4. Order and claims (all dev; predictions are logged in PROGRESS.md before each run)

- G1. The reader and rule across domains with no code change: arithmetic, banking77 intents, AG News topics, 20 Newsgroups, and a mixed roster. Claim: one reader, one rule, one tau; reported per domain with the look-based detectors (Mahalanobis, kNN, vMF-conformal on MiniLM) at the same false-unclaimed rate. Success is not "wins everywhere": the report says where it loses.
- G2. `RoutedPool` against `PhyloMoE` on a text stream.
- G3. Exemplars versus names only, order bias, roster size (retrieval of candidates).
- G4. Generic verifier and group test.

## 5. Non-goals and limits

No growth work, no change to `growmoe/`, no new downloads, no confirmatory claim. Limits already known: the reader depends on the language model's understanding of the card text; a modality without text needs its own reader; two sources of error remain (the reader and the expert).

## 6. What the first measurements changed (added 2026-10-30, after labs G1, G1b, G1c; notebook E-048)

- **R2 held, R3 is new.** The closed-list asking reader is NOT a general novelty detector: in three text domains (customer intents, news topics, newsgroups) the look-based detectors on sentence embeddings find more never-taught messages at the same false-flag rate (C-98, refuted as a general reader). It wins where the look is the same and the wording shifts (arithmetic: known messages flagged 0.12 against 0.38-0.50, AUROC 0.92 against 0.69 or less; C-99). No fixed combination of the two (smallest or largest conformal p-value) works in every domain (C-100).
- **Ruling R3:** the layer holds several readers and must take their trust from the roster's own data. The reader contract of section 2 is unchanged (inputs to probabilities over the cards and "none"; a density gate on vectors fits it too, with the pool's own vMF signature as the look-based reader).
- **What is already known about the obvious repair.** Generic rewrites of teaching messages were tried on 2026-10-28 as extra calibration parts and prototypes for the look-based reader (PROGRESS.md, lab_rewrites; E-046) and did not bring its false alarm down (0.232-0.262 against 0.256) while losing 0.14-0.16 of the found never-taught skills. So rewrites are not a calibration cure. Whether they are a useful DIAGNOSTIC (the rank of a reader's false-flag rate on rewrites of held-out roster messages across domains, used to decide whom to trust) is untested and is lab G3; its prediction has to be written before it runs.
- **Order now:** G3 (diagnostic), then G2 (`RoutedPool` against `PhyloMoE` on one text stream, with the reader mix chosen by G3), then the generic verifier and group test.

## 7. The general routing layer as it stands (2026-10-30, after labs G1 to G4b; notebook E-048 to E-050)

Plain words first. For every incoming message the layer says: **who answers, whether anyone should, and how sure it is**, for any domain, from the pool's own roster. It is built from parts, each measured on its own.

1. **Roster.** One card per expert: the name its teacher gave and two of its own teaching messages. Nothing else about the domain is known to the layer. (`cards.py`)
2. **Two readers that see different things.** The *asking reader* shows a language model the message and the roster as a multiple-choice question ("which of these, or none of these?") and reads the probabilities (`reader.py`); it holds when the wording changes and the function stays the same (arithmetic: flags 0.09 of known messages in new wordings against 0.36-0.42 for the look-based detectors). The *density gate* is the pool's own look-based gate on sentence embeddings (the vMF signature of `growmoe`); it wins when the new thing differs in topic (banking never-taught intents: found 0.84 against 0.50). Neither is enough alone (E-048).
3. **Calibration on outcomes, not on rewrites.** A label, or a stated answer that a taught expert reproduces, says for some messages that they are owned. A random sample of at least 40 such verified messages (nothing is flagged below 19) from the stream's own wording sets every reader's cut (`outcome.py`). A reader that was only accurate in the teaching wording is thereby held to its real false-flag rate in the stream's wording (E-050: arithmetic 0.054 for the mix against 0.350 before). Rewrites by the language model were tried as a diagnostic and do not work (E-049).
4. **Mix.** The readers' calibrated p-values are combined by "either" (answer-more: a message is not mine if any reader says so) or "both" (refuse-more). Which one is a *policy with a price*, not a derived fact (E-049): with identical experts the asking gate answers 92 % of known messages against 62 % for the pool's gate, and also answers 42-50 % of never-taught skills against 24-46 %.
5. **Decision.** Claimed (an expert owns it, Solid or Thin by its measured competence), unclaimed ("not mine": noise, out of scope or a new skill; whether it deserves an expert takes recurrence and feedback, ruling R1), a team of experts sized by probability mass (`route.py`).

What is measured (tuning seeds 150-152, dev, not confirmatory): the mix keeps its false-flag rate near nominal in four domains (arithmetic, banking intents, news topics, newsgroups) and finds as much as or more than the best single reader at that operating point; power at an honest false-flag rate is modest (0.28, 0.68, 0.39, 0.19). What is NOT yet shown: robustness to a noisy outcome check (a quarter never-taught stream costs about a third of the power, borderline at the declared bar, C-105); the layer on the growmoe pool's own experts (the pool is vector-in and cannot be told who owns a sample; `GatedPool` is a container of our own); discovery of cards from an unlabelled stream; any pre-registered run; any domain beyond text. In the arithmetic test bed the lever is expert competence, not routing (the perfect gate still nets about zero, E-049).
