# E-048 - General routing layer: one reader for every domain fails; asking wins under a wording shift, looking wins on topic, no fixed fusion works (labs G1, G1b, G1c)

- **ID:** E-048
- **Date:** 2026-10-30
- **Track:** G
- **Type:** experiment
- **Status:** completed for labs G1, G1b and G1c (the pool, `RoutedPool`, and shift-aware calibration, labs G2 and G3, follow in later entries)
- **Evidence grade:** B (script-generated on the tuning seeds 150-152; every rule and prediction written in PROGRESS.md before the numbers; exploratory, NOT a frozen pre-registration, NOT the single-use seeds; no independent reviewer yet)
- **Code / commit:** branch `knowledge-status-phase1`, commits 30a30f9 to 566166d; code in `research/knowledge_status/routing_general/`, tests `tests/knowledge_status/test_rg_*.py`; worktree D:/shm_ks
- **People:** Claude designed, built and ran it on the user's instruction of 2026-10-30 ("start with the pool wiring ... this seems to be just specific to arithmetic, whereas we need a general algorithm and working system")

## Question
Can ONE reader and ONE decision rule, built only from the pool's own roster (the name of each expert and two of its teaching messages) and containing no domain word, tell which expert owns a message and recognise a message nobody owns, in four different domains (arithmetic in everyday language, customer-service intents, news topics, newsgroups)? If yes, that is the general layer the pool needs; if not, which kinds of novelty need which kind of reader?

## Why we asked it
Entry E-047 built the routing and novelty algorithm v1 and proved it on arithmetic only, with a question that lists the six operations, a boundary question about calculation and a checker that parses numbers. The user (2026-10-30): the algorithm must be general, not arithmetic-specific, and the first job is wiring it to the pool. The pool (`growmoe/pools/phylo.py`) takes only feature vectors, has its own density gate and ignores `ctx`, and the spec forbids editing `growmoe/`; the design note `docs/superpowers/specs/2026-10-30-general-routing-layer-design.md` lists what is arithmetic-shaped and defines a domain-free contract (cards, reader, rule, verifier).

## Prediction (written BEFORE the numbers, if there was one)
In `research/knowledge_status/PROGRESS.md`, blocks of 2026-10-30: the G1 predictions with the rules C1 (same tau in every domain, known claimed >= 0.80 and foreign unclaimed >= 0.85), C2 (the reader finds more never-taught messages than the best look-based detector by >= 0.05 in >= 2 domains), C3 (exemplars beat names by 0.02 in top-1 in >= 2 domains); then G1b/G1c: C4 (under a wording shift the reader's known-flagged rate is lower than every detector's by >= 0.10) and C5 (no fixed fusion works everywhere, which I predicted would HOLD). Result: C1, C2, C3 failed; C4 passed; C5 held.

## Method (how)
- **Cards and the question** (`cards.py`): the template contains no domain word (a test checks it); the options are the taught classes in a fixed order and a last option "none of these". Two arms: class names only, and names plus two of the class's teaching messages.
- **Reader** (`reader.py`): Qwen2.5-1.5B-Instruct, chat template, one forward pass, the probabilities of the option letters; answers cached per question in json, written atomically.
- **Rule** (`route.py`): claimed when the best card has probability >= tau (0.5); a claimed message takes the card's state and a team sized by probability mass; unclaimed means "not mine" and nothing more (ruling R1: unclaimed is not "new skill").
- **Domains** (`domains.py`, `arith_domain.py`): any labelled corpus becomes a world (taught classes, 30 held-out calibration messages per class, a known test, a never-taught test); banking77 (6 taught of 77), AG News (2 of 4), 20 Newsgroups (6 of 20), arithmetic (the test bed of E-045: 3 taught skills, calibration in the taught wordings S1-S3, known test in the NEW wordings S4-S8). Foreign messages: the other text domains and science questions (arithmetic: everyday sentences and the 60 look-alike traps).
- **Look-based detectors** (`baselines.py` of the arithmetic study): Mahalanobis, kNN, vMF conformal on MiniLM vectors, fitted on all the teaching messages (60 per class; arithmetic up to 2000), plus a logistic regression for identification. Everyone is cut at the same 5 % false-flag rate on the same held-out calibration messages (`measures.py`).
- **Fusion** (G1c): the conformal p-values of the reader and one detector combined as the smallest p ("either") or the largest ("both"), recalibrated on the calibration messages by leave-one-out.
- Seeds 150-152 (tuning). Two launch failures on data sizes (loud, before any model ran) and two design flaws found after seeing numbers are in the caveats; no bar changed after a result; `growmoe/` untouched.

## Results (what came out)
Numbers copied by script from `research/knowledge_status/results/ROUTING_GENERAL_G1_150_152.md` (3 seeds, mean). The claim rule here is tau 0.5; the reader columns are the asking reader, the detector rows are the look-based ones.

### arithmetic (3 seeds; 3 taught classes; 80 known, 120 unknown, 236 foreign messages per seed)

Claim rule at tau 0.5 (the same rule in every domain):

| arm | tau | known claimed | known claimed RIGHT | right given claimed | unknown unclaimed | foreign unclaimed |
|---|---|---|---|---|---|---|
| names | 0.5 | 0.942 | 0.883 | 0.936 | 0.244 | 0.975 |
| exemplars | 0.5 | 0.896 | 0.871 | 0.972 | 0.581 | 0.979 |

Same false-flag rate for everyone (5 % of held-out known messages):

| reader or detector | known flagged | unknown found | foreign found | AUROC known vs unknown | AUROC known vs foreign |
|---|---|---|---|---|---|
| ask, names | 0.100 | 0.475 | 0.987 | 0.886 | 0.997 |
| ask, exemplars | 0.121 | 0.669 | 0.986 | 0.921 | 0.995 |
| mahalanobis (MiniLM) | 0.504 | 0.925 | 1.000 | 0.681 | 0.998 |
| knn_distance (MiniLM) | 0.442 | 0.361 | 1.000 | 0.472 | 1.000 |
| vmf_conformal (MiniLM) | 0.383 | 0.756 | 1.000 | 0.685 | 1.000 |
| FUSED either: ask_exemplars + vmf_conformal | 0.371 | 0.800 | 1.000 | 0.719 | 0.825 |
| FUSED both: ask_exemplars + vmf_conformal | 0.208 | 0.919 | 1.000 | 0.893 | 0.996 |
| FUSED either: ask_exemplars + mahalanobis | 0.438 | 0.925 | 1.000 | 0.761 | 0.812 |
| FUSED both: ask_exemplars + mahalanobis | 0.254 | 0.964 | 1.000 | 0.918 | 0.996 |
| FUSED either: ask_names + vmf_conformal | 0.371 | 0.733 | 1.000 | 0.715 | 0.829 |
| FUSED both: ask_names + vmf_conformal | 0.233 | 0.908 | 1.000 | 0.878 | 0.998 |

Who is it? (top-1 over the taught classes, known messages):

| reader or classifier | top-1 |
|---|---|
| ask, names | 0.883 |
| ask, exemplars | 0.900 |
| logistic regression on MiniLM, 60 messages per class | 0.329 |

### banking (3 seeds; 6 taught classes; 180 known, 300 unknown, 150 foreign messages per seed)

Claim rule at tau 0.5 (the same rule in every domain):

| arm | tau | known claimed | known claimed RIGHT | right given claimed | unknown unclaimed | foreign unclaimed |
|---|---|---|---|---|---|---|
| names | 0.5 | 0.978 | 0.831 | 0.849 | 0.234 | 0.440 |
| exemplars | 0.5 | 0.969 | 0.889 | 0.918 | 0.464 | 0.907 |

Same false-flag rate for everyone (5 % of held-out known messages):

| reader or detector | known flagged | unknown found | foreign found | AUROC known vs unknown | AUROC known vs foreign |
|---|---|---|---|---|---|
| ask, names | 0.043 | 0.348 | 0.576 | 0.838 | 0.925 |
| ask, exemplars | 0.048 | 0.501 | 0.916 | 0.865 | 0.985 |
| mahalanobis (MiniLM) | 0.037 | 0.838 | 1.000 | 0.972 | 1.000 |
| knn_distance (MiniLM) | 0.037 | 0.773 | 1.000 | 0.961 | 1.000 |
| vmf_conformal (MiniLM) | 0.035 | 0.860 | 1.000 | 0.979 | 1.000 |
| FUSED either: ask_exemplars + vmf_conformal | 0.043 | 0.833 | 1.000 | 0.965 | 0.996 |
| FUSED both: ask_exemplars + vmf_conformal | 0.039 | 0.713 | 0.987 | 0.937 | 0.997 |
| FUSED either: ask_exemplars + mahalanobis | 0.039 | 0.771 | 1.000 | 0.956 | 0.993 |
| FUSED both: ask_exemplars + mahalanobis | 0.037 | 0.703 | 0.984 | 0.937 | 0.998 |
| FUSED either: ask_names + vmf_conformal | 0.035 | 0.799 | 1.000 | 0.962 | 0.996 |
| FUSED both: ask_names + vmf_conformal | 0.043 | 0.686 | 0.860 | 0.940 | 0.984 |

Who is it? (top-1 over the taught classes, known messages):

| reader or classifier | top-1 |
|---|---|
| ask, names | 0.839 |
| ask, exemplars | 0.902 |
| logistic regression on MiniLM, 60 messages per class | 0.998 |

### agnews (3 seeds; 2 taught classes; 180 known, 300 unknown, 150 foreign messages per seed)

Claim rule at tau 0.5 (the same rule in every domain):

| arm | tau | known claimed | known claimed RIGHT | right given claimed | unknown unclaimed | foreign unclaimed |
|---|---|---|---|---|---|---|
| names | 0.5 | 0.980 | 0.874 | 0.891 | 0.292 | 0.467 |
| exemplars | 0.5 | 0.915 | 0.794 | 0.870 | 0.598 | 0.604 |

Same false-flag rate for everyone (5 % of held-out known messages):

| reader or detector | known flagged | unknown found | foreign found | AUROC known vs unknown | AUROC known vs foreign |
|---|---|---|---|---|---|
| ask, names | 0.039 | 0.393 | 0.507 | 0.816 | 0.826 |
| ask, exemplars | 0.057 | 0.470 | 0.473 | 0.854 | 0.862 |
| mahalanobis (MiniLM) | 0.065 | 0.504 | 0.844 | 0.883 | 0.968 |
| knn_distance (MiniLM) | 0.046 | 0.509 | 0.744 | 0.892 | 0.956 |
| vmf_conformal (MiniLM) | 0.063 | 0.468 | 0.669 | 0.874 | 0.929 |
| FUSED either: ask_exemplars + vmf_conformal | 0.063 | 0.478 | 0.553 | 0.875 | 0.919 |
| FUSED both: ask_exemplars + vmf_conformal | 0.059 | 0.556 | 0.631 | 0.898 | 0.922 |
| FUSED either: ask_exemplars + mahalanobis | 0.067 | 0.506 | 0.693 | 0.873 | 0.942 |
| FUSED both: ask_exemplars + mahalanobis | 0.063 | 0.539 | 0.693 | 0.904 | 0.936 |
| FUSED either: ask_names + vmf_conformal | 0.056 | 0.472 | 0.618 | 0.873 | 0.917 |
| FUSED both: ask_names + vmf_conformal | 0.050 | 0.581 | 0.656 | 0.883 | 0.899 |

Who is it? (top-1 over the taught classes, known messages):

| reader or classifier | top-1 |
|---|---|
| ask, names | 0.885 |
| ask, exemplars | 0.867 |
| logistic regression on MiniLM, 60 messages per class | 0.913 |

### newsgroups (3 seeds; 6 taught classes; 180 known, 300 unknown, 150 foreign messages per seed)

Claim rule at tau 0.5 (the same rule in every domain):

| arm | tau | known claimed | known claimed RIGHT | right given claimed | unknown unclaimed | foreign unclaimed |
|---|---|---|---|---|---|---|
| names | 0.5 | 0.857 | 0.748 | 0.873 | 0.499 | 0.524 |
| exemplars | 0.5 | 0.622 | 0.569 | 0.917 | 0.818 | 0.800 |

Same false-flag rate for everyone (5 % of held-out known messages):

| reader or detector | known flagged | unknown found | foreign found | AUROC known vs unknown | AUROC known vs foreign |
|---|---|---|---|---|---|
| ask, names | 0.020 | 0.119 | 0.064 | 0.773 | 0.784 |
| ask, exemplars | 0.035 | 0.113 | 0.060 | 0.770 | 0.748 |
| mahalanobis (MiniLM) | 0.052 | 0.226 | 0.620 | 0.779 | 0.937 |
| knn_distance (MiniLM) | 0.046 | 0.236 | 0.591 | 0.775 | 0.930 |
| vmf_conformal (MiniLM) | 0.056 | 0.253 | 0.538 | 0.740 | 0.882 |
| FUSED either: ask_exemplars + vmf_conformal | 0.039 | 0.187 | 0.393 | 0.768 | 0.874 |
| FUSED both: ask_exemplars + vmf_conformal | 0.054 | 0.259 | 0.264 | 0.801 | 0.837 |
| FUSED either: ask_exemplars + mahalanobis | 0.035 | 0.201 | 0.504 | 0.784 | 0.914 |
| FUSED both: ask_exemplars + mahalanobis | 0.046 | 0.227 | 0.267 | 0.821 | 0.855 |
| FUSED either: ask_names + vmf_conformal | 0.033 | 0.226 | 0.424 | 0.777 | 0.890 |
| FUSED both: ask_names + vmf_conformal | 0.050 | 0.357 | 0.498 | 0.794 | 0.841 |

Who is it? (top-1 over the taught classes, known messages):

| reader or classifier | top-1 |
|---|---|
| ask, names | 0.778 |
| ask, exemplars | 0.752 |
| logistic regression on MiniLM, 60 messages per class | 0.820 |


## Interpretation (what we understood, and on what basis)
Plain words first. A language model that is asked "which of these does the message belong to, or none?" is NOT a general novelty detector. When the new thing differs in topic from what was taught (a new customer intent, a new news topic, a new newsgroup), an ordinary sentence embedding already separates it and the detectors on it do as well or better (banking: AUROC 0.979 against 0.865). When the look is the same and only the function or the wording differs (arithmetic in everyday language), the embedding follows the wording, not the operation: a plain classifier on it identifies the skill of new-wording messages at chance (0.329 for three skills), the detectors flag 38 to 50 % of known messages, and the asking reader keeps them known (flags 0.12) and ranks never-taught above known with AUROC 0.92 against 0.69 or less. Each reader sees what the other cannot, and no fixed way of combining them (novel if either says so, novel if both say so) works in every domain.

Technical statement: a double dissociation between a semantic-topic signal (representation distance) and a function-under-wording-shift signal (the model's answer to a question). This is the reader-level counterpart of the information-ceiling result (E-044): which signal can see which difference. Measured: the four tables above (3 seeds, dev). Assumed: that the pattern generalises beyond these four domains and beyond Qwen2.5-1.5B and MiniLM.

## Caveats and what went wrong
- **Three seeds, tuning seeds only**, small known test in arithmetic (40-80 messages per seed), one language model and one encoder, class name plus two messages for the reader against all 60 (up to 2000) teaching messages for the detectors: the comparison is fair about the cut and unfair about information, in the detectors' favour on text and not on arithmetic.
- **My predictions missed in the detectors' favour**: banking never-taught found 0.84-0.86 against my 0.25-0.55. I underestimated the embedding detectors on topical novelty.
- **Position bias held**: with names only, 56 % of foreign sentences in banking and 53 % in agnews are claimed by some card; the model prefers a card to "none of these". Exemplars repair banking, not agnews.
- **Design flaw 1 (F-46)**: the first arithmetic run mixed wordings in the never-taught set (43 % in the teaching wordings) while the known set was all in new wordings; this handicapped the detectors in the unknown-found and AUROC columns (kNN AUROC 0.379). Fixed; the corrected run is the one reported; the verdicts did not change.
- **Design flaw 2 (F-47)**: the test bed's calibration repeated 4-6 % of the teaching requests; dropped.
- **Process slip (F-48)**: a commit went in with one failing test (the commit gate does not run tests); fixed in the next commit.
- The traps are flagged by every reader (1.000): they are everyday sentences with arithmetic words, which is easy in a closed list; the v1 weakness on traps came from the open list.
- Not done: no pool is wired yet; fusion uses calibration without a shift; the verifier and the group test are still arithmetic-specific.

## Consequence
Ruling R3 in the design note: the layer holds several readers (the asking reader and the density gate the pool already has) and must learn from the roster's OWN data how far each can be trusted under a wording shift. Next, in this order: G3, generate rewrites of the roster's own teaching messages with the same language model (no domain knowledge), calibrate each reader on originals plus rewrites and test on the truly shifted known messages of arithmetic (predictions to be written first); then G2, the thin `RoutedPool` against `PhyloMoE` on one text stream; then a generic verifier and group test. The claim that one reader suffices is dropped (C-98).

## Evidence
- `research/knowledge_status/results/ROUTING_GENERAL_G1_150_152.md` (generated; the corrected run), `research/knowledge_status/results/routing_general/` (per-seed cells, cached model answers, vectors, run logs)
- `research/knowledge_status/PROGRESS.md` (predictions, first-run verdicts, flaws, corrected run, in order)
- `docs/superpowers/specs/2026-10-30-general-routing-layer-design.md`

## Reproduce
```
cd D:/shm_ks
python research/knowledge_status/routing_general/lab_general.py 150 152
python -m pytest tests/knowledge_status/test_rg_*.py
```

