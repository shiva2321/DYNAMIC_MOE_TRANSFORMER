# Research map: where everything lives

This folder (`D:/spherical_hyperspace_memory`) is the single home of the MoE and geometry research. Since 2026-10-30 there are no sibling worktrees any more
(see section 5). The lab notebook, `docs/labnotebook/`, is the record of what was done and what it showed; this file only says where to look.

## 1. Research threads

| Thread (notebook track) | Code | Results and evidence | Reports and designs |
|---|---|---|---|
| **Dynamic MoE / GrowMoE** (G) | `growmoe/` (pool, router, set gate, growth, baselines) | `research/growmoe_results/` (one folder per study: `phase0-2/`, `m0/`, `m2a/`, `m5/`, `rv2*/`, `rv3/`, `stage4_spikes/`, `merge_study/`, `shootout/`, `diagnosis/`, `arith/`, `cva/`, `r11d/`), `research/moe_diligence_2026-10/`, `research/spikes/` | `research/DYNAMIC_MOE_RESEARCH_AND_DESIGN.md`, `GATEKEEPER_RESEARCH_2026-10.md`, `GROWMOE_STAGE4_RESEARCH.md`, `GROWMOE_MERGE_RESEARCH.md`, `MOE_CONTINUAL_LEARNING_REPORT_2026-10.md`, `ROUTING_RESEARCH_2026-10.md`; specs in `docs/superpowers/specs/` |
| **Knowledge status and routing on language streams** (G) | `research/knowledge_status/*.py` (the code lives with its results) | `research/knowledge_status/results/`, `PROGRESS.md` | `NOVELTY_ROUTING_RESEARCH_2026-10-26.md`, `knowledge_status/ROUTING_ALGORITHM_2026-10-29.md`, `knowledge_status/RELATED_WORK.md`; specs `docs/superpowers/specs/2026-10-26-knowledge-status-and-routing-design.md`, `2026-10-30-general-routing-layer-design.md` |
| **UniMoE** (combined dynamic MoE on real text, Stage 1 and follow-up P3/P4) (G) | `unimoe/` | `results/unimoe/` (kept at this top-level path on purpose: the cached cells are keyed by a hash of the code, so moving the output folder would change that code and invalidate them), `research/unimoe/PROGRESS.md` | `results/unimoe/ANALYSIS.md`, `ANALYSIS_P4.md`; spec `docs/superpowers/specs/2026-10-30-unimoe-design.md`; notebook E-053, E-054. Stage 2 (token-level language model) not started |
| **Geometry atlas** (Z) | `geometry/`, `geomlab/` | `research/geomlab_results/`, `research/results/` | `GEOMETRY_ATLAS_2026-10.md`, `VSA_GEOMETRY_TO_INTELLIGENCE.md`, `ALGORITHM_AND_PROTOCOL.md`, `BIBLIOGRAPHY_CHRONOLOGICAL.md` |
| **Yggdrasil architecture** (Y) | `yggformer/`, `rust_yggdrasil/` | `research/yggdrasil_poc/`, `yggdrasil_results/`, `yggformer_results/` | `YGGDRASIL_ARCHITECTURE_RESEARCH.md` |
| **ISHMN** (the first system; audits are track I) | `ishmn/`, `experiments/`, `benchmarks/`, `crates/ishmn_attention/` | `artifacts/` (dashboards, benchmark JSON; the large `.pt` files are not in git) | notebook E-001 to E-004 |
| Retired code | `quarantine/` (legacy and contaminated variants, kept for the record), `scratch/` (old benchmark scripts) | | |

Tests: `tests/` has one sub-folder per package (`growmoe`, `knowledge_status`, `unimoe`, `geomlab`, `yggformer`) and the older ISHMN tests at its top level.
`python -m pytest tests/unimoe` takes about 20 s (77 tests); `python -m pytest --collect-only -q` finds 1927 tests.

## 2. The record

- `docs/labnotebook/`: entries (`INDEX.md` lists them), claims, failures, decisions, open questions. Start with its `README.md` (rules, evidence grades).
- `docs/superpowers/`: `specs/` (designs), `plans/` (implementation plans), `handoff/` (briefs for the implementing agent and review notes).
- `docs/roadmap/`: the **road plan** (living guidepost, from 2026-11-02): `ROADMAP.md` (goal, milestones, stuck protocol), `STATE_BOARD.md` (what exists and what must exist), `MILESTONES.md`, `IDEA_BOARD.md`, `RESEARCH_DIGEST.md`, `CHANGELOG.md`.

## 3. Data that is not in git

| Path | Size | What it is | If lost |
|---|---|---|---|
| `research/knowledge_status/_cache/` | 10.4 GB (6.4 GB on disk, NTFS compression) | cached sentence representations (`reps/`, 1882 files), wording rewrites and candidate lists (`rewrites/`, `candidates_*.json`), 6 `.pt` files of the knowledge-status labs | model runs to recreate |
| `artifacts/checkpoints/`, `artifacts/*.pt` | about 9 GB | ISHMN model checkpoints | not recreatable without the original runs |
| `results/unimoe/emb/` | 235 MB | cached sentence embeddings of the 7 UniMoE domains (ignored since 2026-10-30) | needs the MiniLM weights and the corpora |
| `data_cache/` | | third-party dataset downloads | re-download |
| `research/growmoe_results/rv3/_cache_real/` | 8 MB | 20news, agnews and banking77 arrays built by `rv3/spike_real_text.py` | re-download and rebuild |

## 4. Work still open in this folder (uncommitted at the time of the consolidation)

- Geometry lab phase 3 (`geomlab/phase3_geocore.py` modified; `research/geomlab_results/phase3/`; notebook entry `E-031-...-geocore.md`): another thread's work, left as it was found.
- `results/unimoe/`: `tune/`, `tune_mix/`, `tune_drift/`, `ask/`, the `*.log` files and the `chain.sh`, `p3chain.sh`, `v2chain.sh` scripts of Stage 1 and P3 were never committed. The notebook does not cite them; decide whether they are evidence (commit) or scratch.
- `research/knowledge_status/results/routing_general/vec/`: 32 embedding files that are not in git (126 others in the same folder are).
- Nothing has been pushed: local `track-g-growmoe` is ahead of `origin/track-g-growmoe` by 157 commits; local `main` is 59 commits behind it.

## 5. Consolidation of 2026-10-30

Four sibling folders each held a git worktree of this repository. They are merged and retired:

| Was | Held | Now |
|---|---|---|
| `D:/shm_um` (branch `unimoe-stage1`) | UniMoE Stage 1, P3, P4 | `unimoe/`, `results/unimoe/`, `research/unimoe/`; its untracked results moved to `results/unimoe/` |
| `D:/shm_ks` (branch `knowledge-status-phase1`, an ancestor of `unimoe-stage1`) | knowledge-status Phase 1 | `research/knowledge_status/`, with its 10.4 GB `_cache/` and the `.superpowers/` plan ledger moved here |
| `D:/shm_lab_wt` (detached at `090ece8`, already in this branch) | lab-notebook and rv3 working copy | already in git; the three dataset caches moved to `rv3/_cache_real/` |
| `D:/shm_adopt` (a broken worktree, already in this branch) | stage-4 adoption checkout | already in git |

How: `track-g-growmoe` was fast-forwarded from `9b277fd` to `f1a8dd7` (no merge commit; to undo: `git reset --keep 9b277fdb08e7b4144d8a3179737423930edff616`).
Untracked data was moved by rename, never overwritten. Each retired folder was checked file by file against git history first; the folders themselves went to the
Recycle Bin (recoverable until it is emptied). The branches `unimoe-stage1`, `knowledge-status-phase1` and `stage4-1-cap-and-telemetry` still exist and are fully merged;
`git branch -d <name>` removes them when wanted.

**Old commands and notes:** replace `D:/shm_um`, `D:/shm_ks`, `D:/shm_lab_wt` (and `/d/shm_um` in `results/unimoe/p4chain*.sh`) by `D:/spherical_hyperspace_memory`. The notebook
entries were left as written, because their evidence hashes cover them.

Checked after the move, in this folder: 77 UniMoE tests pass; the P4 tables regenerate from the cached cells byte for byte; all 1927 tests are collected without import errors.
