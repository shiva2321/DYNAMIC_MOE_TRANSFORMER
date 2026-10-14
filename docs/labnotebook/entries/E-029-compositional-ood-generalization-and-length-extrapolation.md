# E-029 - Compositional OOD generalization and length extrapolation

> **RECONSTRUCTED ENTRY.** The original text of this entry was accidentally overwritten on 2026-10-14 at 14:10 by Claude's generator for a different entry (a file-name search that matched the existing E-029 and wrote over it; the file was not under version control). Everything below is rebuilt from what survived: the index line (title, status, evidence grade), `PREREG.md` and `tables.txt` of phase 2, copied without change. The original author's interpretation, caveats and consequence were lost; the person who wrote them should re-add them. The overwritten text of Claude's own entry is now E-030.

- **ID:** E-029
- **Date:** 2026-10-14
- **Track:** Z
- **Type:** experiment
- **Status:** completed
- **Evidence grade:** A (pre-registered in `PREREG.md`, dev tuning on seeds >= 100, 10 fresh test seeds 0-9, script-generated, paired stats with Bonferroni correction, crash-resilient atomic checkpointing)
- **Code / commit:** `geomlab/phase2_composition.py`, `research/geomlab_results/phase2/run_phase2.py` (not committed at the time of the reconstruction)
- **People:** not recovered (the sibling entries E-027 and E-028 list the user, Gemini as implementer and Claude as reviewer)

## Question and pre-registered design (copied from `research/geomlab_results/phase2/PREREG.md`)
## 1. Objective and Research Questions
This experiment tests whether geometric representations equipped with exact group isometries (Lie group block-unitary operators $U(4)^{D/4}$ and flat Clifford Torus phasors $\mathbb{T}^D$) solve out-of-distribution (OOD) compositional length extrapolation, or whether unconstrained neural representations (standard Transformers and Euclidean RNNs) fail when relational composition depth exceeds the training distribution.

Specifically:
1. **Can neural networks extrapolate relational reasoning to unseen depths?** When trained only on composition paths of length $K \in \{1, 2, 3\}$, can a standard Transformer or Euclidean recurrent network correctly trace paths of length $K \in \{4, 6, 8, 12\}$?
2. **Why do standard architectures fail?** Is failure caused by representation drift (unbounded norm growth or attenuation) and non-isometric error accumulation?
3. **Does an exact isometric group operator eliminate the length barrier?** Because unitary matrices and flat torus rotations preserve $L_2$ norm and metric distances exactly ($\|U v\|_2 = \|v\|_2$), does an isometric representation maintain high accuracy across extended reasoning chains without fine-tuning?

---

## 2. Experimental Task: Multi-Step Graph Relational Composition

- **Graph Structure:** $N = 256$ entities arranged in a multi-relational directed graph with $|\mathcal{R}| = 8$ distinct non-trivial relational generators (permutations / transitions).
- **Relational Composition Sequence:**
  A path is defined by a start entity $e_0 \in \{0, \dots, N-1\}$ and a sequence of relation tokens $(r_1, r_2, \dots, r_K)$, yielding ground truth target entity:
  $$e_K = e_0 \circ r_1 \circ r_2 \circ \dots \circ r_K$$
- **Dataset Partitioning (Zero Data Leakage):**
  - **Training Distribution ($K_{\text{train}} \in \{1, 2, 3\}$):** 6,000 distinct paths of lengths 1, 2, and 3.
  - **In-Distribution Test ($K \in \{1, 2, 3\}$):** 1,500 held-out paths of lengths 1, 2, and 3.
  - **OOD Extrapolation Test ($K \in \{4, 6, 8, 12\}$):** 1,000 held-out paths for EACH length $K \in \{4, 6, 8, 12\}$ (total 4,000 OOD paths). Not a single path of length $\ge 4$ is ever seen during training or validation tuning.
- **Dimensionality:** $D = 256$ across all arms.

---

## 3. Matched Arms and Architectural Specifications

All models have matched embedding budgets ($D = 256$) and are trained with AdamW with their own hyperparameter grid on dev seeds.

1. **`standard_transformer` (Causal Transformer Baseline):**
   - 2 Transformer layers, 4 attention heads, hidden dimension 256, FFN dimension 512, LayerNorm, GELU.
   - Learned positional embeddings.
   - Sequence input: $[e_0, r_1, r_2, \dots, r_K]$.
   - Readout from final token position via linear classifier over the 256 entity classes.
   - Loss: Cross-entropy.

2. **`euclidean_rnn` (Standard Recurrent Baseline):**
   - Hidden state $h_t \in \mathbb{R}^{256}$.
   - Transition: $h_t = \operatorname{LayerNorm}(\operatorname{GELU}(W_h h_{t-1} + W_r r_t))$.
   - Readout: Linear classifier $W_{\text{out}} h_K + b$.
   - Loss: Cross-entropy.

3. **`torus_isometry` (Flat Clifford Torus Phase Binding):**
   - Entities and relations represented as phase angles $\theta \in [-\pi, \pi)^D$.
   - Transition is exact torus rotation / phase addition:
     $$\theta_t = \theta_{t-1} + \phi(r_t) \pmod{2\pi}$$
   - Readout: Cosine similarity between final phase $\theta_K$ and all entity phase prototypes with learned inverse temperature $\beta$.
   - Loss: Cross-entropy.

4. **`block_unitary_isometry` (Block-Unitary Lie Group Composition):**
   - Dimension $D = 256$ partitioned into $B = 64$ blocks of size $4 \times 4$.
   - State: Block-unitary matrix $U_t \in U(4)^{64}$.
   - Initial state: $U_0 = \mathcal{E}(e_0)$.
   - Transition: $U_t = U_{t-1} \mathcal{R}(r_t)$, where $\mathcal{R}(r) \in U(4)^{64}$ is parameterized via skew-Hermitian matrix exponential $\exp(A - A^\dagger)$, ensuring exact isometry ($\|U_t v\|_2 = \|v\|_2$).
   - Readout: Normalized Frobenius inner product / complex trace $\operatorname{Re}\operatorname{Tr}(U_{\text{target}}^\dagger U_K)$ with learned scale $\beta$.
   - Loss: Cross-entropy.

---

## Pre-registered hypotheses and decision rules (copied from PREREG.md)
| Hypothesis | Description | Success Criterion | Falsified If |
|---|---|---|---|
| **H1** | In-Distribution Competence | All 4 models achieve $\ge 80\%$ accuracy on $K \in \{1, 2, 3\}$ | Any model fails to reach $70\%$ in-distribution |
| **H2** | Baseline Length Breakdown | `standard_transformer` and `euclidean_rnn` drop below $30\%$ accuracy at $K=8$ and below $15\%$ at $K=12$ | Either baseline maintains $\ge 50\%$ at $K=12$ |
| **H3** | Isometric Length Extrapolation | `block_unitary_isometry` maintains $\ge 75\%$ accuracy at $K=8$ and $\ge 60\%$ at $K=12$, beating both baselines by $\ge 40$ percentage points ($p < 0.001$) | `block_unitary_isometry` accuracy $< 50\%$ at $K=12$ or difference $< 20$ points |
| **H4** | Isometry Norm Conservation | `block_unitary_isometry` and `torus_isometry` have strictly $0.0\%$ norm drift ($\Delta_{\text{norm}} \equiv 1.0$), while `euclidean_rnn` norm drifts by $\ge 25\%$ at $K=12$ | Isometric norm drifts by $> 1\%$ |

### Decision Rules:
- **Adopt Lie Group / Isometric Operators for Multi-Step Reasoning** if H3 and H4 pass (proving that metric isometries prevent reasoning drift and unlock depth extrapolation).
- **If H2 is falsified (Transformer extrapolates to $K=12$ without degradation):** Conclude that learned attention already handles composition without geometric inductive bias, and do not mandate group-theoretic state transitions.

---

## Results (copied from `research/geomlab_results/phase2/tables.txt`, generated by `run_phase2.py`)
```
================================================================================
PHASE 2: COMPOSITIONAL OOD GENERALIZATION AND LENGTH EXTRAPOLATION RESULTS
================================================================================
Dev Seeds: [100, 101, 102] | Test Seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] (10 seeds, mean +- 95% CI)
Training distribution: K in {1, 2, 3} | OOD Extrapolation: K in {4, 6, 8, 12}

TABLE 1: REASONING ACCURACY ACROSS COMPOSITION LENGTHS (CHANCE = 0.39%)
--------------------------------------------------------------------------------
Arm                      | K<=3 (Train)   | K=4 (OOD)    | K=6 (OOD)    | K=8 (OOD)    | K=12 (OOD)  
--------------------------------------------------------------------------------
standard_transformer     | 11.5% +/- 2.3% | 5.5%         | 3.3%         | 2.3%         | 1.6%        
euclidean_rnn            | 2.9% +/- 0.3%  | 2.2%         | 0.7%         | 0.5%         | 0.5%        
torus_isometry           | 48.3% +/- 1.5% | 26.2%        | 11.9%        | 6.4%         | 1.9%        
block_unitary_isometry   | 78.3% +/- 1.2% | 73.8%        | 68.2%        | 57.8%        | 48.4%       
--------------------------------------------------------------------------------

TABLE 2: REPRESENTATION NORM DRIFT RATIO ||h_K|| / ||h_0|| AT K=12
--------------------------------------------------------------------------------
Arm                      | K=12 Norm Drift    | Theoretical Ideal 
--------------------------------------------------------------------------------
standard_transformer     | 1.0587 +/- 0.0761  | 1.0000            
euclidean_rnn            | 1.0819 +/- 0.0036  | 1.0000            
torus_isometry           | 1.0000 +/- 0.0000  | 1.0000 (Exact Isometry)
block_unitary_isometry   | 1.0000 +/- 0.0000  | 1.0000 (Exact Isometry)
--------------------------------------------------------------------------------

TABLE 3: PRE-REGISTERED HYPOTHESIS VERIFICATION
--------------------------------------------------------------------------------
H1: In-distribution competence (all arms >= 80% on K<=3): REJECTED
    standard_transformer: 11.5%
    euclidean_rnn: 2.9%
    torus_isometry: 48.3%
    block_unitary_isometry: 78.3%
H2: Baseline length breakdown (<30% at K=8, <15% at K=12): PASSED
    Transformer K=8: 2.3%, K=12: 1.6%
    RNN K=8: 0.5%, K=12: 0.5%
H3: Isometric extrapolation resilience (Unitary >=75% at K=8, >=60% at K=12, gain >=40%): REJECTED
    Unitary K=8: 57.8%, K=12: 48.4% (Gain over baselines: +46.8%, p=3.45e-14, dz=25.55)
    Torus K=8: 6.4%, K=12: 1.9%
H4: Isometry norm conservation (drift=0.0%, RNN drift >= 25%): REJECTED
    Unitary: 1.0000, Torus: 1.0000, RNN: 1.0819
================================================================================
```

## What the files state (no interpretation added)
By the pre-registered criteria H1, H3 and H4 were REJECTED and H2 PASSED (the rule "adopt Lie-group / isometric operators if H3 and H4 pass" is therefore not met). In the generated tables the block-unitary model reaches 78.3 % in distribution and 73.8 / 68.2 / 57.8 / 48.4 % at K = 4 / 6 / 8 / 12, the torus model 48.3 % and 26.2 / 11.9 / 6.4 / 1.9 %, the standard transformer 11.5 % and 5.5 / 3.3 / 2.3 / 1.6 %, the Euclidean RNN 2.9 % and 2.2 / 0.7 / 0.5 / 0.5 %; both isometric models keep the norm ratio at exactly 1.0000 while the Euclidean RNN drifts to 1.0819 (the pre-registered threshold for H4 was 25 %).

## Caveats, consequence, interpretation
Lost with the overwritten file. To be re-added by the author of this entry.

## Evidence
- `research/geomlab_results/phase2/PREREG.md`, `run_phase2.py`, `checkpoint.json`, `results.json`, `tables.txt`
- `geomlab/phase2_composition.py`, `tests/geomlab/test_phase2.py`

## Reproduce
`python research/geomlab_results/phase2/run_phase2.py` (checkpointed; see the script header)
