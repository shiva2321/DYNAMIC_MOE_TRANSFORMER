# References

Honesty tag on each: **L** = read at abstract level (title, abstract, sometimes the repository README) when it was collected; **R** = recalled, not rechecked; **V** = verified in full for a specific claim (none yet in this notebook).
arXiv IDs below were collected on 2026-10-09 (routing note) and 2026-10-06 (geometry atlas); they were not re-fetched in the 2026-10-12 notebook pass. The full geometry bibliography (about 340 entries) is `research/BIBLIOGRAPHY_CHRONOLOGICAL.md` (uncommitted, other track).

## Statistical machinery we rely on

| Reference | Used for | Tag |
|---|---|---|
| Benjamini and Hochberg (1995), Controlling the false discovery rate, JRSS-B | the novelty rule | R |
| Vovk, Gammerman, Shafer (2005), Algorithmic Learning in a Random World | conformal p-values | R |
| Bates, Candes, Lei, Romano, Sesia (2023), Testing for outliers with conformal p-values, Annals of Statistics (cited in `growmoe/conformal.py`) | BH on conformal p-values is valid (positive dependence) | R |
| Kim, Xiao, Liu (2022), A theoretical study on solving continual learning, NeurIPS | within-task prediction x task-id prediction decomposition | R |

## Routing, growth and continual learning (collected 2026-10-09; see `research/ROUTING_RESEARCH_2026-10.md` section 3)

| Reference | Why it matters here | Tag |
|---|---|---|
| SAME: router drift vs expert drift, arXiv 2602.01990 | router drift is a distinct cause of forgetting | L |
| LLaVA-DyMoE: ambiguous-token handling for added experts | old-task tokens are pulled to new experts | L |
| Latent-LoRA: Gaussian-mixture router on frozen embeddings, arXiv 2607.23837 | gradient-free router immune to drift | L |
| Expert Gate (Aljundi et al.), arXiv 1611.06194 | one autoencoder per task as the gate | L |
| Arrow: routing prototypes from the SVD of each expert's weights | gradient-free routing | L |
| FlyPrompt: random expansion plus a closed-form ridge router | gradient-free routing | L |
| FLEX, arXiv 2608.01437 | routing looks solved only on short benchmarks; calibrated soft mixtures beat hard selection; routers recover 28-50 % of the oracle gap | L |
| Hierarchically Gated Experts, arXiv 2412.17188 | loss-threshold gating with an outlier / other-task / instability disambiguation stage | L |
| TOOD / confidence-based task prediction, arXiv 2607.29592 | test-time selection beyond input density | L |
| Autonomy-of-Experts, arXiv 2501.13074; Routing-Free MoE, arXiv 2604.00801; Self-Routing, arXiv 2604.00421 | experts route themselves | L |
| PEER (product keys), arXiv 2407.04153; fast feedforward networks, arXiv 2308.14711 | sublinear routing | L |
| Loss-free load balancing (DeepSeek), arXiv 2408.15664 | fixed-pool maintenance | L |
| MaRS (ICLR 2026), SEMA (CVPR 2025), LLaVA-CMoE | closest published relatives to statistically controlled growth | L |
| SEMA, arXiv 2403.18886 (code: huiyiwang01/SEMA-CL, MIT) | method read: autoencoder descriptor per adapter, z-score of reconstruction error, expand when all descriptors flag shift, task boundaries given | L+ |
| AIR-MoE, arXiv 2605.04952 | two-stage routing: codebook shortlist then exact scores; guarantee on routing mass recall; the same pattern as our stage 2 | L+ |
| Task-level MoE (Kudugunta et al.), arXiv 2110.03742 | routing a whole task or sequence together: +1.0 BLEU, 1.9x throughput; the precedent for group answering | L |
| Conformal test martingales, arXiv 2411.11824; testing by betting (Shekhar and Ramdas), arXiv 2112.09162; crepes (martingale module) | sequential, anytime-valid alternatives to a batch test | L |
| RouterRetriever, arXiv 2409.02685 | add an expert without retraining a pilot-embedding router | L |
| Phipson and Smyth (2010), permutation p-values should never be zero | the (1 + count) / (1 + n) form we use, so the smallest p-value is 1 / (1 + n_perm) | R |
| Besag and Clifford (1991), sequential Monte Carlo p-values | stop at the first exceedance: same decision, far fewer permutations for a known batch | R |
| Bates et al. (2023), testing for outliers with conformal p-values | conformal p-values of test points sharing a calibration set are dependent (PRDS): why the Fisher combination over-fires | R |

## Geometry (Tracks Y, Z)

| Reference | Why it matters here | Tag |
|---|---|---|
| Training nGPT, arXiv 2608.01284; the nGPT paper and community reproductions | unit-sphere residual stream (Z1) | L |
| Park (ICLR 2025); Nava and Wyart (2026); audit arXiv 2607.05268 | hyperbolic is not needed for hierarchy; hyperbolic models often train near-Euclidean | L |
| KATA, PaTH, DeltaProduct, Mamba-3, MapFormer, mHC (DeepSeek-V4) | the atlas candidates for state, memory and residual mixing | L |

## Our own earlier documents

`research/DYNAMIC_MOE_RESEARCH_AND_DESIGN.md` (design literature), `research/MOE_CONTINUAL_LEARNING_REPORT_2026-10.md`, `research/GEOMETRY_ATLAS_2026-10.md` (about 50 arXiv IDs), `research/VSA_GEOMETRY_TO_INTELLIGENCE.md`.
