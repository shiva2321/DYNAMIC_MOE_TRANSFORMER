"""Configuration dataclasses. Thresholds are error rates or tolerances with units, never bare constants."""

from dataclasses import dataclass


@dataclass
class CommonCfg:
    d: int = 128                 # latent dimension (unit vectors from the frontend)
    hidden: int = 128            # expert MLP hidden width
    depth: int = 1               # expert residual blocks
    n_classes: int = 64          # global label-space capacity
    lr: float = 3e-3
    batch_size: int = 64
    inner_steps: int = 1         # optimiser steps per routed batch
    replay_size: int = 256       # per-expert replay reservoir
    replay_frac: float = 0.5     # share of each step's batch drawn from replay
    seed: int = 0


@dataclass
class GrowthCfg:
    widen_factors: tuple = (1.5, 4.0, 16.0)   # candidate NEW widths as multiples of the current width. Spike 3: for a width-32
    #                              teacher the gain of widening a width-4 student is +0.0 % (+4 units), +1.7 % (+12), +2.7 % (+28),
    #                              +5.3 % (+60); proposing only +50 % never sees it
    min_widen: int = 4
    train_steps: int = 480       # probe steps for the control and every shadow. Zero-outgoing-weight units need ~100-200
    #                              steps to engage and the probe gain has ~3-6 % training noise. Measured accept rate over
    #                              12 probes: capacity-limited 50 % at 240 steps, 100 % at 480; sufficient-capacity 8 % -> 0 %;
    #                              data-limited 0 % -> 0 %
    batch_size: int = 64
    lr: float = 3e-3
    min_gain: float = 0.02       # required relative held-out loss reduction vs the equally trained control
    alpha: float = 0.05          # family-wise level of the paired-bootstrap test (Bonferroni over candidates)
    n_boot: int = 1000
    backoff0: int = 1            # calls skipped after a first rejection; doubles with every further rejection
    max_backoff: int = 64
    max_hidden: int = 1024
    max_depth: int = 8


@dataclass
class PhyloCfg(CommonCfg):
    q: float = 0.05              # tolerated false-novelty rate / FDR level
    alpha_spawn: float = 0.01    # evidence level: spawn when log-LR > log(1/alpha_spawn)
    delta_forget: float = 0.05   # nats of parent replay loss we may trade to reuse it
    m_min: int = 32              # samples required to consider forming a regime
    min_share: float = 0.05      # fraction of recent stream; probation TTL = ceil(m_min / min_share) samples
    min_neighbors: int = 4       # DBSCAN core-point degree threshold
    edge_null: float = 0.5       # expected edges per node under the random-null threshold tau
    pool_cap: int = 512          # max samples held in the probation buffer
    probe_steps: int = 60        # training steps for extend-vs-spawn shadow probe
    probe_min_affinity: float = 0.05
    init_mode: str = "shrink_perturb"  # "scratch" | "clone" | "inherit"
    rho_max: float = 0.75
    inherit_sigma: float = 0.05
    infant_calls: int = 5        # observe calls during which an expert has infant status
    infant_lr_mult: float = 2.0
    infant_relax: float = 4.0
    dormant_after: int = 400     # observe calls without assignment before becoming dormant
    archive_after: int = 800     # observe calls without assignment before archiving
    merge_every: int = 50        # interval of observe calls to check for expert merges
    delta_merge: float = 0.02    # maximum tolerated accuracy drop on probe sets during merge
    merge_overlap: float = 0.2   # mutual prototype coverage required to propose a merge
    max_experts: int = 256       # maximum live experts before pool abstains
    agem: bool = False           # enable A-GEM gradient projection during training
    competence: bool = True      # enable prequential loss-change monitor
    monitor_alpha: float = 0.01
    monitor_warmup: int = 50
    arm_monitor_after_infancy: bool = True   # an infant's untrained loss is no baseline: feed its competence monitor only once infancy is over.
    #                              ON since 2026-10-10 (M2-A, fresh seeds 10-19: +10.0 / +11.6 points over the best single network on F-15 / F-30).
    batch_vote: bool = True      # routing v2 stage 1: all accepted samples of a batch train the expert most of them were routed to.
    #                              ON since 2026-10-12 (RV2-A, fresh seeds 20-39, all 11 pre-registered statements PASS: +11.4 points on OV-0.3 [+10.0, +12.9],
    #                              +6.4 on F-30, +3.0 on F-15; research/growmoe_results/rv2a/RESULTS_RV2A.md). Phase 1, M0 and M2-A ran with it OFF; pass False to reproduce them.
    batch_vote_share: float = 0.5    # ... when that expert holds at least this share of the batch's accepted samples
    batch_vote_min: int = 8          # ... and at least this many samples were accepted
    grow: bool = True            # enable capacity growth policy
    no_conformal: bool = False   # ablation: fixed score threshold instead of conformal
    no_probation: bool = False   # ablation: bypass probation buffer clustering
    sig_cls: object = None       # alternative signature class (e.g. KnnSignature)
    route_rule: str = "default"  # test prediction routing rule: "default" | "accept_then_logf" (highest log-density among p >= q; skips supersede filter at test time) | "argmax_logf" | "nearest_proto"


@dataclass
class BNPCfg(CommonCfg):
    alpha_dp: float = 1.0        # CRP concentration parameter
    tau_b: float = 2.0           # standard deviations below running mean for novelty threshold
    n_stm: int = 32              # short-term memory buffer threshold to spawn a scratch expert
    vae_steps: int = 60          # training steps for VAE on STM
    vae_lr: float = 2e-3         # learning rate for VAE optimizer
    vae_h: int = 64              # VAE hidden layer width
    vae_z: int = 16              # VAE latent dimension
    q: float = 0.05              # abstain threshold for prediction unknown flag
    max_experts: int = 256       # maximum live experts


@dataclass
class KeyedCfg(CommonCfg):
    c: float = 2.0               # z-score threshold below running mean cosine for novelty (novel if z < -c)
    n_stm: int = 32              # short-term memory buffer threshold to spawn a delta expert
    rank: int = 16               # low-rank delta dimension
    key_ema: float = 0.05        # EMA momentum for updating expert key
    delta_merge: float = 0.05    # maximum allowed held-out loss rise during sibling merge
    merge_every: int = 50        # interval of observe calls to evaluate sibling merges
    merge_sim: float = 0.85      # cosine similarity of keys required to propose sibling merge
    dormant_after: int = 200     # observe calls without assignment before becoming dormant
    archive_after: int = 500     # observe calls without assignment before archiving
    base_steps: int = 60         # bootstrap training steps for the shared base MLP
    max_experts: int = 256       # maximum live experts
    q: float = 0.05              # abstain threshold for prediction unknown flag

