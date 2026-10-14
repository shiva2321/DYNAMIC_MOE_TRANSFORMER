"""Configuration dataclasses. Thresholds are error rates or tolerances with units, never bare constants."""

import math
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
    gate_shortlist: int = 2      # routing v2 stage 2: shortlist this many experts per input by fingerprint before the exact gate (0 = scan every expert).
    #                              ON since 2026-10-13 (RV2-B, fresh seeds 40-59: C1, C2, C3, C4 and G all PASS, research/growmoe_results/rv2b/RESULTS_RV2B.md).
    #                              Acts only with at least gate_shortlist_min_experts experts (below that the exact scan runs, bit for bit);
    #                              Phase 1, M0, M2-A and RV2-A ran with the exact scan; pass gate_shortlist=0 to force it.
    gate_fingerprint_k: int = 1          # sub-centroids per expert fingerprint
    gate_fingerprint_refresh: int = 128  # rebuild an expert's fingerprint after its signature has seen this many more stream points
    gate_shortlist_min_experts: int = 64 # below this many experts the exact scan is used
    set_gate: bool = False       # routing v2 stage 3a: a batch-level test decides whether the accepted samples of a batch are known or new as a SET.
    #                              OFF until its pre-registered run (RV3-5); docs/superpowers/specs/2026-10-13-routing-v2-stage3-design.md.
    set_gate_alpha: float = 0.001        # level of the admission test: an expert admits a batch when the exact permutation p-value exceeds this
    set_gate_perms: int = 999            # permutations per test; the smallest attainable p-value 1/(1+perms) must not exceed the levels
    set_gate_chunk: int = 16             # permutations drawn per early-stop step
    set_gate_candidates: int = 3         # experts tried per batch (those whose nearest reference mean is nearest the batch mean)
    set_gate_stat: str = "meandist"      # "meandist" (distance of means) | "mmd" (squared MMD, RBF kernel, median bandwidth)
    set_gate_ownership: str = "loss"     # who trains an admitted batch: "loss" (lowest label cross-entropy among the admitted) | "nearest"
    set_gate_min_labels: int = 8         # labelled accepted samples needed for loss ownership (fewer: the nearest admitted expert)
    set_gate_min_ref: int = 48           # rows a source window needs before it counts as a reference
    set_gate_windows: int = 4            # source windows kept per expert
    set_gate_window_batches: int = 5     # batches per source window
    set_gate_split_alpha: float = 0.01   # level of the test that opens a new source window
    set_gate_max_rows: int = 512         # rows above which a reference, a stored batch or a tested batch is cut to a random subset (bounds memory and time)
    set_gate_breaker_len: int = 12       # circuit breaker: among the last this-many familiar-looking batches ...
    set_gate_breaker_rate: float = 0.35  # ... more than this share called new as a set pauses the set level ...
    set_gate_breaker_pause: int = 100    # ... for this many steps
    set_gate_confirm: int = 2            # persistence (amendment A-15): a "new as a set" verdict is acted on only when this many decided batches in a row are doubtful;
    #                              the earlier doubtful batches are HELD (their accepted and flagged samples wait, nothing is trained or sent to probation). 1 = act at once.
    group_vote: bool = False     # routing v2 stage 3b: predict(x, groups=...) answers a group by the majority of its inputs' own choices (OFF until RV3-5)
    group_min: int = 8           # smallest group that is answered together
    group_veto: bool = True      # an input whose exact p-value under the group's expert is below group_veto_q keeps its own choice
    group_veto_q: float = 0.2    # amendment A-16: the level of that veto, separate from q (dev seeds: 0.2 balances the three group statements, margins about 3 points each)
    grow: bool = True            # enable capacity growth policy
    no_conformal: bool = False   # ablation: fixed score threshold instead of conformal
    no_probation: bool = False   # ablation: bypass probation buffer clustering
    sig_cls: object = None       # alternative signature class (e.g. KnnSignature)
    route_rule: str = "default"  # test prediction routing rule: "default" | "accept_then_logf" (highest log-density among p >= q; skips supersede filter at test time) | "argmax_logf" | "nearest_proto"

    def __post_init__(self) -> None:
        """Refuse stage-3 settings that would silently disable or break the test (a level the permutation count can never reach, an unknown name)."""
        if self.set_gate_stat not in ("meandist", "mmd"):
            raise ValueError(f"set_gate_stat must be 'meandist' or 'mmd', got {self.set_gate_stat!r}")
        if self.set_gate_ownership not in ("loss", "nearest"):
            raise ValueError(f"set_gate_ownership must be 'loss' or 'nearest', got {self.set_gate_ownership!r}")
        for name in ("set_gate_alpha", "set_gate_split_alpha"):
            level = getattr(self, name)
            if not 0.0 < level < 1.0:
                raise ValueError(f"{name} must lie in (0, 1), got {level}")
            if math.floor(level * (1 + self.set_gate_perms) + 1e-9) < 1:                    # the rule of growmoe.settest.max_exceedances
                raise ValueError(f"set_gate_perms={self.set_gate_perms} is too small for {name}={level}: the smallest attainable p-value 1/(1+perms) exceeds the level, so the test could never reject")
        if not 0.0 <= self.set_gate_breaker_rate < 1.0:
            raise ValueError("set_gate_breaker_rate must lie in [0, 1)")
        if not 0.0 < self.group_veto_q <= 1.0:
            raise ValueError(f"group_veto_q must lie in (0, 1], got {self.group_veto_q}")
        if self.set_gate and self.set_gate_confirm > 1 and self.no_probation:
            raise ValueError("set_gate_confirm > 1 holds doubtful samples for the probation buffer, so it cannot be combined with no_probation")
        for name in ("set_gate_perms", "set_gate_chunk", "set_gate_candidates", "set_gate_min_labels", "set_gate_min_ref", "set_gate_windows",
                     "set_gate_window_batches", "set_gate_max_rows", "set_gate_breaker_len", "set_gate_breaker_pause", "group_min", "set_gate_confirm"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be at least 1, got {getattr(self, name)}")


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

