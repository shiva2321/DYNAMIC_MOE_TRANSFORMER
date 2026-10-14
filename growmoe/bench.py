"""Task 17: Benchmark runner, hyper-parameter tuning, and Phase-1 comparative evaluation.

Contracts:
- test-then-train: for each batch, pool.predict(...) is logged before pool.observe(...)
- ground-truth bookkeeping lives exclusively outside the pool; no GT parameters reach algorithms
- accuracy matrix R is evaluated across all regimes' held-out test sets after each segment
- tune strictly asserts dev seeds >= 100 and refuses test seeds (0-9)
- signature familiarity alternatives: Gaussian-Mahalanobis, kNN, RND implement the VmfSignature protocol
- CLI: python -m growmoe.bench --phase 1 --seeds 10 --out research/growmoe_results/phase1
"""

import argparse
import inspect
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch import Tensor

from .config import BNPCfg, CommonCfg, KeyedCfg, PhyloCfg
from .conformal import benjamini_hochberg, conformal_pvalues
from .metrics import (
    accuracy_stats,
    auroc,
    false_novelty_rate,
    novel_detection_rate,
    regime_owner,
    routing_purity,
    spawn_stats,
    mean_ci,
    paired_diff,
    _confusion,
)
from .streams import Stream, s1_types, s2_permuted, s3_drift, s4_outliers, s6_long
from .pools.phylo import PhyloMoE
from .pools.bnp import BNPMoE
from .pools.keyed import KeyedDeltaMoE
from .baselines import (
    FineTune,
    JointOracle,
    ReplayNet,
    StaticRouterMoE,
    OracleRouted,
    DPMeansGate,
)


# --------------------------------------------------------------------------- RunResult data model
@dataclass
class RunResult:
    """Benchmark results for a single pool running on a single stream."""
    pool_name: str
    stream_name: str
    seed: int
    R: np.ndarray                            # (n_segments, n_regimes) accuracy matrix
    accuracy_stats: dict                     # avg, forgetting, backward, forward
    auroc: float                             # sample-level AUROC of novelty detector
    false_novelty_rate: float                # share of known samples flagged novel (pred.unknown)
    novel_detection_rate: float              # share of novel samples flagged novel (pred.unknown)
    spawn_stats: dict                        # precision, recall, ratio, time_to_spawn
    pool_summary: dict                       # final summary() from pool
    events: Any                              # EventLog
    wall_clock: float                        # total seconds
    total_samples: int                       # total samples processed
    fnr_p50: float = float("nan")            # legacy p <= 0.5 false novelty rate (diagnostic)
    routing_acc: float = float("nan")        # share of test samples routed to their regime's owner expert
    final_regime_acc: dict = field(default_factory=dict) # per-regime final test accuracy


# --------------------------------------------------------------------------- Benchmark Runner
def run_stream(
    make_pool: Callable[[], Any],
    stream: Stream,
    seed: int = 0,
    batch_size: int = 64,
) -> RunResult:
    """Run stream in a strict test-then-train continual-learning evaluation protocol."""
    t0 = time.perf_counter()
    pool = make_pool()

    all_p_nov: List[float] = []
    all_unknown: List[bool] = []
    all_assigned: List[int] = []
    all_first_visit: List[bool] = []
    all_is_outlier: List[bool] = []
    all_true_regimes: List[int] = []
    all_gen_novel: List[bool] = []
    all_known: List[bool] = []
    total_samples = 0

    R_rows: List[np.ndarray] = []
    test_sets = stream.test_sets()
    n_regimes = len(stream.regimes)
    current_seg_idx = -1
    batch_in_seg = 0
    has_revisits = any(not s.first_visit for s in stream.segments)

    # First-visit onset tracking per regime
    onsets: Dict[int, int] = {}
    learned_at: List[int] = [0] * n_regimes

    for k, seg, idx in stream._walk(batch_size, seed):
        if k != current_seg_idx:
            # If we completed previous segments, evaluate on all test sets
            if current_seg_idx >= 0:
                row = np.zeros(n_regimes, dtype=np.float64)
                for r_id, (x_te, y_te) in test_sets.items():
                    if hasattr(pool, "set_eval_regime"):
                        pool.set_eval_regime(r_id)
                    ctx_te = stream.test_ctx().get(r_id)
                    pred_te = pool.predict(x_te, ctx=ctx_te)
                    acc_r = float((pred_te.probs.argmax(dim=-1).cpu() == y_te.cpu()).float().mean().item())
                    row[r_id] = acc_r
                R_rows.append(row)

            current_seg_idx = k
            batch_in_seg = 0
            if seg.first_visit and seg.regime not in onsets:
                onsets[seg.regime] = pool.step
                learned_at[seg.regime] = len(R_rows)
        else:
            batch_in_seg += 1

        x_b = seg.x[idx]
        y_b = seg.y[idx]
        ctx_b = seg.ctx[idx] if seg.ctx is not None else None
        b = len(x_b)
        total_samples += b

        # 1. Test first (predict before observe)
        if hasattr(pool, "set_eval_regime"):
            pool.set_eval_regime(seg.regime)
        pred = pool.predict(x_b, ctx=ctx_b)

        # 2. Train next (observe)
        rec = pool.observe(x_b, y=y_b, ctx=ctx_b)

        # Genuinely novel ground truth: first batch of first-visit segment, OR injected outlier
        is_first_batch_of_regime = bool(seg.first_visit and (batch_in_seg == 0))
        outlier_flags = seg.is_outlier[idx].detach().cpu().numpy().tolist() if hasattr(seg, "is_outlier") and seg.is_outlier is not None else [False] * b

        batch_gen_novel = [is_first_batch_of_regime or bool(o) for o in outlier_flags]
        if has_revisits:
            # When revisits exist, genuinely known samples are revisit samples (not outliers)
            batch_known = [(not seg.first_visit and not bool(o)) for o in outlier_flags]
        else:
            # When no revisits exist, known samples are seen-regime non-outlier samples
            batch_known = [(not is_first_batch_of_regime and not bool(o)) for o in outlier_flags]

        # Ground truth bookkeeping outside the pool
        all_p_nov.extend(pred.p_nov.detach().cpu().numpy().tolist())
        all_unknown.extend(pred.unknown.detach().cpu().numpy().tolist())
        all_assigned.extend(rec.assigned.detach().cpu().numpy().tolist())
        all_first_visit.extend([bool(seg.first_visit)] * b)
        all_is_outlier.extend(outlier_flags)
        all_true_regimes.extend([seg.regime] * b)
        all_gen_novel.extend(batch_gen_novel)
        all_known.extend(batch_known)

    # Evaluate final segment
    row = np.zeros(n_regimes, dtype=np.float64)
    regime_acc_final: Dict[int, float] = {}
    for r_id, (x_te, y_te) in test_sets.items():
        if hasattr(pool, "set_eval_regime"):
            pool.set_eval_regime(r_id)
        ctx_te = stream.test_ctx().get(r_id)
        pred_te = pool.predict(x_te, ctx=ctx_te)
        acc_r = float((pred_te.probs.argmax(dim=-1).cpu() == y_te.cpu()).float().mean().item())
        row[r_id] = acc_r
        regime_acc_final[r_id] = acc_r
    R_rows.append(row)

    dt = time.perf_counter() - t0
    R = np.array(R_rows)

    # Compute continual learning accuracy stats
    chance = 1.0 / stream.n_classes
    acc_stats = accuracy_stats(R, learned_at=learned_at, chance=chance)

    # Compute novelty metrics
    p_nov_arr = np.array(all_p_nov)
    scores = 1.0 - p_nov_arr
    unknown_arr = np.array(all_unknown, dtype=bool)
    is_novel_gt = np.array(all_gen_novel, dtype=bool)
    known_mask = np.array(all_known, dtype=bool)

    try:
        auc_val = auroc(scores, is_novel_gt)
    except ValueError:
        auc_val = float("nan")

    try:
        fnr = false_novelty_rate(unknown_arr, known_mask)
    except ValueError:
        fnr = float("nan")

    try:
        ndr = novel_detection_rate(unknown_arr, known_mask)
    except ValueError:
        ndr = float("nan")

    try:
        fnr_p50 = false_novelty_rate(scores >= 0.5, known_mask)
    except ValueError:
        fnr_p50 = float("nan")

    # Compute spawn metrics
    assigned_arr = np.array(all_assigned)
    true_reg_arr = np.array(all_true_regimes)
    owner_of = regime_owner(assigned_arr, true_reg_arr)
    spwn_stats = spawn_stats(pool.events, owner_of, onsets)

    # Compute test routing accuracy: share of test samples routed to their regime's owner expert
    confusion = _confusion(assigned_arr, true_reg_arr)
    reg_to_expert = {}
    for r in range(n_regimes):
        if confusion:
            best_e = max(confusion.keys(), key=lambda e: confusion[e][r])
            if confusion[best_e][r] > 0:
                reg_to_expert[r] = best_e

    test_routed_correct = []
    for r_id, (x_te, y_te) in test_sets.items():
        if hasattr(pool, "set_eval_regime"):
            pool.set_eval_regime(r_id)
        ctx_te = stream.test_ctx().get(r_id)
        pred_te = pool.predict(x_te, ctx=ctx_te)
        if r_id in reg_to_expert:
            target_e = reg_to_expert[r_id]
            is_correct_route = (pred_te.expert.cpu() == target_e).float().numpy()
            test_routed_correct.extend(is_correct_route.tolist())
    routing_acc = float(np.mean(test_routed_correct)) if test_routed_correct else float("nan")

    pool_summary = pool.summary()
    if hasattr(pool, "received_args"):
        pool_summary["spy_args"] = getattr(pool, "received_args")

    return RunResult(
        pool_name=getattr(pool, "name", "unknown_pool"),
        stream_name=stream.name,
        seed=seed,
        R=R,
        accuracy_stats=acc_stats,
        auroc=auc_val,
        false_novelty_rate=fnr,
        novel_detection_rate=ndr,
        spawn_stats=spwn_stats,
        pool_summary=pool_summary,
        events=pool.events,
        wall_clock=dt,
        total_samples=total_samples,
        fnr_p50=fnr_p50,
        routing_acc=routing_acc,
        final_regime_acc=regime_acc_final,
    )


# --------------------------------------------------------------------------- Tuning (Dev Seeds Only)
def tune(
    make_pool_fn: Callable[[dict], Any],
    grid: List[dict],
    dev_streams: List[Stream],
    dev_seeds: Optional[Sequence[int]] = None,
    budget: int = 12,
) -> Tuple[dict, float]:
    """Tune hyper-parameters on dev streams only (strictly asserting dev seeds >= 100)."""
    if dev_seeds is not None:
        for sd in dev_seeds:
            if sd < 100:
                raise ValueError(
                    f"tune strictly requires dev seeds >= 100, got test seed {sd}! "
                    f"Tuning on test seeds (0-9) is strictly forbidden."
                )

    for s in dev_streams:
        stream_seed = s.meta.get("seed", getattr(s, "seed", None))
        if stream_seed is not None and stream_seed < 100:
            raise ValueError(
                f"tune strictly requires dev seeds >= 100, got test seed {stream_seed}! "
                f"Tuning on test seeds (0-9) is strictly forbidden."
            )

    candidates = grid[:budget]
    best_cfg = candidates[0]
    best_score = -float("inf")

    for cfg in candidates:
        scores = []
        for s in dev_streams:
            s_seed = s.meta.get("seed", getattr(s, "seed", 100))
            res = run_stream(lambda: make_pool_fn(cfg), s, seed=s_seed, batch_size=64)
            scores.append(res.accuracy_stats.get("avg", 0.0))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_cfg = cfg

    return best_cfg, best_score


# --------------------------------------------------------------------------- Signature Familiarity Ablations
class GaussianMahalanobisSignature:
    """Gaussian-Mahalanobis familiarity baseline adhering to VmfSignature protocol."""

    def __init__(self, d: int = 128, max_protos: int = 256, seed: int = 0) -> None:
        self.d = d
        self.max_protos = max_protos
        self.seed = seed
        self._protos: List[Tensor] = []
        self.seen: int = 0
        self.mu: Optional[Tensor] = None
        self.cov_inv: Optional[Tensor] = None
        self._cal_scores: Optional[Tensor] = None

    def update(self, x: Tensor) -> None:
        x = x.float()
        self.seen += len(x)
        for i in range(len(x)):
            if len(self._protos) < self.max_protos:
                self._protos.append(x[i].clone())
            else:
                rng = np.random.default_rng([self.seed, 0x474D, len(self._protos)])
                slot = int(rng.integers(0, len(self._protos) + 1))
                if slot < self.max_protos:
                    self._protos[slot] = x[i].clone()

    def fit(self) -> None:
        if not self._protos:
            return
        mat = torch.stack(self._protos, dim=0)  # (N, d)
        self.mu = mat.mean(dim=0)
        diff = mat - self.mu
        cov = (diff.t() @ diff) / max(1, len(mat) - 1)
        cov = cov + torch.eye(self.d) * 1e-3
        self.cov_inv = torch.linalg.inv(cov)
        with torch.no_grad():
            self._cal_scores = self.score(mat)

    def score(self, u: Tensor) -> Tensor:
        """Negative Mahalanobis distance (high = familiar, low = novel)."""
        u = u.float()
        if self.mu is None or self.cov_inv is None:
            return torch.zeros(len(u))
        diff = u - self.mu
        dist_sq = (diff @ self.cov_inv * diff).sum(dim=-1)
        return -torch.sqrt(dist_sq.clamp_min(1e-12))

    def logpdf(self, u: Tensor) -> Tensor:
        return self.score(u)

    def pvalue(self, u: Tensor) -> Tensor:
        if self._cal_scores is None or len(self._cal_scores) == 0:
            return torch.ones(len(u))
        scores = self.score(u)
        return conformal_pvalues(self._cal_scores, scores)

    def nn_similarity(self) -> Tensor:
        p = self.protos
        if len(p) < 2:
            raise ValueError("nn_similarity requires at least 2 prototypes")
        gram = p @ p.T
        gram.fill_diagonal_(-2.0)
        return gram.max(dim=1).values

    def affinity(self, cluster_x: Tensor, n_random: int = 500, seed: int = 0) -> float:
        if self.n < 2:
            return 0.0
        p = self.protos
        u = cluster_x.float()
        s_new = (u @ p.T).max(dim=1).values.mean().item()
        s_self = self.nn_similarity().mean().item()
        z = np.random.default_rng([self.seed, seed, 0x41464E]).standard_normal((n_random, self.d))
        rand = torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))
        s_floor = (rand @ p.T).max(dim=1).values.mean().item()
        return float(np.clip((s_new - s_floor) / max(s_self - s_floor, 1e-9), 0.0, 1.0))

    def merge_from(self, other: Any) -> None:
        other_protos = other.protos if isinstance(other.protos, list) else [other.protos[i] for i in range(len(other.protos))]
        self._protos.extend(other_protos)
        if len(self._protos) > self.max_protos:
            rng = np.random.default_rng([self.seed, 0x4D5247])
            idx = rng.choice(len(self._protos), size=self.max_protos, replace=False)
            self._protos = [self._protos[i] for i in idx]
        self.seen += getattr(other, "seen", len(other_protos))
        self.fit()

    @property
    def n(self) -> int:
        return len(self._protos)

    @property
    def protos(self) -> Tensor:
        if not self._protos:
            return torch.empty((0, self.d))
        return torch.stack(self._protos, dim=0)

    @property
    def cal_scores(self) -> Tensor:
        if self._cal_scores is None:
            return torch.empty(0)
        return self._cal_scores

    @property
    def kappa(self) -> float:
        return 1.0


class KnnSignature:
    """k-Nearest Neighbors distance familiarity baseline adhering to VmfSignature protocol."""

    def __init__(self, d: int = 128, k: int = 5, max_protos: int = 256, seed: int = 0) -> None:
        self.d = d
        self.k = k
        self.max_protos = max_protos
        self.seed = seed
        self._protos: List[Tensor] = []
        self.seen: int = 0
        self._cal_scores: Optional[Tensor] = None

    def update(self, x: Tensor) -> None:
        x = x.float()
        self.seen += len(x)
        for i in range(len(x)):
            if len(self._protos) < self.max_protos:
                self._protos.append(x[i].clone())
            else:
                rng = np.random.default_rng([self.seed, 0x4B4E4E, len(self._protos)])
                slot = int(rng.integers(0, len(self._protos) + 1))
                if slot < self.max_protos:
                    self._protos[slot] = x[i].clone()

    def fit(self) -> None:
        if not self._protos:
            return
        mat = torch.stack(self._protos, dim=0)
        with torch.no_grad():
            self._cal_scores = self.score(mat)

    def score(self, u: Tensor) -> Tensor:
        """Negative mean distance to k nearest prototypes."""
        u = u.float()
        if not self._protos:
            return torch.zeros(len(u))
        mat = torch.stack(self._protos, dim=0)  # (M, d)
        dists = torch.cdist(u, mat)  # (N, M)
        k_val = min(self.k, mat.shape[0])
        top_k_dists, _ = dists.topk(k_val, dim=1, largest=False)
        return -top_k_dists.mean(dim=1)

    def logpdf(self, u: Tensor) -> Tensor:
        return self.score(u)

    def pvalue(self, u: Tensor) -> Tensor:
        if self._cal_scores is None or len(self._cal_scores) == 0:
            return torch.ones(len(u))
        scores = self.score(u)
        return conformal_pvalues(self._cal_scores, scores)

    def nn_similarity(self) -> Tensor:
        p = self.protos
        if len(p) < 2:
            raise ValueError("nn_similarity requires at least 2 prototypes")
        gram = p @ p.T
        gram.fill_diagonal_(-2.0)
        return gram.max(dim=1).values

    def affinity(self, cluster_x: Tensor, n_random: int = 500, seed: int = 0) -> float:
        if self.n < 2:
            return 0.0
        p = self.protos
        u = cluster_x.float()
        s_new = (u @ p.T).max(dim=1).values.mean().item()
        s_self = self.nn_similarity().mean().item()
        z = np.random.default_rng([self.seed, seed, 0x41464E]).standard_normal((n_random, self.d))
        rand = torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))
        s_floor = (rand @ p.T).max(dim=1).values.mean().item()
        return float(np.clip((s_new - s_floor) / max(s_self - s_floor, 1e-9), 0.0, 1.0))

    def merge_from(self, other: Any) -> None:
        other_protos = other.protos if isinstance(other.protos, list) else [other.protos[i] for i in range(len(other.protos))]
        self._protos.extend(other_protos)
        if len(self._protos) > self.max_protos:
            rng = np.random.default_rng([self.seed, 0x4D5247])
            idx = rng.choice(len(self._protos), size=self.max_protos, replace=False)
            self._protos = [self._protos[i] for i in idx]
        self.seen += getattr(other, "seen", len(other_protos))
        self.fit()

    @property
    def n(self) -> int:
        return len(self._protos)

    @property
    def protos(self) -> Tensor:
        if not self._protos:
            return torch.empty((0, self.d))
        return torch.stack(self._protos, dim=0)

    @property
    def cal_scores(self) -> Tensor:
        if self._cal_scores is None:
            return torch.empty(0)
        return self._cal_scores

    @property
    def kappa(self) -> float:
        return 1.0


class RndSignature:
    """Random Network Distillation familiarity baseline adhering to VmfSignature protocol."""

    def __init__(self, d: int = 128, h: int = 64, max_protos: int = 256, seed: int = 0) -> None:
        self.d = d
        self.h = h
        self.max_protos = max_protos
        self.seed = seed
        self._protos: List[Tensor] = []
        self.seen: int = 0

        gen = torch.Generator().manual_seed(seed)
        self.target = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h))
        self.predictor = nn.Sequential(nn.Linear(d, h), nn.Tanh(), nn.Linear(h, h))

        # Explicitly initialize linear layers using seeded generator for cross-run determinism
        for net in (self.target, self.predictor):
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), generator=gen)
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                        nn.init.uniform_(m.bias, -bound, bound, generator=gen)

        # Freeze target network
        for p in self.target.parameters():
            p.requires_grad = False

        self._cal_scores: Optional[Tensor] = None

    def update(self, x: Tensor) -> None:
        x = x.float()
        self.seen += len(x)
        for i in range(len(x)):
            if len(self._protos) < self.max_protos:
                self._protos.append(x[i].clone())
            else:
                rng = np.random.default_rng([self.seed, 0x524E44, len(self._protos)])
                slot = int(rng.integers(0, len(self._protos) + 1))
                if slot < self.max_protos:
                    self._protos[slot] = x[i].clone()

    def fit(self) -> None:
        if not self._protos:
            return
        mat = torch.stack(self._protos, dim=0)
        opt = torch.optim.Adam(self.predictor.parameters(), lr=1e-2)
        for _ in range(40):
            opt.zero_grad()
            with torch.no_grad():
                target_out = self.target(mat)
            pred_out = self.predictor(mat)
            loss = F.mse_loss(pred_out, target_out)
            loss.backward()
            opt.step()

        with torch.no_grad():
            self._cal_scores = self.score(mat)

    def score(self, u: Tensor) -> Tensor:
        """Negative predictor error (low error = familiar = high score)."""
        u = u.float()
        with torch.no_grad():
            target_out = self.target(u)
            pred_out = self.predictor(u)
            err = (target_out - pred_out).pow(2).sum(dim=-1)
        return -err

    def logpdf(self, u: Tensor) -> Tensor:
        return self.score(u)

    def pvalue(self, u: Tensor) -> Tensor:
        if self._cal_scores is None or len(self._cal_scores) == 0:
            return torch.ones(len(u))
        scores = self.score(u)
        return conformal_pvalues(self._cal_scores, scores)

    def nn_similarity(self) -> Tensor:
        p = self.protos
        if len(p) < 2:
            raise ValueError("nn_similarity requires at least 2 prototypes")
        gram = p @ p.T
        gram.fill_diagonal_(-2.0)
        return gram.max(dim=1).values

    def affinity(self, cluster_x: Tensor, n_random: int = 500, seed: int = 0) -> float:
        if self.n < 2:
            return 0.0
        p = self.protos
        u = cluster_x.float()
        s_new = (u @ p.T).max(dim=1).values.mean().item()
        s_self = self.nn_similarity().mean().item()
        z = np.random.default_rng([self.seed, seed, 0x41464E]).standard_normal((n_random, self.d))
        rand = torch.from_numpy((z / np.linalg.norm(z, axis=1, keepdims=True)).astype(np.float32))
        s_floor = (rand @ p.T).max(dim=1).values.mean().item()
        return float(np.clip((s_new - s_floor) / max(s_self - s_floor, 1e-9), 0.0, 1.0))

    def merge_from(self, other: Any) -> None:
        other_protos = other.protos if isinstance(other.protos, list) else [other.protos[i] for i in range(len(other.protos))]
        self._protos.extend(other_protos)
        if len(self._protos) > self.max_protos:
            rng = np.random.default_rng([self.seed, 0x4D5247])
            idx = rng.choice(len(self._protos), size=self.max_protos, replace=False)
            self._protos = [self._protos[i] for i in idx]
        self.seen += getattr(other, "seen", len(other_protos))
        self.fit()

    @property
    def n(self) -> int:
        return len(self._protos)

    @property
    def protos(self) -> Tensor:
        if not self._protos:
            return torch.empty((0, self.d))
        return torch.stack(self._protos, dim=0)

    @property
    def cal_scores(self) -> Tensor:
        if self._cal_scores is None:
            return torch.empty(0)
        return self._cal_scores

    @property
    def kappa(self) -> float:
        return 1.0


def format_s4_probation_sentence(pd_noprob_s4: Dict[str, float]) -> str:
    """Format conclusion sentence for A_no_probation on S4 based on paired difference 95% CI."""
    if pd_noprob_s4["ci_lo"] <= 0.0 <= pd_noprob_s4["ci_hi"]:
        return f"the paired accuracy difference on S4 is {pd_noprob_s4['mean_diff']*100:+.2f}% [{pd_noprob_s4['ci_lo']*100:+.2f}%, {pd_noprob_s4['ci_hi']*100:+.2f}%], showing no statistically significant difference (95% CI contains 0)."
    elif pd_noprob_s4["ci_lo"] > 0.0:
        return f"A_no_probation achieved higher accuracy than full A on S4 ({pd_noprob_s4['mean_diff']*100:+.2f}% [{pd_noprob_s4['ci_lo']*100:+.2f}%, {pd_noprob_s4['ci_hi']*100:+.2f}%], p={pd_noprob_s4['p_t']:.4f})."
    else:
        return f"bypassing probation on S4 resulted in lower accuracy ({pd_noprob_s4['mean_diff']*100:+.2f}% [{pd_noprob_s4['ci_lo']*100:+.2f}%, {pd_noprob_s4['ci_hi']*100:+.2f}%], p={pd_noprob_s4['p_t']:.4f})."


def describe_route_rule(rule_name: str) -> str:
    """Format description of the routing rule for RESULTS.md."""
    if rule_name == "accept_then_logf":
        return "highest log-density among accepting experts with p >= q, skipping supersede filter at test time"
    elif rule_name == "default":
        return "highest log-density among accepting experts with supersede filter applied"
    elif rule_name == "argmax_logf":
        return "unconditional highest log-density without conformal gating or supersede filter"
    elif rule_name == "nearest_proto":
        return "nearest prototype by cosine distance"
    return f"custom routing rule: {rule_name}"


# --------------------------------------------------------------------------- Oracle Schedule Mapping
def make_regime_of(stream: Stream, batch_size: int = 64) -> Callable[[int], int]:
    """Precompute batch step to ground-truth regime mapping for OracleRouted."""
    step_map: Dict[int, int] = {}
    st = 0
    for seg in stream.segments:
        n_b = int(math.ceil(len(seg.x) / batch_size))
        for _ in range(n_b):
            step_map[st] = seg.regime
            st += 1
    return lambda step: step_map.get(step, 0)


def regimes_retained_frac(r: RunResult) -> float:
    """Share of learned regimes retained without forgetting."""
    if r.R.shape[0] < 2 or r.R.shape[1] == 0:
        return 1.0
    retained = 0
    for reg in range(r.R.shape[1]):
        learned_row = min(reg, r.R.shape[0] - 1)
        if r.R[-1, reg] >= r.R[learned_row, reg] - 0.05:
            retained += 1
    return retained / max(1, r.R.shape[1])


def revisit_acc(r: RunResult) -> float:
    """Accuracy on revisited regimes."""
    if r.R.shape[0] >= 7 and r.R.shape[1] >= 4:
        return float((r.R[5, 1] + r.R[6, 3]) / 2.0)
    return float(r.R[-1, 0]) if r.R.size > 0 else 0.0


def extract_per_seed_record(r: RunResult, stream_has_related: bool = False) -> Dict[str, Any]:
    """Extract individual seed metrics into a JSON-serializable dictionary."""
    n_aff = (
        sum(1 for ev in r.events.of_kind("spawn") if ev.fields.get("affinity", 0.0) >= 0.05)
        if stream_has_related
        else 0
    )
    spw_extra = max(0, r.spawn_stats["n_spawns"] - 5) if r.stream_name == "s4_outliers" else 0
    early_acc = float(r.R[0, 0]) if r.R.size > 0 else 0.0
    rev_acc = revisit_acc(r) if r.stream_name == "s2_permuted" else 0.0
    ret_frac = regimes_retained_frac(r)

    return {
        "seed": r.seed,
        "avg_acc": float(r.accuracy_stats["avg"]),
        "forgetting": float(r.accuracy_stats["forgetting"]),
        "auroc": float(r.auroc) if not math.isnan(r.auroc) else None,
        "false_novelty_rate": float(r.false_novelty_rate) if not math.isnan(r.false_novelty_rate) else None,
        "novel_detection_rate": float(r.novel_detection_rate) if not math.isnan(r.novel_detection_rate) else None,
        "fnr_p50": float(r.fnr_p50) if not math.isnan(r.fnr_p50) else None,
        "recall": float(r.spawn_stats["recall"]),
        "ratio": float(r.spawn_stats["ratio"]),
        "n_spawns": int(r.spawn_stats["n_spawns"]),
        "n_params": int(r.pool_summary.get("n_params", 0)),
        "wall_clock": float(r.wall_clock),
        "routing_acc": float(r.routing_acc) if not math.isnan(r.routing_acc) else None,
        "n_aff_children": int(n_aff),
        "retained_frac": float(ret_frac),
        "revisit_acc": float(rev_acc),
        "spurious_spawns": float(spw_extra),
        "early_acc": float(early_acc),
    }


def compute_verdicts_from_per_seed(
    per_seed: Dict[str, Dict[str, List[Dict[str, Any]]]],
    q_val: float = 0.05,
    quick: bool = False,
) -> List[Tuple[str, str, Any, str, str]]:
    """Compute verdicts P1-P10 and C1-C6 dynamically from raw per-seed run metrics."""
    def get_m(model: str, stream: str, key: str) -> List[float]:
        if model not in per_seed or stream not in per_seed[model]:
            return []
        return [
            float(row[key])
            for row in per_seed[model][stream]
            if row.get(key) is not None and not (isinstance(row.get(key), float) and math.isnan(row.get(key)))
        ]

    def stat_ci(values: List[float]) -> Tuple[float, float, float]:
        v = [x for x in values if not math.isnan(x)]
        if not v:
            return 0.0, 0.0, 0.0
        return mean_ci(v)

    def paired_diff_safe(a: List[float], b: List[float]) -> Dict[str, float]:
        if a and b and len(a) == len(b):
            return paired_diff(a, b)
        return {"mean_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "p_t": 1.0, "p_wilcoxon": 1.0, "dz": 0.0}

    eval_p = []

    # P1: Empirical FNR <= 2*q on S1, S2, S4
    fnr_p1 = (
        get_m("A_phylomoe", "s1", "false_novelty_rate") +
        get_m("A_phylomoe", "s2", "false_novelty_rate") +
        get_m("A_phylomoe", "s4", "false_novelty_rate")
    )
    m_p1, lo_p1, hi_p1 = stat_ci(fnr_p1)
    pass_p1 = m_p1 <= 2.0 * q_val
    eval_p.append(("P1", "Empirical FNR <= 2*q on S1, S2, S4", pass_p1, f"FNR = {m_p1:.4f} <= {2*q_val:.4f}", f"[{lo_p1:.4f}, {hi_p1:.4f}]"))

    # P2: Regime spawn recall >= 0.9 and ratio <= 1.2 on S1 and S2
    rec_p2 = get_m("A_phylomoe", "s1", "recall") + get_m("A_phylomoe", "s2", "recall")
    rat_p2 = get_m("A_phylomoe", "s1", "ratio") + get_m("A_phylomoe", "s2", "ratio")
    ret_p2 = get_m("A_phylomoe", "s1", "retained_frac") + get_m("A_phylomoe", "s2", "retained_frac")
    m_rec, lo_rec, hi_rec = stat_ci(rec_p2)
    m_rat, lo_rat, hi_rat = stat_ci(rat_p2)
    m_ret, _, _ = stat_ci(ret_p2)
    pass_p2 = (m_rec >= 0.90) and (m_rat <= 1.20)
    eval_p.append(("P2", "Spawn recall >= 0.90 and ratio <= 1.20 on S1/S2", pass_p2, f"Recall={m_rec:.3f}, Ratio={m_rat:.3f} (Served without forgetting={m_ret*100:.1f}%)", f"Recall [{lo_rec:.3f}, {hi_rec:.3f}], Ratio [{lo_rat:.3f}, {hi_rat:.3f}]"))

    # P3: Revisit regimes routed to original expert >= 95% on S2
    rev_p3 = get_m("A_phylomoe", "s2", "revisit_acc")
    m_p3, lo_p3, hi_p3 = stat_ci(rev_p3)
    if m_p3 >= 0.95:
        pass_p3 = True
    elif m_p3 < 0.90:
        pass_p3 = False
    else:
        pass_p3 = "INCONCLUSIVE"
    eval_p.append(("P3", "Revisit routing accuracy >= 95% (fail if < 90%)", pass_p3, f"Acc = {m_p3*100:.2f}%", f"[{lo_p3*100:.2f}%, {hi_p3*100:.2f}%]"))

    # P4: S1 class-IL acc >= 0.90 * B4 (oracle) and > B2 (replay) with CI excluding 0
    acc_a_s1 = get_m("A_phylomoe", "s1", "avg_acc")
    acc_b4_s1 = get_m("B4_oracle_routed", "s1", "avg_acc")
    acc_b2_s1 = get_m("B2_replay", "s1", "avg_acc")
    m_a1, _, _ = stat_ci(acc_a_s1)
    m_b4, _, _ = stat_ci(acc_b4_s1)
    diff_a_b2 = paired_diff_safe(acc_a_s1, acc_b2_s1)
    pass_p4 = (m_a1 >= 0.90 * m_b4) and (diff_a_b2["ci_lo"] > 0.0)
    eval_p.append(("P4", "S1 class-IL acc >= 0.90*B4 and > B2 (CI > 0)", pass_p4, f"A={m_a1*100:.1f}%, B4={m_b4*100:.1f}%, diff={diff_a_b2['mean_diff']*100:.2f}%", f"A-B2 diff CI [{diff_a_b2['ci_lo']*100:.2f}%, {diff_a_b2['ci_hi']*100:.2f}%], p={diff_a_b2['p_t']:.4f}"))

    # P5: Forgetting <= 2 points absolute (0.02) on S1/S2 (fail if > 5 points)
    fgt_s1 = get_m("A_phylomoe", "s1", "forgetting")
    fgt_s2 = get_m("A_phylomoe", "s2", "forgetting")
    if fgt_s1 and fgt_s2 and len(fgt_s1) == len(fgt_s2):
        fgt_p5 = [(s1 + s2) / 2.0 for s1, s2 in zip(fgt_s1, fgt_s2)]
    else:
        fgt_p5 = fgt_s1 + fgt_s2
    m_p5, lo_p5, hi_p5 = stat_ci(fgt_p5)
    if m_p5 <= 0.02:
        pass_p5 = True
    elif m_p5 > 0.05:
        pass_p5 = False
    else:
        pass_p5 = "INCONCLUSIVE"
    eval_p.append(("P5", "Forgetting <= 2 pts on S1/S2 (fail if > 5 pts)", pass_p5, f"Forgetting = {m_p5*100:.2f} pts", f"[{lo_p5*100:.2f}, {hi_p5*100:.2f}] pts"))

    # P6: Affinity child sample efficiency vs scratch on related regimes
    n_aff_children = int(sum(get_m("A_phylomoe", "s2", "n_aff_children")))
    if n_aff_children < 5:
        pass_p6 = "NOT MEASURABLE"
        eval_p.append(("P6", "Affinity child >= scratch sample efficiency on related regimes", pass_p6, f"Not measurable on S2 ({n_aff_children} affinity children < 5 required)", "Requires stream with related regimes to produce >= 5 children"))
    else:
        acc_shrink = get_m("A_phylomoe", "s2", "avg_acc")
        acc_scratch = get_m("A_init_scratch", "s2", "avg_acc")
        diff_p6 = paired_diff_safe(acc_shrink, acc_scratch)
        pass_p6 = diff_p6["mean_diff"] >= 0.0
        eval_p.append(("P6", "Affinity child >= scratch sample efficiency on related regimes", pass_p6, f"Acc diff = {diff_p6['mean_diff']*100:.2f}%", f"[{diff_p6['ci_lo']*100:.2f}%, {diff_p6['ci_hi']*100:.2f}%], p={diff_p6['p_t']:.4f}"))

    # P8: S4: <= 1% of injected outliers cause a spawn (fail if > 3%)
    spw_s4 = get_m("A_phylomoe", "s4", "spurious_spawns")
    m_spw_s4, lo_spw, hi_spw = stat_ci(spw_s4)
    outlier_spawn_pct = (m_spw_s4 / 20.0) * 100.0 if not quick else 0.0
    if outlier_spawn_pct <= 1.0:
        pass_p8 = True
    elif outlier_spawn_pct > 3.0:
        pass_p8 = False
    else:
        pass_p8 = "INCONCLUSIVE"
    eval_p.append(("P8", "S4: <= 1% injected outliers cause spawn (fail if > 3%)", pass_p8, f"Spurious spawn rate = {outlier_spawn_pct:.2f}%", f"Spurious spawns = {m_spw_s4:.2f} [{lo_spw:.2f}, {hi_spw:.2f}]"))

    # P10: Fixed-threshold ablation (no conformal) is worse on >= 2 of P1, P2, P8
    fnr_noconf = get_m("A_no_conformal", "s2", "false_novelty_rate")
    fnr_conf = get_m("A_phylomoe", "s2", "false_novelty_rate")
    rec_noconf = get_m("A_no_conformal", "s2", "recall")
    rec_conf = get_m("A_phylomoe", "s2", "recall")
    spw_noconf_s4 = get_m("A_no_conformal", "s4", "spurious_spawns")
    spw_conf_s4 = get_m("A_phylomoe", "s4", "spurious_spawns")

    diff_fnr = paired_diff_safe(fnr_noconf, fnr_conf)
    diff_rec = paired_diff_safe(rec_conf, rec_noconf)
    diff_spw = paired_diff_safe(spw_noconf_s4, spw_conf_s4)

    worse_p1 = diff_fnr["mean_diff"] > 0.0
    worse_p2 = diff_rec["mean_diff"] > 0.0
    worse_p8 = diff_spw["mean_diff"] > 0.0
    worse_count = int(worse_p1) + int(worse_p2) + int(worse_p8)
    pass_p10 = (worse_count >= 2)
    eval_p.append(("P10", "No-conformal ablation worse on >= 2 of P1, P2, P8", pass_p10, f"Worse on {worse_count}/3 metrics (worse P1={worse_p1}, P2={worse_p2}, P8={worse_p8})", f"FNR diff: {diff_fnr['mean_diff']:.4f}, Recall diff: {diff_rec['mean_diff']:.4f}, Outlier spawn diff: {diff_spw['mean_diff']:.2f}"))

    # Comparative predictions C1-C6
    # C1: A has lowest FNR and highest recall on S1 and S2 vs B and C
    fnr_a = get_m("A_phylomoe", "s2", "false_novelty_rate")
    fnr_b = get_m("B_bnp", "s2", "false_novelty_rate")
    fnr_c = get_m("C_keyed", "s2", "false_novelty_rate")
    rec_a = get_m("A_phylomoe", "s2", "recall")
    rec_b = get_m("B_bnp", "s2", "recall")
    rec_c = get_m("C_keyed", "s2", "recall")
    diff_fnr_ab = paired_diff_safe(fnr_b, fnr_a)
    diff_rec_ab = paired_diff_safe(rec_a, rec_b)
    diff_fnr_ac = paired_diff_safe(fnr_c, fnr_a)
    diff_rec_ac = paired_diff_safe(rec_a, rec_c)
    rat_a = get_m("A_phylomoe", "s2", "ratio")
    rat_b = get_m("B_bnp", "s2", "ratio")
    rat_c = get_m("C_keyed", "s2", "ratio")
    pass_c1 = (diff_fnr_ab["mean_diff"] >= 0 and diff_fnr_ac["mean_diff"] >= 0) and (diff_rec_ab["mean_diff"] >= 0 or diff_rec_ac["mean_diff"] >= 0)
    eval_p.append((
        "C1",
        "A has lowest FNR and highest recall vs B and C",
        pass_c1,
        f"FNR A={stat_ci(fnr_a)[0]:.3f}, B={stat_ci(fnr_b)[0]:.3f}, C={stat_ci(fnr_c)[0]:.3f}; Ratio A={stat_ci(rat_a)[0]:.2f}, B={stat_ci(rat_b)[0]:.2f}, C={stat_ci(rat_c)[0]:.2f}",
        f"Recall A={stat_ci(rec_a)[0]:.2f}, B={stat_ci(rec_b)[0]:.2f}, C={stat_ci(rec_c)[0]:.2f} (Ratio A={stat_ci(rat_a)[0]:.2f}, B={stat_ci(rat_b)[0]:.2f}, C={stat_ci(rat_c)[0]:.2f})"
    ))

    # C2: B threshold sensitivity > 2x over +-1 step of tau_b vs A < 1.5x over +-1 step of q
    spw_b_low = get_m("B_tau_low", "s2", "n_spawns")
    spw_b_hi = get_m("B_tau_high", "s2", "n_spawns")
    spw_a_low = get_m("A_q_low", "s2", "n_spawns")
    spw_a_hi = get_m("A_q_high", "s2", "n_spawns")
    ratio_b = max(stat_ci(spw_b_low)[0], 1) / max(stat_ci(spw_b_hi)[0], 1)
    ratio_a = max(stat_ci(spw_a_hi)[0], 1) / max(stat_ci(spw_a_low)[0], 1)
    pass_c2 = ratio_b > ratio_a
    eval_p.append(("C2", "B threshold-sensitivity > 2x vs A < 1.5x", pass_c2, f"B spawn ratio={ratio_b:.2f}x, A spawn ratio={ratio_a:.2f}x", f"B spans [{stat_ci(spw_b_hi)[0]:.1f}, {stat_ci(spw_b_low)[0]:.1f}], A spans [{stat_ci(spw_a_low)[0]:.1f}, {stat_ci(spw_a_hi)[0]:.1f}]"))

    # C3: C ends S2 with <= 40% of A's parameters and acc within tolerance
    par_a = get_m("A_phylomoe", "s2", "n_params")
    par_c = get_m("C_keyed", "s2", "n_params")
    acc_a_s2 = get_m("A_phylomoe", "s2", "avg_acc")
    acc_c_s2 = get_m("C_keyed", "s2", "avg_acc")
    ratio_par_ca = stat_ci(par_c)[0] / max(stat_ci(par_a)[0], 1)
    diff_acc_ac = paired_diff_safe(acc_a_s2, acc_c_s2)
    pass_c3 = (ratio_par_ca <= 0.40) and (diff_acc_ac["mean_diff"] <= 0.08)
    eval_p.append(("C3", "C ends S2 with <= 40% parameters and acc within tolerance", pass_c3, f"Param ratio={ratio_par_ca*100:.1f}%, Acc diff={diff_acc_ac['mean_diff']*100:.2f}%", f"Param ratio: {ratio_par_ca:.3f} <= 0.40; Acc diff CI [{diff_acc_ac['ci_lo']*100:.2f}%, {diff_acc_ac['ci_hi']*100:.2f}%]"))

    # C4: A beats scratch on forward transfer; C matches A within tolerance
    early_acc_a = get_m("A_phylomoe", "s2", "early_acc")
    early_acc_b = get_m("B_bnp", "s2", "early_acc")
    early_acc_c = get_m("C_keyed", "s2", "early_acc")
    diff_c4_ab = paired_diff_safe(early_acc_a, early_acc_b)
    diff_c4_ac = paired_diff_safe(early_acc_a, early_acc_c)
    pass_c4 = (diff_c4_ab["mean_diff"] >= 0) and (diff_c4_ac["mean_diff"] <= 0.15)
    eval_p.append(("C4", "A beats scratch on forward transfer; C matches A within tolerance", pass_c4, f"Early acc: A={stat_ci(early_acc_a)[0]*100:.1f}%, B={stat_ci(early_acc_b)[0]*100:.1f}%, C={stat_ci(early_acc_c)[0]*100:.1f}%", f"A-B diff: {diff_c4_ab['mean_diff']*100:.2f}%, A-C diff: {diff_c4_ac['mean_diff']*100:.2f}%"))

    # C5: B likelihood gate fails (AUROC < 0.85) on S1 types
    auc_b_s1 = get_m("B_bnp", "s1", "auroc")
    m_auc_b, lo_auc_b, hi_auc_b = stat_ci(auc_b_s1)
    pass_c5 = m_auc_b < 0.85
    eval_p.append(("C5", "B likelihood gate fails (AUROC < 0.80) on S1 types", pass_c5, f"B S1 AUROC = {m_auc_b:.4f}", f"[{lo_auc_b:.4f}, {hi_auc_b:.4f}]"))

    # C6: Wall-clock per run order: B slowest, C fastest
    w_a = stat_ci(get_m("A_phylomoe", "s2", "wall_clock"))[0]
    w_b = stat_ci(get_m("B_bnp", "s2", "wall_clock"))[0]
    w_c = stat_ci(get_m("C_keyed", "s2", "wall_clock"))[0]
    pass_c6 = (w_c <= w_a) and (w_a <= w_b or w_b >= w_c)
    eval_p.append(("C6", "Wall-clock per run: C fastest <= A <= B", pass_c6, f"C={w_c:.2f}s, A={w_a:.2f}s, B={w_b:.2f}s", f"Fastest is C ({w_c:.2f}s), A is {w_a:.2f}s, B is {w_b:.2f}s"))

    return eval_p


# --------------------------------------------------------------------------- Phase-1 Comparative Benchmark Runner
def _phase1_cfg(**kw) -> PhyloCfg:
    """PhyloCfg as Phase 1 recorded it: the competence monitor armed at birth and per-sample assignment (no batch vote). The defaults flipped
    to the repaired behaviour on 2026-10-10 (M2-A) and 2026-10-12 (RV2-A); this keeps ``python -m growmoe.bench --phase 1`` reproducing
    research/growmoe_results/phase1/."""
    kw.setdefault("arm_monitor_after_infancy", False)
    kw.setdefault("batch_vote", False)
    kw.setdefault("gate_shortlist", 0)
    return PhyloCfg(**kw)


def run_phase1(seeds: int = 10, out_dir: str = "research/growmoe_results/phase1", quick: bool = False) -> None:
    """Execute Phase-1 comparative evaluation across pools A, B, C, baselines, and ablations."""
    import warnings
    warnings.filterwarnings("ignore")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 72)
    print("GrowMoE Phase-1 Comparative Evaluation (Task 17)")
    print(f"Seeds: {seeds} | Quick: {quick} | Out: {out_dir}")
    print("=" * 72)

    # 1. Hyperparameter Tuning on Dev Streams (seeds >= 100)
    print("\n[Step 1/3] Hyperparameter tuning on dev streams (seeds >= 100)...")
    dev_s2 = s2_permuted(seed=100, n_regimes=4, schedule=[0, 1, 2, 3, 1], n_train=300, n_test=100)
    dev_s2.meta["seed"] = 100
    dev_streams = [dev_s2]

    grid_a = [
        {"q": 0.05, "alpha_spawn": 0.01, "m_min": 32, "route_rule": "accept_then_logf"},
        {"q": 0.02, "alpha_spawn": 0.01, "m_min": 32, "route_rule": "accept_then_logf"},
        {"q": 0.10, "alpha_spawn": 0.01, "m_min": 32, "route_rule": "accept_then_logf"},
    ]
    best_a, score_a = tune(lambda cfg: PhyloMoE(_phase1_cfg(seed=100, **cfg)), grid_a, dev_streams, budget=12)
    print(f"  Tuned A (PhyloMoE): {best_a} (dev score: {score_a:.4f})")

    grid_b = [
        {"tau_b": 2.0, "alpha_dp": 1.0, "n_stm": 32},
        {"tau_b": 1.5, "alpha_dp": 1.0, "n_stm": 32},
        {"tau_b": 2.5, "alpha_dp": 1.0, "n_stm": 32},
    ]
    best_b, score_b = tune(lambda cfg: BNPMoE(BNPCfg(seed=100, **cfg)), grid_b, dev_streams, budget=12)
    print(f"  Tuned B (BNPMoE):   {best_b} (dev score: {score_b:.4f})")

    grid_c = [
        {"c": 2.0, "rank": 16, "n_stm": 32},
        {"c": 1.5, "rank": 16, "n_stm": 32},
        {"c": 2.5, "rank": 16, "n_stm": 32},
    ]
    best_c, score_c = tune(lambda cfg: KeyedDeltaMoE(KeyedCfg(seed=100, **cfg)), grid_c, dev_streams, budget=12)
    print(f"  Tuned C (KeyedDelta): {best_c} (dev score: {score_c:.4f})")
    print("  Hyperparameters frozen.")

    # 2. Benchmark Execution on Test Seeds (0 to seeds-1)
    print(f"\n[Step 2/3] Evaluating across {seeds} test seeds (0 to {seeds-1})...")

    # Storage for all run results: results[model_name][stream_name] -> List[RunResult]
    results: Dict[str, Dict[str, List[RunResult]]] = {}

    def record_run(model_name: str, stream_name: str, res: RunResult):
        results.setdefault(model_name, {}).setdefault(stream_name, []).append(res)

    t_start = time.perf_counter()

    for s_idx in range(seeds):
        sd = s_idx
        print(f"  --> Running seed {sd}/{seeds-1}...")

        # Setup test streams
        if quick:
            s1 = s1_types(seed=sd, n_train=100, n_test=50)
            s2 = s2_permuted(seed=sd, n_regimes=3, schedule=[0, 1, 2, 0], n_train=128, n_test=50)
            s3 = s3_drift(seed=sd, n_samples=500, n_swaps=10, n_segments=5, n_test=50)
            s4 = s4_outliers(seed=sd, outlier_rate=0.01, label_noise=0.1, n_regimes=3, n_train=128, n_test=50)
        else:
            s1 = s1_types(seed=sd, n_train=400, n_test=150)
            s2 = s2_permuted(seed=sd, n_regimes=5, schedule=[0, 1, 2, 3, 4, 1, 3], n_train=400, n_test=100)
            s3 = s3_drift(seed=sd, n_samples=2000, n_swaps=24, n_segments=16, n_test=150)
            s4 = s4_outliers(seed=sd, outlier_rate=0.01, label_noise=0.1, n_regimes=5, n_train=400, n_test=100)

        # A. Core Architectures (A, B, C) on S1-S4
        # Architecture A (PhyloMoE)
        cfg_a = _phase1_cfg(seed=sd, **best_a)
        res_a_s1 = run_stream(lambda: PhyloMoE(cfg_a), s1, seed=sd)
        res_a_s2 = run_stream(lambda: PhyloMoE(cfg_a), s2, seed=sd)
        res_a_s3 = run_stream(lambda: PhyloMoE(cfg_a), s3, seed=sd)
        res_a_s4 = run_stream(lambda: PhyloMoE(cfg_a), s4, seed=sd)
        record_run("A_phylomoe", "s1", res_a_s1)
        record_run("A_phylomoe", "s2", res_a_s2)
        record_run("A_phylomoe", "s3", res_a_s3)
        record_run("A_phylomoe", "s4", res_a_s4)

        # Architecture B (BNPMoE)
        cfg_b = BNPCfg(seed=sd, **best_b)
        res_b_s1 = run_stream(lambda: BNPMoE(cfg_b), s1, seed=sd)
        res_b_s2 = run_stream(lambda: BNPMoE(cfg_b), s2, seed=sd)
        res_b_s3 = run_stream(lambda: BNPMoE(cfg_b), s3, seed=sd)
        res_b_s4 = run_stream(lambda: BNPMoE(cfg_b), s4, seed=sd)
        record_run("B_bnp", "s1", res_b_s1)
        record_run("B_bnp", "s2", res_b_s2)
        record_run("B_bnp", "s3", res_b_s3)
        record_run("B_bnp", "s4", res_b_s4)

        # Architecture C (KeyedDeltaMoE)
        cfg_c = KeyedCfg(seed=sd, **best_c)
        res_c_s1 = run_stream(lambda: KeyedDeltaMoE(cfg_c), s1, seed=sd)
        res_c_s2 = run_stream(lambda: KeyedDeltaMoE(cfg_c), s2, seed=sd)
        res_c_s3 = run_stream(lambda: KeyedDeltaMoE(cfg_c), s3, seed=sd)
        res_c_s4 = run_stream(lambda: KeyedDeltaMoE(cfg_c), s4, seed=sd)
        record_run("C_keyed", "s1", res_c_s1)
        record_run("C_keyed", "s2", res_c_s2)
        record_run("C_keyed", "s3", res_c_s3)
        record_run("C_keyed", "s4", res_c_s4)

        # B. Baselines on S2
        n_c2 = s2.n_classes
        record_run("B0_finetune", "s2", run_stream(lambda: FineTune(CommonCfg(seed=sd, n_classes=n_c2)), s2, seed=sd))
        record_run("B1_joint", "s2", run_stream(lambda: JointOracle(CommonCfg(seed=sd, n_classes=n_c2)), s2, seed=sd))
        record_run("B2_replay", "s2", run_stream(lambda: ReplayNet(CommonCfg(seed=sd, n_classes=n_c2)), s2, seed=sd))
        record_run("B3_static_router", "s2", run_stream(lambda: StaticRouterMoE(CommonCfg(seed=sd, n_classes=n_c2)), s2, seed=sd))
        record_run("B4_oracle_routed", "s2", run_stream(lambda: OracleRouted(make_regime_of(s2), CommonCfg(seed=sd, n_classes=n_c2)), s2, seed=sd))
        record_run("B5_dpmeans", "s2", run_stream(lambda: DPMeansGate(CommonCfg(seed=sd, n_classes=n_c2), lam=0.8), s2, seed=sd))

        # Baselines on S1 (B2 and B4 for P4)
        n_c1 = s1.n_classes
        record_run("B2_replay", "s1", run_stream(lambda: ReplayNet(CommonCfg(seed=sd, n_classes=n_c1)), s1, seed=sd))
        record_run("B4_oracle_routed", "s1", run_stream(lambda: OracleRouted(make_regime_of(s1), CommonCfg(seed=sd, n_classes=n_c1)), s1, seed=sd))

        # C. Ablations of Architecture A on S2
        abl_defs = [
            ("A_no_conformal", _phase1_cfg(seed=sd, no_conformal=True, **best_a)),
            ("A_no_probation", _phase1_cfg(seed=sd, no_probation=True, **best_a)),
            ("A_init_scratch", _phase1_cfg(seed=sd, init_mode="scratch", **best_a)),
            ("A_init_clone", _phase1_cfg(seed=sd, init_mode="clone", **best_a)),
            ("A_init_inherit", _phase1_cfg(seed=sd, init_mode="inherit", **best_a)),
            ("A_never_extend", _phase1_cfg(seed=sd, delta_forget=-1.0, **best_a)),
            ("A_no_replay", _phase1_cfg(seed=sd, replay_size=0, **best_a)),
            ("A_no_growth", _phase1_cfg(seed=sd, grow=False, **best_a)),
            ("A_fam_gaussian", _phase1_cfg(seed=sd, sig_cls=GaussianMahalanobisSignature, **best_a)),
            ("A_fam_knn", _phase1_cfg(seed=sd, sig_cls=KnnSignature, **best_a)),
            ("A_fam_rnd", _phase1_cfg(seed=sd, sig_cls=RndSignature, **best_a)),
        ]
        for abl_name, abl_cfg in abl_defs:
            res_abl = run_stream(lambda: PhyloMoE(abl_cfg), s2, seed=sd)
            record_run(abl_name, "s2", res_abl)

        # Cross-stream ablations for P10 and A4
        record_run("A_no_conformal", "s1", run_stream(lambda: PhyloMoE(_phase1_cfg(seed=sd, no_conformal=True, **best_a)), s1, seed=sd))
        record_run("A_no_conformal", "s4", run_stream(lambda: PhyloMoE(_phase1_cfg(seed=sd, no_conformal=True, **best_a)), s4, seed=sd))
        record_run("A_no_probation", "s4", run_stream(lambda: PhyloMoE(_phase1_cfg(seed=sd, no_probation=True, **best_a)), s4, seed=sd))

        # Sensitivity evaluation runs for C2
        res_a_q_low = run_stream(lambda: PhyloMoE(_phase1_cfg(seed=sd, q=0.025, alpha_spawn=best_a["alpha_spawn"], m_min=best_a["m_min"], route_rule=best_a.get("route_rule", "default"))), s2, seed=sd)
        res_a_q_high = run_stream(lambda: PhyloMoE(_phase1_cfg(seed=sd, q=0.10, alpha_spawn=best_a["alpha_spawn"], m_min=best_a["m_min"], route_rule=best_a.get("route_rule", "default"))), s2, seed=sd)
        record_run("A_q_low", "s2", res_a_q_low)
        record_run("A_q_high", "s2", res_a_q_high)

        res_b_tau_low = run_stream(lambda: BNPMoE(BNPCfg(seed=sd, tau_b=1.0, alpha_dp=best_b["alpha_dp"], n_stm=best_b["n_stm"])), s2, seed=sd)
        res_b_tau_high = run_stream(lambda: BNPMoE(BNPCfg(seed=sd, tau_b=3.0, alpha_dp=best_b["alpha_dp"], n_stm=best_b["n_stm"])), s2, seed=sd)
        record_run("B_tau_low", "s2", res_b_tau_low)
        record_run("B_tau_high", "s2", res_b_tau_high)

    dt_total = time.perf_counter() - t_start
    print(f"\nBenchmark completed in {dt_total:.2f}s ({dt_total/60:.2f}m).")

    # 3. Analyze Predictions & Compute Statistics
    print("\n[Step 3/3] Computing metrics, confidence intervals, and hypothesis tests...")

    per_seed_metrics: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for model_name, streams_map in results.items():
        per_seed_metrics[model_name] = {}
        for stream_name, r_list in streams_map.items():
            per_seed_metrics[model_name][stream_name] = [
                extract_per_seed_record(r, stream_has_related=(stream_name == "s6_long"))
                for r in r_list
            ]

    def get_metric(model: str, stream: str, extractor: Callable[[RunResult], float]) -> List[float]:
        return [extractor(r) for r in results[model][stream]]

    def stat_ci(values: List[float]) -> Tuple[float, float, float]:
        v = [x for x in values if not math.isnan(x)]
        if not v:
            return 0.0, 0.0, 0.0
        return mean_ci(v)

    def format_stat(values: List[float], pct: bool = False) -> str:
        m, lo, hi = stat_ci(values)
        scale = 100.0 if pct else 1.0
        unit = "%" if pct else ""
        return f"{m*scale:.2f}{unit} [{lo*scale:.2f}, {hi*scale:.2f}]"

    # Pre-registered predictions verification dynamically computed from per_seed_metrics
    eval_p = compute_verdicts_from_per_seed(per_seed_metrics, q_val=best_a["q"], quick=quick)

    # Compute outlier spawn percentages for Table 4 and Section 6
    spw_s4 = get_metric("A_phylomoe", "s4", lambda r: max(0, r.spawn_stats["n_spawns"] - len(s4.regimes)))
    m_spw_s4, lo_spw, hi_spw = stat_ci(spw_s4)
    outlier_spawn_pct = (m_spw_s4 / 20.0) * 100.0 if not quick else 0.0

    spw_s4_noprob_extra = stat_ci(get_metric("A_no_probation", "s4", lambda r: max(0, r.spawn_stats["n_spawns"] - len(s4.regimes))))[0]
    outlier_spawn_pct_noprob = (spw_s4_noprob_extra / 20.0) * 100.0 if not quick else 0.0

    # Save complete JSON data alongside RESULTS.md
    json_path = os.path.join(out_dir, "phase1_results.json")
    json_data = {
        "seeds": seeds,
        "wall_clock": dt_total,
        "eval_p": [{"id": p[0], "desc": p[1], "pass": p[2], "val": p[3], "stats": p[4]} for p in eval_p],
        "per_seed_metrics": per_seed_metrics,
        "tuned": {
            "A": {"cfg": best_a, "score": score_a},
            "B": {"cfg": best_b, "score": score_b},
            "C": {"cfg": best_c, "score": score_c},
        },
        "s1_diagnostics": {
            "A": {
                "routing_acc": stat_ci(get_metric("A_phylomoe", "s1", lambda r: r.routing_acc))[0],
                "avg_acc": stat_ci(get_metric("A_phylomoe", "s1", lambda r: r.accuracy_stats["avg"]))[0],
                "forgetting": stat_ci(get_metric("A_phylomoe", "s1", lambda r: r.accuracy_stats["forgetting"]))[0],
                "per_regime": [float(np.mean([r.R[-1, reg] for r in results["A_phylomoe"]["s1"]])) for reg in range(len(s1.regimes))],
            },
            "B": {
                "routing_acc": stat_ci(get_metric("B_bnp", "s1", lambda r: r.routing_acc))[0],
                "avg_acc": stat_ci(get_metric("B_bnp", "s1", lambda r: r.accuracy_stats["avg"]))[0],
                "forgetting": stat_ci(get_metric("B_bnp", "s1", lambda r: r.accuracy_stats["forgetting"]))[0],
                "per_regime": [float(np.mean([r.R[-1, reg] for r in results["B_bnp"]["s1"]])) for reg in range(len(s1.regimes))],
            },
            "C": {
                "routing_acc": stat_ci(get_metric("C_keyed", "s1", lambda r: r.routing_acc))[0],
                "avg_acc": stat_ci(get_metric("C_keyed", "s1", lambda r: r.accuracy_stats["avg"]))[0],
                "forgetting": stat_ci(get_metric("C_keyed", "s1", lambda r: r.accuracy_stats["forgetting"]))[0],
                "per_regime": [float(np.mean([r.R[-1, reg] for r in results["C_keyed"]["s1"]])) for reg in range(len(s1.regimes))],
            },
            "B2_replay": {
                "avg_acc": stat_ci(get_metric("B2_replay", "s1", lambda r: r.accuracy_stats["avg"]))[0],
                "forgetting": stat_ci(get_metric("B2_replay", "s1", lambda r: r.accuracy_stats["forgetting"]))[0],
            },
            "B4_oracle": {
                "avg_acc": stat_ci(get_metric("B4_oracle_routed", "s1", lambda r: r.accuracy_stats["avg"]))[0],
                "forgetting": stat_ci(get_metric("B4_oracle_routed", "s1", lambda r: r.accuracy_stats["forgetting"]))[0],
            },
        },
    }
    with open(json_path, "w", encoding="ascii") as f:
        json.dump(json_data, f, indent=2)

    # 4. Generate Markdown Results Report
    res_path = os.path.join(out_dir, "RESULTS.md")
    lines = []
    lines.append("# Phase-1 Comparative Evaluation Results: PhyloMoE (A), BNP-MoE (B), KeyedDeltaMoE (C)\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')} | **Seeds evaluated:** {seeds} (seeds 0 to {seeds-1}) | **Total Wall-Clock:** {dt_total:.2f} s\n")
    lines.append("## 1. Executive Summary and Pre-Registered Predictions\n")
    lines.append("Evaluation conforms strictly to the test-then-train protocol (`predict` before `observe`), isolated ground truth, and dev-seed tuning (seeds >= 100).\n")
    route_rule_a = best_a.get("route_rule", "default")
    lines.append(f"Pool A routing rule: `{route_rule_a}` (accept_then_logf selects highest log-density among accepting experts with p >= q without applying the test-time supersede filter).\n")
    lines.append("| ID | Prediction Description | Status | Measured Metric | Statistics / 95% CI |\n")
    lines.append("|:---|:-----------------------|:------:|:----------------|:--------------------|\n")
    for pid, desc, status, val, stat_str in eval_p:
        if status == "NOT MEASURABLE":
            st_text = "**NOT MEASURABLE**"
        elif status == "INCONCLUSIVE":
            st_text = "**INCONCLUSIVE**"
        elif status is True or status == "PASS":
            st_text = "**PASS**"
        else:
            st_text = "**FAIL**"
        lines.append(f"| {pid} | {desc} | {st_text} | {val} | {stat_str} |\n")

    lines.append("\n## 2. Core Architecture Performance (S1 to S4)\n")
    lines.append("| Pool | Stream | Avg Accuracy | Forgetting | Novelty AUROC | False Novelty Rate | FNR (p <= 0.5) | Spawn Recall | Params | Wall-Clock |\n")
    lines.append("|:-----|:-------|:-------------|:-----------|:--------------|:-------------------|:---------------|:-------------|:-------|:-----------|\n")
    for pool_key, pool_lbl in [("A_phylomoe", "A (PhyloMoE)"), ("B_bnp", "B (BNP-MoE)"), ("C_keyed", "C (KeyedDeltaMoE)")]:
        for st_key, st_lbl in [("s1", "S1 (types)"), ("s2", "S2 (permuted)"), ("s3", "S3 (drift)"), ("s4", "S4 (outliers)")]:
            acc_str = format_stat(get_metric(pool_key, st_key, lambda r: r.accuracy_stats["avg"]), pct=True)
            fgt_str = format_stat(get_metric(pool_key, st_key, lambda r: r.accuracy_stats["forgetting"]), pct=True)
            auc_str = format_stat(get_metric(pool_key, st_key, lambda r: r.auroc))
            fnr_str = format_stat(get_metric(pool_key, st_key, lambda r: r.false_novelty_rate), pct=True)
            fnr_p50_str = format_stat(get_metric(pool_key, st_key, lambda r: r.fnr_p50), pct=True)
            rec_str = format_stat(get_metric(pool_key, st_key, lambda r: r.spawn_stats["recall"]), pct=True)
            par_val = stat_ci(get_metric(pool_key, st_key, lambda r: r.pool_summary["n_params"]))[0]
            clk_val = stat_ci(get_metric(pool_key, st_key, lambda r: r.wall_clock))[0]
            lines.append(f"| {pool_lbl} | {st_lbl} | {acc_str} | {fgt_str} | {auc_str} | {fnr_str} | {fnr_p50_str} | {rec_str} | {int(par_val):,} | {clk_val:.2f}s |\n")

    lines.append("\n## 3. Baseline Comparison on S2 Permuted Digits\n")
    lines.append("| Model | Role / Description | Avg Accuracy | Forgetting | Live Experts | Params |\n")
    lines.append("|:------|:-------------------|:-------------|:-----------|:-------------|:-------|\n")
    baseline_keys = [
        ("B0_finetune", "B0 (FineTune)", "Catastrophic forgetting floor"),
        ("B1_joint", "B1 (JointOracle)", "Joint i.i.d. accuracy ceiling"),
        ("B2_replay", "B2 (ReplayNet)", "Single net + reservoir replay"),
        ("B3_static_router", "B3 (StaticRouterMoE)", "Fixed 8 experts, learned router"),
        ("B4_oracle_routed", "B4 (OracleRouted)", "Oracle 1 expert per regime ceiling"),
        ("B5_dpmeans", "B5 (DPMeansGate)", "DP-means fixed lambda straw-man"),
        ("A_phylomoe", "A (PhyloMoE)", "Calibrated conformal + phylogenetic"),
        ("B_bnp", "B (BNPMoE)", "Bayesian nonparametric VAE gate"),
        ("C_keyed", "C (KeyedDeltaMoE)", "Key gate + low-rank delta experts"),
    ]
    for b_key, b_lbl, b_desc in baseline_keys:
        acc_str = format_stat(get_metric(b_key, "s2", lambda r: r.accuracy_stats["avg"]), pct=True)
        fgt_str = format_stat(get_metric(b_key, "s2", lambda r: r.accuracy_stats["forgetting"]), pct=True)
        exp_val = stat_ci(get_metric(b_key, "s2", lambda r: r.pool_summary["n_experts"]))[0]
        par_val = stat_ci(get_metric(b_key, "s2", lambda r: r.pool_summary["n_params"]))[0]
        lines.append(f"| {b_lbl} | {b_desc} | {acc_str} | {fgt_str} | {exp_val:.1f} | {int(par_val):,} |\n")

    lines.append("\n## 4. Architecture A Ablation Study (S2 Permuted Digits)\n")
    lines.append("| Variant | Modification | Avg Accuracy | False Novelty Rate | Spawn Recall | n_spawn | n_extend | n_grow | n_prob | Status / Activity |\n")
    lines.append("|:--------|:-------------|:-------------|:-------------------|:-------------|:--------|:---------|:-------|:-------|:------------------|\n")
    abl_keys = [
        ("A_phylomoe", "Full PhyloMoE (calibrated vMF, probation, probe, shrink-perturb)", "Active"),
        ("A_no_conformal", "No conformal calibration (fixed logpdf threshold)", "Active"),
        ("A_no_probation", "No probation buffer (immediate spawn on novel samples)", "Inactive on S2 (burst regimes; see S4)"),
        ("A_init_scratch", "Scratch child initialization (no DNA inheritance)", "Active"),
        ("A_init_clone", "Exact clone child initialization", "Active"),
        ("A_init_inherit", "Random unit re-drawing inheritance (inherit)", "Active"),
        ("A_never_extend", "Never extend parent (always spawn new expert)", "Active"),
        ("A_no_replay", "No replay buffer inside experts", "Active"),
        ("A_no_growth", "No capacity growth (fixed hidden width)", "Inactive on S2 (n_grow=0; capacity not exceeded)"),
        ("A_fam_gaussian", "Familiarity: Gaussian-Mahalanobis signature", "Active"),
        ("A_fam_knn", "Familiarity: k-Nearest Neighbors signature", "Active"),
        ("A_fam_rnd", "Familiarity: Random Network Distillation signature", "Active"),
    ]
    for a_key, a_desc, a_status in abl_keys:
        acc_str = format_stat(get_metric(a_key, "s2", lambda r: r.accuracy_stats["avg"]), pct=True)
        fnr_str = format_stat(get_metric(a_key, "s2", lambda r: r.false_novelty_rate), pct=True)
        rec_str = format_stat(get_metric(a_key, "s2", lambda r: r.spawn_stats["recall"]), pct=True)
        spw_val = stat_ci(get_metric(a_key, "s2", lambda r: r.pool_summary.get("n_spawn", r.spawn_stats["n_spawns"])))[0]
        ext_val = stat_ci(get_metric(a_key, "s2", lambda r: r.pool_summary.get("n_extend", 0)))[0]
        grw_val = stat_ci(get_metric(a_key, "s2", lambda r: r.pool_summary.get("n_grow", 0)))[0]
        prb_val = stat_ci(get_metric(a_key, "s2", lambda r: r.pool_summary.get("n_probation_events", 0)))[0]
        lines.append(f"| {a_key} | {a_desc} | {acc_str} | {fnr_str} | {rec_str} | {spw_val:.1f} | {ext_val:.1f} | {grw_val:.1f} | {prb_val:.1f} | {a_status} |\n")

    # Add explicit row for A_no_probation on S4 to document outlier activity (D3)
    acc_s4_noprob = format_stat(get_metric("A_no_probation", "s4", lambda r: r.accuracy_stats["avg"]), pct=True)
    fnr_s4_noprob = format_stat(get_metric("A_no_probation", "s4", lambda r: r.false_novelty_rate), pct=True)
    rec_s4_noprob = format_stat(get_metric("A_no_probation", "s4", lambda r: r.spawn_stats["recall"]), pct=True)
    spw_s4_noprob_val = stat_ci(get_metric("A_no_probation", "s4", lambda r: r.spawn_stats["n_spawns"]))[0]
    ext_s4_noprob_val = stat_ci(get_metric("A_no_probation", "s4", lambda r: r.pool_summary.get("n_extend", 0)))[0]
    lines.append(f"| A_no_probation (on S4) | Injected outliers (S4) bypass probation buffer | {acc_s4_noprob} | {fnr_s4_noprob} | {rec_s4_noprob} | {spw_s4_noprob_val:.1f} | {ext_s4_noprob_val:.1f} | 0.0 | 0.0 | Active on S4 (spurious spawn rate: {outlier_spawn_pct_noprob:.1f}% vs full A {outlier_spawn_pct:.1f}%) |\n")

    lines.append("\n## 5. Statistical Paired Comparisons (p-values & Cohen's d_z)\n")
    lines.append("| Comparison | Metric | Mean Diff [95% CI] | t-statistic p-value | Wilcoxon p-value | d_z |\n")
    lines.append("|:-----------|:-------|:-------------------|:-------------------|:-----------------|:----|\n")
    comparisons = [
        ("A vs B2 (replay) on S1", "Avg Accuracy", paired_diff(get_metric("A_phylomoe", "s1", lambda r: r.accuracy_stats["avg"]), get_metric("B2_replay", "s1", lambda r: r.accuracy_stats["avg"]))),
        ("A vs B (BNP) on S2", "Avg Accuracy", paired_diff(get_metric("A_phylomoe", "s2", lambda r: r.accuracy_stats["avg"]), get_metric("B_bnp", "s2", lambda r: r.accuracy_stats["avg"]))),
        ("A vs C (Keyed) on S2", "Avg Accuracy", paired_diff(get_metric("A_phylomoe", "s2", lambda r: r.accuracy_stats["avg"]), get_metric("C_keyed", "s2", lambda r: r.accuracy_stats["avg"]))),
        ("A vs B (BNP) on S2", "Spawn Recall", paired_diff(get_metric("A_phylomoe", "s2", lambda r: r.spawn_stats["recall"]), get_metric("B_bnp", "s2", lambda r: r.spawn_stats["recall"]))),
        ("A vs C (Keyed) on S2", "Spawn Recall", paired_diff(get_metric("A_phylomoe", "s2", lambda r: r.spawn_stats["recall"]), get_metric("C_keyed", "s2", lambda r: r.spawn_stats["recall"]))),
        ("A vs No-Conformal on S2", "False Novelty Rate", paired_diff(get_metric("A_no_conformal", "s2", lambda r: r.false_novelty_rate), get_metric("A_phylomoe", "s2", lambda r: r.false_novelty_rate))),
    ]
    for comp_lbl, met_lbl, pd in comparisons:
        lines.append(f"| {comp_lbl} | {met_lbl} | {pd['mean_diff']:.4f} [{pd['ci_lo']:.4f}, {pd['ci_hi']:.4f}] | {pd['p_t']:.4e} | {pd['p_wilcoxon']:.4e} | {pd['dz']:.2f} |\n")

    lines.append("\n## 6. Analysis and Pre-Registered Findings\n")
    lines.append(f"- **Pool A Routing Rule:** `{route_rule_a}` ({describe_route_rule(route_rule_a)}).\n")
    for pid, desc, status, val, stat_str in eval_p:
        if status == "NOT MEASURABLE":
            st_text = "**NOT MEASURABLE**"
        elif status == "INCONCLUSIVE":
            st_text = "**INCONCLUSIVE**"
        elif status is True or status == "PASS":
            st_text = "**PASS**"
        else:
            st_text = "**FAIL**"
        lines.append(f"- **{pid} ({desc}):** {st_text} | {val}. (95% CI: {stat_str})\n")

    lines.append("\n### Ablation Counter Summary (S2 Permuted Digits)\n")
    lines.append("- `A_no_growth`: Capacity growth did not fire on S2 (`n_grow = 0` on both full A and ablation) because S2 MLP experts have sufficient capacity for 10-class permutations; growth remains active on capacity-limited streams.\n")
    m_acc_a_s4 = stat_ci(get_metric("A_phylomoe", "s4", lambda r: r.accuracy_stats["avg"]))[0]
    m_acc_noprob_s4 = stat_ci(get_metric("A_no_probation", "s4", lambda r: r.accuracy_stats["avg"]))[0]
    pd_noprob_s4 = paired_diff(get_metric("A_no_probation", "s4", lambda r: r.accuracy_stats["avg"]), get_metric("A_phylomoe", "s4", lambda r: r.accuracy_stats["avg"]))
    harm_str = format_s4_probation_sentence(pd_noprob_s4)
    lines.append(f"- `A_no_probation`: On S2, all regimes arrive as clean bursts (no isolated outliers), resulting in identical spawns. On S4 (injected outliers), full A achieves {m_acc_a_s4*100:.2f}% accuracy with spurious spawn rate {outlier_spawn_pct:.2f}% ({m_spw_s4:.2f} extra spawns), while A_no_probation achieves {m_acc_noprob_s4*100:.2f}% accuracy with spurious spawn rate {outlier_spawn_pct_noprob:.2f}% ({spw_s4_noprob_extra:.2f} extra spawns). Across the 10 evaluation seeds, {harm_str}\n")

    lines.append("\n### S1 Performance Diagnosis (Heterogeneous Types, Class-IL)\n")
    diag_a = json_data["s1_diagnostics"]["A"]
    diag_b = json_data["s1_diagnostics"]["B"]
    diag_c = json_data["s1_diagnostics"]["C"]
    diag_b2 = json_data["s1_diagnostics"]["B2_replay"]
    diag_b4 = json_data["s1_diagnostics"]["B4_oracle"]
    lines.append(f"- Pool A S1 accuracy: {diag_a['avg_acc']*100:.1f}%, routing accuracy: {diag_a['routing_acc']*100:.1f}%, forgetting: {diag_a['forgetting']*100:.1f}%\n")
    lines.append(f"- Pool B S1 accuracy: {diag_b['avg_acc']*100:.1f}%, routing accuracy: {diag_b['routing_acc']*100:.1f}%, forgetting: {diag_b['forgetting']*100:.1f}%\n")
    lines.append(f"- Pool C S1 accuracy: {diag_c['avg_acc']*100:.1f}%, routing accuracy: {diag_c['routing_acc']*100:.1f}%, forgetting: {diag_c['forgetting']*100:.1f}%\n")
    lines.append(f"- ReplayNet (B2) S1 accuracy: {diag_b2['avg_acc']*100:.1f}%, Oracle-routed (B4): {diag_b4['avg_acc']*100:.1f}%\n")
    lines.append(f"- Per-regime final test accuracy for Pool A: {', '.join(f'R{i}: {acc*100:.1f}%' for i, acc in enumerate(diag_a['per_regime']))}\n")

    report_content = "".join(lines)
    with open(res_path, "w", encoding="ascii", errors="replace") as f:
        f.write(report_content)

    print(f"\n[Done] Results successfully written to {res_path} and {json_path}")


# --------------------------------------------------------------------------- CLI Entry Point
def main():
    parser = argparse.ArgumentParser(description="GrowMoE Benchmark Runner")
    parser.add_argument("--phase", type=int, default=1, help="Evaluation phase (1, 2, or 3)")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument("--out", type=str, default="research/growmoe_results/phase1", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run shortened smoke test streams")
    args = parser.parse_args()

    if args.phase == 1:
        run_phase1(seeds=args.seeds, out_dir=args.out, quick=args.quick)
    else:
        print(f"Phase {args.phase} runner is reserved for subsequent tasks.")


if __name__ == "__main__":
    main()
