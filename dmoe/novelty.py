"""
Unified Multi-Scale Novelty Detection Algorithm (M02 + M13 + M24 + M26).

Detects novelty across any input stream by combining four calibrated channels:
1. Coarse Sequence Distribution Shift (seq_dist):
   Cosine distance between pooled sequence representation and expert centroid.
2. Local Token / Span Distribution Shift (tok_dist):
   Peak (top-k) token-level cosine distance against expert token prototypes.
   Prevents sequence mean-pooling from washing out 1-2 novel tokens in a long prompt.
3. Latent & Tail Surprise (surp_mean, surp_tail):
   Both sequence-mean and top-k peak token surprise from SurpriseDrivenLatentThinker
   (or normalized predictive entropy), catching localized contradictions/nuance.
4. Outcome / Prefix NLL Surprise (nll_mean, nll_tail):
   When next-token or outcome labels are available, separates harmless surface
   rewordings (low post-thinking error) from genuine semantic/rule nuance flips.

Each channel is standardized online (EMA mean & variance) per expert so thresholds
adapt automatically to each expert's own in-distribution variance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NoveltyVerdict:
    is_novel: bool
    novelty_score: float
    novelty_type: str  # "none" | "coarse_domain" | "near_twin_rule" | "local_nuance"
    channel_z: Dict[str, float] = field(default_factory=dict)
    novel_token_mask: Optional[torch.Tensor] = None


class ExpertRunningStats:
    """Lightweight online EMA mean & variance tracker for a single expert."""

    def __init__(self, ema_decay: float = 0.95):
        self.ema_decay = ema_decay
        self.count: int = 0
        self.key_counts: Dict[str, int] = {}
        self.means: Dict[str, float] = {}
        self.vars: Dict[str, float] = {}

    def update(self, metrics: Dict[str, float]) -> None:
        self.count += 1
        for k, v in metrics.items():
            kc = self.key_counts.get(k, 0) + 1
            self.key_counts[k] = kc
            alpha = 1.0 / float(kc) if kc < 20 else (1.0 - self.ema_decay)
            if k not in self.means:
                self.means[k] = v
                self.vars[k] = 0.01
            else:
                prev_m = self.means[k]
                new_m = (1.0 - alpha) * prev_m + alpha * v
                diff2 = (v - prev_m) * (v - new_m)
                self.means[k] = new_m
                self.vars[k] = max((1.0 - alpha) * self.vars[k] + alpha * diff2, 1e-4)

    def z_score(self, key: str, value: float, min_std: float = 0.04) -> float:
        if key not in self.means or self.key_counts.get(key, self.count) < 5:
            return 0.0
        std = max(self.vars[key] ** 0.5, min_std)
        return max(0.0, (value - self.means[key]) / std)


class UnifiedNoveltyDetector(nn.Module):
    """
    Multi-scale, calibrated novelty detector for coarse domain shifts,
    near-twin functional shifts, and localized single-token nuance.
    """

    def __init__(
        self,
        d_model: int,
        seq_dist_threshold: float = 0.45,
        tok_dist_threshold: float = 0.55,
        surp_threshold: float = 0.42,
        tail_fraction: float = 0.20,
        z_threshold: float = 2.75,
        max_token_protos: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_dist_threshold = float(seq_dist_threshold)
        self.tok_dist_threshold = float(tok_dist_threshold)
        self.surp_threshold = float(surp_threshold)
        self.tail_fraction = float(tail_fraction)
        self.z_threshold = float(z_threshold)
        self.max_token_protos = int(max_token_protos)

        self.expert_stats: List[ExpertRunningStats] = []
        self.token_protos: List[Optional[torch.Tensor]] = []

    def sync_num_experts(self, num_experts: int) -> None:
        while len(self.expert_stats) < num_experts:
            self.expert_stats.append(ExpertRunningStats())
            self.token_protos.append(None)
        if len(self.expert_stats) > num_experts:
            self.expert_stats = self.expert_stats[:num_experts]
            self.token_protos = self.token_protos[:num_experts]

    def prune_expert(self, idx: int) -> None:
        if 0 <= idx < len(self.expert_stats):
            self.expert_stats.pop(idx)
            self.token_protos.pop(idx)

    @torch.no_grad()
    def update_token_prototypes(self, expert_idx: int, token_latents: torch.Tensor) -> None:
        """
        Store a bounded rolling buffer of normalized token-level latents [N, d_model]
        for expert_idx to enable local sub-sequence / single-token novelty checks.
        """
        if expert_idx >= len(self.token_protos):
            return
        flat = F.normalize(token_latents.detach().reshape(-1, self.d_model).float(), p=2, dim=-1)
        if flat.size(0) == 0:
            return
        # Subsample to keep buffer bounded and diverse
        if flat.size(0) > 32:
            idx = torch.randperm(flat.size(0), device=flat.device)[:32]
            flat = flat[idx]
        cur = self.token_protos[expert_idx]
        if cur is None:
            self.token_protos[expert_idx] = flat[: self.max_token_protos].clone()
        else:
            comb = torch.cat([cur.to(flat.device), flat], dim=0)
            if comb.size(0) > self.max_token_protos:
                comb = comb[-self.max_token_protos :]
            self.token_protos[expert_idx] = comb.clone()

    @torch.no_grad()
    def evaluate(
        self,
        expert_idx: int,
        seq_sig: torch.Tensor,
        expert_centroid: Optional[torch.Tensor],
        token_latents: Optional[torch.Tensor] = None,
        token_surprise: Optional[torch.Tensor] = None,
        token_nll: Optional[torch.Tensor] = None,
        expert_is_converged: bool = False,
        update_baseline: bool = True,
        external_novel: bool = False,
    ) -> NoveltyVerdict:
        """
        Evaluate whether the current batch exhibits coarse, near-twin, or local-nuance novelty
        relative to expert_idx's calibrated in-distribution baseline.
        """
        self.sync_num_experts(max(expert_idx + 1, len(self.expert_stats)))
        stats = self.expert_stats[expert_idx]

        # 1. Coarse Sequence Distance
        if expert_centroid is not None and expert_centroid.numel() > 0:
            cent = F.normalize(expert_centroid.mean(dim=0, keepdim=True).float(), p=2, dim=-1).to(seq_sig.device)
            seq_sims = (seq_sig.float() @ cent.T).squeeze(-1)
            seq_dist = float((1.0 - seq_sims).mean().item())
        else:
            seq_dist = 0.0

        # 2. Local Token / Span Distance (catches single-token OOD inside familiar sequences)
        tok_dist_tail = 0.0
        tok_dist_map: Optional[torch.Tensor] = None
        protos = self.token_protos[expert_idx]
        if token_latents is not None and protos is not None and protos.numel() > 0:
            B, L, D = token_latents.shape
            norm_tok = F.normalize(token_latents.float(), p=2, dim=-1)
            p_dev = protos.to(norm_tok.device)
            # [B, L, M] -> max similarity to any stored prototype
            sims = torch.matmul(norm_tok, p_dev.T).max(dim=-1).values
            tok_dist_map = 1.0 - sims  # [B, L]
            k_tail = max(1, int(round(L * self.tail_fraction)))
            tok_dist_tail = float(torch.topk(tok_dist_map, k=k_tail, dim=-1).values.mean().item())

        # 3. Surprise Mean & Tail (catches localized contradiction / nuance spikes)
        surp_mean = 0.0
        surp_tail = 0.0
        if token_surprise is not None and token_surprise.numel() > 0:
            L = token_surprise.size(-1)
            k_tail = max(1, int(round(L * self.tail_fraction)))
            surp_mean = float(token_surprise.float().mean().item())
            surp_tail = float(torch.topk(token_surprise.float(), k=k_tail, dim=-1).values.mean().item())

        # 4. Outcome / Prefix NLL Mean & Tail (when labels/targets are observed)
        nll_mean = 0.0
        nll_tail = 0.0
        if token_nll is not None and token_nll.numel() > 0:
            L = token_nll.size(-1)
            k_tail = max(1, int(round(L * self.tail_fraction)))
            nll_mean = float(token_nll.float().mean().item())
            nll_tail = float(torch.topk(token_nll.float(), k=k_tail, dim=-1).values.mean().item())

        # Compute calibrated z-scores against this expert's own in-distribution history
        z_seq = stats.z_score("seq_dist", seq_dist, min_std=0.035)
        z_tok = stats.z_score("tok_dist_tail", tok_dist_tail, min_std=0.04)
        z_surp_m = stats.z_score("surp_mean", surp_mean, min_std=0.05) if expert_is_converged else 0.0
        z_surp_t = stats.z_score("surp_tail", surp_tail, min_std=0.06) if expert_is_converged else 0.0
        z_nll_m = stats.z_score("nll_mean", nll_mean, min_std=0.15) if expert_is_converged else 0.0
        z_nll_t = stats.z_score("nll_tail", nll_tail, min_std=0.15) if expert_is_converged else 0.0

        # Absolute floor guards ensure small statistical fluctuations on near-zero baselines
        # don't trigger unless the absolute signal is also meaningful.
        # Once calibrated (stats.count >= 5), triggers require significant excess over
        # the expert's own in-distribution baseline so high-entropy real text streams
        # do not false-alarm on static thresholds.
        is_calibrated = stats.count >= 5
        base_seq = stats.means.get("seq_dist", 0.0)
        base_tok = stats.means.get("tok_dist_tail", 0.0)
        base_surp = stats.means.get("surp_mean", 0.0)
        base_surp_t = stats.means.get("surp_tail", 0.0)
        base_nll_m = stats.means.get("nll_mean", 0.0)
        base_nll_t = stats.means.get("nll_tail", 0.0)

        coarse_trigger = (
            seq_dist >= max(self.seq_dist_threshold, base_seq + 0.18)
        ) or (is_calibrated and z_seq >= self.z_threshold and seq_dist >= max(0.28, base_seq + 0.12))

        if is_calibrated:
            near_twin_trigger = expert_is_converged and (
                (z_surp_m >= self.z_threshold and surp_mean >= max(self.surp_threshold, base_surp + 0.10))
                or (token_nll is not None and z_nll_m >= self.z_threshold and nll_mean >= max(0.50, base_nll_m + 0.35))
            )
            local_nuance_trigger = (
                (tok_dist_map is not None and z_tok >= self.z_threshold and tok_dist_tail >= max(self.tok_dist_threshold, base_tok + 0.08))
                or (expert_is_converged and z_surp_t >= self.z_threshold and surp_tail >= max(0.35, base_surp_t + 0.08))
                or (expert_is_converged and token_nll is not None and z_nll_t >= self.z_threshold and nll_tail >= max(0.80, base_nll_t + 0.40))
            )
        elif not update_baseline:
            # Uncalibrated one-shot evaluation fallback
            near_twin_trigger = expert_is_converged and (surp_mean >= self.surp_threshold)
            local_nuance_trigger = (
                tok_dist_map is not None and tok_dist_tail >= self.tok_dist_threshold
            )
        else:
            # Cold-start warmup (first 5 in-distribution batches): allow baseline stats
            # to calibrate before firing surprise/token-tail triggers.
            near_twin_trigger = False
            local_nuance_trigger = False

        if coarse_trigger:
            novelty_type = "coarse_domain"
        elif near_twin_trigger:
            novelty_type = "near_twin_rule"
        elif local_nuance_trigger:
            novelty_type = "local_nuance"
        else:
            novelty_type = "none"

        is_novel = novelty_type != "none"

        max_z = max(z_seq, z_tok, z_surp_m, z_surp_t, z_nll_m, z_nll_t)
        raw_max = max(
            seq_dist / max(self.seq_dist_threshold, 1e-6),
            tok_dist_tail / max(self.tok_dist_threshold, 1e-6),
            (surp_tail if expert_is_converged else 0.0) / max(self.surp_threshold, 1e-6),
        )
        novelty_score = float(torch.sigmoid(torch.tensor(max(max_z - self.z_threshold, (raw_max - 1.0) * 3.0))).item())

        # Token-level mask pinpointing which positions drove local nuance / surprise
        novel_token_mask: Optional[torch.Tensor] = None
        if token_surprise is not None:
            novel_token_mask = token_surprise >= self.surp_threshold
        if tok_dist_map is not None:
            tok_flag = tok_dist_map >= self.tok_dist_threshold
            novel_token_mask = tok_flag if novel_token_mask is None else (novel_token_mask | tok_flag)

        # Only update in-distribution baseline stats when the batch is NOT novel
        if update_baseline and (not is_novel) and (not external_novel):
            obs: Dict[str, float] = {"seq_dist": seq_dist}
            if tok_dist_map is not None:
                obs["tok_dist_tail"] = tok_dist_tail
            if expert_is_converged:
                if token_surprise is not None:
                    obs["surp_mean"] = surp_mean
                    obs["surp_tail"] = surp_tail
                if token_nll is not None:
                    obs["nll_mean"] = nll_mean
                    obs["nll_tail"] = nll_tail
            stats.update(obs)
            if token_latents is not None:
                self.update_token_prototypes(expert_idx, token_latents)

        return NoveltyVerdict(
            is_novel=is_novel,
            novelty_score=novelty_score,
            novelty_type=novelty_type,
            channel_z={
                "z_seq": z_seq,
                "z_tok": z_tok,
                "z_surp_mean": z_surp_m,
                "z_surp_tail": z_surp_t,
                "z_nll_mean": z_nll_m,
                "z_nll_tail": z_nll_t,
            },
            novel_token_mask=novel_token_mask,
        )

