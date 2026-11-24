import functools
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cap_rows(x: torch.Tensor, max_rows: int, gen: torch.Generator) -> torch.Tensor:
    """Return at most max_rows rows from x, uniformly sampled without replacement."""
    n = x.size(0)
    if n <= max_rows:
        return x
    perm = torch.randperm(n, generator=gen, device=x.device)[:max_rows]
    return x[perm.sort().values]


def _meandist_statistic(x: torch.Tensor, ref: torch.Tensor) -> float:
    """Euclidean distance between sample centroids."""
    return float((x.mean(dim=0) - ref.mean(dim=0)).norm().item())


def perm_test_exchangeability(
    x: torch.Tensor,
    ref: torch.Tensor,
    alpha: float = 0.05,
    num_perms: int = 100,
    seed: int = 42,
) -> Tuple[bool, float]:
    """
    Exact two-sample permutation test (adapted from growmoe.settest in spherical_hyperspace_memory).
    Tests the null hypothesis: x and ref come from the same underlying distribution.
    Returns (rejected: bool, p_value: float).
    rejected=True means the batch x is NOT exchangeable with ref (i.e. x is novel to this expert).
    """
    n_x = x.size(0)
    n_ref = ref.size(0)
    if n_x < 2 or n_ref < 2:
        return False, 1.0

    t_obs = _meandist_statistic(x, ref)
    pooled = torch.cat([x, ref], dim=0)
    n_total = n_x + n_ref

    gen = torch.Generator(device=x.device if x.is_cuda else "cpu")
    gen.manual_seed(seed)

    exceedances = 0
    max_exceed = math.floor(alpha * (1 + num_perms) + 1e-9) - 1

    for _ in range(num_perms):
        perm = torch.randperm(n_total, generator=gen, device=pooled.device)
        perm_x = pooled[perm[:n_x]]
        perm_ref = pooled[perm[n_x:]]
        t_perm = _meandist_statistic(perm_x, perm_ref)
        if t_perm >= t_obs:
            exceedances += 1
            if exceedances > max_exceed:
                # Early stop: impossible to reject at alpha level
                return False, (1.0 + exceedances) / (1.0 + num_perms)

    p_value = (1.0 + exceedances) / (1.0 + num_perms)
    rejected = p_value <= alpha
    return rejected, p_value


class SetGate(nn.Module):
    """
    Batch-Level Hypothesis Gating adapted from spherical_hyperspace_memory (growmoe.setgate).
    Routes incoming chunk representations [B, S_chunks, d_model] to existing experts
    by testing exchangeability against each expert's stored exemplar signature buffer.
    
    If no existing expert accepts the batch (p <= alpha for all), decision is 'novel'.
    """

    def __init__(
        self,
        d_model: int,
        alpha: float = 0.05,
        num_perms: int = 100,
        max_ref_rows: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.alpha = float(alpha)
        self.num_perms = int(num_perms)
        self.max_ref_rows = int(max_ref_rows)

    def route_batch(
        self,
        batch_latents: torch.Tensor,
        expert_exemplars: List[torch.Tensor],
        step: int = 0,
    ) -> Tuple[str, int, float]:
        """
        batch_latents: [N, d_model] flattened chunk latent representations.
        expert_exemplars: list of [M_k, d_model] tensors of reference latents per expert.
        
        Returns:
            verdict: 'known' | 'novel'
            owner_expert_idx: int (0..E-1 if known, -1 if novel)
            best_p_value: float
        """
        n_experts = len(expert_exemplars)
        if n_experts == 0:
            return "novel", -1, 0.0

        gen = torch.Generator(device="cpu")
        gen.manual_seed(step * 10007 + 42)

        # Cap batch rows for fast constant-time permutation testing
        capped_batch = _cap_rows(batch_latents.detach().cpu(), self.max_ref_rows, gen)
        batch_centroid = capped_batch.mean(dim=0)

        # Rank candidate experts by distance between reference centroid and batch centroid
        candidates: List[Tuple[float, int, torch.Tensor]] = []
        for e_idx, ref in enumerate(expert_exemplars):
            if ref is None or ref.numel() == 0 or ref.size(0) < 2:
                continue
            ref_capped = _cap_rows(ref.detach().cpu(), self.max_ref_rows, gen)
            dist = float((ref_capped.mean(dim=0) - batch_centroid).norm().item())
            candidates.append((dist, e_idx, ref_capped))

        if not candidates:
            return "novel", -1, 0.0

        # Sort nearest centroid first
        candidates.sort(key=lambda item: item[0])

        best_p = 0.0
        # Test candidate experts in order of distance
        for _, e_idx, ref_capped in candidates:
            seed_k = step * 10007 + e_idx * 997 + 13
            rejected, p_val = perm_test_exchangeability(
                capped_batch,
                ref_capped,
                alpha=self.alpha,
                num_perms=self.num_perms,
                seed=seed_k,
            )
            if p_val > best_p:
                best_p = p_val
            if not rejected:
                # Admitted: batch matches this expert's reference distribution
                return "known", e_idx, p_val

        # All existing experts rejected the batch -> Truly NOVEL domain!
        return "novel", -1, best_p


class SequenceContentRouter(nn.Module):
    """
    Per-Sequence Content-Addressed Router for Mixed Streams (B2).
    Evaluates each sequence individually against expert centroids in normalized latent space.
    Operates in O(B * E * d) entirely on GPU with zero host-device synchronization.
    """

    def __init__(self, d_model: int, sim_threshold: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.sim_threshold = float(sim_threshold)

    @torch.no_grad()
    def extract_signature(self, x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        Extract normalized sequence content signature [B, d_model].
        Modality-agnostic: pooled byte embedding for bytes, token embedding for text, patch for image.
        """
        if hasattr(encoder, "byte_chunker") and encoder.byte_chunker is not None:
            emb = encoder.byte_chunker.byte_embed(x.long().clamp(0, 257))
            sig = emb.mean(dim=1)
        elif hasattr(encoder, "text_embed") and encoder.text_embed is not None:
            emb = encoder.text_embed(x.long())
            sig = emb.mean(dim=1)
        elif x.dim() == 4 and hasattr(encoder, "image_patch_proj"):
            patches = encoder.image_patch_proj(x).flatten(2).transpose(1, 2)
            sig = patches.mean(dim=1)
        elif x.dim() == 3:
            sig = x.mean(dim=1)
        else:
            sig = x.float().reshape(x.size(0), -1)[:, : self.d_model]

        return F.normalize(sig, p=2, dim=-1)

    @torch.no_grad()
    def route_sequences(
        self,
        signatures: torch.Tensor,
        expert_exemplars: List[Optional[torch.Tensor]],
        fallback_expert: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route each sequence individually based on normalized cosine similarity to expert centroids.

        Args:
            signatures: [B, d_model] normalized sequence signatures.
            expert_exemplars: list of [M_e, d_model] tensors per expert.
            fallback_expert: default expert index if no expert has exemplars.

        Returns:
            assigned_experts: [B] long tensor of expert indices.
            max_sims: [B] float tensor of similarity to winning expert.
            is_novel: [B] bool tensor indicating whether max_sim < sim_threshold.
        """
        bsz = signatures.size(0)
        device = signatures.device

        valid_experts = []
        centroids = []
        for e_idx, ex in enumerate(expert_exemplars):
            if ex is not None and ex.numel() > 0:
                cent = F.normalize(ex.mean(dim=0, keepdim=True).float(), p=2, dim=-1)
                valid_experts.append(e_idx)
                centroids.append(cent)

        if not centroids:
            return (
                torch.full((bsz,), fallback_expert, dtype=torch.long, device=device),
                torch.zeros(bsz, device=device),
                torch.ones(bsz, dtype=torch.bool, device=device),
            )

        stacked_centroids = torch.cat(centroids, dim=0).to(device)
        sims = torch.matmul(signatures, stacked_centroids.T)
        max_sims, best_sub_idx = sims.max(dim=-1)

        expert_map = torch.tensor(valid_experts, dtype=torch.long, device=device)
        assigned_experts = expert_map[best_sub_idx]
        is_novel = max_sims < self.sim_threshold

        return assigned_experts, max_sims, is_novel


class CausalPrototypeRouter(nn.Module):
    """
    Header-Free Multi-Prototype & Closed-Form LDA Sequence Router (M04 + M15 upgrade).
    Extracts strictly causal sequence signatures capturing unigram + directed bigram
    + lag-2 transition geometry (rotational Hadamard binding) over shared vocabularies,
    and routes via cached Cholesky LDA across up to `max_protos` per expert plus
    per-prototype max-cosine density for multimodal OOD/novelty detection.
    """

    def __init__(
        self,
        d_model: int,
        max_protos: int = 32,
        lda_reg: float = 0.05,
        sim_threshold: float = 0.55,
        lda_min_protos: int = 4,
        tail_window: int = 0,
        use_idf_weights: bool = False,
        center_multi_expert: bool = False,
        revisit_sim_margin: float = 0.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.sig_dim = 3 * int(d_model)
        self.max_protos = int(max_protos)
        self.lda_reg = float(lda_reg)
        self.sim_threshold = float(sim_threshold)
        self.lda_min_protos = int(lda_min_protos)
        self.tail_window = max(0, int(tail_window))
        self.use_idf_weights = bool(use_idf_weights)
        self.center_multi_expert = bool(center_multi_expert)
        self.revisit_sim_margin = float(revisit_sim_margin)

        self.register_buffer("_byte_ema", torch.ones(258, dtype=torch.float32) / 258.0)
        self._byte_obs_steps: int = 0

        self.expert_protos: List[Optional[torch.Tensor]] = []
        self._sim_means: List[float] = []
        self._sim_vars: List[float] = []
        self._sim_counts: List[int] = []
        self._dirty: bool = True
        self._lda_w: Optional[torch.Tensor] = None
        self._lda_b: Optional[torch.Tensor] = None
        self._lda_valid_mask: Optional[torch.Tensor] = None
        self.last_proto_sims: Optional[torch.Tensor] = None
        self.last_thr_per_exp: Optional[torch.Tensor] = None

    @staticmethod
    def _extract_token_embeddings(x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        if isinstance(encoder, nn.Embedding):
            return encoder(x.long())
        if hasattr(encoder, "byte_embed") and encoder.byte_embed is not None:
            return encoder.byte_embed(x.long())
        if hasattr(encoder, "byte_chunker") and encoder.byte_chunker is not None:
            return encoder.byte_chunker.byte_embed(x.long().clamp(0, 257))
        if hasattr(encoder, "text_embed") and encoder.text_embed is not None:
            return encoder.text_embed(x.long())
        if x.dim() == 3:
            return x.float()
        raise ValueError("Unsupported encoder for CausalPrototypeRouter.extract_signature")

    @torch.no_grad()
    def extract_signature(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        tail_window: Optional[int] = None,
        context_h: Optional[torch.Tensor] = None,
        context_blend: float = 0.0,
    ) -> torch.Tensor:
        """
        Compute causal sequence transition signature [B, 3 * d_model].
        Prefix x[:, :t] depends strictly on tokens <= t-1.
        When tail_window > 0, computes the signature over the most recent `tail_window`
        tokens so mid-stream domain transitions adapt rapidly without early-token dilution.
        When context_h and context_blend > 0 are provided, blends the contextualized
        Layer-0 trunk representation into the unigram semantic channel.
        """
        win = self.tail_window if tail_window is None else int(tail_window)
        x_slice = x[:, -win:] if (win > 0 and x.size(1) > win) else x
        emb = self._extract_token_embeddings(x_slice, encoder).float()  # [B, L, D]
        B, L, D = emb.shape
        if self.use_idf_weights and x_slice.dim() == 2 and x_slice.dtype in (torch.long, torch.int64, torch.int32, torch.uint8):
            x_ids = x_slice.long().clamp(0, 257)
            if self.training:
                counts = torch.bincount(x_ids.reshape(-1), minlength=258).float().to(self._byte_ema.device)
                freq = counts / counts.sum().clamp_min(1.0)
                decay = 0.90 if self._byte_obs_steps < 10 else 0.98
                self._byte_ema.mul_(decay).add_(freq * (1.0 - decay))
                self._byte_obs_steps += 1
            p_tok = self._byte_ema.to(emb.device)[x_ids].clamp_min(1e-4)
            w_tok = (1.0 / torch.sqrt(p_tok)).clamp(max=8.0)
            w_tok = w_tok / w_tok.mean(dim=1, keepdim=True).clamp_min(1e-6)
            u_raw = (emb * w_tok.unsqueeze(-1)).mean(dim=1)
        else:
            w_tok = None
            u_raw = emb.mean(dim=1)
        u_mean = F.normalize(u_raw, p=2, dim=-1, eps=1e-6)
        if (
            context_h is not None
            and context_blend > 0.0
            and context_h.dim() == 3
            and context_h.size(-1) == D
        ):
            h_slice = context_h[:, -win:] if (win > 0 and context_h.size(1) > win) else context_h
            if w_tok is not None:
                ctx_raw = (h_slice.detach().float() * w_tok.unsqueeze(-1)).mean(dim=1)
            else:
                ctx_raw = h_slice.detach().float().mean(dim=1)
            ctx_mean = F.normalize(ctx_raw, p=2, dim=-1, eps=1e-6)
            u_mean = F.normalize(
                (1.0 - float(context_blend)) * u_mean + float(context_blend) * ctx_mean,
                p=2,
                dim=-1,
                eps=1e-6,
            )
        if L >= 2:
            # Non-commutative HRR cyclic-shift binding preserves directed transitions a -> b
            bi_prod = torch.roll(emb[:, :-1, :], shifts=1, dims=-1) * emb[:, 1:, :]
            if w_tok is not None:
                w_bi = 0.5 * (w_tok[:, :-1] + w_tok[:, 1:])
                b_mean = F.normalize((bi_prod * w_bi.unsqueeze(-1)).mean(dim=1), p=2, dim=-1, eps=1e-6)
            else:
                b_mean = F.normalize(bi_prod.mean(dim=1), p=2, dim=-1, eps=1e-6)
        else:
            b_mean = u_mean
        if L >= 3:
            skip_prod = torch.roll(emb[:, :-2, :], shifts=2, dims=-1) * emb[:, 2:, :]
            if w_tok is not None:
                w_sk = 0.5 * (w_tok[:, :-2] + w_tok[:, 2:])
                s_mean = F.normalize((skip_prod * w_sk.unsqueeze(-1)).mean(dim=1), p=2, dim=-1, eps=1e-6)
            else:
                s_mean = F.normalize(skip_prod.mean(dim=1), p=2, dim=-1, eps=1e-6)
        else:
            s_mean = b_mean
        return F.normalize(torch.cat([u_mean, b_mean, s_mean], dim=-1), p=2, dim=-1, eps=1e-6)

    @torch.no_grad()
    def sync_num_experts(self, num_experts: int) -> None:
        while len(self.expert_protos) < num_experts:
            self.expert_protos.append(None)
            self._sim_means.append(0.0)
            self._sim_vars.append(0.01)
            self._sim_counts.append(0)
            self._dirty = True
        if len(self.expert_protos) > num_experts:
            self.expert_protos = self.expert_protos[:num_experts]
            self._sim_means = self._sim_means[:num_experts]
            self._sim_vars = self._sim_vars[:num_experts]
            self._sim_counts = self._sim_counts[:num_experts]
            self._dirty = True

    @torch.no_grad()
    def prune_expert(self, expert_idx: int) -> None:
        if 0 <= expert_idx < len(self.expert_protos):
            del self.expert_protos[expert_idx]
            if expert_idx < len(self._sim_means):
                del self._sim_means[expert_idx]
                del self._sim_vars[expert_idx]
                del self._sim_counts[expert_idx]
            self._dirty = True

    @torch.no_grad()
    def update_prototypes(self, expert_idx: int, sigs: torch.Tensor) -> None:
        self.sync_num_experts(max(len(self.expert_protos), expert_idx + 1))
        flat = F.normalize(sigs.detach().reshape(-1, self.sig_dim).float(), p=2, dim=-1, eps=1e-6)
        if flat.numel() == 0:
            return
        cur = self.expert_protos[expert_idx]
        if cur is not None and cur.numel() > 0:
            # Measure incoming in-distribution similarity against existing stored prototypes
            # so threshold adapts cleanly across dense vs shared-vocab streams.
            in_sims = (flat @ cur.to(flat.device).T).max(dim=-1).values
            batch_mean = float(in_sims.mean().item())
            batch_var = float(in_sims.var(unbiased=False).item()) if in_sims.numel() > 1 else 0.0025
            cnt = self._sim_counts[expert_idx] + 1
            alpha = 1.0 / float(cnt) if cnt < 4 else max(1.0 / float(cnt), 0.15)
            if self._sim_counts[expert_idx] == 0:
                self._sim_means[expert_idx] = batch_mean
                self._sim_vars[expert_idx] = max(batch_var, 1e-3)
            else:
                self._sim_means[expert_idx] = (1.0 - alpha) * self._sim_means[expert_idx] + alpha * batch_mean
                self._sim_vars[expert_idx] = max(
                    (1.0 - alpha) * self._sim_vars[expert_idx] + alpha * batch_var,
                    1e-3,
                )
            self._sim_counts[expert_idx] = cnt
        if cur is None or cur.numel() == 0:
            new_buf = flat[: self.max_protos].clone()
        else:
            combined = torch.cat([cur.to(flat.device), flat], dim=0)
            if combined.size(0) > self.max_protos:
                # Strided retention across a bounded anchor fraction + recent tail prevents cold-start drift
                keep_old = max(1, self.max_protos // 4)
                keep_new = self.max_protos - keep_old
                old_part = cur.to(flat.device)
                if old_part.size(0) > keep_old:
                    idx_old = torch.linspace(0, old_part.size(0) - 1, steps=keep_old, device=flat.device).long()
                    old_part = old_part[idx_old]
                new_part = combined[-keep_new:]
                combined = torch.cat([old_part, new_part], dim=0)[-self.max_protos :]
            new_buf = combined.clone()
        self.expert_protos[expert_idx] = new_buf
        self._dirty = True

    @torch.no_grad()
    def _fit_lda(self, device: torch.device) -> None:
        k = len(self.expert_protos)
        if k == 0:
            self._lda_w = None
            self._lda_b = None
            self._lda_valid_mask = None
            self._dirty = False
            return

        xs: List[torch.Tensor] = []
        ys: List[torch.Tensor] = []
        counts = torch.zeros(k, dtype=torch.long, device=device)
        for e_idx, buf in enumerate(self.expert_protos):
            if buf is not None and buf.size(0) >= 1:
                b_dev = buf.to(device=device, dtype=torch.float64)
                xs.append(b_dev)
                ys.append(torch.full((b_dev.size(0),), e_idx, dtype=torch.long, device=device))
                counts[e_idx] = b_dev.size(0)

        valid_mask = counts >= self.lda_min_protos
        if int(valid_mask.sum().item()) < 2:
            self._lda_w = None
            self._lda_b = None
            self._lda_valid_mask = valid_mask
            self._dirty = False
            return

        x_all = torch.cat(xs, dim=0)  # [N, D]
        y_all = torch.cat(ys, dim=0)  # [N]
        keep_rows = valid_mask[y_all]
        x_use = x_all[keep_rows]
        y_use = y_all[keep_rows]
        n_use = x_use.size(0)
        d = x_use.size(1)

        mu = torch.zeros(k, d, dtype=torch.float64, device=device)
        for e_idx in range(k):
            if valid_mask[e_idx]:
                mu[e_idx] = x_use[y_use == e_idx].mean(dim=0)

        xc = x_use - mu[y_use]
        denom = max(1, n_use - int(valid_mask.sum().item()))
        cov = (xc.T @ xc) / float(denom) + self.lda_reg * torch.eye(d, dtype=torch.float64, device=device)
        try:
            chol = torch.linalg.cholesky(cov)
            w = torch.cholesky_solve(mu.T, chol).T  # [k, d]
        except RuntimeError:
            w = torch.linalg.solve(cov + 1e-3 * torch.eye(d, dtype=torch.float64, device=device), mu.T).T

        # Balanced class priors across valid experts so equalized counts don't skew routing
        n_valid = max(1, int(valid_mask.sum().item()))
        log_prior = math.log(1.0 / float(n_valid))
        b = -0.5 * (mu * w).sum(dim=1) + log_prior
        b = b.masked_fill(~valid_mask, -1e9)
        w = w * valid_mask.unsqueeze(1).to(dtype=torch.float64)

        self._lda_w = w.float()
        self._lda_b = b.float()
        self._lda_valid_mask = valid_mask
        self._dirty = False

    @torch.no_grad()
    def route_sequences(
        self,
        signatures: torch.Tensor,
        dormant_mask: Optional[List[bool]] = None,
        fallback_expert: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route sequences [B, 3 * d_model] using cached LDA logits + max-proto cosine novelty.
        Returns: (assigned_experts [B], max_proto_sims [B], is_novel [B])
        """
        bsz = signatures.size(0)
        device = signatures.device
        sigs = F.normalize(signatures.float(), p=2, dim=-1, eps=1e-6)
        k = len(self.expert_protos)
        if k == 0:
            self.last_proto_sims = None
            self.last_thr_per_exp = None
            return (
                torch.full((bsz,), fallback_expert, dtype=torch.long, device=device),
                torch.zeros(bsz, device=device),
                torch.ones(bsz, dtype=torch.bool, device=device),
            )

        if self._dirty or (self._lda_w is not None and self._lda_w.device != device):
            self._fit_lda(device)

        dormant_t = torch.tensor(
            dormant_mask[:k] if dormant_mask is not None else [False] * k,
            dtype=torch.bool,
            device=device,
        )
        if dormant_t.numel() < k:
            dormant_t = F.pad(dormant_t, (0, k - dormant_t.numel()), value=False)

        # Compute per-expert max-prototype cosine similarity [B, k]
        proto_sims = torch.full((bsz, k), -1.0, dtype=torch.float32, device=device)
        has_any = torch.zeros(k, dtype=torch.bool, device=device)
        for e_idx, buf in enumerate(self.expert_protos):
            if buf is not None and buf.numel() > 0 and not bool(dormant_t[e_idx].item()):
                b_dev = buf.to(device=device, dtype=torch.float32)
                proto_sims[:, e_idx] = (sigs @ b_dev.T).max(dim=-1).values
                has_any[e_idx] = True

        if not bool(has_any.any().item()):
            self.last_proto_sims = None
            self.last_thr_per_exp = None
            return (
                torch.full((bsz,), fallback_expert, dtype=torch.long, device=device),
                torch.zeros(bsz, device=device),
                torch.ones(bsz, dtype=torch.bool, device=device),
            )

        max_sims = proto_sims.max(dim=-1).values.clamp(-1.0, 1.0)
        # Per-expert calibrated similarity floor: falls back to sim_threshold on cold start
        # and adapts to each expert's intra-domain dispersion once calibrated.
        thr_per_exp = torch.full((k,), self.sim_threshold, dtype=torch.float32, device=device)
        for e_idx in range(min(k, len(self._sim_counts))):
            if self._sim_counts[e_idx] >= 4:
                std_e = max(self._sim_vars[e_idx] ** 0.5, 0.04)
                cal_thr = max(0.18, self._sim_means[e_idx] - 2.75 * std_e)
                if self._sim_means[e_idx] > self.sim_threshold + 0.12:
                    thr_per_exp[e_idx] = max(self.sim_threshold, min(0.85, cal_thr))
                else:
                    thr_per_exp[e_idx] = min(self.sim_threshold, cal_thr)
        self.last_proto_sims = proto_sims.detach()
        self.last_thr_per_exp = thr_per_exp.detach()

        if self._lda_w is not None and self._lda_valid_mask is not None:
            usable = self._lda_valid_mask & (~dormant_t)
            if int(usable.sum().item()) >= 2:
                lda_scores = sigs @ self._lda_w.T + self._lda_b.unsqueeze(0)
                if self.center_multi_expert:
                    # Symmetric cross-expert prototype centering removes background vocab direction
                    mus = [
                        buf.to(device=device, dtype=torch.float32).mean(dim=0)
                        for ei, buf in enumerate(self.expert_protos)
                        if buf is not None and buf.numel() > 0 and not bool(dormant_t[ei].item())
                    ]
                    if len(mus) >= 2:
                        grand_mu = torch.stack(mus, dim=0).mean(dim=0, keepdim=True)
                        c_sigs = F.normalize(sigs - grand_mu, p=2, dim=-1, eps=1e-6)
                        c_mus = F.normalize(torch.stack(mus, dim=0) - grand_mu, p=2, dim=-1, eps=1e-6)
                        # Blend centered cosine discriminative score into LDA logits cleanly
                        valid_idx = [ei for ei, buf in enumerate(self.expert_protos) if buf is not None and buf.numel() > 0 and not bool(dormant_t[ei].item())]
                        c_bonus = torch.zeros_like(lda_scores)
                        c_bonus[:, valid_idx] = (c_sigs @ c_mus.T) * 4.0
                        lda_scores = lda_scores + c_bonus
                lda_scores = lda_scores.masked_fill(~usable.unsqueeze(0), -1e9)
                assigned = lda_scores.argmax(dim=-1)
                assigned_sim = proto_sims.gather(1, assigned.unsqueeze(1)).squeeze(1)
                is_novel = assigned_sim < thr_per_exp[assigned]
                return assigned, max_sims, is_novel

        # Fallback to max prototype similarity across valid experts
        masked_sims = proto_sims.masked_fill(~has_any.unsqueeze(0), -1e9)
        assigned = masked_sims.argmax(dim=-1)
        is_novel = max_sims < thr_per_exp[assigned]
        return assigned, max_sims, is_novel

    def storage_bytes(self) -> int:
        total = 0
        for buf in self.expert_protos:
            if buf is not None:
                total += buf.numel() * 4
        if self._lda_w is not None:
            total += self._lda_w.numel() * 4
        if self._lda_b is not None:
            total += self._lda_b.numel() * 4
        return total


@functools.lru_cache(maxsize=64)
def _causal_ema_weights(seq_len: int, decay: float, device_str: str, dtype_str: str) -> torch.Tensor:
    """Lower-triangular causal EMA matrix `W [T, T]` whose rows sum to `1.0`."""
    dev = torch.device(device_str)
    dtype = getattr(torch, dtype_str.split(".")[-1], torch.float32)
    idx = torch.arange(seq_len, device=dev, dtype=torch.float32)
    diff = idx.unsqueeze(1) - idx.unsqueeze(0)  # [T, T], >=0 on causal lower triangle
    causal = diff >= 0
    weights = torch.where(causal, decay ** diff.clamp_min(0.0), torch.zeros_like(diff))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weights.to(dtype=dtype)


def build_token_dispatch_plan(
    top_ids: torch.Tensor,
    top_w: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """
    Build a sorted sparse dispatch plan `(tok_idx, weights, counts)` ONCE per forward pass.
    `top_ids, top_w` may have shape `[B, T, K]` or `[N, K]`.
    """
    flat_ids = top_ids.reshape(-1, top_ids.size(-1))
    flat_w = top_w.reshape(-1, top_w.size(-1))
    mask = flat_w > 0.0
    nz = mask.nonzero(as_tuple=False)
    E = max(1, int(num_experts))
    if nz.numel() == 0:
        empty_idx = flat_ids.new_zeros((0,), dtype=torch.long)
        empty_w = flat_w.new_zeros((0,))
        return empty_idx, empty_w, [0] * E

    tok_idx = nz[:, 0]
    slot_idx = nz[:, 1]
    exp_id = flat_ids[tok_idx, slot_idx].clamp(0, E - 1)
    weights = flat_w[tok_idx, slot_idx]
    order = torch.argsort(exp_id, stable=True)
    tok_idx = tok_idx[order]
    exp_id = exp_id[order]
    weights = weights[order]
    counts = torch.bincount(exp_id, minlength=E).tolist()
    return tok_idx, weights, [int(c) for c in counts]


def sparse_weighted_expert_dispatch(
    x_flat: torch.Tensor,
    modules: Any,
    dispatch_plan: Tuple[torch.Tensor, torch.Tensor, List[int]],
    fn_call: Optional[Any] = None,
    out_dim: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Shared top-k sorted scatter-add dispatcher over `dispatch_plan = (tok_idx, weights, counts)`.
    Returns `[N, out_dim]` additive residual or `None` when active count is zero.
    """
    tok_idx, weights, counts = dispatch_plan
    if tok_idx.numel() == 0:
        return None
    split_sizes = [int(c) for c in counts]
    if sum(split_sizes) != tok_idx.numel():
        return None
    x_gather = x_flat.index_select(0, tok_idx)
    chunks = torch.split(x_gather, split_sizes, dim=0)
    E = len(modules)
    out_chunks: List[torch.Tensor] = []
    keep_idx: List[torch.Tensor] = []
    keep_w: List[torch.Tensor] = []
    offset = 0
    for e, ci in enumerate(split_sizes):
        if ci > 0 and e < E:
            y_e = fn_call(modules[e], chunks[e], e) if fn_call is not None else modules[e](chunks[e])
            out_chunks.append(y_e)
            keep_idx.append(tok_idx[offset : offset + ci])
            keep_w.append(weights[offset : offset + ci])
        offset += ci
    if not out_chunks:
        return None
    cat_out = torch.cat(out_chunks, dim=0) * torch.cat(keep_w, dim=0).to(dtype=out_chunks[0].dtype).unsqueeze(-1)
    d_out = cat_out.size(-1) if out_dim is None else int(out_dim)
    out = x_flat.new_zeros((x_flat.size(0), d_out), dtype=cat_out.dtype)
    out.index_add_(0, torch.cat(keep_idx, dim=0), cat_out)
    return out


def compute_chunk_surface_features(
    x: torch.Tensor,
    chunk_size: int = 4,
    num_chunks: Optional[int] = None,
    format_only: bool = False,
) -> torch.Tensor:
    """
    Compute strictly causal surface-formatting descriptors `[B, T, d_s]` directly from
    raw byte IDs. When `format_only=True`, returns the 6-D pure formatting subspace
    `(pad_frac, space_frac, upper_frac, case_diff, rel_pos, causal_len_frac)` so whitening
    removes length/padding/whitespace/case shortcuts without blinding code/math lexical cues.
    Uses zero task or role labels.
    """
    B, L = x.shape
    C = max(1, int(chunk_size))
    T = int(num_chunks) if num_chunks is not None else max(1, (L + C - 1) // C)
    need_L = T * C
    if L < need_L:
        xc = F.pad(x.long(), (0, need_L - L), value=257)
    else:
        xc = x[:, :need_L].long()
    xc = xc.view(B, T, C)
    dev = x.device

    is_pad = xc == 257
    valid = (~is_pad) & (xc < 256)
    v_f = valid.float()
    v_cnt = v_f.sum(dim=-1).clamp_min(1.0)

    pad_frac = is_pad.float().mean(dim=-1)
    is_space = (xc == 32) | (xc == 9) | (xc == 10) | (xc == 13)
    space_frac = is_space.float().mean(dim=-1)
    lower_frac = ((xc >= 97) & (xc <= 122)).float().mean(dim=-1)
    upper_frac = ((xc >= 65) & (xc <= 90)).float().mean(dim=-1)
    t_grid = torch.arange(T, device=dev, dtype=torch.float32)
    rel_pos = (t_grid / float(max(1, T - 1))).unsqueeze(0).expand(B, T)
    chunk_non_pad = (~is_pad).any(dim=-1).float()
    causal_len_frac = torch.cumsum(chunk_non_pad, dim=1) / (t_grid.unsqueeze(0) + 1.0)

    if format_only:
        case_diff = upper_frac - lower_frac
        return torch.stack(
            [pad_frac, space_frac, upper_frac, case_diff, rel_pos, causal_len_frac],
            dim=-1,
        )

    digit_frac = ((xc >= 48) & (xc <= 57)).float().mean(dim=-1)
    is_punct = (
        ((xc >= 33) & (xc <= 47))
        | ((xc >= 58) & (xc <= 64))
        | ((xc >= 91) & (xc <= 96))
        | ((xc >= 123) & (xc <= 126))
    )
    punct_frac = is_punct.float().mean(dim=-1)
    non_ascii_frac = ((xc >= 128) & (xc < 256)).float().mean(dim=-1)

    xb = xc.clamp(0, 255).float() * v_f
    mean_byte = (xb.sum(dim=-1) / v_cnt) / 255.0
    var_byte = (((xb - (mean_byte.unsqueeze(-1) * 255.0)) ** 2) * v_f).sum(dim=-1) / v_cnt
    std_byte = torch.sqrt(var_byte.clamp_min(0.0)) / 128.0
    if C >= 2:
        uniq_frac = (xc[..., 1:] != xc[..., :-1]).float().mean(dim=-1)
    else:
        uniq_frac = torch.zeros((B, T), device=dev, dtype=torch.float32)

    return torch.stack(
        [
            pad_frac,
            space_frac,
            lower_frac,
            upper_frac,
            digit_frac,
            punct_frac,
            non_ascii_frac,
            mean_byte,
            std_byte,
            uniq_frac,
            rel_pos,
            causal_len_frac,
        ],
        dim=-1,
    )


class CausalContextTokenRouter(nn.Module):
    """
    v7 Zero-Replay-Safe Contextual Per-Token/Chunk Router.

    Runs after Layer-3 ThreeLevelAttention (`z_ctx [B, T, D]`):
      - Projects contextual latents via GPM-protected `q_proj` and `ctx_proj`.
      - Matches against frozen-when-mature multi-key expert centroids `K_e in [M_k, d_route]`.
      - Computes differentiable `k in {0, 1, 2}` conditional activation per chunk,
        immediately escalating `nuance_mask == True` chunks to `k = 2`.
    """

    def __init__(
        self,
        d_model: int,
        d_route: int = 96,
        max_experts: int = 16,
        initial_experts: int = 1,
        num_keys: int = 8,
        grad_scale: float = 0.25,
        ema_decay: float = 0.85,
        tau_skip: float = 0.38,
        tau_top2: float = 0.82,
        delta_mix: float = 0.05,
        scale: float = 12.0,
        active_rate_low: float = 0.35,
        active_rate_high: float = 0.80,
        warmup_steps: int = 20,
        sequence_level: bool = False,
        use_loss_free_balance: bool = True,
        bias_lr: float = 0.05,
        enable_surface_whitening: bool = False,
        surface_dim: int = 12,
        surface_ema_decay: float = 0.94,
        surface_ridge: float = 0.05,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.d_route = int(d_route)
        self.max_experts = int(max_experts)
        self.num_experts = max(1, int(initial_experts))
        self.num_keys = max(1, int(num_keys))
        self.grad_scale = float(grad_scale)
        self.ema_decay = float(ema_decay)
        self.tau_skip = float(tau_skip)
        self.tau_top2 = float(tau_top2)
        self.delta_mix = float(delta_mix)
        self.scale = float(scale)
        self.active_rate_low = float(active_rate_low)
        self.active_rate_high = float(active_rate_high)
        self.warmup_steps = int(warmup_steps)
        self.sequence_level = bool(sequence_level)
        self.use_loss_free_balance = bool(use_loss_free_balance)
        self.bias_lr = float(bias_lr)
        self.enable_surface_whitening = bool(enable_surface_whitening)
        self.surface_dim = int(surface_dim)
        self.surface_ema_decay = float(surface_ema_decay)
        self.surface_ridge = float(surface_ridge)

        self.route_norm = nn.LayerNorm(self.d_model)
        self.q_proj = nn.Linear(self.d_model, self.d_route, bias=False)
        self.ctx_proj = nn.Linear(self.d_model, self.d_route, bias=False)
        hidden_need = max(32, self.d_model // 2)
        self.need_head = nn.Sequential(
            nn.Linear(self.d_model, hidden_need),
            nn.GELU(),
            nn.Linear(hidden_need, 1),
        )
        self.spec_gate_scale = nn.Parameter(torch.tensor(0.5))

        gen = torch.Generator().manual_seed(1337)
        init_keys = F.normalize(torch.randn(self.max_experts, self.num_keys, self.d_route, generator=gen), dim=-1)
        self.register_buffer("expert_keys", init_keys)
        self.register_buffer("key_initialized", torch.zeros(self.max_experts, self.num_keys, dtype=torch.bool))
        self.register_buffer("key_counts", torch.zeros(self.max_experts, self.num_keys, dtype=torch.float32))
        self.register_buffer("expert_frozen", torch.zeros(self.max_experts, dtype=torch.bool))
        self.register_buffer("expert_dormant", torch.zeros(self.max_experts, dtype=torch.bool))
        self.register_buffer("expert_bias", torch.zeros(self.max_experts, dtype=torch.float32))
        self.register_buffer("steps_seen", torch.tensor(0, dtype=torch.long))
        if self.enable_surface_whitening:
            self.register_buffer("surf_mu_h", torch.zeros(self.d_model, dtype=torch.float32))
            self.register_buffer("surf_mu_s", torch.zeros(self.surface_dim, dtype=torch.float32))
            self.register_buffer("surf_C_hs", torch.zeros(self.d_model, self.surface_dim, dtype=torch.float32))
            self.register_buffer("surf_C_ss", torch.eye(self.surface_dim, dtype=torch.float32) * 1e-2)
            self.register_buffer("surf_proj_basis", torch.zeros(self.d_model, self.surface_dim, dtype=torch.float32))
            self.register_buffer("surf_steps", torch.tensor(0, dtype=torch.long))
        self._surf_steps_i: int = 0

    def sync_num_experts(self, n: int) -> None:
        self.num_experts = max(1, min(int(n), self.max_experts))

    @torch.no_grad()
    def freeze_expert(self, idx: int) -> None:
        if 0 <= idx < self.max_experts:
            self.expert_frozen[idx] = True

    @torch.no_grad()
    def set_dormant(self, idx: int, dormant: bool = True) -> None:
        if 0 <= idx < self.max_experts:
            self.expert_dormant[idx] = bool(dormant)

    @torch.no_grad()
    def seed_expert_from_vectors(self, idx: int, vecs: torch.Tensor) -> None:
        """Seed multi-key prototypes of `idx` using farthest-point coverage over cluster vectors."""
        if not (0 <= idx < self.max_experts) or vecs.numel() == 0:
            return
        v = F.normalize(vecs.detach().float().to(self.expert_keys.device), dim=-1)
        n = v.size(0)
        k_fill = min(self.num_keys, n)
        chosen: List[torch.Tensor] = [v.mean(dim=0)]
        chosen[0] = F.normalize(chosen[0], dim=0)
        for _ in range(1, k_fill):
            c_mat = torch.stack(chosen, dim=0)
            max_sim = (v @ c_mat.t()).max(dim=-1).values
            far_i = int(torch.argmin(max_sim).item())
            chosen.append(v[far_i])
        for j in range(k_fill):
            self.expert_keys[idx, j] = chosen[j]
            self.key_initialized[idx, j] = True
            self.key_counts[idx, j] = 1.0
        for j in range(k_fill, self.num_keys):
            self.expert_keys[idx, j] = chosen[j % k_fill]
            self.key_initialized[idx, j] = False
            self.key_counts[idx, j] = 0.0
        self.expert_frozen[idx] = False
        self.expert_dormant[idx] = False

    @torch.no_grad()
    def _update_surface_subspace(self, h_route: torch.Tensor, surface_feats: torch.Tensor) -> None:
        """Update running surface cross-covariance and soft-orthogonal null-space basis `surf_proj_basis`."""
        H = h_route.detach().float().reshape(-1, self.d_model)
        S = surface_feats.detach().float().reshape(-1, self.surface_dim).to(H.device)
        n = H.size(0)
        if n < 2:
            return
        mu_h_b = H.mean(dim=0)
        mu_s_b = S.mean(dim=0)
        steps = self._surf_steps_i
        alpha = max(1.0 - self.surface_ema_decay, 1.0 / float(steps + 1))
        self.surf_mu_h.mul_(1.0 - alpha).add_(mu_h_b * alpha)
        self.surf_mu_s.mul_(1.0 - alpha).add_(mu_s_b * alpha)
        Hc = H - self.surf_mu_h.unsqueeze(0)
        Sc = S - self.surf_mu_s.unsqueeze(0)
        c_hs_b = (Hc.t() @ Sc) / float(n)
        c_ss_b = (Sc.t() @ Sc) / float(n)
        if steps == 0:
            self.surf_C_hs.copy_(c_hs_b)
            self.surf_C_ss.copy_(c_ss_b)
        else:
            self.surf_C_hs.mul_(1.0 - alpha).add_(c_hs_b * alpha)
            self.surf_C_ss.mul_(1.0 - alpha).add_(c_ss_b * alpha)
        new_steps = steps + 1
        self._surf_steps_i = new_steps
        self.surf_steps.fill_(new_steps)

        # Recompute SVD basis on warmup steps (2..8) and every 8 steps thereafter to avoid per-step CUDA syncs
        if new_steps >= 2 and (new_steps <= 8 or (new_steps % 8 == 0)):
            eye_s = torch.eye(self.surface_dim, device=H.device, dtype=torch.float32)
            # W_surf = C_hs (C_ss + ridge I)^-1  ->  solve (C_ss + ridge I) W^T = C_hs^T
            w_t = torch.linalg.solve(self.surf_C_ss + self.surface_ridge * eye_s, self.surf_C_hs.t())
            W_surf = w_t.t()  # [D, d_s]
            U, sv, _ = torch.linalg.svd(W_surf, full_matrices=False)
            # Exact orthogonal projection on active surface singular vectors (avoids soft-residual leakage
            # while skipping zero-variance directions when a surface feature is constant).
            active = (sv > 0.05).to(dtype=U.dtype)
            self.surf_proj_basis.copy_(U * active.unsqueeze(0))

    def compute_queries(
        self,
        h_attn3: torch.Tensor,
        expert_states: Optional[List[int]] = None,
        surface_feats: Optional[torch.Tensor] = None,
        update_surface_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Compute strictly causal query representations `q_tok`, `need`, `d_ctx`, and `probs`."""
        B, T, _ = h_attn3.shape
        h_mix = self.grad_scale * h_attn3 + (1.0 - self.grad_scale) * h_attn3.detach()
        h_route = self.route_norm(h_mix)
        if self.enable_surface_whitening:
            if update_surface_stats and self.training and surface_feats is not None:
                self._update_surface_subspace(h_route, surface_feats)
            if self._surf_steps_i >= 2:
                h_c = h_route.float() - self.surf_mu_h.view(1, 1, -1)
                basis = self.surf_proj_basis  # [D, d_s]
                proj = torch.matmul(torch.matmul(h_c, basis), basis.t())
                h_route = (h_c - proj).to(h_route.dtype)
        if self.sequence_level:
            h_route = h_route.mean(dim=1, keepdim=True).expand_as(h_route)
            h_ema = h_route
        else:
            W_ema = _causal_ema_weights(T, self.ema_decay, str(h_route.device), str(h_route.dtype))
            h_ema = torch.matmul(W_ema.unsqueeze(0), h_route)
        q_tok = F.normalize(self.q_proj(h_route) + self.ctx_proj(h_ema), p=2, dim=-1)
        need = torch.sigmoid(self.need_head(h_route).squeeze(-1))

        E = self.num_experts
        keys_e = F.normalize(self.expert_keys[:E].float(), dim=-1)  # [E, M_k, d_r]
        key_sims = torch.einsum("btd,emd->btem", q_tok.float(), keys_e)
        init_mask = self.key_initialized[:E]
        # Prefer initialized slots once an expert has at least one initialized key; before init use full bank
        has_init = init_mask.any(dim=-1)  # [E]
        for e in range(E):
            if bool(has_init[e].item()):
                key_sims[:, :, e, ~init_mask[e]] = -1.0

        max_key_sim = key_sims.max(dim=-1).values  # [B, T, E]
        dormant_mask = self.expert_dormant[:E].clone()
        if expert_states is not None:
            for e in range(min(E, len(expert_states))):
                if int(expert_states[e]) == 2:
                    dormant_mask[e] = True
        if bool(dormant_mask.any().item()):
            max_key_sim = max_key_sim.masked_fill(dormant_mask.view(1, 1, E), -2.0)

        d_ctx = (1.0 - max_key_sim.max(dim=-1).values).clamp(0.0, 2.0)
        logits_r = self.scale * max_key_sim
        if bool(dormant_mask.any().item()):
            logits_r = logits_r.masked_fill(dormant_mask.view(1, 1, E), -1e4)
        probs = F.softmax(logits_r, dim=-1)

        if self.use_loss_free_balance and E > 1:
            biased_sim = max_key_sim + self.expert_bias[:E].view(1, 1, E)
            if bool(dormant_mask.any().item()):
                biased_sim = biased_sim.masked_fill(dormant_mask.view(1, 1, E), -2.0)
            routing_scores = F.softmax(self.scale * biased_sim, dim=-1)
        else:
            routing_scores = probs

        return {
            "q_tok": q_tok,
            "need": need,
            "d_ctx": d_ctx,
            "max_key_sim": max_key_sim,
            "probs": probs,
            "routing_scores": routing_scores,
            "dormant_mask": dormant_mask,
        }

    def select_experts(
        self,
        query_state: Dict[str, torch.Tensor],
        nuance_mask: Optional[torch.Tensor] = None,
        training_step: Optional[int] = None,
        content_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Select per-chunk `k in {0, 1, 2}` specialists, immediately escalating nuanced chunks to `k=2`."""
        q_tok = query_state["q_tok"]
        need = query_state["need"]
        probs = query_state["probs"]
        routing_scores = query_state.get("routing_scores", probs)
        d_ctx = query_state["d_ctx"]
        max_key_sim = query_state.get("max_key_sim", None)
        B, T, E = probs.shape
        dev = probs.device

        nuanced = (
            nuance_mask.to(device=dev, dtype=torch.bool)
            if nuance_mask is not None
            else torch.zeros(B, T, dtype=torch.bool, device=dev)
        )
        valid_c = (
            content_mask.to(device=dev, dtype=torch.bool)
            if content_mask is not None
            else torch.ones(B, T, dtype=torch.bool, device=dev)
        )
        cur_step = int(self.steps_seen.item()) if training_step is None else int(training_step)
        in_warmup = self.training and cur_step < self.warmup_steps
        spec_scale = F.softplus(self.spec_gate_scale)
        gate_val = spec_scale * (0.4 + 0.6 * need)

        if E == 1:
            is_active = (torch.ones_like(nuanced) if in_warmup else ((need >= self.tau_skip) | nuanced)) & valid_c
            k_used = is_active.long()
            w1 = torch.where(is_active, gate_val, torch.zeros_like(gate_val))
            top_ids = torch.zeros(B, T, 2, dtype=torch.long, device=dev)
            top_w = torch.stack([w1, torch.zeros_like(w1)], dim=-1)
            load_bal = probs.new_tensor(0.0)
            if max_key_sim is not None:
                erc_loss = (1.0 - max_key_sim[..., 0].clamp(-1.0, 1.0)).pow(2).mean()
            else:
                erc_loss = probs.new_tensor(0.0)
        else:
            _, p_ids = torch.topk(routing_scores, k=2, dim=-1)
            p_vals = torch.gather(probs, -1, p_ids)
            margin = p_vals[..., 0] - p_vals[..., 1]
            is_skip = (~valid_c) | (torch.zeros_like(nuanced) if in_warmup else ((need < self.tau_skip) & (~nuanced)))
            is_top2 = (~is_skip) & ((margin < self.delta_mix) | (need >= self.tau_top2) | nuanced)
            k_used = torch.where(is_skip, 0, torch.where(is_top2, 2, 1))

            mix_sum = (p_vals[..., 0] + p_vals[..., 1]).clamp_min(1e-6)
            w_top2 = (p_vals / mix_sum.unsqueeze(-1)) * gate_val.unsqueeze(-1)
            w_top1 = torch.stack([p_vals[..., 0] * gate_val, torch.zeros_like(gate_val)], dim=-1)
            w_zero = torch.zeros_like(w_top2)

            top_ids = p_ids
            top_w = torch.where(
                is_skip.unsqueeze(-1),
                w_zero,
                torch.where(is_top2.unsqueeze(-1), w_top2, w_top1),
            )
            # Load-balance across active non-dormant experts
            dormant_mask = query_state["dormant_mask"]
            valid_e = (~dormant_mask).float()
            n_valid = valid_e.sum().clamp_min(1.0)
            one_hot_top1 = F.one_hot(p_ids[..., 0], num_classes=E).float() * (~is_skip).float().unsqueeze(-1)
            frac_assigned = one_hot_top1.mean(dim=(0, 1))
            mean_prob = probs.mean(dim=(0, 1))
            load_bal = n_valid * (frac_assigned * mean_prob * valid_e).sum()
            # Expert-Router Coupling (ERC): match q_tok closely to assigned key + encourage distinct functional modes
            if max_key_sim is not None:
                assigned_sim = torch.gather(max_key_sim, -1, p_ids[..., :1]).squeeze(-1)
                active_f = (~is_skip).float()
                erc_align = (((1.0 - assigned_sim) * active_f).pow(2).sum()) / active_f.sum().clamp_min(1.0)
                k_means = F.normalize(self.expert_keys[:E].mean(dim=1).float(), dim=-1)
                off_cos = (k_means @ k_means.t()) * (1.0 - torch.eye(E, device=dev))
                erc_loss = erc_align + 0.15 * off_cos.pow(2).mean()
            else:
                erc_loss = probs.new_tensor(0.0)

        # Soft target-rate regularizer keeps dynamic k alive without forcing degenerate all-0 or all-2
        mean_need = need.mean()
        rate_reg = F.relu(self.active_rate_low - mean_need).pow(2) + F.relu(mean_need - self.active_rate_high).pow(2)
        if self.use_loss_free_balance:
            # DeepSeek-V3 style loss-free bias handles load balance; avoid homogenizing softmax aux penalty
            balance_loss = 0.25 * rate_reg + 0.15 * erc_loss
        else:
            balance_loss = 0.05 * load_bal + 0.25 * rate_reg + 0.10 * erc_loss

        return {
            "top_ids": top_ids,
            "top_w": top_w,
            "k_used": k_used,
            "d_ctx": d_ctx,
            "probs": probs,
            "q_tok": q_tok,
            "need": need,
            "balance_loss": balance_loss,
            "erc_loss": erc_loss,
        }

    def forward(
        self,
        h_attn3: torch.Tensor,
        nuance_mask: Optional[torch.Tensor] = None,
        expert_states: Optional[List[int]] = None,
        training_step: Optional[int] = None,
        surface_feats: Optional[torch.Tensor] = None,
        update_surface_stats: bool = False,
    ) -> Dict[str, torch.Tensor]:
        q_state = self.compute_queries(
            h_attn3,
            expert_states=expert_states,
            surface_feats=surface_feats,
            update_surface_stats=update_surface_stats,
        )
        return self.select_experts(q_state, nuance_mask=nuance_mask, training_step=training_step)

    @torch.no_grad()
    def update_keys(
        self,
        q_tok: torch.Tensor,
        top_ids: torch.Tensor,
        top_w: torch.Tensor,
        expert_states: Optional[List[int]] = None,
        new_subkey_sim: float = 0.72,
    ) -> None:
        """
        Update multi-key prototypes ONLY for `STATE_ACTIVE` experts & adjust loss-free load bias.
        Permanently frozen (`STATE_FROZEN` / `expert_frozen[e]`) experts receive zero key writes.
        """
        self.steps_seen += 1
        q_flat = F.normalize(q_tok.detach().float().reshape(-1, self.d_route), dim=-1)
        ids_flat = top_ids.detach().reshape(-1, top_ids.size(-1))
        w_flat = top_w.detach().reshape(-1, top_w.size(-1))
        E = self.num_experts

        if self.use_loss_free_balance and E > 1:
            valid_mask = torch.ones(E, dtype=torch.bool, device=q_flat.device)
            for e in range(E):
                if bool(self.expert_dormant[e].item()) or (
                    expert_states is not None and e < len(expert_states) and int(expert_states[e]) == 2
                ):
                    valid_mask[e] = False
            n_val = max(1, int(valid_mask.sum().item()))
            active_tok = w_flat[:, 0] > 0.0
            if bool(active_tok.any().item()):
                counts_e = torch.bincount(ids_flat[active_tok, 0], minlength=E)[:E].float()
                frac = counts_e / counts_e.sum().clamp_min(1.0)
                target = 1.0 / float(n_val)
                delta = self.bias_lr * torch.sign(target - frac)
                for e in range(E):
                    if valid_mask[e]:
                        self.expert_bias[e] = (self.expert_bias[e] + delta[e]).clamp(-0.35, 0.35)

        for e in range(E):
            if bool(self.expert_frozen[e].item()):
                continue
            if expert_states is not None and e < len(expert_states) and int(expert_states[e]) != 0:
                continue
            mask_e = ((ids_flat[:, 0] == e) & (w_flat[:, 0] > 0.0)) | (
                (ids_flat[:, 1] == e) & (w_flat[:, 1] > 0.0)
            )
            if not bool(mask_e.any().item()):
                continue
            qe = q_flat[mask_e]
            if qe.size(0) > 64:
                qe = qe[:64]

            keys_e = self.expert_keys[e]
            init_e = self.key_initialized[e]
            cnt_e = self.key_counts[e]
            if not bool(init_e[0].item()):
                keys_e[0] = F.normalize(qe.mean(dim=0), dim=0)
                init_e[0] = True
                cnt_e[0] = float(qe.size(0))
                if qe.size(0) == 1:
                    continue
            # Batch-assign remaining queries to nearest initialized key or allocate next uninitialized key
            init_idx = init_e.nonzero(as_tuple=False).squeeze(-1)
            sims = qe @ keys_e[init_idx].t()
            best_sim, best_pos = sims.max(dim=-1)
            best_slot = init_idx[best_pos]

            uninit_slots = (~init_e).nonzero(as_tuple=False).squeeze(-1).tolist()
            if uninit_slots:
                far_mask = best_sim < new_subkey_sim
                if bool(far_mask.any().item()):
                    far_vecs = qe[far_mask]
                    slot_new = uninit_slots[0]
                    keys_e[slot_new] = F.normalize(far_vecs.mean(dim=0), dim=0)
                    init_e[slot_new] = True
                    cnt_e[slot_new] = float(far_vecs.size(0))
                    qe = qe[~far_mask]
                    best_slot = best_slot[~far_mask]

            if qe.numel() > 0:
                for slot_val in best_slot.unique().tolist():
                    m_s = best_slot == int(slot_val)
                    c_old = float(cnt_e[slot_val].item())
                    c_new = float(m_s.sum().item())
                    eta = min(0.25, c_new / max(1.0, c_old + c_new))
                    updated = (1.0 - eta) * keys_e[slot_val] + eta * F.normalize(qe[m_s].mean(dim=0), dim=0)
                    keys_e[slot_val] = F.normalize(updated, dim=0)
                    cnt_e[slot_val] = min(1000.0, c_old + c_new)


