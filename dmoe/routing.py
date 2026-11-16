import math
from typing import Dict, List, Optional, Tuple

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
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.sig_dim = 3 * int(d_model)
        self.max_protos = int(max_protos)
        self.lda_reg = float(lda_reg)
        self.sim_threshold = float(sim_threshold)
        self.lda_min_protos = int(lda_min_protos)

        self.expert_protos: List[Optional[torch.Tensor]] = []
        self._sim_means: List[float] = []
        self._sim_vars: List[float] = []
        self._sim_counts: List[int] = []
        self._dirty: bool = True
        self._lda_w: Optional[torch.Tensor] = None
        self._lda_b: Optional[torch.Tensor] = None
        self._lda_valid_mask: Optional[torch.Tensor] = None

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
    def extract_signature(self, x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """
        Compute causal sequence transition signature [B, 3 * d_model].
        Prefix x[:, :t] depends strictly on tokens <= t-1.
        """
        emb = self._extract_token_embeddings(x, encoder).float()  # [B, L, D]
        B, L, D = emb.shape
        u_mean = F.normalize(emb.mean(dim=1), p=2, dim=-1, eps=1e-6)
        if L >= 2:
            # Non-commutative HRR cyclic-shift binding preserves directed transitions a -> b
            bi_prod = torch.roll(emb[:, :-1, :], shifts=1, dims=-1) * emb[:, 1:, :]
            b_mean = F.normalize(bi_prod.mean(dim=1), p=2, dim=-1, eps=1e-6)
        else:
            b_mean = u_mean
        if L >= 3:
            skip_prod = torch.roll(emb[:, :-2, :], shifts=2, dims=-1) * emb[:, 2:, :]
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
        self.sync_num_experts(expert_idx + 1)
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
                thr_per_exp[e_idx] = min(self.sim_threshold, cal_thr)

        if self._lda_w is not None and self._lda_valid_mask is not None:
            usable = self._lda_valid_mask & (~dormant_t)
            if int(usable.sum().item()) >= 2:
                lda_scores = sigs @ self._lda_w.T + self._lda_b.unsqueeze(0)
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

