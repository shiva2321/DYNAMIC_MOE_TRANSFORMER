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

