"""
Parameter-matched Stage 3 continual learning baselines & budgeted dictionary control.

Includes:
- DenseRoPEContinualBaseline: causal RoPE Transformer (NoReplay / Replay128 / +Upkeeper)
- StaticMoEContinualBaseline: fixed multi-expert causal RoPE MoE with learned top-1 router
- BudgetedDictionaryBaseline: purely non-neural counting dictionary capped at `max_total_bytes`
"""

from collections import OrderedDict
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.rope import RotaryEmbedding


class _CausalRoPEBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.head_dim)
        self.norm2 = nn.LayerNorm(d_model)

    def attn(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope.apply_rotary(q, k)
        ctx = (
            F.scaled_dot_product_attention(q, k, v, is_causal=True)
            .transpose(1, 2)
            .contiguous()
            .view(B, L, D)
        )
        return x + self.out_proj(ctx)


class DenseRoPEContinualBaseline(nn.Module):
    """Parameter-matched causal RoPE dense Transformer baseline."""

    def __init__(
        self,
        d_model: int = 48,
        num_heads: int = 4,
        d_ff: int = 384,
        num_layers: int = 2,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.byte_embed = nn.Embedding(258, d_model)
        self.blocks = nn.ModuleList([_CausalRoPEBlock(d_model, num_heads) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.byte_embed(x)
        for blk, ffn in zip(self.blocks, self.ffns):
            h = blk.attn(h)
            h = h + ffn(blk.norm2(h))
        logits = self.out_head(self.final_norm(h))
        return {"output": logits.unsqueeze(2), "latent": h}


class StaticMoEContinualBaseline(nn.Module):
    """
    Parameter-matched standard Static Top-1 MoE causal RoPE Transformer.
    Trained continually end-to-end without expert freezing, prototype routing, or GPM.
    """

    def __init__(
        self,
        d_model: int = 48,
        num_heads: int = 4,
        d_ff: int = 160,
        num_layers: int = 2,
        num_experts: int = 3,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.vocab_size = vocab_size
        self.byte_embed = nn.Embedding(258, d_model)
        self.blocks = nn.ModuleList([_CausalRoPEBlock(d_model, num_heads) for _ in range(num_layers)])
        self.routers = nn.ModuleList([nn.Linear(d_model, num_experts) for _ in range(num_layers)])
        self.experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(num_experts)
            ])
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_head = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.byte_embed(x)
        aux_loss = h.new_tensor(0.0)
        assigned = None
        for l_idx, (blk, router, exps) in enumerate(zip(self.blocks, self.routers, self.experts)):
            h = blk.attn(h)
            h_n = blk.norm2(h)
            logits_r = router(h_n)  # [B, L, E]
            probs_r = F.softmax(logits_r, dim=-1)
            top1_idx = probs_r.argmax(dim=-1)  # [B, L]
            if l_idx == 0:
                # Sequence-level mode for routing diagnostic comparison
                seq_votes = F.one_hot(top1_idx, num_classes=self.num_experts).float().mean(dim=1)
                assigned = seq_votes.argmax(dim=-1)
            # Straight-through top-1 dispatch
            hard_mask = F.one_hot(top1_idx, num_classes=self.num_experts).float()
            st_gate = hard_mask - probs_r.detach() + probs_r
            moe_out = torch.zeros_like(h_n)
            for e_idx, exp in enumerate(exps):
                moe_out = moe_out + exp(h_n) * st_gate[:, :, e_idx : e_idx + 1]
            h = h + moe_out
            # Standard Switch-style load balancing auxiliary loss
            f_e = hard_mask.mean(dim=(0, 1))
            p_e = probs_r.mean(dim=(0, 1))
            aux_loss = aux_loss + float(self.num_experts) * (f_e * p_e).sum()

        logits = self.out_head(self.final_norm(h))
        return {
            "output": logits.unsqueeze(2),
            "latent": h,
            "aux_loss": 0.01 * aux_loss,
            "assigned_experts": assigned,
        }


class BudgetedDictionaryBaseline:
    """
    Pure Non-Neural Key-Value Counting Store bounded strictly by `max_total_bytes`.
    Follows the same evidence-accumulation & multi-source flip rules as CalibratedBayesianUpkeeper
    and outputs calibrated Laplace-smoothed log-probabilities over the active symbol range.
    """

    BYTES_PER_PRIMARY = 16
    BYTES_PER_TENTATIVE = 52

    def __init__(
        self,
        max_total_bytes: int = 4096,
        min_sources_for_flip: int = 2,
        min_flip_count: int = 3,
        max_prior_count: int = 16,
        flip_threshold_ratio: float = 0.5,
        vocab_start: int = 32,
        vocab_size_active: int = 16,
        vocab_total: int = 256,
    ):
        self.max_total_bytes = int(max_total_bytes)
        self.max_primary = max(4, int(self.max_total_bytes * 0.55) // self.BYTES_PER_PRIMARY)
        rem_bytes = max(0, self.max_total_bytes - self.max_primary * self.BYTES_PER_PRIMARY)
        self.max_tentative = max(2, rem_bytes // self.BYTES_PER_TENTATIVE)
        self.min_sources_for_flip = int(min_sources_for_flip)
        self.min_flip_count = int(min_flip_count)
        self.max_prior_count = int(max_prior_count)
        self.flip_threshold_ratio = float(flip_threshold_ratio)
        self.vocab_start = int(vocab_start)
        self.vocab_size_active = int(vocab_size_active)
        self.vocab_total = int(vocab_total)

        self.primary_store: OrderedDict[int, Tuple[int, int]] = OrderedDict()
        self.tentative_store: OrderedDict[int, Tuple[int, int, set, int]] = OrderedDict()
        self._call_step: int = 0

    def _evict_lowest_primary(self) -> None:
        if not self.primary_store:
            return
        victim_key = None
        victim_cnt = 1 << 30
        for idx, (k, (_, cnt)) in enumerate(self.primary_store.items()):
            if cnt < victim_cnt:
                victim_cnt = cnt
                victim_key = k
            if idx >= 15:
                break
        if victim_key is not None:
            del self.primary_store[victim_key]

    def _put_primary(self, key: int, val: int, count: int) -> None:
        if key in self.primary_store:
            self.primary_store.move_to_end(key)
        else:
            while len(self.primary_store) >= self.max_primary:
                self._evict_lowest_primary()
        self.primary_store[key] = (int(val), min(int(count), self.max_prior_count))

    def _record_tentative(self, key: int, cand_val: int, source_id: str) -> Tuple[int, int]:
        step = self._call_step
        if key in self.tentative_store:
            cur_cand, cur_cnt, sources, _ = self.tentative_store[key]
            self.tentative_store.move_to_end(key)
            if cur_cand == cand_val:
                sources.add(source_id)
                new_cnt = cur_cnt + 1
                self.tentative_store[key] = (cand_val, new_cnt, sources, step)
                return new_cnt, len(sources)
            if cur_cnt <= 1:
                sources = {source_id}
                self.tentative_store[key] = (cand_val, 1, sources, step)
                return 1, 1
            return 0, len(sources)
        if len(self.tentative_store) >= self.max_tentative:
            victim_key = None
            victim_score = None
            for k, (_, cnt, srcs, l_step) in self.tentative_store.items():
                is_fresh = 0 if (step - l_step) >= 8 else 1
                score = (is_fresh, len(srcs), min(cnt, 4))
                if victim_score is None or score < victim_score:
                    victim_score = score
                    victim_key = k
                    if score == (0, 1, 1):
                        break
            if victim_score is not None and victim_score[0] == 1 and victim_score[2] >= 2:
                return 0, 0
            if victim_key is not None:
                del self.tentative_store[victim_key]
        self.tentative_store[key] = (int(cand_val), 1, {source_id}, step)
        return 1, 1

    def update(self, x: torch.Tensor, y: torch.Tensor, source_id: str = "default") -> None:
        self._call_step += 1
        B, L = x.shape
        if L < 2:
            return
        keys_list = (
            (((x[:, :-1].detach().cpu().to(torch.int64) & 0xFFFF) << 16) | (x[:, 1:].detach().cpu().to(torch.int64) & 0xFFFF)).tolist()
        )
        y_list = y.detach().cpu().tolist()

        n_cached_hits = 0
        n_cached_contra = 0
        for kb, yb in zip(keys_list, y_list):
            for t_idx, key in enumerate(kb):
                entry = self.primary_store.get(key)
                if entry is not None:
                    n_cached_hits += 1
                    if yb[t_idx + 1] != entry[0]:
                        n_cached_contra += 1
        high_contra_batch = (n_cached_hits >= 8) and (n_cached_contra >= 0.30 * n_cached_hits)
        flips_this_call = 0

        for kb, yb in zip(keys_list, y_list):
            for t_idx, key in enumerate(kb):
                tgt = yb[t_idx + 1]
                entry = self.primary_store.get(key)
                if entry is None:
                    if not (high_contra_batch and flips_this_call == 0 and len(self.primary_store) >= self.max_primary):
                        self._put_primary(key, tgt, 1)
                else:
                    cur_y, prior_cnt = entry
                    if tgt == cur_y:
                        if not (high_contra_batch and flips_this_call == 0):
                            self._put_primary(key, cur_y, prior_cnt + 1)
                        if key in self.tentative_store:
                            del self.tentative_store[key]
                    else:
                        evid_cnt, n_src = self._record_tentative(key, tgt, source_id)
                        req_cnt = max(self.min_flip_count, int(math.ceil(prior_cnt * self.flip_threshold_ratio)))
                        if evid_cnt >= req_cnt and n_src >= self.min_sources_for_flip:
                            self._put_primary(key, tgt, max(2, evid_cnt))
                            if key in self.tentative_store:
                                del self.tentative_store[key]
                            flips_this_call += 1

    @torch.no_grad()
    def predict_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated log-probs [B, L, vocab_total] for fair Acc + NLL evaluation."""
        B, L = x.shape
        dev = x.device
        # Generous smoothed uniform prior over active domain tokens [vocab_start .. vocab_start + vocab_size_active)
        probs = torch.full((B, L, self.vocab_total), 1e-6, dtype=torch.float32, device=dev)
        active_mass = (1.0 - 1e-6 * float(self.vocab_total - self.vocab_size_active)) / float(self.vocab_size_active)
        probs[:, :, self.vocab_start : self.vocab_start + self.vocab_size_active] = active_mass

        x_list = x.detach().cpu().tolist()
        hits_b, hits_t, hits_y, hits_w = [], [], [], []
        for b, xb in enumerate(x_list):
            # Position 0 fallback: copy token
            tok0 = xb[0]
            if 0 <= tok0 < self.vocab_total:
                hits_b.append(b)
                hits_t.append(0)
                hits_y.append(tok0)
                hits_w.append(0.85)
            for t in range(1, L):
                key = ((xb[t - 1] & 0xFFFF) << 16) | (xb[t] & 0xFFFF)
                entry = self.primary_store.get(key)
                if entry is not None:
                    val, cnt = entry
                    if 0 <= val < self.vocab_total:
                        w = float(cnt) / (1.0 + float(cnt))
                        hits_b.append(b)
                        hits_t.append(t)
                        hits_y.append(val)
                        hits_w.append(min(w, 0.95))

        if hits_b:
            hb = torch.tensor(hits_b, dtype=torch.long, device=dev)
            ht = torch.tensor(hits_t, dtype=torch.long, device=dev)
            hy = torch.tensor(hits_y, dtype=torch.long, device=dev)
            hw = torch.tensor(hits_w, dtype=torch.float32, device=dev).unsqueeze(-1)
            sel = probs[hb, ht] * (1.0 - hw)
            sel.scatter_add_(-1, hy.unsqueeze(-1), hw)
            probs[hb, ht] = sel

        return torch.log(probs.clamp_min(1e-7))

    def storage_bytes(self) -> int:
        return (
            len(self.primary_store) * self.BYTES_PER_PRIMARY
            + len(self.tentative_store) * self.BYTES_PER_TENTATIVE
        )
