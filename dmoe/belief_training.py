"""
Experimental Training Algorithms for Continual Belief & Meaning Learning.

Implements the three research ideas and matched baselines for `belief_training_lab_v1`:
  1. Idea 1 (Concept-Keyed Wake-Chunk-Sleep Triage):
     Sparse concept bottleneck + SIGReg + ConceptBeliefStore with multi-source burst
     filtering, context-conditional belief splitting, Mechanism P ripple demotion,
     RHO-style reducible-error chunked fast updates (LaCT + Elastic anchor), and sleep.
  2. Idea 2 (Equivalence-and-Contradiction Learning):
     Cross-view equivalence attraction on canonical expression latents stabilized by
     SIGReg (LeJEPA Epps-Pulley), contradiction context separation, and concept readout.
  3. Idea 3 (Inference-Before-Plasticity + BONG Precision Synapses):
     - MuPCProspectiveNetwork: strictly zero `.backward()` calls; prospective local
       activity equilibration + local energy-gradient updates + BONG diagonal precision.
     - BONGPrecisionOptimizer + ContinualBackprop unit utility recycling.
"""

from __future__ import annotations

from collections import OrderedDict
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.rope import RotaryEmbedding


K_CONCEPTS = 16
NUM_OPS = 4
SYM_OFFSET = 4
Q_VAL = 20
Q_ATTR = 21
Q_PROP = 22
VOCAB_SIZE = 24
SEQ_LEN = 6


def _random_derangement_from(base_perm: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Return a random permutation of [0..K-1] differing from base_perm at every index."""
    n = int(base_perm.numel())
    p = torch.randperm(n, generator=gen)
    for i in range(n):
        if int(p[i].item()) == int(base_perm[i].item()):
            j = (i + 1) % n
            if int(p[j].item()) == int(base_perm[i].item()) or int(p[i].item()) == int(base_perm[j].item()):
                j = (i + 2) % n
            tmp = int(p[i].item())
            p[i] = p[j]
            p[j] = tmp
    # Guaranteed fallback if any fixed point remains
    for i in range(n):
        if int(p[i].item()) == int(base_perm[i].item()):
            p[i] = (base_perm[i] + 1) % n
    return p


class BeliefMeaningTask:
    """
    Zero-header, shared-vocabulary continual belief & meaning environment.

    Input sequence [6]: [op_tok in 0..3, a_tok in 4..19, b_tok in 4..19,
                         c1_tok in 4..19, c2_tok in 4..19, q_tok in {20, 21, 22}]
    Target class y in [0..15] representing canonical concept / attribute / property.
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        gen = torch.Generator().manual_seed(self.seed * 911 + 42)

        self.phi = torch.randperm(K_CONCEPTS, generator=gen)
        self.inv_phi = torch.argsort(self.phi)

        self.g_base = torch.randperm(K_CONCEPTS, generator=gen)
        self.g_alt = _random_derangement_from(self.g_base, gen)
        self.h_init = torch.randperm(K_CONCEPTS, generator=gen)
        self.h_rev = _random_derangement_from(self.h_init, gen)

        # Disjoint concept splits: 4 context-split, 4 law-revised, 8 static
        c_perm = torch.randperm(K_CONCEPTS, generator=gen)
        self.s_split = c_perm[0:4].tolist()
        self.s_rev = c_perm[4:8].tolist()
        self.s_static = c_perm[8:16].tolist()
        self.s_split_mask = torch.zeros(K_CONCEPTS, dtype=torch.bool)
        self.s_split_mask[self.s_split] = True
        self.s_rev_mask = torch.zeros(K_CONCEPTS, dtype=torch.bool)
        self.s_rev_mask[self.s_rev] = True

        # Stratified 75/25 pair split (192 seen, 64 heldout)
        self.seen_mask = torch.ones(K_CONCEPTS, K_CONCEPTS, dtype=torch.bool)
        self.seen_pairs: List[Tuple[int, int]] = []
        self.heldout_pairs: List[Tuple[int, int]] = []
        for a in range(K_CONCEPTS):
            row_perm = torch.randperm(K_CONCEPTS, generator=gen)
            for b in row_perm[:4].tolist():
                self.seen_mask[a, b] = False
            for b in range(K_CONCEPTS):
                if self.seen_mask[a, b]:
                    self.seen_pairs.append((a, b))
                else:
                    self.heldout_pairs.append((a, b))

        # Precompute canonical meaning table [4, 16, 16]: 4 shifted algebraic views over shared phi
        ga = self.phi.unsqueeze(1).expand(K_CONCEPTS, K_CONCEPTS)
        gb = self.phi.unsqueeze(0).expand(K_CONCEPTS, K_CONCEPTS)
        op_shifts = [0, 5, 11, 14]
        self.val_table = torch.stack(
            [self.inv_phi[(ga + 3 * gb + s) % K_CONCEPTS] for s in op_shifts],
            dim=0,
        )  # [4, 16, 16]

        # Precompute pair tensors for fast vectorized sampling
        self.seen_a = torch.tensor([p[0] for p in self.seen_pairs], dtype=torch.long)
        self.seen_b = torch.tensor([p[1] for p in self.seen_pairs], dtype=torch.long)
        self.held_a = torch.tensor([p[0] for p in self.heldout_pairs], dtype=torch.long)
        self.held_b = torch.tensor([p[1] for p in self.heldout_pairs], dtype=torch.long)

    def ctx_regime(self, c1: torch.Tensor, c2: torch.Tensor) -> torch.Tensor:
        """Compositional phi-parity XOR check: exact 50/50 balanced, zero single-token leakage."""
        b1 = self.phi[c1] >= (K_CONCEPTS // 2)
        b2 = self.phi[c2] >= (K_CONCEPTS // 2)
        return (b1 ^ b2).long()

    def _compute_targets(
        self,
        ops: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        q_toks: torch.Tensor,
        truth_stage: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = self.val_table[ops, a, b]
        ctx = self.ctx_regime(c1, c2)

        attr_y = self.g_base[m].clone()
        if truth_stage in ("p2", "p3"):
            split_hit = self.s_split_mask[m] & (ctx == 1)
            attr_y = torch.where(split_hit, self.g_alt[m], attr_y)

        prop_y = self.h_init[m].clone()
        if truth_stage == "p3":
            rev_hit = self.s_rev_mask[m]
            prop_y = torch.where(rev_hit, self.h_rev[m], prop_y)

        y = m.clone()
        y = torch.where(q_toks == Q_ATTR, attr_y, y)
        y = torch.where(q_toks == Q_PROP, prop_y, y)

        if truth_stage == "pb":
            corrupt = (q_toks == Q_ATTR) | (q_toks == Q_PROP)
            y = torch.where(corrupt, (y + 5) % K_CONCEPTS, y)

        return y, m, ctx

    def sample_stream_batch(
        self,
        phase: str,
        batch_size: int,
        gen: torch.Generator,
        confirm_frac: float = 0.25,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample a training batch for `phase` in {'p1', 'pb', 'p2', 'p3', 'joint_oracle'}.
        Returns (x [B, 6], y [B]) without exposing ground-truth m or ctx_regime.
        """
        B = int(batch_size)
        if phase == "p1":
            q_choices = torch.tensor([Q_VAL, Q_VAL, Q_ATTR, Q_PROP], dtype=torch.long)
            q_toks = q_choices[torch.randint(0, q_choices.numel(), (B,), generator=gen)]
            ops_val = torch.randint(0, 4, (B,), generator=gen)
            ops_other = torch.randint(0, 3, (B,), generator=gen)
            ops = torch.where(q_toks == Q_VAL, ops_val, ops_other)
            p_idx = torch.randint(0, self.seen_a.numel(), (B,), generator=gen)
            a = self.seen_a[p_idx]
            b = self.seen_b[p_idx]
            c1 = torch.randint(0, K_CONCEPTS, (B,), generator=gen)
            c2 = torch.randint(0, K_CONCEPTS, (B,), generator=gen)
            y, _, _ = self._compute_targets(ops, a, b, c1, c2, q_toks, "p1")

        elif phase == "pb":
            ops = torch.randint(0, 2, (B,), generator=gen)
            q_choices = torch.tensor([Q_ATTR, Q_PROP], dtype=torch.long)
            q_toks = q_choices[torch.randint(0, 2, (B,), generator=gen)]
            p_idx = torch.randint(0, self.seen_a.numel(), (B,), generator=gen)
            a = self.seen_a[p_idx]
            b = self.seen_b[p_idx]
            c1 = torch.randint(0, K_CONCEPTS, (B,), generator=gen)
            c2 = torch.randint(0, K_CONCEPTS, (B,), generator=gen)
            y, _, _ = self._compute_targets(ops, a, b, c1, c2, q_toks, "pb")

        elif phase == "p2":
            # Mix focused S_split ATTR updates (op in {0,1}) with confirmation VAL/PROP/ATTR samples
            n_focus = int(round(B * (1.0 - float(confirm_frac))))
            n_conf = B - n_focus
            ops_f, a_f, b_f = self._sample_filtered_seen(n_focus, [0, 1], self.s_split, gen)
            c1_f = torch.randint(0, K_CONCEPTS, (n_focus,), generator=gen)
            c2_f = torch.randint(0, K_CONCEPTS, (n_focus,), generator=gen)
            q_f = torch.full((n_focus,), Q_ATTR, dtype=torch.long)

            q_choices = torch.tensor([Q_VAL, Q_ATTR, Q_PROP], dtype=torch.long)
            q_c = q_choices[torch.randint(0, 3, (n_conf,), generator=gen)]
            ops_c = torch.where(
                q_c == Q_VAL,
                torch.randint(0, 4, (n_conf,), generator=gen),
                torch.randint(0, 3, (n_conf,), generator=gen),
            )
            p_idx = torch.randint(0, self.seen_a.numel(), (n_conf,), generator=gen)
            a_c = self.seen_a[p_idx]
            b_c = self.seen_b[p_idx]
            c1_c = torch.randint(0, K_CONCEPTS, (n_conf,), generator=gen)
            c2_c = torch.randint(0, K_CONCEPTS, (n_conf,), generator=gen)

            ops = torch.cat([ops_f, ops_c], dim=0)
            a = torch.cat([a_f, a_c], dim=0)
            b = torch.cat([b_f, b_c], dim=0)
            c1 = torch.cat([c1_f, c1_c], dim=0)
            c2 = torch.cat([c2_f, c2_c], dim=0)
            q_toks = torch.cat([q_f, q_c], dim=0)
            perm = torch.randperm(B, generator=gen)
            ops, a, b, c1, c2, q_toks = ops[perm], a[perm], b[perm], c1[perm], c2[perm], q_toks[perm]
            y, _, _ = self._compute_targets(ops, a, b, c1, c2, q_toks, "p2")

        elif phase == "p3":
            # Focused S_rev PROP updates strictly on op=0 ONLY; confirmations avoid S_rev PROP on op!=0
            n_focus = int(round(B * (1.0 - float(confirm_frac))))
            n_conf = B - n_focus
            ops_f, a_f, b_f = self._sample_filtered_seen(n_focus, [0], self.s_rev, gen)
            c1_f = torch.randint(0, K_CONCEPTS, (n_focus,), generator=gen)
            c2_f = torch.randint(0, K_CONCEPTS, (n_focus,), generator=gen)
            q_f = torch.full((n_focus,), Q_PROP, dtype=torch.long)

            # Confirmations: VAL (all ops), ATTR (op 0..2), PROP (only non-S_rev concepts so op=1..3 S_rev never leaks)
            q_choices = torch.tensor([Q_VAL, Q_ATTR, Q_PROP], dtype=torch.long)
            q_c = q_choices[torch.randint(0, 3, (n_conf,), generator=gen)]
            ops_c = torch.where(
                q_c == Q_VAL,
                torch.randint(0, 4, (n_conf,), generator=gen),
                torch.randint(0, 3, (n_conf,), generator=gen),
            )
            p_idx = torch.randint(0, self.seen_a.numel(), (n_conf,), generator=gen)
            a_c = self.seen_a[p_idx]
            b_c = self.seen_b[p_idx]
            # Replace any accidental (q_c==PROP & m in S_rev & ops_c != 0) with non-S_rev
            m_c = self.val_table[ops_c, a_c, b_c]
            bad = (q_c == Q_PROP) & self.s_rev_mask[m_c] & (ops_c != 0)
            if bad.any():
                non_rev = [i for i in range(K_CONCEPTS) if i not in self.s_rev]
                n_bad = int(bad.sum().item())
                _, ra, rb = self._sample_filtered_seen(n_bad, [0, 1, 2], non_rev, gen)
                a_c[bad] = ra
                b_c[bad] = rb
            c1_c = torch.randint(0, K_CONCEPTS, (n_conf,), generator=gen)
            c2_c = torch.randint(0, K_CONCEPTS, (n_conf,), generator=gen)

            ops = torch.cat([ops_f, ops_c], dim=0)
            a = torch.cat([a_f, a_c], dim=0)
            b = torch.cat([b_f, b_c], dim=0)
            c1 = torch.cat([c1_f, c1_c], dim=0)
            c2 = torch.cat([c2_f, c2_c], dim=0)
            q_toks = torch.cat([q_f, q_c], dim=0)
            perm = torch.randperm(B, generator=gen)
            ops, a, b, c1, c2, q_toks = ops[perm], a[perm], b[perm], c1[perm], c2[perm], q_toks[perm]
            y, _, _ = self._compute_targets(ops, a, b, c1, c2, q_toks, "p3")

        elif phase == "joint_oracle":
            q_choices = torch.tensor([Q_VAL, Q_ATTR, Q_PROP], dtype=torch.long)
            q_toks = q_choices[torch.randint(0, 3, (B,), generator=gen)]
            ops = torch.randint(0, 4, (B,), generator=gen)
            p_idx = torch.randint(0, self.seen_a.numel(), (B,), generator=gen)
            a = self.seen_a[p_idx]
            b = self.seen_b[p_idx]
            c1 = torch.randint(0, K_CONCEPTS, (B,), generator=gen)
            c2 = torch.randint(0, K_CONCEPTS, (B,), generator=gen)
            y, _, _ = self._compute_targets(ops, a, b, c1, c2, q_toks, "p3")
        else:
            raise ValueError(f"Unknown phase: {phase}")

        x = torch.stack([ops, a + SYM_OFFSET, b + SYM_OFFSET, c1 + SYM_OFFSET, c2 + SYM_OFFSET, q_toks], dim=-1)
        return x, y

    def _sample_filtered_seen(
        self,
        n: int,
        op_list: List[int],
        concept_list: List[int],
        gen: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n <= 0:
            z = torch.zeros(0, dtype=torch.long)
            return z, z, z
        allowed_c = torch.zeros(K_CONCEPTS, dtype=torch.bool)
        allowed_c[concept_list] = True
        valid_ops, valid_a, valid_b = [], [], []
        for op in op_list:
            m_grid = self.val_table[op]
            mask = self.seen_mask & allowed_c[m_grid]
            idx = mask.nonzero(as_tuple=False)
            valid_ops.append(torch.full((idx.size(0),), op, dtype=torch.long))
            valid_a.append(idx[:, 0])
            valid_b.append(idx[:, 1])
        all_ops = torch.cat(valid_ops, dim=0)
        all_a = torch.cat(valid_a, dim=0)
        all_b = torch.cat(valid_b, dim=0)
        pick = torch.randint(0, all_ops.numel(), (n,), generator=gen)
        return all_ops[pick], all_a[pick], all_b[pick]

    def sample_eval_subset(
        self,
        q_tok: int,
        truth_stage: str,
        op_list: List[int],
        pair_split: str,
        concept_subset: Optional[List[int]],
        ctx_filter: Optional[int],
        n_samples: int,
        gen: torch.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build exact held-out evaluation batch for a specified subset."""
        pool_a = self.seen_a if pair_split == "seen" else self.held_a
        pool_b = self.seen_b if pair_split == "seen" else self.held_b
        base_mask = self.seen_mask if pair_split == "seen" else (~self.seen_mask)
        allowed_c = torch.ones(K_CONCEPTS, dtype=torch.bool)
        if concept_subset is not None:
            allowed_c.zero_()
            allowed_c[concept_subset] = True

        valid_ops, valid_a, valid_b = [], [], []
        for op in op_list:
            m_grid = self.val_table[op]
            mask = base_mask & allowed_c[m_grid]
            idx = mask.nonzero(as_tuple=False)
            if idx.numel() > 0:
                valid_ops.append(torch.full((idx.size(0),), op, dtype=torch.long))
                valid_a.append(idx[:, 0])
                valid_b.append(idx[:, 1])
        if not valid_ops:
            pick = torch.randint(0, pool_a.numel(), (n_samples,), generator=gen)
            ops = torch.tensor(op_list, dtype=torch.long)[torch.randint(0, len(op_list), (n_samples,), generator=gen)]
            a, b = pool_a[pick], pool_b[pick]
        else:
            all_ops = torch.cat(valid_ops, dim=0)
            all_a = torch.cat(valid_a, dim=0)
            all_b = torch.cat(valid_b, dim=0)
            pick = torch.randint(0, all_ops.numel(), (n_samples,), generator=gen)
            ops, a, b = all_ops[pick], all_a[pick], all_b[pick]

        c1 = torch.randint(0, K_CONCEPTS, (n_samples,), generator=gen)
        if ctx_filter is None:
            c2 = torch.randint(0, K_CONCEPTS, (n_samples,), generator=gen)
        else:
            # For each c1, exactly 8 of 16 c2 values yield ctx_regime == ctx_filter
            all_c2 = torch.arange(K_CONCEPTS, dtype=torch.long)
            b_c1 = (self.phi[c1] >= (K_CONCEPTS // 2)).unsqueeze(1)  # [N, 1]
            b_c2 = (self.phi[all_c2] >= (K_CONCEPTS // 2)).unsqueeze(0)  # [1, 16]
            valid_c2 = (b_c1 ^ b_c2).long() == int(ctx_filter)
            # Sample uniformly among the 8 matching c2 choices per row
            probs_c2 = valid_c2.float()
            c2 = torch.multinomial(probs_c2, num_samples=1, generator=gen).squeeze(-1)

        q_toks = torch.full((n_samples,), int(q_tok), dtype=torch.long)
        y, _, _ = self._compute_targets(ops, a, b, c1, c2, q_toks, truth_stage)
        x = torch.stack([ops, a + SYM_OFFSET, b + SYM_OFFSET, c1 + SYM_OFFSET, c2 + SYM_OFFSET, q_toks], dim=-1)
        return x, y


class _CausalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
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
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = x + self.out_proj(ctx)
        x = x + self.ffn(self.norm2(x))
        return x


class ConceptBeliefBackbone(nn.Module):
    """
    Strictly causal RoPE Transformer backbone exposing causal latents:
      - z_expr: position 2 latent (strictly causal over [op, a, b])
      - z_ctx:  position 4 latent (strictly causal over [op, a, b, c1, c2])
      - h_query: position 5 latent (causal over full sequence ending at q_tok)
    Also provides an optional concept-anchored readout + fast context adapter.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 192,
        vocab_size: int = VOCAB_SIZE,
        num_classes: int = K_CONCEPTS,
        use_concept_readout: bool = False,
        d_ctx: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ctx = d_ctx
        self.use_concept_readout = bool(use_concept_readout)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([_CausalBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

        self.expr_proj = nn.Linear(d_model, d_model)
        self.ctx_proj = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_ctx),
        )
        self.fast_adapter = nn.Sequential(
            nn.Linear(d_model + d_ctx, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model + d_ctx),
        )
        # Zero-init fast_adapter out layer so initial behavior equals base backbone
        nn.init.zeros_(self.fast_adapter[2].weight)
        nn.init.zeros_(self.fast_adapter[2].bias)

        if self.use_concept_readout:
            self.val_head = nn.Linear(d_model, num_classes)
            self.attr_head = nn.Linear(d_model + d_ctx, num_classes)
            self.prop_head = nn.Linear(d_model, num_classes)
        else:
            self.out_head = nn.Linear(d_model, num_classes)

    def _build_ctx_feat(self, h0: torch.Tensor, h_n: torch.Tensor) -> torch.Tensor:
        h0_ctx = F.layer_norm(h0[:, 3:5, :], (self.d_model,))
        return torch.cat([h_n[:, 4, :], h0_ctx[:, 0, :], h0_ctx[:, 1, :]], dim=-1)

    def encode_trunk(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute strictly causal Transformer trunk latents once per batch."""
        h0 = self.tok_emb(x)
        h = h0
        for blk in self.blocks:
            h = blk(h)
        h_n = self.final_norm(h)
        z_expr = self.expr_proj(h_n[:, 2, :])
        ctx_feat = self._build_ctx_feat(h0, h_n)
        return {"z_expr": z_expr, "ctx_feat": ctx_feat, "h_query": h_n[:, 5, :], "hidden": h_n}

    def forward_fast_heads(
        self,
        q_tok: torch.Tensor,
        z_expr: torch.Tensor,
        ctx_feat: torch.Tensor,
        use_fast_adapter: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Run fast context/readout heads on precomputed causal trunk latents."""
        z_ctx = self.ctx_proj(ctx_feat)
        z_expr_fast = z_expr
        z_ctx_fast = z_ctx
        if use_fast_adapter:
            delta = self.fast_adapter(torch.cat([z_expr, z_ctx], dim=-1))
            z_expr_fast = z_expr + delta[:, : self.d_model]
            z_ctx_fast = z_ctx + delta[:, self.d_model :]
        if self.use_concept_readout:
            logits_val = self.val_head(z_expr)
            logits_attr = self.attr_head(torch.cat([z_expr_fast, z_ctx_fast], dim=-1))
            logits_prop = self.prop_head(z_expr)
            logits = torch.where((q_tok == Q_ATTR).unsqueeze(-1), logits_attr, logits_val)
            logits = torch.where((q_tok == Q_PROP).unsqueeze(-1), logits_prop, logits)
        else:
            logits = self.out_head(z_expr_fast)
        return {"logits": logits, "z_expr": z_expr, "z_ctx": z_ctx_fast}

    def forward(
        self,
        x: torch.Tensor,
        use_fast_adapter: bool = False,
    ) -> Dict[str, torch.Tensor]:
        h0 = self.tok_emb(x)
        h = h0
        for blk in self.blocks:
            h = blk(h)
        h_n = self.final_norm(h)

        z_expr = self.expr_proj(h_n[:, 2, :])
        z_ctx = self.ctx_proj(self._build_ctx_feat(h0, h_n))
        h_query = h_n[:, 5, :]
        z_expr_fast = z_expr
        z_ctx_fast = z_ctx
        if use_fast_adapter:
            delta = self.fast_adapter(torch.cat([z_expr, z_ctx], dim=-1))
            z_expr_fast = z_expr + delta[:, : self.d_model]
            z_ctx_fast = z_ctx + delta[:, self.d_model :]
            h_query = h_query + delta[:, : self.d_model]

        if self.use_concept_readout:
            q_tok = x[:, 5]
            logits_val = self.val_head(z_expr)
            logits_attr = self.attr_head(torch.cat([z_expr_fast, z_ctx_fast], dim=-1))
            logits_prop = self.prop_head(z_expr)
            logits = torch.where((q_tok == Q_ATTR).unsqueeze(-1), logits_attr, logits_val)
            logits = torch.where((q_tok == Q_PROP).unsqueeze(-1), logits_prop, logits)
        else:
            logits = self.out_head(h_query)

        return {
            "logits": logits,
            "z_expr": z_expr,
            "z_ctx": z_ctx_fast,
            "h_query": h_query,
            "hidden": h_n,
        }


def sigreg_loss(z: torch.Tensor, num_slices: int = 128, knots: int = 17) -> torch.Tensor:
    """
    LeJEPA Sketched Isotropic Gaussian Regularization (Balestriero & LeCun, arXiv:2511.08544).
    Enforces isotropic N(0, I) over random 1D projections via the Epps-Pulley test.
    """
    if z.size(0) < 4:
        return z.new_tensor(0.0)
    D = z.size(-1)
    u = F.normalize(torch.randn(num_slices, D, device=z.device, dtype=z.dtype), dim=1)
    proj = torch.matmul(u, z.t())  # [M, N]
    t = torch.linspace(0.0, 3.0, steps=knots, device=z.device, dtype=z.dtype)
    target_cf = torch.exp(-0.5 * (t ** 2)).view(1, -1)
    angle = proj.unsqueeze(2) * t.view(1, 1, -1)
    emp_real = torch.cos(angle).mean(dim=1)
    emp_imag = torch.sin(angle).mean(dim=1)
    return ((emp_real - target_cf) ** 2 + emp_imag ** 2).mean()


def equivalence_and_contradiction_losses(
    z_expr: torch.Tensor,
    z_ctx: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    shuffle_equiv: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Idea 2 losses computed purely within-batch from observed (x, y) pairs:
      - L_equiv: for VAL samples sharing the same observed y, attract their z_expr vectors.
        If `shuffle_equiv=True`, randomly permute y across VAL samples (X-47 control).
      - L_contra: when two ATTR samples share close z_expr (same concept) yet have different y,
        push their z_ctx vectors apart (contradiction-driven context splitting).
    """
    q_tok = x[:, 5]
    val_mask = q_tok == Q_VAL
    l_equiv = z_expr.new_tensor(0.0)
    if int(val_mask.sum().item()) >= 4:
        zv = F.normalize(z_expr[val_mask], dim=-1)
        yv = y[val_mask]
        if shuffle_equiv:
            perm = torch.randperm(yv.numel(), device=yv.device)
            yv = yv[perm]
        same_y = (yv.unsqueeze(1) == yv.unsqueeze(0)).float()
        same_y.fill_diagonal_(0.0)
        n_pos = same_y.sum()
        if float(n_pos.item()) > 0.0:
            cos_sim = zv @ zv.t()
            # Pull positive pairs close in cosine similarity and push negative pairs slightly orthogonal
            diff_y = 1.0 - same_y
            diff_y.fill_diagonal_(0.0)
            pos_loss = ((1.0 - cos_sim) * same_y).sum() / n_pos
            n_neg = diff_y.sum().clamp_min(1.0)
            neg_loss = (F.relu(cos_sim).pow(2) * diff_y).sum() / n_neg
            l_equiv = pos_loss + 0.5 * neg_loss

    attr_mask = q_tok == Q_ATTR
    l_contra = z_ctx.new_tensor(0.0)
    if int(attr_mask.sum().item()) >= 4:
        ze_a = F.normalize(z_expr[attr_mask].detach(), dim=-1)
        zc_a = F.normalize(z_ctx[attr_mask], dim=-1)
        ya = y[attr_mask]
        same_concept = (ze_a @ ze_a.t()) > 0.75
        diff_target = ya.unsqueeze(1) != ya.unsqueeze(0)
        contra_pairs = (same_concept & diff_target).float()
        same_pairs = (same_concept & (~diff_target)).float()
        same_pairs.fill_diagonal_(0.0)
        n_c = contra_pairs.sum()
        if float(n_c.item()) > 0.0:
            ctx_sim = zc_a @ zc_a.t()
            push = ((ctx_sim + 0.5).pow(2) * contra_pairs).sum() / n_c
            n_s = same_pairs.sum().clamp_min(1.0)
            pull = ((1.0 - ctx_sim) * same_pairs).sum() / n_s
            l_contra = push + 0.5 * pull

    return l_equiv, l_contra


class SparseConceptQuantizer(nn.Module):
    """
    Self-trained sparse concept dictionary mapping continuous z_expr into discrete
    concept IDs without labels. Tracks usage to revive unused atoms.
    """

    def __init__(self, d_model: int, num_atoms: int = 48):
        super().__init__()
        self.d_model = d_model
        self.num_atoms = num_atoms
        atoms = F.normalize(torch.randn(num_atoms, d_model), dim=-1)
        self.atoms = nn.Parameter(atoms)
        self.register_buffer("atom_usage", torch.zeros(num_atoms, dtype=torch.float32))
        self.register_buffer("mean_z_norm", torch.tensor(1.0, dtype=torch.float32))

    @torch.no_grad()
    def assign(self, z_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_norm = F.normalize(z_expr.detach().float(), dim=-1)
        a_norm = F.normalize(self.atoms.detach().float(), dim=-1)
        sims = z_norm @ a_norm.t()
        best_sim, ids = sims.max(dim=-1)
        return ids, best_sim

    @torch.no_grad()
    def scaled_atoms(self) -> torch.Tensor:
        return F.normalize(self.atoms.detach().float(), dim=-1) * self.mean_z_norm

    def quantize_loss(self, z_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_norm = F.normalize(z_expr, dim=-1)
        a_norm = F.normalize(self.atoms, dim=-1)
        sims = z_norm @ a_norm.t()
        ids = sims.detach().argmax(dim=-1)
        matched = a_norm[ids]
        # Pull atom toward encoder embeddings (detached z) + commit encoder lightly to atom
        loss_dict = F.mse_loss(matched, z_norm.detach())
        loss_commit = 0.25 * F.mse_loss(z_norm, matched.detach())
        # Diversity loss encouraging uniform atom usage
        probs = F.softmax(sims * 8.0, dim=-1).mean(dim=0)
        loss_div = (probs * torch.log(probs.clamp_min(1e-7))).sum() + math.log(self.num_atoms)
        with torch.no_grad():
            counts = torch.bincount(ids, minlength=self.num_atoms).float()
            self.atom_usage.mul_(0.95).add_(0.05 * counts)
            batch_norm = z_expr.detach().float().norm(dim=-1).mean()
            self.mean_z_norm.mul_(0.95).add_(0.05 * batch_norm)
        return loss_dict + loss_commit + 0.01 * loss_div, ids

    @torch.no_grad()
    def revive_dead_atoms(self, z_expr: torch.Tensor, min_usage: float = 0.02) -> int:
        dead = (self.atom_usage < min_usage).nonzero(as_tuple=False).squeeze(-1)
        if dead.numel() == 0 or z_expr.size(0) == 0:
            return 0
        z_norm = F.normalize(z_expr.detach(), dim=-1)
        n_rep = min(int(dead.numel()), z_norm.size(0))
        self.atoms.data[dead[:n_rep]] = z_norm[:n_rep]
        self.atom_usage[dead[:n_rep]] = 0.5
        return n_rep


class ConceptBeliefStore:
    """
    Concept-keyed Bayesian belief store with:
      - Key = (concept_id, q_role), NOT raw surface tokens.
      - Multi-source burst immunity (TentativeBeliefStore logic).
      - Context-conditional belief splitting ('apple -> red' AND 'apple -> green').
      - Mechanism P selective demotion of stale explained entries on genuine law flips.
      - Strict honest memory byte accounting including split prototypes and probe buffers.
    """

    BYTES_PER_PRIMARY = 21       # 8B key + 4B val + 4B count + 4B mismatch + 1B explained flag
    BYTES_PER_TENTATIVE = 52
    BYTES_PER_SPLIT_PROTO = 144  # 2 x 16-float prototypes (128B) + 2 x (val, cnt) (16B)
    BYTES_PER_CTX_OBS = 68       # 16-float vec (64B) + 4B label

    def __init__(
        self,
        d_ctx: int = 16,
        max_total_bytes: int = 16384,
        min_sources_for_flip: int = 2,
        min_flip_count: int = 3,
        max_prior_count: int = 24,
        flip_threshold_ratio: float = 0.45,
        kappa: float = 2.5,
        min_fuse_count: int = 2,
        max_fuse_weight: float = 0.94,
        ctx_split_thresh: float = 0.78,
        enable_ctx_split: bool = True,
        enable_mech_p: bool = True,
    ):
        self.d_ctx = int(d_ctx)
        self.max_total_bytes = int(max_total_bytes)
        self.min_sources_for_flip = int(min_sources_for_flip)
        self.min_flip_count = int(min_flip_count)
        self.max_prior_count = int(max_prior_count)
        self.flip_threshold_ratio = float(flip_threshold_ratio)
        self.kappa = float(kappa)
        self.min_fuse_count = int(min_fuse_count)
        self.max_fuse_weight = float(max_fuse_weight)
        self.ctx_split_thresh = float(ctx_split_thresh)
        self.enable_ctx_split = bool(enable_ctx_split)
        self.enable_mech_p = bool(enable_mech_p)

        # key = (concept_id, q_role) -> dict(val, count, explained)
        self.primary: OrderedDict[Tuple[int, int], Dict[str, Any]] = OrderedDict()
        # key -> dict(proto0, val0, cnt0, proto1, val1, cnt1)
        self.splits: OrderedDict[Tuple[int, int], Dict[str, Any]] = OrderedDict()
        # key -> dict(cand_val, cnt, sources, last_step)
        self.tentative: OrderedDict[Tuple[int, int], Dict[str, Any]] = OrderedDict()
        # key -> list of (z_ctx_list, y_int) capped at 12 while contested
        self.ctx_window: OrderedDict[Tuple[int, int], List[Tuple[List[float], int]]] = OrderedDict()

        self._step = 0
        self.flips_executed = 0
        self.splits_executed = 0
        self.bursts_rejected = 0
        self.p_demotions = 0
        self.confirmed_since_rev: set = set()
        self.rev_event_open = False
        self.rev_sweeps_empty = 0
        self._contra_sources: List[Tuple[int, str]] = []

    def storage_bytes(self) -> int:
        obs_count = sum(len(v) for v in self.ctx_window.values())
        return (
            len(self.primary) * self.BYTES_PER_PRIMARY
            + len(self.splits) * self.BYTES_PER_SPLIT_PROTO
            + len(self.tentative) * self.BYTES_PER_TENTATIVE
            + obs_count * self.BYTES_PER_CTX_OBS
        )

    def _enforce_budget(self) -> None:
        while self.storage_bytes() > self.max_total_bytes:
            if self.ctx_window:
                self.ctx_window.popitem(last=False)
            elif self.tentative:
                self.tentative.popitem(last=False)
            elif self.splits:
                self.splits.popitem(last=False)
            elif self.primary:
                self.primary.popitem(last=False)
            else:
                break

    def _record_ctx_obs(self, key: Tuple[int, int], zc: List[float], tgt: int) -> None:
        if not self.enable_ctx_split:
            return
        buf = self.ctx_window.get(key)
        if buf is None:
            buf = []
            self.ctx_window[key] = buf
        else:
            self.ctx_window.move_to_end(key)
        buf.append((zc, int(tgt)))
        if len(buf) > 16:
            buf.pop(0)

    def _try_context_split(self, key: Tuple[int, int], old_y: int, new_y: int) -> bool:
        if not self.enable_ctx_split or key not in self.ctx_window:
            return False
        buf = self.ctx_window[key]
        old_vecs = [v for v, lbl in buf if lbl == old_y]
        new_vecs = [v for v, lbl in buf if lbl == new_y]
        if len(old_vecs) < 4 or len(new_vecs) < 4:
            return False

        t_old = torch.tensor(old_vecs, dtype=torch.float32)
        t_new = torch.tensor(new_vecs, dtype=torch.float32)
        t_old_n = F.normalize(t_old, dim=-1)
        t_new_n = F.normalize(t_new, dim=-1)
        c0_full = F.normalize(t_old_n.mean(dim=0), dim=0)
        c1_full = F.normalize(t_new_n.mean(dim=0), dim=0)
        if (1.0 - float((c0_full @ c1_full).item())) < 0.25:
            return False
        # Leave-one-out cosine prototype separability
        correct = 0
        total = t_old_n.size(0) + t_new_n.size(0)
        for i in range(t_old_n.size(0)):
            other_old = torch.cat([t_old_n[:i], t_old_n[i + 1 :]], dim=0).mean(dim=0)
            c0 = F.normalize(other_old, dim=0)
            c1 = c1_full
            if float((t_old_n[i] @ c0) > (t_old_n[i] @ c1)):
                correct += 1
        for i in range(t_new_n.size(0)):
            other_new = torch.cat([t_new_n[:i], t_new_n[i + 1 :]], dim=0).mean(dim=0)
            c0 = c0_full
            c1 = F.normalize(other_new, dim=0)
            if float((t_new_n[i] @ c1) > (t_new_n[i] @ c0)):
                correct += 1

        if (correct / max(1, total)) < max(0.80, self.ctx_split_thresh):
            return False

        p0 = c0_full.tolist()
        p1 = c1_full.tolist()
        self.splits[key] = {
            "proto0": p0,
            "val0": int(old_y),
            "cnt0": len(old_vecs),
            "proto1": p1,
            "val1": int(new_y),
            "cnt1": len(new_vecs),
            "purity": 1.0,
        }
        self.splits_executed += 1
        self.tentative.pop(key, None)
        self.ctx_window.pop(key, None)
        self._enforce_budget()
        return True

    def observe_batch(
        self,
        concept_ids: torch.Tensor,
        q_toks: torch.Tensor,
        z_ctx: torch.Tensor,
        targets: torch.Tensor,
        neural_logits: torch.Tensor,
        source_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Process one batch through Tier-0 / Tier-1 store logic (zero backprop):
          - Confirmations strengthen prior count and set Mechanism-P `explained` bit.
          - Single-source contradictions are quarantined.
          - Multi-source contradictions either split by context or flip & open P event.
        Returns summary dict including `needs_chunk_mask` [B] bool for Tier-2 surprise.
        """
        self._step += 1
        cids = concept_ids.detach().cpu().tolist()
        qs = q_toks.detach().cpu().tolist()
        ys = targets.detach().cpu().tolist()
        zcs = F.normalize(z_ctx.detach().float(), dim=-1).cpu().tolist()
        n_preds = neural_logits.detach().argmax(dim=-1).cpu().tolist()
        n_probs = F.softmax(neural_logits.detach().float(), dim=-1).cpu()

        B = len(cids)
        # Pre-check high contradiction rate on established primary entries (burst shield)
        hits, contra = 0, 0
        for i in range(B):
            key = (int(cids[i]), int(qs[i]))
            if key in self.splits:
                hits += 1
                sp = self.splits[key]
                v = torch.tensor(zcs[i], dtype=torch.float32)
                s0 = float((v * torch.tensor(sp["proto0"])).sum())
                s1 = float((v * torch.tensor(sp["proto1"])).sum())
                pred_sp = sp["val0"] if s0 >= s1 else sp["val1"]
                if ys[i] != pred_sp:
                    contra += 1
            elif key in self.primary:
                hits += 1
                if ys[i] != self.primary[key]["val"]:
                    contra += 1
        high_contra = hits >= 6 and contra >= 0.30 * hits
        severe_burst = hits >= 6 and contra >= 0.55 * hits
        if severe_burst:
            self._contra_sources.append((self._step, source_id))
        self._contra_sources = [(st_i, sr_i) for st_i, sr_i in self._contra_sources if (self._step - st_i) <= 8]
        active_contra_srcs = len({sr_i for _, sr_i in self._contra_sources})
        consecutive_same_src = (
            len(self._contra_sources) >= 2
            and self._contra_sources[-1][1] == self._contra_sources[-2][1]
            and (self._step - self._contra_sources[-2][0]) <= 3
        )
        single_source_burst = severe_burst and (active_contra_srcs < self.min_sources_for_flip or consecutive_same_src)

        flips_here = 0
        needs_chunk = [False] * B

        for i in range(B):
            key = (int(cids[i]), int(qs[i]))
            tgt = int(ys[i])
            zc = zcs[i]
            np_i = int(n_preds[i])
            p_tgt = float(n_probs[i, tgt].item())

            if key in self.splits:
                sp = self.splits[key]
                v = torch.tensor(zc, dtype=torch.float32)
                s0 = float((v * torch.tensor(sp["proto0"])).sum())
                s1 = float((v * torch.tensor(sp["proto1"])).sum())
                use0 = s0 >= s1
                branch_val = sp["val0"] if use0 else sp["val1"]
                if tgt == branch_val:
                    sp["purity"] = 0.90 * float(sp.get("purity", 1.0)) + 0.10
                    if use0:
                        sp["cnt0"] = min(sp["cnt0"] + 1, self.max_prior_count)
                        new_p = F.normalize(0.85 * torch.tensor(sp["proto0"]) + 0.15 * v, dim=0)
                        sp["proto0"] = new_p.tolist()
                    else:
                        sp["cnt1"] = min(sp["cnt1"] + 1, self.max_prior_count)
                        new_p = F.normalize(0.85 * torch.tensor(sp["proto1"]) + 0.15 * v, dim=0)
                        sp["proto1"] = new_p.tolist()
                elif tgt == (sp["val1"] if use0 else sp["val0"]):
                    # Context separator predicted opposite branch -> decay split purity unless confirmed
                    sp["purity"] = float(sp.get("purity", 1.0)) * 0.85
                    needs_chunk[i] = True
                    if tgt == sp["val0"]:
                        sp["proto0"] = F.normalize(0.7 * torch.tensor(sp["proto0"]) + 0.3 * v, dim=0).tolist()
                    else:
                        sp["proto1"] = F.normalize(0.7 * torch.tensor(sp["proto1"]) + 0.3 * v, dim=0).tolist()
                else:
                    if not high_contra:
                        sp["purity"] = float(sp.get("purity", 1.0)) * 0.80
                    self.bursts_rejected += 1
                continue

            entry = self.primary.get(key)
            if entry is None:
                if high_contra and flips_here == 0:
                    self.bursts_rejected += 1
                else:
                    self.primary[key] = {
                        "val": tgt,
                        "count": 1,
                        "purity": 1.0,
                        "explained": 1 if np_i == tgt else 0,
                    }
                    if (not single_source_burst) and p_tgt < 0.45:
                        needs_chunk[i] = True
            else:
                cur_y = entry["val"]
                prior_cnt = entry["count"]
                purity = float(entry.get("purity", 1.0))
                if tgt == cur_y:
                    if key in self.tentative:
                        old_tent = self.tentative[key]
                        if len(old_tent["sources"]) >= self.min_sources_for_flip and old_tent["cnt"] >= 2:
                            # Same atom simultaneously receiving cur_y AND multi-source alt label => colliding atom
                            purity = purity * 0.78
                        elif (self._step - old_tent["last_step"]) > 4 and len(old_tent["sources"]) < self.min_sources_for_flip:
                            self.tentative.pop(key, None)
                    if not (high_contra and flips_here == 0):
                        entry["count"] = min(prior_cnt + 1, self.max_prior_count)
                        if key not in self.tentative:
                            purity = 0.90 * purity + 0.10
                        entry["purity"] = purity
                        entry["explained"] = 1 if np_i == cur_y else 0
                        self.primary.move_to_end(key)
                    if key in self.ctx_window:
                        self._record_ctx_obs(key, zc, tgt)
                    if self.rev_event_open:
                        self.confirmed_since_rev.add(key)
                    if (not single_source_burst) and entry["count"] >= 2 and p_tgt < 0.55:
                        needs_chunk[i] = True
                else:
                    # Contradiction! Record in tentative store & context window
                    if purity >= 0.55:
                        self._record_ctx_obs(key, zc, tgt)
                    tent = self.tentative.get(key)
                    if tent is None:
                        self.tentative[key] = {
                            "cand_val": tgt,
                            "cnt": 1,
                            "sources": {source_id},
                            "last_step": self._step,
                        }
                        evid_cnt, n_src = 1, 1
                    else:
                        if (self._step - tent["last_step"]) > 4 and len(tent["sources"]) < self.min_sources_for_flip:
                            tent = None
                        if tent is None:
                            self.tentative[key] = {
                                "cand_val": tgt,
                                "cnt": 1,
                                "sources": {source_id},
                                "last_step": self._step,
                            }
                            evid_cnt, n_src = 1, 1
                        else:
                            self.tentative.move_to_end(key)
                            if tent["cand_val"] == tgt:
                                tent["cnt"] += 1
                                tent["sources"].add(source_id)
                                tent["last_step"] = self._step
                                evid_cnt, n_src = tent["cnt"], len(tent["sources"])
                            elif tent["cnt"] <= 1 or (
                                len(tent["sources"]) < self.min_sources_for_flip and source_id not in tent["sources"]
                            ):
                                if not high_contra and prior_cnt <= 3:
                                    purity = purity * 0.75
                                tent["cand_val"] = tgt
                                tent["cnt"] = 1
                                tent["sources"] = {source_id}
                                tent["last_step"] = self._step
                                evid_cnt, n_src = 1, 1
                            else:
                                if not high_contra:
                                    purity = purity * 0.80
                                evid_cnt, n_src = 0, len(tent["sources"])
                    entry["purity"] = purity

                    req_cnt = max(self.min_flip_count, int(math.ceil(prior_cnt * self.flip_threshold_ratio)))
                    req_srcs = max(self.min_sources_for_flip, 3) if (severe_burst and flips_here == 0) else self.min_sources_for_flip
                    if (not single_source_burst) and evid_cnt >= req_cnt and n_src >= req_srcs:
                        # Persistent multi-source contradiction: split by context OR flip law
                        did_split = self._try_context_split(key, cur_y, tgt)
                        if did_split:
                            needs_chunk[i] = True
                        else:
                            # If old_y still keeps appearing alongside new_y, keep chunking context until separable
                            buf = self.ctx_window.get(key, [])
                            old_recent = sum(1 for _, lbl in buf if lbl == cur_y)
                            new_recent = sum(1 for _, lbl in buf if lbl == tgt)
                            if self.enable_ctx_split and old_recent >= 3 and new_recent >= 3:
                                needs_chunk[i] = True
                            else:
                                # Genuine law revision vs ambiguous impure atom collision
                                self.primary[key] = {
                                    "val": tgt,
                                    "count": max(2, evid_cnt),
                                    "purity": max(0.85, purity) if (purity >= 0.65 or prior_cnt >= 4) else purity * 0.85,
                                    "explained": 1 if np_i == tgt else 0,
                                }
                                self.primary.move_to_end(key)
                                self.tentative.pop(key, None)
                                self.ctx_window.pop(key, None)
                                if purity >= 0.65 or prior_cnt >= 4:
                                    self.flips_executed += 1
                                    flips_here += 1
                                    needs_chunk[i] = True
                                    if self.enable_mech_p and not self.rev_event_open:
                                        self.rev_event_open = True
                                        self.confirmed_since_rev = {key}
                                        self.rev_sweeps_empty = 0
                                    elif self.rev_event_open:
                                        self.confirmed_since_rev.add(key)
                    else:
                        self.bursts_rejected += 1
                        # Only feed Tier-2 chunk if multiple sources actually corroborated THIS candidate
                        if (not single_source_burst) and evid_cnt >= 2 and n_src >= self.min_sources_for_flip and (purity >= 0.65 or p_tgt < 0.45):
                            needs_chunk[i] = True

        self._enforce_budget()
        return {
            "needs_chunk_mask": torch.tensor(needs_chunk, dtype=torch.bool, device=targets.device),
            "flips_here": flips_here,
        }

    def run_mechanism_p_sweep(
        self,
        concept_prototypes: torch.Tensor,
        model: ConceptBeliefBackbone,
        device: torch.device,
        tau_derive: float = 0.65,
        use_fast_adapter: bool = True,
    ) -> int:
        """
        Mechanism P selective demotion on primary entries whose concept prototype
        is now confidently contradicted by the updated neural lane.
        """
        if not self.enable_mech_p or not self.rev_event_open:
            return 0
        demoted = 0
        to_check = [
            (k, ent)
            for k, ent in self.primary.items()
            if ent["explained"] == 1 and k not in self.confirmed_since_rev
        ]
        if not to_check:
            self.rev_sweeps_empty += 1
            if self.rev_sweeps_empty >= 2:
                self.rev_event_open = False
            return 0

        with torch.no_grad():
            for (cid, q_tok), ent in to_check:
                if cid >= concept_prototypes.size(0):
                    continue
                ze_base = concept_prototypes[cid : cid + 1].to(device)
                ze_fast = ze_base
                zc_fast = torch.zeros(1, model.d_ctx, device=device)
                if use_fast_adapter:
                    delta = model.fast_adapter(torch.cat([ze_base, zc_fast], dim=-1))
                    ze_fast = ze_base + delta[:, : model.d_model]
                    zc_fast = zc_fast + delta[:, model.d_model :]
                if model.use_concept_readout:
                    if q_tok == Q_PROP:
                        logits = model.prop_head(ze_base)
                    elif q_tok == Q_ATTR:
                        logits = model.attr_head(torch.cat([ze_fast, zc_fast], dim=-1))
                    else:
                        logits = model.val_head(ze_base)
                else:
                    continue
                probs = F.softmax(logits.float(), dim=-1)
                top_p, top_y = probs.max(dim=-1)
                if int(top_y.item()) != ent["val"] and float(top_p.item()) >= tau_derive:
                    ent["count"] = 1  # below min_fuse_count => fusion weight drops to 0
                    demoted += 1
                    self.p_demotions += 1

        if demoted == 0:
            self.rev_sweeps_empty += 1
            if self.rev_sweeps_empty >= 2:
                self.rev_event_open = False
        else:
            self.rev_sweeps_empty = 0
        return demoted

    @torch.no_grad()
    def fuse_logits(
        self,
        concept_ids: torch.Tensor,
        q_toks: torch.Tensor,
        z_ctx: torch.Tensor,
        neural_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Calibrated Dirichlet-mixture logit fusion over neural lane + concept belief store."""
        B, C = neural_logits.shape
        log_probs_n = F.log_softmax(neural_logits.float(), dim=-1)
        if not self.primary and not self.splits:
            return log_probs_n.to(dtype=neural_logits.dtype)

        probs_n = log_probs_n.exp()
        cids = concept_ids.detach().cpu().tolist()
        qs = q_toks.detach().cpu().tolist()
        zcs = F.normalize(z_ctx.detach().float(), dim=-1).cpu()

        hits_b, hits_y, hits_w = [], [], []
        for i in range(B):
            key = (int(cids[i]), int(qs[i]))
            if key in self.splits:
                sp = self.splits[key]
                s0 = float((zcs[i] * torch.tensor(sp["proto0"])).sum())
                s1 = float((zcs[i] * torch.tensor(sp["proto1"])).sum())
                val, cnt = (sp["val0"], sp["cnt0"]) if s0 >= s1 else (sp["val1"], sp["cnt1"])
                sp_purity = float(sp.get("purity", 1.0))
                sp_gate = max(0.0, (sp_purity - 0.72) / 0.28)
                if cnt >= self.min_fuse_count and sp_gate > 0.0 and 0 <= val < C:
                    w = min(float(cnt) / (self.kappa + float(cnt)), self.max_fuse_weight) * sp_gate
                    hits_b.append(i)
                    hits_y.append(val)
                    hits_w.append(w)
            elif key in self.primary:
                ent = self.primary[key]
                val, cnt = ent["val"], ent["count"]
                purity = float(ent.get("purity", 1.0))
                purity_gate = max(0.0, (purity - 0.72) / 0.28)
                if cnt >= self.min_fuse_count and purity_gate > 0.0 and 0 <= val < C:
                    w = min(float(cnt) / (self.kappa + float(cnt)), self.max_fuse_weight) * purity_gate
                    hits_b.append(i)
                    hits_y.append(val)
                    hits_w.append(w)

        if not hits_b:
            return log_probs_n.to(dtype=neural_logits.dtype)

        dev = neural_logits.device
        hb = torch.tensor(hits_b, dtype=torch.long, device=dev)
        hy = torch.tensor(hits_y, dtype=torch.long, device=dev)
        hw = torch.tensor(hits_w, dtype=torch.float32, device=dev).unsqueeze(-1)
        probs_fused = probs_n.clone()
        sel = probs_n[hb] * (1.0 - hw)
        sel.scatter_add_(-1, hy.unsqueeze(-1), hw)
        probs_fused[hb] = sel
        return torch.log(probs_fused.clamp_min(1e-7)).to(dtype=neural_logits.dtype)


class BONGPrecisionOptimizer:
    """
    Bayesian Online Natural Gradient (Jones, Chang, & Murphy, NeurIPS 2024)
    diagonal-precision step with decoupled Gaussian prior shrinkage and
    Continual Backprop unit utility recycling (Dohare, Mahmood, & Sutton, Nature 2024).
    """

    def __init__(
        self,
        params: List[nn.Parameter],
        lr: float = 3e-3,
        init_prec: float = 0.02,
        gamma: float = 0.996,
        damping: float = 0.01,
        max_prec: float = 100.0,
        grad_clip: float = 2.0,
        prior_wd: float = 0.03,
        min_prec: float = 0.005,
        low_wd_param_ids: Optional[set] = None,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = float(lr)
        self.init_prec = float(init_prec)
        self.gamma = float(gamma)
        self.damping = float(damping)
        self.max_prec = float(max_prec)
        self.min_prec = float(min_prec)
        self.grad_clip = float(grad_clip)
        self.prior_wd = float(prior_wd)
        self.low_wd_param_ids = set(low_wd_param_ids or set())
        self.precisions: List[torch.Tensor] = [
            torch.full_like(p.data, self.init_prec) for p in self.params
        ]
        self.steps = 0

    @torch.no_grad()
    def step(self, grads: Optional[List[Optional[torch.Tensor]]] = None) -> None:
        self.steps += 1
        g_list = grads if grads is not None else [p.grad for p in self.params]
        for p, prec, g in zip(self.params, self.precisions, g_list):
            if g is None:
                continue
            gn = g.detach()
            g_norm = float(gn.norm().item())
            if g_norm > self.grad_clip:
                gn = gn * (self.grad_clip / max(g_norm, 1e-8))
            prec.mul_(self.gamma).add_(gn * gn)
            prec.clamp_(min=self.min_prec, max=self.max_prec)
            wd_use = min(self.prior_wd, 0.01) if id(p) in self.low_wd_param_ids else self.prior_wd
            if wd_use > 0.0 and p.ndim >= 2:
                p.data.mul_(1.0 - self.lr * wd_use)
            p.data.add_(gn / (prec + self.damping), alpha=-self.lr)

    @torch.no_grad()
    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def precision_bytes(self) -> int:
        return sum(pr.numel() * 4 for pr in self.precisions)


@torch.no_grad()
def continual_backprop_recycle(
    layer_in: nn.Linear,
    layer_out: nn.Linear,
    utility_buf: torch.Tensor,
    act_vals: torch.Tensor,
    replace_rate: float = 0.02,
    maturity_threshold: float = 0.05,
    bong_opt: Optional[BONGPrecisionOptimizer] = None,
) -> int:
    """
    Track hidden unit contribution utility and reinitialize dead/low-utility units
    (Dohare et al., Nature 2024) while resetting their precision if using BONG.
    """
    u_inst = act_vals.detach().abs().mean(dim=tuple(range(act_vals.ndim - 1))) * layer_out.weight.detach().abs().mean(dim=0)
    utility_buf.mul_(0.98).add_(0.02 * u_inst)
    n_units = utility_buf.numel()
    k = max(1, int(round(n_units * replace_rate)))
    low_vals, low_idx = torch.topk(utility_buf, k=k, largest=False)
    recycled = 0
    for idx, val in zip(low_idx.tolist(), low_vals.tolist()):
        if val < maturity_threshold:
            bound = 1.0 / math.sqrt(max(1, layer_in.in_features))
            layer_in.weight.data[idx].uniform_(-bound, bound)
            if layer_in.bias is not None:
                layer_in.bias.data[idx] = 0.0
            layer_out.weight.data[:, idx] = 0.0
            utility_buf[idx] = float(utility_buf.median().item()) + 1e-3
            if bong_opt is not None:
                for p, pr in zip(bong_opt.params, bong_opt.precisions):
                    if p is layer_in.weight:
                        pr[idx].fill_(bong_opt.init_prec)
                    elif p is layer_out.weight:
                        pr[:, idx].fill_(bong_opt.init_prec)
            recycled += 1
    return recycled


class MuPCProspectiveNetwork(nn.Module):
    """
    Strictly backprop-free (.backward() NEVER called) Predictive Coding network
    with uPC depth scaling (1/sqrt(L)), weak-nudged prospective activity relaxation,
    exact z_0 restoring energy, local Jacobian backward passes, and BONG or AdamW updates.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        seq_len: int = SEQ_LEN,
        d_emb: int = 20,
        d_pc: int = 120,
        num_layers: int = 3,
        num_classes: int = K_CONCEPTS,
        k_train: int = 10,
        eta_z: float = 0.12,
        beta_out: float = 0.10,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.d_emb = d_emb
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.k_train = int(k_train)
        self.eta_z = float(eta_z)
        self.beta_out = float(beta_out)
        self.depth_scale = 1.0 / math.sqrt(float(num_layers))

        self.emb = nn.Parameter(torch.randn(vocab_size, d_emb) * 0.15)
        self.in_proj = nn.Parameter(torch.randn(d_pc, seq_len * d_emb) * (1.0 / math.sqrt(seq_len * d_emb)))
        self.in_bias = nn.Parameter(torch.zeros(d_pc))

        self.w_hid = nn.ParameterList([
            nn.Parameter(torch.randn(d_pc, d_pc) * (1.0 / math.sqrt(d_pc)))
            for _ in range(num_layers)
        ])
        self.b_hid = nn.ParameterList([
            nn.Parameter(torch.zeros(d_pc))
            for _ in range(num_layers)
        ])
        self.w_out_layer = nn.ParameterList([
            nn.Parameter(torch.randn(d_pc, d_pc) * (self.depth_scale / math.sqrt(d_pc)))
            for _ in range(num_layers)
        ])
        self.readout = nn.Parameter(torch.randn(num_classes, d_pc) * (1.0 / math.sqrt(d_pc)))
        self.readout_bias = nn.Parameter(torch.zeros(num_classes))

    def ordered_params(self) -> List[nn.Parameter]:
        return [
            self.emb,
            self.in_proj,
            self.in_bias,
            *list(self.w_hid),
            *list(self.b_hid),
            *list(self.w_out_layer),
            self.readout,
            self.readout_bias,
        ]

    def _embed_flat(self, x: torch.Tensor) -> torch.Tensor:
        return self.emb[x].reshape(x.size(0), self.seq_len * self.d_emb)

    def _layer_pred(self, l_idx: int, z_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pre = z_prev @ self.w_hid[l_idx].t() + self.b_hid[l_idx]
        act = F.relu(pre)
        d_act = (pre > 0).to(dtype=pre.dtype)
        pred = z_prev + (act @ self.w_out_layer[l_idx].t()) * self.depth_scale
        return pred, act, d_act

    @torch.no_grad()
    def equilibrate(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Forward predictive-coding sweep initializing the zero-error state z_0..z_L (no y)."""
        x_flat = self._embed_flat(x)
        pre0 = x_flat @ self.in_proj.t() + self.in_bias
        z0 = F.relu(pre0)
        dz0 = (pre0 > 0).to(dtype=pre0.dtype)
        zs = [z0]
        cur = z0
        for l_idx in range(self.num_layers):
            pred, _, _ = self._layer_pred(l_idx, cur)
            zs.append(pred)
            cur = pred
        logits = zs[-1] @ self.readout.t() + self.readout_bias
        return logits, zs, x_flat, dz0

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits, zs, _, _ = self.equilibrate(x)
        return {"logits": logits, "z_top": zs[-1]}

    @torch.no_grad()
    def local_update_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        opt: Any,
        skip_threshold: Optional[float] = None,
        ema_ref_loss: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Weak-nudged prospective target relaxation + exact local energy-gradient updates.
        NEVER calls `.backward()`.
        """
        logits_eq, zs_eq, x_flat, dz0 = self.equilibrate(x)
        ce_val = float(F.cross_entropy(logits_eq, y).item())
        if skip_threshold is not None and ema_ref_loss is not None:
            if (ce_val - ema_ref_loss) < skip_threshold and ce_val < 0.35:
                return {"ce_loss": ce_val, "skipped": 1.0}

        z0_init = zs_eq[0]
        zs = [z.clone() for z in zs_eq]
        B = float(x.size(0))
        y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
        beta = self.beta_out

        for _ in range(self.k_train):
            logits_t = zs[-1] @ self.readout.t() + self.readout_bias
            eo = beta * (F.softmax(logits_t, dim=-1) - y_onehot)
            pred_top, _, _ = self._layer_pred(self.num_layers - 1, zs[self.num_layers - 1])
            zs[-1] = zs[-1] - self.eta_z * ((eo @ self.readout) + (zs[-1] - pred_top))

            for l_idx in range(self.num_layers - 1, 0, -1):
                pred_prev, _, _ = self._layer_pred(l_idx - 1, zs[l_idx - 1])
                err_prev = zs[l_idx] - pred_prev
                pred_next, _, da_next = self._layer_pred(l_idx, zs[l_idx])
                err_next = zs[l_idx + 1] - pred_next
                j_t_e = err_next + (((err_next @ self.w_out_layer[l_idx]) * da_next) @ self.w_hid[l_idx]) * self.depth_scale
                zs[l_idx] = zs[l_idx] - self.eta_z * (err_prev - j_t_e)

            pred_0, _, da_0 = self._layer_pred(0, zs[0])
            err_0 = zs[1] - pred_0
            j0_t_e = err_0 + (((err_0 @ self.w_out_layer[0]) * da_0) @ self.w_hid[0]) * self.depth_scale
            zs[0] = zs[0] - self.eta_z * ((zs[0] - z0_init) - j0_t_e)

        # Exact local parameter gradients evaluated at weak-nudged prospective states zs*
        logits_f = zs[-1] @ self.readout.t() + self.readout_bias
        e_out_f = (F.softmax(logits_f, dim=-1) - y_onehot) / B
        g_readout = e_out_f.t() @ zs[-1]
        g_readout_b = e_out_f.sum(dim=0)

        scale = 1.0 / beta
        g_w_out, g_w_hid, g_b_hid = [], [], []
        for l_idx in range(self.num_layers):
            pred, act, da = self._layer_pred(l_idx, zs[l_idx])
            e_l = scale * (pred - zs[l_idx + 1]) / B
            g_wo = (e_l.t() @ act) * self.depth_scale
            e_act = ((e_l @ self.w_out_layer[l_idx]) * self.depth_scale) * da
            g_wh = e_act.t() @ zs[l_idx]
            g_bh = e_act.sum(dim=0)
            g_w_out.append(g_wo)
            g_w_hid.append(g_wh)
            g_b_hid.append(g_bh)

        e_z0 = (scale * (z0_init - zs[0]) / B) * dz0
        g_in_proj = e_z0.t() @ x_flat
        g_in_bias = e_z0.sum(dim=0)
        g_x_flat = (e_z0 @ self.in_proj).reshape(x.size(0), self.seq_len, self.d_emb)
        g_emb = torch.zeros_like(self.emb)
        g_emb.index_add_(0, x.reshape(-1), g_x_flat.reshape(-1, self.d_emb))

        param_grads = [
            g_emb, g_in_proj, g_in_bias,
            *g_w_hid, *g_b_hid, *g_w_out,
            g_readout, g_readout_b,
        ]
        if isinstance(opt, BONGPrecisionOptimizer):
            opt.step(param_grads)
        else:
            for p, g in zip(self.ordered_params(), param_grads):
                p.grad = g
            opt.step()
            opt.zero_grad()

        return {"ce_loss": ce_val, "skipped": 0.0}


class NullSpaceGuard:
    """
    Online activation covariance tracker with relaxed AlphaEdit+-style null-space
    gradient projection: P = (1 - relax) * (I - U U^T) + relax * I.
    """

    def __init__(self, model: ConceptBeliefBackbone, energy_thresh: float = 0.96, relax: float = 0.10):
        self.model = model
        self.energy_thresh = float(energy_thresh)
        self.relax = float(relax)
        self.covs: Dict[str, torch.Tensor] = {}
        self.p_null: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
        self.tracking = True
        self._register()

    def _targets(self) -> Dict[str, nn.Linear]:
        mods: Dict[str, nn.Linear] = {}
        for idx, blk in enumerate(self.model.blocks):
            mods[f"b{idx}.out"] = blk.out_proj
            mods[f"b{idx}.ff0"] = blk.ffn[0]
            mods[f"b{idx}.ff2"] = blk.ffn[2]
        mods["expr_proj"] = self.model.expr_proj
        return mods

    def _register(self) -> None:
        for name, mod in self._targets().items():
            def _hook(_m, args, _out, k=name):
                if not self.tracking or not _m.training:
                    return
                with torch.no_grad():
                    x = args[0].reshape(-1, args[0].size(-1)).detach().float()
                    cov = x.t() @ x
                    if k not in self.covs:
                        self.covs[k] = cov
                    else:
                        self.covs[k] = 0.99 * self.covs[k] + cov
            self.hooks.append(mod.register_forward_hook(_hook))

    @torch.no_grad()
    def consolidate(self) -> None:
        for name, cov in self.covs.items():
            evals, evecs = torch.linalg.eigh(cov.float())
            idx = torch.argsort(evals, descending=True)
            evals = evals[idx].clamp_min(0.0)
            evecs = evecs[:, idx]
            tot = float(evals.sum().item())
            if tot <= 1e-8:
                continue
            cum = torch.cumsum(evals, dim=0) / tot
            r = int((cum < self.energy_thresh).sum().item()) + 1
            r = min(r, max(1, int(0.65 * cov.size(0))))
            U = evecs[:, :r]
            eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            self.p_null[name] = (1.0 - self.relax) * (eye - U @ U.t()) + self.relax * eye

    @torch.no_grad()
    def project_grads(self) -> None:
        if not self.p_null:
            return
        for name, mod in self._targets().items():
            if name in self.p_null and mod.weight.grad is not None:
                p = self.p_null[name].to(mod.weight.grad.device, mod.weight.grad.dtype)
                mod.weight.grad.copy_(mod.weight.grad @ p)

    def remove(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class BudgetedSurfaceDict:
    """
    Pure non-neural counting dictionary baseline keyed strictly on raw 6-token surface
    sequences (op, a, b, c1, c2, q), bounded by `max_total_bytes`.
    """

    BYTES_PER_PRIMARY = 16
    BYTES_PER_TENTATIVE = 52

    def __init__(
        self,
        max_total_bytes: int = 16384,
        min_sources_for_flip: int = 2,
        min_flip_count: int = 3,
        max_prior_count: int = 24,
        flip_threshold_ratio: float = 0.45,
        num_classes: int = K_CONCEPTS,
    ):
        self.max_total_bytes = int(max_total_bytes)
        self.max_primary = max(4, int(self.max_total_bytes * 0.65) // self.BYTES_PER_PRIMARY)
        rem = max(0, self.max_total_bytes - self.max_primary * self.BYTES_PER_PRIMARY)
        self.max_tentative = max(2, rem // self.BYTES_PER_TENTATIVE)
        self.min_sources_for_flip = int(min_sources_for_flip)
        self.min_flip_count = int(min_flip_count)
        self.max_prior_count = int(max_prior_count)
        self.flip_threshold_ratio = float(flip_threshold_ratio)
        self.num_classes = int(num_classes)

        self.primary: OrderedDict[Tuple[int, ...], Tuple[int, int]] = OrderedDict()
        self.tentative: OrderedDict[Tuple[int, ...], Tuple[int, int, set, int]] = OrderedDict()
        self._step = 0

    def storage_bytes(self) -> int:
        return len(self.primary) * self.BYTES_PER_PRIMARY + len(self.tentative) * self.BYTES_PER_TENTATIVE

    def update(self, x: torch.Tensor, y: torch.Tensor, source_id: str = "default") -> None:
        self._step += 1
        xs = x.detach().cpu().tolist()
        ys = y.detach().cpu().tolist()
        hits, contra = 0, 0
        for xb, yb in zip(xs, ys):
            ent = self.primary.get(tuple(xb))
            if ent is not None:
                hits += 1
                if ent[0] != yb:
                    contra += 1
        high_contra = hits >= 6 and contra >= 0.30 * hits
        flips = 0
        for xb, yb in zip(xs, ys):
            k = tuple(xb)
            ent = self.primary.get(k)
            if ent is None:
                if not (high_contra and flips == 0 and len(self.primary) >= self.max_primary):
                    while len(self.primary) >= self.max_primary:
                        self.primary.popitem(last=False)
                    self.primary[k] = (int(yb), 1)
            else:
                cur_y, cnt = ent
                if yb == cur_y:
                    if not (high_contra and flips == 0):
                        self.primary.move_to_end(k)
                        self.primary[k] = (cur_y, min(cnt + 1, self.max_prior_count))
                    self.tentative.pop(k, None)
                else:
                    tent = self.tentative.get(k)
                    if tent is None:
                        while len(self.tentative) >= self.max_tentative:
                            self.tentative.popitem(last=False)
                        self.tentative[k] = (int(yb), 1, {source_id}, self._step)
                        evid, n_src = 1, 1
                    else:
                        c_val, c_cnt, srcs, _ = tent
                        self.tentative.move_to_end(k)
                        if c_val == yb:
                            srcs.add(source_id)
                            self.tentative[k] = (c_val, c_cnt + 1, srcs, self._step)
                            evid, n_src = c_cnt + 1, len(srcs)
                        elif c_cnt <= 1:
                            self.tentative[k] = (int(yb), 1, {source_id}, self._step)
                            evid, n_src = 1, 1
                        else:
                            evid, n_src = 0, len(srcs)
                    req = max(self.min_flip_count, int(math.ceil(cnt * self.flip_threshold_ratio)))
                    if evid >= req and n_src >= self.min_sources_for_flip:
                        self.primary.move_to_end(k)
                        self.primary[k] = (int(yb), max(2, evid))
                        self.tentative.pop(k, None)
                        flips += 1

    @torch.no_grad()
    def predict_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        xs = x.detach().cpu().tolist()
        dev = x.device
        probs = torch.full((len(xs), self.num_classes), 1.0 / self.num_classes, dtype=torch.float32, device=dev)
        hb, hy, hw = [], [], []
        for i, xb in enumerate(xs):
            ent = self.primary.get(tuple(xb))
            if ent is not None:
                val, cnt = ent
                w = min(float(cnt) / (2.0 + float(cnt)), 0.94)
                hb.append(i)
                hy.append(val)
                hw.append(w)
        if hb:
            t_b = torch.tensor(hb, dtype=torch.long, device=dev)
            t_y = torch.tensor(hy, dtype=torch.long, device=dev)
            t_w = torch.tensor(hw, dtype=torch.float32, device=dev).unsqueeze(-1)
            sel = probs[t_b] * (1.0 - t_w)
            sel.scatter_add_(-1, t_y.unsqueeze(-1), t_w)
            probs[t_b] = sel
        return torch.log(probs.clamp_min(1e-7))
