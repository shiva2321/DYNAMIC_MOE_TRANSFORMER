"""Task 16: DeltaExpert — low-rank parameter-efficient expert with copy-on-write fork.

An expert comprises:
- A SHARED FROZEN base MLP (MLPExpert)
- A lineage of frozen ancestor low-rank deltas: sum_{a in lineage} A_a B_a^T
- An active learnable low-rank delta: A B^T (rank r)
- A seen-class mask: classes never observed are masked to -1e9

Operations:
- fork(): copy-on-write child inheriting frozen snapshot of parent delta (function-preserving)
- rank_up(k): expand rank by k with zero-padding (function-preserving)
- add_block(): add zero-output residual delta block (function-preserving)
- flatten(r): truncated SVD compression of accumulated lineage deltas
- merge_siblings(other): merge sibling deltas via low-rank SVD averaging
- train_step(): standard training step with seen-class masking and optional replay / A-GEM
"""

import copy
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .experts import MLPExpert, agem_project

MASK_VALUE = -1e9


class DeltaBlock(nn.Module):
    """Low-rank delta for one residual block."""

    def __init__(self, d: int, hidden: int, rank: int, gen: torch.Generator) -> None:
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.rank = rank

        # w1 delta: hidden x d -> A1 (hidden, r), B1 (d, r)
        self.w1_A = nn.Parameter(torch.zeros(hidden, rank))
        self.w1_B = nn.Parameter(torch.randn(d, rank, generator=gen) / math.sqrt(rank))

        # w2 delta: d x hidden -> A2 (d, r), B2 (hidden, r)
        self.w2_A = nn.Parameter(torch.zeros(d, rank))
        self.w2_B = nn.Parameter(torch.randn(hidden, rank, generator=gen) / math.sqrt(rank))

    def forward_delta_w1(self, x: Tensor, lineage: List[Dict[str, Tensor]]) -> Tensor:
        # x is (B, d). Effective delta: x @ (sum_a B_a A_a^T + B A^T) = sum_a (x @ B_a) @ A_a^T + (x @ B) @ A^T
        out = (x @ self.w1_B) @ self.w1_A.t()
        for snap in lineage:
            if "w1_A" in snap and "w1_B" in snap:
                out = out + (x @ snap["w1_B"]) @ snap["w1_A"].t()
        return out

    def forward_delta_w2(self, x: Tensor, lineage: List[Dict[str, Tensor]]) -> Tensor:
        # x is (B, hidden)
        out = (x @ self.w2_B) @ self.w2_A.t()
        for snap in lineage:
            if "w2_A" in snap and "w2_B" in snap:
                out = out + (x @ snap["w2_B"]) @ snap["w2_A"].t()
        return out

    def snapshot(self) -> Dict[str, Tensor]:
        return {
            "w1_A": self.w1_A.detach().clone(),
            "w1_B": self.w1_B.detach().clone(),
            "w2_A": self.w2_A.detach().clone(),
            "w2_B": self.w2_B.detach().clone(),
        }


class DeltaExpert(nn.Module):
    """Low-rank delta expert sharing a frozen base MLP."""

    def __init__(
        self,
        base: MLPExpert,
        rank: int = 16,
        lineage: Optional[List[List[Dict[str, Tensor]]]] = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.d_in = base.d_in
        self.hidden = base.hidden
        self.n_classes = base.n_classes
        self.seed = seed

        # Freeze base parameters
        for p in self.base.parameters():
            p.requires_grad = False

        self.gen = torch.Generator().manual_seed(seed)

        # Delta blocks for each base block
        self.delta_blocks = nn.ModuleList([
            DeltaBlock(self.d_in, self.hidden, rank, self.gen)
            for _ in range(len(base.blocks))
        ])

        # Head delta: C x d -> Ahead (C, r), Bhead (d, r)
        self.head_A = nn.Parameter(torch.zeros(self.n_classes, rank))
        self.head_B = nn.Parameter(torch.randn(self.d_in, rank, generator=self.gen) / math.sqrt(rank))

        # Additional residual blocks added via add_block
        self.extra_blocks = nn.ModuleList()

        # Frozen lineage snapshots from ancestors
        # lineage is a list of snapshots, each containing block deltas + head deltas
        self.lineage: List[Dict[str, Tensor]] = []
        if lineage is not None:
            for snap in lineage:
                self.lineage.append({k: v.detach().clone() for k, v in snap.items()})

        # Seen-class mask
        self.register_buffer("seen", torch.zeros(self.n_classes, dtype=torch.bool))

    def mark_seen(self, y: Tensor) -> None:
        y = y.long()
        valid = (y >= 0) & (y < self.n_classes)
        if valid.any():
            self.seen[y[valid]] = True

    def num_params(self) -> int:
        """Count trainable parameters in own delta (excluding shared frozen base and lineage)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, u: Tensor) -> Tensor:
        h = u.float()

        # Pass through base residual blocks + delta blocks
        for i, (base_block, delta_block) in enumerate(zip(self.base.blocks, self.delta_blocks)):
            # ln
            h_ln = base_block.ln(h)
            # w1 + delta_w1
            block_lineage_snaps = [
                {"w1_A": snap[f"b{i}_w1_A"], "w1_B": snap[f"b{i}_w1_B"]}
                for snap in self.lineage if f"b{i}_w1_A" in snap
            ]
            w1_out = F.linear(h_ln, base_block.w1, base_block.b1) + delta_block.forward_delta_w1(h_ln, block_lineage_snaps)
            h_act = F.gelu(w1_out)

            # w2 + delta_w2
            w2_lineage_snaps = [
                {"w2_A": snap[f"b{i}_w2_A"], "w2_B": snap[f"b{i}_w2_B"]}
                for snap in self.lineage if f"b{i}_w2_A" in snap
            ]
            w2_out = F.linear(h_act, base_block.w2, base_block.b2) + delta_block.forward_delta_w2(h_act, w2_lineage_snaps)
            h = h + w2_out

        # Pass through extra blocks (if any)
        for extra in self.extra_blocks:
            h = extra(h)

        # Final LayerNorm
        h = self.base.norm(h)

        # Head + head delta
        head_lineage_snaps = [
            {"head_A": snap["head_A"], "head_B": snap["head_B"]}
            for snap in self.lineage if "head_A" in snap
        ]
        logits = F.linear(h, self.base.head.weight, self.base.head.bias) + (h @ self.head_B) @ self.head_A.t()
        for snap in head_lineage_snaps:
            logits = logits + (h @ snap["head_B"]) @ snap["head_A"].t()

        # Seen-class mask
        logits = logits.masked_fill(~self.seen, MASK_VALUE)
        return logits

    def proba(self, x: Tensor) -> Tensor:
        return F.softmax(self(x), dim=-1)

    def fork(self, seed: Optional[int] = None) -> "DeltaExpert":
        """Copy-on-write fork: snapshot parent delta into frozen lineage, fresh zero-init delta."""
        child_seed = seed if seed is not None else self.seed + 1
        snapshot: Dict[str, Tensor] = {}
        for i, db in enumerate(self.delta_blocks):
            snap_b = db.snapshot()
            snapshot[f"b{i}_w1_A"] = snap_b["w1_A"]
            snapshot[f"b{i}_w1_B"] = snap_b["w1_B"]
            snapshot[f"b{i}_w2_A"] = snap_b["w2_A"]
            snapshot[f"b{i}_w2_B"] = snap_b["w2_B"]
        snapshot["head_A"] = self.head_A.detach().clone()
        snapshot["head_B"] = self.head_B.detach().clone()

        new_lineage = [dict(s) for s in self.lineage] + [snapshot]
        child = DeltaExpert(base=self.base, rank=self.rank, lineage=new_lineage, seed=child_seed)
        child.seen = self.seen.clone()
        return child

    def rank_up(self, k: int) -> None:
        """Expand delta rank by k columns with zero padding (function-preserving)."""
        with torch.no_grad():
            for db in self.delta_blocks:
                # w1_A: (hidden, r) -> pad k columns of zeros
                new_w1_A = torch.cat([db.w1_A.data, torch.zeros(db.hidden, k, device=db.w1_A.device)], dim=1)
                new_w1_B = torch.cat([db.w1_B.data, torch.zeros(db.d, k, device=db.w1_B.device)], dim=1)
                db.w1_A = nn.Parameter(new_w1_A)
                db.w1_B = nn.Parameter(new_w1_B)

                new_w2_A = torch.cat([db.w2_A.data, torch.zeros(db.d, k, device=db.w2_A.device)], dim=1)
                new_w2_B = torch.cat([db.w2_B.data, torch.zeros(db.hidden, k, device=db.w2_B.device)], dim=1)
                db.w2_A = nn.Parameter(new_w2_A)
                db.w2_B = nn.Parameter(new_w2_B)
                db.rank += k

            new_head_A = torch.cat([self.head_A.data, torch.zeros(self.n_classes, k, device=self.head_A.device)], dim=1)
            new_head_B = torch.cat([self.head_B.data, torch.zeros(self.d_in, k, device=self.head_B.device)], dim=1)
            self.head_A = nn.Parameter(new_head_A)
            self.head_B = nn.Parameter(new_head_B)
            self.rank += k

    def add_block(self) -> None:
        """Add a zero-output residual block (function-preserving)."""
        # Identity block on residual stream: h + W2(gelu(W1 ln(h))) with W2=0
        class IdentityResBlock(nn.Module):
            def __init__(self, d: int, hidden: int, gen: torch.Generator):
                super().__init__()
                self.ln = nn.LayerNorm(d)
                self.w1 = nn.Parameter(torch.randn(hidden, d, generator=gen) / math.sqrt(d))
                self.b1 = nn.Parameter(torch.zeros(hidden))
                self.w2 = nn.Parameter(torch.zeros(d, hidden))
                self.b2 = nn.Parameter(torch.zeros(d))

            def forward(self, h: Tensor) -> Tensor:
                return h + F.linear(F.gelu(F.linear(self.ln(h), self.w1, self.b1)), self.w2, self.b2)

        self.extra_blocks.append(IdentityResBlock(self.d_in, self.hidden, self.gen))

    def flatten(self, r: Optional[int] = None) -> float:
        """Compress accumulated lineage deltas and own delta via truncated SVD.

        Returns maximum truncation error bound across layers.
        """
        r_target = r if r is not None else self.rank
        max_err = 0.0

        with torch.no_grad():
            for i, db in enumerate(self.delta_blocks):
                # Total delta w1
                delta_w1 = db.w1_A.data @ db.w1_B.data.t()
                for snap in self.lineage:
                    if f"b{i}_w1_A" in snap:
                        delta_w1 = delta_w1 + snap[f"b{i}_w1_A"] @ snap[f"b{i}_w1_B"].t()

                U, S, Vh = torch.linalg.svd(delta_w1, full_matrices=False)
                r_k = min(r_target, len(S))
                err = float(S[r_k].item()) if r_k < len(S) else 0.0
                max_err = max(max_err, err)

                sqrt_s = torch.diag(torch.sqrt(S[:r_k]))
                db.w1_A = nn.Parameter(U[:, :r_k] @ sqrt_s)
                db.w1_B = nn.Parameter(Vh[:r_k, :].t() @ sqrt_s)

                # Total delta w2
                delta_w2 = db.w2_A.data @ db.w2_B.data.t()
                for snap in self.lineage:
                    if f"b{i}_w2_A" in snap:
                        delta_w2 = delta_w2 + snap[f"b{i}_w2_A"] @ snap[f"b{i}_w2_B"].t()

                U2, S2, Vh2 = torch.linalg.svd(delta_w2, full_matrices=False)
                r_k2 = min(r_target, len(S2))
                err2 = float(S2[r_k2].item()) if r_k2 < len(S2) else 0.0
                max_err = max(max_err, err2)

                sqrt_s2 = torch.diag(torch.sqrt(S2[:r_k2]))
                db.w2_A = nn.Parameter(U2[:, :r_k2] @ sqrt_s2)
                db.w2_B = nn.Parameter(Vh2[:r_k2, :].t() @ sqrt_s2)
                db.rank = r_k

            # Head delta
            delta_head = self.head_A.data @ self.head_B.data.t()
            for snap in self.lineage:
                if "head_A" in snap:
                    delta_head = delta_head + snap["head_A"] @ snap["head_B"].t()

            Uh, Sh, Vhh = torch.linalg.svd(delta_head, full_matrices=False)
            r_kh = min(r_target, len(Sh))
            err_h = float(Sh[r_kh].item()) if r_kh < len(Sh) else 0.0
            max_err = max(max_err, err_h)

            sqrt_sh = torch.diag(torch.sqrt(Sh[:r_kh]))
            self.head_A = nn.Parameter(Uh[:, :r_kh] @ sqrt_sh)
            self.head_B = nn.Parameter(Vhh[:r_kh, :].t() @ sqrt_sh)

            self.rank = r_target
            self.lineage.clear()

        return max_err

    def merge_siblings(self, other: "DeltaExpert") -> None:
        """Merge sibling expert deltas via low-rank SVD averaging."""
        with torch.no_grad():
            for i, (db_self, db_other) in enumerate(zip(self.delta_blocks, other.delta_blocks)):
                # w1 average
                delta_w1_self = db_self.w1_A.data @ db_self.w1_B.data.t()
                delta_w1_other = db_other.w1_A.data @ db_other.w1_B.data.t()
                delta_w1_avg = 0.5 * (delta_w1_self + delta_w1_other)

                U, S, Vh = torch.linalg.svd(delta_w1_avg, full_matrices=False)
                r_k = min(self.rank, len(S))
                sqrt_s = torch.diag(torch.sqrt(S[:r_k]))
                db_self.w1_A = nn.Parameter(U[:, :r_k] @ sqrt_s)
                db_self.w1_B = nn.Parameter(Vh[:r_k, :].t() @ sqrt_s)

                # w2 average
                delta_w2_self = db_self.w2_A.data @ db_self.w2_B.data.t()
                delta_w2_other = db_other.w2_A.data @ db_other.w2_B.data.t()
                delta_w2_avg = 0.5 * (delta_w2_self + delta_w2_other)

                U2, S2, Vh2 = torch.linalg.svd(delta_w2_avg, full_matrices=False)
                r_k2 = min(self.rank, len(S2))
                sqrt_s2 = torch.diag(torch.sqrt(S2[:r_k2]))
                db_self.w2_A = nn.Parameter(U2[:, :r_k2] @ sqrt_s2)
                db_self.w2_B = nn.Parameter(Vh2[:r_k2, :].t() @ sqrt_s2)

            # Head average
            delta_head_self = self.head_A.data @ self.head_B.data.t()
            delta_head_other = other.head_A.data @ other.head_B.data.t()
            delta_head_avg = 0.5 * (delta_head_self + delta_head_other)

            Uh, Sh, Vhh = torch.linalg.svd(delta_head_avg, full_matrices=False)
            r_kh = min(self.rank, len(Sh))
            sqrt_sh = torch.diag(torch.sqrt(Sh[:r_kh]))
            self.head_A = nn.Parameter(Uh[:, :r_kh] @ sqrt_sh)
            self.head_B = nn.Parameter(Vhh[:r_kh, :].t() @ sqrt_sh)

            # Union seen mask
            self.seen = self.seen | other.seen

    def clone(self) -> "DeltaExpert":
        """Deep copy of the delta expert sharing the same frozen base."""
        cloned = DeltaExpert(base=self.base, rank=self.rank, lineage=self.lineage, seed=self.seed)
        cloned.load_state_dict(self.state_dict())
        cloned.seen = self.seen.clone()
        return cloned

    @staticmethod
    def _flat_grad(loss: Tensor, params) -> Tensor:
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for g, p in zip(grads, params)])

    def train_step(self, u: Tensor, y: Tensor, opt, replay=None, project: bool = False) -> float:
        """One optimizer step on (u, y); returns loss before the step."""
        self.train()
        y = y.long()
        self.mark_seen(y)
        if replay is not None:
            xr, yr = replay[0], replay[1].long()
            self.mark_seen(yr)

        if replay is not None and project:
            params = [p for p in self.parameters() if p.requires_grad]
            loss_new = F.cross_entropy(self(u), y)
            g = agem_project(
                self._flat_grad(loss_new, params),
                self._flat_grad(F.cross_entropy(self(xr), yr), params),
            )
            opt.zero_grad(set_to_none=True)
            offset = 0
            for p in params:
                p.grad = g[offset:offset + p.numel()].view_as(p).clone()
                offset += p.numel()
            opt.step()
            return float(loss_new.detach())

        if replay is not None:
            x_all, y_all = torch.cat([u, xr]), torch.cat([y, yr])
        else:
            x_all, y_all = u, y

        logits = self(x_all)
        loss = F.cross_entropy(logits, y_all)
        loss_val = float(loss.detach().item())

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss_val
