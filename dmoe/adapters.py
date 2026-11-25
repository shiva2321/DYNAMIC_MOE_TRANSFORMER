"""Per-expert low-rank residual adapters shared across Attention, STM, and Thinker modules."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from dmoe.routing import sparse_weighted_expert_dispatch


class ExpertAdapter(nn.Module):
    """Zero-init pre-norm low-rank residual bottleneck adapter (preserves function at spawn)."""

    def __init__(self, d_model: int, rank: int = 24):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = max(4, int(rank))
        self.norm = nn.LayerNorm(self.d_model)
        self.down = nn.Linear(self.d_model, self.rank)
        self.act = nn.GELU()
        self.up = nn.Linear(self.rank, self.d_model)
        with torch.no_grad():
            self.up.weight.zero_()
            self.up.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.act(self.down(self.norm(x))))


class RoutedAdapterBank(nn.Module):
    """Lifecycle-managed bank of per-expert adapters (`spawn`, `freeze`, `prune`, batched dispatch)."""

    def __init__(self, d_model: int, rank: int = 24, initial_experts: int = 1):
        super().__init__()
        self.d_model = int(d_model)
        self.rank = max(4, int(rank))
        self.adapters = nn.ModuleList([ExpertAdapter(d_model, rank) for _ in range(initial_experts)])

    def spawn(self, source_idx: Optional[int] = None, fresh_init: bool = False) -> int:
        dev = next(self.parameters()).device
        new_adp = ExpertAdapter(self.d_model, self.rank).to(dev)
        if not fresh_init and source_idx is not None and 0 <= source_idx < len(self.adapters):
            src = self.adapters[source_idx]
            with torch.no_grad():
                for ps, pd in zip(src.parameters(), new_adp.parameters()):
                    pd.copy_(ps)
        self.adapters.append(new_adp)
        return len(self.adapters) - 1

    def freeze(self, idx: int) -> None:
        if 0 <= idx < len(self.adapters):
            for p in self.adapters[idx].parameters():
                p.requires_grad_(False)

    def prune(self, idx: int) -> bool:
        if len(self.adapters) <= 1 or not (0 <= idx < len(self.adapters)):
            return False
        del self.adapters[idx]
        return True

    def dispatch_sparse(
        self,
        x: torch.Tensor,
        dispatch_plan: object,
    ) -> torch.Tensor:
        """Apply per-token weighted sparse adapters using vectorized masked-GEMM (`E <= 8`) or sorted scatter-add."""
        if isinstance(dispatch_plan, dict) and dispatch_plan.get("mode") == "dense_mask":
            one_hot_w = dispatch_plan["one_hot_w"]
            E = len(self.adapters)
            if E == 0:
                return torch.zeros_like(x)
            x_flat = x.reshape(-1, self.d_model)
            w = one_hot_w[:, :E].to(dtype=x.dtype)
            if E == 1:
                return (self.adapters[0](x_flat) * w[:, :1]).view_as(x)
            x_n = F.layer_norm(x_flat, (self.d_model,), eps=1e-5)
            norm_w = torch.stack([a.norm.weight for a in self.adapters], dim=0)  # [E, D]
            norm_b = torch.stack([a.norm.bias for a in self.adapters], dim=0)    # [E, D]
            down_w = torch.stack([a.down.weight for a in self.adapters], dim=0)  # [E, R, D]
            down_b = torch.stack([a.down.bias for a in self.adapters], dim=0)    # [E, R]
            up_w = torch.cat([a.up.weight for a in self.adapters], dim=1)        # [D, E*R]
            up_b = torch.stack([a.up.bias for a in self.adapters], dim=0)        # [E, D]
            x_e = x_n.unsqueeze(1) * norm_w.unsqueeze(0) + norm_b.unsqueeze(0)   # [N, E, D]
            h = self.adapters[0].act(torch.einsum("ned,erd->ner", x_e, down_w) + down_b.unsqueeze(0))
            h_m = (h * w.unsqueeze(-1)).reshape(x_flat.size(0), E * self.rank)
            out = F.linear(h_m, up_w) + torch.matmul(w, up_b)
            return out.to(dtype=x.dtype).view_as(x)
        res = sparse_weighted_expert_dispatch(
            x.reshape(-1, self.d_model),
            self.adapters,
            dispatch_plan,
            out_dim=self.d_model,
        )
        return res.to(dtype=x.dtype).view_as(x) if res is not None else torch.zeros_like(x)

    def forward(
        self,
        x: torch.Tensor,
        assigned_experts: Optional[torch.Tensor] = None,
        force_idx: Optional[int] = None,
        skip_uniform_check: bool = False,
        dispatch_plan: Optional[Tuple[torch.Tensor, torch.Tensor, List[int]]] = None,
    ) -> torch.Tensor:
        if dispatch_plan is not None:
            return self.dispatch_sparse(x, dispatch_plan)
        if len(self.adapters) == 1:
            return self.adapters[0](x)
        if force_idx is not None:
            return self.adapters[min(int(force_idx), len(self.adapters) - 1)](x)
        B = x.size(0)
        if assigned_experts is not None and assigned_experts.numel() == B:
            if not skip_uniform_check:
                e0 = int(assigned_experts[0].item())
                if bool((assigned_experts == e0).all().item()):
                    return self.adapters[min(e0, len(self.adapters) - 1)](x)
            was_2d = x.dim() == 2
            x3 = x.unsqueeze(1) if was_2d else x
            ids = assigned_experts.clamp(0, len(self.adapters) - 1)
            norm_w = torch.stack([a.norm.weight for a in self.adapters], dim=0)[ids]
            norm_b = torch.stack([a.norm.bias for a in self.adapters], dim=0)[ids]
            down_w = torch.stack([a.down.weight for a in self.adapters], dim=0)[ids]
            down_b = torch.stack([a.down.bias for a in self.adapters], dim=0)[ids]
            up_w = torch.stack([a.up.weight for a in self.adapters], dim=0)[ids]
            up_b = torch.stack([a.up.bias for a in self.adapters], dim=0)[ids]
            h = F.layer_norm(x3, (self.d_model,), eps=1e-5) * norm_w.unsqueeze(1) + norm_b.unsqueeze(1)
            h = self.adapters[0].act(torch.bmm(h, down_w.transpose(1, 2)) + down_b.unsqueeze(1))
            out = torch.bmm(h, up_w.transpose(1, 2)) + up_b.unsqueeze(1)
            return out.squeeze(1) if was_2d else out
        return self.adapters[-1](x)
