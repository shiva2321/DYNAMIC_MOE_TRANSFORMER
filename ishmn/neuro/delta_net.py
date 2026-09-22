"""
Chunked Gated DeltaNet Layer (Pure PyTorch, Triton-Free).
Harvested from E:\\AI_Research\\src\\concerto\\delta_net_chunked_gated.py.

Provides chunk-parallel gated delta rule fast-weight memory:
    v_old_t = S_{t-1} @ k_t
    S_t     = alpha_t * S_{t-1} + beta_t * outer(v_t - alpha_t * v_old_t, k_t)
    o_t     = S_t @ q_t

Derived via unit lower-triangular WY block elimination for chunk size C=16:
    T[i,j] = (A_i/A_j) * (k_j . k_i)  for j < i, else 0
    (I + diag(Beta) @ T) @ W = diag(Beta) @ (V - diag(A) @ U0)
    solved via torch.linalg.solve_triangular in O(C) chunk parallel steps.
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkedGatedDeltaRuleLayer(nn.Module):
    """
    Chunk-parallel Gated Delta Rule Layer.
    100% Triton-free, pure PyTorch, strictly bounded numerical stability.
    """
    def __init__(self, d_in: int, d_k: int, d_v: int, chunk_size: int = 16):
        super().__init__()
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.C = chunk_size

        self.Wk = nn.Linear(d_in, d_k, bias=False)
        self.Wv = nn.Linear(d_in, d_v, bias=False)
        self.Wq = nn.Linear(d_in, d_k, bias=False)
        self.Wbeta = nn.Linear(d_in, 1)
        self.Walpha = nn.Linear(d_in, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.Wk.weight)
        nn.init.xavier_uniform_(self.Wv.weight)
        nn.init.xavier_uniform_(self.Wq.weight)
        nn.init.zeros_(self.Wbeta.bias)
        nn.init.zeros_(self.Walpha.bias)

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, L, d_in]
            initial_state: Optional [B, d_v, d_k]
        Returns:
            out: [B, L, d_v]
            final_state: [B, d_v, d_k]
        """
        B, L, _ = x.shape
        C = self.C
        device, dtype = x.device, x.dtype

        k_full = F.normalize(self.Wk(x), dim=-1)
        v_full = self.Wv(x)
        q_full = F.normalize(self.Wq(x), dim=-1)
        beta_full = torch.sigmoid(self.Wbeta(x)).squeeze(-1)       # [B, L]
        log_alpha_full = F.logsigmoid(self.Walpha(x)).squeeze(-1)  # [B, L] <= 0

        if initial_state is not None:
            S = initial_state.clone()
        else:
            S = x.new_zeros(B, self.d_v, self.d_k)

        outputs = []
        n_chunks = (L + C - 1) // C
        eye_C = torch.eye(C, device=device, dtype=dtype).unsqueeze(0)
        NEG_BIG = -50.0

        for c in range(n_chunks):
            lo, hi = c * C, min((c + 1) * C, L)
            c_len = hi - lo
            K = k_full[:, lo:hi]
            V = v_full[:, lo:hi]
            Q = q_full[:, lo:hi]
            Beta = beta_full[:, lo:hi]
            LogAlpha = log_alpha_full[:, lo:hi]

            log_A = torch.cumsum(LogAlpha, dim=1)  # [B, c_len]
            raw_log_ratio = log_A.unsqueeze(-1) - log_A.unsqueeze(-2)  # [B, c_len, c_len]
            idx = torch.arange(c_len, device=device)
            strictly_lower = idx.unsqueeze(-1) > idx.unsqueeze(-2)
            lower_incl = idx.unsqueeze(-1) >= idx.unsqueeze(-2)

            safe_for_T = torch.where(strictly_lower.unsqueeze(0), raw_log_ratio, NEG_BIG)
            ratio_T = torch.exp(safe_for_T)
            safe_for_M = torch.where(lower_incl.unsqueeze(0), raw_log_ratio, NEG_BIG)
            ratio_M = torch.exp(safe_for_M)

            A = torch.exp(log_A)

            KK = torch.einsum("bik,bjk->bij", K, K)
            T = torch.tril(ratio_T * KK, diagonal=-1)

            U0 = torch.einsum("bik,bvk->biv", K, S)
            U0_scaled = A.unsqueeze(-1) * U0

            lhs = eye_C[:, :c_len, :c_len] + Beta.unsqueeze(-1) * T
            rhs = Beta.unsqueeze(-1) * (V - U0_scaled)
            W = torch.linalg.solve_triangular(lhs, rhs, upper=False, unitriangular=True)

            Bmat = torch.einsum("bik,bjk->bij", K, Q)
            KQ_row_i = Bmat.transpose(-1, -2)
            M = torch.tril(ratio_M * KQ_row_i, diagonal=0)

            U0q = torch.einsum("bik,bvk->biv", Q, S)
            U0q_scaled = A.unsqueeze(-1) * U0q

            O = U0q_scaled + torch.einsum("bij,bjv->biv", M, W)
            outputs.append(O)

            # Chunk-final carry
            A_last = A[:, -1:]                                         # [B, 1]
            last_log_A = log_A[:, -1:]                                 # [B, 1]
            log_ratio_last = last_log_A - log_A                        # [B, c_len] <= 0
            ratio_last = torch.exp(log_ratio_last).unsqueeze(-1)       # [B, c_len, 1]
            W_scaled = W * ratio_last                                  # [B, c_len, d_v]
            delta_S = torch.einsum("biv,bik->bvk", W_scaled, K)
            S = A_last.unsqueeze(-1) * S + delta_S

        out = torch.cat(outputs, dim=1)
        return out, S
