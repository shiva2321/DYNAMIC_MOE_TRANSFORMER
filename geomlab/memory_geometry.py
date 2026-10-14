"""
Memory Geometry Benchmark Module (Track Z Round Z2).
Implements candidate associative memory architectures for Multi-Query Associative Recall (MQAR):
1. SoftmaxAttentionMemory (Reference ceiling, O(T) state)
2. LinearAttentionMemory (elu+1 feature map, matrix state)
3. GatedDeltaMemory (Gated DeltaNet, matrix state)
4. DeltaL2Memory (L2-normalised keys, Householder projection)
5. KataPsdMemory (Self-dual homogeneous cone / PSD features, arXiv 2607.17419)
6. TriadicMemory (Coupled bilinear / tensor-product state, arXiv 2609.36529)
7. PostDecayMemory (PoST geometrically spaced decay spectrum, arXiv 2604.07658)
8. LegacyVMFMemoryAdapter (ISHMN InfiniteSphericalMemory read-only adapter)
"""

import math
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Read-only import from ishmn
from ishmn.memory import InfiniteSphericalMemory


def compute_state_bytes(
    arm: str,
    dim: int,
    num_slots: int = 32,
    hyperspace_dim: int = 8,
    batch_size: int = 1,
    seq_len: int = 128,
) -> int:
    """
    Computes exact persistent recurrent/cache state size in bytes per batch element.
    Assuming float32 (4 bytes per element).
    """
    bytes_per_elem = 4
    if arm == "softmax":
        # Softmax caches K and V of shape [seq_len, dim]
        return 2 * seq_len * dim * bytes_per_elem * batch_size
    elif arm == "linear":
        # S: [dim, dim], z: [dim]
        return (dim * dim + dim) * bytes_per_elem * batch_size
    elif arm in ("gated_delta", "delta_l2", "post_decay"):
        # S: [dim, dim]
        return (dim * dim) * bytes_per_elem * batch_size
    elif arm == "kata_psd":
        # S: [dim, 2 * dim]
        return (dim * (2 * dim)) * bytes_per_elem * batch_size
    elif arm == "triadic":
        # Coupled tensor state: S1 [dim, dim] + S2 [dim, dim]
        return (2 * dim * dim) * bytes_per_elem * batch_size
    elif arm == "legacy_vmf":
        # u: [num_slots, hyperspace_dim], r: [num_slots, 1], v: [num_slots, dim]
        # usage: [num_slots], energy: [num_slots]
        total_floats = num_slots * (hyperspace_dim + 1 + dim)
        return total_floats * bytes_per_elem * batch_size
    else:
        raise ValueError(f"Unknown memory arm: {arm}")


class SoftmaxAttentionMemory(nn.Module):
    """Reference ceiling: exact causal attention caching all keys and values."""
    def __init__(self, dim: int, scale: Optional[float] = None):
        super().__init__()
        self.dim = dim
        self.scale = scale if scale is not None else 1.0 / math.sqrt(dim)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, Any]:
        return {"keys": [], "values": []}

    def write_step(
        self,
        state: Dict[str, Any],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Dict[str, Any]:
        state["keys"].append(k_t.unsqueeze(1))
        state["values"].append(v_t.unsqueeze(1))
        return state

    def read_step(self, state: Dict[str, Any], q_t: torch.Tensor) -> torch.Tensor:
        if not state["keys"]:
            return torch.zeros_like(q_t)
        K = torch.cat(state["keys"], dim=1)    # [B, T, d]
        V = torch.cat(state["values"], dim=1)  # [B, T, d]
        q = q_t.unsqueeze(1)                   # [B, 1, d]
        scores = torch.bmm(q, K.transpose(1, 2)) * self.scale  # [B, 1, T]
        attn = F.softmax(scores, dim=-1)
        out = torch.bmm(attn, V).squeeze(1)     # [B, d]
        return out


class LinearAttentionMemory(nn.Module):
    """Causal linear attention with positive feature map phi(x) = elu(x) + 1."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device("cpu")
        return {
            "S": torch.zeros(batch_size, self.dim, self.dim, device=dev),
            "z": torch.zeros(batch_size, self.dim, device=dev),
        }

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def write_step(
        self,
        state: Dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        phi_k = self._feature_map(k_t)  # [B, d]
        # S_t = S_{t-1} + v_t * phi(k_t)^T
        update = torch.bmm(v_t.unsqueeze(-1), phi_k.unsqueeze(1))  # [B, d, d]
        new_S = state["S"] * alpha + update * beta
        new_z = state["z"] * alpha + phi_k * beta
        return {"S": new_S, "z": new_z}

    def read_step(self, state: Dict[str, torch.Tensor], q_t: torch.Tensor) -> torch.Tensor:
        phi_q = self._feature_map(q_t)  # [B, d]
        # numerator: S * phi_q
        num = torch.bmm(state["S"], phi_q.unsqueeze(-1)).squeeze(-1)  # [B, d]
        den = (phi_q * state["z"]).sum(dim=-1, keepdim=True).clamp(min=1e-5)  # [B, 1]
        return num / den


class GatedDeltaMemory(nn.Module):
    """Gated DeltaNet associative memory with continuous matrix state."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj_alpha = nn.Linear(dim, 1)
        self.proj_beta = nn.Linear(dim, 1)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device("cpu")
        return {"S": torch.zeros(batch_size, self.dim, self.dim, device=dev)}

    def write_step(
        self,
        state: Dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        B = k_t.size(0)
        S = state["S"]  # [B, d, d]
        if alpha is None:
            a_t = torch.sigmoid(self.proj_alpha(k_t)).unsqueeze(-1)  # [B, 1, 1]
        else:
            a_t = torch.tensor(alpha, device=k_t.device).view(1, 1, 1)
        if beta is None:
            b_t = torch.sigmoid(self.proj_beta(k_t)).unsqueeze(-1)   # [B, 1, 1]
        else:
            b_t = torch.tensor(beta, device=k_t.device).view(1, 1, 1)

        # Erasure: S * (I - beta * k * k^T)
        # S_k = S * k -> [B, d]
        S_k = torch.bmm(S, k_t.unsqueeze(-1)).squeeze(-1)
        # erasure update: S - beta * S_k * k^T
        erasure = torch.bmm(S_k.unsqueeze(-1), k_t.unsqueeze(1))  # [B, d, d]
        # new write: beta * v * k^T
        add = torch.bmm(v_t.unsqueeze(-1), k_t.unsqueeze(1))      # [B, d, d]

        new_S = a_t * (S - b_t * erasure) + b_t * add
        return {"S": new_S}

    def read_step(self, state: Dict[str, torch.Tensor], q_t: torch.Tensor) -> torch.Tensor:
        # Retrieval: S * q
        return torch.bmm(state["S"], q_t.unsqueeze(-1)).squeeze(-1)


class DeltaL2Memory(nn.Module):
    """
    Gated DeltaNet with strictly L2-normalised keys.
    Key projection to S^{d-1} makes (I - k k^T) an exact orthogonal projector.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj_alpha = nn.Linear(dim, 1)
        self.proj_beta = nn.Linear(dim, 1)

    def normalize_key(self, k: torch.Tensor) -> torch.Tensor:
        # Float32 precision safe L2 normalization
        return F.normalize(k.float(), p=2, dim=-1, eps=1e-8).to(k.dtype)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device("cpu")
        return {"S": torch.zeros(batch_size, self.dim, self.dim, device=dev)}

    def write_step(
        self,
        state: Dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        k_hat = self.normalize_key(k_t)
        S = state["S"]
        if alpha is None:
            a_t = torch.sigmoid(self.proj_alpha(k_hat)).unsqueeze(-1)
        else:
            a_t = torch.tensor(alpha, device=k_t.device).view(1, 1, 1)
        if beta is None:
            b_t = torch.sigmoid(self.proj_beta(k_hat)).unsqueeze(-1)
        else:
            b_t = torch.tensor(beta, device=k_t.device).view(1, 1, 1)

        S_k = torch.bmm(S, k_hat.unsqueeze(-1)).squeeze(-1)
        erasure = torch.bmm(S_k.unsqueeze(-1), k_hat.unsqueeze(1))
        add = torch.bmm(v_t.unsqueeze(-1), k_hat.unsqueeze(1))

        new_S = a_t * (S - b_t * erasure) + b_t * add
        return {"S": new_S}

    def read_step(self, state: Dict[str, torch.Tensor], q_t: torch.Tensor) -> torch.Tensor:
        q_hat = self.normalize_key(q_t)
        return torch.bmm(state["S"], q_hat.unsqueeze(-1)).squeeze(-1)


class KataPsdMemory(nn.Module):
    """
    KATA-style Self-Dual Homogeneous Cone / PSD features (arXiv 2607.17419).
    Maps key/query vectors into expanded cone features Phi(x) in R^{2d},
    breaking the capacity wall d while preserving fast outer-product updates.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.cone_dim = 2 * dim
        # Structured orthogonal cone projection
        self.register_buffer("proj_cone", torch.randn(dim, dim) / math.sqrt(dim))
        self.proj_alpha = nn.Linear(dim, 1)
        self.proj_beta = nn.Linear(dim, 1)

    def _cone_map(self, x: torch.Tensor) -> torch.Tensor:
        # Rank-one PSD / self-dual cone feature mapping
        x_rot = torch.matmul(x, self.proj_cone)
        # Combined positive features: [relu(x), relu(x_rot)]
        f = torch.cat([F.relu(x), F.relu(x_rot)], dim=-1)
        return F.normalize(f.float(), p=2, dim=-1, eps=1e-8).to(x.dtype)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device("cpu")
        return {"S": torch.zeros(batch_size, self.dim, self.cone_dim, device=dev)}

    def write_step(
        self,
        state: Dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        phi_k = self._cone_map(k_t)  # [B, cone_dim]
        S = state["S"]               # [B, dim, cone_dim]
        if alpha is None:
            a_t = torch.sigmoid(self.proj_alpha(k_t)).unsqueeze(-1)
        else:
            a_t = torch.tensor(alpha, device=k_t.device).view(1, 1, 1)
        if beta is None:
            b_t = torch.sigmoid(self.proj_beta(k_t)).unsqueeze(-1)
        else:
            b_t = torch.tensor(beta, device=k_t.device).view(1, 1, 1)

        S_k = torch.bmm(S, phi_k.unsqueeze(-1)).squeeze(-1)
        erasure = torch.bmm(S_k.unsqueeze(-1), phi_k.unsqueeze(1))
        add = torch.bmm(v_t.unsqueeze(-1), phi_k.unsqueeze(1))

        new_S = a_t * (S - b_t * erasure) + b_t * add
        return {"S": new_S}

    def read_step(self, state: Dict[str, torch.Tensor], q_t: torch.Tensor) -> torch.Tensor:
        phi_q = self._cone_map(q_t)
        return torch.bmm(state["S"], phi_q.unsqueeze(-1)).squeeze(-1)


class TriadicMemory(nn.Module):
    """
    Triadic / tensor-product state attention (arXiv 2609.36529).
    Maintains coupled matrix states S1, S2 representing 3-way interactions,
    scaling associative memory capacity toward d^2.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.proj_alpha = nn.Linear(dim, 1)
        self.proj_beta = nn.Linear(dim, 1)
        self.couple = nn.Linear(dim, dim)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device("cpu")
        return {
            "S1": torch.zeros(batch_size, self.dim, self.dim, device=dev),
            "S2": torch.zeros(batch_size, self.dim, self.dim, device=dev),
        }

    def write_step(
        self,
        state: Dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        k_hat = F.normalize(k_t.float(), p=2, dim=-1, eps=1e-8).to(k_t.dtype)
        if alpha is None:
            a_t = torch.sigmoid(self.proj_alpha(k_hat)).unsqueeze(-1)
        else:
            a_t = torch.tensor(alpha, device=k_t.device).view(1, 1, 1)
        if beta is None:
            b_t = torch.sigmoid(self.proj_beta(k_hat)).unsqueeze(-1)
        else:
            b_t = torch.tensor(beta, device=k_t.device).view(1, 1, 1)

        # Coupled write into S1 and S2
        add1 = torch.bmm(v_t.unsqueeze(-1), k_hat.unsqueeze(1))
        v2 = self.couple(v_t)
        add2 = torch.bmm(v2.unsqueeze(-1), k_hat.unsqueeze(1))

        new_S1 = a_t * state["S1"] + b_t * add1
        new_S2 = a_t * state["S2"] + b_t * add2
        return {"S1": new_S1, "S2": new_S2}

    def read_step(self, state: Dict[str, torch.Tensor], q_t: torch.Tensor) -> torch.Tensor:
        q_hat = F.normalize(q_t.float(), p=2, dim=-1, eps=1e-8).to(q_t.dtype)
        o1 = torch.bmm(state["S1"], q_hat.unsqueeze(-1)).squeeze(-1)
        o2 = torch.bmm(state["S2"], q_hat.unsqueeze(-1)).squeeze(-1)
        return o1 + 0.5 * o2


class PostDecayMemory(nn.Module):
    """
    Gated DeltaNet with Power-Spectrum of Time (PoST, arXiv 2604.07658).
    Geometrically spaced decay rates covering multiple timescales.
    """
    def __init__(self, dim: int, decay_min: float = 0.01, decay_max: float = 0.99):
        super().__init__()
        self.dim = dim
        self.decay_min = decay_min
        self.decay_max = decay_max

        # Initialize geometrically spaced log-decay parameter
        log_min = math.log(decay_min)
        log_max = math.log(decay_max)
        geom_rates = torch.exp(torch.linspace(log_min, log_max, steps=dim))
        self.log_decay = nn.Parameter(torch.logit(geom_rates.clamp(1e-4, 1.0 - 1e-4)))
        self.proj_beta = nn.Linear(dim, 1)

    def get_decay_rates(self) -> torch.Tensor:
        return torch.sigmoid(self.log_decay)

    def normalize_key(self, k: torch.Tensor) -> torch.Tensor:
        return F.normalize(k.float(), p=2, dim=-1, eps=1e-8).to(k.dtype)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        dev = device if device is not None else torch.device("cpu")
        return {"S": torch.zeros(batch_size, self.dim, self.dim, device=dev)}

    def write_step(
        self,
        state: Dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        k_hat = self.normalize_key(k_t)
        S = state["S"]
        if alpha is None:
            a_t = self.get_decay_rates().view(1, 1, self.dim)  # [1, 1, d] column-wise decay
        else:
            a_t = torch.tensor(alpha, device=k_t.device).view(1, 1, 1)
        if beta is None:
            b_t = torch.sigmoid(self.proj_beta(k_hat)).unsqueeze(-1)
        else:
            b_t = torch.tensor(beta, device=k_t.device).view(1, 1, 1)

        S_k = torch.bmm(S, k_hat.unsqueeze(-1)).squeeze(-1)
        erasure = torch.bmm(S_k.unsqueeze(-1), k_hat.unsqueeze(1))
        add = torch.bmm(v_t.unsqueeze(-1), k_hat.unsqueeze(1))

        new_S = (S - b_t * erasure) * a_t + b_t * add
        return {"S": new_S}

    def read_step(self, state: Dict[str, torch.Tensor], q_t: torch.Tensor) -> torch.Tensor:
        q_hat = self.normalize_key(q_t)
        return torch.bmm(state["S"], q_hat.unsqueeze(-1)).squeeze(-1)


class LegacyVMFMemoryAdapter(nn.Module):
    """
    Non-intrusive adapter for ISHMN InfiniteSphericalMemory (ishmn/memory.py).
    Adapts sequence queries and key-value writes into von Mises-Fisher spherical slots.
    """
    def __init__(self, dim: int, num_slots: int = 32, hyperspace_dim: int = 8):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.hyperspace_dim = hyperspace_dim

        # Underlying ISHMN memory module imported read-only
        self.inner_mem = InfiniteSphericalMemory(
            num_slots=num_slots,
            hyperspace_dim=hyperspace_dim,
            payload_dim=dim,
            initial_radius_shells=4,
            max_slots=num_slots,
        )

        # Projections to hyperspace S^{D-1} and radial shell R+
        self.proj_u = nn.Linear(dim, hyperspace_dim)
        self.proj_r = nn.Linear(dim, 1)
        self.proj_gate = nn.Linear(dim, 1)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        return self.inner_mem.init_memory(batch_size=batch_size, device=device)

    def write_step(
        self,
        state: Dict[str, torch.Tensor],
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        u_write = F.normalize(self.proj_u(k_t).float(), p=2, dim=-1, eps=1e-8).to(k_t.dtype)
        r_write = F.softplus(self.proj_r(k_t)) + 1.0
        gate = torch.sigmoid(self.proj_gate(k_t))
        erase_vec = torch.ones_like(v_t)
        add_vec = v_t
        kappa = torch.full((k_t.size(0), 1), 10.0, device=k_t.device)
        beta_r = torch.full((k_t.size(0), 1), 1.0, device=k_t.device)

        _, new_state = self.inner_mem.write(
            state=state,
            u_write=u_write,
            r_write=r_write,
            add_vector=add_vec,
            erase_vector=erase_vec,
            write_gate=gate,
            kappa_w=kappa,
            beta_rw=beta_r,
        )
        return new_state

    def read_step(self, state: Dict[str, torch.Tensor], q_t: torch.Tensor) -> torch.Tensor:
        u_query = F.normalize(self.proj_u(q_t).float(), p=2, dim=-1, eps=1e-8).to(q_t.dtype)
        r_query = F.softplus(self.proj_r(q_t)) + 1.0
        kappa = torch.full((q_t.size(0), 1), 10.0, device=q_t.device)
        beta_r = torch.full((q_t.size(0), 1), 1.0, device=q_t.device)

        read_payload, _, _ = self.inner_mem.read(
            state=state,
            u_query=u_query,
            r_query=r_query,
            kappa=kappa,
            beta_r=beta_r,
        )
        return read_payload
