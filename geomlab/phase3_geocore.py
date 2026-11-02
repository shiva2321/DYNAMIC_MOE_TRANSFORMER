"""
Phase 3: Unified Heterogeneous Multi-Manifold Architecture (GeoCore).
Implements the tasks, models, and attractor cleanup dynamics specified in
research/geomlab_results/phase3/PREREG.md:
- Multi-Step Bound Variable Tracking Dataset on real Banking77 text
- BaselineTransformer
- BaselineRecurrent
- UncleanedGeoCore (ablation without attractor dynamics)
- FullGeoCore (complete heterogeneous architecture with Metric Hopfield Cleanup)
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from growmoe.frontend import HashFrontend
from growmoe.streams_real import _load_banking77


# ---------------------------------------------------------------------------
# Dataset Generation: Multi-Step Bound Variable Tracking on Real Text
# ---------------------------------------------------------------------------

def build_bound_tracking_dataset(
    num_entities: int = 128,
    num_roles: int = 4,
    num_train: int = 4000,
    num_val: int = 1000,
    num_ood: int = 1000,
    train_lengths: Sequence[int] = (1, 2),
    ood_lengths: Sequence[int] = (4, 8),
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Constructs multi-step role-filler tracking benchmark on real Banking77 text queries.
    Initial state binds 4 roles to 4 distinct entities.
    Sequential transformations (e.g. cyclic shift, swap, reversal) permute role assignments.
    At step K, a query role R_q is probed; target is the entity currently bound to R_q.
    """
    rng = np.random.RandomState(seed)

    # 1. Load real Banking77 text samples for entities
    df_train, df_test = _load_banking77()
    unique_labels = sorted(df_train["label"].unique())
    # Select one representative text query per intent class
    entity_texts = []
    for lbl in unique_labels:
        sub = df_train[df_train["label"] == lbl]
        entity_texts.append(sub["text"].iloc[0])
    
    # If fewer than num_entities, cycle or draw additional
    while len(entity_texts) < num_entities:
        idx = len(entity_texts) % len(df_test)
        entity_texts.append(df_test["text"].iloc[idx])
    entity_texts = entity_texts[:num_entities]

    # Pre-extract frontend features: (N, 1024)
    fe = HashFrontend(d=1024)
    entity_features = fe(entity_texts).float()  # (128, 1024)

    # 4 transformation operators on role slots [0, 1, 2, 3]
    # 0: cyclic shift (0->1->2->3->0)
    # 1: swap (0 <-> 2)
    # 2: swap (1 <-> 3)
    # 3: reverse (0 <-> 3, 1 <-> 2)
    def apply_trans(slots: List[int], trans_id: int) -> List[int]:
        s = list(slots)
        if trans_id == 0:
            return [s[3], s[0], s[1], s[2]]
        elif trans_id == 1:
            s[0], s[2] = s[2], s[0]
            return s
        elif trans_id == 2:
            s[1], s[3] = s[3], s[1]
            return s
        elif trans_id == 3:
            return s[::-1]
        return s

    def generate_split(count: int, lengths: Sequence[int], existing: set) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        init_fillers = []
        transitions = []
        queries = []
        targets = []
        path_lens = []
        max_len = max(lengths)

        attempts = 0
        while len(init_fillers) < count and attempts < count * 30:
            attempts += 1
            K = rng.choice(lengths)
            # Choose 4 distinct entity fillers for the 4 role slots
            fillers = rng.choice(num_entities, size=num_roles, replace=False).tolist()
            # Transformation sequence of length K
            trans_seq = rng.randint(0, 4, size=K).tolist()
            # Query role probed at the end
            q_role = rng.randint(0, num_roles)

            tuple_key = (tuple(fillers), tuple(trans_seq), q_role)
            if tuple_key in existing:
                continue
            existing.add(tuple_key)

            # Trace execution
            curr_slots = list(fillers)
            for t_id in trans_seq:
                curr_slots = apply_trans(curr_slots, t_id)
            target_entity = curr_slots[q_role]

            # Pad transitions to max_len
            padded_trans = trans_seq + [0] * (max_len - K)

            init_fillers.append(fillers)
            transitions.append(padded_trans)
            queries.append(q_role)
            targets.append(target_entity)
            path_lens.append(K)

        return (
            torch.tensor(init_fillers, dtype=torch.long),
            torch.tensor(transitions, dtype=torch.long),
            torch.tensor(queries, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(path_lens, dtype=torch.long),
        )

    seen = set()
    tr_fillers, tr_trans, tr_queries, tr_targets, tr_lens = generate_split(num_train, train_lengths, seen)
    val_fillers, val_trans, val_queries, val_targets, val_lens = generate_split(num_val, train_lengths, seen)

    ood_dict = {}
    for ood_k in ood_lengths:
        f_k, t_k, q_k, y_k, l_k = generate_split(num_ood, [ood_k], seen)
        ood_dict[ood_k] = {
            "fillers": f_k,
            "trans": t_k,
            "queries": q_k,
            "targets": y_k,
            "lengths": l_k,
        }

    return {
        "entity_features": entity_features,
        "train": {
            "fillers": tr_fillers,
            "trans": tr_trans,
            "queries": tr_queries,
            "targets": tr_targets,
            "lengths": tr_lens,
        },
        "val": {
            "fillers": val_fillers,
            "trans": val_trans,
            "queries": val_queries,
            "targets": val_targets,
            "lengths": val_lens,
        },
        "ood_dict": ood_dict,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class BaselineTransformer(nn.Module):
    """
    Standard deep Transformer baseline with LayerNorm and learned position embeddings.
    """
    def __init__(
        self,
        entity_features: torch.Tensor,
        num_roles: int = 4,
        num_trans: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        num_entities, in_dim = entity_features.shape
        self.num_entities = num_entities

        # Linear projection for continuous text features
        self.proj = nn.Linear(in_dim, d_model)
        self.register_buffer("entity_features", entity_features)

        self.role_emb = nn.Embedding(num_roles, d_model)
        self.trans_emb = nn.Embedding(num_trans, d_model)
        self.pos_emb = nn.Embedding(32, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_entities)

    def forward(
        self,
        fillers: torch.Tensor,
        trans: torch.Tensor,
        queries: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # fillers: (B, 4)
        # trans: (B, L_trans)
        # queries: (B,)
        B, L_trans = trans.shape
        device = fillers.device

        # Encode 4 role-filler bindings
        ent_embs = self.proj(self.entity_features[fillers])  # (B, 4, D)
        role_embs = self.role_emb(torch.arange(4, device=device)).unsqueeze(0).expand(B, 4, -1)  # (B, 4, D)
        bound_inputs = ent_embs + role_embs  # (B, 4, D)

        # Transformation embeddings: (B, L_trans, D)
        trans_inputs = self.trans_emb(trans)

        # Query role input: (B, 1, D)
        query_input = self.role_emb(queries).unsqueeze(1)

        # Sequence: [bound_0, bound_1, bound_2, bound_3, trans_1, ..., trans_K, query]
        seq = torch.cat([bound_inputs, trans_inputs, query_input], dim=1)  # (B, 4 + L_trans + 1, D)
        total_L = seq.shape[1]

        positions = torch.arange(total_L, device=device).unsqueeze(0).expand(B, total_L)
        h = seq + self.pos_emb(positions)

        # Attention mask and key padding mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(total_L, device=device)
        col_indices = torch.arange(total_L, device=device).unsqueeze(0).expand(B, total_L)
        is_trans = (col_indices >= 4) & (col_indices < 4 + L_trans)
        padding_mask = is_trans & ((col_indices - 4) >= lengths.unsqueeze(1))
        h_out = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        h_out = self.ln_f(h_out)

        # Final token represents the probed query
        final_h = h_out[:, -1, :]  # (B, D)
        logits = self.classifier(final_h)
        return logits, final_h


class BaselineRecurrent(nn.Module):
    """
    Standard recurrent baseline with LayerNorm and GELU.
    """
    def __init__(
        self,
        entity_features: torch.Tensor,
        num_roles: int = 4,
        num_trans: int = 4,
        d_model: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        num_entities, in_dim = entity_features.shape
        self.num_entities = num_entities

        self.proj = nn.Linear(in_dim, d_model)
        self.register_buffer("entity_features", entity_features)

        self.role_emb = nn.Embedding(num_roles, d_model)
        self.trans_emb = nn.Embedding(num_trans, d_model)

        self.w_h = nn.Linear(d_model, d_model)
        self.w_t = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.probe = nn.Linear(2 * d_model, num_entities)

    def forward(
        self,
        fillers: torch.Tensor,
        trans: torch.Tensor,
        queries: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_trans = trans.shape
        device = fillers.device

        # Initial memory bundle: sum_{m=1}^4 (Role_m + Filler_m)
        ent_embs = self.proj(self.entity_features[fillers])  # (B, 4, D)
        role_embs = self.role_emb(torch.arange(4, device=device)).unsqueeze(0).expand(B, 4, -1)
        h = (ent_embs + role_embs).sum(dim=1)  # (B, D)

        for t in range(L_trans):
            mask = (t < lengths).unsqueeze(-1)
            t_vec = self.trans_emb(trans[:, t])
            nxt_h = self.norm(F.gelu(self.w_h(h) + self.w_t(t_vec)))
            h = torch.where(mask, nxt_h, h)

        # Readout conditioned on query role
        q_vec = self.role_emb(queries)
        probe_input = torch.cat([h, q_vec], dim=-1)
        logits = self.probe(probe_input)
        return logits, h


class UncleanedGeoCore(nn.Module):
    """
    Heterogeneous Geometric Core without attractor dynamics (ablation):
    - Spherical input normalization
    - Toroidal role-filler phase binding
    - Block-unitary state transitions
    - Unbinding via query role subtraction (no attractor cleanup)
    """
    def __init__(
        self,
        entity_features: torch.Tensor,
        num_roles: int = 4,
        num_trans: int = 4,
        d_model: int = 256,
        block_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.num_blocks = d_model // block_size
        num_entities, in_dim = entity_features.shape
        self.num_entities = num_entities

        # Perceptual encoder mapping continuous text to unit hypersphere
        self.proj1 = nn.Linear(in_dim, 2 * d_model)
        self.proj2 = nn.Linear(2 * d_model, 2 * d_model)
        self.register_buffer("entity_features", entity_features)

        # Toroidal role phases
        self.role_phases = nn.Parameter(torch.empty(num_roles, d_model).uniform_(-math.pi, math.pi))

        # Block-unitary Lie group generator parameters for transformations
        self.trans_gens_real = nn.Parameter(torch.randn(num_trans, self.num_blocks, block_size, block_size) * 0.1)
        self.trans_gens_imag = nn.Parameter(torch.randn(num_trans, self.num_blocks, block_size, block_size) * 0.1)

        self.beta = nn.Parameter(torch.tensor(12.0))

    def _get_entity_phases(self) -> torch.Tensor:
        # Map text features to unit phasors on T^D
        h = F.gelu(self.proj1(self.entity_features))
        uv = self.proj2(h).reshape(-1, self.d_model, 2)
        uv_norm = F.normalize(uv, p=2, dim=-1)
        return torch.atan2(uv_norm[:, :, 1], uv_norm[:, :, 0])  # (N, D)

    def _exp_unitary(self) -> torch.Tensor:
        g = torch.complex(self.trans_gens_real, self.trans_gens_imag)
        s = 0.5 * (g - g.conj().transpose(-2, -1))
        return torch.linalg.matrix_exp(s)  # (|T|, num_blocks, b, b)

    def forward(
        self,
        fillers: torch.Tensor,
        trans: torch.Tensor,
        queries: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_trans = trans.shape
        device = fillers.device

        entity_phases = self._get_entity_phases()  # (N, D)
        ent_p = entity_phases[fillers]            # (B, 4, D)
        role_p = self.role_phases.unsqueeze(0).expand(B, 4, -1)  # (B, 4, D)

        # Toroidal binding: phi_m = role_p + ent_p
        bound_phases = (role_p + ent_p + math.pi) % (2 * math.pi) - math.pi
        # Superposition bundle: S_0 = sum_{m=1}^4 e^{i phi_m}
        phasors_bundle = torch.exp(1j * bound_phases).sum(dim=1)  # (B, D)

        # Convert to block representation for unitary transformation
        # (B, num_blocks, block_size)
        state_blocks = phasors_bundle.reshape(B, self.num_blocks, self.block_size, 1)

        u_trans = self._exp_unitary()  # (|T|, num_blocks, b, b)

        for t in range(L_trans):
            mask = (t < lengths).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            step_u = u_trans[trans[:, t]]  # (B, num_blocks, b, b)
            nxt_state = torch.matmul(step_u, state_blocks)
            state_blocks = torch.where(mask, nxt_state, state_blocks)

        # Flatten transformed state: (B, D)
        state_final = state_blocks.squeeze(-1).reshape(B, self.d_model)

        # Unbinding via query role conjugate phasor: S_K * e^{-i role_q}
        q_role_phase = self.role_phases[queries]  # (B, D)
        unbound_phasor = state_final * torch.exp(-1j * q_role_phase)  # (B, D)

        # Readout: cosine similarity with clean entity prototypes
        # Re(unbound_phasor @ cb_phasors^H)
        cb_phasors = torch.exp(1j * entity_phases)  # (N, D)
        sim = (unbound_phasor.unsqueeze(1) * cb_phasors.conj().unsqueeze(0)).real.mean(dim=-1)  # (B, N)

        logits = self.beta * sim
        return logits, state_final.real


class FullGeoCore(UncleanedGeoCore):
    """
    Complete Heterogeneous Geometric Intelligence Core (`GeoCore`):
    - Spherical perceptual input encoding
    - Toroidal role-filler phase binding
    - Block-unitary sequential state transitions
    - Continuous Metric Hopfield Attractor Dynamics for crosstalk cleanup
    """
    def __init__(
        self,
        entity_features: torch.Tensor,
        num_roles: int = 4,
        num_trans: int = 4,
        d_model: int = 256,
        block_size: int = 4,
        hopfield_beta: float = 30.0,
    ):
        super().__init__(entity_features, num_roles, num_trans, d_model, block_size)
        self.hopfield_beta = hopfield_beta

    def _hopfield_cleanup(self, unbound_phasor: torch.Tensor, entity_phases: torch.Tensor) -> torch.Tensor:
        """
        Continuous Metric Hopfield Attractor update on the Clifford Torus:
        Computes energy-weighted projection against codebook phasors to snap
        noisy superpositions back to clean prototype representations.
        """
        cb_phasors = torch.exp(1j * entity_phases)  # (N, D)

        # Compute metric overlap / cosine similarity: (B, N)
        cos_overlap = (unbound_phasor.unsqueeze(1) * cb_phasors.conj().unsqueeze(0)).real.mean(dim=-1)
        # Softmax energy weighting
        attn_weights = F.softmax(self.hopfield_beta * cos_overlap, dim=-1)  # (B, N)

        # Attractor state projection
        cleaned_phasor = torch.matmul(attn_weights.to(cb_phasors.dtype), cb_phasors)  # (B, D)
        # Re-project to unit circle (Clifford Torus manifold projection)
        cleaned_phasor_unit = cleaned_phasor / (cleaned_phasor.abs() + 1e-12)
        return cleaned_phasor_unit

    def forward(
        self,
        fillers: torch.Tensor,
        trans: torch.Tensor,
        queries: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L_trans = trans.shape
        device = fillers.device

        entity_phases = self._get_entity_phases()  # (N, D)
        ent_p = entity_phases[fillers]            # (B, 4, D)
        role_p = self.role_phases.unsqueeze(0).expand(B, 4, -1)

        # Toroidal binding
        bound_phases = (role_p + ent_p + math.pi) % (2 * math.pi) - math.pi
        phasors_bundle = torch.exp(1j * bound_phases).sum(dim=1)  # (B, D)

        # Block-unitary transition
        state_blocks = phasors_bundle.reshape(B, self.num_blocks, self.block_size, 1)
        u_trans = self._exp_unitary()

        for t in range(L_trans):
            mask = (t < lengths).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            step_u = u_trans[trans[:, t]]
            nxt_state = torch.matmul(step_u, state_blocks)
            state_blocks = torch.where(mask, nxt_state, state_blocks)

        state_final = state_blocks.squeeze(-1).reshape(B, self.d_model)

        # Unbinding via query role
        q_role_phase = self.role_phases[queries]
        unbound_phasor = state_final * torch.exp(-1j * q_role_phase)

        # Continuous Metric Hopfield Attractor Cleanup
        cleaned_phasor = self._hopfield_cleanup(unbound_phasor, entity_phases)

        # Final Readout against clean codebook prototypes
        cb_phasors = torch.exp(1j * entity_phases)
        sim = (cleaned_phasor.unsqueeze(1) * cb_phasors.conj().unsqueeze(0)).real.mean(dim=-1)

        logits = self.beta * sim
        return logits, state_final.real
