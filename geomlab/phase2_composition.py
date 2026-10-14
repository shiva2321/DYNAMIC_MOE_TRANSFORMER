"""
Phase 2: Compositional Out-of-Distribution Generalization and Length Extrapolation.
Implements the tasks, models, and length extrapolation evaluation specified in
research/geomlab_results/phase2/PREREG.md:
- Multi-Step Toroidal Grid Relational Composition dataset
- StandardTransformer baseline
- EuclideanRNN baseline
- TorusIsometry model (flat Clifford torus phase addition)
- BlockUnitaryIsometry model (U(4)^64 matrix exponential composition)
"""

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dataset Generation: Toroidal Grid Relational Composition
# ---------------------------------------------------------------------------

def build_graph_composition_dataset(
    num_entities: int = 256,
    num_relations: int = 8,
    num_train: int = 6000,
    num_val: int = 1500,
    num_ood_per_length: int = 1000,
    train_lengths: Sequence[int] = (1, 2, 3),
    ood_lengths: Sequence[int] = (4, 6, 8, 12),
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Generates a multi-relational graph over N = 16 x 16 = 256 entities with 8 spatial generators:
    cardinal shifts (+x, -x, +y, -y) and diagonal shifts (+x+y, -x-y, +x-y, -x+y) modulo 16.
    Entities are assigned randomized abstract token IDs to prevent trivial coordinate leakage.
    Samples paths of length K: e_0 -r_1-> e_1 ... -r_K-> e_K.
    Training and validation use lengths in train_lengths.
    OOD test sets use lengths in ood_lengths.
    Guarantees zero path overlap between train and test/OOD sets.
    """
    rng = np.random.RandomState(seed)

    grid_dim = int(math.isqrt(num_entities))
    if grid_dim * grid_dim != num_entities:
        grid_dim = 16
        num_entities = 256

    # Random permutation of entity token IDs
    entity_perm = rng.permutation(num_entities)
    coord_to_entity = {i: int(entity_perm[i]) for i in range(num_entities)}
    entity_to_coord = {int(entity_perm[i]): i for i in range(num_entities)}

    # 8 generators on 2D torus
    def step_coord(c: int, r: int) -> int:
        x, y = c % grid_dim, c // grid_dim
        if r == 0:    x = (x + 1) % grid_dim
        elif r == 1:  x = (x - 1) % grid_dim
        elif r == 2:  y = (y + 1) % grid_dim
        elif r == 3:  y = (y - 1) % grid_dim
        elif r == 4:  x = (x + 1) % grid_dim; y = (y + 1) % grid_dim
        elif r == 5:  x = (x - 1) % grid_dim; y = (y - 1) % grid_dim
        elif r == 6:  x = (x + 1) % grid_dim; y = (y - 1) % grid_dim
        elif r == 7:  x = (x - 1) % grid_dim; y = (y + 1) % grid_dim
        return y * grid_dim + x

    def step_entity(e: int, r: int) -> int:
        c = entity_to_coord[e]
        nxt_c = step_coord(c, r)
        return coord_to_entity[nxt_c]

    # Precompute transition permutations for verification
    permutations = []
    for r in range(num_relations):
        p = np.array([step_entity(e, r) for e in range(num_entities)], dtype=np.int64)
        permutations.append(p)

    def generate_unique_paths(lengths: Sequence[int], count: int, existing_paths: set) -> Tuple[List[List[int]], List[int], List[int]]:
        paths = []
        targets = []
        path_lens = []
        attempts = 0
        max_attempts = count * 30

        while len(paths) < count and attempts < max_attempts:
            attempts += 1
            K = rng.choice(lengths)
            e0 = rng.randint(0, num_entities)
            rels = rng.randint(0, num_relations, size=K).tolist()
            path_tuple = (e0, tuple(rels))

            if path_tuple in existing_paths:
                continue

            existing_paths.add(path_tuple)

            # Compute target entity
            curr = e0
            for r in rels:
                curr = permutations[r][curr]

            seq = [e0] + rels
            paths.append(seq)
            targets.append(int(curr))
            path_lens.append(K)

        return paths, targets, path_lens

    seen_paths = set()

    # Train paths (K in 1, 2, 3)
    train_paths, train_targets, train_lens = generate_unique_paths(train_lengths, num_train, seen_paths)

    # In-distribution validation paths (K in 1, 2, 3)
    val_paths, val_targets, val_lens = generate_unique_paths(train_lengths, num_val, seen_paths)

    # OOD paths per length K in ood_lengths
    ood_dict = {}
    for ood_k in ood_lengths:
        p_k, t_k, l_k = generate_unique_paths([ood_k], num_ood_per_length, seen_paths)
        pad_len = ood_k + 1
        arr = np.zeros((len(p_k), pad_len), dtype=np.int64)
        for i, p in enumerate(p_k):
            arr[i, :len(p)] = p
        ood_dict[ood_k] = {
            "paths": torch.from_numpy(arr),
            "targets": torch.tensor(t_k, dtype=torch.long),
            "lengths": torch.tensor(l_k, dtype=torch.long),
        }

    max_train_len = max(train_lengths) + 1
    max_val_len = max(train_lengths) + 1

    def pad_paths(paths_list: List[List[int]], pad_len: int) -> torch.Tensor:
        arr = np.zeros((len(paths_list), pad_len), dtype=np.int64)
        for i, p in enumerate(paths_list):
            arr[i, :len(p)] = p
        return torch.from_numpy(arr)

    return {
        "train_paths": pad_paths(train_paths, max_train_len),
        "train_targets": torch.tensor(train_targets, dtype=torch.long),
        "train_lengths": torch.tensor(train_lens, dtype=torch.long),
        "val_paths": pad_paths(val_paths, max_val_len),
        "val_targets": torch.tensor(val_targets, dtype=torch.long),
        "val_lengths": torch.tensor(val_lens, dtype=torch.long),
        "ood_dict": ood_dict,
        "permutations": permutations,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class StandardTransformer(nn.Module):
    """
    Standard causal Transformer baseline with LayerNorm and learned position embeddings.
    """
    def __init__(
        self,
        num_entities: int = 256,
        num_relations: int = 8,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_entities = num_entities
        self.num_relations = num_relations
        total_vocab = num_entities + num_relations

        self.token_emb = nn.Embedding(total_vocab, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

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

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = seq.shape
        device = seq.device

        vocab_seq = seq.clone()
        if L > 1:
            vocab_seq[:, 1:] = seq[:, 1:] + self.num_entities

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        h = self.token_emb(vocab_seq) + self.pos_emb(positions)

        # Causal mask and key padding mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
        # key_padding_mask: True where padding (col index > lengths[row])
        col_indices = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        padding_mask = col_indices > lengths.unsqueeze(1)

        h_out = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask, is_causal=True)
        h_out = self.ln_f(h_out)

        batch_idx = torch.arange(B, device=device)
        final_h = h_out[batch_idx, lengths]  # (B, d_model)

        logits = self.classifier(final_h)
        return logits, final_h


class EuclideanRNN(nn.Module):
    """
    Standard recurrent baseline with LayerNorm, GELU, and explicit step masking.
    """
    def __init__(
        self,
        num_entities: int = 256,
        num_relations: int = 8,
        d_model: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.entity_emb = nn.Embedding(num_entities, d_model)
        self.rel_emb = nn.Embedding(num_relations, d_model)

        self.w_h = nn.Linear(d_model, d_model)
        self.w_r = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_entities)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = seq.shape
        h0 = self.entity_emb(seq[:, 0])  # (B, d_model)
        h = h0

        for t in range(1, L):
            mask = (t <= lengths).unsqueeze(-1)
            r_vec = self.rel_emb(seq[:, t])
            nxt_h = self.norm(F.gelu(self.w_h(h) + self.w_r(r_vec)))
            h = torch.where(mask, nxt_h, h)

        logits = self.classifier(h)
        return logits, h


class TorusIsometry(nn.Module):
    """
    Flat Clifford Torus T^D model with exact isometric phase addition.
    theta_t = theta_{t-1} + phi(r_t) (mod 2pi).
    Zero energy drift: ||e^{i theta}||_2 = 1.0 identically.
    """
    def __init__(
        self,
        num_entities: int = 256,
        num_relations: int = 8,
        d_model: int = 256,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_entities = num_entities
        self.entity_phases = nn.Parameter(torch.empty(num_entities, d_model).uniform_(-math.pi, math.pi))
        self.rel_phases = nn.Parameter(torch.empty(num_relations, d_model).uniform_(-math.pi, math.pi))
        self.beta = nn.Parameter(torch.tensor(12.0))

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = seq.shape
        curr_phase = self.entity_phases[seq[:, 0]]  # (B, d_model)

        for t in range(1, L):
            mask = (t <= lengths).unsqueeze(-1)
            delta = self.rel_phases[seq[:, t]]
            nxt_phase = (curr_phase + delta + math.pi) % (2 * math.pi) - math.pi
            curr_phase = torch.where(mask, nxt_phase, curr_phase)

        diff = curr_phase.unsqueeze(1) - self.entity_phases.unsqueeze(0)
        cos_sim = torch.cos(diff).mean(dim=-1)
        logits = self.beta * cos_sim
        return logits, curr_phase


class BlockUnitaryIsometry(nn.Module):
    """
    Non-abelian Lie group model U(b)^{D/b} with exact matrix-exponential composition.
    State: U_t = U_{t-1} R(r_t).
    Every transition is an exact unitary isometry.
    """
    def __init__(
        self,
        num_entities: int = 256,
        num_relations: int = 8,
        d_model: int = 256,
        block_size: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.block_size = block_size
        self.num_blocks = d_model // block_size
        self.num_entities = num_entities

        self.entity_gens_real = nn.Parameter(torch.randn(num_entities, self.num_blocks, block_size, block_size) * 0.1)
        self.entity_gens_imag = nn.Parameter(torch.randn(num_entities, self.num_blocks, block_size, block_size) * 0.1)

        self.rel_gens_real = nn.Parameter(torch.randn(num_relations, self.num_blocks, block_size, block_size) * 0.1)
        self.rel_gens_imag = nn.Parameter(torch.randn(num_relations, self.num_blocks, block_size, block_size) * 0.1)

        self.beta = nn.Parameter(torch.tensor(12.0))

    def _exp_unitary(self, g_real: torch.Tensor, g_imag: torch.Tensor) -> torch.Tensor:
        g = torch.complex(g_real, g_imag)
        s = 0.5 * (g - g.conj().transpose(-2, -1))
        return torch.linalg.matrix_exp(s)

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L = seq.shape

        u_entities = self._exp_unitary(self.entity_gens_real, self.entity_gens_imag)  # (N, B, b, b)
        u_rels = self._exp_unitary(self.rel_gens_real, self.rel_gens_imag)          # (|R|, B, b, b)

        state = u_entities[seq[:, 0]]  # (B, num_blocks, b, b)

        for t in range(1, L):
            mask = (t <= lengths).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            nxt = torch.matmul(state, u_rels[seq[:, t]])
            state = torch.where(mask, nxt, state)

        state_vec = state.reshape(B, -1)
        entity_vec = u_entities.reshape(self.num_entities, -1)

        trace_sim = torch.matmul(state_vec, entity_vec.conj().T).real / (self.num_blocks * self.block_size)
        logits = self.beta * trace_sim
        return logits, state_vec.real
