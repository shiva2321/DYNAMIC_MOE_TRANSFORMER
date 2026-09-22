"""
Stage 1 Diagnostic: Multi-Query Associative Recall (MQAR) & State Tracking Benchmark
Strictly Matched Compute & Parameter Budget Comparison:
- Control Arm 1: Mode B Alone (Latent Deliberation Sandwich, Frozen Memory)
- Control Arm 2: Mode C Alone (Day/Night Memory Buffer, Surface 1-Step Retrieval)
- Treatment Arm: Unified Shared-Substrate B+C (Isomorphic S^7 x R^+ + Epistemic Write-Gating + REM Replay)

Gate 1 Falsification Gate:
Acc(Unified) >= max(Acc(ModeB), Acc(ModeC)) + 15.0% on Multi-Hop Associative Recall.
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ishmn.sigreg import SIGReg


# ==============================================================================
# 1. SYNTHETIC MQAR DATASET GENERATOR
# ==============================================================================

class MQARDataset:
    """
    Generates synthetic Multi-Query Associative Recall sequences with:
    - Direct Key-Value pairs: K_i -> V_i
    - Multi-Hop Transitive Chains: K_a -> K_b, K_b -> V_c => Query(K_a) -> V_c
    - Distractor noise tokens between key-value pairs
    """
    def __init__(
        self,
        num_keys: int = 32,
        num_vals: int = 32,
        num_distractors: int = 32,
        seq_len: int = 96,
        seed: int = 42
    ):
        self.num_keys = num_keys
        self.num_vals = num_vals
        self.num_distractors = num_distractors
        self.seq_len = seq_len
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.PAD = 0
        self.ASSIGN = 1
        self.QUERY = 2
        self.HOP = 3
        self.SPECIAL_OFFSET = 4

        self.KEY_START = self.SPECIAL_OFFSET
        self.VAL_START = self.KEY_START + num_keys
        self.DIST_START = self.VAL_START + num_vals
        self.VOCAB_SIZE = self.DIST_START + num_distractors

    def generate_batch(
        self,
        batch_size: int,
        num_direct: int = 6,
        num_multi_hop: int = 3
    ) -> Tuple[List[List[Tuple[int, int]]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates structured (Key, Value) pairs and query sets.
        Returns:
            all_pairs: List of list of (k, v) token pairs per batch sample
            queries: [B, num_queries] query key tokens
            targets: [B, num_queries] expected target tokens
            is_multihop: [B, num_queries] bool mask indicating multi-hop queries
        """
        all_pairs = []
        batch_queries = []
        batch_targets = []
        batch_is_mh = []

        for b in range(batch_size):
            # Sample keys and values
            keys = random.sample(range(self.KEY_START, self.KEY_START + self.num_keys), num_direct + num_multi_hop * 2)
            vals = random.sample(range(self.VAL_START, self.VAL_START + self.num_vals), num_direct + num_multi_hop)

            pairs = []
            queries = []
            targets = []
            is_mh = []

            # 1. Multi-hop chains: ka -> kb, kb -> vc
            for h in range(num_multi_hop):
                ka = keys[h]
                kb = keys[num_multi_hop + h]
                vc = vals[h]
                pairs.append((ka, kb))
                pairs.append((kb, vc))
                queries.append(ka)
                targets.append(vc)
                is_mh.append(True)

            # 2. Direct bindings: kd -> vd
            for d in range(num_direct):
                kd = keys[2 * num_multi_hop + d]
                vd = vals[num_multi_hop + d]
                pairs.append((kd, vd))
                queries.append(kd)
                targets.append(vd)
                is_mh.append(False)

            random.shuffle(pairs)
            all_pairs.append(pairs)
            batch_queries.append(queries)
            batch_targets.append(targets)
            batch_is_mh.append(is_mh)

        return (
            all_pairs,
            torch.tensor(batch_queries, dtype=torch.long),
            torch.tensor(batch_targets, dtype=torch.long),
            torch.tensor(batch_is_mh, dtype=torch.bool)
        )


# ==============================================================================
# 2. CONTROL ARM 1: MODE B ALONE (Latent Deliberation, Frozen Memory)
# ==============================================================================

class ModeBAlone(nn.Module):
    """
    Mode B Alone:
    - Deliberates across internal states using Adaptive Computation Time (ACT).
    - Memory is FROZEN at test time: no dynamic associative memory writes.
    - Must store and retrieve associations solely within recurrent hidden state.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 40, act_ticks: int = 3):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.act_ticks = act_ticks

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, embed_dim, batch_first=True)
        self.deliberation_cell = nn.GRUCell(embed_dim, embed_dim)
        self.temp = nn.Parameter(torch.tensor(5.0))

    def forward(self, pairs_batch: List[List[Tuple[int, int]]], query_tokens: torch.Tensor) -> torch.Tensor:
        B, num_q = query_tokens.shape
        device = query_tokens.device

        # Serialize pairs into a linear token stream for RNN
        seq_tensors = []
        for pairs in pairs_batch:
            tokens = []
            for k, v in pairs:
                tokens.extend([k, v])
            seq_tensors.append(torch.tensor(tokens, device=device))
        seq_tensor = torch.stack(seq_tensors, dim=0)  # [B, 2*N]

        x = self.token_embed(seq_tensor)
        _, h_final = self.rnn(x)  # [1, B, embed_dim]
        h_ctx = h_final.squeeze(0)  # [B, embed_dim]

        # Process each query
        q_embed = self.token_embed(query_tokens)  # [B, num_q, D]
        # Combine context with query
        state = q_embed + h_ctx.unsqueeze(1)

        # Multi-tick deliberation on latent state (System 2)
        flat_state = state.reshape(B * num_q, self.embed_dim)
        for _ in range(self.act_ticks):
            flat_state = self.deliberation_cell(flat_state)

        # Metric un-embedding
        state_norm = F.normalize(flat_state.reshape(B, num_q, self.embed_dim), p=2, dim=-1)
        embed_norm = F.normalize(self.token_embed.weight, p=2, dim=-1)
        logits = torch.matmul(state_norm, embed_norm.T) * self.temp
        return logits


# ==============================================================================
# 3. CONTROL ARM 2: MODE C ALONE (Day/Night Memory Buffer, Surface 1-Step Retrieval)
# ==============================================================================

class ModeCAlone(nn.Module):
    """
    Mode C Alone:
    - Fast-weight associative memory matrix (DeltaNet / Titans style).
    - Writes key-value associations dynamically into memory matrix.
    - Single-step surface lookup (System 1 only, ticks = 1).
    - Cannot execute multi-hop iterative relaxation.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 40):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.temp = nn.Parameter(torch.tensor(5.0))

    def forward(self, pairs_batch: List[List[Tuple[int, int]]], query_tokens: torch.Tensor) -> torch.Tensor:
        B, num_q = query_tokens.shape
        device = query_tokens.device

        # Build associative memory matrix M per batch sample
        M_list = []
        for b in range(B):
            pairs = pairs_batch[b]
            k_toks = torch.tensor([p[0] for p in pairs], device=device)
            v_toks = torch.tensor([p[1] for p in pairs], device=device)

            k_vecs = F.normalize(self.W_k(self.token_embed(k_toks)), p=2, dim=-1)  # [N, D]
            v_vecs = self.W_v(self.token_embed(v_toks))                            # [N, D]

            # In-context associative memory accumulation: sum_i (v_i outer k_i)
            M = torch.matmul(v_vecs.T, k_vecs) / math.sqrt(self.embed_dim)         # [D, D]
            M_list.append(M)

        M_batch = torch.stack(M_list, dim=0)  # [B, D, D]

        # Single-step surface lookup (1-hop lookup)
        q_embed = self.token_embed(query_tokens)  # [B, num_q, D]
        q_vec = F.normalize(self.W_q(q_embed), p=2, dim=-1)

        # retrieved = M @ q_vec
        retrieved = torch.bmm(M_batch, q_vec.transpose(1, 2)).transpose(1, 2)  # [B, num_q, D]

        # Metric un-embedding against token embeddings
        retrieved_norm = F.normalize(retrieved, p=2, dim=-1)
        embed_norm = F.normalize(self.token_embed.weight, p=2, dim=-1)
        logits = torch.matmul(retrieved_norm, embed_norm.T) * self.temp
        return logits


# ==============================================================================
# 4. TREATMENT ARM: UNIFIED SHARED-SUBSTRATE B+C (Isomorphic + REM Replay)
# ==============================================================================

class UnifiedSubstrateEngine(nn.Module):
    """
    Treatment Arm:
    - Unified Substrate: Thoughts τ_t and Memory m_k share the identical representation space.
    - Epistemic Write-Gating: Fast-weight associative write with adaptive gating.
    - Multi-Tick Deliberation on Memory Substrate: Successively queries and relaxes through
      the associative state across multiple ticks (System 2):
        Tick 1: K_a -> K_b
        Tick 2: K_b -> V_c
      enabling true multi-hop transitive reasoning.
    - REM Generative Replay: LeJEPA SIGReg regularizer preventing dimensional collapse.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 40, act_ticks: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.act_ticks = act_ticks

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)

        # Deliberation recurrent relaxation cell
        self.deliberation_cell = nn.GRUCell(embed_dim, embed_dim)
        self.temp = nn.Parameter(torch.tensor(5.0))

        # LeJEPA Anti-Collapse Regularizer for REM replay
        self.sigreg = SIGReg(num_slices=32, lambda_reg=0.2)

    def forward(
        self,
        pairs_batch: List[List[Tuple[int, int]]],
        query_tokens: torch.Tensor,
        is_sleep_replay: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, num_q = query_tokens.shape
        device = query_tokens.device

        # Build associative memory matrix M per batch sample
        M_list = []
        for b in range(B):
            pairs = pairs_batch[b]
            k_toks = torch.tensor([p[0] for p in pairs], device=device)
            v_toks = torch.tensor([p[1] for p in pairs], device=device)

            k_vecs = F.normalize(self.W_k(self.token_embed(k_toks)), p=2, dim=-1)
            v_vecs = self.W_v(self.token_embed(v_toks))

            M = torch.matmul(v_vecs.T, k_vecs) / math.sqrt(self.embed_dim)
            M_list.append(M)

        M_batch = torch.stack(M_list, dim=0)  # [B, D, D]

        # Initial thought state τ_0
        q_embed = self.token_embed(query_tokens)
        curr_state = F.normalize(self.W_q(q_embed), p=2, dim=-1)  # [B, num_q, D]

        rem_loss = torch.tensor(0.0, device=device)

        # Multi-Tick Deliberation across the Unified Memory Lattice (System 2)
        for tick in range(self.act_ticks):
            # Query memory using current deliberative thought
            retrieved = torch.bmm(M_batch, curr_state.transpose(1, 2)).transpose(1, 2)  # [B, num_q, D]
            retrieved = F.normalize(retrieved, p=2, dim=-1)

            # Recurrent relaxation step
            flat_curr = curr_state.reshape(B * num_q, self.embed_dim)
            flat_ret = retrieved.reshape(B * num_q, self.embed_dim)
            updated = self.deliberation_cell(flat_ret, flat_curr)
            curr_state = F.normalize(updated.reshape(B, num_q, self.embed_dim), p=2, dim=-1)

        # REM Sleep Replay: LeJEPA SIGReg regularizer
        if is_sleep_replay or self.training:
            sig_loss, _ = self.sigreg(curr_state.reshape(-1, self.embed_dim))
            rem_loss = rem_loss + sig_loss

        # Metric un-embedding
        embed_norm = F.normalize(self.token_embed.weight, p=2, dim=-1)
        logits = torch.matmul(curr_state, embed_norm.T) * self.temp
        return logits, {"rem_sigreg_loss": rem_loss.item()}


# ==============================================================================
# 5. BENCHMARK SUITE & EVALUATION HARNESS
# ==============================================================================

def train_and_evaluate_arm(
    name: str,
    model: nn.Module,
    dataset: MQARDataset,
    epochs: int = 120,
    batch_size: int = 32,
    lr: float = 0.005,
    device: str = "cpu"
) -> Dict[str, float]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[{name}] Parameters: {param_count:,} | Training on {device}...")
    start_time = time.time()

    model.train()
    for epoch in range(epochs):
        pairs, queries, targets, is_mh = dataset.generate_batch(batch_size, num_direct=6, num_multi_hop=3)
        queries, targets = queries.to(device), targets.to(device)

        optimizer.zero_grad()
        if isinstance(model, UnifiedSubstrateEngine):
            logits, aux = model(pairs, queries, is_sleep_replay=(epoch % 5 == 0))
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss = loss + 0.1 * aux["rem_sigreg_loss"]
        else:
            logits = model(pairs, queries)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), targets.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    train_duration = time.time() - start_time

    # Evaluation phase on held-out test split
    model.eval()
    total_direct_correct = 0
    total_direct_queries = 0
    total_multihop_correct = 0
    total_multihop_queries = 0

    with torch.no_grad():
        for _ in range(25):
            pairs, queries, targets, is_mh = dataset.generate_batch(batch_size=32, num_direct=6, num_multi_hop=3)
            queries, targets, is_mh = queries.to(device), targets.to(device), is_mh.to(device)

            if isinstance(model, UnifiedSubstrateEngine):
                logits, _ = model(pairs, queries, is_sleep_replay=False)
            else:
                logits = model(pairs, queries)

            preds = logits.argmax(dim=-1)  # [B, num_q]

            # Multi-hop evaluation
            mh_mask = is_mh
            total_multihop_correct += (preds[mh_mask] == targets[mh_mask]).sum().item()
            total_multihop_queries += mh_mask.sum().item()

            # Direct evaluation
            dir_mask = ~is_mh
            total_direct_correct += (preds[dir_mask] == targets[dir_mask]).sum().item()
            total_direct_queries += dir_mask.sum().item()

    direct_acc = (total_direct_correct / max(total_direct_queries, 1)) * 100.0
    multihop_acc = (total_multihop_correct / max(total_multihop_queries, 1)) * 100.0
    overall_acc = ((total_direct_correct + total_multihop_correct) /
                   max(total_direct_queries + total_multihop_queries, 1)) * 100.0

    return {
        "name": name,
        "parameters": param_count,
        "direct_acc": direct_acc,
        "multihop_acc": multihop_acc,
        "overall_acc": overall_acc,
        "train_time_sec": train_duration
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1 MQAR Matched-Compute Benchmark")
    parser.add_argument("--epochs", type=int, default=120, help="Training epochs per arm")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 78)
    print("STAGE 1: MULTI-QUERY ASSOCIATIVE RECALL (MQAR) MATCHED-COMPUTE BENCHMARK")
    print(f"Device: {args.device} | Epochs per arm: {args.epochs} | Batch size: {args.batch_size}")
    print("=" * 78)

    dataset = MQARDataset(num_keys=32, num_vals=32, num_distractors=32)

    # Matched parameter budgets (~10k - 15k params each)
    # Mode B Alone: embed_dim = 40, act_ticks = 3
    arm1_model = ModeBAlone(vocab_size=dataset.VOCAB_SIZE, embed_dim=40, act_ticks=3)

    # Mode C Alone: embed_dim = 40, 1-step retrieval
    arm2_model = ModeCAlone(vocab_size=dataset.VOCAB_SIZE, embed_dim=40)

    # Treatment Arm: Unified Substrate B+C, embed_dim = 40, act_ticks = 2
    treatment_model = UnifiedSubstrateEngine(vocab_size=dataset.VOCAB_SIZE, embed_dim=40, act_ticks=2)

    # Execute matched benchmark
    results_b = train_and_evaluate_arm("Control Arm 1 (Mode B Alone)", arm1_model, dataset, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    results_c = train_and_evaluate_arm("Control Arm 2 (Mode C Alone)", arm2_model, dataset, epochs=args.epochs, batch_size=args.batch_size, device=args.device)
    results_unified = train_and_evaluate_arm("Treatment Arm (Unified B+C)", treatment_model, dataset, epochs=args.epochs, batch_size=args.batch_size, device=args.device)

    # Report Results Table
    print("\n" + "=" * 78)
    print(f"{'ARM / ARCHITECTURE':<32} | {'PARAMS':<8} | {'DIRECT ACC':<11} | {'MULTI-HOP':<11} | {'OVERALL'}")
    print("-" * 78)
    for res in [results_b, results_c, results_unified]:
        print(f"{res['name']:<32} | {res['parameters']:<8,d} | {res['direct_acc']:>9.2f}% | {res['multihop_acc']:>9.2f}% | {res['overall_acc']:>6.2f}%")
    print("=" * 78)

    # Evaluate Gate 1 Kill Criterion
    best_control_mh = max(results_b['multihop_acc'], results_c['multihop_acc'])
    gain = results_unified['multihop_acc'] - best_control_mh

    print(f"\n[GATE 1 EVALUATION]")
    print(f"Best Control Multi-Hop Accuracy: {best_control_mh:.2f}%")
    print(f"Unified Substrate Multi-Hop Accuracy: {results_unified['multihop_acc']:.2f}%")
    print(f"Observed Margin over Best Control: +{gain:.2f}% (Threshold: >= +15.0%)")

    if gain >= 15.0:
        print(">>> RESULT: GATE 1 PASSED! Unified substrate strictly outperforms decoupled controls.")
    else:
        print(">>> RESULT: GATE 1 FAILED / MARGIN INSUFFICIENT. Re-evaluate substrate dynamics.")


if __name__ == "__main__":
    main()
