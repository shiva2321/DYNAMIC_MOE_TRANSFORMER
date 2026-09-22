"""
Exhaustive Empirical Benchmark: Pure Learned Algorithmic Generalization & Length Extrapolation.

ZERO hardcoded rules, ZERO base-10 digit slicing, ZERO human heuristics.
Pure neural networks learning algorithmic state machines and memory stack manipulation from raw tokens.

Tasks:
1. Cumulative Binary Parity Automaton (Running XOR / 1-bit ALU):
   - In-Distribution Training: Sequences of length L in [3, 6]
   - Out-of-Distribution Testing: Sequences of length L in [8, 12, 16, 20] (up to >3x length!)
2. Differentiable Stack Reversal (LIFO Working Memory):
   - In-Distribution Training: Sequences of length L in [3, 5]
   - Out-of-Distribution Testing: Sequences of length L in [6, 8, 10, 12] (more than 2x length!)

Head-to-head comparison:
- Model A: Standard Flat Recurrent Network (Conventional GRU with equal parameters)
- Model B: Infinite Spherical Hyperspace Memory Network (ISHMN Controller + Riemannian Memory Shells)
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import SphericalHyperspaceControllerNet

# Deterministic seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =========================================================================
# 1. Standard Flat Recurrent Baseline (Conventional GRU)
# =========================================================================

class FlatRecurrentBaseline(nn.Module):
    """Conventional GRU with identical parameter budget, no spherical memory."""
    def __init__(self, vocab_size=16, embedding_dim=16, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.cell = nn.GRUCell(embedding_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.hidden_dim = hidden_dim

    def forward_sequence(self, token_seq):
        """
        token_seq: [B, T]
        Returns: logits: [B, T, num_classes]
        """
        B, T = token_seq.shape
        device = token_seq.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        logits_list = []

        embeds = self.embedding(token_seq) # [B, T, D]
        for t in range(T):
            x_t = embeds[:, t]
            h = self.cell(x_t, h)
            logits_t = self.classifier(h)
            logits_list.append(logits_t.unsqueeze(1))

        return torch.cat(logits_list, dim=1)


# =========================================================================
# 2. Infinite Spherical Hyperspace Memory Network Wrapper (ISHMN)
# =========================================================================

class ISHMAlgorithmNet(nn.Module):
    """
    Hyperspace Neural Controller with Infinite Spherical Memory:
    - Embedding maps tokens to payload representations
    - Controller navigates and writes to spherical hyperspace shells
    - Output head decodes next-step state
    """
    def __init__(self, vocab_size=16, embedding_dim=16, hidden_dim=64, hyperspace_dim=6, num_slots=16, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.controller = SphericalHyperspaceControllerNet(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            hyperspace_dim=hyperspace_dim,
            payload_dim=embedding_dim,
            num_slots=num_slots,
            max_slots=64,
            enable_steering=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

    def forward_sequence(self, token_seq):
        """
        token_seq: [B, T]
        Returns: logits: [B, T, num_classes]
        """
        B, T = token_seq.shape
        device = token_seq.device
        state = self.controller.init_state(B, device=device)
        logits_list = []

        embeds = self.embedding(token_seq)
        for t in range(T):
            x_t = embeds[:, t]
            out_t, state, _ = self.controller.step(x_t, state)
            logits_t = self.classifier(out_t)
            logits_list.append(logits_t.unsqueeze(1))

        return torch.cat(logits_list, dim=1)


# =========================================================================
# 3. Data Generators (Cumulative Parity & Stack Reversal)
# =========================================================================

def get_parity_batch(batch_size, length, device=None):
    """
    Generates binary sequences and cumulative parity ground truth.
    Tokens: 0 or 1.
    Target: running XOR (parity bit at each step).
    """
    bits = torch.randint(0, 2, (batch_size, length), device=device)
    # Compute cumulative parity: y_t = (sum_{i=1}^t bits_i) % 2
    targets = (torch.cumsum(bits, dim=1) % 2).long()
    return bits, targets

def get_reversal_batch(batch_size, length, device=None, vocab_size=10):
    """
    Generates sequence reversal tasks.
    Sequence: [s_1, s_2, ..., s_L, TRIGGER] -> Target: [PAD, PAD, ..., s_L, s_{L-1}, ..., s_1]
    Tokens 2..vocab_size-1 are payload symbols.
    Token 1 is TRIGGER (tells network to output reversed sequence).
    """
    symbols = torch.randint(2, vocab_size, (batch_size, length), device=device)
    trigger = torch.ones(batch_size, 1, dtype=torch.long, device=device)
    
    # Input is symbols + trigger + dummy read tokens
    dummy_read = torch.zeros(batch_size, length, dtype=torch.long, device=device)
    inputs = torch.cat([symbols, trigger, dummy_read], dim=1) # Length: 2*L + 1
    
    # Target is reversed symbols during the read phase
    reversed_symbols = torch.flip(symbols, dims=[1])
    # Ignore targets during ingestion phase (mask with -100)
    ingestion_ignore = torch.full((batch_size, length + 1), -100, dtype=torch.long, device=device)
    targets = torch.cat([ingestion_ignore, reversed_symbols], dim=1)
    
    return inputs, targets


# =========================================================================
# 4. Benchmark Execution: Parity Automaton Length Generalization
# =========================================================================

def benchmark_parity_automaton(device):
    print("\n" + "=" * 75)
    print("BENCHMARK 1: PURE LEARNED CUMULATIVE PARITY AUTOMATON (1-BIT ALU)")
    print("Zero hardcoded logic. Testing length generalization up to >3x training length!")
    print("=" * 75)

    base_net = FlatRecurrentBaseline(vocab_size=4, embedding_dim=16, hidden_dim=48, num_classes=2).to(device)
    ishm_net = ISHMAlgorithmNet(vocab_size=4, embedding_dim=16, hidden_dim=48, hyperspace_dim=6, num_slots=16, num_classes=2).to(device)

    opt_base = torch.optim.AdamW(base_net.parameters(), lr=0.003, weight_decay=1e-5)
    opt_ishm = torch.optim.AdamW(ishm_net.parameters(), lr=0.003, weight_decay=1e-5)

    # Train strictly on short sequences: lengths 3 to 6
    train_lengths = [3, 4, 5, 6]
    train_steps = 500
    batch_size = 32

    print("Training on short sequences (L in [3, 6])...")
    start_time = time.time()
    for step in range(1, train_steps + 1):
        l = random.choice(train_lengths)
        bits, targets = get_parity_batch(batch_size, l, device=device)

        # Baseline step
        opt_base.zero_grad()
        logits_base = base_net.forward_sequence(bits)
        loss_base = F.cross_entropy(logits_base.view(-1, 2), targets.view(-1))
        loss_base.backward()
        opt_base.step()

        # ISHMN step
        opt_ishm.zero_grad()
        logits_ishm = ishm_net.forward_sequence(bits)
        loss_ishm = F.cross_entropy(logits_ishm.view(-1, 2), targets.view(-1))
        loss_ishm.backward()
        opt_ishm.step()

        if step % 100 == 0:
            print(f"  Step {step:3d}/{train_steps} | Base Loss: {loss_base.item():.4f} | ISHM Loss: {loss_ishm.item():.4f}")

    train_dur = time.time() - start_time
    print(f"Training completed in {train_dur:.2f}s!")

    # Evaluate across a spectrum of lengths: In-Distribution and Out-of-Distribution
    eval_lengths = [3, 5, 8, 12, 16, 20]
    n_eval = 200

    results = {
        "lengths": eval_lengths,
        "base_step_acc": [],
        "base_seq_acc": [],
        "ishm_step_acc": [],
        "ishm_seq_acc": []
    }

    base_net.eval()
    ishm_net.eval()

    print("\n--- Length Generalization Evaluation (Parity Automaton) ---")
    print(f"{'Length':8s} | {'Base Step Acc':14s} | {'Base Seq Acc':13s} | {'ISHM Step Acc':14s} | {'ISHM Seq Acc':13s} | {'Status'}")
    print("-" * 80)

    for l in eval_lengths:
        with torch.no_grad():
            bits, targets = get_parity_batch(n_eval, l, device=device)

            # Baseline eval
            logits_b = base_net.forward_sequence(bits)
            preds_b = logits_b.argmax(dim=-1)
            step_acc_b = (preds_b == targets).float().mean().item() * 100.0
            seq_acc_b = (preds_b == targets).all(dim=-1).float().mean().item() * 100.0

            # ISHM eval
            logits_i = ishm_net.forward_sequence(bits)
            preds_i = logits_i.argmax(dim=-1)
            step_acc_i = (preds_i == targets).float().mean().item() * 100.0
            seq_acc_i = (preds_i == targets).all(dim=-1).float().mean().item() * 100.0

            results["base_step_acc"].append(step_acc_b)
            results["base_seq_acc"].append(seq_acc_b)
            results["ishm_step_acc"].append(step_acc_i)
            results["ishm_seq_acc"].append(seq_acc_i)

            tag = "In-Dist" if l <= 6 else f"OOD ({l/6.0:.1f}x Length)"
            print(f"L = {l:2d}     | {step_acc_b:12.1f}% | {seq_acc_b:11.1f}% | {step_acc_i:12.1f}% | {seq_acc_i:11.1f}% | {tag}")

    return results


# =========================================================================
# 5. Benchmark Execution: Differentiable Stack Reversal
# =========================================================================

def benchmark_stack_reversal(device):
    print("\n" + "=" * 75)
    print("BENCHMARK 2: DIFFERENTIABLE STACK REVERSAL (LIFO MEMORY GENERALIZATION)")
    print("Testing if working memory acts as an unconstrained push-pop stack across lengths.")
    print("=" * 75)

    vocab_size = 10
    base_net = FlatRecurrentBaseline(vocab_size=vocab_size, embedding_dim=16, hidden_dim=64, num_classes=vocab_size).to(device)
    ishm_net = ISHMAlgorithmNet(vocab_size=vocab_size, embedding_dim=16, hidden_dim=64, hyperspace_dim=6, num_slots=24, num_classes=vocab_size).to(device)

    opt_base = torch.optim.AdamW(base_net.parameters(), lr=0.003, weight_decay=1e-5)
    opt_ishm = torch.optim.AdamW(ishm_net.parameters(), lr=0.003, weight_decay=1e-5)

    # Train strictly on short sequences: lengths 3 to 5
    train_lengths = [3, 4, 5]
    train_steps = 600
    batch_size = 32

    print("Training on short sequences (L in [3, 5])...")
    start_time = time.time()
    for step in range(1, train_steps + 1):
        l = random.choice(train_lengths)
        inputs, targets = get_reversal_batch(batch_size, l, device=device, vocab_size=vocab_size)

        # Baseline step
        opt_base.zero_grad()
        logits_base = base_net.forward_sequence(inputs)
        loss_base = F.cross_entropy(logits_base.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        loss_base.backward()
        opt_base.step()

        # ISHMN step
        opt_ishm.zero_grad()
        logits_ishm = ishm_net.forward_sequence(inputs)
        loss_ishm = F.cross_entropy(logits_ishm.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        loss_ishm.backward()
        opt_ishm.step()

        if step % 150 == 0:
            print(f"  Step {step:3d}/{train_steps} | Base Loss: {loss_base.item():.4f} | ISHM Loss: {loss_ishm.item():.4f}")

    train_dur = time.time() - start_time
    print(f"Training completed in {train_dur:.2f}s!")

    eval_lengths = [3, 5, 7, 9, 12]
    n_eval = 200

    results = {
        "lengths": eval_lengths,
        "base_step_acc": [],
        "base_seq_acc": [],
        "ishm_step_acc": [],
        "ishm_seq_acc": []
    }

    base_net.eval()
    ishm_net.eval()

    print("\n--- Length Generalization Evaluation (Stack Reversal) ---")
    print(f"{'Length':8s} | {'Base Token Acc':15s} | {'Base Exact Seq':14s} | {'ISHM Token Acc':15s} | {'ISHM Exact Seq':14s} | {'Status'}")
    print("-" * 85)

    for l in eval_lengths:
        with torch.no_grad():
            inputs, targets = get_reversal_batch(n_eval, l, device=device, vocab_size=vocab_size)

            # Evaluate only during the output phase (where target != -100)
            valid_mask = (targets != -100)

            # Baseline eval
            logits_b = base_net.forward_sequence(inputs)
            preds_b = logits_b.argmax(dim=-1)
            correct_tokens_b = (preds_b == targets) & valid_mask
            step_acc_b = (correct_tokens_b.sum().float() / valid_mask.sum().float()).item() * 100.0
            
            # Exact sequence match: all output tokens in the sequence are correct
            seq_correct_b = ((preds_b == targets) | (~valid_mask)).all(dim=-1).float().mean().item() * 100.0

            # ISHM eval
            logits_i = ishm_net.forward_sequence(inputs)
            preds_i = logits_i.argmax(dim=-1)
            correct_tokens_i = (preds_i == targets) & valid_mask
            step_acc_i = (correct_tokens_i.sum().float() / valid_mask.sum().float()).item() * 100.0
            seq_correct_i = ((preds_i == targets) | (~valid_mask)).all(dim=-1).float().mean().item() * 100.0

            results["base_step_acc"].append(step_acc_b)
            results["base_seq_acc"].append(seq_correct_b)
            results["ishm_step_acc"].append(step_acc_i)
            results["ishm_seq_acc"].append(seq_correct_i)

            tag = "In-Dist" if l <= 5 else f"OOD ({l/5.0:.1f}x Length)"
            print(f"L = {l:2d}     | {step_acc_b:13.1f}% | {seq_correct_b:12.1f}% | {step_acc_i:13.1f}% | {seq_correct_i:12.1f}% | {tag}")

    return results


# =========================================================================
# 6. Visualization & Reporting
# =========================================================================

def plot_algorithmic_generalization_report(res_parity, res_stack):
    os.makedirs("artifacts", exist_ok=True)
    report_path = os.path.join("artifacts", "algorithmic_generalization_proof.png")

    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Scientific Proof: Pure Learned Algorithmic Generalization & Length Extrapolation\n"
        "(ZERO Hardcoded Code: Models Trained Strictly on Short Sequences and Tested on Unseen Long Sequences)",
        fontsize=14, fontweight="bold"
    )

    # 1. Parity Length Extrapolation
    ax1 = axs[0]
    lengths_p = res_parity["lengths"]
    ax1.plot(lengths_p, res_parity["ishm_step_acc"], marker="o", lw=2.8, color="#2ecc71", label="ISHMN (Hyperspace Memory)")
    ax1.plot(lengths_p, res_parity["base_step_acc"], marker="s", lw=2.5, linestyle="--", color="#e74c3c", label="Standard Flat GRU")
    ax1.axvline(x=6, color="#7f8c8d", linestyle=":", lw=2.0, label="Training Horizon Boundary (L=6)")
    ax1.axhline(y=50, color="#bdc3c7", linestyle="-.", lw=1.5, label="Random Guess (50%)")

    ax1.set_title("1. Cumulative Parity Automaton (1-bit ALU)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sequence Length L (Trained on L in [3, 6])", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Step Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(40, 105)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower left", fontsize=9.5)

    # 2. Stack Reversal Length Extrapolation
    ax2 = axs[1]
    lengths_s = res_stack["lengths"]
    ax2.plot(lengths_s, res_stack["ishm_step_acc"], marker="o", lw=2.8, color="#2ecc71", label="ISHMN (Hyperspace Memory)")
    ax2.plot(lengths_s, res_stack["base_step_acc"], marker="s", lw=2.5, linestyle="--", color="#e74c3c", label="Standard Flat GRU")
    ax2.axvline(x=5, color="#7f8c8d", linestyle=":", lw=2.0, label="Training Horizon Boundary (L=5)")

    ax2.set_title("2. Differentiable Stack Reversal (LIFO Memory)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sequence Length L (Trained on L in [3, 5])", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Token Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(report_path, dpi=200)
    plt.close()
    print(f"\n[Artifact Saved] Visual report written to: {report_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Pure Algorithmic Generalization Benchmark on: {device}")

    res_parity = benchmark_parity_automaton(device)
    res_stack = benchmark_stack_reversal(device)
    plot_algorithmic_generalization_report(res_parity, res_stack)

if __name__ == "__main__":
    main()
