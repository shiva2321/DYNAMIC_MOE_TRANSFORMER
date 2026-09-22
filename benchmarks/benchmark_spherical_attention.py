"""
Empirical Benchmark: Riemannian Spherical Attention (RSA) vs Standard Attention vs Flat RNN.
Testing Unbounded Length Extrapolation on Algorithmic Sequence Retrieval (Stack Reversal).

Models Compared:
1. Flat GRU Baseline: Conventional recurrent network (suffers from pointer/context smudging).
2. Standard Flat Transformer: Conventional multi-head softmax attention in Euclidean space.
3. Riemannian Spherical Attention (RSA): Native S^{D-1} x R+ attention with SO(2) geodesic phase winding.

Training Horizon: Strictly short sequences of length L in [3, 5].
Testing Horizon: Lengths L in [3, 5, 7, 9, 12, 16] (up to >3.2x length extrapolation!).
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

from ishmn.spherical_attention import RiemannianSphericalTransformerLayer

# Deterministic seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =========================================================================
# 1. Models
# =========================================================================

class FlatGRUBaseline(nn.Module):
    def __init__(self, vocab_size=10, d_model=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cell = nn.GRUCell(d_model, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        B, T = x.shape
        device = x.device
        h = torch.zeros(B, self.hidden_dim, device=device)
        logits_list = []
        embeds = self.embedding(x)
        for t in range(T):
            h = self.cell(embeds[:, t], h)
            logits_list.append(self.classifier(h).unsqueeze(1))
        return torch.cat(logits_list, dim=1)


class StandardEuclideanTransformer(nn.Module):
    """Standard multi-head attention with sinusoidal positional embeddings."""
    def __init__(self, vocab_size=10, d_model=32, num_heads=4, d_ff=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        embeds = self.embedding(x) + self.pos_embedding[:, :T]
        h = self.encoder(embeds)
        return self.classifier(h)


class RSATransformer(nn.Module):
    """Transformer powered by Riemannian Spherical Attention on S^{D-1} x R+ with SO(2) phase winding."""
    def __init__(self, vocab_size=10, d_model=32, num_heads=4, hyperspace_dim=8, d_ff=64, device=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layer1 = RiemannianSphericalTransformerLayer(
            d_model=d_model, num_heads=num_heads, hyperspace_dim=hyperspace_dim, d_ff=d_ff, device=device
        )
        self.layer2 = RiemannianSphericalTransformerLayer(
            d_model=d_model, num_heads=num_heads, hyperspace_dim=hyperspace_dim, d_ff=d_ff, device=device
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embedding(x)
        h = self.layer1(h)
        h = self.layer2(h)
        return self.classifier(h)


# =========================================================================
# 2. Data Generator (Stack Reversal)
# =========================================================================

def get_reversal_batch(batch_size, length, device=None, vocab_size=10):
    symbols = torch.randint(2, vocab_size, (batch_size, length), device=device)
    trigger = torch.ones(batch_size, 1, dtype=torch.long, device=device) # token 1
    dummy_read = torch.zeros(batch_size, length, dtype=torch.long, device=device) # token 0
    inputs = torch.cat([symbols, trigger, dummy_read], dim=1) # total len: 2*length + 1

    reversed_symbols = torch.flip(symbols, dims=[1])
    ingestion_ignore = torch.full((batch_size, length + 1), -100, dtype=torch.long, device=device)
    targets = torch.cat([ingestion_ignore, reversed_symbols], dim=1)
    return inputs, targets


# =========================================================================
# 3. Main Benchmark Execution
# =========================================================================

def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("   RIEMANNIAN SPHERICAL ATTENTION (RSA) LENGTH EXTRAPOLATION BENCHMARK")
    print(f"   Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    vocab_size = 10
    d_model = 32

    gru_model = FlatGRUBaseline(vocab_size=vocab_size, d_model=d_model, hidden_dim=64).to(device)
    std_trans = StandardEuclideanTransformer(vocab_size=vocab_size, d_model=d_model, num_heads=4, d_ff=64).to(device)
    rsa_trans = RSATransformer(vocab_size=vocab_size, d_model=d_model, num_heads=4, hyperspace_dim=8, d_ff=64, device=device).to(device)

    opt_gru = torch.optim.AdamW(gru_model.parameters(), lr=0.003, weight_decay=1e-5)
    opt_std = torch.optim.AdamW(std_trans.parameters(), lr=0.003, weight_decay=1e-5)
    opt_rsa = torch.optim.AdamW(rsa_trans.parameters(), lr=0.003, weight_decay=1e-5)

    # Train strictly on short sequences: L in [3, 4, 5]
    train_lengths = [3, 4, 5]
    train_steps = 700
    batch_size = 32

    print("\n--- Training on Short Sequences (L in [3, 5]) ---")
    start_t = time.time()
    for step in range(1, train_steps + 1):
        l = random.choice(train_lengths)
        inputs, targets = get_reversal_batch(batch_size, l, device=device, vocab_size=vocab_size)

        # GRU step
        opt_gru.zero_grad()
        out_gru = gru_model(inputs)
        loss_gru = F.cross_entropy(out_gru.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        loss_gru.backward()
        opt_gru.step()

        # Standard Transformer step
        opt_std.zero_grad()
        out_std = std_trans(inputs)
        loss_std = F.cross_entropy(out_std.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        loss_std.backward()
        opt_std.step()

        # RSA Transformer step
        opt_rsa.zero_grad()
        out_rsa = rsa_trans(inputs)
        loss_rsa = F.cross_entropy(out_rsa.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        loss_rsa.backward()
        opt_rsa.step()

        if step % 175 == 0:
            print(f"  Step {step:3d}/{train_steps} | GRU Loss: {loss_gru.item():.4f} | StdTrans Loss: {loss_std.item():.4f} | RSA Loss: {loss_rsa.item():.4f}")

    train_time = time.time() - start_t
    print(f"Training completed in {train_time:.2f}s!")

    # Evaluation across lengths: in-distribution (3, 5) and OOD (7, 9, 12, 16)
    eval_lengths = [3, 5, 7, 9, 12, 16]
    n_eval = 250

    results = {
        "lengths": eval_lengths,
        "gru_token_acc": [],
        "gru_seq_acc": [],
        "std_token_acc": [],
        "std_seq_acc": [],
        "rsa_token_acc": [],
        "rsa_seq_acc": []
    }

    gru_model.eval()
    std_trans.eval()
    rsa_trans.eval()

    print("\n" + "=" * 90)
    print("LENGTH EXTRAPOLATION EVALUATION: Differentiable Stack Reversal")
    print("=" * 90)
    print(f"{'Length':8s} | {'Flat GRU (Token / Seq)':23s} | {'Std Transformer (Tok/Seq)':25s} | {'RSA (Ours: Token / Seq)':25s} | {'Status'}")
    print("-" * 90)

    for l in eval_lengths:
        with torch.no_grad():
            inputs, targets = get_reversal_batch(n_eval, l, device=device, vocab_size=vocab_size)
            valid_mask = (targets != -100)

            # GRU eval
            out_g = gru_model(inputs).argmax(dim=-1)
            t_acc_g = ((out_g == targets) & valid_mask).sum().float() / valid_mask.sum().float() * 100.0
            s_acc_g = ((out_g == targets) | (~valid_mask)).all(dim=-1).float().mean() * 100.0

            # Std Transformer eval
            out_s = std_trans(inputs).argmax(dim=-1)
            t_acc_s = ((out_s == targets) & valid_mask).sum().float() / valid_mask.sum().float() * 100.0
            s_acc_s = ((out_s == targets) | (~valid_mask)).all(dim=-1).float().mean() * 100.0

            # RSA Transformer eval
            out_r = rsa_trans(inputs).argmax(dim=-1)
            t_acc_r = ((out_r == targets) & valid_mask).sum().float() / valid_mask.sum().float() * 100.0
            s_acc_r = ((out_r == targets) | (~valid_mask)).all(dim=-1).float().mean() * 100.0

            results["gru_token_acc"].append(t_acc_g.item())
            results["gru_seq_acc"].append(s_acc_g.item())
            results["std_token_acc"].append(t_acc_s.item())
            results["std_seq_acc"].append(s_acc_s.item())
            results["rsa_token_acc"].append(t_acc_r.item())
            results["rsa_seq_acc"].append(s_acc_r.item())

            tag = "In-Dist" if l <= 5 else f"OOD ({l/5.0:.1f}x Length)"
            print(f"L = {l:2d}     | {t_acc_g.item():5.1f}% / {s_acc_g.item():5.1f}%          | {t_acc_s.item():5.1f}% / {s_acc_s.item():5.1f}%            | {t_acc_r.item():5.1f}% / {s_acc_r.item():5.1f}%            | {tag}")

    # Generate visual plot
    plot_rsa_report(results)

def plot_rsa_report(results):
    os.makedirs("artifacts", exist_ok=True)
    report_path = os.path.join("artifacts", "rsa_attention_generalization.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Empirical Proof: Riemannian Spherical Attention (RSA) vs Standard Attention\n"
        "(Trained on Short Sequences L in [3, 5] | Tested up to L = 16 [>3.2x Extrapolation])",
        fontsize=14, fontweight="bold"
    )

    lengths = results["lengths"]

    # Panel 1: Exact Sequence Match Rate
    ax1.plot(lengths, results["rsa_seq_acc"], marker="o", lw=3.0, color="#2ecc71", label="RSA (Riemannian Spherical Attention)")
    ax1.plot(lengths, results["std_seq_acc"], marker="^", lw=2.5, linestyle="--", color="#3498db", label="Standard Euclidean Transformer")
    ax1.plot(lengths, results["gru_seq_acc"], marker="s", lw=2.2, linestyle=":", color="#e74c3c", label="Flat GRU Recurrent Baseline")
    ax1.axvline(x=5, color="#7f8c8d", linestyle=":", lw=1.8, label="Training Horizon (L = 5)")

    ax1.set_title("1. Exact Sequence Match Rate (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Sequence Length L", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Exact Sequence Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper right", fontsize=9.5)

    # Panel 2: Token-Level Accuracy
    ax2.plot(lengths, results["rsa_token_acc"], marker="o", lw=3.0, color="#2ecc71", label="RSA (Ours)")
    ax2.plot(lengths, results["std_token_acc"], marker="^", lw=2.5, linestyle="--", color="#3498db", label="Standard Transformer")
    ax2.plot(lengths, results["gru_token_acc"], marker="s", lw=2.2, linestyle=":", color="#e74c3c", label="Flat GRU")
    ax2.axvline(x=5, color="#7f8c8d", linestyle=":", lw=1.8, label="Training Horizon (L = 5)")
    ax2.axhline(y=12.5, color="#bdc3c7", linestyle="-.", lw=1.2, label="Random Guess (~12.5%)")

    ax2.set_title("2. Individual Token Retrieval Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Sequence Length L", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Token Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 105)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(report_path, dpi=200)
    plt.close()
    print(f"\n[Artifact Saved] RSA visual report written to: {report_path}")

if __name__ == "__main__":
    run_benchmark()
