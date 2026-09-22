"""
Domain-Agnostic Algorithmic Scratchpad Deliberation Training.

Trains LoopedRiemannianTransformer on universal algorithmic reasoning sequences:
1. Algorithmic State Tracking (Cumulative parity automaton / memory stack).
2. Arbitrary-length Sequence Manipulation (Extrapolation on LIFO reversal).
3. Stepwise Decomposition Scratchpads (Iterative sub-goal execution without base-10 digit slicing cheats).
"""

import os
import sys
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.spherical_attention import (
    RiemannianSphericalAttention,
    LoopedRiemannianTransformer
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHARS = list("0123456789+-*=/.:>< \n[]#")
CHAR2ID = {c: i for i, c in enumerate(CHARS)}
ID2CHAR = {i: c for i, c in enumerate(CHARS)}
PAD_ID = CHAR2ID[" "]
VOCAB_SIZE = len(CHARS)


class AlgorithmicScratchpadModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_model=48, num_heads=4, hyperspace_dim=8, d_ff=96, max_loops=6, device=None):
        super().__init__()
        self.device = device or DEVICE
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.looped_transformer = LoopedRiemannianTransformer(
            d_model=d_model,
            num_heads=num_heads,
            hyperspace_dim=hyperspace_dim,
            d_ff=d_ff,
            max_loops=max_loops,
            freq_base=35.0,
            freq_scale=0.6,
            device=self.device
        )
        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, x, is_causal=True):
        embeds = self.embed(x)
        h, meta = self.looped_transformer(embeds, is_causal=is_causal)
        logits = self.classifier(h)
        return logits, meta


def generate_scratchpad_dataset(num_samples=2500):
    """Generates multi-domain algorithmic sequences with intermediate deliberation steps."""
    sequences = []

    # 1. State Tracking: Parity Automaton sequences (e.g. "[1 0 1 1]->P:0\n")
    for _ in range(num_samples // 3):
        length = random.randint(3, 8)
        bits = [random.choice([0, 1]) for _ in range(length)]
        parity = sum(bits) % 2
        bit_str = " ".join(str(b) for b in bits)
        sequences.append(f"[{bit_str}]->P:{parity}\n")

    # 2. Sequence Reversal / LIFO Working Memory (e.g. "[3 8 2]->R:[2 8 3]\n")
    for _ in range(num_samples // 3):
        length = random.randint(3, 6)
        digits = [random.randint(0, 9) for _ in range(length)]
        d_str = " ".join(str(d) for d in digits)
        rev_str = " ".join(str(d) for d in reversed(digits))
        sequences.append(f"[{d_str}]->R:[{rev_str}]\n")

    # 3. Stepwise Sub-Goal Decomposition Scratchpad (e.g. "45-12: >sub 10=35 >sub 2=33 #33\n")
    for _ in range(num_samples // 3):
        a = random.randint(15, 999)
        b = random.randint(1, 49)
        op = random.choice(["+", "-"])
        if op == "+":
            b_high = (b // 10) * 10
            b_low = b % 10
            s1 = a + b_high
            ans = s1 + b_low
            if b_high > 0:
                seq = f"{a}+{b}: >add {b_high}={s1} >add {b_low}={ans} #{ans}\n"
            else:
                seq = f"{a}+{b}: >add {b_low}={ans} #{ans}\n"
        else:
            if a < b:
                a, b = b, a
            b_high = (b // 10) * 10
            b_low = b % 10
            s1 = a - b_high
            ans = s1 - b_low
            if b_high > 0:
                seq = f"{a}-{b}: >sub {b_high}={s1} >sub {b_low}={ans} #{ans}\n"
            else:
                seq = f"{a}-{b}: >sub {b_low}={ans} #{ans}\n"
        sequences.append(seq)

    random.shuffle(sequences)
    return sequences


def train_scratchpad_reasoner(epochs=35):
    print("\n" + "=" * 75)
    print("TRAINING DOMAIN-AGNOSTIC ALGORITHMIC SCRATCHPAD REASONER")
    print("=" * 75)
    data = generate_scratchpad_dataset(3000)
    print(f"[*] Generated {len(data)} algorithmic sequences across 3 diverse domains...")
    print(f"    Example Parity:     {data[0].strip()}")
    print(f"    Example Reversal:   {data[1].strip()}")
    print(f"    Example Scratchpad: {data[2].strip()}")

    model = AlgorithmicScratchpadModel(vocab_size=VOCAB_SIZE, d_model=48, max_loops=6, device=DEVICE).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)

    max_len = max(len(s) for s in data)
    N = len(data)

    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(data)
        total_loss = 0.0
        steps = 0
        for b_start in range(0, N, 64):
            batch = data[b_start:b_start + 64]
            padded = []
            for s in batch:
                ids = [CHAR2ID[c] for c in s if c in CHAR2ID]
                padded.append(ids + [PAD_ID] * (max_len - len(ids)))
            tokens = torch.tensor(padded, dtype=torch.long, device=DEVICE)
            inp = tokens[:, :-1]
            tgt = tokens[:, 1:]

            logits, meta = model(inp, is_causal=True)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), tgt.reshape(-1), ignore_index=PAD_ID)
            loss = loss + 0.005 * meta.get("ponder_cost", 0.0)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            steps += 1

        scheduler.step()
        if ep % 7 == 0 or ep == epochs or ep == 1:
            avg_loss = total_loss / max(steps, 1)
            print(f"    Epoch {ep:2d}/{epochs:2d} | CrossEntropy: {avg_loss:.4f} | Avg Thinking Steps: {meta['steps_taken']:.2f}")

    model.eval()
    os.makedirs("artifacts", exist_ok=True)
    out_path = "artifacts/algorithmic_scratchpad_model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"[+] Saved trained algorithmic scratchpad model to: {out_path}")
    return model


if __name__ == "__main__":
    train_scratchpad_reasoner(epochs=28)
