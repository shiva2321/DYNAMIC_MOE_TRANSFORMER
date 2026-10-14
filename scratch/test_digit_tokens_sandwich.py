"""
test_digit_tokens_sandwich.py - Digit-by-digit token-based arithmetic representation.
Tests whether representing numbers as sequences of base-10 digits allows exact arithmetic
and tests length generalization (training on 2-digit, testing on 3-digit and 4-digit).
ASCII output only.
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.layers import YggdrasilProjector, YggdrasilMemoryCortex

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_DIGITS = 6  # supports up to 6-digit results (0 to 999,999)


class DigitWiseSandwichNet(nn.Module):
    """
    Digit-wise arithmetic network:
    Input: Digits of A (padded to MAX_DIGITS) + Digits of B (padded to MAX_DIGITS) + Op one-hot
    Encoder -> Yggdrasil Manifold Projector -> Memory Cortex -> Multi-head Digit Classifiers (10 classes per digit)
    """
    def __init__(self, max_digits=MAX_DIGITS, hidden_dim=128, num_prototypes=32, dim_h=16, dim_t=16):
        super().__init__()
        self.max_digits = max_digits
        # Each number has max_digits (one-hot or normalized), plus op (5)
        in_features = max_digits * 2 + 5

        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.projector = YggdrasilProjector(in_features=hidden_dim, dim_h=dim_h, dim_t=dim_t)
        self.cortex = YggdrasilMemoryCortex(num_prototypes=num_prototypes, dim_h=dim_h, dim_t=dim_t, out_features=hidden_dim)

        # Multi-head classifier: one 10-class head for each digit position (units, tens, hundreds, thousands...)
        self.digit_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 10) for _ in range(max_digits)
        ])

    def forward(self, x):
        feat = self.encoder(x)
        state = self.projector(feat)
        _, ctx = self.cortex(state)
        rep = feat + ctx
        # List of logits: [ (Batch, 10), (Batch, 10), ... ]
        logits = [head(rep) for head in self.digit_heads]
        return logits


def num_to_digits(n_arr, max_digits=MAX_DIGITS):
    """Decomposes an array of integers into an array of shape (N, max_digits)."""
    digits = np.zeros((len(n_arr), max_digits), dtype=np.float32)
    temp = np.abs(n_arr)
    for i in range(max_digits):
        digits[:, i] = (temp % 10).astype(np.float32) / 9.0  # normalize [0, 1]
        temp //= 10
    return digits


def make_dataset(num_samples, min_val, max_val, op='add', seed=42):
    rng = np.random.default_rng(seed)
    a = rng.integers(min_val, max_val + 1, size=num_samples)
    b = rng.integers(min_val, max_val + 1, size=num_samples)
    if op == 'add':
        c = a + b
        op_idx = 0
    elif op == 'mul':
        c = a * b
        op_idx = 2
    else:
        raise ValueError()

    a_dig = num_to_digits(a)
    b_dig = num_to_digits(b)
    op_onehot = np.zeros((num_samples, 5), dtype=np.float32)
    op_onehot[:, op_idx] = 1.0

    x = np.concatenate([a_dig, b_dig, op_onehot], axis=1)

    # True target digits (unnormalized integer classes 0..9)
    y_dig = np.zeros((num_samples, MAX_DIGITS), dtype=np.int64)
    temp = np.abs(c)
    for i in range(MAX_DIGITS):
        y_dig[:, i] = (temp % 10).astype(np.int64)
        temp //= 10

    return torch.tensor(x, dtype=torch.float32, device=DEVICE), torch.tensor(y_dig, dtype=torch.long, device=DEVICE), c


def run_digit_benchmark():
    print("================================================================")
    print("  DIGIT-WISE POSITIONAL ARITHMETIC ON YGGDRASIL SANDWICH")
    print("  Training on 1-2 digits [0..99].")
    print("  Testing Length Generalization on 3 digits [100..999] and 4 digits [1000..9999].")
    print("================================================================")

    for op in ['add', 'mul']:
        print(f"\n--- Operation: {op.upper()} ---")
        x_tr, y_tr, _ = make_dataset(4000, 0, 99, op=op, seed=100)
        x_te_seen, y_te_seen, _ = make_dataset(1000, 0, 99, op=op, seed=200)
        x_te_3dig, y_te_3dig, _ = make_dataset(1000, 100, 999, op=op, seed=300)
        x_te_4dig, y_te_4dig, _ = make_dataset(1000, 1000, 9999, op=op, seed=400)

        model = DigitWiseSandwichNet().to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

        model.train()
        for epoch in range(120):
            opt.zero_grad()
            logits_list = model(x_tr)
            loss = sum(F.cross_entropy(logits_list[i], y_tr[:, i]) for i in range(MAX_DIGITS))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            def evaluate(x_in, y_in):
                logits = model(x_in)
                preds = [l.argmax(dim=-1) for l in logits]
                # Exact integer match across all digits
                exact_match = torch.ones(x_in.shape[0], dtype=torch.bool, device=DEVICE)
                for i in range(MAX_DIGITS):
                    exact_match &= (preds[i] == y_in[:, i])
                return float(exact_match.float().mean().item())

            acc_seen = evaluate(x_te_seen, y_te_seen)
            acc_3dig = evaluate(x_te_3dig, y_te_3dig)
            acc_4dig = evaluate(x_te_4dig, y_te_4dig)

            print(f"  Exact Match Accuracy (Seen 1-2 Digits [0..99]): {acc_seen*100:6.2f}%")
            print(f"  Exact Match Accuracy (Unseen 3 Digits [100..999]): {acc_3dig*100:6.2f}%")
            print(f"  Exact Match Accuracy (Unseen 4 Digits [1000..9999]): {acc_4dig*100:6.2f}%")


if __name__ == "__main__":
    run_digit_benchmark()
